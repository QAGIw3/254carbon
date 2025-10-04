"""
CAISO compliance reporting service for regulatory requirements.

Generates:
- Settlement data report (node-level aggregates)
- Resource adequacy report (entity-level compliance snapshot)
- Renewable portfolio report (RPS progress and actions)

Entitlements are enforced to restrict access to downloads where required
(pilot configuration) and API-only calls as appropriate.
"""
import logging
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from auth import verify_token
from entitlements import check_entitlement
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import db

logger = logging.getLogger(__name__)

# CAISO compliance router
caiso_compliance_router = APIRouter(prefix="/api/v1/caiso/compliance", tags=["CAISO Compliance"])


class ComplianceReportType(str, Enum):
    """Types of CAISO compliance reports."""
    SETTLEMENT_DATA = "settlement_data"
    RESOURCE_ADEQUACY = "resource_adequacy"
    RENEWABLE_PORTFOLIO = "renewable_portfolio"
    TRANSMISSION_USAGE = "transmission_usage"
    CONGESTION_REVENUE = "congestion_revenue"
    ANCILLARY_SERVICES = "ancillary_services"


class ComplianceReportRequest(BaseModel):
    """Request for CAISO compliance report."""
    report_type: ComplianceReportType
    start_date: date
    end_date: date
    entity_id: Optional[str] = None  # Specific entity for RA reports
    format: str = "json"  # json, csv, pdf


class SettlementDataReport(BaseModel):
    """CAISO settlement data report."""
    report_period: str
    total_settled_volume: float
    total_settlement_amount: float
    average_price: float
    settlement_summary: Dict[str, Any]
    node_details: List[Dict[str, Any]]
    compliance_status: str
    generated_at: datetime


class ResourceAdequacyReport(BaseModel):
    """CAISO resource adequacy compliance report."""
    entity_id: str
    report_month: str
    ra_requirement_mw: float
    ra_capacity_committed_mw: float
    ra_surplus_deficit_mw: float
    compliance_percentage: float
    penalty_assessment: float
    monthly_details: List[Dict[str, Any]]
    recommendations: List[str]


class RenewablePortfolioReport(BaseModel):
    """CAISO renewable portfolio standard compliance report."""
    entity_id: str
    compliance_year: int
    rps_target_percentage: float
    actual_renewable_percentage: float
    shortfall_surplus_percentage: float
    penalty_cost: float
    renewable_sources: List[Dict[str, Any]]
    compliance_actions: List[str]


# CAISO compliance endpoints

@caiso_compliance_router.post("/reports/settlement", response_model=SettlementDataReport)
async def generate_settlement_report(
    request: ComplianceReportRequest,
    user=Depends(verify_token),
):
    """
    Generate CAISO settlement data compliance report.

    Data sourcing
    - ClickHouse: market_intelligence.market_price_ticks filtered by market=CAISO and date range
    - Aggregates volume, amount, and price statistics by instrument
    """
    # Check CAISO entitlement
    await check_entitlement(user, "market", "power", "downloads")

    try:
        # Get settlement data from ClickHouse
        clickhouse = await db.get_clickhouse_client()

        query = """
            SELECT
                instrument_id,
                SUM(volume) as settled_volume,
                SUM(volume * value) as settlement_amount,
                AVG(value) as avg_price,
                COUNT(*) as settlement_count
            FROM market_intelligence.market_price_ticks
            WHERE market = 'CAISO'
                AND event_time >= toDateTime('{start_date}')
                AND event_time < toDateTime('{end_date}') + INTERVAL 1 DAY
            GROUP BY instrument_id
            ORDER BY settlement_amount DESC
        """

        results = await clickhouse.fetch_all(query)

        # Calculate totals
        total_volume = sum(row['settled_volume'] for row in results)
        total_amount = sum(row['settlement_amount'] for row in results)
        avg_price = total_amount / total_volume if total_volume > 0 else 0

        # Generate report
        report = SettlementDataReport(
            report_period=f"{request.start_date} to {request.end_date}",
            total_settled_volume=total_volume,
            total_settlement_amount=total_amount,
            average_price=avg_price,
            settlement_summary={
                "total_transactions": sum(row['settlement_count'] for row in results),
                "unique_nodes": len(results),
                "avg_transaction_size": total_volume / len(results) if results else 0,
                "price_range": {
                    "min": min((row['avg_price'] for row in results), default=0),
                    "max": max((row['avg_price'] for row in results), default=0)
                }
            },
            node_details=[
                {
                    "node_id": row['instrument_id'],
                    "settled_volume": row['settled_volume'],
                    "settlement_amount": row['settlement_amount'],
                    "average_price": row['avg_price'],
                    "transaction_count": row['settlement_count']
                }
                for row in results
            ],
            compliance_status="compliant",
            generated_at=datetime.utcnow()
        )

        return report

    except Exception as e:
        logger.error(f"Error generating CAISO settlement report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@caiso_compliance_router.post("/reports/resource-adequacy", response_model=ResourceAdequacyReport)
async def generate_resource_adequacy_report(
    request: ComplianceReportRequest,
    user=Depends(verify_token),
):
    """
    Generate CAISO resource adequacy compliance report.

    Notes
    - Mock calculation; replace with actual RA datasets and formulas
    - Enforces presence of entity_id
    """
    # Check CAISO entitlement
    await check_entitlement(user, "market", "power", "downloads")

    if not request.entity_id:
        raise HTTPException(status_code=400, detail="entity_id required for RA reports")

    try:
        # Calculate RA compliance (mock implementation)
        # In reality, this would query actual RA data

        report_month = request.start_date.strftime("%Y-%m")

        # Mock RA data
        ra_requirement = 1000.0  # MW
        ra_committed = 950.0     # MW
        ra_deficit = ra_requirement - ra_committed
        compliance_pct = (ra_committed / ra_requirement) * 100 if ra_requirement > 0 else 0
        penalty = ra_deficit * 1000  # $1000/MW penalty

        report = ResourceAdequacyReport(
            entity_id=request.entity_id,
            report_month=report_month,
            ra_requirement_mw=ra_requirement,
            ra_capacity_committed_mw=ra_committed,
            ra_surplus_deficit_mw=-ra_deficit,  # Negative indicates deficit
            compliance_percentage=compliance_pct,
            penalty_assessment=penalty,
            monthly_details=[
                {
                    "day": (request.start_date + timedelta(days=i)).isoformat(),
                    "peak_demand": 800 + i * 10,
                    "capacity_committed": 950 - i * 5,
                    "shortfall": max(0, (800 + i * 10) - (950 - i * 5))
                }
                for i in range((request.end_date - request.start_date).days + 1)
            ],
            recommendations=[
                "Increase capacity commitments for peak demand periods",
                "Consider bilateral capacity contracts",
                "Monitor RA market prices for optimization opportunities"
            ] if compliance_pct < 100 else [
                "Maintain current capacity commitments",
                "Monitor for RA market price increases"
            ]
        )

        return report

    except Exception as e:
        logger.error(f"Error generating CAISO RA report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@caiso_compliance_router.post("/reports/renewable-portfolio", response_model=RenewablePortfolioReport)
async def generate_renewable_portfolio_report(
    request: ComplianceReportRequest,
    user=Depends(verify_token),
):
    """
    Generate CAISO renewable portfolio standard compliance report.

    Notes
    - Mock values for sources/action plan; replace with portfolio analytics
    """
    # Check CAISO entitlement
    await check_entitlement(user, "market", "power", "downloads")

    if not request.entity_id:
        raise HTTPException(status_code=400, detail="entity_id required for RPS reports")

    try:
        # Calculate RPS compliance (mock implementation)
        target_pct = 0.60  # 60% RPS target
        actual_pct = 0.55  # 55% actual renewable
        shortfall_pct = target_pct - actual_pct
        penalty_cost = shortfall_pct * 1000000  # Penalty calculation

        report = RenewablePortfolioReport(
            entity_id=request.entity_id,
            compliance_year=request.start_date.year,
            rps_target_percentage=target_pct * 100,
            actual_renewable_percentage=actual_pct * 100,
            shortfall_surplus_percentage=shortfall_pct * 100,
            penalty_cost=penalty_cost,
            renewable_sources=[
                {
                    "source_type": "Solar",
                    "capacity_mw": 150.0,
                    "generation_mwh": 250000.0,
                    "percentage": 35.0
                },
                {
                    "source_type": "Wind",
                    "capacity_mw": 100.0,
                    "generation_mwh": 180000.0,
                    "percentage": 25.0
                },
                {
                    "source_type": "Hydro",
                    "capacity_mw": 50.0,
                    "generation_mwh": 120000.0,
                    "percentage": 17.0
                },
                {
                    "source_type": "Geothermal",
                    "capacity_mw": 25.0,
                    "generation_mwh": 160000.0,
                    "percentage": 23.0
                }
            ],
            compliance_actions=[
                "Procure additional renewable energy contracts",
                "Consider renewable energy credit (REC) purchases",
                "Evaluate new renewable project development"
            ] if shortfall_pct > 0 else [
                "Maintain current renewable portfolio",
                "Monitor REC market for selling opportunities"
            ]
        )

        return report

    except Exception as e:
        logger.error(f"Error generating CAISO RPS report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@caiso_compliance_router.get("/requirements/current")
async def get_caiso_compliance_requirements(
    entity_type: str = Query("load_serving_entity", description="Entity type"),
    user=Depends(verify_token),
):
    """Get current CAISO compliance requirements (static snapshot)."""
    # Check CAISO entitlement
    await check_entitlement(user, "market", "power", "api")

    try:
        # Return current compliance requirements
        requirements = {
            "resource_adequacy": {
                "monthly_requirement_mw": 1000.0,
                "peak_demand_forecast": 1200.0,
                "planning_reserve_margin": 0.15,
                "must_offer_obligation": True,
                "flexible_ra_requirement": 100.0
            },
            "renewable_portfolio_standard": {
                "current_year_target": 0.60,
                "long_term_target": 1.00,
                "interim_targets": {
                    "2025": 0.65,
                    "2030": 0.80,
                    "2045": 1.00
                },
                "eligible_sources": [
                    "Solar", "Wind", "Hydro", "Geothermal",
                    "Biomass", "Fuel Cell", "Wave", "Tidal"
                ]
            },
            "transmission_access_charge": {
                "tac_rate": 0.015,  # $/kW-month
                "wheeling_rate": 0.008,
                "congestion_revenue_rights": "available"
            },
            "ancillary_services": {
                "regulation_up_requirement": 50.0,
                "regulation_down_requirement": 50.0,
                "spinning_reserve_requirement": 100.0,
                "non_spinning_reserve_requirement": 100.0
            },
            "last_updated": datetime.utcnow().isoformat()
        }

        return requirements

    except Exception as e:
        logger.error(f"Error fetching CAISO compliance requirements: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@caiso_compliance_router.get("/penalties/calculator")
async def calculate_compliance_penalties(
    shortfall_mw: float = Query(..., description="Capacity shortfall in MW"),
    penalty_type: str = Query("resource_adequacy", description="Type of penalty"),
    user=Depends(verify_token),
):
    """Calculate potential compliance penalties based on simple rates."""
    # Check CAISO entitlement
    await check_entitlement(user, "market", "power", "api")

    try:
        # Penalty calculations based on CAISO rules
        penalty_rates = {
            "resource_adequacy": 1000.0,  # $/MW-month
            "renewable_portfolio": 50.0,   # $/MWh shortfall
            "ancillary_services": 500.0,   # $/MW-month
        }

        rate = penalty_rates.get(penalty_type, 1000.0)

        # Calculate penalties for different time periods
        penalties = {
            "monthly_penalty": shortfall_mw * rate,
            "quarterly_penalty": shortfall_mw * rate * 3,
            "annual_penalty": shortfall_mw * rate * 12,
            "penalty_rate": rate,
            "calculation_basis": f"${rate}/MW-month for {penalty_type}",
            "mitigation_strategies": [
                "Procure additional capacity contracts",
                "Participate in capacity market auctions",
                "Consider demand response programs",
                "Evaluate bilateral agreements"
            ]
        }

        return penalties

    except Exception as e:
        logger.error(f"Error calculating compliance penalties: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@caiso_compliance_router.post("/reports/batch")
async def generate_batch_compliance_reports(
    report_types: List[ComplianceReportType],
    start_date: date,
    end_date: date,
    entity_id: Optional[str] = None,
    recipients: List[str] = [],
    user=Depends(verify_token),
):
    """Generate batch compliance reports for multiple report types."""
    # Check CAISO entitlement
    await check_entitlement(user, "market", "power", "downloads")

    try:
        reports = {}

        # Generate each requested report
        for report_type in report_types:
            request = ComplianceReportRequest(
                report_type=report_type,
                start_date=start_date,
                end_date=end_date,
                entity_id=entity_id,
                format="json"
            )

            if report_type == ComplianceReportType.SETTLEMENT_DATA:
                reports["settlement"] = await generate_settlement_report(request, user)
            elif report_type == ComplianceReportType.RESOURCE_ADEQUACY:
                reports["resource_adequacy"] = await generate_resource_adequacy_report(request, user)
            elif report_type == ComplianceReportType.RENEWABLE_PORTFOLIO:
                reports["renewable_portfolio"] = await generate_renewable_portfolio_report(request, user)

        # Send batch report email if recipients provided
        if recipients:
            await send_compliance_report_batch(recipients, reports, start_date, end_date)

        return {
            "batch_id": f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "reports_generated": len(reports),
            "report_types": [rt.value for rt in report_types],
            "period": f"{start_date} to {end_date}",
            "reports": reports,
            "generated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error generating batch compliance reports: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def send_compliance_report_batch(recipients: List[str], reports: Dict, start_date: date, end_date: date):
    """Send batch compliance reports via email."""
    try:
        # This would integrate with email service
        for recipient in recipients:
            logger.info(f"Sending compliance report batch to {recipient}")
            logger.info(f"Report period: {start_date} to {end_date}")
            logger.info(f"Reports included: {list(reports.keys())}")

            # In real implementation, would send formatted email with report summaries

    except Exception as e:
        logger.error(f"Error sending compliance report batch: {e}")
