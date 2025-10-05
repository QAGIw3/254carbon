"""
RegTech Compliance API Router

Purpose
-------
Exposes endpoints for tracking regulatory updates, analyzing compliance gaps,
and generating reports. Uses the `RegTechEngine` to encapsulate domain logic.

Endpoints
---------
- GET `/api/v1/regtech/regulations` — updates since a date
- GET `/api/v1/regtech/compliance/gaps` — gap analysis by entity
- POST `/api/v1/regtech/reports/generate` — report generation
- GET `/api/v1/regtech/penalty-risk/{entity_id}` — penalty risk summary
"""
import logging
from datetime import date
from typing import List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from regtech.compliance_engine import RegTechEngine, Jurisdiction, RegulationType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/regtech", tags=["regtech"])

# Initialize RegTech engine
regtech = RegTechEngine()


class RegulationUpdate(BaseModel):
    """Regulation update."""
    regulation_id: str
    jurisdiction: Jurisdiction
    title: str
    regulation_type: RegulationType
    effective_date: date
    summary: str
    impact_assessment: str


class ComplianceGap(BaseModel):
    """Compliance gap."""
    gap_id: str
    regulation_id: str
    jurisdiction: Jurisdiction
    requirement: str
    current_status: str
    risk_level: str
    remediation_cost_usd: float


class ComplianceReport(BaseModel):
    """Compliance report."""
    report_id: str
    report_type: str
    jurisdiction: Jurisdiction
    reporting_period: str
    data_completeness_pct: float
    validation_status: str
    errors: List[str]


@router.get("/regulations", response_model=List[RegulationUpdate])
async def get_regulatory_updates(
    jurisdiction: Jurisdiction,
    since_date: date = Query(...),
):
    """Get regulatory updates since specified date."""
    try:
        updates = regtech.track_regulations(jurisdiction, since_date)
        return [RegulationUpdate(**u) for u in updates]
    except Exception as e:
        logger.error(f"Error tracking regulations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compliance/gaps", response_model=List[ComplianceGap])
async def analyze_compliance(
    entity_id: str = Query(...),
    jurisdiction: Jurisdiction = Query(...),
):
    """Analyze compliance gaps for an entity."""
    try:
        gaps = regtech.analyze_compliance_gaps(entity_id, jurisdiction)
        return [ComplianceGap(**g) for g in gaps]
    except Exception as e:
        logger.error(f"Error analyzing compliance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reports/generate", response_model=ComplianceReport)
async def generate_report(
    report_type: str = Query(...),
    entity_id: str = Query(...),
    reporting_period: str = Query(...),
):
    """Generate automated compliance report."""
    try:
        report = regtech.generate_compliance_report(report_type, entity_id, reporting_period)
        return ComplianceReport(**report)
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/penalty-risk/{entity_id}")
async def assess_penalty_risk(entity_id: str, jurisdiction: Jurisdiction):
    """Assess penalty risk from compliance gaps."""
    try:
        gaps = regtech.analyze_compliance_gaps(entity_id, jurisdiction)
        risk = regtech.assess_penalty_risk(entity_id, gaps)
        return risk
    except Exception as e:
        logger.error(f"Error assessing penalty risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))
