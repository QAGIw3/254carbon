"""
CAISO-specific curve scenarios for California market analysis.
Custom scenarios tailored for CAISO pilot customers including regulatory compliance and regional factors.
"""
import logging
from datetime import date, datetime, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from main import ScenarioSpec, execute_scenario_pipeline

logger = logging.getLogger(__name__)

# CAISO-specific router
caiso_scenarios_router = APIRouter(prefix="/api/v1/caiso/scenarios", tags=["CAISO Scenarios"])


class CAISOMarketCondition(str, Enum):
    """CAISO market conditions for scenario modeling."""
    NORMAL = "normal"
    HIGH_DEMAND = "high_demand"
    LOW_HYDRO = "low_hydro"
    HIGH_RENEWABLES = "high_renewables"
    TRANSMISSION_CONSTRAINTS = "transmission_constraints"
    REGULATORY_CHANGE = "regulatory_change"
    CARBON_POLICY = "carbon_policy"


class CAISORegion(str, Enum):
    """CAISO regions for localized analysis."""
    SP15 = "SP15"  # Southern California
    NP15 = "NP15"  # Northern California
    ZP26 = "ZP26"  # Pacific Northwest
    ALL = "ALL"    # Entire CAISO footprint


class CAISOCurveScenario(BaseModel):
    """CAISO-specific curve scenario configuration."""
    scenario_id: str
    name: str
    description: str
    market_condition: CAISOMarketCondition
    region: CAISORegion
    as_of_date: date
    horizon_months: int = 24
    include_regulatory_impact: bool = True
    include_carbon_pricing: bool = True
    include_transmission_constraints: bool = True
    custom_assumptions: Optional[Dict[str, Any]] = None


class CAISOCurveResult(BaseModel):
    """Results from CAISO curve scenario execution."""
    scenario_id: str
    run_id: str
    status: str
    curves: Dict[str, List[Dict[str, Any]]]  # instrument_id -> curve points
    summary_metrics: Dict[str, Any]
    regional_insights: Dict[str, Any]
    regulatory_impact: Optional[Dict[str, Any]] = None
    completed_at: datetime


class CAISORegulatoryScenario(BaseModel):
    """CAISO regulatory compliance scenarios."""
    scenario_name: str
    regulation_type: str  # "resource_adequacy", "renewable_portfolio", "carbon_cap", etc.
    compliance_year: int
    target_percentage: float
    penalty_cost: float
    implementation_timeline: List[Dict[str, Any]]


# CAISO-specific scenario templates

CAISO_SCENARIO_TEMPLATES = {
    CAISOMarketCondition.NORMAL: {
        "macro": {
            "gdp_growth": 0.025,
            "inflation": 0.022,
            "interest_rate": 0.045
        },
        "fuels": {
            "natural_gas_price": 4.50,  # $/MMBtu
            "coal_price": 65.00,       # $/ton
        },
        "power": {
            "load_growth": 0.015,      # Annual load growth
            "renewable_penetration": 0.35,
            "battery_storage_growth": 0.20
        },
        "policy": {
            "carbon_price": 0.0,       # $/ton CO2
            "renewable_mandate": 0.60  # RPS target
        }
    },

    CAISOMarketCondition.HIGH_DEMAND: {
        "macro": {
            "gdp_growth": 0.035,
            "inflation": 0.025,
            "interest_rate": 0.05
        },
        "fuels": {
            "natural_gas_price": 5.50,
            "coal_price": 75.00,
        },
        "power": {
            "load_growth": 0.025,
            "peak_demand_multiplier": 1.15,
            "renewable_penetration": 0.30
        },
        "policy": {
            "carbon_price": 25.0,
            "renewable_mandate": 0.70
        }
    },

    CAISOMarketCondition.LOW_HYDRO: {
        "macro": {
            "gdp_growth": 0.025,
            "inflation": 0.022,
            "interest_rate": 0.045
        },
        "fuels": {
            "natural_gas_price": 5.00,
            "hydro_availability": 0.65  # Reduced hydro generation
        },
        "power": {
            "load_growth": 0.015,
            "renewable_penetration": 0.35,
            "hydro_capacity_factor": 0.65
        },
        "policy": {
            "carbon_price": 15.0,
            "renewable_mandate": 0.65
        }
    },

    CAISOMarketCondition.HIGH_RENEWABLES: {
        "macro": {
            "gdp_growth": 0.025,
            "inflation": 0.022,
            "interest_rate": 0.045
        },
        "fuels": {
            "natural_gas_price": 4.00,
            "solar_cost_decline": 0.15,
            "wind_cost_decline": 0.12
        },
        "power": {
            "load_growth": 0.015,
            "renewable_penetration": 0.50,
            "battery_storage_growth": 0.35,
            "solar_capacity_factor": 0.28,
            "wind_capacity_factor": 0.35
        },
        "policy": {
            "carbon_price": 35.0,
            "renewable_mandate": 0.80,
            "battery_incentives": 0.15
        }
    },

    CAISOMarketCondition.TRANSMISSION_CONSTRAINTS: {
        "macro": {
            "gdp_growth": 0.025,
            "inflation": 0.022,
            "interest_rate": 0.045
        },
        "fuels": {
            "natural_gas_price": 4.50,
        },
        "power": {
            "load_growth": 0.015,
            "transmission_capacity": 0.85,  # Reduced transmission capacity
            "congestion_frequency": 0.25,
            "renewable_penetration": 0.35
        },
        "policy": {
            "carbon_price": 20.0,
            "transmission_investment": 0.10
        }
    },

    CAISOMarketCondition.REGULATORY_CHANGE: {
        "macro": {
            "gdp_growth": 0.025,
            "inflation": 0.022,
            "interest_rate": 0.045
        },
        "fuels": {
            "natural_gas_price": 4.50,
        },
        "power": {
            "load_growth": 0.015,
            "renewable_penetration": 0.35
        },
        "policy": {
            "carbon_price": 45.0,
            "renewable_mandate": 0.90,
            "capacity_market_reform": True,
            "resource_adequacy_changes": True
        }
    },

    CAISOMarketCondition.CARBON_POLICY: {
        "macro": {
            "gdp_growth": 0.025,
            "inflation": 0.022,
            "interest_rate": 0.045
        },
        "fuels": {
            "natural_gas_price": 4.50,
            "carbon_capture_cost": 50.0
        },
        "power": {
            "load_growth": 0.015,
            "renewable_penetration": 0.35,
            "ccs_capacity": 0.05
        },
        "policy": {
            "carbon_price": 75.0,
            "renewable_mandate": 0.85,
            "carbon_capture_subsidy": 0.20,
            "clean_energy_standard": True
        }
    }
}


# CAISO-specific endpoints

@caiso_scenarios_router.post("/generate", response_model=CAISOCurveResult)
async def generate_caiso_curve_scenario(
    scenario_config: CAISOCurveScenario,
    user=Depends(verify_token),
):
    """Generate CAISO-specific curve scenario."""
    try:
        # Create scenario specification from template
        template = CAISO_SCENARIO_TEMPLATES[scenario_config.market_condition]

        # Apply regional overrides if specified
        regional_overrides = {}
        if scenario_config.region != CAISORegion.ALL:
            regional_overrides = get_regional_overrides(scenario_config.region)

        # Merge templates with regional overrides and custom assumptions
        scenario_spec_dict = {
            **template,
            **regional_overrides,
            **(scenario_config.custom_assumptions or {})
        }

        # Create scenario spec
        spec = ScenarioSpec(
            as_of_date=scenario_config.as_of_date,
            macro=scenario_spec_dict.get("macro", {}),
            fuels=scenario_spec_dict.get("fuels", {}),
            power=scenario_spec_dict.get("power", {}),
            policy=scenario_spec_dict.get("policy", {}),
            market_overrides={
                "market": "CAISO",
                "region": scenario_config.region.value,
                "horizon_months": scenario_config.horizon_months
            }
        )

        # Execute scenario
        run_id = f"CAISO_{scenario_config.scenario_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # For now, return mock results structure
        # In real implementation, this would call execute_scenario_pipeline
        result = CAISOCurveResult(
            scenario_id=scenario_config.scenario_id,
            run_id=run_id,
            status="completed",
            curves={
                "CAISO.SP15": [
                    {
                        "delivery_date": (scenario_config.as_of_date + timedelta(days=30*i)).isoformat(),
                        "price": 45.0 + i * 0.5,
                        "confidence": 0.85 - i * 0.01
                    }
                    for i in range(min(scenario_config.horizon_months, 24))
                ]
            },
            summary_metrics={
                "avg_price_12m": 48.50,
                "avg_price_24m": 52.25,
                "price_volatility": 0.15,
                "renewable_correlation": 0.72,
                "load_correlation": 0.68
            },
            regional_insights={
                "sp15_avg_price": 47.50,
                "np15_avg_price": 49.25,
                "transmission_basis": 1.75,
                "congestion_risk": 0.23
            },
            regulatory_impact={
                "carbon_price_impact": 3.25,
                "rps_compliance_cost": 1.50,
                "capacity_market_impact": 2.00
            } if scenario_config.include_regulatory_impact else None,
            completed_at=datetime.utcnow()
        )

        return result

    except Exception as e:
        logger.error(f"Error generating CAISO curve scenario: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@caiso_scenarios_router.get("/templates", response_model=List[Dict[str, Any]])
async def get_caiso_scenario_templates(user=Depends(verify_token)):
    """Get available CAISO scenario templates."""
    templates = []

    for condition, config in CAISO_SCENARIO_TEMPLATES.items():
        templates.append({
            "condition": condition.value,
            "name": condition.value.replace("_", " ").title(),
            "description": get_condition_description(condition),
            "key_assumptions": config,
            "suitable_for": get_suitable_regions(condition)
        })

    return templates


@caiso_scenarios_router.post("/regulatory-scenario")
async def create_caiso_regulatory_scenario(
    scenario: CAISORegulatoryScenario,
    user=Depends(verify_token),
):
    """Create CAISO regulatory compliance scenario."""
    try:
        # Create regulatory scenario specification
        regulatory_spec = {
            "regulation_type": scenario.regulation_type,
            "compliance_year": scenario.compliance_year,
            "target_percentage": scenario.target_percentage,
            "penalty_cost": scenario.penalty_cost,
            "implementation_timeline": scenario.implementation_timeline,
            "caiso_specific": {
                "resource_adequacy_impact": True,
                "renewable_integration_required": True,
                "transmission_planning_impact": True
            }
        }

        # Create scenario spec for regulatory analysis
        spec = ScenarioSpec(
            as_of_date=datetime.utcnow().date(),
            macro={"gdp_growth": 0.025, "inflation": 0.022},
            fuels={"natural_gas_price": 4.50},
            power={"load_growth": 0.015, "renewable_penetration": 0.35},
            policy={
                "carbon_price": 25.0,
                "renewable_mandate": scenario.target_percentage,
                "regulatory_scenario": regulatory_spec
            },
            market_overrides={
                "market": "CAISO",
                "regulatory_focus": True,
                "compliance_analysis": True
            }
        )

        # Execute regulatory scenario
        run_id = f"CAISO_REG_{scenario.scenario_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Mock results for regulatory analysis
        result = {
            "scenario_id": scenario.scenario_name,
            "run_id": run_id,
            "status": "completed",
            "regulatory_analysis": {
                "compliance_cost": scenario.penalty_cost * (1 - scenario.target_percentage),
                "implementation_phases": len(scenario.implementation_timeline),
                "total_impact": scenario.penalty_cost * scenario.target_percentage,
                "risk_assessment": {
                    "penalty_risk": "high" if scenario.penalty_cost > 100000 else "medium",
                    "implementation_risk": "medium",
                    "market_risk": "low"
                }
            },
            "curve_impact": {
                "avg_price_increase": 2.50,
                "price_volatility_increase": 0.05,
                "capacity_price_impact": 15.0
            },
            "recommendations": [
                "Accelerate renewable procurement to meet RPS targets",
                "Consider capacity market participation for RA compliance",
                "Monitor transmission constraints for compliance delivery"
            ],
            "completed_at": datetime.utcnow().isoformat()
        }

        return result

    except Exception as e:
        logger.error(f"Error creating CAISO regulatory scenario: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def get_regional_overrides(region: CAISORegion) -> Dict[str, Any]:
    """Get regional overrides for CAISO regions."""
    regional_configs = {
        CAISORegion.SP15: {
            "power": {
                "load_growth": 0.020,  # Higher growth in Southern California
                "renewable_penetration": 0.40,
                "solar_capacity_factor": 0.32,
                "transmission_import_dependence": 0.15
            }
        },
        CAISORegion.NP15: {
            "power": {
                "load_growth": 0.012,  # Lower growth in Northern California
                "renewable_penetration": 0.45,
                "hydro_capacity_factor": 0.75,
                "wind_capacity_factor": 0.28
            }
        },
        CAISORegion.ZP26: {
            "power": {
                "load_growth": 0.008,  # Lowest growth in Pacific Northwest
                "renewable_penetration": 0.55,
                "hydro_capacity_factor": 0.80,
                "wind_capacity_factor": 0.35,
                "transmission_export_capacity": 0.25
            }
        }
    }

    return regional_configs.get(region, {})


def get_condition_description(condition: CAISOMarketCondition) -> str:
    """Get human-readable description for market condition."""
    descriptions = {
        CAISOMarketCondition.NORMAL: "Standard market conditions with baseline assumptions",
        CAISOMarketCondition.HIGH_DEMAND: "Elevated demand scenario with economic growth acceleration",
        CAISOMarketCondition.LOW_HYDRO: "Reduced hydroelectric generation due to drought conditions",
        CAISOMarketCondition.HIGH_RENEWABLES: "Accelerated renewable energy deployment scenario",
        CAISOMarketCondition.TRANSMISSION_CONSTRAINTS: "Increased transmission congestion and capacity limitations",
        CAISOMarketCondition.REGULATORY_CHANGE: "Major regulatory reforms and policy shifts",
        CAISOMarketCondition.CARBON_POLICY: "Aggressive carbon pricing and clean energy policies"
    }

    return descriptions.get(condition, "Custom market condition")


def get_suitable_regions(condition: CAISOMarketCondition) -> List[str]:
    """Get regions most suitable for each market condition."""
    suitability = {
        CAISOMarketCondition.NORMAL: [CAISORegion.ALL],
        CAISOMarketCondition.HIGH_DEMAND: [CAISORegion.SP15, CAISORegion.ALL],
        CAISOMarketCondition.LOW_HYDRO: [CAISORegion.SP15, CAISORegion.ALL],
        CAISOMarketCondition.HIGH_RENEWABLES: [CAISORegion.SP15, CAISORegion.NP15, CAISORegion.ALL],
        CAISOMarketCondition.TRANSMISSION_CONSTRAINTS: [CAISORegion.SP15, CAISORegion.ALL],
        CAISOMarketCondition.REGULATORY_CHANGE: [CAISORegion.ALL],
        CAISOMarketCondition.CARBON_POLICY: [CAISORegion.ALL]
    }

    return [region.value for region in suitability.get(condition, [CAISORegion.ALL])]
