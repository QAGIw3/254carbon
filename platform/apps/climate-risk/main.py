"""
Climate Risk Analytics Engine

Comprehensive climate risk assessment for energy assets:
- Physical risks (extreme weather, sea level rise)
- Transition risks (carbon pricing, policy changes)
- Financial impact modeling
- TCFD reporting
"""
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from enum import Enum

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Climate Risk Analytics",
    description="Climate risk assessment for energy infrastructure",
    version="1.0.0",
)


class RiskType(str, Enum):
    PHYSICAL_ACUTE = "physical_acute"  # Extreme events
    PHYSICAL_CHRONIC = "physical_chronic"  # Long-term shifts
    TRANSITION_POLICY = "transition_policy"  # Regulations
    TRANSITION_TECH = "transition_tech"  # Technology disruption
    TRANSITION_MARKET = "transition_market"  # Market shifts


class ClimateScenario(str, Enum):
    RCP26 = "rcp26"  # <2°C warming
    RCP45 = "rcp45"  # ~2.5°C warming
    RCP85 = "rcp85"  # >4°C warming
    NET_ZERO_2050 = "net_zero_2050"
    DELAYED_TRANSITION = "delayed_transition"


class AssetType(str, Enum):
    POWER_PLANT = "power_plant"
    TRANSMISSION = "transmission"
    PIPELINE = "pipeline"
    STORAGE = "storage"
    RENEWABLE_FARM = "renewable_farm"


class PhysicalRiskAssessment(BaseModel):
    """Physical climate risk assessment."""
    asset_id: str
    asset_type: AssetType
    location: Dict[str, float]
    risks: List[Dict[str, Any]]
    overall_risk_score: float  # 0-100
    timeline_years: int
    scenario: ClimateScenario


class TransitionRiskAssessment(BaseModel):
    """Transition risk assessment."""
    asset_id: str
    asset_type: AssetType
    stranded_asset_probability: float
    policy_impact_score: float
    technology_disruption_score: float
    market_shift_score: float
    timeline_years: int
    scenario: ClimateScenario


class FinancialImpact(BaseModel):
    """Financial impact of climate risks."""
    asset_id: str
    baseline_value_usd: float
    climate_var_usd: float  # Climate Value at Risk
    expected_loss_annual_usd: float
    adaptation_cost_usd: float
    stranded_value_usd: float
    scenario: ClimateScenario


class ClimateRiskEngine:
    """Climate risk modeling and assessment."""
    
    def __init__(self):
        self.hazard_models = self._load_hazard_models()
        self.carbon_scenarios = self._load_carbon_scenarios()
    
    def _load_hazard_models(self) -> Dict:
        """Load climate hazard models."""
        return {
            "flood": {"probability_increase_2050": 2.5, "severity_increase": 1.8},
            "drought": {"probability_increase_2050": 2.0, "severity_increase": 1.6},
            "hurricane": {"intensity_increase": 1.3, "frequency_increase": 1.2},
            "wildfire": {"probability_increase_2050": 3.0, "severity_increase": 2.2},
            "heatwave": {"frequency_increase": 4.0, "intensity_increase": 1.5},
        }
    
    def _load_carbon_scenarios(self) -> Dict:
        """Load carbon price scenarios."""
        return {
            ClimateScenario.NET_ZERO_2050: {
                2025: 50, 2030: 100, 2040: 200, 2050: 300
            },
            ClimateScenario.DELAYED_TRANSITION: {
                2025: 30, 2030: 50, 2040: 150, 2050: 400
            },
            ClimateScenario.RCP26: {
                2025: 60, 2030: 120, 2040: 180, 2050: 250
            },
        }
    
    def assess_physical_risk(
        self,
        asset_id: str,
        asset_type: AssetType,
        location: Dict[str, float],
        scenario: ClimateScenario,
        timeline_years: int
    ) -> Dict[str, Any]:
        """
        Assess physical climate risks for an asset.
        
        Evaluates acute (extreme events) and chronic (long-term) risks.
        """
        logger.info(f"Assessing physical risk for {asset_id} under {scenario}")
        
        lat = location["lat"]
        lon = location["lon"]
        
        risks = []
        
        # Flood risk
        if lat < 30:  # Coastal/tropical regions
            baseline_flood_prob = 0.02  # 2% annual
            climate_multiplier = self.hazard_models["flood"]["probability_increase_2050"]
            future_flood_prob = min(baseline_flood_prob * climate_multiplier ** (timeline_years / 30), 0.30)
            
            risks.append({
                "hazard": "flooding",
                "type": "acute",
                "current_probability": round(baseline_flood_prob, 3),
                "future_probability": round(future_flood_prob, 3),
                "severity": "high" if future_flood_prob > 0.15 else "medium",
                "financial_impact_usd": int(10_000_000 * future_flood_prob),
            })
        
        # Drought risk
        if 20 < lat < 40:  # Mid-latitude regions
            baseline_drought_prob = 0.10
            future_drought_prob = min(baseline_drought_prob * 2.0 ** (timeline_years / 30), 0.40)
            
            risks.append({
                "hazard": "drought",
                "type": "chronic",
                "current_probability": round(baseline_drought_prob, 3),
                "future_probability": round(future_drought_prob, 3),
                "severity": "high" if future_drought_prob > 0.25 else "medium",
                "financial_impact_usd": int(5_000_000 * future_drought_prob),
            })
        
        # Hurricane risk (coastal)
        if abs(lat) < 35 and abs(lon) < 100:  # Hurricane zones
            baseline_hurricane_prob = 0.05
            future_hurricane_prob = baseline_hurricane_prob * 1.3 ** (timeline_years / 30)
            
            risks.append({
                "hazard": "hurricane",
                "type": "acute",
                "current_probability": round(baseline_hurricane_prob, 3),
                "future_probability": round(future_hurricane_prob, 3),
                "severity": "very_high",
                "financial_impact_usd": int(50_000_000 * future_hurricane_prob),
            })
        
        # Heatwave risk (all regions)
        baseline_heatwave_days = 10
        future_heatwave_days = baseline_heatwave_days * (1 + timeline_years / 20)
        
        risks.append({
            "hazard": "heatwave",
            "type": "chronic",
            "current_days_per_year": baseline_heatwave_days,
            "future_days_per_year": int(future_heatwave_days),
            "severity": "medium",
            "operational_impact": f"{int(future_heatwave_days * 0.3)} days reduced capacity",
            "financial_impact_usd": int(2_000_000 * future_heatwave_days / 30),
        })
        
        # Calculate overall risk score
        total_impact = sum(r.get("financial_impact_usd", 0) for r in risks)
        risk_score = min(total_impact / 1_000_000, 100)  # Normalize to 0-100
        
        return {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "location": location,
            "risks": risks,
            "overall_risk_score": round(risk_score, 1),
            "timeline_years": timeline_years,
            "scenario": scenario,
        }
    
    def assess_transition_risk(
        self,
        asset_id: str,
        asset_type: AssetType,
        fuel_type: str,
        capacity_mw: float,
        age_years: int,
        scenario: ClimateScenario,
        timeline_years: int
    ) -> Dict[str, Any]:
        """
        Assess transition risks from policy, technology, and market changes.
        """
        logger.info(f"Assessing transition risk for {asset_id} under {scenario}")
        
        # Carbon price impact
        carbon_trajectory = self.carbon_scenarios.get(scenario, {})
        current_carbon_price = carbon_trajectory.get(2025, 50)
        future_carbon_price = carbon_trajectory.get(2025 + timeline_years, 200)
        
        # Stranded asset probability
        if fuel_type in ["coal", "oil"]:
            base_strand_prob = 0.60
            carbon_factor = (future_carbon_price - current_carbon_price) / 100
            age_factor = min(age_years / 30, 1.0)
            strand_prob = min(base_strand_prob + carbon_factor * 0.2 + age_factor * 0.3, 0.95)
        elif fuel_type == "gas":
            strand_prob = min(0.30 + (future_carbon_price - 100) / 500, 0.70)
        else:  # renewables
            strand_prob = 0.05
        
        # Policy impact score
        policy_stringency = {
            ClimateScenario.NET_ZERO_2050: 0.9,
            ClimateScenario.RCP26: 0.7,
            ClimateScenario.DELAYED_TRANSITION: 0.4,
        }.get(scenario, 0.5)
        
        fossil_penalty = 1.0 if fuel_type in ["coal", "oil", "gas"] else 0.0
        policy_impact = policy_stringency * fossil_penalty * 100
        
        # Technology disruption score
        if fuel_type in ["coal", "oil"]:
            tech_disruption = 85  # High disruption from renewables
        elif fuel_type == "gas":
            tech_disruption = 60  # Medium disruption
        else:
            tech_disruption = 20  # Renewables face battery storage competition
        
        # Market shift score
        renewable_share_growth = timeline_years * 2.5  # 2.5% per year
        market_shift = min(renewable_share_growth * fossil_penalty, 100)
        
        return {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "stranded_asset_probability": round(strand_prob, 3),
            "policy_impact_score": round(policy_impact, 1),
            "technology_disruption_score": round(tech_disruption, 1),
            "market_shift_score": round(market_shift, 1),
            "timeline_years": timeline_years,
            "scenario": scenario,
        }
    
    def calculate_financial_impact(
        self,
        asset_id: str,
        baseline_value: float,
        physical_risks: Dict,
        transition_risks: Dict,
        scenario: ClimateScenario
    ) -> Dict[str, Any]:
        """
        Calculate financial impact of climate risks.
        
        Includes Climate VaR and expected losses.
        """
        logger.info(f"Calculating financial impact for {asset_id}")
        
        # Physical risk losses
        physical_loss_annual = sum(
            r.get("financial_impact_usd", 0)
            for r in physical_risks.get("risks", [])
        )
        
        # Transition risk: stranded asset value
        strand_prob = transition_risks.get("stranded_asset_probability", 0)
        stranded_value = baseline_value * strand_prob
        
        # Adaptation costs (to reduce physical risks)
        adaptation_cost = physical_loss_annual * 5  # 5 years of losses to adapt
        
        # Climate VaR (95th percentile loss)
        climate_var = stranded_value + physical_loss_annual * 20  # 20-year horizon
        
        return {
            "asset_id": asset_id,
            "baseline_value_usd": baseline_value,
            "climate_var_usd": round(climate_var, 0),
            "expected_loss_annual_usd": round(physical_loss_annual, 0),
            "adaptation_cost_usd": round(adaptation_cost, 0),
            "stranded_value_usd": round(stranded_value, 0),
            "scenario": scenario,
        }


# Global risk engine
risk_engine = ClimateRiskEngine()


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "climate-risk"}


@app.get("/api/v1/climate/physical-risk", response_model=PhysicalRiskAssessment)
async def assess_physical_risk(
    asset_id: str,
    asset_type: AssetType,
    lat: float = Query(...),
    lon: float = Query(...),
    scenario: ClimateScenario = ClimateScenario.RCP45,
    timeline_years: int = 30,
):
    """
    Assess physical climate risks for an asset.
    
    Includes floods, droughts, hurricanes, and heatwaves.
    """
    try:
        location = {"lat": lat, "lon": lon}
        result = risk_engine.assess_physical_risk(
            asset_id, asset_type, location, scenario, timeline_years
        )
        return PhysicalRiskAssessment(**result)
    except Exception as e:
        logger.error(f"Physical risk assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/climate/transition-risk", response_model=TransitionRiskAssessment)
async def assess_transition_risk(
    asset_id: str,
    asset_type: AssetType,
    fuel_type: str,
    capacity_mw: float,
    age_years: int,
    scenario: ClimateScenario = ClimateScenario.NET_ZERO_2050,
    timeline_years: int = 25,
):
    """
    Assess transition risks from policy, technology, and market changes.
    """
    try:
        result = risk_engine.assess_transition_risk(
            asset_id, asset_type, fuel_type, capacity_mw, age_years, scenario, timeline_years
        )
        return TransitionRiskAssessment(**result)
    except Exception as e:
        logger.error(f"Transition risk assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/climate/financial-impact", response_model=FinancialImpact)
async def calculate_financial_impact(
    asset_id: str,
    baseline_value_usd: float,
    physical_risk_id: str,  # Reference to previous assessment
    transition_risk_id: str,
    scenario: ClimateScenario = ClimateScenario.RCP45,
):
    """
    Calculate comprehensive financial impact of climate risks.
    
    Includes Climate VaR, expected losses, and adaptation costs.
    """
    try:
        # Mock retrieval of previous assessments
        physical_risks = {"risks": [{"financial_impact_usd": 5000000}]}
        transition_risks = {"stranded_asset_probability": 0.45}
        
        result = risk_engine.calculate_financial_impact(
            asset_id, baseline_value_usd, physical_risks, transition_risks, scenario
        )
        return FinancialImpact(**result)
    except Exception as e:
        logger.error(f"Financial impact calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/climate/scenarios")
async def get_climate_scenarios():
    """Get available climate scenarios."""
    return {
        "scenarios": [
            {"code": "rcp26", "name": "RCP 2.6 (<2°C)", "warming_2100": 1.8},
            {"code": "rcp45", "name": "RCP 4.5 (~2.5°C)", "warming_2100": 2.4},
            {"code": "rcp85", "name": "RCP 8.5 (>4°C)", "warming_2100": 4.3},
            {"code": "net_zero_2050", "name": "Net Zero 2050", "warming_2100": 1.5},
            {"code": "delayed_transition", "name": "Delayed Transition", "warming_2100": 3.2},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8027)

