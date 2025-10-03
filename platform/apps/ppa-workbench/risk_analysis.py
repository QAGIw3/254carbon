"""
Risk analysis for PPA contracts.
"""
import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PPARiskAnalyzer:
    """Analyze risks in PPA contracts."""
    
    def calculate_ppa_risk(
        self,
        contract,
        cash_flows: List[Dict],
        forward_curve: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics for PPA.
        
        Risks:
        - Price risk (for index-based PPAs)
        - Volume risk (for renewable PPAs)
        - Basis risk (for hub+ PPAs)
        - Shape risk (generation profile vs price profile)
        - Counterparty credit risk
        """
        risks = {}
        
        # Price risk
        if contract.contract_type in ["index", "hub_plus", "collar"]:
            risks["price_risk"] = self._calculate_price_risk(
                contract,
                forward_curve,
            )
        else:
            risks["price_risk"] = {
                "exposure": "none",
                "type": "fixed_price",
            }
        
        # Volume risk
        if contract.shape_profile in ["solar", "wind"]:
            risks["volume_risk"] = self._calculate_volume_risk(contract)
        else:
            risks["volume_risk"] = {
                "exposure": "low",
                "type": "baseload",
            }
        
        # Basis risk
        if contract.contract_type == "hub_plus":
            risks["basis_risk"] = self._calculate_basis_risk(contract)
        else:
            risks["basis_risk"] = {"exposure": "none"}
        
        # Shape risk
        risks["shape_risk"] = self._calculate_shape_risk(contract)
        
        # NPV volatility
        risks["npv_volatility"] = self._calculate_npv_volatility(cash_flows)
        
        return risks
    
    def _calculate_price_risk(
        self,
        contract,
        forward_curve: Optional[pd.Series],
    ) -> Dict:
        """Calculate price risk exposure."""
        if forward_curve is None:
            return {"exposure": "unknown"}
        
        # Calculate price volatility
        returns = forward_curve.pct_change().dropna()
        annual_vol = returns.std() * np.sqrt(12)  # Monthly to annual
        
        # Exposure depends on contract type
        if contract.contract_type == "collar":
            # Limited by floor and cap
            exposure_factor = 0.3  # 30% exposure (remaining after collar)
        else:
            exposure_factor = 1.0  # Full exposure
        
        # Annual value at risk (simplified)
        annual_revenue = contract.annual_mwh * (contract.fixed_price or forward_curve.mean())
        price_var = annual_revenue * annual_vol * exposure_factor
        
        return {
            "exposure": "high" if exposure_factor > 0.7 else "medium",
            "annual_volatility": float(annual_vol),
            "annual_var_95": float(price_var * 1.645),
            "exposure_factor": exposure_factor,
        }
    
    def _calculate_volume_risk(self, contract) -> Dict:
        """Calculate volume/generation risk."""
        # Renewable generation has inter-annual variability
        
        if contract.shape_profile == "solar":
            cv = 0.10  # 10% coefficient of variation
        elif contract.shape_profile == "wind":
            cv = 0.15  # 15% CV
        else:
            cv = 0.02  # Very low for baseload
        
        # Volume at risk
        expected_mwh = contract.annual_mwh
        volume_std = expected_mwh * cv
        volume_var_95 = volume_std * 1.645
        
        return {
            "exposure": "high" if cv > 0.10 else "low",
            "coefficient_of_variation": cv,
            "expected_annual_mwh": expected_mwh,
            "volume_var_95_mwh": float(volume_var_95),
        }
    
    def _calculate_basis_risk(self, contract) -> Dict:
        """Calculate nodal basis risk."""
        # Risk that nodal price diverges from hub
        
        # Typical basis volatility: 20-30% of price level
        basis_vol = 0.25
        
        return {
            "exposure": "medium",
            "basis_adder": contract.basis_adder,
            "basis_volatility": basis_vol,
            "note": "Node may decouple from hub during congestion",
        }
    
    def _calculate_shape_risk(self, contract) -> Dict:
        """Calculate shape mismatch risk."""
        if contract.shape_profile in ["solar", "wind"]:
            return {
                "exposure": "high",
                "type": contract.shape_profile,
                "note": "Generation profile may not match peak prices",
            }
        else:
            return {
                "exposure": "low",
                "type": "baseload",
            }
    
    def _calculate_npv_volatility(self, cash_flows: List[Dict]) -> Dict:
        """Calculate NPV volatility."""
        pvs = [cf["pv"] for cf in cash_flows]
        
        return {
            "npv_std": float(np.std(pvs)),
            "coefficient_of_variation": float(np.std(pvs) / np.mean(pvs)) if np.mean(pvs) > 0 else 0,
        }

