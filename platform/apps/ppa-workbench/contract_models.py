"""
PPA contract modeling classes.
"""
import logging
from typing import List, Dict
import numpy as np

logger = logging.getLogger(__name__)


class PPAContractBase:
    """Base class for PPA contracts."""
    
    def __init__(self, contract_spec: Dict):
        self.contract_id = contract_spec["contract_id"]
        self.start_date = contract_spec["start_date"]
        self.end_date = contract_spec["end_date"]
        self.capacity_mw = contract_spec["capacity_mw"]
        self.annual_mwh = contract_spec["annual_mwh"]
    
    def calculate_payment(
        self,
        market_price: float,
        delivered_mwh: float,
    ) -> float:
        """Calculate payment based on contract terms."""
        raise NotImplementedError


class FixedPricePPA(PPAContractBase):
    """Fixed price PPA contract."""
    
    def __init__(self, contract_spec: Dict):
        super().__init__(contract_spec)
        self.fixed_price = contract_spec["fixed_price"]
    
    def calculate_payment(
        self,
        market_price: float,
        delivered_mwh: float,
    ) -> float:
        """Payment = delivered MWh × fixed price."""
        return delivered_mwh * self.fixed_price


class CollarPPA(PPAContractBase):
    """Collar PPA with floor and cap."""
    
    def __init__(self, contract_spec: Dict):
        super().__init__(contract_spec)
        self.floor_price = contract_spec["floor_price"]
        self.cap_price = contract_spec["cap_price"]
    
    def calculate_payment(
        self,
        market_price: float,
        delivered_mwh: float,
    ) -> float:
        """
        Payment = delivered MWh × min(max(market_price, floor), cap).
        
        Provides downside protection and upside participation (capped).
        """
        effective_price = min(max(market_price, self.floor_price), self.cap_price)
        return delivered_mwh * effective_price


class HubPlusPPA(PPAContractBase):
    """Hub + basis PPA."""
    
    def __init__(self, contract_spec: Dict):
        super().__init__(contract_spec)
        self.index_hub = contract_spec["index_hub"]
        self.basis_adder = contract_spec["basis_adder"]
    
    def calculate_payment(
        self,
        market_price: float,
        delivered_mwh: float,
    ) -> float:
        """
        Payment = delivered MWh × (hub_price + basis_adder).
        
        Tracks market but with fixed basis.
        """
        # market_price should be hub price
        effective_price = market_price + self.basis_adder
        return delivered_mwh * effective_price


class ShapeRiskAnalyzer:
    """Analyze shape risk for PPAs."""
    
    @staticmethod
    def calculate_shape_risk(
        contract_profile: str,
        delivered_profile: np.ndarray,
        market_prices: np.ndarray,
    ) -> Dict:
        """
        Calculate shape risk.
        
        Shape risk = risk that generation doesn't match contract shape.
        """
        # Compare delivered to contract shape
        if contract_profile == "baseload":
            expected_shape = np.ones(len(delivered_profile)) / len(delivered_profile)
        elif contract_profile == "peak":
            # Peak hours (7am-11pm)
            expected_shape = np.array([
                1.0 if 7 <= h < 23 else 0.0 for h in range(24)
            ])
            expected_shape /= expected_shape.sum()
        else:
            expected_shape = delivered_profile / delivered_profile.sum()
        
        # Shape deviation
        shape_deviation = np.abs(
            delivered_profile / delivered_profile.sum() - expected_shape
        ).mean()
        
        # Value impact
        delivered_normalized = delivered_profile / delivered_profile.sum()
        value_delivered = np.sum(delivered_normalized * market_prices)
        value_expected = np.sum(expected_shape * market_prices)
        
        shape_penalty = value_expected - value_delivered
        
        return {
            "shape_deviation": float(shape_deviation),
            "shape_penalty": float(shape_penalty),
            "correlation_to_price": float(np.corrcoef(delivered_profile, market_prices)[0, 1]),
        }

