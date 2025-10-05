"""
Base class for market adapters.
"""
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class MarketAdapter(ABC):
    """
    Base class for all market adapters.
    
    Provides common functionality and enforces consistent interface.
    """
    
    def __init__(self, market_name: str):
        """
        Initialize market adapter.
        
        Args:
            market_name: Name of the market (e.g., "MISO", "CAISO")
        """
        self.market_name = market_name
        self.logger = logging.getLogger(f"adapter.{market_name.lower()}")
        self.logger.info(f"{market_name} adapter initialized")
    
    async def validate_entitlement(
        self,
        user: dict,
        resource: str,
        channel: str = "api",
    ) -> bool:
        """
        Validate user entitlement for market-specific resource.
        
        Args:
            user: User claims from authentication
            resource: Resource identifier (instrument, product, etc.)
            channel: Access channel (api, downloads, stream, hub)
        
        Returns:
            bool: True if user is entitled to access resource
        """
        # Default implementation - override in subclasses for custom logic
        tenant_id = user.get("tenant_id")
        if not tenant_id:
            self.logger.warning("User has no tenant_id")
            return False
        
        # Basic validation - can be overridden
        return True
    
    def get_market_config(self) -> Dict[str, Any]:
        """
        Get market-specific configuration.
        
        Returns:
            dict: Market configuration
        """
        return {
            "market_name": self.market_name,
            "supported_products": self.get_supported_products(),
            "features": self.get_features(),
        }
    
    @abstractmethod
    def get_supported_products(self) -> list:
        """
        Get list of products supported by this adapter.
        
        Returns:
            list: Product identifiers
        """
        pass
    
    @abstractmethod
    def get_features(self) -> list:
        """
        Get list of features provided by this adapter.
        
        Returns:
            list: Feature names
        """
        pass
    
    def log_metric(self, metric_name: str, value: Any):
        """
        Log adapter-specific metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
        """
        self.logger.info(f"Metric {metric_name}: {value}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform adapter health check.
        
        Returns:
            dict: Health status
        """
        return {
            "adapter": self.market_name,
            "status": "healthy",
        }

