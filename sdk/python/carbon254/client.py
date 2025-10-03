"""
Main client class for 254Carbon API.
"""
import asyncio
from typing import List, Optional, Dict, Any
from datetime import date, datetime

import httpx
import pandas as pd

from .models import Instrument, PriceTick, ForwardCurve, Scenario, ScenarioRun
from .exceptions import CarbonAPIError, AuthenticationError, RateLimitError, NotFoundError


class CarbonClient:
    """
    254Carbon API Client
    
    Example:
        >>> from carbon254 import CarbonClient
        >>> client = CarbonClient(api_key="your_api_key")
        >>> instruments = client.get_instruments(market="power")
        >>> df = client.get_prices_dataframe("MISO.HUB.INDIANA", start_date="2025-01-01")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.254carbon.ai",
        timeout: int = 30,
    ):
        """
        Initialize Carbon client.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for API (default: production)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._get_headers(),
        )
        
        self._async_client = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers including authentication."""
        headers = {
            "User-Agent": f"carbon254-python-sdk/1.0.0",
            "Content-Type": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    def _handle_response(self, response: httpx.Response) -> Any:
        """Handle API response and raise appropriate exceptions."""
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            raise AuthenticationError("Invalid API key or unauthorized")
        elif response.status_code == 403:
            raise AuthenticationError("Access forbidden - check entitlements")
        elif response.status_code == 404:
            raise NotFoundError("Resource not found")
        elif response.status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        else:
            try:
                error_data = response.json()
                detail = error_data.get("detail", "Unknown error")
            except:
                detail = response.text
            
            raise CarbonAPIError(f"API error ({response.status_code}): {detail}")
    
    # Instruments API
    
    def get_instruments(
        self,
        market: Optional[str] = None,
        product: Optional[str] = None,
    ) -> List[Instrument]:
        """
        Get available instruments.
        
        Args:
            market: Filter by market (power, gas, env, lng)
            product: Filter by product (lmp, curve, rec, etc.)
        
        Returns:
            List of Instrument objects
        """
        params = {}
        if market:
            params["market"] = market
        if product:
            params["product"] = product
        
        response = self._client.get("/api/v1/instruments", params=params)
        data = self._handle_response(response)
        
        return [Instrument(**item) for item in data]
    
    # Prices API
    
    def get_prices(
        self,
        instrument_id: str,
        start_time: datetime,
        end_time: datetime,
        price_type: str = "mid",
    ) -> List[PriceTick]:
        """
        Get historical price ticks.
        
        Args:
            instrument_id: Instrument identifier
            start_time: Start datetime (UTC)
            end_time: End datetime (UTC)
            price_type: Price type (mid, bid, ask, settle)
        
        Returns:
            List of PriceTick objects
        """
        params = {
            "instrument_id": [instrument_id],
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "price_type": price_type,
        }
        
        response = self._client.get("/api/v1/prices/ticks", params=params)
        data = self._handle_response(response)
        
        return [PriceTick(**item) for item in data]
    
    def get_prices_dataframe(
        self,
        instrument_id: str,
        start_time: datetime,
        end_time: datetime,
        price_type: str = "mid",
    ) -> pd.DataFrame:
        """
        Get historical prices as pandas DataFrame.
        
        Args:
            instrument_id: Instrument identifier
            start_time: Start datetime (UTC)
            end_time: End datetime (UTC)
            price_type: Price type (mid, bid, ask, settle)
        
        Returns:
            pandas DataFrame with price data
        """
        ticks = self.get_prices(instrument_id, start_time, end_time, price_type)
        
        if not ticks:
            return pd.DataFrame()
        
        data = [
            {
                "timestamp": tick.event_time,
                "price": tick.value,
                "volume": tick.volume,
                "instrument_id": tick.instrument_id,
                "source": tick.source,
            }
            for tick in ticks
        ]
        
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        
        return df
    
    # Forward Curves API
    
    def get_forward_curve(
        self,
        instrument_id: str,
        as_of_date: date,
        scenario_id: str = "BASE",
    ) -> ForwardCurve:
        """
        Get forward curve.
        
        Args:
            instrument_id: Instrument identifier
            as_of_date: Curve as-of date
            scenario_id: Scenario ID (default: BASE)
        
        Returns:
            ForwardCurve object
        """
        params = {
            "instrument_id": [instrument_id],
            "as_of_date": as_of_date.isoformat(),
            "scenario_id": scenario_id,
        }
        
        response = self._client.get("/api/v1/curves/forward", params=params)
        points = self._handle_response(response)
        
        return ForwardCurve(
            instrument_id=instrument_id,
            as_of_date=as_of_date,
            scenario_id=scenario_id,
            points=points,
        )
    
    def get_curve_dataframe(
        self,
        instrument_id: str,
        as_of_date: date,
        scenario_id: str = "BASE",
    ) -> pd.DataFrame:
        """
        Get forward curve as pandas DataFrame.
        
        Args:
            instrument_id: Instrument identifier
            as_of_date: Curve as-of date
            scenario_id: Scenario ID
        
        Returns:
            pandas DataFrame with curve data
        """
        curve = self.get_forward_curve(instrument_id, as_of_date, scenario_id)
        
        df = pd.DataFrame(curve.points)
        if not df.empty:
            df["delivery_start"] = pd.to_datetime(df["delivery_start"])
            df.set_index("delivery_start", inplace=True)
        
        return df
    
    # Scenarios API
    
    def create_scenario(
        self,
        title: str,
        description: str,
        assumptions: Dict[str, Any],
    ) -> str:
        """
        Create new scenario.
        
        Args:
            title: Scenario title
            description: Scenario description
            assumptions: Scenario assumptions (DSL)
        
        Returns:
            Scenario ID
        """
        data = {
            "title": title,
            "description": description,
            "assumptions": assumptions,
        }
        
        response = self._client.post("/api/v1/scenarios", json=data)
        result = self._handle_response(response)
        
        return result["scenario_id"]
    
    def run_scenario(
        self,
        scenario_id: str,
    ) -> str:
        """
        Execute scenario run.
        
        Args:
            scenario_id: Scenario ID
        
        Returns:
            Run ID
        """
        response = self._client.post(f"/api/v1/scenarios/{scenario_id}/runs")
        result = self._handle_response(response)
        
        return result["run_id"]
    
    def get_run_status(
        self,
        scenario_id: str,
        run_id: str,
    ) -> Dict[str, Any]:
        """
        Get scenario run status.
        
        Args:
            scenario_id: Scenario ID
            run_id: Run ID
        
        Returns:
            Status dict
        """
        response = self._client.get(
            f"/api/v1/scenarios/{scenario_id}/runs/{run_id}"
        )
        return self._handle_response(response)
    
    # Async methods
    
    async def get_instruments_async(
        self,
        market: Optional[str] = None,
        product: Optional[str] = None,
    ) -> List[Instrument]:
        """Async version of get_instruments."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=self._get_headers(),
            )
        
        params = {}
        if market:
            params["market"] = market
        if product:
            params["product"] = product
        
        response = await self._async_client.get("/api/v1/instruments", params=params)
        data = self._handle_response(response)
        
        return [Instrument(**item) for item in data]
    
    # Context manager support
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Close HTTP client."""
        if self._client:
            self._client.close()
        if self._async_client:
            asyncio.run(self._async_client.aclose())

