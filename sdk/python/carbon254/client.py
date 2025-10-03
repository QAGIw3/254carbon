"""
Main client class for 254Carbon API.
"""
import asyncio
import json
import time
from typing import List, Optional, Dict, Any, Callable, Union
from datetime import date, datetime

import httpx
import pandas as pd
import websockets
from websockets.exceptions import ConnectionClosedError

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
        base_url: Optional[str] = None,
        timeout: int = 30,
        local_dev: bool = True,
    ):
        """
        Initialize Carbon client.

        Args:
            api_key: API key for authentication
            base_url: Base URL for API (auto-detected if None)
            timeout: Request timeout in seconds
            local_dev: Enable local development mode with mock data fallbacks
        """
        # Auto-detect environment
        if base_url is None:
            if local_dev:
                base_url = "http://localhost:8000"
                api_key = api_key or "dev-key"
            else:
                base_url = "https://api.254carbon.ai"

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.local_dev = local_dev

        # WebSocket URL for streaming
        self.ws_url = self.base_url.replace("http", "ws")

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._get_headers(),
        )

        self._async_client = None
        self._ws_client = None
    
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
        """Close HTTP and WebSocket clients."""
        if self._client:
            self._client.close()
        if self._async_client:
            asyncio.run(self._async_client.aclose())
        if self._ws_client:
            asyncio.run(self._ws_client.close())

    # WebSocket Streaming

    async def stream_prices(
        self,
        instrument_ids: List[str],
        callback: Callable[[PriceTick], None],
        reconnect: bool = True,
    ) -> None:
        """
        Stream real-time price updates via WebSocket.

        Args:
            instrument_ids: List of instrument IDs to subscribe to
            callback: Function to call with each price update
            reconnect: Auto-reconnect on connection loss
        """
        uri = f"{self.ws_url}/api/v1/stream"

        while True:
            try:
                async with websockets.connect(uri) as websocket:
                    # Subscribe to instruments
                    subscription = {
                        "type": "subscribe",
                        "instruments": instrument_ids,
                        "api_key": self.api_key
                    }
                    await websocket.send(json.dumps(subscription))

                    while True:
                        try:
                            message = await websocket.recv()
                            data = json.loads(message)

                            if data.get("type") == "price_update":
                                tick_data = data.get("data", {})
                                price_tick = PriceTick(**tick_data)
                                callback(price_tick)

                        except ConnectionClosedError:
                            break

            except Exception as e:
                if not reconnect:
                    raise
                print(f"WebSocket connection lost, reconnecting in 5 seconds: {e}")
                await asyncio.sleep(5)

    def stream_prices_sync(
        self,
        instrument_ids: List[str],
        callback: Callable[[PriceTick], None],
    ) -> None:
        """Synchronous wrapper for price streaming."""
        asyncio.run(self.stream_prices(instrument_ids, callback))

    # Advanced Analytics

    def get_portfolio_var(
        self,
        positions: List[Dict[str, Any]],
        confidence_level: float = 0.95,
        method: str = "historical",
        lookback_days: int = 252,
    ) -> Dict[str, Any]:
        """
        Calculate Value at Risk for a portfolio.

        Args:
            positions: List of position dicts with 'instrument_id' and 'quantity'
            confidence_level: Confidence level (0.95 = 95%)
            method: VaR method ('historical', 'parametric', 'monte_carlo')
            lookback_days: Historical lookback period

        Returns:
            Dict with VaR results
        """
        data = {
            "positions": positions,
            "confidence_level": confidence_level,
            "method": method,
            "lookback_days": lookback_days
        }

        try:
            response = self._client.post("/api/v1/risk/var", json=data)
            return self._handle_response(response)
        except (CarbonAPIError, httpx.TimeoutException):
            if self.local_dev:
                return self._generate_mock_var(positions, confidence_level)
            raise

    def get_correlation_matrix(
        self,
        instrument_ids: List[str],
        start_date: datetime,
        end_date: datetime,
        window: int = 30,
    ) -> pd.DataFrame:
        """
        Get correlation matrix for instruments.

        Args:
            instrument_ids: List of instrument IDs
            start_date: Start date for analysis
            end_date: End date for analysis
            window: Rolling window for correlation calculation

        Returns:
            pandas DataFrame with correlation matrix
        """
        params = {
            "instruments": instrument_ids,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "window": window
        }

        try:
            response = self._client.get("/api/v1/analytics/correlation", params=params)
            data = self._handle_response(response)

            # Convert to DataFrame
            df = pd.DataFrame(data)
            df.set_index("instrument_id", inplace=True)

            return df

        except (CarbonAPIError, httpx.TimeoutException):
            if self.local_dev:
                return self._generate_mock_correlation_matrix(instrument_ids)
            raise

    def get_price_forecast(
        self,
        instrument_id: str,
        horizon_days: int = 30,
        model_type: str = "ensemble",
    ) -> pd.DataFrame:
        """
        Get ML-based price forecasts.

        Args:
            instrument_id: Instrument to forecast
            horizon_days: Forecast horizon in days
            model_type: Model type ('ensemble', 'neural', 'tree')

        Returns:
            pandas DataFrame with forecast data
        """
        params = {
            "instrument_id": instrument_id,
            "horizon_days": horizon_days,
            "model_type": model_type
        }

        try:
            response = self._client.get("/api/v1/analytics/forecast", params=params)
            data = self._handle_response(response)

            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            return df

        except (CarbonAPIError, httpx.TimeoutException):
            if self.local_dev:
                return self._generate_mock_forecast(instrument_id, horizon_days)
            raise

    # Mock Data Generation for Local Development

    def _generate_mock_var(
        self,
        positions: List[Dict[str, Any]],
        confidence_level: float,
    ) -> Dict[str, Any]:
        """Generate mock VaR data for local development."""
        import random

        total_var = 0
        position_vars = []

        for position in positions:
            instrument_id = position["instrument_id"]
            quantity = position["quantity"]

            # Generate realistic volatility by instrument
            base_vol = 0.25 if "MISO" in instrument_id else 0.22 if "PJM" in instrument_id else 0.30
            position_value = abs(quantity) * 50  # Assume $50/MWh

            # Simple VaR calculation
            var = position_value * base_vol * (2.326 if confidence_level == 0.99 else 1.645)
            position_vars.append({
                "instrument_id": instrument_id,
                "quantity": quantity,
                "var_value": var,
                "var_percentage": var / position_value
            })
            total_var += var

        return {
            "total_var": total_var,
            "confidence_level": confidence_level,
            "method": "parametric",
            "positions": position_vars,
            "currency": "USD"
        }

    def _generate_mock_correlation_matrix(self, instrument_ids: List[str]) -> pd.DataFrame:
        """Generate mock correlation matrix for local development."""
        import numpy as np

        n = len(instrument_ids)
        # Generate realistic correlation matrix (high correlation within markets)
        base_corr = np.eye(n)  # Start with identity matrix

        # Add realistic correlations based on market relationships
        for i in range(n):
            for j in range(i + 1, n):
                inst1, inst2 = instrument_ids[i], instrument_ids[j]

                # High correlation within same market
                if inst1.split(".")[0] == inst2.split(".")[0]:
                    corr = 0.7 + np.random.normal(0, 0.1)
                else:
                    corr = 0.3 + np.random.normal(0, 0.15)

                corr = max(-1, min(1, corr))  # Clamp to [-1, 1]
                base_corr[i, j] = corr
                base_corr[j, i] = corr

        return pd.DataFrame(base_corr, index=instrument_ids, columns=instrument_ids)

    def _generate_mock_forecast(
        self,
        instrument_id: str,
        horizon_days: int
    ) -> pd.DataFrame:
        """Generate mock forecast data for local development."""
        import numpy as np

        # Base price for instrument
        if "MISO" in instrument_id:
            base_price = 35.0
        elif "PJM" in instrument_id:
            base_price = 40.0
        elif "CAISO" in instrument_id:
            base_price = 45.0
        else:
            base_price = 40.0

        dates = pd.date_range(start=datetime.now(), periods=horizon_days, freq='D')
        prices = []

        current_price = base_price
        for i in range(horizon_days):
            # Add trend and noise
            trend = 0.1 * i / horizon_days  # Slight upward trend
            noise = np.random.normal(0, base_price * 0.05)
            price = base_price + trend + noise
            prices.append(max(0, price))  # Ensure non-negative

        return pd.DataFrame({
            "timestamp": dates,
            "forecast_price": prices,
            "lower_bound": [p * 0.9 for p in prices],
            "upper_bound": [p * 1.1 for p in prices],
            "confidence": 0.8
        })

