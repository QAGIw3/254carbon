"""
Real-Time Forecast Service
High-frequency (5-minute) price forecasting with <500ms latency.
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel
import redis.asyncio as redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real-Time Forecast Service",
    description="Sub-hourly price forecasting with ultra-low latency",
    version="1.0.0",
)


class RealtimeForecastRequest(BaseModel):
    instrument_id: str
    intervals_ahead: int = 12  # 12 x 5min = 1 hour ahead
    include_confidence: bool = True


class ForecastInterval(BaseModel):
    interval_ending: datetime
    forecast_price: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


class RealtimeForecastResponse(BaseModel):
    instrument_id: str
    forecast_time: datetime
    intervals: List[ForecastInterval]
    latency_ms: float
    model_version: str


class StreamingForecaster:
    """
    Streaming forecaster with <500ms latency.
    
    Uses:
    - Lightweight models (pre-loaded in memory)
    - Redis for feature caching
    - Async processing
    - Connection pooling
    """
    
    def __init__(self):
        self.models = {}
        self.redis_client = None
        self._load_models()
    
    def _load_models(self):
        """Pre-load lightweight models into memory."""
        # In production, load actual trained models
        # For now, use mock models
        logger.info("Loading streaming forecast models...")
        
        # Mock: simple linear models for speed
        self.models["default"] = {
            "coefficients": np.array([1.0, 0.05, -0.02, 0.1]),
            "intercept": 40.0,
            "version": "stream_v1",
        }
    
    async def get_redis(self):
        """Get Redis client for feature caching."""
        if self.redis_client is None:
            self.redis_client = await redis.from_url(
                "redis://redis-cluster:6379",
                decode_responses=True,
            )
        return self.redis_client
    
    async def get_realtime_features(self, instrument_id: str) -> np.ndarray:
        """
        Get real-time features for forecasting.
        
        Features cached in Redis for <10ms access.
        """
        # Try cache first
        r = await self.get_redis()
        cache_key = f"rt_features:{instrument_id}"
        
        cached = await r.get(cache_key)
        if cached:
            return np.array(json.loads(cached))
        
        # Calculate features (simplified)
        # In production, would query recent prices, load, weather, etc.
        features = np.array([
            datetime.utcnow().hour,  # Hour of day
            datetime.utcnow().minute / 60,  # Fraction of hour
            np.sin(2 * np.pi * datetime.utcnow().hour / 24),  # Hour cycle
            45.0,  # Last known price (mock)
        ])
        
        # Cache for 1 minute
        await r.setex(cache_key, 60, json.dumps(features.tolist()))
        
        return features
    
    async def predict(
        self,
        instrument_id: str,
        intervals_ahead: int,
    ) -> List[Dict]:
        """
        Generate predictions for next N 5-minute intervals.
        
        Target latency: <500ms
        """
        start_time = datetime.utcnow()
        
        # Get model
        model = self.models.get(instrument_id, self.models["default"])
        
        # Get features (cached, ~10ms)
        features = await self.get_realtime_features(instrument_id)
        
        # Generate predictions
        predictions = []
        base_time = datetime.utcnow().replace(second=0, microsecond=0)
        base_time = base_time.replace(minute=(base_time.minute // 5) * 5)
        
        for i in range(intervals_ahead):
            interval_time = base_time + timedelta(minutes=5 * (i + 1))
            
            # Feature vector for this interval
            interval_features = features.copy()
            interval_features[0] = interval_time.hour
            interval_features[1] = interval_time.minute / 60
            interval_features[2] = np.sin(2 * np.pi * interval_time.hour / 24)
            
            # Predict (linear model for speed)
            forecast = (
                model["intercept"] +
                np.dot(model["coefficients"], interval_features)
            )
            
            # Simple confidence interval
            std_error = 3.0  # Mock
            ci_lower = forecast - 1.96 * std_error
            ci_upper = forecast + 1.96 * std_error
            
            predictions.append({
                "interval_ending": interval_time,
                "forecast_price": float(forecast),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
            })
        
        # Calculate latency
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        logger.info(f"Forecast generated in {latency_ms:.2f}ms")
        
        return predictions, latency_ms, model["version"]


# Global forecaster instance
forecaster = StreamingForecaster()


@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": len(forecaster.models)}


@app.post("/api/v1/forecast/realtime", response_model=RealtimeForecastResponse)
async def realtime_forecast(request: RealtimeForecastRequest):
    """
    Generate real-time forecast for next N 5-minute intervals.
    
    Ultra-low latency: <500ms target.
    """
    try:
        predictions, latency_ms, model_version = await forecaster.predict(
            request.instrument_id,
            request.intervals_ahead,
        )
        
        intervals = [
            ForecastInterval(
                interval_ending=p["interval_ending"],
                forecast_price=p["forecast_price"],
                ci_lower=p["ci_lower"] if request.include_confidence else None,
                ci_upper=p["ci_upper"] if request.include_confidence else None,
            )
            for p in predictions
        ]
        
        return RealtimeForecastResponse(
            instrument_id=request.instrument_id,
            forecast_time=datetime.utcnow(),
            intervals=intervals,
            latency_ms=latency_ms,
            model_version=model_version,
        )
        
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/forecast/{instrument_id}")
async def websocket_forecast(websocket: WebSocket, instrument_id: str):
    """
    WebSocket stream of continuous forecasts.
    
    Pushes updated forecast every 5 minutes.
    """
    await websocket.accept()
    
    try:
        while True:
            # Generate forecast
            predictions, latency_ms, model_version = await forecaster.predict(
                instrument_id,
                intervals_ahead=12,
            )
            
            # Send to client
            await websocket.send_json({
                "instrument_id": instrument_id,
                "forecast_time": datetime.utcnow().isoformat(),
                "intervals": predictions,
                "latency_ms": latency_ms,
            })
            
            # Wait 5 minutes
            await asyncio.sleep(300)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)

