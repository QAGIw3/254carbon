# Integration Guide: Multi-Source Redundancy Router

This guide shows how to integrate the Multi-Source Redundancy Router with existing gateway endpoints and data connectors.

## Gateway Integration

### Option 1: Transparent Routing (Recommended)

Add routing middleware to gateway endpoints that transparently routes requests to the best available sources:

```python
# platform/apps/gateway/routing_middleware.py
import httpx
from typing import List, Dict, Any
from fastapi import Request

ROUTING_SERVICE_URL = "http://routing-service:8007"

async def fetch_with_routing(
    metric_key: str,
    source_fetchers: Dict[str, callable],
    mode: str = "routed"
) -> Dict[str, Any]:
    """
    Fetch metric from multiple sources and use routing service to select best value.
    
    Args:
        metric_key: Metric identifier (e.g., "carbon_intensity.us_east")
        source_fetchers: Dict of {source_id: async_fetch_function}
        mode: "routed" for active routing, "advisory" for shadow mode
    
    Returns:
        Routed value with metadata
    """
    # Fetch from all sources in parallel
    candidate_values = []
    
    async with httpx.AsyncClient() as client:
        for source_id, fetcher in source_fetchers.items():
            try:
                value = await fetcher()
                candidate_values.append({
                    "source_id": source_id,
                    "value": value,
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.warning(f"Failed to fetch from {source_id}: {e}")
        
        if not candidate_values:
            raise HTTPException(
                status_code=503,
                detail="No sources available"
            )
        
        # Route request
        response = await client.post(
            f"{ROUTING_SERVICE_URL}/api/v1/routing/route",
            json={
                "metric_key": metric_key,
                "candidate_values": candidate_values,
                "mode": mode
            }
        )
        response.raise_for_status()
        
        return response.json()


# Example usage in gateway endpoint
@app.get("/api/v1/carbon-intensity/{region}")
async def get_carbon_intensity(
    region: str,
    mode: str = Query("routed", description="routed|advisory|raw")
):
    """Get carbon intensity with intelligent source routing."""
    
    metric_key = f"carbon_intensity.{region}"
    
    # Define source fetchers
    source_fetchers = {
        "watttime": lambda: fetch_watttime_carbon(region),
        "eia": lambda: fetch_eia_carbon(region),
        "caiso": lambda: fetch_caiso_carbon(region) if region in ["us_west"] else None
    }
    
    if mode in ["routed", "advisory"]:
        result = await fetch_with_routing(metric_key, source_fetchers, mode)
        
        if mode == "advisory":
            # Also fetch current single-source value for comparison
            current_value = await source_fetchers["watttime"]()
            result["advisory"] = {
                "routed_value": result["value"],
                "current_value": current_value,
                "difference": result["value"] - current_value,
                "confidence_improvement": result["confidence"]
            }
        
        return result
    else:
        # Raw mode: return first available source
        for source_id, fetcher in source_fetchers.items():
            try:
                value = await fetcher()
                return {
                    "value": value,
                    "source": source_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            except Exception:
                continue
        
        raise HTTPException(status_code=503, detail="All sources unavailable")
```

### Option 2: Explicit Routing Endpoint

Add dedicated routing endpoint for advanced users:

```python
# platform/apps/gateway/main.py

@app.post("/api/v1/routing/request")
async def request_routed_value(
    metric_key: str,
    sources: Optional[List[str]] = None,
    user=Depends(verify_token)
):
    """
    Request routed value from multiple sources.
    
    Allows users to explicitly request routing with control over which sources to use.
    """
    # Check entitlement
    await check_entitlement(user, "routing", "advanced", "api")
    
    # Fetch from specified sources or all available
    if sources is None:
        sources = await get_available_sources(metric_key)
    
    candidate_values = []
    for source_id in sources:
        try:
            value = await fetch_from_source(source_id, metric_key)
            candidate_values.append({
                "source_id": source_id,
                "value": value,
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as e:
            logger.warning(f"Failed to fetch from {source_id}: {e}")
    
    # Route via routing service
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ROUTING_SERVICE_URL}/api/v1/routing/route",
            json={
                "metric_key": metric_key,
                "candidate_values": candidate_values,
                "mode": "routed"
            }
        )
        result = response.json()
    
    # Audit routing decision
    await AuditLogger.log_access(
        user_id=user.get("sub"),
        tenant_id=user.get("tenant_id"),
        action="routing_decision",
        resource_type="metric",
        resource_id=metric_key,
        request=request,
        success=True,
        details={
            "strategy": result["strategy"],
            "confidence": result["confidence"],
            "sources": [s["source_id"] for s in result["sources"]]
        }
    )
    
    return result
```

## Connector Health Emission

Add health metric emission hooks to existing data connectors:

### CAISO Connector Example

```python
# platform/data/connectors/caiso_connector.py
import httpx

class CAISOConnector(Ingestor):
    
    async def emit_health_metric(
        self,
        metric_key: str,
        freshness_lag_sec: int,
        response_latency_ms: int,
        error_occurred: bool,
        last_value: Optional[float] = None
    ):
        """Emit health metrics to routing service."""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    "http://routing-service:8007/api/v1/routing/health",
                    json={
                        "source_id": self.source_id,
                        "metric_key": metric_key,
                        "freshness_lag_sec": freshness_lag_sec,
                        "response_latency_ms": response_latency_ms,
                        "error_rate_win": 1.0 if error_occurred else 0.0,
                        "completeness_pct": 100.0 if last_value is not None else 0.0,
                        "last_value": last_value
                    },
                    timeout=5.0
                )
        except Exception as e:
            logger.warning(f"Failed to emit health metric: {e}")
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull LMP data with health tracking."""
        last_checkpoint = self.load_checkpoint()
        last_time = self._resolve_last_time(last_checkpoint)
        
        logger.info(f"Fetching CAISO {self.market_type} LMP since {last_time}")
        
        for node in self.get_nodes():
            metric_key = f"lmp.caiso.{node}"
            start_time = time.time()
            error_occurred = False
            last_value = None
            
            try:
                # Fetch data
                df = self._fetch_lmp_data(node, last_time)
                
                if not df.empty:
                    last_value = df['lmp'].iloc[-1]
                    freshness_lag = int(time.time() - df['timestamp'].iloc[-1].timestamp())
                    
                    # Emit data points
                    for _, row in df.iterrows():
                        yield {
                            'source_id': self.source_id,
                            'metric_key': metric_key,
                            'timestamp': row['timestamp'],
                            'value': row['lmp'],
                            'metadata': {
                                'node': node,
                                'market': self.market_type
                            }
                        }
                else:
                    freshness_lag = 600  # Assume stale if no data
                
            except Exception as e:
                logger.error(f"Error fetching {node} data: {e}")
                error_occurred = True
                freshness_lag = 999
            
            # Emit health metric
            response_latency_ms = int((time.time() - start_time) * 1000)
            asyncio.run(self.emit_health_metric(
                metric_key=metric_key,
                freshness_lag_sec=freshness_lag,
                response_latency_ms=response_latency_ms,
                error_occurred=error_occurred,
                last_value=last_value
            ))
```

### Generic Connector Health Hook

For any connector, add this helper:

```python
# platform/data/connectors/base_connector.py

class HealthAwareConnector:
    """Base class with health tracking capabilities."""
    
    def __init__(self, source_id: str, routing_service_url: str = "http://routing-service:8007"):
        self.source_id = source_id
        self.routing_service_url = routing_service_url
        self.health_buffer = []
        self.health_flush_interval = 60  # seconds
        self.last_health_flush = time.time()
    
    async def track_fetch(
        self,
        metric_key: str,
        fetch_func: callable
    ) -> Any:
        """
        Wrap a fetch operation with health tracking.
        
        Usage:
            result = await connector.track_fetch(
                "carbon_intensity.us_east",
                lambda: fetch_watttime_carbon("us_east")
            )
        """
        start_time = time.time()
        error_occurred = False
        last_value = None
        
        try:
            result = await fetch_func()
            if isinstance(result, (int, float)):
                last_value = result
            return result
        except Exception as e:
            error_occurred = True
            raise
        finally:
            response_latency_ms = int((time.time() - start_time) * 1000)
            
            # Buffer health metric
            self.health_buffer.append({
                "metric_key": metric_key,
                "freshness_lag_sec": 30,  # Estimate or compute from timestamp
                "response_latency_ms": response_latency_ms,
                "error_occurred": error_occurred,
                "last_value": last_value
            })
            
            # Flush buffer periodically
            if time.time() - self.last_health_flush > self.health_flush_interval:
                await self._flush_health_buffer()
    
    async def _flush_health_buffer(self):
        """Flush accumulated health metrics to routing service."""
        if not self.health_buffer:
            return
        
        try:
            async with httpx.AsyncClient() as client:
                for metric in self.health_buffer:
                    await client.post(
                        f"{self.routing_service_url}/api/v1/routing/health",
                        json={
                            "source_id": self.source_id,
                            **metric,
                            "error_rate_win": 1.0 if metric["error_occurred"] else 0.0,
                            "completeness_pct": 100.0 if metric["last_value"] is not None else 0.0
                        },
                        timeout=5.0
                    )
            
            self.health_buffer.clear()
            self.last_health_flush = time.time()
            
        except Exception as e:
            logger.warning(f"Failed to flush health buffer: {e}")
```

## Cache Integration

Integrate routing decisions with the existing cache layer:

```python
# platform/apps/gateway/cache.py (extend existing CacheManager)

class CacheManager:
    
    async def get_routed_value(
        self,
        metric_key: str,
        source_fetchers: Dict[str, callable],
        ttl: Optional[int] = None
    ) -> Any:
        """
        Get value with routing, caching the routed decision.
        
        Caches both the value and the routing decision metadata.
        """
        cache_key = f"routed:{metric_key}"
        
        # Try cache first
        cached = await self.get(cache_key)
        if cached is not None:
            return cached
        
        # Fetch with routing
        result = await fetch_with_routing(metric_key, source_fetchers)
        
        # Cache with adaptive TTL based on confidence
        if ttl is None:
            # Higher confidence = longer TTL
            if result["confidence"] > 0.9:
                ttl = 300  # 5 minutes
            elif result["confidence"] > 0.7:
                ttl = 180  # 3 minutes
            else:
                ttl = 60   # 1 minute
        
        await self.set(cache_key, result, ttl=ttl)
        
        return result
```

## Testing the Integration

### Shadow Mode Testing

```python
# Test routing in shadow mode without affecting production traffic
async def test_shadow_routing():
    """Compare routing decisions with current single-source approach."""
    
    metrics = ["carbon_intensity.us_east", "lmp.caiso.sp15", "lmp.miso.indiana"]
    
    for metric_key in metrics:
        # Get current single-source value
        current_value = await fetch_current_source(metric_key)
        
        # Get routed value in advisory mode
        routed_result = await fetch_with_routing(metric_key, mode="advisory")
        
        # Compare
        difference = abs(routed_result["value"] - current_value)
        confidence = routed_result["confidence"]
        
        print(f"{metric_key}:")
        print(f"  Current: {current_value}")
        print(f"  Routed:  {routed_result['value']}")
        print(f"  Diff:    {difference}")
        print(f"  Confidence: {confidence}")
        print(f"  Strategy: {routed_result['strategy']}")
```

## Monitoring Integration

Add routing metrics to existing monitoring:

```python
# platform/apps/gateway/metrics.py

from routing_service.monitoring.metrics import (
    record_routing_decision,
    record_health_update
)

# Track routing decisions in gateway
@app.middleware("http")
async def track_routing_metrics(request: Request, call_next):
    if "/api/v1/carbon-intensity" in request.url.path:
        start_time = time.time()
        response = await call_next(request)
        latency = time.time() - start_time
        
        # Extract routing metadata from response if available
        if hasattr(response, "routing_metadata"):
            record_routing_decision(
                strategy=response.routing_metadata["strategy"],
                metric_key=response.routing_metadata["metric_key"],
                confidence=response.routing_metadata["confidence"],
                latency_seconds=latency,
                is_synthetic=response.routing_metadata["is_synthetic"],
                num_sources=len(response.routing_metadata["sources"])
            )
        
        return response
    
    return await call_next(request)
```

## Rollout Checklist

- [ ] Phase 1: Deploy routing service to staging
- [ ] Phase 2: Add health emission hooks to connectors (shadow mode)
- [ ] Phase 3: Test advisory mode on gateway endpoints
- [ ] Phase 4: Enable active routing for non-critical metrics
- [ ] Phase 5: Monitor for 1 week, validate improvements
- [ ] Phase 6: Enable routing for critical metrics
- [ ] Phase 7: Enable synthetic fallback
- [ ] Phase 8: Add advanced blending strategies

## Troubleshooting

### High Synthetic Fallback Rate
- Check source health metrics
- Review trust scores
- Verify sources are emitting health updates
- Check circuit breaker states

### Low Routing Confidence
- Review source dispersion
- Check for outliers in source values
- Verify source trust scores are reasonable
- Consider adjusting policy thresholds

### High Latency
- Enable caching for routing decisions
- Optimize source fetch parallelization
- Consider reducing number of sources
- Check database connection pool

## Support

For integration questions: platform-team@254carbon.ai
Slack: #routing-service
