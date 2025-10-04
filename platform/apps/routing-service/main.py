"""
Multi-Source Redundancy Router - FastAPI Service

REST API service for source routing, health monitoring, and decision auditing.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel
import asyncpg
import json

from engine import (
    RoutingPolicyEngine,
    TrustScoringEngine,
    CircuitBreaker,
    SourceValue,
    RoutingDecision,
    RoutingStrategy
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Source Redundancy Router",
    description="Data source routing, health monitoring, and trust scoring",
    version="1.0.0"
)

# Global state
pool: Optional[asyncpg.Pool] = None
routing_engine: Optional[RoutingPolicyEngine] = None
circuit_breaker = CircuitBreaker(failure_threshold=5, cooldown_sec=300)


# Pydantic Models
class HealthMetricUpdate(BaseModel):
    """Source health metric update."""
    source_id: str
    metric_key: str
    freshness_lag_sec: int
    response_latency_ms: int
    error_rate_win: float
    completeness_pct: float
    deviation_from_blend: float = 0.0
    anomaly_flag: int = 0
    last_value: Optional[float] = None
    stddev_value: Optional[float] = None


class SourceValueInput(BaseModel):
    """Source value for routing request."""
    source_id: str
    value: float
    timestamp: Optional[datetime] = None


class RoutingRequest(BaseModel):
    """Request for routing decision."""
    metric_key: str
    candidate_values: List[SourceValueInput]
    mode: str = "routed"  # routed|advisory


class RoutingResponse(BaseModel):
    """Routing decision response."""
    value: float
    timestamp: datetime
    strategy: str
    confidence: float
    sources: List[Dict[str, Any]]
    decision_id: str
    is_synthetic: bool


class TrustScoreResponse(BaseModel):
    """Trust score response."""
    source_id: str
    metric_key: str
    trust_score: float
    components: Dict[str, float]
    timestamp: datetime


class SourceHealthResponse(BaseModel):
    """Source health status response."""
    source_id: str
    metric_key: str
    trust_score: float
    freshness_lag_sec: int
    error_rate: float
    uptime_pct: float
    circuit_state: str
    last_updated: datetime


# Database Connection
async def get_db_pool() -> asyncpg.Pool:
    """Get database connection pool."""
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(
            host="postgresql",
            port=5432,
            database="market_intelligence",
            user="market_user",
            password="market_pass",
            min_size=2,
            max_size=10
        )
    return pool


async def load_policy_config() -> Dict[str, Any]:
    """Load active routing policy configuration from database."""
    db = await get_db_pool()
    async with db.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT policy_version, min_trust, max_fresh_lag_sec, 
                   stable_dispersion, switch_margin, weights_json,
                   synthetic_decay_factor, synthetic_max_consecutive,
                   hysteresis_min_intervals, mad_k_factor
            FROM pg.routing_policy
            WHERE active = true
            ORDER BY created_at DESC
            LIMIT 1
        """)
        
        if not row:
            # Return default config
            return {
                'policy_version': 'v1',
                'min_trust': 0.55,
                'max_fresh_lag_sec': 180,
                'stable_dispersion': 0.012,
                'switch_margin': 0.07,
                'weights_json': {
                    'freshness': 0.30,
                    'error_rate': 0.20,
                    'deviation': 0.15,
                    'consistency': 0.15,
                    'uptime': 0.20
                },
                'synthetic_decay_factor': 0.85,
                'synthetic_max_consecutive': 12,
                'hysteresis_min_intervals': 3,
                'mad_k_factor': 3.0
            }
        
        return dict(row)


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    global routing_engine
    
    logger.info("Starting Multi-Source Redundancy Router service...")
    
    # Initialize database pool
    await get_db_pool()
    
    # Load routing policy
    policy_config = await load_policy_config()
    routing_engine = RoutingPolicyEngine(policy_config)
    
    logger.info(f"Loaded routing policy version: {policy_config['policy_version']}")
    logger.info("Service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global pool
    if pool:
        await pool.close()
    logger.info("Service shutdown complete")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "multi-source-redundancy-router",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/v1/routing/route", response_model=RoutingResponse)
async def route_metric(request: RoutingRequest):
    """
    Route a metric request to best available source(s).
    
    Main routing endpoint that accepts candidate source values
    and returns the optimal routed value with confidence scoring.
    """
    try:
        db = await get_db_pool()
        
        # Get previous decision for hysteresis
        previous_decision = None
        async with db.acquire() as conn:
            prev_row = await conn.fetchrow("""
                SELECT decision_id, value, strategy, confidence, is_synthetic
                FROM pg.routing_decisions
                WHERE metric_key = $1
                ORDER BY ts DESC
                LIMIT 1
            """, request.metric_key)
            
            if prev_row:
                previous_decision = RoutingDecision(
                    decision_id=str(prev_row['decision_id']),
                    metric_key=request.metric_key,
                    value=prev_row['value'],
                    strategy=RoutingStrategy(prev_row['strategy']),
                    confidence=prev_row['confidence'],
                    sources=[],
                    rationale_hash='',
                    policy_version='',
                    is_synthetic=prev_row['is_synthetic']
                )
        
        # Get latest trust scores for each source
        source_values = []
        for candidate in request.candidate_values:
            # Check circuit breaker
            if not circuit_breaker.is_available(candidate.source_id):
                logger.warning(f"Source {candidate.source_id} circuit breaker is open")
                continue
            
            # Get latest health metrics and trust score
            async with db.acquire() as conn:
                health_row = await conn.fetchrow("""
                    SELECT freshness_lag_sec, response_latency_ms
                    FROM pg.source_health
                    WHERE source_id = $1 AND metric_key = $2
                    ORDER BY ts DESC
                    LIMIT 1
                """, candidate.source_id, request.metric_key)
                
                trust_row = await conn.fetchrow("""
                    SELECT trust_score
                    FROM pg.trust_scores
                    WHERE source_id = $1 AND metric_key = $2
                    ORDER BY ts DESC
                    LIMIT 1
                """, candidate.source_id, request.metric_key)
            
            trust_score = trust_row['trust_score'] if trust_row else 0.8
            freshness_lag = health_row['freshness_lag_sec'] if health_row else 60
            latency = health_row['response_latency_ms'] if health_row else 100
            
            source_values.append(SourceValue(
                source_id=candidate.source_id,
                value=candidate.value,
                trust_score=trust_score,
                freshness_lag_sec=freshness_lag,
                response_latency_ms=latency,
                timestamp=candidate.timestamp or datetime.utcnow()
            ))
        
        if not source_values:
            raise HTTPException(
                status_code=503,
                detail="No available sources for routing"
            )
        
        # Make routing decision
        decision = routing_engine.route(
            metric_key=request.metric_key,
            candidate_values=source_values,
            previous_decision=previous_decision
        )
        
        # Store decision in database
        async with db.acquire() as conn:
            await conn.execute("""
                INSERT INTO pg.routing_decisions
                (decision_id, ts, metric_key, strategy, value, confidence,
                 sources_json, rationale_hash, policy_version, is_synthetic,
                 previous_decision_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
                decision.decision_id,
                datetime.utcnow(),
                decision.metric_key,
                decision.strategy.value,
                decision.value,
                decision.confidence,
                json.dumps(decision.sources),
                decision.rationale_hash,
                decision.policy_version,
                decision.is_synthetic,
                decision.previous_decision_id
            )
        
        # Log decision with audit logger
        logger.info(
            f"Routing decision: {decision.strategy} for {request.metric_key}, "
            f"value={decision.value:.2f}, confidence={decision.confidence:.2f}"
        )
        
        return RoutingResponse(
            value=decision.value,
            timestamp=datetime.utcnow(),
            strategy=decision.strategy.value,
            confidence=decision.confidence,
            sources=decision.sources,
            decision_id=decision.decision_id,
            is_synthetic=decision.is_synthetic
        )
        
    except Exception as e:
        logger.error(f"Error in routing decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/routing/health")
async def update_health_metric(update: HealthMetricUpdate):
    """
    Update health metrics for a source.
    
    Used by connectors to push health status updates.
    """
    try:
        db = await get_db_pool()
        
        async with db.acquire() as conn:
            # Insert health metric
            await conn.execute("""
                INSERT INTO pg.source_health
                (ts, source_id, metric_key, freshness_lag_sec, response_latency_ms,
                 error_rate_win, completeness_pct, deviation_from_blend, anomaly_flag,
                 last_value, stddev_value)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
                datetime.utcnow(),
                update.source_id,
                update.metric_key,
                update.freshness_lag_sec,
                update.response_latency_ms,
                update.error_rate_win,
                update.completeness_pct,
                update.deviation_from_blend,
                update.anomaly_flag,
                update.last_value,
                update.stddev_value
            )
            
            # Get source config
            source_row = await conn.fetchrow("""
                SELECT sla_freshness_sec, reliability_baseline
                FROM pg.source_registry
                WHERE source_id = $1
            """, update.source_id)
            
            if not source_row:
                raise HTTPException(
                    status_code=404,
                    detail=f"Source {update.source_id} not found"
                )
            
            # Compute trust score
            sla_freshness = source_row['sla_freshness_sec']
            
            # Get historical metrics for consistency and uptime
            stats = await conn.fetchrow("""
                SELECT 
                    COALESCE(STDDEV(last_value), 0) as value_stddev,
                    COALESCE(AVG(CASE WHEN error_rate_win < 0.05 THEN 1 ELSE 0 END), 0) as uptime_ratio
                FROM pg.source_health
                WHERE source_id = $1 AND metric_key = $2
                  AND ts > $3
            """, update.source_id, update.metric_key, datetime.utcnow() - timedelta(hours=24))
            
            consistency_ratio = 1.0 - min(stats['value_stddev'] / 100.0, 1.0) if stats['value_stddev'] else 0.95
            uptime_roll = stats['uptime_ratio']
            
            # Compute trust score
            trust_score, components = routing_engine.scoring_engine.compute_trust_score(
                freshness_lag_sec=update.freshness_lag_sec,
                sla_freshness_sec=sla_freshness,
                error_rate_win=update.error_rate_win,
                deviation_from_blend=update.deviation_from_blend,
                consistency_ratio=consistency_ratio,
                uptime_roll=uptime_roll
            )
            
            # Store trust score
            await conn.execute("""
                INSERT INTO pg.trust_scores
                (ts, source_id, metric_key, trust_score, freshness_component,
                 error_component, deviation_component, consistency_component,
                 uptime_component, policy_version)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
                datetime.utcnow(),
                update.source_id,
                update.metric_key,
                trust_score,
                components['freshness'],
                components['error_rate'],
                components['deviation'],
                components['consistency'],
                components['uptime'],
                routing_engine.policy_version
            )
        
        # Update circuit breaker
        if update.error_rate_win > 0.5:
            circuit_breaker.record_failure(update.source_id)
        else:
            circuit_breaker.record_success(update.source_id)
        
        return {
            "status": "updated",
            "source_id": update.source_id,
            "metric_key": update.metric_key,
            "trust_score": trust_score,
            "circuit_state": "open" if not circuit_breaker.is_available(update.source_id) else "closed"
        }
        
    except Exception as e:
        logger.error(f"Error updating health metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/routing/sources/health")
async def get_sources_health(
    metric_key: Optional[str] = Query(None, description="Filter by metric key")
):
    """Get health status of all sources."""
    try:
        db = await get_db_pool()
        
        async with db.acquire() as conn:
            if metric_key:
                rows = await conn.fetch("""
                    SELECT DISTINCT ON (sh.source_id)
                        sh.source_id,
                        sh.metric_key,
                        ts.trust_score,
                        sh.freshness_lag_sec,
                        sh.error_rate_win,
                        sh.ts as last_updated
                    FROM pg.source_health sh
                    LEFT JOIN pg.trust_scores ts ON 
                        sh.source_id = ts.source_id AND 
                        sh.metric_key = ts.metric_key AND
                        sh.ts = ts.ts
                    WHERE sh.metric_key = $1
                    ORDER BY sh.source_id, sh.ts DESC
                """, metric_key)
            else:
                rows = await conn.fetch("""
                    SELECT DISTINCT ON (sh.source_id, sh.metric_key)
                        sh.source_id,
                        sh.metric_key,
                        ts.trust_score,
                        sh.freshness_lag_sec,
                        sh.error_rate_win,
                        sh.ts as last_updated
                    FROM pg.source_health sh
                    LEFT JOIN pg.trust_scores ts ON 
                        sh.source_id = ts.source_id AND 
                        sh.metric_key = ts.metric_key AND
                        sh.ts = ts.ts
                    ORDER BY sh.source_id, sh.metric_key, sh.ts DESC
                """)
        
        sources = []
        for row in rows:
            circuit_state = "open" if not circuit_breaker.is_available(row['source_id']) else "closed"
            uptime_pct = (1.0 - row['error_rate_win']) * 100
            
            sources.append(SourceHealthResponse(
                source_id=row['source_id'],
                metric_key=row['metric_key'],
                trust_score=row['trust_score'] or 0.0,
                freshness_lag_sec=row['freshness_lag_sec'],
                error_rate=row['error_rate_win'],
                uptime_pct=uptime_pct,
                circuit_state=circuit_state,
                last_updated=row['last_updated']
            ))
        
        return {
            "sources": sources,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting sources health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/routing/decisions")
async def get_routing_decisions(
    metric_key: Optional[str] = Query(None, description="Filter by metric key"),
    strategy: Optional[str] = Query(None, description="Filter by strategy"),
    limit: int = Query(100, ge=1, le=1000, description="Number of decisions to return")
):
    """Get recent routing decisions."""
    try:
        db = await get_db_pool()
        
        query = """
            SELECT decision_id, ts, metric_key, strategy, value, confidence,
                   sources_json, is_synthetic
            FROM pg.routing_decisions
            WHERE 1=1
        """
        params = []
        param_idx = 1
        
        if metric_key:
            query += f" AND metric_key = ${param_idx}"
            params.append(metric_key)
            param_idx += 1
        
        if strategy:
            query += f" AND strategy = ${param_idx}"
            params.append(strategy)
            param_idx += 1
        
        query += f" ORDER BY ts DESC LIMIT ${param_idx}"
        params.append(limit)
        
        async with db.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        decisions = []
        for row in rows:
            decisions.append({
                "decision_id": str(row['decision_id']),
                "timestamp": row['ts'].isoformat(),
                "metric_key": row['metric_key'],
                "strategy": row['strategy'],
                "value": row['value'],
                "confidence": row['confidence'],
                "sources": json.loads(row['sources_json']),
                "is_synthetic": row['is_synthetic']
            })
        
        return {
            "decisions": decisions,
            "count": len(decisions)
        }
        
    except Exception as e:
        logger.error(f"Error getting routing decisions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/routing/policy/reload")
async def reload_policy():
    """Reload routing policy configuration from database."""
    global routing_engine
    
    try:
        policy_config = await load_policy_config()
        routing_engine = RoutingPolicyEngine(policy_config)
        
        logger.info(f"Reloaded routing policy version: {policy_config['policy_version']}")
        
        return {
            "status": "reloaded",
            "policy_version": policy_config['policy_version'],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error reloading policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
