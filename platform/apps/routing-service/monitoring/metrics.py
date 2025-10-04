"""
Prometheus metrics collection for Multi-Source Redundancy Router.
"""
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from typing import Dict, Any

# Create registry
registry = CollectorRegistry()

# Counters
routing_decisions_total = Counter(
    'routing_decisions_total',
    'Total number of routing decisions',
    ['strategy', 'metric_key'],
    registry=registry
)

source_health_updates_total = Counter(
    'source_health_updates_total',
    'Total number of source health updates',
    ['source_id', 'metric_key'],
    registry=registry
)

circuit_breaker_state_changes = Counter(
    'circuit_breaker_state_changes_total',
    'Total circuit breaker state changes',
    ['source_id', 'new_state'],
    registry=registry
)

# Gauges
routing_confidence_avg = Gauge(
    'routing_confidence_avg',
    'Average confidence score for routing decisions',
    ['metric_key'],
    registry=registry
)

synthetic_fallback_ratio = Gauge(
    'synthetic_fallback_ratio',
    'Ratio of synthetic fallback decisions',
    ['metric_key'],
    registry=registry
)

source_trust_score = Gauge(
    'source_trust_score',
    'Current trust score for source',
    ['source_id', 'metric_key'],
    registry=registry
)

circuit_breaker_state = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=half_open, 2=open)',
    ['source_id'],
    registry=registry
)

active_sources_count = Gauge(
    'active_sources_count',
    'Number of active sources available',
    ['metric_key'],
    registry=registry
)

# Histograms
routing_decision_latency = Histogram(
    'routing_decision_latency_seconds',
    'Latency of routing decisions',
    ['strategy'],
    registry=registry
)

source_freshness_lag = Histogram(
    'source_freshness_lag_seconds',
    'Freshness lag for source data',
    ['source_id', 'metric_key'],
    registry=registry
)

source_response_latency = Histogram(
    'source_response_latency_seconds',
    'Response latency for source queries',
    ['source_id', 'metric_key'],
    registry=registry
)


def record_routing_decision(
    strategy: str,
    metric_key: str,
    confidence: float,
    latency_seconds: float,
    is_synthetic: bool,
    num_sources: int
):
    """Record metrics for a routing decision."""
    routing_decisions_total.labels(
        strategy=strategy,
        metric_key=metric_key
    ).inc()
    
    routing_confidence_avg.labels(
        metric_key=metric_key
    ).set(confidence)
    
    routing_decision_latency.labels(
        strategy=strategy
    ).observe(latency_seconds)
    
    active_sources_count.labels(
        metric_key=metric_key
    ).set(num_sources)
    
    # Track synthetic fallback ratio
    if is_synthetic:
        # This is a simplified version; in production, use a sliding window
        synthetic_fallback_ratio.labels(
            metric_key=metric_key
        ).inc()


def record_health_update(
    source_id: str,
    metric_key: str,
    trust_score: float,
    freshness_lag_sec: int,
    response_latency_ms: int
):
    """Record metrics for a source health update."""
    source_health_updates_total.labels(
        source_id=source_id,
        metric_key=metric_key
    ).inc()
    
    source_trust_score.labels(
        source_id=source_id,
        metric_key=metric_key
    ).set(trust_score)
    
    source_freshness_lag.labels(
        source_id=source_id,
        metric_key=metric_key
    ).observe(freshness_lag_sec)
    
    source_response_latency.labels(
        source_id=source_id,
        metric_key=metric_key
    ).observe(response_latency_ms / 1000.0)


def record_circuit_breaker_state_change(source_id: str, new_state: str):
    """Record circuit breaker state change."""
    circuit_breaker_state_changes.labels(
        source_id=source_id,
        new_state=new_state
    ).inc()
    
    # Map state to numeric value
    state_map = {'closed': 0, 'half_open': 1, 'open': 2}
    circuit_breaker_state.labels(
        source_id=source_id
    ).set(state_map.get(new_state, 0))
