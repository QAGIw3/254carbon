-- Multi-Source Redundancy Router Schema
-- Schema for source health monitoring, trust scoring, and routing decisions

-- Source Registry Extensions (extends existing pg.source_registry)
-- Add columns to track redundancy routing metadata
ALTER TABLE pg.source_registry ADD COLUMN IF NOT EXISTS metric_types TEXT[] DEFAULT '{}';
ALTER TABLE pg.source_registry ADD COLUMN IF NOT EXISTS cadence_sec INT DEFAULT 300;
ALTER TABLE pg.source_registry ADD COLUMN IF NOT EXISTS default_weight FLOAT DEFAULT 1.0;
ALTER TABLE pg.source_registry ADD COLUMN IF NOT EXISTS license_class TEXT DEFAULT 'standard';
ALTER TABLE pg.source_registry ADD COLUMN IF NOT EXISTS reliability_baseline FLOAT DEFAULT 0.95;
ALTER TABLE pg.source_registry ADD COLUMN IF NOT EXISTS sla_freshness_sec INT DEFAULT 180;
ALTER TABLE pg.source_registry ADD COLUMN IF NOT EXISTS enabled BOOLEAN DEFAULT true;
ALTER TABLE pg.source_registry ADD COLUMN IF NOT EXISTS fallback_role TEXT DEFAULT 'primary'; -- primary|secondary|synthetic

-- Source Health Monitoring
-- Time-series health metrics for each source
CREATE TABLE IF NOT EXISTS pg.source_health (
    ts TIMESTAMPTZ NOT NULL,
    source_id TEXT NOT NULL REFERENCES pg.source_registry(source_id),
    metric_key TEXT NOT NULL,
    freshness_lag_sec INT,
    response_latency_ms INT,
    error_rate_win FLOAT,
    completeness_pct FLOAT,
    deviation_from_blend FLOAT,
    anomaly_flag INT DEFAULT 0,
    last_value FLOAT,
    stddev_value FLOAT,
    successful_intervals INT DEFAULT 0,
    total_intervals INT DEFAULT 0,
    PRIMARY KEY (ts, source_id, metric_key)
);

CREATE INDEX IF NOT EXISTS idx_source_health_ts ON pg.source_health(ts DESC);
CREATE INDEX IF NOT EXISTS idx_source_health_source ON pg.source_health(source_id, metric_key, ts DESC);

-- Trust Score History
-- Computed trust scores over time
CREATE TABLE IF NOT EXISTS pg.trust_scores (
    ts TIMESTAMPTZ NOT NULL,
    source_id TEXT NOT NULL REFERENCES pg.source_registry(source_id),
    metric_key TEXT NOT NULL,
    trust_score FLOAT NOT NULL,
    freshness_component FLOAT,
    error_component FLOAT,
    deviation_component FLOAT,
    consistency_component FLOAT,
    uptime_component FLOAT,
    policy_version TEXT NOT NULL,
    PRIMARY KEY (ts, source_id, metric_key)
);

CREATE INDEX IF NOT EXISTS idx_trust_scores_ts ON pg.trust_scores(ts DESC);
CREATE INDEX IF NOT EXISTS idx_trust_scores_source ON pg.trust_scores(source_id, metric_key, ts DESC);

-- Routing Decisions
-- Audit trail of all routing decisions
CREATE TABLE IF NOT EXISTS pg.routing_decisions (
    decision_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    metric_key TEXT NOT NULL,
    strategy TEXT NOT NULL, -- single|blend|fallback|synthetic
    value FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    sources_json JSONB NOT NULL,
    rationale_hash TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    previous_decision_id UUID REFERENCES pg.routing_decisions(decision_id),
    is_synthetic BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_routing_decisions_ts ON pg.routing_decisions(ts DESC);
CREATE INDEX IF NOT EXISTS idx_routing_decisions_metric ON pg.routing_decisions(metric_key, ts DESC);
CREATE INDEX IF NOT EXISTS idx_routing_decisions_strategy ON pg.routing_decisions(strategy, ts DESC);

-- Operator Overrides
-- Manual override table with TTL for temporary routing adjustments
CREATE TABLE IF NOT EXISTS pg.routing_overrides (
    override_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_key TEXT NOT NULL,
    forced_source_id TEXT REFERENCES pg.source_registry(source_id),
    forced_strategy TEXT, -- single|blend|synthetic
    forced_weight_adjustments JSONB,
    reason TEXT NOT NULL,
    created_by TEXT NOT NULL,
    valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_until TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_routing_overrides_active ON pg.routing_overrides(metric_key, valid_from, valid_until)
    WHERE valid_until > now();

-- Circuit Breaker State
-- Track sources that have been temporarily removed from rotation
CREATE TABLE IF NOT EXISTS pg.circuit_breaker_state (
    source_id TEXT PRIMARY KEY REFERENCES pg.source_registry(source_id),
    metric_key TEXT NOT NULL,
    state TEXT NOT NULL, -- open|half_open|closed
    failure_count INT DEFAULT 0,
    last_failure_ts TIMESTAMPTZ,
    cooldown_until TIMESTAMPTZ,
    reason TEXT,
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_circuit_breaker_state ON pg.circuit_breaker_state(state, cooldown_until);

-- Routing Policy Configuration
-- Versioned routing policy settings
CREATE TABLE IF NOT EXISTS pg.routing_policy (
    policy_version TEXT PRIMARY KEY,
    min_trust FLOAT NOT NULL DEFAULT 0.55,
    max_fresh_lag_sec INT NOT NULL DEFAULT 180,
    stable_dispersion FLOAT NOT NULL DEFAULT 0.012,
    switch_margin FLOAT NOT NULL DEFAULT 0.07,
    weights_json JSONB NOT NULL,
    synthetic_decay_factor FLOAT NOT NULL DEFAULT 0.85,
    synthetic_max_consecutive INT NOT NULL DEFAULT 12,
    hysteresis_min_intervals INT NOT NULL DEFAULT 3,
    mad_k_factor FLOAT NOT NULL DEFAULT 3.0,
    active BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Insert default policy
INSERT INTO pg.routing_policy (
    policy_version,
    min_trust,
    max_fresh_lag_sec,
    stable_dispersion,
    switch_margin,
    weights_json,
    synthetic_decay_factor,
    synthetic_max_consecutive,
    hysteresis_min_intervals,
    mad_k_factor,
    active
) VALUES (
    'v1',
    0.55,
    180,
    0.012,
    0.07,
    '{"freshness": 0.30, "error_rate": 0.20, "deviation": 0.15, "consistency": 0.15, "uptime": 0.20}'::jsonb,
    0.85,
    12,
    3,
    3.0,
    true
) ON CONFLICT (policy_version) DO NOTHING;

-- Source Heartbeat Events
-- Real-time heartbeat tracking for push-mode monitoring
CREATE TABLE IF NOT EXISTS pg.source_heartbeats (
    source_id TEXT NOT NULL REFERENCES pg.source_registry(source_id),
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    status TEXT NOT NULL, -- healthy|degraded|failed
    latency_ms INT,
    error_message TEXT,
    metadata JSONB,
    PRIMARY KEY (source_id, ts)
);

CREATE INDEX IF NOT EXISTS idx_source_heartbeats_ts ON pg.source_heartbeats(ts DESC);

-- Synthetic Fallback Models
-- Configuration for synthetic fallback strategies
CREATE TABLE IF NOT EXISTS pg.synthetic_models (
    model_id TEXT PRIMARY KEY,
    metric_key TEXT NOT NULL,
    model_type TEXT NOT NULL, -- arima|ets|ml_regression|last_value_carry
    model_config JSONB NOT NULL,
    training_start_date DATE,
    training_end_date DATE,
    model_accuracy FLOAT,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_synthetic_models_metric ON pg.synthetic_models(metric_key, active);

-- Grant permissions (assuming standard role)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA pg TO market_intelligence_app;
