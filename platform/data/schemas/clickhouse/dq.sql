-- Data Quality and Imputation Schema (ClickHouse)
-- Databases
CREATE DATABASE IF NOT EXISTS market_intelligence;

-- Data quality issues (row-level flags)
CREATE TABLE IF NOT EXISTS market_intelligence.data_quality_issues
(
    event_time           DateTime64(3, 'UTC'),
    source               LowCardinality(String),
    instrument_id        LowCardinality(String),
    commodity_type       LowCardinality(String),
    dimension            LowCardinality(String),   -- completeness|consistency|validity|accuracy|timeliness
    severity             LowCardinality(String),   -- info|warning|error
    rule_id              String,
    value                String,
    expected             Nullable(String),
    metadata             JSON,
    run_id               UUID,
    created_at           DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY toYYYYMM(event_time)
ORDER BY (instrument_id, commodity_type, dimension, event_time, source);

-- Daily quality scores (aggregated)
CREATE TABLE IF NOT EXISTS market_intelligence.data_quality_scores
(
    date                 Date,
    source               LowCardinality(String),
    commodity_type       LowCardinality(String),
    score                Float64,
    components           JSON,
    created_at           DateTime DEFAULT now()
)
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (source, commodity_type, date);

-- Cross-source validation results
CREATE TABLE IF NOT EXISTS market_intelligence.cross_source_validation
(
    ts                   DateTime64(3, 'UTC'),
    primary_source       LowCardinality(String),
    secondary_source     LowCardinality(String),
    instrument_id        LowCardinality(String),
    metric_name          LowCardinality(String),
    rel_diff             Float64,
    within_tolerance     UInt8,
    reconciled_value     Nullable(Float64),
    run_id               UUID,
    created_at           DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY toYYYYMM(ts)
ORDER BY (instrument_id, metric_name, ts, primary_source, secondary_source);

-- Hourly imputed prices (kept separate from raw)
CREATE TABLE IF NOT EXISTS market_intelligence.market_price_hourly_imputed
(
    hour_start           DateTime('UTC'),
    market               LowCardinality(String),
    instrument_id        LowCardinality(String),
    price_type           LowCardinality(String),
    imputed_price        Float64,
    method               LowCardinality(String),   -- linear|carry_forward|seasonal_mean|model
    gap_length_hours     UInt32,
    run_id               UUID,
    created_at           DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY toYYYYMM(hour_start)
ORDER BY (instrument_id, price_type, hour_start);

-- Daily imputed aggregates
CREATE TABLE IF NOT EXISTS market_intelligence.market_price_daily_imputed
(
    date                 Date,
    market               LowCardinality(String),
    instrument_id        LowCardinality(String),
    price_type           LowCardinality(String),
    imputed_obs          UInt32,
    avg_imputed_price    Float64,
    max_gap_hours        UInt32,
    run_id               UUID,
    created_at           DateTime DEFAULT now()
)
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (instrument_id, price_type, date);

-- Indexes for performance
ALTER TABLE market_intelligence.data_quality_issues 
    ADD INDEX IF NOT EXISTS idx_dq_inst instrument_id TYPE bloom_filter GRANULARITY 4;
ALTER TABLE market_intelligence.data_quality_issues 
    ADD INDEX IF NOT EXISTS idx_dq_comm commodity_type TYPE set(64) GRANULARITY 4;
ALTER TABLE market_intelligence.cross_source_validation 
    ADD INDEX IF NOT EXISTS idx_rel_diff rel_diff TYPE minmax GRANULARITY 8;

-- Rollup MV: daily issue counts by source and severity
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.mv_dq_issues_daily
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (source, severity, date)
AS SELECT
    toDate(event_time) AS date,
    source,
    severity,
    count() AS issue_count
FROM market_intelligence.data_quality_issues
GROUP BY date, source, severity;


