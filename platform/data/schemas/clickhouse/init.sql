-- ClickHouse Schema Initialization
-- Market Intelligence Platform

-- Market price ticks table
CREATE DATABASE IF NOT EXISTS ch;

CREATE TABLE IF NOT EXISTS ch.market_price_ticks
(
    event_time        DateTime64(3, 'UTC'),
    arrival_time      DateTime64(3, 'UTC') DEFAULT now(),
    market            LowCardinality(String),
    product           LowCardinality(String),
    instrument_id     LowCardinality(String),
    location_code     LowCardinality(String),
    price_type        LowCardinality(String),
    value             Float64,
    volume            Nullable(Float64),
    currency          FixedString(3),
    unit              LowCardinality(String),
    source            LowCardinality(String),
    version_id        UInt32 DEFAULT 1
)
ENGINE = ReplacingMergeTree(version_id)
PARTITION BY toYYYYMM(event_time)
ORDER BY (instrument_id, price_type, event_time, source);

-- Forward curve points table
CREATE TABLE IF NOT EXISTS ch.forward_curve_points
(
    as_of_date        Date,
    market            LowCardinality(String),
    product           LowCardinality(String),
    instrument_id     LowCardinality(String),
    curve_id          LowCardinality(String),
    scenario_id       LowCardinality(String) DEFAULT 'BASE',
    delivery_start    Date,
    delivery_end      Date,
    tenor_type        LowCardinality(String),
    price             Float64,
    currency          FixedString(3),
    unit              LowCardinality(String),
    source            LowCardinality(String),
    run_id            UUID,
    version_id        UInt32 DEFAULT 1,
    created_at        DateTime64(3, 'UTC') DEFAULT now()
)
ENGINE = ReplacingMergeTree(version_id)
PARTITION BY toYYYYMM(as_of_date)
ORDER BY (curve_id, scenario_id, delivery_start, as_of_date);

-- Fundamentals time series table
CREATE TABLE IF NOT EXISTS ch.fundamentals_series
(
    ts                DateTime64(3, 'UTC'),
    market            LowCardinality(String),
    entity_id         LowCardinality(String),
    variable          LowCardinality(String),
    value             Float64,
    unit              LowCardinality(String),
    scenario_id       LowCardinality(String) DEFAULT 'BASE',
    source            LowCardinality(String),
    version_id        UInt32 DEFAULT 1
)
ENGINE = ReplacingMergeTree(version_id)
PARTITION BY toYYYYMM(ts)
ORDER BY (entity_id, variable, ts, scenario_id, source);

-- Indexes for performance
ALTER TABLE ch.market_price_ticks ADD INDEX idx_location location_code TYPE bloom_filter GRANULARITY 4;
ALTER TABLE ch.forward_curve_points ADD INDEX idx_scenario scenario_id TYPE bloom_filter GRANULARITY 4;
ALTER TABLE ch.fundamentals_series ADD INDEX idx_entity entity_id TYPE bloom_filter GRANULARITY 4;

