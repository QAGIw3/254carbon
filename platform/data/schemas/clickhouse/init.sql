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

-- Materialized views for common query patterns

-- Daily price summary for dashboard KPIs
CREATE MATERIALIZED VIEW IF NOT EXISTS ch.market_price_daily_summary
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (market, instrument_id, date)
AS SELECT
    toDate(event_time) as date,
    market,
    instrument_id,
    location_code,
    price_type,
    avg(value) as avg_price,
    min(value) as min_price,
    max(value) as max_price,
    count() as tick_count,
    sum(volume) as total_volume
FROM ch.market_price_ticks
GROUP BY date, market, instrument_id, location_code, price_type;

-- Hourly curve points summary for performance monitoring
CREATE MATERIALIZED VIEW IF NOT EXISTS ch.curve_hourly_summary
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(as_of_date)
ORDER BY (market, curve_id, as_of_date, hour)
AS SELECT
    toDate(as_of_date) as date,
    toHour(as_of_date) as hour,
    market,
    curve_id,
    scenario_id,
    count() as point_count,
    avg(price) as avg_price,
    min(price) as min_price,
    max(price) as max_price
FROM ch.forward_curve_points
GROUP BY date, hour, market, curve_id, scenario_id;

-- Market data freshness tracking
CREATE MATERIALIZED VIEW IF NOT EXISTS ch.market_freshness
ENGINE = ReplacingMergeTree()
PARTITION BY toYYYYMM(event_time)
ORDER BY (market, source, event_time)
AS SELECT
    market,
    source,
    max(event_time) as last_update,
    now() - max(event_time) as staleness_seconds,
    countIf(event_time > now() - INTERVAL 5 MINUTE) as recent_count
FROM ch.market_price_ticks
GROUP BY market, source;

-- Forecast accuracy metrics for business KPIs
CREATE MATERIALIZED VIEW IF NOT EXISTS ch.forecast_accuracy_daily
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (market, model_type, date)
AS SELECT
    toDate(event_time) as date,
    market,
    'forecast_accuracy' as model_type,
    count() as sample_count,
    avg(abs(predicted_price - actual_price) / actual_price) * 100 as mape,
    avg(abs(predicted_price - actual_price)) as mae,
    sqrt(avg(pow(predicted_price - actual_price, 2))) as rmse
FROM (
    SELECT
        event_time,
        market,
        instrument_id,
        -- Get actual price from market_price_ticks
        value as actual_price,
        -- Get forecasted price (this would need to be joined with forecast data)
        lagInFrame(value) OVER (PARTITION BY market, instrument_id ORDER BY event_time ROWS 1 PRECEDING) as predicted_price
    FROM ch.market_price_ticks
    WHERE event_time >= now() - INTERVAL 30 DAY
)
WHERE predicted_price IS NOT NULL
GROUP BY date, market;
-- ClickHouse initialization script for analytics tables
-- Defines engines, partitions, ordering keys for performance
