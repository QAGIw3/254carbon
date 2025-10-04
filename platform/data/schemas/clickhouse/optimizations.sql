-- ClickHouse Performance Optimizations
-- Market Intelligence Platform

-- =========================================================================
-- MATERIALIZED VIEWS FOR COMMON QUERY PATTERNS
-- =========================================================================

-- Hourly price aggregations for dashboards
CREATE MATERIALIZED VIEW IF NOT EXISTS ch.hourly_price_aggregations
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(hour_start)
ORDER BY (market, instrument_id, hour_start)
POPULATE
AS SELECT
    toStartOfHour(event_time) as hour_start,
    market,
    product,
    instrument_id,
    price_type,
    avg(value) as avg_price,
    min(value) as min_price,
    max(value) as max_price,
    stddevPop(value) as price_stddev,
    count() as tick_count,
    sum(volume) as total_volume
FROM ch.market_price_ticks
GROUP BY hour_start, market, product, instrument_id, price_type;

-- Daily price summaries for reporting
CREATE MATERIALIZED VIEW IF NOT EXISTS ch.daily_price_summaries
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (market, instrument_id, date)
POPULATE
AS SELECT
    toDate(event_time) as date,
    market,
    product,
    instrument_id,
    location_code,
    price_type,
    avg(value) as avg_price,
    min(value) as min_price,
    max(value) as max_price,
    stddevPop(value) as price_stddev,
    quantile(0.25)(value) as p25_price,
    quantile(0.50)(value) as median_price,
    quantile(0.75)(value) as p75_price,
    count() as tick_count,
    sum(volume) as total_volume,
    argMin(event_time, value) as min_price_time,
    argMax(event_time, value) as max_price_time
FROM ch.market_price_ticks
GROUP BY date, market, product, instrument_id, location_code, price_type;

-- Forward curve aggregations for scenario comparison
CREATE MATERIALIZED VIEW IF NOT EXISTS ch.curve_scenario_summaries
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(as_of_date)
ORDER BY (curve_id, scenario_id, tenor_type)
POPULATE
AS SELECT
    curve_id,
    scenario_id,
    tenor_type,
    toDate(as_of_date) as as_of_date,
    market,
    product,
    avgState(price) as avg_price_state,
    minState(price) as min_price_state,
    maxState(price) as max_price_state,
    countState() as point_count_state
FROM ch.forward_curve_points
GROUP BY curve_id, scenario_id, tenor_type, as_of_date, market, product;

-- Real-time price monitoring (latest price per instrument)
CREATE MATERIALIZED VIEW IF NOT EXISTS ch.latest_prices
ENGINE = ReplacingMergeTree(event_time)
ORDER BY (instrument_id, price_type)
AS SELECT
    instrument_id,
    price_type,
    market,
    product,
    location_code,
    value,
    volume,
    currency,
    unit,
    source,
    event_time,
    arrival_time
FROM ch.market_price_ticks
ORDER BY event_time DESC
LIMIT 1 BY instrument_id, price_type;

-- LMP component aggregations (for nodal analysis)
CREATE MATERIALIZED VIEW IF NOT EXISTS ch.lmp_component_hourly
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(hour_start)
ORDER BY (instrument_id, hour_start)
POPULATE
AS SELECT
    toStartOfHour(event_time) as hour_start,
    instrument_id,
    location_code,
    market,
    -- Decompose LMP using naming conventions (Energy, Congestion, Loss)
    sumIf(value, position(lower(price_type), 'energy') > 0) as total_energy,
    sumIf(value, position(lower(price_type), 'congestion') > 0) as total_congestion,
    sumIf(value, position(lower(price_type), 'loss') > 0) as total_loss,
    avgIf(value, position(lower(price_type), 'lmp') > 0) as avg_lmp,
    count() as observations
FROM ch.market_price_ticks
WHERE market IN ('PJM', 'MISO', 'ERCOT', 'CAISO')  -- Nodal markets
GROUP BY hour_start, instrument_id, location_code, market;

-- =========================================================================
-- SKIP INDEXES FOR FREQUENTLY FILTERED COLUMNS
-- =========================================================================

-- Bloom filter index on location_code for nodal queries
ALTER TABLE ch.market_price_ticks 
ADD INDEX IF NOT EXISTS idx_location_bloom location_code TYPE bloom_filter GRANULARITY 4;

-- Min-max index on value for price range filters
ALTER TABLE ch.market_price_ticks 
ADD INDEX IF NOT EXISTS idx_value_minmax value TYPE minmax GRANULARITY 8;

-- Set index on market for quick market filtering
ALTER TABLE ch.market_price_ticks 
ADD INDEX IF NOT EXISTS idx_market_set market TYPE set(100) GRANULARITY 4;

-- Bloom filter on scenario_id for curve queries
ALTER TABLE ch.forward_curve_points 
ADD INDEX IF NOT EXISTS idx_scenario_bloom scenario_id TYPE bloom_filter GRANULARITY 4;

-- Min-max index on delivery_start for date range queries
ALTER TABLE ch.forward_curve_points 
ADD INDEX IF NOT EXISTS idx_delivery_minmax delivery_start TYPE minmax GRANULARITY 8;

-- =========================================================================
-- DATA RETENTION POLICIES (TTL)
-- =========================================================================

-- Keep raw ticks for 2 years, delete older data
ALTER TABLE ch.market_price_ticks
MODIFY TTL event_time + INTERVAL 2 YEAR;

-- Keep forward curves for 5 years
ALTER TABLE ch.forward_curve_points
MODIFY TTL as_of_date + INTERVAL 5 YEAR;

-- Keep fundamentals for 10 years
ALTER TABLE ch.fundamentals_series
MODIFY TTL ts + INTERVAL 10 YEAR;

-- Keep aggregated views for 5 years
ALTER TABLE ch.hourly_price_aggregations
MODIFY TTL hour_start + INTERVAL 5 YEAR;

ALTER TABLE ch.daily_price_summaries
MODIFY TTL date + INTERVAL 5 YEAR;

-- =========================================================================
-- QUERY OPTIMIZATION SETTINGS
-- =========================================================================

-- Optimize merge settings for high-throughput ingestion
ALTER TABLE ch.market_price_ticks
MODIFY SETTING max_parts_in_total = 10000,
               parts_to_delay_insert = 300,
               parts_to_throw_insert = 500;

-- Enable adaptive granularity for sparse partitions
ALTER TABLE ch.forward_curve_points
MODIFY SETTING index_granularity_bytes = 1024000;

-- =========================================================================
-- COMPRESSION OPTIMIZATION
-- =========================================================================

-- Optimize compression for timestamp columns (Delta encoding)
ALTER TABLE ch.market_price_ticks
MODIFY COLUMN event_time CODEC(DoubleDelta, LZ4);

-- Optimize compression for price columns (T64 + LZ4)
ALTER TABLE ch.market_price_ticks
MODIFY COLUMN value CODEC(T64, LZ4);

-- Optimize compression for enum-like columns (ZSTD level 1)
ALTER TABLE ch.market_price_ticks
MODIFY COLUMN instrument_id CODEC(ZSTD(1));

-- =========================================================================
-- DICTIONARIES FOR FAST LOOKUPS
-- =========================================================================

-- Instrument metadata dictionary for fast enrichment
CREATE DICTIONARY IF NOT EXISTS ch.instrument_dict
(
    instrument_id String,
    market String,
    product String,
    location String,
    timezone String,
    currency String
)
PRIMARY KEY instrument_id
SOURCE(POSTGRESQL(
    host 'postgresql'
    port 5432
    db 'market_intelligence'
    user 'market_intelligence'
    password ''
    table 'instrument'
))
LIFETIME(MIN 300 MAX 600)  -- Reload every 5-10 minutes
LAYOUT(HASHED());

-- Market metadata dictionary
CREATE DICTIONARY IF NOT EXISTS ch.market_dict
(
    market_code String,
    market_name String,
    region String,
    timezone String,
    currency String,
    settlement_period_minutes UInt16
)
PRIMARY KEY market_code
SOURCE(POSTGRESQL(
    host 'postgresql'
    port 5432
    db 'market_intelligence'
    user 'market_intelligence'
    password ''
    table 'market'
))
LIFETIME(MIN 600 MAX 1200)  -- Reload every 10-20 minutes
LAYOUT(HASHED());

-- =========================================================================
-- QUERY MONITORING TABLE
-- =========================================================================

-- Track slow queries for optimization
CREATE TABLE IF NOT EXISTS ch.query_log_slow
(
    event_time DateTime,
    query_duration_ms UInt64,
    query_id String,
    query_kind String,
    query TEXT,
    read_rows UInt64,
    read_bytes UInt64,
    result_rows UInt64,
    result_bytes UInt64,
    memory_usage UInt64,
    user String
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_time)
ORDER BY (event_time, query_duration_ms DESC)
TTL event_time + INTERVAL 90 DAY;

-- =========================================================================
-- OPTIMIZATION NOTES
-- =========================================================================

-- Performance Best Practices:
-- 1. Use PREWHERE for primary filtering on indexed columns
-- 2. Prefer materialized views for aggregations instead of repeated GROUP BY
-- 3. Use dictionaries for dimension enrichment instead of JOINs
-- 4. Partition by month for time-series data to enable partition pruning
-- 5. Order by columns that are commonly filtered for efficient data skip
-- 6. Use LowCardinality for columns with <10K distinct values
-- 7. Apply compression codecs appropriate to data patterns
-- 8. Set TTL to manage data growth and costs

-- Query Pattern Examples:
--
-- Fast hourly query using materialized view:
-- SELECT hour_start, avg_price 
-- FROM ch.hourly_price_aggregations
-- WHERE instrument_id = 'PJM.HUB.WEST' 
--   AND hour_start >= today() - INTERVAL 7 DAY;
--
-- Fast latest price using view:
-- SELECT value, event_time
-- FROM ch.latest_prices
-- WHERE instrument_id = 'MISO.HUB.INDIANA';
--
-- Fast dictionary lookup for enrichment:
-- SELECT t.*, dictGet('ch.instrument_dict', 'timezone', t.instrument_id) as tz
-- FROM ch.market_price_ticks t
-- WHERE ...;
-- ClickHouse optimizations and materialized views
-- Applies TTLs, projections, and sample aggregation views
