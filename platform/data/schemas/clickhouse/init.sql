-- ClickHouse Schema Initialization
-- Market Intelligence Platform

-- Market price ticks table
CREATE DATABASE IF NOT EXISTS market_intelligence;
CREATE DATABASE IF NOT EXISTS ch;

-- Legacy partitioning preserved for migration support (kept for compatibility)
CREATE TABLE IF NOT EXISTS market_intelligence.market_price_ticks_legacy
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
    commodity_type    LowCardinality(String) DEFAULT 'power',
    version_id        UInt32 DEFAULT 1
)
ENGINE = ReplacingMergeTree(version_id)
PARTITION BY toYYYYMM(event_time)
ORDER BY (instrument_id, price_type, event_time, source);

CREATE TABLE IF NOT EXISTS market_intelligence.market_price_ticks
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
    commodity_type    LowCardinality(String) DEFAULT 'power',
    version_id        UInt32 DEFAULT 1
)
ENGINE = ReplacingMergeTree(version_id)
PARTITION BY (toYYYYMM(event_time), commodity_type)
ORDER BY (commodity_type, instrument_id, price_type, event_time, source);

-- Forward curve points table
CREATE TABLE IF NOT EXISTS market_intelligence.forward_curve_points
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
CREATE TABLE IF NOT EXISTS market_intelligence.fundamentals_series
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
ALTER TABLE market_intelligence.market_price_ticks ADD INDEX IF NOT EXISTS idx_location location_code TYPE bloom_filter GRANULARITY 4;
ALTER TABLE market_intelligence.forward_curve_points ADD INDEX IF NOT EXISTS idx_scenario scenario_id TYPE bloom_filter GRANULARITY 4;
ALTER TABLE market_intelligence.fundamentals_series ADD INDEX IF NOT EXISTS idx_entity entity_id TYPE bloom_filter GRANULARITY 4;
ALTER TABLE market_intelligence.market_price_ticks ADD INDEX IF NOT EXISTS idx_commodity commodity_type TYPE set(32) GRANULARITY 4;
ALTER TABLE market_intelligence.market_price_ticks ADD INDEX IF NOT EXISTS idx_price_type price_type TYPE bloom_filter GRANULARITY 4;

-- Materialized views for common query patterns

-- Daily price summary for dashboard KPIs
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.market_price_daily_summary
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
FROM market_intelligence.market_price_ticks
GROUP BY date, market, instrument_id, location_code, price_type;

-- Hourly curve points summary for performance monitoring
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.curve_hourly_summary
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
FROM market_intelligence.forward_curve_points
GROUP BY date, hour, market, curve_id, scenario_id;

-- Market data freshness tracking
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.market_freshness
ENGINE = ReplacingMergeTree()
PARTITION BY toYYYYMM(event_time)
ORDER BY (market, source, event_time)
AS SELECT
    market,
    source,
    max(event_time) as last_update,
    now() - max(event_time) as staleness_seconds,
    countIf(event_time > now() - INTERVAL 5 MINUTE) as recent_count
FROM market_intelligence.market_price_ticks
GROUP BY market, source;

-- Forecast accuracy metrics for business KPIs
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.forecast_accuracy_daily
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
    FROM market_intelligence.market_price_ticks
    WHERE event_time >= now() - INTERVAL 30 DAY
)
WHERE predicted_price IS NOT NULL
GROUP BY date, market;
-- Commodity specifications table for contract details
CREATE TABLE IF NOT EXISTS market_intelligence.commodity_specifications
(
    commodity_code        LowCardinality(String),
    commodity_type        LowCardinality(String),
    instrument_root       LowCardinality(String),
    exchange              LowCardinality(String),
    contract_unit         LowCardinality(String),
    contract_size         Float64 DEFAULT 1,
    tick_size             Float64 DEFAULT 0.01,
    currency              FixedString(3) DEFAULT 'USD',
    delivery_location     LowCardinality(String),
    settlement_terms      Nullable(String),
    listing_rules         JSON,
    version_id            UInt32 DEFAULT 1,
    created_at            DateTime64(3, 'UTC') DEFAULT now()
)
ENGINE = ReplacingMergeTree(version_id)
ORDER BY (commodity_code, exchange, instrument_root);

CREATE TABLE IF NOT EXISTS market_intelligence.quality_specifications
(
    commodity_code        LowCardinality(String),
    grade_id              LowCardinality(String),
    quality               JSON,
    delivery_location     LowCardinality(String),
    reference_spec        LowCardinality(String),
    source                LowCardinality(String),
    version_id            UInt32 DEFAULT 1,
    created_at            DateTime64(3, 'UTC') DEFAULT now()
)
ENGINE = ReplacingMergeTree(version_id)
ORDER BY (commodity_code, grade_id);

CREATE TABLE IF NOT EXISTS market_intelligence.futures_curves
(
    as_of_date            Date,
    commodity_code        LowCardinality(String),
CREATE TABLE IF NOT EXISTS market_intelligence.contract_rollovers
(
    commodity_code        LowCardinality(String),
    instrument_root       LowCardinality(String),
    trade_date            Date,
    active_contract       LowCardinality(String),
    next_contract         LowCardinality(String),
    rollover_reason       LowCardinality(String),
    rollover_ratio        Float64,
    created_at            DateTime64(3, 'UTC') DEFAULT now(),
    metadata              JSON
)
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY toYYYYMM(trade_date)
ORDER BY (commodity_code, instrument_root, trade_date);

    contract_month        Date,
    tenor_code            LowCardinality(String),
    settlement_price      Float64,
    open_interest         Int64,
    volume                Int64,
    currency              FixedString(3),
    unit                  LowCardinality(String),
    exchange              LowCardinality(String),
    source                LowCardinality(String),
    version_id            UInt32 DEFAULT 1,
    created_at            DateTime64(3, 'UTC') DEFAULT now()
)
ENGINE = ReplacingMergeTree(version_id)
PARTITION BY (toYYYYMM(as_of_date), commodity_code)
ORDER BY (commodity_code, contract_month, as_of_date, exchange, source);

ALTER TABLE market_intelligence.futures_curves ADD INDEX IF NOT EXISTS idx_futures_commodity commodity_code TYPE set(64) GRANULARITY 4;
ALTER TABLE market_intelligence.futures_curves ADD INDEX IF NOT EXISTS idx_futures_exchange exchange TYPE bloom_filter GRANULARITY 4;

-- Commodity research analytics tables
CREATE TABLE IF NOT EXISTS market_intelligence.commodity_decomposition
(
    snapshot_date Date,
    instrument_id LowCardinality(String),
    method LowCardinality(String),
    trend Float64,
    seasonal Float64,
    residual Float64,
    version LowCardinality(String) DEFAULT 'v1',
    created_at DateTime64(3, 'UTC') DEFAULT now()
)
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY toYYYYMM(snapshot_date)
ORDER BY (instrument_id, method, snapshot_date, version);

CREATE TABLE IF NOT EXISTS market_intelligence.volatility_regimes
(
    date Date,
    instrument_id LowCardinality(String),
    regime_label LowCardinality(String),
    regime_features String,
    method LowCardinality(String),
    n_regimes UInt8,
    fit_version LowCardinality(String) DEFAULT 'v1',
    created_at DateTime64(3, 'UTC') DEFAULT now()
)
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY toYYYYMM(date)
ORDER BY (instrument_id, method, date, regime_label, fit_version);

CREATE TABLE IF NOT EXISTS market_intelligence.supply_demand_metrics
(
    date Date,
    entity_id LowCardinality(String),
    instrument_id Nullable(LowCardinality(String)),
    metric_name LowCardinality(String),
    metric_value Float64,
    unit LowCardinality(String),
    version LowCardinality(String) DEFAULT 'v1',
    created_at DateTime64(3, 'UTC') DEFAULT now()
)
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY toYYYYMM(date)
ORDER BY (entity_id, metric_name, date, version);

CREATE TABLE IF NOT EXISTS market_intelligence.weather_impact
(
    date Date,
    entity_id LowCardinality(String),
    coef_type LowCardinality(String),
    coefficient Float64,
    r2 Nullable(Float64),
    p_value Nullable(Float64),
    window LowCardinality(String),
    model_version LowCardinality(String) DEFAULT 'v1',
    extreme_event_count UInt32,
    diagnostics Nullable(String),
    method LowCardinality(String) DEFAULT 'ols',
    created_at DateTime64(3, 'UTC') DEFAULT now()
)
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY toYYYYMM(date)
ORDER BY (entity_id, coef_type, date, model_version);

CREATE TABLE IF NOT EXISTS market_intelligence.gas_storage_arbitrage
(
    as_of_date Date,
    hub LowCardinality(String),
    region Nullable(LowCardinality(String)),
    curve_reference LowCardinality(String),
    expected_storage_value Float64,
    breakeven_spread Nullable(Float64),
    optimal_schedule String,
    cost_parameters String,
    constraint_summary Nullable(String),
    diagnostics Nullable(String),
    model_version LowCardinality(String) DEFAULT 'v1',
    created_at DateTime64(3, 'UTC') DEFAULT now()
)
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY toYYYYMM(as_of_date)
ORDER BY (hub, region, as_of_date, model_version);

CREATE TABLE IF NOT EXISTS market_intelligence.coal_gas_switching
(
    as_of_date Date,
    region LowCardinality(String),
    coal_cost_mwh Float64,
    gas_cost_mwh Float64,
    co2_price Float64,
    breakeven_gas_price Float64,
    switch_share Float64,
    diagnostics Nullable(String),
    model_version LowCardinality(String) DEFAULT 'v1',
    created_at DateTime64(3, 'UTC') DEFAULT now()
)
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY toYYYYMM(as_of_date)
ORDER BY (region, as_of_date, model_version);

CREATE TABLE IF NOT EXISTS market_intelligence.gas_basis_models
(
    as_of_date Date,
    hub LowCardinality(String),
    predicted_basis Float64,
    actual_basis Nullable(Float64),
    feature_snapshot String,
    diagnostics Nullable(String),
    method LowCardinality(String) DEFAULT 'linear_regression',
    model_version LowCardinality(String) DEFAULT 'v1',
    created_at DateTime64(3, 'UTC') DEFAULT now()
)
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY toYYYYMM(as_of_date)
ORDER BY (hub, as_of_date, model_version);

CREATE TABLE IF NOT EXISTS market_intelligence.lng_routing_optimization
(
    as_of_date Date,
    route_id LowCardinality(String),
    export_terminal LowCardinality(String),
    import_terminal LowCardinality(String),
    vessel_type LowCardinality(String),
    cargo_size_bcf Float64,
    vessel_speed_knots Float64,
    fuel_price_usd_per_tonne Float64,
    distance_nm Float64,
    voyage_time_days Float64,
    fuel_consumption_tonnes Float64,
    fuel_cost_usd Float64,
    charter_cost_usd Float64,
    port_cost_usd Nullable(Float64),
    total_cost_usd Float64,
    cost_per_mmbtu_usd Float64,
    is_optimal_route UInt8,
    assumptions Nullable(String),
    diagnostics Nullable(String),
    model_version LowCardinality(String) DEFAULT 'v1',
    created_at DateTime64(3, 'UTC') DEFAULT now()
)
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY toYYYYMM(as_of_date)
ORDER BY (export_terminal, import_terminal, as_of_date, route_id, model_version);

CREATE TABLE IF NOT EXISTS market_intelligence.coal_transport_costs
(
    as_of_month Date,
    route_id LowCardinality(String),
    origin_region LowCardinality(String),
    destination_region LowCardinality(String),
    transport_mode LowCardinality(String),
    vessel_type Nullable(LowCardinality(String)),
    cargo_tonnes Float64,
    fuel_price_usd_per_tonne Float64,
    freight_cost_usd Float64,
    bunker_cost_usd Nullable(Float64),
    port_fees_usd Nullable(Float64),
    congestion_premium_usd Nullable(Float64),
    carbon_cost_usd Nullable(Float64),
    demurrage_cost_usd Nullable(Float64),
    rail_cost_usd Nullable(Float64),
    truck_cost_usd Nullable(Float64),
    total_cost_usd Float64,
    currency FixedString(3) DEFAULT 'USD',
    assumptions Nullable(String),
    diagnostics Nullable(String),
    model_version LowCardinality(String) DEFAULT 'v1',
    created_at DateTime64(3, 'UTC') DEFAULT now()
)
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY toYYYYMM(as_of_month)
ORDER BY (origin_region, destination_region, as_of_month, transport_mode, model_version);

CREATE TABLE IF NOT EXISTS market_intelligence.pipeline_congestion_forecast
(
    forecast_date Date,
    pipeline_id LowCardinality(String),
    market LowCardinality(String),
    segment LowCardinality(String),
    horizon_days UInt16,
    utilization_forecast_pct Float64,
    utilization_actual_pct Nullable(Float64),
    congestion_probability Float64,
    risk_score Float64,
    risk_tier LowCardinality(String),
    alert_level Nullable(LowCardinality(String)),
    drivers Nullable(String),
    diagnostics Nullable(String),
    model_version LowCardinality(String) DEFAULT 'v1',
    created_at DateTime64(3, 'UTC') DEFAULT now()
)
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY toYYYYMM(forecast_date)
ORDER BY (pipeline_id, forecast_date, horizon_days, model_version);

CREATE TABLE IF NOT EXISTS market_intelligence.seasonal_demand_forecast
(
    forecast_date Date,
    region LowCardinality(String),
    sector Nullable(LowCardinality(String)),
    scenario_id LowCardinality(String) DEFAULT 'BASE',
    base_demand_mw Float64,
    weather_adjustment_mw Nullable(Float64),
    economic_adjustment_mw Nullable(Float64),
    holiday_adjustment_mw Nullable(Float64),
    final_forecast_mw Float64,
    peak_risk_score Nullable(Float64),
    confidence_low_mw Nullable(Float64),
    confidence_high_mw Nullable(Float64),
    diagnostics Nullable(String),
    model_version LowCardinality(String) DEFAULT 'v1',
    created_at DateTime64(3, 'UTC') DEFAULT now()
)
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY toYYYYMM(forecast_date)
ORDER BY (region, forecast_date, scenario_id, model_version);

-- Aggregations for research dashboards
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.mv_commodity_decomposition_daily
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(snapshot_date)
ORDER BY (instrument_id, method, snapshot_date)
AS
SELECT
    snapshot_date,
    instrument_id,
    method,
    avgState(trend) AS avg_trend_state,
    avgState(abs(seasonal)) AS seasonal_intensity_state
FROM market_intelligence.commodity_decomposition
GROUP BY snapshot_date, instrument_id, method;

CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.mv_volatility_regime_share_monthly
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(month)
ORDER BY (instrument_id, regime_label, month)
AS
SELECT
    toStartOfMonth(date) AS month,
    instrument_id,
    regime_label,
    count() AS observation_count
FROM market_intelligence.volatility_regimes
GROUP BY month, instrument_id, regime_label;

-- Commodity-specific indexes
ALTER TABLE market_intelligence.futures_curves ADD INDEX IF NOT EXISTS idx_commodity commodity_code TYPE bloom_filter GRANULARITY 4;

-- Materialized views for commodity analytics

-- Daily commodity price summary
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.commodity_price_daily_summary
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (commodity_type, market, instrument_id, date)
AS SELECT
    toDate(event_time) as date,
    commodity_type,
    market,
    instrument_id,
    location_code,
    price_type,
    avg(value) as avg_price,
    min(value) as min_price,
    max(value) as max_price,
    count() as tick_count,
    sum(volume) as total_volume,
    argMin(value, event_time) as first_price,
    argMax(value, event_time) as last_price
FROM market_intelligence.market_price_ticks
GROUP BY date, commodity_type, market, instrument_id, location_code, price_type;

-- Commodity volatility surface
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.commodity_volatility_surface
ENGINE = ReplacingMergeTree()
PARTITION BY toYYYYMM(as_of_date)
ORDER BY (commodity_type, instrument_id, as_of_date)
AS SELECT
    toDate(event_time) as as_of_date,
    commodity_type,
    instrument_id,
    -- Calculate realized volatility over different horizons
    stddevPop(value) OVER (
        PARTITION BY commodity_type, instrument_id
        ORDER BY event_time
        ROWS BETWEEN 30 PRECEDING AND CURRENT ROW
    ) as volatility_30d,
    stddevPop(value) OVER (
        PARTITION BY commodity_type, instrument_id
        ORDER BY event_time
        ROWS BETWEEN 90 PRECEDING AND CURRENT ROW
    ) as volatility_90d,
    stddevPop(value) OVER (
        PARTITION BY commodity_type, instrument_id
        ORDER BY event_time
        ROWS BETWEEN 365 PRECEDING AND CURRENT ROW
    ) as volatility_365d
FROM market_intelligence.market_price_ticks
WHERE event_time >= now() - INTERVAL 365 DAY;

-- Cross-commodity correlation matrix
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.commodity_correlations
ENGINE = ReplacingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, commodity1, commodity2)
AS SELECT
    toDate(event_time) as date,
    i1.commodity_type as commodity1,
    i2.commodity_type as commodity2,
    i1.instrument_id as instrument1,
    i2.instrument_id as instrument2,
    corr(i1.value, i2.value) as correlation_coefficient,
    count(*) as sample_count
FROM market_intelligence.market_price_ticks i1
JOIN market_intelligence.market_price_ticks i2 ON
    i1.event_time = i2.event_time AND
    i1.commodity_type != i2.commodity_type
WHERE i1.event_time >= now() - INTERVAL 90 DAY
GROUP BY date, commodity1, commodity2, instrument1, instrument2;

-- Daily commodity feature store for analytics
CREATE TABLE IF NOT EXISTS market_intelligence.daily_commodity_features
(
    snapshot_date     Date,
    instrument_id     LowCardinality(String),
    feature_name      LowCardinality(String),
    feature_category  LowCardinality(String),
    feature_value     Float64,
    created_at        DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree()
PARTITION BY toYYYYMM(snapshot_date)
ORDER BY (instrument_id, feature_name, snapshot_date);

-- Aggregated view of daily commodity features by category
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.daily_commodity_feature_aggregates
ENGINE = ReplacingMergeTree()
PARTITION BY toYYYYMM(snapshot_date)
ORDER BY (instrument_id, feature_category, snapshot_date)
AS SELECT
    snapshot_date,
    instrument_id,
    feature_category,
    avg(feature_value) as avg_feature_value,
    max(feature_value) as max_feature_value,
    min(feature_value) as min_feature_value,
    count() as observations,
    now() as computed_at
FROM market_intelligence.daily_commodity_features
GROUP BY snapshot_date, instrument_id, feature_category;

-- Commodity freshness tracking
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.commodity_freshness
ENGINE = ReplacingMergeTree()
PARTITION BY toYYYYMM(event_time)
ORDER BY (commodity_type, market, source, event_time)
AS SELECT
    commodity_type,
    market,
    source,
    max(event_time) as last_update,
    now() - max(event_time) as staleness_seconds,
    countIf(event_time > now() - INTERVAL 5 MINUTE) as recent_count,
    countIf(event_time > now() - INTERVAL 1 HOUR) as hourly_count
FROM market_intelligence.market_price_ticks
GROUP BY commodity_type, market, source;

-- Performance optimization: Data tiering materialized views

-- Hot data tier (last 7 days) - for real-time queries
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.market_price_ticks_hot
ENGINE = MergeTree()
PARTITION BY (toYYYYMM(event_time), commodity_type)
ORDER BY (commodity_type, instrument_id, event_time)
TTL event_time + INTERVAL 7 DAY
AS
SELECT *
FROM market_intelligence.market_price_ticks
WHERE event_time >= now() - INTERVAL 7 DAY;

-- Warm data tier (last 90 days) - for analytics queries
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.market_price_ticks_warm
ENGINE = MergeTree()
PARTITION BY (toYYYYMM(event_time), commodity_type)
ORDER BY (commodity_type, instrument_id, event_time)
TTL event_time + INTERVAL 90 DAY
AS
SELECT *
FROM market_intelligence.market_price_ticks
WHERE event_time >= now() - INTERVAL 90 DAY
  AND event_time < now() - INTERVAL 7 DAY;

-- Cold data tier (older than 90 days) - for historical analysis
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.market_price_ticks_cold
ENGINE = MergeTree()
PARTITION BY (toYYYYMM(event_time), commodity_type)
ORDER BY (commodity_type, instrument_id, event_time)
AS
SELECT *
FROM market_intelligence.market_price_ticks
WHERE event_time < now() - INTERVAL 90 DAY;

-- Compressed analytics tables for massive scale

-- Hourly aggregated data for dashboard performance
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.market_price_hourly_agg
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(toDate(event_time))
ORDER BY (commodity_type, instrument_id, toDate(event_time), toHour(event_time))
AS SELECT
    toDate(event_time) as date,
    toHour(event_time) as hour,
    commodity_type,
    instrument_id,
    location_code,
    price_type,
    avg(value) as avg_price,
    min(value) as min_price,
    max(value) as max_price,
    count() as tick_count,
    sum(volume) as total_volume,
    stddevPop(value) as price_volatility,
    argMin(value, event_time) as first_price,
    argMax(value, event_time) as last_price
FROM market_intelligence.market_price_ticks
GROUP BY commodity_type, instrument_id, location_code, price_type, date, hour;

-- Daily aggregated data for reporting
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.market_price_daily_agg
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (commodity_type, instrument_id, date)
AS SELECT
    toDate(event_time) as date,
    commodity_type,
    instrument_id,
    location_code,
    price_type,
    avg(value) as avg_price,
    min(value) as min_price,
    max(value) as max_price,
    count() as tick_count,
    sum(volume) as total_volume,
    stddevPop(value) as price_volatility,
    avg(value) as open_price,  -- Simplified for performance
    avg(value) as close_price  -- Simplified for performance
FROM market_intelligence.market_price_ticks
GROUP BY commodity_type, instrument_id, location_code, price_type, date;

-- Real-time analytics for streaming data
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.market_price_realtime_stats
ENGINE = ReplacingMergeTree()
PARTITION BY toYYYYMM(event_time)
ORDER BY (commodity_type, instrument_id, event_time)
AS SELECT
    commodity_type,
    instrument_id,
    max(event_time) as last_update,
    count() as total_ticks,
    avg(value) as current_price,
    stddevPop(value) as price_volatility,
    min(value) as price_min_1h,
    max(value) as price_max_1h,
    countIf(event_time > now() - INTERVAL 1 HOUR) as ticks_last_hour
FROM market_intelligence.market_price_ticks
WHERE event_time >= now() - INTERVAL 1 HOUR
GROUP BY commodity_type, instrument_id;

-- ClickHouse initialization script for analytics tables
-- Defines engines, partitions, ordering keys for performance
ALTER TABLE market_intelligence.market_price_ticks
MODIFY TTL event_time + INTERVAL 2 YEAR;
CREATE OR REPLACE VIEW ch.market_price_ticks AS
    SELECT * FROM market_intelligence.market_price_ticks;
CREATE OR REPLACE VIEW ch.forward_curve_points AS
    SELECT * FROM market_intelligence.forward_curve_points;
CREATE OR REPLACE VIEW ch.fundamentals_series AS
    SELECT * FROM market_intelligence.fundamentals_series;
CREATE OR REPLACE VIEW ch.commodity_decomposition AS
    SELECT * FROM market_intelligence.commodity_decomposition;
CREATE OR REPLACE VIEW ch.volatility_regimes AS
    SELECT * FROM market_intelligence.volatility_regimes;
CREATE OR REPLACE VIEW ch.supply_demand_metrics AS
    SELECT * FROM market_intelligence.supply_demand_metrics;
CREATE OR REPLACE VIEW ch.weather_impact AS
    SELECT * FROM market_intelligence.weather_impact;
CREATE OR REPLACE VIEW ch.gas_storage_arbitrage AS
    SELECT * FROM market_intelligence.gas_storage_arbitrage;
CREATE OR REPLACE VIEW ch.coal_gas_switching AS
    SELECT * FROM market_intelligence.coal_gas_switching;
CREATE OR REPLACE VIEW ch.gas_basis_models AS
    SELECT * FROM market_intelligence.gas_basis_models;
CREATE OR REPLACE VIEW ch.lng_routing_optimization AS
    SELECT * FROM market_intelligence.lng_routing_optimization;
CREATE OR REPLACE VIEW ch.coal_transport_costs AS
    SELECT * FROM market_intelligence.coal_transport_costs;
CREATE OR REPLACE VIEW ch.pipeline_congestion_forecast AS
    SELECT * FROM market_intelligence.pipeline_congestion_forecast;
CREATE OR REPLACE VIEW ch.seasonal_demand_forecast AS
    SELECT * FROM market_intelligence.seasonal_demand_forecast;
CREATE OR REPLACE VIEW ch.commodity_specifications AS
    SELECT * FROM market_intelligence.commodity_specifications;
CREATE OR REPLACE VIEW ch.quality_specifications AS
    SELECT * FROM market_intelligence.quality_specifications;
CREATE OR REPLACE VIEW ch.futures_curves AS
    SELECT * FROM market_intelligence.futures_curves;

-- Kafka landing tables for commodities ingestion
CREATE TABLE IF NOT EXISTS market_intelligence.market_price_ticks_kafka
(
    event_time_utc DateTime64(3, 'UTC'),
    arrival_time_utc Nullable(DateTime64(3, 'UTC')),
    market String,
    product String,
    instrument_id String,
    location_code String,
    price_type String,
    value Float64,
    volume Nullable(Float64),
    currency String,
    unit String,
    source String,
    commodity_type String,
    version_id UInt32 DEFAULT 1
)
ENGINE = Kafka
SETTINGS kafka_broker_list = 'kafka:9092',
        kafka_topic_list = 'commodities.ticks.v1',
        kafka_group_name = 'ch-commodities-ticks-ingestor',
        kafka_format = 'JSONEachRow',
        kafka_num_consumers = 1,
        kafka_handle_error_mode = 'stream';

CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.market_price_ticks_mv
TO market_intelligence.market_price_ticks
AS
SELECT
    toDateTime64(event_time_utc, 3, 'UTC') AS event_time,
    coalesce(arrival_time_utc, now64(3)) AS arrival_time,
    market,
    product,
    instrument_id,
    location_code,
    price_type,
    value,
    volume,
    upper(currency) AS currency,
    unit,
    source,
    coalesce(nullIf(commodity_type, ''), 'oil') AS commodity_type,
    version_id
FROM market_intelligence.market_price_ticks_kafka;

CREATE TABLE IF NOT EXISTS market_intelligence.futures_curves_kafka
(
    as_of_date Date,
    contract_month Date,
    commodity_code String,
    settlement_price Float64,
    open_interest Int64,
    volume Int64,
    currency String,
    unit String,
    exchange String,
    source String,
    version_id UInt32 DEFAULT 1
)
ENGINE = Kafka
SETTINGS kafka_broker_list = 'kafka:9092',
        kafka_topic_list = 'commodities.futures.v1',
        kafka_group_name = 'ch-commodities-futures-ingestor',
        kafka_format = 'JSONEachRow',
        kafka_num_consumers = 1,
        kafka_handle_error_mode = 'stream';

CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.futures_curves_mv
TO market_intelligence.futures_curves
AS
SELECT
    as_of_date,
    commodity_code,
    contract_month,
    settlement_price,
    open_interest,
    volume,
    upper(currency) AS currency,
    unit,
    exchange,
    source,
    version_id,
    now64(3) AS created_at
FROM market_intelligence.futures_curves_kafka;
