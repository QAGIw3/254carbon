-- ClickHouse Infrastructure Time Series Schema
-- High-performance storage for infrastructure metrics

-- Infrastructure metrics time series
CREATE TABLE IF NOT EXISTS market_intelligence.infrastructure_metrics
(
    ts                  DateTime64(3, 'UTC'),
    asset_id            LowCardinality(String),
    asset_type          LowCardinality(String),
    metric              LowCardinality(String),
    value               Float64,
    unit                LowCardinality(String),
    country             LowCardinality(String),
    region              LowCardinality(String),
    operator            LowCardinality(String),
    coordinates         Tuple(lat Float64, lon Float64),
    metadata            String,  -- JSON string
    source              LowCardinality(String),
    version_id          UInt32 DEFAULT 1
)
ENGINE = ReplacingMergeTree(version_id)
PARTITION BY toYYYYMM(ts)
ORDER BY (asset_type, asset_id, metric, ts);

-- LNG terminal inventory and flows
CREATE TABLE IF NOT EXISTS market_intelligence.lng_terminal_data
(
    ts                  DateTime64(3, 'UTC'),
    terminal_id         LowCardinality(String),
    terminal_name       LowCardinality(String),
    country             LowCardinality(String),
    inventory_gwh       Float64,
    inventory_mcm       Float64,
    fullness_pct        Float64,
    send_out_gwh        Float64,
    ship_arrivals       UInt32,
    capacity_gwh        Float64,
    coordinates         Tuple(lat Float64, lon Float64),
    source              LowCardinality(String),
    version_id          UInt32 DEFAULT 1
)
ENGINE = ReplacingMergeTree(version_id)
PARTITION BY toYYYYMM(ts)
ORDER BY (terminal_id, ts);

-- Power plant generation and capacity
CREATE TABLE IF NOT EXISTS market_intelligence.power_plant_data
(
    ts                  DateTime64(3, 'UTC'),
    plant_id            LowCardinality(String),
    plant_name          String,
    country             LowCardinality(String),
    fuel_type           LowCardinality(String),
    capacity_mw         Float64,
    generation_mwh      Float64,
    capacity_factor     Float64,
    availability_pct    Float64,
    emissions_tco2      Float64,
    heat_rate           Float64,
    coordinates         Tuple(lat Float64, lon Float64),
    source              LowCardinality(String),
    version_id          UInt32 DEFAULT 1
)
ENGINE = ReplacingMergeTree(version_id)
PARTITION BY toYYYYMM(ts)
ORDER BY (country, fuel_type, plant_id, ts);

-- Transmission line flows and availability
CREATE TABLE IF NOT EXISTS market_intelligence.transmission_flows
(
    ts                  DateTime64(3, 'UTC'),
    line_id             LowCardinality(String),
    from_zone           LowCardinality(String),
    to_zone             LowCardinality(String),
    flow_mw             Float64,
    capacity_mw         Float64,
    utilization_pct     Float64,
    availability_pct    Float64,
    congestion_hours    Float64,
    voltage_kv          Float64,
    line_type           LowCardinality(String),
    countries           Array(String),
    source              LowCardinality(String),
    version_id          UInt32 DEFAULT 1
)
ENGINE = ReplacingMergeTree(version_id)
PARTITION BY toYYYYMM(ts)
ORDER BY (line_id, ts);

-- Renewable resource assessments
CREATE TABLE IF NOT EXISTS market_intelligence.renewable_resources
(
    location_id         String,  -- Encoded lat/lon grid point
    latitude            Float64,
    longitude           Float64,
    resource_type       LowCardinality(String),
    annual_average      Float64,
    monthly_avg_jan     Float64,
    monthly_avg_feb     Float64,
    monthly_avg_mar     Float64,
    monthly_avg_apr     Float64,
    monthly_avg_may     Float64,
    monthly_avg_jun     Float64,
    monthly_avg_jul     Float64,
    monthly_avg_aug     Float64,
    monthly_avg_sep     Float64,
    monthly_avg_oct     Float64,
    monthly_avg_nov     Float64,
    monthly_avg_dec     Float64,
    unit                LowCardinality(String),
    data_year           UInt16,
    resolution_km       Float32,
    data_source         LowCardinality(String),
    created_at          DateTime64(3, 'UTC') DEFAULT now()
)
ENGINE = ReplacingMergeTree()
ORDER BY (resource_type, location_id);

-- Infrastructure project tracking
CREATE TABLE IF NOT EXISTS market_intelligence.infrastructure_projects
(
    update_date         Date,
    project_id          LowCardinality(String),
    project_name        String,
    project_type        LowCardinality(String),
    countries           Array(String),
    status              LowCardinality(String),
    capacity_mw         Float64,
    voltage_kv          Float64,
    length_km           Float64,
    progress_pct        Float64,
    estimated_cost_musd Float64,
    start_year          UInt16,
    completion_year     UInt16,
    developer           String,
    metadata            String,  -- JSON
    source              LowCardinality(String),
    version_id          UInt32 DEFAULT 1
)
ENGINE = ReplacingMergeTree(version_id)
PARTITION BY toYYYYMM(update_date)
ORDER BY (project_type, status, project_id, update_date);

-- Aggregated infrastructure statistics (daily rollups)
CREATE TABLE IF NOT EXISTS market_intelligence.infrastructure_daily_stats
(
    date                Date,
    country             LowCardinality(String),
    asset_type          LowCardinality(String),
    fuel_type           LowCardinality(String),
    total_capacity      Float64,
    available_capacity  Float64,
    total_generation    Float64,
    avg_capacity_factor Float64,
    num_assets          UInt32,
    num_operational     UInt32,
    metadata            String,
    source              LowCardinality(String)
)
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (country, asset_type, fuel_type, date);

-- Create materialized views for common aggregations

-- LNG terminal inventory by country
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.lng_inventory_by_country
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (country, date)
AS SELECT
    toDate(ts) as date,
    country,
    sum(inventory_gwh) as total_inventory_gwh,
    sum(capacity_gwh) as total_capacity_gwh,
    avg(fullness_pct) as avg_fullness_pct,
    sum(send_out_gwh) as total_send_out_gwh,
    sum(ship_arrivals) as total_ship_arrivals,
    count() as num_terminals
FROM market_intelligence.lng_terminal_data
GROUP BY date, country;

-- Power generation by fuel type
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.generation_by_fuel
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (country, fuel_type, date)
AS SELECT
    toDate(ts) as date,
    country,
    fuel_type,
    sum(generation_mwh) as total_generation_mwh,
    sum(capacity_mw) as total_capacity_mw,
    avg(capacity_factor) as avg_capacity_factor,
    sum(emissions_tco2) as total_emissions_tco2,
    count(DISTINCT plant_id) as num_plants
FROM market_intelligence.power_plant_data
GROUP BY date, country, fuel_type;

-- Transmission utilization summary
CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.transmission_utilization
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (from_zone, to_zone, date)
AS SELECT
    toDate(ts) as date,
    from_zone,
    to_zone,
    avgState(flow_mw) as avg_flow_state,
    maxState(flow_mw) as max_flow_state,
    minState(flow_mw) as min_flow_state,
    avgState(utilization_pct) as avg_utilization_state,
    sumState(congestion_hours) as congestion_hours_state
FROM market_intelligence.transmission_flows
GROUP BY date, from_zone, to_zone;

-- Create indexes for performance
ALTER TABLE market_intelligence.infrastructure_metrics 
    ADD INDEX IF NOT EXISTS idx_country country TYPE bloom_filter GRANULARITY 4;
ALTER TABLE market_intelligence.infrastructure_metrics 
    ADD INDEX IF NOT EXISTS idx_coordinates coordinates TYPE minmax GRANULARITY 4;

ALTER TABLE market_intelligence.lng_terminal_data 
    ADD INDEX IF NOT EXISTS idx_terminal terminal_id TYPE bloom_filter GRANULARITY 4;
ALTER TABLE market_intelligence.lng_terminal_data 
    ADD INDEX IF NOT EXISTS idx_country country TYPE bloom_filter GRANULARITY 4;

ALTER TABLE market_intelligence.power_plant_data 
    ADD INDEX IF NOT EXISTS idx_fuel fuel_type TYPE set(32) GRANULARITY 4;
ALTER TABLE market_intelligence.power_plant_data 
    ADD INDEX IF NOT EXISTS idx_country country TYPE bloom_filter GRANULARITY 4;

ALTER TABLE market_intelligence.transmission_flows 
    ADD INDEX IF NOT EXISTS idx_zones (from_zone, to_zone) TYPE bloom_filter GRANULARITY 4;

ALTER TABLE market_intelligence.renewable_resources 
    ADD INDEX IF NOT EXISTS idx_type resource_type TYPE bloom_filter GRANULARITY 4;

-- Create dictionaries for asset lookups
CREATE DICTIONARY IF NOT EXISTS market_intelligence.infrastructure_assets_dict
(
    asset_id String,
    asset_name String,
    asset_type String,
    country String,
    latitude Float64,
    longitude Float64,
    capacity_mw Float64,
    operator String
)
PRIMARY KEY asset_id
SOURCE(CLICKHOUSE(
    HOST 'localhost'
    PORT 9000
    USER 'default'
    TABLE 'infrastructure_assets'
    DB 'market_intelligence'
))
LIFETIME(MIN 3600 MAX 7200)
LAYOUT(HASHED());
