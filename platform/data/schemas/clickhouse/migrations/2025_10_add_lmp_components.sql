-- Migration: add LMP component columns to market_price_ticks
-- Safe to run multiple times (IF NOT EXISTS guards)

ALTER TABLE market_intelligence.market_price_ticks
    ADD COLUMN IF NOT EXISTS energy_component Nullable(Decimal(10,4)) CODEC(T64, LZ4) AFTER value;

ALTER TABLE market_intelligence.market_price_ticks
    ADD COLUMN IF NOT EXISTS congestion_component Nullable(Decimal(10,4)) CODEC(T64, LZ4) AFTER energy_component;

ALTER TABLE market_intelligence.market_price_ticks
    ADD COLUMN IF NOT EXISTS loss_component Nullable(Decimal(10,4)) CODEC(T64, LZ4) AFTER congestion_component;

-- Optional indexes (sparse) for component analytics; adjust if needed
-- ALTER TABLE market_intelligence.market_price_ticks ADD INDEX IF NOT EXISTS idx_component_nonnull (isNotNull(energy_component)) TYPE minmax GRANULARITY 1;

-- Example Kafka → ClickHouse ingestion mapping (JSONEachRow)
-- Adjust topic, brokers, and format to your environment (AvroConfluent, etc.).
--
-- ENGINE = Kafka
-- SETTINGS kafka_broker_list = 'kafka:9092',
--          kafka_topic_list = 'market.price.ticks',
--          kafka_group_name = 'ch-ticks-ingestor',
--          kafka_format = 'JSONEachRow',
--          kafka_num_consumers = 1;
--
-- CREATE MATERIALIZED VIEW IF NOT EXISTS market_intelligence.mv_market_price_ticks
-- TO market_intelligence.market_price_ticks
-- AS
-- SELECT
--   toDateTime64(event_time_utc, 3) AS event_time,
--   instrument_id,
--   location_code,
--   price_type,
--   toDecimal64(value, 4) AS value,
--   toDecimal64OrNull(volume, 4) AS volume,
--   currency,
--   unit,
--   source,
--   toDecimal64OrNull(energy_component, 4) AS energy_component,
--   toDecimal64OrNull(congestion_component, 4) AS congestion_component,
--   toDecimal64OrNull(loss_component, 4) AS loss_component,
--   now() AS created_at
-- FROM market_intelligence.market_price_ticks_kafka;

