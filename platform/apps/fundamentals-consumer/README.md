Fundamentals Consumer

Consumes JSON messages from Kafka topic `market.fundamentals`, maps them to the
ClickHouse table `ch.fundamentals_series`, and inserts in small batches.

Assumptions
- Messages follow the connector canonical JSON with keys: event_time_utc, market,
  product (mapped to variable), instrument_id (mapped to entity_id), value, unit, source.

Run locally
- Environment: CLICKHOUSE_HOST, CLICKHOUSE_PORT, KAFKA_BOOTSTRAP
- python main.py

