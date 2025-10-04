"""
Kafka → ClickHouse fundamentals consumer.

Reads JSON messages from topic 'market.fundamentals' and inserts rows into
ch.fundamentals_series with mapping:
  ts = from event_time_utc (ms)
  market = market
  entity_id = instrument_id (or location_code)
  variable = product
  value = value
  unit = unit
  scenario_id = 'BASE'
  source = source
"""
import os
import json
import ujson
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from kafka import KafkaConsumer
from clickhouse_driver import Client as ClickHouseClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fundamentals_consumer")


def get_clickhouse_client() -> ClickHouseClient:
    host = os.getenv("CLICKHOUSE_HOST", "clickhouse")
    port = int(os.getenv("CLICKHOUSE_PORT", "9000"))
    return ClickHouseClient(host=host, port=port, database="ch", send_receive_timeout=60, settings={
        'async_insert': 1,
        'wait_for_async_insert': 0,
        'max_threads': 16,
        'use_uncompressed_cache': 0,
    })


def to_row(msg: Dict[str, Any]) -> List[Any]:
    ts_ms = msg.get("event_time_utc")
    if isinstance(ts_ms, str):
        try:
            ts_ms = int(ts_ms)
        except Exception:
            ts_ms = None
    ts_dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) if ts_ms else datetime.now(timezone.utc)
    market = msg.get("market", "unknown")
    entity = msg.get("instrument_id") or msg.get("location_code") or "UNKNOWN"
    variable = msg.get("product", "unknown")
    value = float(msg.get("value", 0.0))
    unit = msg.get("unit", "unit")
    source = msg.get("source", "unknown")
    return [ts_dt, market, entity, variable, value, unit, "BASE", source, 1]


def consume():
    bootstrap = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
    topic = os.getenv("FUNDAMENTALS_TOPIC", "market.fundamentals")
    group_id = os.getenv("CONSUMER_GROUP", "fundamentals-ch-writer")

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        group_id=group_id,
        enable_auto_commit=True,
        auto_offset_reset="latest",
        value_deserializer=lambda m: ujson.loads(m.decode("utf-8")),
        consumer_timeout_ms=0,
        max_poll_records=1000,
    )

    ch = get_clickhouse_client()
    logger.info(f"Consuming {topic} from {bootstrap} → ClickHouse fundamentals_series")

    batch: List[List[Any]] = []
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))

    try:
        for message in consumer:
            val = message.value
            try:
                row = to_row(val)
                batch.append(row)
            except Exception as e:
                logger.warning(f"Skip malformed message: {e}; value={val}
")
                continue

            if len(batch) >= BATCH_SIZE:
                ch.execute(
                    "INSERT INTO ch.fundamentals_series (ts, market, entity_id, variable, value, unit, scenario_id, source, version_id) VALUES",
                    batch,
                    types_check=True,
                )
                logger.info(f"Inserted batch of {len(batch)} rows")
                batch = []
    except KeyboardInterrupt:
        logger.info("Shutting down consumer...")
    finally:
        if batch:
            ch.execute(
                "INSERT INTO ch.fundamentals_series (ts, market, entity_id, variable, value, unit, scenario_id, source, version_id) VALUES",
                batch,
                types_check=True,
            )
            logger.info(f"Inserted final batch of {len(batch)} rows")


if __name__ == "__main__":
    consume()

