import os
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from kafka import KafkaConsumer, KafkaProducer
from clickhouse_driver import Client as ClickHouseClient

from platform.shared.data_quality_framework import DataQualityFramework

logger = logging.getLogger(__name__)


def _get_ch() -> ClickHouseClient:
    return ClickHouseClient(
        host=os.getenv("CLICKHOUSE_HOST", "clickhouse"),
        port=int(os.getenv("CLICKHOUSE_PORT", "9000")),
        database="market_intelligence",
        settings={"async_insert": 1, "wait_for_async_insert": 0},
    )


def start_stream_worker():
    bootstrap = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
    topic = os.getenv("DQ_INPUT_TOPIC", "commodities.ticks.v1")
    flags_topic = os.getenv("DQ_FLAGS_TOPIC", "dq.flags.v1")

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        group_id="dq-stream-worker",
        enable_auto_commit=True,
        auto_offset_reset="latest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        max_poll_records=500,
    )

    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        acks="all",
        enable_idempotence=True,
        compression_type="zstd",
        linger_ms=20,
        batch_size=262144,
        value_serializer=lambda v: json.dumps(v, separators=(",", ":")).encode("utf-8"),
        key_serializer=lambda v: v.encode("utf-8"),
    )

    dq = DataQualityFramework()
    ch = _get_ch()

    for message in consumer:
        val: Dict[str, Any] = message.value
        try:
            instrument_id = val.get("instrument_id") or "UNKNOWN"
            commodity_type = val.get("commodity_type") or val.get("market") or "unknown"
            event_time = val.get("event_time_utc")
            product = val.get("product")
            if isinstance(event_time, (int, float)):
                ts = datetime.fromtimestamp(event_time / 1000, tz=timezone.utc)
            else:
                ts = datetime.now(timezone.utc)

            # Simple window-less outlier check via framework (placeholder)
            outlier = False
            score = 0.0
            rule_id = "stream_outlier_threshold"
            severity = "warning"
            value = val.get("value")
            numeric_value = None
            if value is not None:
                try:
                    numeric_value = float(value)
                except Exception:
                    numeric_value = None

            if numeric_value is not None:
                refined_bounds = dq.get_metric_rules("refined_products").get(instrument_id)
                transport_bounds = dq.get_metric_rules("transport_fundamentals").get(product or "")
                rin_bounds = dq.get_metric_rules("rins").get(instrument_id)
                carbon_bounds = dq.get_metric_rules("carbon_prices").get(instrument_id)

                if refined_bounds and not (refined_bounds["value_min"] <= numeric_value <= refined_bounds["value_max"]):
                    lower = refined_bounds["value_min"]
                    upper = refined_bounds["value_max"]
                    delta = lower - numeric_value if numeric_value < lower else numeric_value - upper
                    outlier = True
                    score = min(1.0, abs(delta) / max(upper, 1))
                    rule_id = "refined_product_bounds"
                    severity = "critical"
                elif carbon_bounds and not (carbon_bounds["value_min"] <= numeric_value <= carbon_bounds["value_max"]):
                    lower = carbon_bounds["value_min"]
                    upper = carbon_bounds["value_max"]
                    delta = lower - numeric_value if numeric_value < lower else numeric_value - upper
                    outlier = True
                    score = min(1.0, abs(delta) / max(upper, 1))
                    rule_id = "carbon_price_bounds"
                    severity = "warning"
                elif rin_bounds and not (rin_bounds["value_min"] <= numeric_value <= rin_bounds["value_max"]):
                    lower = rin_bounds["value_min"]
                    upper = rin_bounds["value_max"]
                    delta = lower - numeric_value if numeric_value < lower else numeric_value - upper
                    outlier = True
                    score = min(1.0, abs(delta) / max(upper, 1))
                    rule_id = "rin_price_bounds"
                    severity = "critical"
                elif transport_bounds and not (transport_bounds["value_min"] <= numeric_value <= transport_bounds["value_max"]):
                    outlier = True
                    score = 0.7
                    rule_id = "transport_fundamental_bounds"
                    severity = "warning"

            if not outlier and numeric_value is not None and abs(numeric_value) > 1e6:
                outlier = True
                score = 1.0

            if outlier:
                # Write issue row (lightweight)
                ch.execute(
                    """
                    INSERT INTO market_intelligence.data_quality_issues
                    (event_time, source, instrument_id, commodity_type, dimension, severity, rule_id, value, expected, metadata, run_id)
                    VALUES
                    """,
                    [[
                        ts,
                        val.get("source", "stream"),
                        instrument_id,
                        commodity_type,
                        "validity",
                        severity,
                        rule_id,
                        str(value),
                        None,
                        {"score": score},
                        os.getenv("RUN_ID", "00000000-0000-0000-0000-000000000000"),
                    ]],
                    types_check=True,
                )

                # Publish flag
                producer.send(
                    flags_topic,
                    key=instrument_id,
                    value={
                        "instrument_id": instrument_id,
                        "commodity_type": commodity_type,
                        "event_time": ts.isoformat(),
                        "rule": rule_id,
                        "score": score,
                    },
                )
        except Exception:
            logger.exception("stream worker error")
