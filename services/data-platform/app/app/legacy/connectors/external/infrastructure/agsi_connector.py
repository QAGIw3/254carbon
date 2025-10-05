"""
GIE AGSI+ Gas Storage Connector
---------------------------------

Overview
--------
Fetches European gas storage transparency data from the Gas Infrastructure
Europe (GIE) Aggregated Gas Storage Inventory (AGSI+) API. Emits canonical
fundamentals for storage level, injections/withdrawals, and fullness.

Data Flow
---------
AGSI+ API → normalize facility/country/EU entries → canonical fundamentals → Kafka

Configuration
-------------
- `api_base`: Base API endpoint (default: https://agsi.gie.eu/api/v1).
- `api_key`: Required API key for AGSI+ access.
- `granularity`: `facility` (default) | `country` | `eu`.
- `entities`: Optional list of facility or country codes to filter.
- `lookback_days`: Window used to determine start date when no checkpoint.
- `include_rollups`: When `facility`, also emit country/EU aggregates.
- `kafka.topic`/`kafka.bootstrap_servers`.

Operational Notes
-----------------
- HTTP calls are retried with exponential backoff and surfaced on failure.
- Units are normalized to GWh where feasible; unknown units are forwarded.
- Checkpoints store the last event time to define the next window.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional

import requests
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ...base import CommodityType, Ingestor

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StorageRecord:
    """Normalized storage observation returned by AGSI+."""

    as_of: datetime
    entity: str
    entity_type: str  # facility|country|eu
    working_gwh: Optional[float]
    fullness_pct: Optional[float]
    injection_gwh: Optional[float]
    withdrawal_gwh: Optional[float]
    capacity_gwh: Optional[float]
    region: Optional[str]
    raw: Dict[str, Any]


class AGSIConnector(Ingestor):
    """Connector for the GIE Aggregated Gas Storage Inventory (AGSI+)."""

    DEFAULT_API_BASE = "https://agsi.gie.eu/api/v1"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.GAS

        self.api_base = config.get("api_base", self.DEFAULT_API_BASE).rstrip("/")
        self.api_key = config.get("api_key")
        self.granularity = (config.get("granularity") or "facility").lower()
        self.entities: Optional[List[str]] = config.get("entities")
        self.lookback_days = int(config.get("lookback_days", 7))
        self.include_rollups = bool(config.get("include_rollups", False))

        kafka_cfg = config.get("kafka") or {}
        if not kafka_cfg:
            self.kafka_topic = "market.fundamentals"
            self.kafka_bootstrap_servers = config.get("kafka_bootstrap", "kafka:9092")

        self.session = requests.Session()
        self._validate_config()

    # ------------------------------------------------------------------
    # Connector lifecycle
    # ------------------------------------------------------------------

    def discover(self) -> Dict[str, Any]:
        """Describe exposed storage metrics and selected entities."""
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "granularity": self.granularity,
            "entities": self.entities or "ALL",
            "streams": [
                {
                    "name": "gas_storage",
                    "variables": [
                        "ng_storage_level_gwh",
                        "ng_storage_pct_full",
                        "ng_injection_gwh",
                        "ng_withdrawal_gwh",
                    ],
                    "frequency": "daily",
                }
            ],
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Fetch storage rows and yield per-metric events (daily cadence)."""
        start, end = self._determine_window()
        logger.info(
            "Fetching AGSI+ data granularity=%s window=%s→%s entities=%s",
            self.granularity,
            start.date(),
            end.date(),
            self.entities or "ALL",
        )

        records = list(self._fetch_records(start, end))

        # Optionally roll up facility data to country aggregates
        if self.granularity == "facility" and self.include_rollups:
            records.extend(self._aggregate_records(records, target="country"))
            records.extend(self._aggregate_records(records, target="eu"))

        for record in records:
            yield from self._record_to_events(record)

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map a flattened storage record into the canonical event format."""
        ts: datetime = raw["as_of"]
        metric: str = raw["metric"]
        unit: str = raw["unit"]
        entity: str = raw["entity"]
        entity_type: str = raw["entity_type"]

        instrument_id = self._build_instrument_id(entity, entity_type)

        payload = {
            "event_time_utc": int(ts.timestamp() * 1000),
            "market": "infra",
            "product": metric,
            "instrument_id": instrument_id,
            "location_code": entity,
            "price_type": "observation",
            "value": raw["value"],
            "unit": unit,
            "source": self.source_id,
            "commodity_type": self.commodity_type.value,
            "metadata": json.dumps(
                {
                    "entity_type": entity_type,
                    "region": raw.get("region"),
                    "capacity_gwh": raw.get("capacity_gwh"),
                }
            ),
        }
        return payload

    def emit(self, events: Iterable[Dict[str, Any]]) -> int:
        """Emit flattened metric events (already normalized)."""
        return super().emit(events)

    def checkpoint(self, state: Dict[str, Any]) -> None:
        super().checkpoint(state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record_to_events(self, record: StorageRecord) -> Iterator[Dict[str, Any]]:
        base = {
            "as_of": record.as_of,
            "entity": record.entity,
            "entity_type": record.entity_type,
            "region": record.region,
            "capacity_gwh": record.capacity_gwh,
        }

        if record.working_gwh is not None:
            yield {**base, "metric": "ng_storage_level_gwh", "value": record.working_gwh, "unit": "GWh"}
        if record.fullness_pct is not None:
            yield {**base, "metric": "ng_storage_pct_full", "value": record.fullness_pct, "unit": "pct"}
        if record.injection_gwh is not None:
            injection = max(record.injection_gwh, 0.0)
            yield {**base, "metric": "ng_injection_gwh", "value": injection, "unit": "GWh"}
        if record.withdrawal_gwh is not None:
            withdrawal = max(record.withdrawal_gwh, 0.0)
            yield {**base, "metric": "ng_withdrawal_gwh", "value": withdrawal, "unit": "GWh"}

    def _build_instrument_id(self, entity: str, entity_type: str) -> str:
        if entity_type == "facility":
            return f"AGSI.{entity}.GAS_STORAGE"
        if entity_type == "country":
            return f"AGSI.{entity}.GAS_STORAGE"
        return "AGSI.EU.GAS_STORAGE"

    def _determine_window(self) -> tuple[datetime, datetime]:
        state = self.checkpoint_state or {}
        last_event_ms = state.get("last_event_time")
        if last_event_ms:
            start = datetime.fromtimestamp(last_event_ms / 1000, tz=timezone.utc)
        else:
            start = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        end = datetime.now(timezone.utc)
        return start.replace(hour=0, minute=0, second=0, microsecond=0), end

    def _fetch_records(self, start: datetime, end: datetime) -> Iterator[StorageRecord]:
        """Query AGSI+ endpoint and yield parsed StorageRecord entries."""
        params = self._build_params(start, end)
        endpoint = self._endpoint_for_granularity()

        try:
            payload = self._request_json(endpoint, params)
        except RetryError as exc:
            logger.error("AGSI+ request failed after retries: %s", exc)
            raise

        data = payload.get("data") or []
        logger.info("AGSI+ returned %s rows", len(data))

        for row in data:
            record = self._parse_row(row)
            if record:
                yield record

    def _build_params(self, start: datetime, end: datetime) -> Dict[str, Any]:
        """Construct request parameters for the selected granularity/entities."""
        params: Dict[str, Any] = {
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
        }
        if self.entities:
            key = {
                "facility": "facility",
                "country": "country",
                "eu": "country",
            }[self.granularity]
            params[key] = ",".join(self.entities)
        return params

    def _endpoint_for_granularity(self) -> str:
        if self.granularity == "facility":
            return f"{self.api_base}/facilities"
        if self.granularity == "country":
            return f"{self.api_base}/countries"
        if self.granularity == "eu":
            return f"{self.api_base}/eu"
        raise ValueError(f"Unsupported granularity {self.granularity}")

    def _parse_row(self, row: Dict[str, Any]) -> Optional[StorageRecord]:
        try:
            gas_day = row.get("gasDayStart")
            if not gas_day:
                return None
            as_of = datetime.fromisoformat(f"{gas_day}T00:00:00+00:00")

            entity_code = row.get("code") or row.get("key") or row.get("name")
            if not entity_code:
                return None

            entity_type = self._infer_entity_type(row)
            unit = (row.get("unit") or "GWh").upper()
            country = row.get("country") or row.get("name")

            working = self._convert_to_gwh(row.get("workingGasVolume"), unit)
            fullness = self._safe_float(row.get("full"))
            injection = self._convert_to_gwh(row.get("injection"), unit)
            withdrawal = self._convert_to_gwh(row.get("withdrawal"), unit)
            capacity = self._convert_to_gwh(row.get("infrastructureVolume"), unit)

            return StorageRecord(
                as_of=as_of,
                entity=entity_code,
                entity_type=entity_type,
                working_gwh=working,
                fullness_pct=fullness,
                injection_gwh=injection,
                withdrawal_gwh=withdrawal,
                capacity_gwh=capacity,
                region=country,
                raw=row,
            )
        except Exception as exc:
            logger.warning("Failed to parse AGSI row %s: %s", row, exc)
            return None

    def _aggregate_records(self, records: List[StorageRecord], target: str) -> List[StorageRecord]:
        if not records:
            return []
        aggregates: Dict[tuple[str, str], Dict[str, Any]] = {}
        for record in records:
            if not record.as_of:
                continue
            if target == "country" and record.region:
                key_entity = record.region
            elif target == "country":
                continue
            else:
                key_entity = "EU"

            key = (record.as_of.isoformat(), key_entity)
            agg = aggregates.setdefault(
                key,
                {
                    "working": 0.0,
                    "capacity": 0.0,
                    "injection": 0.0,
                    "withdrawal": 0.0,
                    "fullness_sum": 0.0,
                    "fullness_count": 0,
                },
            )

            if record.working_gwh is not None:
                agg["working"] += record.working_gwh
            if record.capacity_gwh is not None:
                agg["capacity"] += record.capacity_gwh
            if record.injection_gwh is not None:
                agg["injection"] += record.injection_gwh
            if record.withdrawal_gwh is not None:
                agg["withdrawal"] += record.withdrawal_gwh
            if record.fullness_pct is not None:
                agg["fullness_sum"] += record.fullness_pct
                agg["fullness_count"] += 1

        aggregated_records: List[StorageRecord] = []
        for (iso_ts, entity), metrics in aggregates.items():
            as_of = datetime.fromisoformat(iso_ts)
            fullness_avg = (
                metrics["fullness_sum"] / metrics["fullness_count"]
                if metrics["fullness_count"]
                else None
            )
            entity_type = "country" if target == "country" else "eu"
            aggregated_records.append(
                StorageRecord(
                    as_of=as_of,
                    entity=entity,
                    entity_type=entity_type,
                    working_gwh=metrics["working"],
                    fullness_pct=fullness_avg,
                    injection_gwh=metrics["injection"],
                    withdrawal_gwh=metrics["withdrawal"],
                    capacity_gwh=metrics["capacity"],
                    region=None if target == "eu" else entity,
                    raw={"aggregated": True, "target": target},
                )
            )
        return aggregated_records

    def _infer_entity_type(self, row: Dict[str, Any]) -> str:
        if self.granularity in {"facility", "country", "eu"}:
            return self.granularity
        if row.get("facilityName"):
            return "facility"
        if row.get("country"):
            return "country"
        return "eu"

    def _convert_to_gwh(self, value: Optional[Any], unit: Optional[str]) -> Optional[float]:
        val = self._safe_float(value)
        if val is None:
            return None
        unit = (unit or "GWh").upper()
        if unit == "GWH":
            return val
        if unit == "MWH":
            return val / 1000.0
        if unit == "KWH":
            return val / 1_000_000.0
        # AGSI occasionally returns thousand cubic meters (kWh(mil)) – leave as-is
        return val

    def _safe_float(self, value: Optional[Any]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _validate_config(self) -> None:
        if not self.api_key:
            raise ValueError("AGSI API key must be provided")
        if self.granularity not in {"facility", "country", "eu"}:
            raise ValueError("granularity must be facility|country|eu")

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(
            (requests.HTTPError, requests.ConnectionError, requests.Timeout)
        ),
        reraise=True,
    )
    def _request_json(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "x-key": self.api_key,
            "Accept": "application/json",
        }
        resp = self.session.get(endpoint, params=params, headers=headers, timeout=30)
        if resp.status_code == 401:
            raise requests.HTTPError("Unauthorized access to AGSI API")
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", "15"))
            logger.warning("AGSI rate limit hit, sleeping %s seconds", retry_after)
            time.sleep(retry_after)
        resp.raise_for_status()
        return resp.json()
