"""
ENTSO-E Transparency Platform Connector

Coverage: European grid fundamentals — generation, load, transmission, balancing.
Portal: https://transparency.entsoe.eu/

This scaffold emits representative observations and is structured to accept
real API credentials using the official ENTSO-E API when configured.

Data Flow
---------
ENTSO-E Web API → XML/JSON time series → canonical fundamentals → Kafka
"""
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

from kafka import KafkaProducer
import json
import requests
import xml.etree.ElementTree as ET

from ...base import Ingestor

logger = logging.getLogger(__name__)


class ENTSOETransparencyConnector(Ingestor):
    """ENTSO-E transparency platform connector."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # ENTSO-E Web API base (requires token): https://web-api.tp.entsoe.eu/
        # Generic endpoint: GET /api?securityToken=...&documentType=A44&in_Domain=10Y...&out_Domain=...&periodStart=YYYYMMDDHHMM&periodEnd=...
        self.api_base = config.get("api_base", "https://web-api.tp.entsoe.eu")
        self.api_token = config.get("api_token")
        self.area = config.get("area", "10Y1001A1001A83F")  # default: DE (example EIC)
        self.out_area: Optional[str] = config.get("out_area")  # for cross-border flows
        self.live: bool = bool(config.get("live", False))
        # Modes to fetch in live mode: any of load, generation, flows, da_price
        self.modes: List[str] = config.get("modes", ["load", "generation"])  # sensible default
        # Period bounds (UTC) in format YYYYMMDDHHMM; default: last 24h
        self.period_start: Optional[str] = config.get("period_start")
        self.period_end: Optional[str] = config.get("period_end")
        # Optional friendly name mapping for EIC codes
        self.area_names: Dict[str, str] = config.get("area_names", {}) or {}
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "load", "market": "infra", "product": "load", "unit": "MW", "update_freq": "hourly"},
                {"name": "generation", "market": "infra", "product": "generation", "unit": "MW", "update_freq": "hourly"},
                {"name": "cross_border_flows", "market": "infra", "product": "transmission", "unit": "MW", "update_freq": "hourly"},
                {"name": "balancing_prices", "market": "infra", "product": "balancing", "unit": "EUR/MWh", "update_freq": "15min"},
            ],
            "endpoint_examples": {
                "day_ahead_prices": (
                    "GET {base}/api?securityToken=...&documentType=A44&in_Domain=10YDE-..."
                    "&periodStart=202501010000&periodEnd=202501022300"
                ),
                "load": (
                    "GET {base}/api?securityToken=...&documentType=A65&processType=A16"
                    "&outBiddingZone_Domain=10YDE-...&periodStart=...&periodEnd=..."
                ),
                "generation": (
                    "GET {base}/api?securityToken=...&documentType=A75&in_Domain=10YDE-..."
                    "&periodStart=...&periodEnd=..."
                ),
                "cross_border_flows": (
                    "GET {base}/api?securityToken=...&documentType=A11&in_Domain=10Y...&out_Domain=10Y...&periodStart=...&periodEnd=..."
                ),
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        if not self.live or not self.api_token:
            now = datetime.utcnow().isoformat()
            yield {"timestamp": now, "entity": self.area, "variable": "load", "value": 342_000.0, "unit": "MW"}
            yield {"timestamp": now, "entity": self.area, "variable": "generation", "value": 338_500.0, "unit": "MW"}
            if self.out_area:
                yield {"timestamp": now, "entity": f"{self.area}->{self.out_area}", "variable": "transmission", "value": 3_200.0, "unit": "MW"}
            yield {"timestamp": now, "entity": self.area, "variable": "da_price_eur_mwh", "value": 85.3, "unit": "EUR/MWh"}
            return

        start, end = self._compute_period_bounds()

        for mode in self.modes:
            mode = mode.lower()
            try:
                if mode == "load":
                    yield from self._fetch_timeseries(
                        {
                            "documentType": "A65",
                            "processType": "A16",  # Realised
                            "outBiddingZone_Domain": self.area,
                            "periodStart": start,
                            "periodEnd": end,
                        },
                        variable="load",
                        unit="MW",
                        entity=self.area,
                    )
                elif mode == "generation":
                    # Sum over PSR types by aggregating points across series
                    yield from self._fetch_timeseries(
                        {
                            "documentType": "A75",
                            "in_Domain": self.area,
                            "periodStart": start,
                            "periodEnd": end,
                        },
                        variable="generation",
                        unit="MW",
                        entity=self.area,
                        aggregate_series=True,
                    )
                elif mode == "flows" and self.out_area:
                    yield from self._fetch_timeseries(
                        {
                            "documentType": "A11",
                            "in_Domain": self.area,
                            "out_Domain": self.out_area,
                            "periodStart": start,
                            "periodEnd": end,
                        },
                        variable="transmission",
                        unit="MW",
                        entity=f"{self.area}->{self.out_area}",
                    )
                elif mode == "da_price":
                    yield from self._fetch_timeseries(
                        {
                            "documentType": "A44",
                            "in_Domain": self.area,
                            "periodStart": start,
                            "periodEnd": end,
                        },
                        variable="da_price_eur_mwh",
                        unit="EUR/MWh",
                        entity=self.area,
                    )
            except Exception as e:
                logger.warning(f"ENTSO-E fetch failed for mode={mode}: {e}")

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = self._friendly_label(raw.get("entity", "EU"))
        variable = raw.get("variable", "load")
        instrument = f"ENTSOE.{entity}.{variable.upper()}"
        payload = {
            "event_time_utc": int(ts.timestamp() * 1000),
            "market": "infra",
            "product": variable,
            "instrument_id": instrument,
            "location_code": instrument,
            "price_type": "observation",
            "value": float(raw.get("value", 0.0)),
            "volume": None,
            "currency": "EUR" if variable.endswith("eur_mwh") else "USD",
            "unit": raw.get("unit", "unit"),
            "source": self.source_id,
            "seq": int(time.time() * 1000000),
        }
        return payload

    def emit(self, events: Iterator[Dict[str, Any]]) -> int:
        if self.producer is None:
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_bootstrap,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        count = 0
        for event in events:
            try:
                self.producer.send(self.kafka_topic, value=event)
                count += 1
            except Exception as e:
                logger.error(f"Kafka send error: {e}")
        if self.producer is not None:
            self.producer.flush()
        logger.info(f"Emitted {count} ENTSO-E events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"ENTSO-E checkpoint saved: {state}")

    # --- Helpers ---
    def _compute_period_bounds(self) -> Tuple[str, str]:
        if self.period_start and self.period_end:
            return self.period_start, self.period_end
        now = datetime.now(timezone.utc)
        start = (now - timedelta(hours=24)).replace(minute=0, second=0, microsecond=0)
        end = now.replace(minute=0, second=0, microsecond=0)
        fmt = "%Y%m%d%H%M"
        return start.strftime(fmt), end.strftime(fmt)

    def _build_url(self, params: Dict[str, Any]) -> str:
        endpoint = f"{self.api_base.rstrip('/')}/api"
        q = {"securityToken": self.api_token}
        q.update(params)
        # requests handles params, we just return base
        return endpoint, q

    def _fetch_timeseries(
        self,
        params: Dict[str, Any],
        variable: str,
        unit: str,
        entity: str,
        aggregate_series: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        url, q = self._build_url(params)
        resp = requests.get(url, params=q, timeout=60)
        resp.raise_for_status()
        xml = resp.text
        root = ET.fromstring(xml)

        # Parse timeseries: sum across series if requested
        # Collect per timestamp values
        values_by_ts: Dict[str, float] = {}

        for ts in root.iter():
            # Identify Timeseries nodes by localname 'TimeSeries'
            if ts.tag.endswith('TimeSeries'):
                # Resolution and start come from Period/timeInterval and resolution
                for period in ts:
                    if not period.tag.endswith('Period'):
                        continue
                    time_interval = None
                    resolution = None
                    for ch in period:
                        if ch.tag.endswith('timeInterval'):
                            # Expect two children: start, end
                            parts = [c.text for c in ch]
                            if parts:
                                time_interval = parts[0]
                        elif ch.tag.endswith('resolution'):
                            resolution = ch.text
                        elif ch.tag.endswith('Point'):
                            position = None
                            quantity = None
                            for pch in ch:
                                if pch.tag.endswith('position'):
                                    position = int(pch.text)
                                elif pch.tag.endswith('quantity') or pch.tag.endswith('price.amount'):
                                    try:
                                        quantity = float(pch.text)
                                    except Exception:
                                        quantity = None
                            if position is None or quantity is None or not time_interval or not resolution:
                                continue
                            ts_iso = self._calc_point_time(time_interval, resolution, position)
                            if aggregate_series:
                                values_by_ts[ts_iso] = values_by_ts.get(ts_iso, 0.0) + float(quantity)
                            else:
                                yield {"timestamp": ts_iso, "entity": entity, "variable": variable, "value": float(quantity), "unit": unit}

        if aggregate_series:
            for ts_iso, val in sorted(values_by_ts.items()):
                yield {"timestamp": ts_iso, "entity": entity, "variable": variable, "value": val, "unit": unit}

    def _calc_point_time(self, start_iso: str, resolution: str, position: int) -> str:
        # start_iso like 2025-01-01T00:00Z; resolution like PT15M/PT60M
        try:
            start = datetime.fromisoformat(start_iso.replace('Z', '+00:00'))
        except Exception:
            # Fallback: treat as plain UTC
            start = datetime.strptime(start_iso[:16], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
        minutes = 60
        if resolution and resolution.startswith('PT') and resolution.endswith('M'):
            try:
                minutes = int(resolution[2:-1])
            except Exception:
                minutes = 60
        delta = timedelta(minutes=minutes * (position - 1))
        t = start + delta
        return t.replace(tzinfo=timezone.utc).isoformat().replace('+00:00', 'Z')

    def _friendly(self, code: str) -> str:
        return self.area_names.get(code, code)

    def _friendly_label(self, label: str) -> str:
        # Handle flow labels in form "A->B"
        if '->' in label:
            a, b = label.split('->', 1)
            return f"{self._friendly(a)}->{self._friendly(b)}"
        return self._friendly(label)


if __name__ == "__main__":
    connector = ENTSOETransparencyConnector({"source_id": "entsoe_transparency"})
    connector.run()
