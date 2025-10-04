"""
Open Infrastructure Map Connector

Coverage: Global power lines, substations, pipelines (derived from OSM).
Portal: https://openinframap.org/

Live mode uses Overpass API queries to summarize infrastructure within a
bounding box or OSM area: total power line length (km), substation count,
and pipeline length (km). Falls back to safe mocks otherwise.

Data Flow
---------
Overpass/OSM → infra summaries (lines_km, substation_count, pipelines_km) → Kafka
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

from kafka import KafkaProducer
import json
import requests
import math

from ...base import Ingestor

logger = logging.getLogger(__name__)


class OpenInfrastructureMapConnector(Ingestor):
    """OpenInfraMap connector (power infra summaries)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "https://openinframap.org/")
        self.region = config.get("region", "WORLD")
        self.live: bool = bool(config.get("live", False))
        # Overpass API endpoint and region (bbox or area id)
        self.overpass_url: str = config.get("overpass_url", "https://overpass-api.de/api/interpreter")
        # bbox: [south, west, north, east]
        self.bbox: Optional[List[float]] = config.get("bbox")
        # area_id: OSM area id (e.g., 3600062421). If both provided, bbox wins.
        self.area_id: Optional[int] = config.get("area_id")
        self.region_name: str = config.get("region_name", self.region)
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "power_lines", "market": "infra", "product": "lines_km", "unit": "km", "update_freq": "ad-hoc"},
                {"name": "substations", "market": "infra", "product": "substation_count", "unit": "count", "update_freq": "ad-hoc"},
                {"name": "pipelines", "market": "infra", "product": "pipelines_km", "unit": "km", "update_freq": "ad-hoc"},
            ],
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        if not self.live:
            now = datetime.utcnow().isoformat()
            yield {"timestamp": now, "entity": self.region_name, "variable": "lines_km", "value": 1_250_000.0, "unit": "km"}
            yield {"timestamp": now, "entity": self.region_name, "variable": "substation_count", "value": 85_000.0, "unit": "count"}
            yield {"timestamp": now, "entity": self.region_name, "variable": "pipelines_km", "value": 320_000.0, "unit": "km"}
            return

        # Build Overpass QL for bbox or area
        bbox_clause = None
        area_clause = None
        if self.bbox and len(self.bbox) == 4:
            s, w, n, e = self.bbox
            bbox_clause = f"({s},{w},{n},{e})"
        elif self.area_id:
            area_clause = f"area({self.area_id});"

        def overpass(query_body: str) -> Dict[str, Any]:
            q = f"[out:json][timeout:60];{query_body}out body;>\nout skel qt;"
            resp = requests.post(self.overpass_url, data={"data": q}, timeout=120)
            resp.raise_for_status()
            return resp.json()

        # Power lines
        try:
            if bbox_clause:
                q_body = f"way['power'='line']{bbox_clause};"
            else:
                q_body = f"{area_clause}way(area)['power'='line'];"
            data = overpass(q_body)
            lines_km = self._sum_way_lengths_km(data)
        except Exception as e:
            logger.warning(f"Overpass power lines failed: {e}")
            lines_km = 0.0

        # Substations
        try:
            if bbox_clause:
                q_body = f"(node['power'='substation']{bbox_clause};way['power'='substation']{bbox_clause};);"
            else:
                q_body = f"{area_clause}(node(area)['power'='substation'];way(area)['power'='substation'];);"
            data = overpass(q_body)
            substation_count = self._count_elements(data)
        except Exception as e:
            logger.warning(f"Overpass substations failed: {e}")
            substation_count = 0

        # Pipelines
        try:
            if bbox_clause:
                q_body = f"way['man_made'='pipeline']{bbox_clause};"
            else:
                q_body = f"{area_clause}way(area)['man_made'='pipeline'];"
            data = overpass(q_body)
            pipelines_km = self._sum_way_lengths_km(data)
        except Exception as e:
            logger.warning(f"Overpass pipelines failed: {e}")
            pipelines_km = 0.0

        ts = datetime.utcnow().isoformat()
        yield {"timestamp": ts, "entity": self.region_name, "variable": "lines_km", "value": float(lines_km), "unit": "km"}
        yield {"timestamp": ts, "entity": self.region_name, "variable": "substation_count", "value": float(substation_count), "unit": "count"}
        yield {"timestamp": ts, "entity": self.region_name, "variable": "pipelines_km", "value": float(pipelines_km), "unit": "km"}

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", "WORLD")
        variable = raw.get("variable", "lines_km")
        instrument = f"OIM.{entity}.{variable.upper()}"
        return {
            "event_time_utc": int(ts.timestamp() * 1000),
            "market": "infra",
            "product": variable,
            "instrument_id": instrument,
            "location_code": instrument,
            "price_type": "observation",
            "value": float(raw.get("value", 0.0)),
            "volume": None,
            "currency": "USD",
            "unit": raw.get("unit", "unit"),
            "source": self.source_id,
            "seq": int(time.time() * 1000000),
        }

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
        logger.info(f"Emitted {count} OpenInfraMap events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"OpenInfraMap checkpoint saved: {state}")

    # --- Geodesy helpers ---
    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = phi2 - phi1
        dl = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    def _sum_way_lengths_km(self, overpass_json: Dict[str, Any]) -> float:
        nodes: Dict[int, Tuple[float, float]] = {}
        for el in overpass_json.get('elements', []):
            if el.get('type') == 'node':
                nodes[el['id']] = (el['lat'], el['lon'])
        total = 0.0
        for el in overpass_json.get('elements', []):
            if el.get('type') == 'way':
                coords: List[Tuple[float, float]] = []
                if 'geometry' in el:
                    coords = [(p['lat'], p['lon']) for p in el['geometry']]
                else:
                    for nid in el.get('nodes', []):
                        if nid in nodes:
                            coords.append(nodes[nid])
                for (lat1, lon1), (lat2, lon2) in zip(coords, coords[1:]):
                    total += self._haversine_km(lat1, lon1, lat2, lon2)
        return total

    @staticmethod
    def _count_elements(overpass_json: Dict[str, Any]) -> int:
        return sum(1 for el in overpass_json.get('elements', []) if el.get('type') in ('node', 'way'))


if __name__ == "__main__":
    connector = OpenInfrastructureMapConnector({"source_id": "open_inframap"})
    connector.run()
