"""
OpenStreetMap Connector

Coverage: Global roads, transport, infrastructure, land use.
Portal: https://www.openstreetmap.org/

Live mode uses Overpass API queries to summarize infrastructure within a
bounding box or OSM area:
- roads_km: total length of ways tagged 'highway' (excluding footpath-only types)
- rail_km: total length of ways tagged 'railway'
- power_lines_km: total length of ways tagged power=line

Falls back to safe mocks when live is disabled.
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

from kafka import KafkaProducer
import json

from ...base import Ingestor
import requests
import math

logger = logging.getLogger(__name__)


class OpenStreetMapConnector(Ingestor):
    """OpenStreetMap (OSM) geospatial connector."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Overpass API for OSM extracts: https://overpass-api.de/api/interpreter
        # Example query (power lines in bbox):
        # [out:json][timeout:25];(way["power"="line"]({s},{w},{n},{e}););out ids qt;>
        self.overpass_url = config.get("overpass_url", "https://overpass-api.de/api/interpreter")
        self.region = config.get("region", "WORLD")
        self.live: bool = bool(config.get("live", False))
        # bbox: [south, west, north, east] or area_id: OSM area id (e.g., 36000...)
        self.bbox: Optional[List[float]] = config.get("bbox")
        self.area_id: Optional[int] = config.get("area_id")
        self.region_name: str = config.get("region_name", self.region)
        self.kafka_topic = config.get("kafka_topic", "market.fundamentals")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer: KafkaProducer | None = None

    def discover(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "streams": [
                {"name": "road_length", "market": "geospatial", "product": "roads_km", "unit": "km", "update_freq": "ad-hoc"},
                {"name": "rail_length", "market": "geospatial", "product": "rail_km", "unit": "km", "update_freq": "ad-hoc"},
                {"name": "power_lines_length", "market": "geospatial", "product": "power_lines_km", "unit": "km", "update_freq": "ad-hoc"},
            ],
            "endpoint_examples": {
                "overpass_power_lines": "POST {overpass} with data='[out:json];way[\"power\"=\"line\"](s,w,n,e);out;>'",
            },
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        if not self.live:
            now = datetime.utcnow().isoformat()
            yield {"timestamp": now, "entity": self.region_name, "variable": "roads_km", "value": 35_000_000.0, "unit": "km"}
            yield {"timestamp": now, "entity": self.region_name, "variable": "rail_km", "value": 1_150_000.0, "unit": "km"}
            yield {"timestamp": now, "entity": self.region_name, "variable": "power_lines_km", "value": 1_900_000.0, "unit": "km"}
            return

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

        # Roads: highway=* exclude footways/paths/steps
        try:
            if bbox_clause:
                q_body = (
                    f"way['highway']{bbox_clause};"
                    f"way['highway'~'footway|path|steps']{bbox_clause};out ids;"
                )
                # Fetch all highways then subtract excluded types by recomputing lengths with filter
                data = overpass(f"way['highway']{bbox_clause};")
                excl = overpass(f"way['highway'~'footway|path|steps']{bbox_clause};")
            else:
                data = overpass(f"{area_clause}way(area)['highway'];")
                excl = overpass(f"{area_clause}way(area)['highway'~'footway|path|steps'];")
            roads_km_all = self._sum_way_lengths_km(data)
            roads_km_excl = self._sum_way_lengths_km(excl)
            roads_km = max(0.0, roads_km_all - roads_km_excl)
        except Exception as e:
            logger.warning(f"Overpass roads failed: {e}")
            roads_km = 0.0

        # Rails
        try:
            if bbox_clause:
                data = overpass(f"way['railway']{bbox_clause};")
            else:
                data = overpass(f"{area_clause}way(area)['railway'];")
            rail_km = self._sum_way_lengths_km(data)
        except Exception as e:
            logger.warning(f"Overpass rail failed: {e}")
            rail_km = 0.0

        # Power lines
        try:
            if bbox_clause:
                data = overpass(f"way['power'='line']{bbox_clause};")
            else:
                data = overpass(f"{area_clause}way(area)['power'='line'];")
            power_lines_km = self._sum_way_lengths_km(data)
        except Exception as e:
            logger.warning(f"Overpass power lines failed: {e}")
            power_lines_km = 0.0

        ts = datetime.utcnow().isoformat()
        yield {"timestamp": ts, "entity": self.region_name, "variable": "roads_km", "value": float(roads_km), "unit": "km"}
        yield {"timestamp": ts, "entity": self.region_name, "variable": "rail_km", "value": float(rail_km), "unit": "km"}
        yield {"timestamp": ts, "entity": self.region_name, "variable": "power_lines_km", "value": float(power_lines_km), "unit": "km"}

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        ts = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        entity = raw.get("entity", "WORLD")
        variable = raw.get("variable", "roads_km")
        instrument = f"OSM.{entity}.{variable.upper()}"
        return {
            "event_time_utc": int(ts.timestamp() * 1000),
            "market": "geospatial",
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
        logger.info(f"Emitted {count} OSM events to {self.kafka_topic}")
        return count

    def checkpoint(self, state: Dict[str, Any]) -> None:
        self.checkpoint_state = state
        logger.debug(f"OSM checkpoint saved: {state}")

    # --- geometry helpers ---
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


if __name__ == "__main__":
    connector = OpenStreetMapConnector({"source_id": "openstreetmap"})
    connector.run()
