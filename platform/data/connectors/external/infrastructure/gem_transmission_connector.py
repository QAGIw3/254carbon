"""
Global Energy Monitor Transmission Infrastructure Connector
-----------------------------------------------------------

Overview
--------
Fetches transmission infrastructure data from Global Energy Monitor (GEM),
including operational lines, interconnectors, substations, and projects.
In this scaffold, representative samples are used to illustrate the mapping
to canonical infrastructure events and network topology.

Data Flow
---------
GEM API (or sample) → parse lines/projects/network → canonical fundamentals → Kafka

Configuration
-------------
- `api_base`/`api_key`: GEM API settings (if integrating live).
- `regions`: Regions to include in discovery and fetch.
- `min_voltage_kv`: Filter for transmission voltage threshold.
- `include_projects`: Toggle project ingestion.
- `project_statuses`: Allowed statuses for project emissions.
- `kafka.topic`/`kafka.bootstrap_servers`.

Operational Notes
-----------------
- The current implementation uses representative data; replace fetch helpers
  with live API calls when credentials and endpoints are finalized.
- Voltage class classification aids aggregation/filters in downstream analytics.
- The connector emits line capacity/voltage/length, basic project metrics, and
  a topology skeleton for graph-based analyses.

Data source: https://globalenergymonitor.org/
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple
from enum import Enum

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import (
    InfrastructureConnector,
    TransmissionLine,
    GeoLocation,
    OperationalStatus,
    InfrastructureType,
)
from ....base import CommodityType

logger = logging.getLogger(__name__)


class VoltageLevel(Enum):
    """Standard transmission voltage levels."""
    HV = "HV"  # High Voltage: 100-230 kV
    EHV = "EHV"  # Extra High Voltage: 345-765 kV
    UHV = "UHV"  # Ultra High Voltage: >765 kV
    HVDC = "HVDC"  # High Voltage Direct Current


class ProjectStatus(Enum):
    """Infrastructure project status."""
    ANNOUNCED = "announced"
    PRE_CONSTRUCTION = "pre_construction"
    CONSTRUCTION = "construction"
    COMMISSIONED = "commissioned"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class TransmissionProject:
    """Transmission infrastructure project."""
    
    project_id: str
    name: str
    project_type: str  # line, substation, interconnector
    countries: List[str]
    status: ProjectStatus
    voltage_kv: float
    capacity_mw: Optional[float]
    length_km: Optional[float]
    line_type: str  # AC, DC
    developer: Optional[str]
    estimated_cost_million_usd: Optional[float]
    start_year: Optional[int]
    completion_year: Optional[int]
    coordinates: List[GeoLocation]  # Route waypoints
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NetworkNode:
    """Network node (substation or interconnection point)."""
    
    node_id: str
    name: str
    node_type: str  # substation, border_point, generator, load
    location: GeoLocation
    country: str
    voltage_levels: List[float]  # kV
    connected_lines: List[str]  # Line IDs
    metadata: Dict[str, Any] = field(default_factory=dict)


class GEMTransmissionConnector(InfrastructureConnector):
    """Connector for Global Energy Monitor transmission data."""
    
    DEFAULT_API_BASE = "https://api.globalenergymonitor.org/v1"
    
    # Voltage level classification
    VOLTAGE_CLASSES = {
        VoltageLevel.HV: (100, 230),
        VoltageLevel.EHV: (345, 765),
        VoltageLevel.UHV: (765, float('inf')),
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.POWER
        
        self.api_base = config.get("api_base", self.DEFAULT_API_BASE)
        self.api_key = config.get("api_key")
        self.regions = config.get("regions", ["global"])
        self.min_voltage_kv = config.get("min_voltage_kv", 100)
        self.include_projects = config.get("include_projects", True)
        self.project_statuses = config.get("project_statuses", ["construction", "commissioned"])
        
        kafka_cfg = config.get("kafka") or {}
        if not kafka_cfg:
            self.kafka_topic = "market.fundamentals"
            self.kafka_bootstrap_servers = config.get("kafka_bootstrap", "kafka:9092")
        
        self.session = requests.Session()
        self._lines: Dict[str, TransmissionLine] = {}
        self._projects: Dict[str, TransmissionProject] = {}
        self._nodes: Dict[str, NetworkNode] = {}
        self._network_topology: Dict[str, Set[str]] = {}  # node_id -> connected_node_ids
    
    def discover(self) -> Dict[str, Any]:
        """Discover available data streams and coverage for observability."""
        
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "infrastructure_type": "transmission",
            "coverage": self.regions,
            "streams": [
                {
                    "name": "transmission_lines",
                    "variables": [
                        "line_capacity_mw",
                        "line_voltage_kv",
                        "line_length_km",
                        "line_availability_pct",
                    ],
                    "frequency": "monthly",
                },
                {
                    "name": "transmission_projects",
                    "variables": [
                        "project_status",
                        "project_capacity_mw",
                        "project_progress_pct",
                    ],
                    "frequency": "quarterly",
                },
                {
                    "name": "interconnector_flows",
                    "variables": [
                        "flow_mw",
                        "utilization_pct",
                    ],
                    "frequency": "hourly",
                }
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Fetch or synthesize transmission infrastructure data events.

        In live mode, replace sample generators with GEM API requests for lines,
        projects, and network relationships, then map via the same helpers.
        """
        
        logger.info(
            "Fetching GEM transmission data for regions=%s min_voltage=%s kV",
            self.regions,
            self.min_voltage_kv,
        )
        
        # Fetch existing transmission lines
        yield from self._fetch_transmission_lines()
        
        # Fetch infrastructure projects if enabled
        if self.include_projects:
            yield from self._fetch_transmission_projects()
        
        # Fetch network topology
        yield from self._fetch_network_topology()
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw line/project/network dicts to canonical event payloads."""
        
        data_type = raw.get("data_type")
        
        if data_type == "transmission_line":
            return self._map_line_data(raw)
        elif data_type == "project":
            return self._map_project_data(raw)
        elif data_type == "network":
            return self._map_network_data(raw)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint state."""
        
        state["lines_processed"] = len(self._lines)
        state["projects_tracked"] = len(self._projects)
        state["nodes_mapped"] = len(self._nodes)
        super().checkpoint(state)
    
    # ------------------------------------------------------------------
    # Transmission line methods
    # ------------------------------------------------------------------
    
    def _fetch_transmission_lines(self) -> Iterator[Dict[str, Any]]:
        """Fetch operational transmission lines."""
        
        # In production, this would call the actual GEM API
        # For now, using representative sample data
        
        sample_lines = [
            {
                "line_id": "US_TX_CREZ_001",
                "name": "CREZ Transmission Line 1",
                "from_lat": 31.9686,
                "from_lon": -99.9018,
                "to_lat": 32.7767,
                "to_lon": -96.7970,
                "voltage_kv": 345.0,
                "capacity_mw": 3000.0,
                "length_km": 400.0,
                "line_type": "AC",
                "countries": ["US"],
                "operator": "ERCOT",
                "commissioned": "2013-12-01",
            },
            {
                "line_id": "EU_NORDLINK_001",
                "name": "NordLink HVDC",
                "from_lat": 58.0, 
                "from_lon": 7.0,
                "to_lat": 53.5,
                "to_lon": 9.0,
                "voltage_kv": 525.0,
                "capacity_mw": 1400.0,
                "length_km": 623.0,
                "line_type": "DC",
                "countries": ["NO", "DE"],
                "operator": "Statnett/TenneT",
                "commissioned": "2021-05-01",
            },
        ]
        
        for line_data in sample_lines:
            line = self._parse_transmission_line(line_data)
            if line:
                self._lines[line.asset_id] = line
                self.assets[line.asset_id] = line
                
                # Emit capacity event
                yield {
                    "data_type": "transmission_line",
                    "line": line,
                    "metric": "line_capacity_mw",
                    "value": line.capacity_mw,
                }
                
                # Emit voltage event
                yield {
                    "data_type": "transmission_line",
                    "line": line,
                    "metric": "line_voltage_kv",
                    "value": line.voltage_kv,
                }
                
                # Emit length event
                if line.length_km:
                    yield {
                        "data_type": "transmission_line",
                        "line": line,
                        "metric": "line_length_km",
                        "value": line.length_km,
                    }
    
    def _parse_transmission_line(self, data: Dict[str, Any]) -> Optional[TransmissionLine]:
        """Parse transmission line data."""
        
        try:
            from_location = GeoLocation(
                lat=float(data["from_lat"]),
                lon=float(data["from_lon"])
            )
            to_location = GeoLocation(
                lat=float(data["to_lat"]),
                lon=float(data["to_lon"])
            )
            
            # Use midpoint as primary location
            mid_lat = (from_location.lat + to_location.lat) / 2
            mid_lon = (from_location.lon + to_location.lon) / 2
            location = GeoLocation(mid_lat, mid_lon)
            
            # Determine primary country (first in list)
            countries = data.get("countries", [])
            country = countries[0] if countries else "INTL"
            
            commissioned_date = None
            if data.get("commissioned"):
                commissioned_date = datetime.fromisoformat(data["commissioned"]).date()
            
            line = TransmissionLine(
                asset_id=data["line_id"],
                name=data["name"],
                location=location,
                country=country,
                from_location=from_location,
                to_location=to_location,
                voltage_kv=float(data["voltage_kv"]),
                capacity_mw=float(data["capacity_mw"]),
                length_km=data.get("length_km"),
                line_type=data.get("line_type", "AC"),
                status=OperationalStatus.OPERATIONAL,
                commissioned_date=commissioned_date,
                operator=data.get("operator"),
                metadata={
                    "countries": countries,
                    "voltage_class": self._classify_voltage(float(data["voltage_kv"])),
                }
            )
            
            return line
            
        except Exception as e:
            logger.warning(f"Failed to parse transmission line: {e}")
            return None
    
    def _map_line_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map transmission line data to canonical schema."""
        
        line: TransmissionLine = raw["line"]
        metric = raw["metric"]
        value = raw["value"]
        
        return self.create_infrastructure_event(
            asset=line,
            metric=metric,
            value=value,
            unit=self._get_unit_for_metric(metric),
            event_time=datetime.now(timezone.utc),
            metadata={
                "line_type": line.line_type,
                "voltage_class": line.metadata.get("voltage_class"),
                "countries": line.metadata.get("countries", []),
                "from_location": line.from_location.to_dict(),
                "to_location": line.to_location.to_dict(),
            }
        )
    
    # ------------------------------------------------------------------
    # Project tracking methods
    # ------------------------------------------------------------------
    
    def _fetch_transmission_projects(self) -> Iterator[Dict[str, Any]]:
        """Fetch transmission infrastructure projects."""
        
        # Sample project data
        sample_projects = [
            {
                "project_id": "US_GRAIN_BELT_EXPRESS",
                "name": "Grain Belt Express",
                "project_type": "line",
                "countries": ["US"],
                "status": "construction",
                "voltage_kv": 600.0,
                "capacity_mw": 4000.0,
                "length_km": 1240.0,
                "line_type": "DC",
                "developer": "Invenergy",
                "estimated_cost_million_usd": 2500.0,
                "start_year": 2023,
                "completion_year": 2027,
                "waypoints": [
                    {"lat": 38.5, "lon": -99.0},
                    {"lat": 38.8, "lon": -94.5},
                    {"lat": 38.6, "lon": -90.2},
                ],
            },
            {
                "project_id": "EU_CELTIC_INTERCONNECTOR",
                "name": "Celtic Interconnector",
                "project_type": "interconnector",
                "countries": ["IE", "FR"],
                "status": "construction",
                "voltage_kv": 320.0,
                "capacity_mw": 700.0,
                "length_km": 575.0,
                "line_type": "DC",
                "developer": "EirGrid/RTE",
                "estimated_cost_million_usd": 1000.0,
                "start_year": 2022,
                "completion_year": 2026,
                "waypoints": [
                    {"lat": 51.8, "lon": -8.0},
                    {"lat": 48.4, "lon": -4.5},
                ],
            },
        ]
        
        for proj_data in sample_projects:
            project = self._parse_project(proj_data)
            if project and project.status.value in self.project_statuses:
                self._projects[project.project_id] = project
                
                # Emit project status
                status_value = float(list(ProjectStatus).index(project.status))
                yield {
                    "data_type": "project",
                    "project": project,
                    "metric": "project_status_code",
                    "value": status_value,
                }
                
                # Emit project capacity
                if project.capacity_mw:
                    yield {
                        "data_type": "project",
                        "project": project,
                        "metric": "project_capacity_mw",
                        "value": project.capacity_mw,
                    }
                
                # Emit project progress
                progress = self._calculate_project_progress(project)
                yield {
                    "data_type": "project",
                    "project": project,
                    "metric": "project_progress_pct",
                    "value": progress,
                }
    
    def _parse_project(self, data: Dict[str, Any]) -> Optional[TransmissionProject]:
        """Parse transmission project data."""
        
        try:
            waypoints = []
            for wp in data.get("waypoints", []):
                waypoints.append(GeoLocation(wp["lat"], wp["lon"]))
            
            status_map = {
                "announced": ProjectStatus.ANNOUNCED,
                "pre_construction": ProjectStatus.PRE_CONSTRUCTION,
                "construction": ProjectStatus.CONSTRUCTION,
                "commissioned": ProjectStatus.COMMISSIONED,
                "suspended": ProjectStatus.SUSPENDED,
                "cancelled": ProjectStatus.CANCELLED,
            }
            status = status_map.get(data["status"], ProjectStatus.ANNOUNCED)
            
            return TransmissionProject(
                project_id=data["project_id"],
                name=data["name"],
                project_type=data["project_type"],
                countries=data["countries"],
                status=status,
                voltage_kv=float(data["voltage_kv"]),
                capacity_mw=data.get("capacity_mw"),
                length_km=data.get("length_km"),
                line_type=data.get("line_type", "AC"),
                developer=data.get("developer"),
                estimated_cost_million_usd=data.get("estimated_cost_million_usd"),
                start_year=data.get("start_year"),
                completion_year=data.get("completion_year"),
                coordinates=waypoints,
                metadata=data,
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse project: {e}")
            return None
    
    def _calculate_project_progress(self, project: TransmissionProject) -> float:
        """Calculate project progress percentage."""
        
        if project.status == ProjectStatus.COMMISSIONED:
            return 100.0
        elif project.status == ProjectStatus.CANCELLED:
            return 0.0
        elif project.status in [ProjectStatus.CONSTRUCTION, ProjectStatus.PRE_CONSTRUCTION]:
            if project.start_year and project.completion_year:
                current_year = datetime.now().year
                if current_year < project.start_year:
                    return 0.0
                elif current_year >= project.completion_year:
                    return 95.0  # Nearly complete
                else:
                    total_years = project.completion_year - project.start_year
                    elapsed_years = current_year - project.start_year
                    return min(elapsed_years / total_years * 100, 95.0)
            else:
                # Default progress by status
                return 50.0 if project.status == ProjectStatus.CONSTRUCTION else 25.0
        else:
            return 10.0  # Announced/planning stage
    
    def _map_project_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map project data to canonical schema."""
        
        project: TransmissionProject = raw["project"]
        metric = raw["metric"]
        value = raw["value"]
        
        # Use midpoint of route as location
        if project.coordinates:
            avg_lat = sum(loc.lat for loc in project.coordinates) / len(project.coordinates)
            avg_lon = sum(loc.lon for loc in project.coordinates) / len(project.coordinates)
            location = GeoLocation(avg_lat, avg_lon)
        else:
            location = GeoLocation(0, 0)  # Unknown location
        
        return {
            "event_time_utc": int(datetime.now(timezone.utc).timestamp() * 1000),
            "market": "infra",
            "product": metric,
            "instrument_id": f"TRANSMISSION.PROJECT.{project.project_id}",
            "location_code": "_".join(project.countries),
            "price_type": "observation",
            "value": value,
            "unit": self._get_unit_for_metric(metric),
            "source": self.source_id,
            "commodity_type": self.commodity_type.value,
            "metadata": json.dumps({
                "project_name": project.name,
                "project_type": project.project_type,
                "status": project.status.value,
                "voltage_kv": project.voltage_kv,
                "line_type": project.line_type,
                "developer": project.developer,
                "countries": project.countries,
                "coordinates": location.to_dict(),
                "start_year": project.start_year,
                "completion_year": project.completion_year,
                "estimated_cost_million_usd": project.estimated_cost_million_usd,
            })
        }
    
    # ------------------------------------------------------------------
    # Network topology methods
    # ------------------------------------------------------------------
    
    def _fetch_network_topology(self) -> Iterator[Dict[str, Any]]:
        """Fetch network topology data."""
        
        # Sample network nodes
        sample_nodes = [
            {
                "node_id": "US_TX_HOUSTON_SUB",
                "name": "Houston Substation",
                "node_type": "substation",
                "lat": 29.7604,
                "lon": -95.3698,
                "country": "US",
                "voltage_levels": [345, 138, 69],
                "connected_lines": ["US_TX_CREZ_001"],
            },
            {
                "node_id": "NO_TONSTAD",
                "name": "Tonstad",
                "node_type": "substation",
                "lat": 58.6667,
                "lon": 6.7167,
                "country": "NO",
                "voltage_levels": [525, 420, 300],
                "connected_lines": ["EU_NORDLINK_001"],
            },
        ]
        
        for node_data in sample_nodes:
            node = self._parse_network_node(node_data)
            if node:
                self._nodes[node.node_id] = node
                
                # Build topology
                for line_id in node.connected_lines:
                    if node.node_id not in self._network_topology:
                        self._network_topology[node.node_id] = set()
                    self._network_topology[node.node_id].add(line_id)
                
                # Emit node events
                yield {
                    "data_type": "network",
                    "node": node,
                    "metric": "node_voltage_levels",
                    "value": float(len(node.voltage_levels)),
                }
    
    def _parse_network_node(self, data: Dict[str, Any]) -> Optional[NetworkNode]:
        """Parse network node data."""
        
        try:
            location = GeoLocation(
                lat=float(data["lat"]),
                lon=float(data["lon"])
            )
            
            return NetworkNode(
                node_id=data["node_id"],
                name=data["name"],
                node_type=data["node_type"],
                location=location,
                country=data["country"],
                voltage_levels=data["voltage_levels"],
                connected_lines=data.get("connected_lines", []),
                metadata=data,
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse network node: {e}")
            return None
    
    def _map_network_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map network topology data to canonical schema."""
        
        node: NetworkNode = raw["node"]
        metric = raw["metric"]
        value = raw["value"]
        
        return {
            "event_time_utc": int(datetime.now(timezone.utc).timestamp() * 1000),
            "market": "infra",
            "product": metric,
            "instrument_id": f"NETWORK.{node.country}.{node.node_id}",
            "location_code": node.country,
            "price_type": "observation",
            "value": value,
            "unit": self._get_unit_for_metric(metric),
            "source": self.source_id,
            "commodity_type": self.commodity_type.value,
            "metadata": json.dumps({
                "node_name": node.name,
                "node_type": node.node_type,
                "coordinates": node.location.to_dict(),
                "voltage_levels": node.voltage_levels,
                "connected_lines": node.connected_lines,
            })
        }
    
    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    
    def _classify_voltage(self, voltage_kv: float) -> str:
        """Classify voltage level."""
        
        for level, (min_v, max_v) in self.VOLTAGE_CLASSES.items():
            if min_v <= voltage_kv < max_v:
                return level.value
        
        # Check for HVDC
        if voltage_kv >= 100:
            return VoltageLevel.HVDC.value
        
        return "MV"  # Medium voltage
    
    def _get_unit_for_metric(self, metric: str) -> str:
        """Get unit for a specific metric."""
        
        unit_map = {
            "line_capacity_mw": "MW",
            "line_voltage_kv": "kV",
            "line_length_km": "km",
            "line_availability_pct": "pct",
            "project_capacity_mw": "MW",
            "project_progress_pct": "pct",
            "project_status_code": "code",
            "flow_mw": "MW",
            "utilization_pct": "pct",
            "node_voltage_levels": "count",
        }
        
        return unit_map.get(metric, "unit")
