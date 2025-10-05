"""
NREL RE Data Explorer Connector
--------------------------------

Overview
--------
Fetches renewable energy resource data and project information from the NREL
RE Data Explorer API. Provides resource assessments (solar, wind) and a catalog
of projects for selected regions.

Data Flow
---------
RE Explorer API (or sample) → resource grid/projects → canonical fundamentals → Kafka

Configuration
-------------
- `api_key`: Required for live API usage.
- `regions`: List of region labels for discovery/coverage.
- `resource_types`: e.g., solar_ghi, solar_dni, wind_speed_100m.
- `include_projects`: Toggle project ingestion.
- `grid_bounds`: Optional {lat_min, lat_max, lon_min, lon_max} to scope grids.
- `lookback_days`: Used for recency checks in some resources.
- `kafka.topic`/`kafka.bootstrap_servers`.

Operational Notes
-----------------
- This scaffold emits representative data; replace fetch helpers with live API
  requests and map responses via the provided mappers.
- A simple in-memory cache avoids redundant API calls within a run.

API reference: https://re-explorer.org/
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from typing import Any, Dict, Iterator, List, Optional, Tuple
from enum import Enum

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import (
    InfrastructureConnector,
    PowerPlant,
    RenewableResource,
    GeoLocation,
    FuelType,
    OperationalStatus,
)
from ....base import CommodityType

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of renewable resources available."""
    SOLAR_GHI = "solar_ghi"  # Global Horizontal Irradiance
    SOLAR_DNI = "solar_dni"  # Direct Normal Irradiance
    WIND_SPEED_100M = "wind_speed_100m"
    WIND_SPEED_50M = "wind_speed_50m"
    HYDRO_POTENTIAL = "hydro_potential"


@dataclass(slots=True)
class REProject:
    """Renewable energy project from RE Explorer."""
    
    project_id: str
    name: str
    technology: FuelType
    capacity_mw: float
    location: GeoLocation
    country: str
    status: OperationalStatus
    commissioned_date: Optional[date]
    developer: Optional[str]
    capacity_factor: Optional[float]
    annual_generation_gwh: Optional[float]
    metadata: Dict[str, Any]


@dataclass(slots=True)
class ResourceAssessment:
    """Resource assessment for a specific location."""
    
    location: GeoLocation
    resource_type: ResourceType
    annual_average: float
    monthly_averages: List[float]
    unit: str
    data_year: int
    resolution_km: float
    metadata: Dict[str, Any]


class REExplorerConnector(InfrastructureConnector):
    """Connector for NREL RE Data Explorer."""
    
    DEFAULT_API_BASE = "https://developer.nrel.gov/api/reexplorer/v1"
    DEFAULT_SOLAR_API = "https://developer.nrel.gov/api/solar/solar_resource/v1"
    DEFAULT_WIND_API = "https://developer.nrel.gov/api/windtoolkit/v2/wind"
    
    # Grid resolution for resource queries (degrees)
    GRID_RESOLUTION = 0.5
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.RENEWABLES
        
        self.api_key = config.get("api_key")
        self.regions = config.get("regions", ["global"])
        self.resource_types = config.get("resource_types", ["solar_ghi", "wind_speed_100m"])
        self.include_projects = config.get("include_projects", True)
        self.grid_bounds = config.get("grid_bounds")  # Optional: {"lat_min": -90, "lat_max": 90, ...}
        self.lookback_days = config.get("lookback_days", 30)
        
        kafka_cfg = config.get("kafka") or {}
        if not kafka_cfg:
            self.kafka_topic = "market.fundamentals"
            self.kafka_bootstrap_servers = config.get("kafka_bootstrap", "kafka:9092")
        
        self.session = requests.Session()
        self._validate_config()
        
        # Cache for resource data to avoid repeated API calls
        self._resource_cache: Dict[str, ResourceAssessment] = {}
        self._projects_cache: List[REProject] = []
    
    def discover(self) -> Dict[str, Any]:
        """Discover available data streams."""
        
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "regions": self.regions,
            "streams": [
                {
                    "name": "renewable_resources",
                    "variables": [
                        "solar_ghi_annual_kwh_m2",
                        "solar_dni_annual_kwh_m2",
                        "wind_speed_100m_avg_ms",
                        "wind_capacity_factor_pct",
                    ],
                    "frequency": "static",
                },
                {
                    "name": "renewable_projects",
                    "variables": [
                        "project_capacity_mw",
                        "project_status",
                        "capacity_factor_pct",
                    ],
                    "frequency": "monthly",
                }
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull renewable resource and project data."""
        
        logger.info(
            "Fetching RE Explorer data for regions=%s resource_types=%s",
            self.regions,
            self.resource_types,
        )
        
        # Fetch renewable projects if enabled
        if self.include_projects:
            yield from self._fetch_projects()
        
        # Fetch resource assessments for grid points
        yield from self._fetch_resource_grid()
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw data to canonical schema."""
        
        data_type = raw.get("data_type")
        
        if data_type == "project":
            return self._map_project_data(raw)
        elif data_type == "resource":
            return self._map_resource_data(raw)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint state."""
        super().checkpoint(state)
    
    # ------------------------------------------------------------------
    # Project data methods
    # ------------------------------------------------------------------
    
    def _fetch_projects(self) -> Iterator[Dict[str, Any]]:
        """Fetch renewable energy projects."""
        
        # In production, this would call the actual RE Explorer API
        # For now, using sample data structure
        sample_projects = [
            {
                "project_id": "US_SOLAR_001",
                "name": "Mojave Solar Park",
                "technology": "solar",
                "capacity_mw": 280.0,
                "lat": 35.0117,
                "lon": -117.5584,
                "country": "US",
                "status": "operational",
                "commissioned_date": "2014-12-01",
                "developer": "Abengoa Solar",
                "capacity_factor": 0.26,
            },
            {
                "project_id": "US_WIND_001", 
                "name": "Alta Wind Energy Center",
                "technology": "wind",
                "capacity_mw": 1548.0,
                "lat": 35.0825,
                "lon": -118.3265,
                "country": "US",
                "status": "operational",
                "commissioned_date": "2011-04-01",
                "developer": "Terra-Gen Power",
                "capacity_factor": 0.32,
            },
        ]
        
        for proj_data in sample_projects:
            project = self._parse_project(proj_data)
            if project:
                self._projects_cache.append(project)
                
                # Emit project capacity event
                yield {
                    "data_type": "project",
                    "project": project,
                    "metric": "project_capacity_mw",
                    "value": project.capacity_mw,
                }
                
                # Emit capacity factor if available
                if project.capacity_factor:
                    yield {
                        "data_type": "project",
                        "project": project,
                        "metric": "capacity_factor_pct",
                        "value": project.capacity_factor * 100,
                    }
    
    def _parse_project(self, data: Dict[str, Any]) -> Optional[REProject]:
        """Parse project data into REProject object."""
        
        try:
            location = GeoLocation(
                lat=float(data["lat"]),
                lon=float(data["lon"])
            )
            
            tech_map = {
                "solar": FuelType.SOLAR,
                "wind": FuelType.WIND,
                "hydro": FuelType.HYDRO,
            }
            technology = tech_map.get(data["technology"], FuelType.OTHER)
            
            status_map = {
                "operational": OperationalStatus.OPERATIONAL,
                "construction": OperationalStatus.CONSTRUCTION,
                "planned": OperationalStatus.PLANNED,
            }
            status = status_map.get(data["status"], OperationalStatus.UNKNOWN)
            
            commissioned = None
            if data.get("commissioned_date"):
                commissioned = datetime.fromisoformat(data["commissioned_date"]).date()
            
            # Create power plant asset
            plant = PowerPlant(
                asset_id=data["project_id"],
                name=data["name"],
                location=location,
                country=data["country"],
                status=status,
                commissioned_date=commissioned,
                capacity_mw=float(data["capacity_mw"]),
                primary_fuel=technology,
                capacity_factor=data.get("capacity_factor"),
                developer=data.get("developer"),
            )
            
            # Store in assets registry
            self.assets[plant.asset_id] = plant
            
            return REProject(
                project_id=data["project_id"],
                name=data["name"],
                technology=technology,
                capacity_mw=float(data["capacity_mw"]),
                location=location,
                country=data["country"],
                status=status,
                commissioned_date=commissioned,
                developer=data.get("developer"),
                capacity_factor=data.get("capacity_factor"),
                annual_generation_gwh=data.get("annual_generation_gwh"),
                metadata=data,
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse project: {e}")
            return None
    
    def _map_project_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map project data to canonical schema."""
        
        project: REProject = raw["project"]
        metric = raw["metric"]
        value = raw["value"]
        
        # Get the power plant asset
        plant = self.assets.get(project.project_id)
        
        if plant:
            return self.create_infrastructure_event(
                asset=plant,
                metric=metric,
                value=value,
                unit="MW" if "capacity" in metric else "pct",
                event_time=datetime.now(timezone.utc),
                metadata={
                    "technology": project.technology.value,
                    "developer": project.developer,
                    "annual_generation_gwh": project.annual_generation_gwh,
                }
            )
        else:
            # Fallback if asset not found
            return {
                "event_time_utc": int(datetime.now(timezone.utc).timestamp() * 1000),
                "market": "infra",
                "product": metric,
                "instrument_id": f"RENEWABLE.{project.country}.{project.project_id}",
                "location_code": project.country,
                "price_type": "observation",
                "value": value,
                "unit": "MW" if "capacity" in metric else "pct",
                "source": self.source_id,
                "commodity_type": self.commodity_type.value,
                "metadata": json.dumps({
                    "project_name": project.name,
                    "technology": project.technology.value,
                    "coordinates": project.location.to_dict(),
                })
            }
    
    # ------------------------------------------------------------------
    # Resource assessment methods
    # ------------------------------------------------------------------
    
    def _fetch_resource_grid(self) -> Iterator[Dict[str, Any]]:
        """Fetch resource assessments for grid points."""
        
        grid_points = self._generate_grid_points()
        
        for lat, lon in grid_points:
            location = GeoLocation(lat, lon)
            
            for resource_type in self.resource_types:
                assessment = self._fetch_resource_assessment(location, resource_type)
                
                if assessment:
                    self._resource_cache[f"{lat},{lon},{resource_type}"] = assessment
                    
                    yield {
                        "data_type": "resource",
                        "assessment": assessment,
                        "metric": f"{resource_type}_annual",
                        "value": assessment.annual_average,
                    }
    
    def _generate_grid_points(self) -> List[Tuple[float, float]]:
        """Generate grid points for resource assessment."""
        
        if self.grid_bounds:
            lat_min = self.grid_bounds["lat_min"]
            lat_max = self.grid_bounds["lat_max"]
            lon_min = self.grid_bounds["lon_min"]
            lon_max = self.grid_bounds["lon_max"]
        else:
            # Default to global grid (sampled)
            lat_min, lat_max = -60, 60
            lon_min, lon_max = -180, 180
        
        grid_points = []
        lat = lat_min
        
        while lat <= lat_max:
            lon = lon_min
            while lon <= lon_max:
                grid_points.append((lat, lon))
                lon += self.GRID_RESOLUTION
            lat += self.GRID_RESOLUTION
        
        logger.info(f"Generated {len(grid_points)} grid points for resource assessment")
        return grid_points
    
    def _fetch_resource_assessment(
        self,
        location: GeoLocation,
        resource_type: str
    ) -> Optional[ResourceAssessment]:
        """Fetch resource assessment for a specific location."""
        
        # Check cache first
        cache_key = f"{location.lat},{location.lon},{resource_type}"
        if cache_key in self._resource_cache:
            return self._resource_cache[cache_key]
        
        try:
            if resource_type.startswith("solar"):
                return self._fetch_solar_resource(location, resource_type)
            elif resource_type.startswith("wind"):
                return self._fetch_wind_resource(location, resource_type)
            else:
                logger.warning(f"Unsupported resource type: {resource_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch resource assessment: {e}")
            return None
    
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(
            (requests.HTTPError, requests.ConnectionError, requests.Timeout)
        ),
    )
    def _fetch_solar_resource(
        self,
        location: GeoLocation,
        resource_type: str
    ) -> Optional[ResourceAssessment]:
        """Fetch solar resource data from NREL Solar API."""
        
        # For demonstration, return sample data
        # In production, this would call the actual NREL API
        
        if resource_type == "solar_ghi":
            annual_avg = 1825.5  # kWh/m2/year (typical for mid-latitudes)
            unit = "kWh/m2/year"
        else:  # solar_dni
            annual_avg = 2155.2  # kWh/m2/year
            unit = "kWh/m2/year"
        
        # Generate realistic monthly averages
        monthly = [
            annual_avg * 0.06,  # Jan
            annual_avg * 0.07,  # Feb
            annual_avg * 0.09,  # Mar
            annual_avg * 0.10,  # Apr
            annual_avg * 0.11,  # May
            annual_avg * 0.12,  # Jun
            annual_avg * 0.11,  # Jul
            annual_avg * 0.10,  # Aug
            annual_avg * 0.09,  # Sep
            annual_avg * 0.08,  # Oct
            annual_avg * 0.06,  # Nov
            annual_avg * 0.05,  # Dec
        ]
        
        return ResourceAssessment(
            location=location,
            resource_type=ResourceType(resource_type),
            annual_average=annual_avg,
            monthly_averages=monthly,
            unit=unit,
            data_year=2023,
            resolution_km=self.GRID_RESOLUTION * 111,  # Convert degrees to km
            metadata={
                "source": "NREL",
                "dataset": "NSRDB",
            }
        )
    
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(
            (requests.HTTPError, requests.ConnectionError, requests.Timeout)
        ),
    )
    def _fetch_wind_resource(
        self,
        location: GeoLocation,
        resource_type: str
    ) -> Optional[ResourceAssessment]:
        """Fetch wind resource data from NREL Wind Toolkit."""
        
        # For demonstration, return sample data
        # In production, this would call the actual NREL API
        
        if resource_type == "wind_speed_100m":
            annual_avg = 7.5  # m/s (typical for good wind sites)
            unit = "m/s"
        else:  # wind_speed_50m
            annual_avg = 6.8  # m/s
            unit = "m/s"
        
        # Generate realistic monthly averages (higher in winter)
        monthly = [
            annual_avg * 1.15,  # Jan
            annual_avg * 1.12,  # Feb
            annual_avg * 1.10,  # Mar
            annual_avg * 1.05,  # Apr
            annual_avg * 0.95,  # May
            annual_avg * 0.85,  # Jun
            annual_avg * 0.80,  # Jul
            annual_avg * 0.85,  # Aug
            annual_avg * 0.95,  # Sep
            annual_avg * 1.05,  # Oct
            annual_avg * 1.10,  # Nov
            annual_avg * 1.13,  # Dec
        ]
        
        return ResourceAssessment(
            location=location,
            resource_type=ResourceType(resource_type),
            annual_average=annual_avg,
            monthly_averages=monthly,
            unit=unit,
            data_year=2023,
            resolution_km=self.GRID_RESOLUTION * 111,
            metadata={
                "source": "NREL",
                "dataset": "Wind Toolkit",
                "hub_height": "100m" if "100m" in resource_type else "50m",
            }
        )
    
    def _map_resource_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map resource assessment data to canonical schema."""
        
        assessment: ResourceAssessment = raw["assessment"]
        metric = raw["metric"]
        value = raw["value"]
        
        resource = RenewableResource(
            location=assessment.location,
            resource_type=assessment.resource_type.value,
            annual_average=assessment.annual_average,
            unit=assessment.unit,
            resolution_km=assessment.resolution_km,
            data_source="NREL",
            metadata=assessment.metadata,
        )
        
        return {
            "event_time_utc": int(datetime.now(timezone.utc).timestamp() * 1000),
            "market": "infra",
            "product": metric,
            "instrument_id": f"RESOURCE.{assessment.resource_type.value}.GRID",
            "location_code": f"{assessment.location.lat:.2f},{assessment.location.lon:.2f}",
            "price_type": "observation",
            "value": value,
            "unit": assessment.unit,
            "source": self.source_id,
            "commodity_type": self.commodity_type.value,
            "metadata": json.dumps({
                "coordinates": assessment.location.to_dict(),
                "resource_type": assessment.resource_type.value,
                "data_year": assessment.data_year,
                "resolution_km": assessment.resolution_km,
                "monthly_averages": assessment.monthly_averages,
                **assessment.metadata,
            })
        }
    
    def _validate_config(self) -> None:
        """Validate connector configuration."""
        
        if not self.api_key:
            raise ValueError("NREL API key must be provided")
        
        valid_resources = {
            "solar_ghi", "solar_dni", 
            "wind_speed_100m", "wind_speed_50m",
            "hydro_potential"
        }
        
        for rt in self.resource_types:
            if rt not in valid_resources:
                raise ValueError(f"Invalid resource type: {rt}")
