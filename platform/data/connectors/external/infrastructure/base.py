"""
Infrastructure Connector Base Utilities
---------------------------------------

Shared utilities and data models for infrastructure connectors including:
- LNG terminals and storage
- Power plants and generation assets  
- Transmission infrastructure
- Renewable resource mapping

Provides common functionality for geospatial data handling, facility aggregation,
and infrastructure-specific data quality checks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from geopy.distance import geodesic

from ....base import Ingestor, CommodityType

logger = logging.getLogger(__name__)


class InfrastructureType(Enum):
    """Types of energy infrastructure assets."""
    LNG_TERMINAL = "lng_terminal"
    GAS_STORAGE = "gas_storage"
    POWER_PLANT = "power_plant"
    TRANSMISSION_LINE = "transmission_line"
    SUBSTATION = "substation"
    RENEWABLE_RESOURCE = "renewable_resource"
    INTERCONNECTOR = "interconnector"


class FuelType(Enum):
    """Fuel types for power generation assets."""
    COAL = "coal"
    NATURAL_GAS = "natural_gas"
    NUCLEAR = "nuclear"
    HYDRO = "hydro"
    WIND = "wind"
    SOLAR = "solar"
    BIOMASS = "biomass"
    OIL = "oil"
    GEOTHERMAL = "geothermal"
    BATTERY = "battery"
    OTHER = "other"


class OperationalStatus(Enum):
    """Operational status of infrastructure assets."""
    OPERATIONAL = "operational"
    CONSTRUCTION = "construction"
    PLANNED = "planned"
    DECOMMISSIONED = "decommissioned"
    MOTHBALLED = "mothballed"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class GeoLocation:
    """Geographic location with coordinate validation."""
    lat: float
    lon: float
    
    def __post_init__(self):
        if not -90 <= self.lat <= 90:
            raise ValueError(f"Invalid latitude: {self.lat}")
        if not -180 <= self.lon <= 180:
            raise ValueError(f"Invalid longitude: {self.lon}")
    
    def distance_to(self, other: GeoLocation) -> float:
        """Calculate distance in kilometers to another location."""
        return geodesic((self.lat, self.lon), (other.lat, other.lon)).km
    
    def to_dict(self) -> Dict[str, float]:
        return {"lat": self.lat, "lon": self.lon}


@dataclass(slots=True)
class InfrastructureAsset:
    """Base representation of an infrastructure asset."""
    asset_id: str
    name: str
    asset_type: InfrastructureType
    location: GeoLocation
    country: str
    region: Optional[str] = None
    operator: Optional[str] = None
    owner: Optional[str] = None
    status: OperationalStatus = OperationalStatus.UNKNOWN
    commissioned_date: Optional[date] = None
    decommissioned_date: Optional[date] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_operational(self, as_of: date = None) -> bool:
        """Check if asset is operational as of given date."""
        if self.status != OperationalStatus.OPERATIONAL:
            return False
        
        check_date = as_of or date.today()
        
        if self.commissioned_date and check_date < self.commissioned_date:
            return False
            
        if self.decommissioned_date and check_date >= self.decommissioned_date:
            return False
            
        return True
    
    def to_instrument_id(self) -> str:
        """Generate canonical instrument ID for the asset."""
        type_map = {
            InfrastructureType.LNG_TERMINAL: "LNG",
            InfrastructureType.GAS_STORAGE: "GAS_STORAGE",
            InfrastructureType.POWER_PLANT: "POWER",
            InfrastructureType.TRANSMISSION_LINE: "TRANSMISSION",
            InfrastructureType.RENEWABLE_RESOURCE: "RENEWABLE",
        }
        prefix = type_map.get(self.asset_type, "INFRA")
        return f"{prefix}.{self.country}.{self.asset_id}"


@dataclass(slots=True)
class PowerPlant(InfrastructureAsset):
    """Power generation facility with capacity and fuel information."""
    capacity_mw: float
    primary_fuel: FuelType
    secondary_fuel: Optional[FuelType] = None
    efficiency_pct: Optional[float] = None
    capacity_factor: Optional[float] = None
    annual_generation_gwh: Optional[float] = None
    emissions_rate_tco2_mwh: Optional[float] = None
    
    def __post_init__(self):
        self.asset_type = InfrastructureType.POWER_PLANT
        if self.capacity_mw <= 0:
            raise ValueError(f"Invalid capacity: {self.capacity_mw} MW")


@dataclass(slots=True)
class LNGTerminal(InfrastructureAsset):
    """LNG terminal with storage and regasification capacity."""
    storage_capacity_gwh: float
    regasification_capacity_gwh_d: float
    num_tanks: Optional[int] = None
    berth_capacity: Optional[int] = None
    send_out_capacity_gwh_d: Optional[float] = None
    
    def __post_init__(self):
        self.asset_type = InfrastructureType.LNG_TERMINAL


@dataclass(slots=True)
class TransmissionLine(InfrastructureAsset):
    """Transmission infrastructure connecting two points."""
    from_location: GeoLocation
    to_location: GeoLocation
    voltage_kv: float
    capacity_mw: float
    length_km: Optional[float] = None
    line_type: str = "AC"  # AC or DC
    
    def __post_init__(self):
        self.asset_type = InfrastructureType.TRANSMISSION_LINE
        if self.length_km is None:
            self.length_km = self.from_location.distance_to(self.to_location)


@dataclass(slots=True)
class RenewableResource:
    """Renewable resource potential at a location."""
    location: GeoLocation
    resource_type: str  # solar_ghi, wind_100m, etc.
    annual_average: float
    unit: str
    resolution_km: float
    data_source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class InfrastructureConnector(Ingestor):
    """Base class for infrastructure data connectors."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.assets: Dict[str, InfrastructureAsset] = {}
        self.spatial_index: Optional[Any] = None
        
    def create_infrastructure_event(
        self,
        asset: InfrastructureAsset,
        metric: str,
        value: float,
        unit: str,
        event_time: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create canonical infrastructure data event."""
        
        event = {
            "event_time_utc": int(event_time.timestamp() * 1000),
            "market": "infra",
            "product": metric,
            "instrument_id": asset.to_instrument_id(),
            "location_code": f"{asset.country}_{asset.region or 'national'}",
            "price_type": "observation",
            "value": value,
            "unit": unit,
            "source": self.source_id,
            "commodity_type": self._infer_commodity_type(asset),
            "metadata": json.dumps({
                "asset_id": asset.asset_id,
                "asset_name": asset.name,
                "asset_type": asset.asset_type.value,
                "coordinates": asset.location.to_dict(),
                "operator": asset.operator,
                "status": asset.status.value,
                **(metadata or {})
            })
        }
        
        return event
    
    def _infer_commodity_type(self, asset: InfrastructureAsset) -> str:
        """Infer commodity type from asset type."""
        if isinstance(asset, LNGTerminal):
            return CommodityType.GAS.value
        elif isinstance(asset, PowerPlant):
            if asset.primary_fuel in [FuelType.WIND, FuelType.SOLAR, FuelType.HYDRO]:
                return CommodityType.RENEWABLES.value
            elif asset.primary_fuel == FuelType.COAL:
                return CommodityType.COAL.value
            elif asset.primary_fuel == FuelType.NATURAL_GAS:
                return CommodityType.GAS.value
            else:
                return CommodityType.POWER.value
        else:
            return CommodityType.POWER.value
    
    def aggregate_by_region(
        self,
        assets: List[InfrastructureAsset],
        metric_func: callable,
        group_by: str = "country"
    ) -> Dict[str, float]:
        """Aggregate infrastructure metrics by geographic region."""
        
        aggregates = {}
        
        for asset in assets:
            if not asset.is_operational():
                continue
                
            if group_by == "country":
                key = asset.country
            elif group_by == "region":
                key = f"{asset.country}_{asset.region or 'national'}"
            else:
                key = "global"
            
            if key not in aggregates:
                aggregates[key] = []
            
            aggregates[key].append(asset)
        
        results = {}
        for key, group_assets in aggregates.items():
            results[key] = metric_func(group_assets)
            
        return results
    
    def find_nearest_assets(
        self,
        location: GeoLocation,
        asset_type: Optional[InfrastructureType] = None,
        radius_km: float = 100,
        limit: int = 10
    ) -> List[Tuple[InfrastructureAsset, float]]:
        """Find nearest infrastructure assets to a location."""
        
        candidates = []
        
        for asset in self.assets.values():
            if asset_type and asset.asset_type != asset_type:
                continue
                
            distance = location.distance_to(asset.location)
            
            if distance <= radius_km:
                candidates.append((asset, distance))
        
        # Sort by distance and return top N
        candidates.sort(key=lambda x: x[1])
        return candidates[:limit]
    
    def validate_infrastructure_data(self, event: Dict[str, Any]) -> bool:
        """Validate infrastructure-specific data quality rules."""
        
        # Call parent validation first
        if not self.validate_event(event):
            return False
        
        # Infrastructure-specific validations
        metadata = json.loads(event.get("metadata", "{}"))
        
        # Check coordinates
        coords = metadata.get("coordinates", {})
        if coords:
            try:
                GeoLocation(coords["lat"], coords["lon"])
            except ValueError:
                logger.warning(f"Invalid coordinates: {coords}")
                return False
        
        # Check capacity values
        if "capacity" in event["product"]:
            if event["value"] < 0:
                logger.warning(f"Negative capacity value: {event['value']}")
                return False
        
        # Check utilization percentages
        if "utilization" in event["product"] or "pct" in event["unit"]:
            if not 0 <= event["value"] <= 100:
                logger.warning(f"Invalid percentage value: {event['value']}")
                return False
        
        return True
    
    def calculate_availability_factor(
        self,
        operational_hours: float,
        period_hours: float
    ) -> float:
        """Calculate availability factor for infrastructure assets."""
        if period_hours <= 0:
            return 0.0
        return min(operational_hours / period_hours, 1.0)
    
    def calculate_capacity_factor(
        self,
        actual_output: float,
        rated_capacity: float,
        period_hours: float
    ) -> float:
        """Calculate capacity factor for generation assets."""
        if rated_capacity <= 0 or period_hours <= 0:
            return 0.0
        max_output = rated_capacity * period_hours
        return min(actual_output / max_output, 1.0)
