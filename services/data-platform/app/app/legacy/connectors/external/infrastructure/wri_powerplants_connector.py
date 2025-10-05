"""
WRI Global Power Plant Database Connector
------------------------------------------

Fetches comprehensive power plant data from the World Resources Institute
Global Power Plant Database, containing information on ~30,000 power plants
worldwide.

Capabilities
~~~~~~~~~~~~
- Downloads and parses CSV/JSON data files
- Tracks power plant capacity, fuel type, and generation
- Handles multi-fuel plants and capacity changes
- Geocoded plant locations
- Emits canonical infrastructure events

Data source: https://datasets.wri.org/dataset/globalpowerplantdatabase
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import Any, Dict, Iterator, List, Optional, Set
from urllib.parse import urlparse

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import (
    InfrastructureConnector,
    PowerPlant,
    GeoLocation,
    FuelType,
    OperationalStatus,
)
from ....base import CommodityType

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WRIPowerPlant:
    """Power plant record from WRI database."""
    
    gppd_id: str
    name: str
    country: str
    country_long: str
    latitude: float
    longitude: float
    primary_fuel: str
    other_fuels: List[str]
    capacity_mw: float
    owner: Optional[str]
    source: Optional[str]
    url: Optional[str]
    commissioning_year: Optional[int]
    retirement_year: Optional[int]
    generation_gwh: Dict[int, float]  # Year -> GWh
    estimated_generation: Dict[int, bool]  # Year -> is_estimated
    raw_data: Dict[str, Any]


class WRIPowerPlantsConnector(InfrastructureConnector):
    """Connector for WRI Global Power Plant Database."""
    
    # Default data URL (version 1.3.0)
    DEFAULT_DATA_URL = "https://wri-dataportal-prod.s3.amazonaws.com/manual/global_power_plant_database_v_1_3.zip"
    
    # Fuel type mappings from WRI to our canonical types
    FUEL_TYPE_MAP = {
        "Coal": FuelType.COAL,
        "Gas": FuelType.NATURAL_GAS,
        "Oil": FuelType.OIL,
        "Nuclear": FuelType.NUCLEAR,
        "Hydro": FuelType.HYDRO,
        "Wind": FuelType.WIND,
        "Solar": FuelType.SOLAR,
        "Geothermal": FuelType.GEOTHERMAL,
        "Biomass": FuelType.BIOMASS,
        "Biogas": FuelType.BIOMASS,
        "Waste": FuelType.BIOMASS,
        "Wave and Tidal": FuelType.HYDRO,
        "Petcoke": FuelType.COAL,
        "Cogeneration": FuelType.OTHER,
        "Other": FuelType.OTHER,
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.data_url = config.get("data_url", self.DEFAULT_DATA_URL)
        self.countries = config.get("countries")  # Optional filter
        self.min_capacity_mw = config.get("min_capacity_mw", 0)
        self.fuel_types = config.get("fuel_types")  # Optional filter
        self.include_generation = config.get("include_generation", True)
        self.data_dir = config.get("data_dir", tempfile.gettempdir())
        
        kafka_cfg = config.get("kafka") or {}
        if not kafka_cfg:
            self.kafka_topic = "market.fundamentals"
            self.kafka_bootstrap_servers = config.get("kafka_bootstrap", "kafka:9092")
        
        self.session = requests.Session()
        self._plants: Dict[str, WRIPowerPlant] = {}
        self._countries_seen: Set[str] = set()
    
    def discover(self) -> Dict[str, Any]:
        """Discover available data streams."""
        
        return {
            "source_id": self.source_id,
            "data_source": "WRI Global Power Plant Database",
            "version": "1.3.0",
            "record_count": "~30,000 power plants",
            "coverage": "Global",
            "streams": [
                {
                    "name": "power_plant_capacity",
                    "variables": [
                        "installed_capacity_mw",
                        "primary_fuel_type",
                        "commissioning_year",
                    ],
                    "frequency": "quarterly",
                },
                {
                    "name": "power_plant_generation",
                    "variables": [
                        "annual_generation_gwh",
                        "capacity_factor",
                    ],
                    "frequency": "annual",
                    "years": "2013-2017",
                }
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Download and parse WRI power plant data."""
        
        logger.info("Downloading WRI Global Power Plant Database from %s", self.data_url)
        
        # Download and extract data
        data_file = self._download_data()
        
        if not data_file:
            logger.error("Failed to download WRI data")
            return
        
        # Parse CSV data
        plants = list(self._parse_csv(data_file))
        logger.info("Parsed %d power plants from WRI database", len(plants))
        
        # Store plants in registry
        for plant in plants:
            self._plants[plant.gppd_id] = plant
            self._countries_seen.add(plant.country)
        
        # Emit capacity events
        for plant in plants:
            if self._should_include_plant(plant):
                yield from self._plant_to_events(plant)
        
        # Clean up temporary file
        if os.path.exists(data_file):
            os.unlink(data_file)
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw plant data to canonical schema."""
        
        plant_data: WRIPowerPlant = raw["plant"]
        metric = raw["metric"]
        value = raw["value"]
        year = raw.get("year")
        
        # Create power plant asset
        location = GeoLocation(plant_data.latitude, plant_data.longitude)
        
        # Determine operational status
        if plant_data.retirement_year and plant_data.retirement_year <= datetime.now().year:
            status = OperationalStatus.DECOMMISSIONED
        else:
            status = OperationalStatus.OPERATIONAL
        
        # Map fuel type
        fuel_type = self.FUEL_TYPE_MAP.get(plant_data.primary_fuel, FuelType.OTHER)
        
        # Create or get power plant asset
        asset_id = f"WRI_{plant_data.gppd_id}"
        
        if asset_id not in self.assets:
            commissioned_date = None
            if plant_data.commissioning_year:
                commissioned_date = date(plant_data.commissioning_year, 1, 1)
            
            decommissioned_date = None
            if plant_data.retirement_year:
                decommissioned_date = date(plant_data.retirement_year, 12, 31)
            
            plant = PowerPlant(
                asset_id=asset_id,
                name=plant_data.name,
                location=location,
                country=plant_data.country,
                status=status,
                commissioned_date=commissioned_date,
                decommissioned_date=decommissioned_date,
                capacity_mw=plant_data.capacity_mw,
                primary_fuel=fuel_type,
                owner=plant_data.owner,
                metadata={
                    "gppd_id": plant_data.gppd_id,
                    "country_long": plant_data.country_long,
                    "source": plant_data.source,
                    "url": plant_data.url,
                    "other_fuels": plant_data.other_fuels,
                }
            )
            self.assets[asset_id] = plant
        
        plant = self.assets[asset_id]
        
        # Add year to metadata if this is generation data
        metadata = {}
        if year:
            metadata["year"] = year
            metadata["estimated"] = plant_data.estimated_generation.get(year, False)
        
        return self.create_infrastructure_event(
            asset=plant,
            metric=metric,
            value=value,
            unit=raw.get("unit", "MW"),
            event_time=datetime.now(timezone.utc),
            metadata=metadata
        )
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint state."""
        
        state["countries_processed"] = list(self._countries_seen)
        state["plant_count"] = len(self._plants)
        super().checkpoint(state)
    
    # ------------------------------------------------------------------
    # Data download and parsing
    # ------------------------------------------------------------------
    
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(
            (requests.HTTPError, requests.ConnectionError, requests.Timeout)
        ),
    )
    def _download_data(self) -> Optional[str]:
        """Download and extract WRI data file."""
        
        try:
            # Check if we need to download ZIP or direct CSV
            parsed_url = urlparse(self.data_url)
            filename = os.path.basename(parsed_url.path)
            
            # Download file
            response = self.session.get(self.data_url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Save to temporary file
            temp_path = os.path.join(self.data_dir, f"wri_temp_{datetime.now().timestamp()}")
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract if ZIP file
            if filename.endswith('.zip'):
                extract_dir = temp_path + "_extracted"
                os.makedirs(extract_dir, exist_ok=True)
                
                with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Find CSV file in extracted contents
                csv_files = [
                    os.path.join(extract_dir, f) 
                    for f in os.listdir(extract_dir) 
                    if f.endswith('.csv') and 'global_power_plant' in f
                ]
                
                if csv_files:
                    os.unlink(temp_path)  # Remove ZIP
                    return csv_files[0]
                else:
                    logger.error("No CSV file found in ZIP archive")
                    return None
            else:
                # Assume direct CSV download
                return temp_path
                
        except Exception as e:
            logger.error(f"Failed to download WRI data: {e}")
            return None
    
    def _parse_csv(self, csv_file: str) -> Iterator[WRIPowerPlant]:
        """Parse WRI CSV file into power plant records."""
        
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                plant = self._parse_plant_row(row)
                if plant:
                    yield plant
    
    def _parse_plant_row(self, row: Dict[str, str]) -> Optional[WRIPowerPlant]:
        """Parse single CSV row into WRIPowerPlant object."""
        
        try:
            # Required fields
            gppd_id = row['gppd_idnr']
            name = row['name']
            country = row['country']
            country_long = row['country_long']
            
            # Parse coordinates
            latitude = float(row['latitude'])
            longitude = float(row['longitude'])
            
            # Validate coordinates
            if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
                logger.warning(f"Invalid coordinates for plant {gppd_id}: {latitude}, {longitude}")
                return None
            
            # Parse fuel types
            primary_fuel = row.get('primary_fuel', 'Other')
            other_fuels = []
            for i in range(1, 5):  # other_fuel1 through other_fuel4
                fuel = row.get(f'other_fuel{i}')
                if fuel and fuel.strip():
                    other_fuels.append(fuel)
            
            # Parse capacity
            capacity_mw = float(row.get('capacity_mw', 0))
            if capacity_mw <= 0:
                logger.warning(f"Invalid capacity for plant {gppd_id}: {capacity_mw}")
                return None
            
            # Optional fields
            owner = row.get('owner') or None
            source = row.get('source') or None
            url = row.get('url') or None
            
            # Parse years
            commissioning_year = None
            if row.get('commissioning_year'):
                try:
                    commissioning_year = int(float(row['commissioning_year']))
                except (ValueError, TypeError):
                    pass
            
            retirement_year = None
            if row.get('year_of_capacity_data'):
                try:
                    retirement_year = int(float(row['year_of_capacity_data']))
                    # Only use as retirement year if it's marked as retired
                    if row.get('status') != 'Retired':
                        retirement_year = None
                except (ValueError, TypeError):
                    pass
            
            # Parse generation data (2013-2017)
            generation_gwh = {}
            estimated_generation = {}
            
            for year in range(2013, 2018):
                gen_key = f'generation_gwh_{year}'
                est_key = f'estimated_generation_gwh_{year}'
                
                if gen_key in row and row[gen_key]:
                    try:
                        generation_gwh[year] = float(row[gen_key])
                        # Check if generation was estimated
                        estimated_generation[year] = (
                            row.get(est_key) and 
                            row[est_key].lower() in ['true', '1', 'yes']
                        )
                    except (ValueError, TypeError):
                        pass
            
            return WRIPowerPlant(
                gppd_id=gppd_id,
                name=name,
                country=country,
                country_long=country_long,
                latitude=latitude,
                longitude=longitude,
                primary_fuel=primary_fuel,
                other_fuels=other_fuels,
                capacity_mw=capacity_mw,
                owner=owner,
                source=source,
                url=url,
                commissioning_year=commissioning_year,
                retirement_year=retirement_year,
                generation_gwh=generation_gwh,
                estimated_generation=estimated_generation,
                raw_data=row,
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse plant row: {e}")
            return None
    
    # ------------------------------------------------------------------
    # Event generation
    # ------------------------------------------------------------------
    
    def _should_include_plant(self, plant: WRIPowerPlant) -> bool:
        """Check if plant matches filter criteria."""
        
        # Country filter
        if self.countries and plant.country not in self.countries:
            return False
        
        # Capacity filter
        if plant.capacity_mw < self.min_capacity_mw:
            return False
        
        # Fuel type filter
        if self.fuel_types:
            plant_fuel = self.FUEL_TYPE_MAP.get(plant.primary_fuel, FuelType.OTHER)
            if plant_fuel.value not in self.fuel_types:
                return False
        
        return True
    
    def _plant_to_events(self, plant: WRIPowerPlant) -> Iterator[Dict[str, Any]]:
        """Convert power plant to infrastructure events."""
        
        # Emit capacity event
        yield {
            "plant": plant,
            "metric": "installed_capacity_mw",
            "value": plant.capacity_mw,
            "unit": "MW",
        }
        
        # Emit fuel type as numeric code
        fuel_code = self.FUEL_TYPE_MAP.get(plant.primary_fuel, FuelType.OTHER)
        yield {
            "plant": plant,
            "metric": "primary_fuel_code",
            "value": float(list(FuelType).index(fuel_code)),
            "unit": "code",
        }
        
        # Emit generation data if available
        if self.include_generation and plant.generation_gwh:
            for year, gwh in plant.generation_gwh.items():
                yield {
                    "plant": plant,
                    "metric": "annual_generation_gwh",
                    "value": gwh,
                    "unit": "GWh",
                    "year": year,
                }
                
                # Calculate and emit capacity factor
                if plant.capacity_mw > 0:
                    # Capacity factor = actual generation / (capacity * hours_in_year)
                    max_generation = plant.capacity_mw * 8760 / 1000  # Convert MW to GWh
                    capacity_factor = gwh / max_generation if max_generation > 0 else 0
                    
                    yield {
                        "plant": plant,
                        "metric": "capacity_factor_pct",
                        "value": capacity_factor * 100,
                        "unit": "pct",
                        "year": year,
                    }
