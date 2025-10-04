"""
Pipeline Flow Data Connector

Ingests natural gas pipeline flow data from:
- Genscape (real-time pipeline flows and capacity)
- PointLogic (pipeline nominations and scheduling)
- EIA (weekly storage and flow data)
- European gas storage data (GIE AGSI+)

Provides critical data for:
- Pipeline congestion analysis
- Storage arbitrage opportunities
- Regional basis modeling
- Supply chain optimization
"""
import logging
from datetime import datetime, timedelta, timezone, date
from typing import Iterator, Dict, Any, Optional, List
import time
import requests
import json

from .base import Ingestor, CommodityType

logger = logging.getLogger(__name__)


class PipelineFlowConnector(Ingestor):
    """
    Natural gas pipeline flow and capacity data connector.

    Responsibilities:
    - Ingest real-time pipeline flow data from Genscape
    - Collect pipeline nominations from PointLogic
    - Gather storage data from EIA and GIE
    - Map pipeline data to canonical schema
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.commodity_type = CommodityType.GAS

        # API configurations
        self.genscape_api_url = config.get("genscape_api_url", "https://api.genscape.com")
        self.genscape_api_key = config.get("genscape_api_key")
        self.pointlogic_api_url = config.get("pointlogic_api_url", "https://api.pointlogic.com")
        self.pointlogic_api_key = config.get("pointlogic_api_key")
        self.eia_api_key = config.get("eia_api_key")

        # Key pipelines and storage facilities to track
        self._register_pipeline_specifications()

    def _register_pipeline_specifications(self) -> None:
        """Register specifications for major pipelines and storage facilities."""

        # Major US natural gas pipelines
        self.pipelines = {
            "transco": {
                "name": "Transcontinental Gas Pipe Line",
                "operator": "Williams",
                "capacity": 10.2,  # Bcf/d
                "zones": ["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5", "Zone 6"]
            },
            "tetco": {
                "name": "Texas Eastern Transmission",
                "operator": "Enbridge",
                "capacity": 8.1,  # Bcf/d
                "zones": ["ETX", "STX", "WLA", "ELA"]
            },
            "ngpl": {
                "name": "Natural Gas Pipeline Company of America",
                "operator": "Kinder Morgan",
                "capacity": 6.4,  # Bcf/d
                "zones": ["Gulf Coast", "Midcontinent", "North", "South"]
            },
            "columbia": {
                "name": "Columbia Gas Transmission",
                "operator": "TC Energy",
                "capacity": 9.9,  # Bcf/d
                "zones": ["Appalachia", "TCO Pool"]
            }
        }

        # Major storage facilities
        self.storage_facilities = {
            "eastern": {
                "name": "Eastern Storage Facilities",
                "type": "depleted_reservoir",
                "working_capacity": 2500,  # Bcf
                "operators": ["Dominion", "National Fuel", "UGI"]
            },
            "producing": {
                "name": "Producing Region Storage",
                "type": "salt_dome",
                "working_capacity": 800,  # Bcf
                "operators": ["Kinder Morgan", "Enterprise"]
            },
            "western": {
                "name": "Western Storage Facilities",
                "type": "aquifer",
                "working_capacity": 400,  # Bcf
                "operators": ["Southwest Gas", "Questar"]
            }
        }

    def discover(self) -> Dict[str, Any]:
        """Discover available pipeline flow and storage data streams."""
        return {
            "source_id": self.source_id,
            "commodity_type": self.commodity_type.value,
            "data_streams": [
                {
                    "type": "pipeline_flows",
                    "description": "Real-time pipeline flow data",
                    "frequency": "hourly",
                    "pipelines": list(self.pipelines.keys())
                },
                {
                    "type": "pipeline_nominations",
                    "description": "Pipeline nomination and scheduling data",
                    "frequency": "daily",
                    "pipelines": list(self.pipelines.keys())
                },
                {
                    "type": "storage_data",
                    "description": "Natural gas storage levels and flows",
                    "frequency": "daily",
                    "regions": ["Eastern", "Producing", "Western"]
                }
            ]
        }

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull pipeline flow and storage data.

        For production: Use Genscape, PointLogic, and EIA APIs
        For development: Generate realistic mock data
        """
        try:
            # Get current date
            current_date = datetime.now(timezone.utc).date()

            # Pull pipeline flow data
            for pipeline_id, pipeline_info in self.pipelines.items():
                try:
                    flow_data = self._fetch_pipeline_flow_data(pipeline_id, current_date)

                    for zone, flow_info in flow_data.items():
                        yield {
                            "event_time": datetime.combine(current_date, datetime.min.time(), timezone.utc),
                            "arrival_time": datetime.now(timezone.utc),
                            "market": self.commodity_type.value,
                            "product": "pipeline_flow",
                            "instrument_id": f"{pipeline_id}_{zone}",
                            "location_code": zone,
                            "price_type": "flow_rate",
                            "value": flow_info["flow_rate"],
                            "volume": flow_info["capacity_utilization"],
                            "currency": "USD",  # For capacity value
                            "unit": "MMcf/d",
                            "source": self.source_id,
                            "commodity_type": self.commodity_type.value,
                            "version_id": 1,
                            "metadata": json.dumps({
                                "pipeline": pipeline_id,
                                "zone": zone,
                                "capacity": pipeline_info["capacity"],
                                "operator": pipeline_info["operator"]
                            })
                        }

                except Exception as e:
                    logger.error(f"Error fetching pipeline data for {pipeline_id}: {e}")
                    continue

            # Pull storage data
            for region, storage_info in self.storage_facilities.items():
                try:
                    storage_data = self._fetch_storage_data(region, current_date)

                    yield {
                        "event_time": datetime.combine(current_date, datetime.min.time(), timezone.utc),
                        "arrival_time": datetime.now(timezone.utc),
                        "market": self.commodity_type.value,
                        "product": "storage",
                        "instrument_id": f"storage_{region}",
                        "location_code": region,
                        "price_type": "inventory",
                        "value": storage_data["inventory_level"],
                        "volume": storage_data["injection_rate"],
                        "currency": "USD",
                        "unit": "Bcf",
                        "source": self.source_id,
                        "commodity_type": self.commodity_type.value,
                        "version_id": 1,
                        "metadata": json.dumps({
                            "region": region,
                            "capacity": storage_info["working_capacity"],
                            "type": storage_info["type"]
                        })
                    }

                except Exception as e:
                    logger.error(f"Error fetching storage data for {region}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in pipeline flow connector: {e}")
            raise

    def _fetch_pipeline_flow_data(self, pipeline_id: str, current_date: date) -> Dict[str, Any]:
        """Fetch pipeline flow data for a specific pipeline."""

        # Mock pipeline flow data for development
        # In production: Query Genscape or PointLogic APIs

        pipeline_info = self.pipelines[pipeline_id]
        zones = pipeline_info["zones"]

        flow_data = {}
        for zone in zones:
            # Generate realistic flow data
            import random

            # Base capacity utilization (60-90% typical)
            utilization = random.uniform(0.6, 0.9)

            # Calculate flow rate based on capacity
            flow_rate = pipeline_info["capacity"] * utilization * random.uniform(0.8, 1.2)  # Add some variation

            flow_data[zone] = {
                "flow_rate": round(flow_rate, 1),
                "capacity_utilization": round(utilization * 100, 1),
                "direction": "forward" if random.random() > 0.1 else "reverse",  # Mostly forward flow
                "pressure": round(random.uniform(800, 1200), 1)  # Pipeline pressure in psi
            }

        return flow_data

    def _fetch_storage_data(self, region: str, current_date: date) -> Dict[str, Any]:
        """Fetch storage data for a specific region."""

        # Mock storage data for development
        # In production: Query EIA or GIE APIs

        storage_info = self.storage_facilities[region]

        # Generate realistic storage data
        import random

        # Current inventory as percentage of capacity
        inventory_pct = random.uniform(0.3, 0.9)  # 30-90% full

        # Calculate actual inventory
        inventory_level = storage_info["working_capacity"] * inventory_pct

        # Injection/withdrawal rate
        injection_rate = random.uniform(-50, 100)  # Can be negative (withdrawal)

        return {
            "inventory_level": round(inventory_level, 1),
            "injection_rate": round(injection_rate, 1),
            "inventory_pct": round(inventory_pct * 100, 1),
            "working_capacity": storage_info["working_capacity"],
            "cushion_gas": storage_info["working_capacity"] * 0.2  # 20% cushion gas
        }

    def _authenticate_genscape_api(self) -> Dict[str, str]:
        """Authenticate with Genscape API."""
        return {
            "Authorization": f"Bearer {self.genscape_api_key}",
            "Content-Type": "application/json"
        }

    def _authenticate_pointlogic_api(self) -> Dict[str, str]:
        """Authenticate with PointLogic API."""
        return {
            "Authorization": f"Bearer {self.pointlogic_api_key}",
            "Content-Type": "application/json"
        }
