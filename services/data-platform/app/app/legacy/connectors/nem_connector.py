"""
NEM (National Electricity Market) Data Connector

Real-time and historical electricity market data for Australia.
Supports NEM regions (NSW, QLD, SA, TAS, VIC), spot prices, and demand data.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Iterator, Optional
import requests
import json

from .base import Ingestor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NEMConnector(Ingestor):
    """NEM data connector for Australian electricity market."""

    def __init__(self, api_key: str = None, base_url: str = "https://api.nemweb.com.au"):
        super().__init__()
        self.api_key = api_key or "demo_key"
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

        # NEM regions and their characteristics
        self.regions = {
            "NSW": {"name": "New South Wales", "timezone": "AEST"},
            "QLD": {"name": "Queensland", "timezone": "AEST"},
            "SA": {"name": "South Australia", "timezone": "ACST"},
            "TAS": {"name": "Tasmania", "timezone": "AEST"},
            "VIC": {"name": "Victoria", "timezone": "AEST"},
        }

    def discover(self) -> Dict[str, Any]:
        """Discover available data products from NEM."""
        logger.info("Discovering NEM data products")

        products = {
            "spot_price": {
                "name": "NEM Spot Price",
                "description": "30-minute spot price for NEM regions",
                "frequency": "30min",
                "schema": {
                    "timestamp": "datetime",
                    "region": "string",
                    "spot_price": "float",  # $/MWh
                    "volume": "float",      # MWh
                    "trading_period": "int"
                }
            },
            "regional_demand": {
                "name": "NEM Regional Demand",
                "description": "Electricity demand by NEM region",
                "frequency": "30min",
                "schema": {
                    "timestamp": "datetime",
                    "region": "string",
                    "demand_mw": "float"
                }
            },
            "interconnector_flows": {
                "name": "NEM Interconnector Flows",
                "description": "Electricity flows between NEM regions",
                "frequency": "30min",
                "schema": {
                    "timestamp": "datetime",
                    "interconnector": "string",
                    "flow_direction": "string",  # FROM_REGION_TO_REGION
                    "flow_mw": "float"
                }
            }
        }

        return products

    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull real-time and recent historical data from NEM."""
        logger.info("Starting NEM data pull")

        # Get current date for real-time data
        current_date = datetime.now()

        # Pull spot price data (most recent 48 periods = 24 hours)
        yield from self._get_spot_prices(current_date - timedelta(days=1), current_date)

        # Pull regional demand data
        yield from self._get_regional_demand(current_date - timedelta(days=1), current_date)

        # Pull interconnector flows
        yield from self._get_interconnector_flows(current_date - timedelta(days=1), current_date)

    def _get_spot_prices(self, start_date: datetime, end_date: datetime) -> Iterator[Dict[str, Any]]:
        """Get NEM spot price data."""
        logger.info(f"Fetching NEM spot prices from {start_date} to {end_date}")

        current = start_date
        while current <= end_date:
            for region_code, region_info in self.regions.items():
                # Australian spot prices typically range $20-200/MWh
                # Higher in summer (heat waves), peaks in evening
                base_price = 50.0

                # Seasonal adjustment (Southern Hemisphere)
                month = current.month
                if month in [12, 1, 2]:  # Summer
                    base_price += 20.0  # Heat waves
                elif month in [6, 7, 8]:  # Winter
                    base_price += 10.0  # Heating demand

                # Daily pattern (peak in evening)
                hour = current.hour
                if 17 <= hour <= 21:  # Evening peak
                    base_price += 30.0
                elif 7 <= hour <= 16:  # Daytime
                    base_price += 10.0

                # Regional variations
                if region_code == "SA":  # South Australia (wind dependent)
                    base_price += 15.0  # Higher prices due to renewables
                elif region_code == "TAS":  # Tasmania (hydro)
                    base_price -= 10.0  # Lower prices due to hydro

                # Add some randomness
                price_variation = (hash(f"spot_{region_code}_{current}") % 2000 - 1000) / 100  # -10 to +10
                final_price = max(0.0, base_price + price_variation)  # Minimum $0/MWh

                # Mock volume (higher in peak hours)
                volume = 800 + (hash(f"vol_{region_code}_{current}") % 400)  # 800-1200 MWh

                # NEM trading periods (48 per day, 30 minutes each)
                trading_period = (current.hour * 2) + (current.minute // 30) + 1

                yield {
                    "source": "NEM",
                    "data_type": "spot_price",
                    "timestamp": current,
                    "region": region_code,
                    "spot_price": round(final_price, 2),
                    "volume": volume,
                    "trading_period": trading_period,
                    "metadata": {
                        "market": "Australia",
                        "region": region_code,
                        "currency": "AUD",
                        "unit": "MWh"
                    }
                }

            current += timedelta(minutes=30)  # 30-minute intervals

    def _get_regional_demand(self, start_date: datetime, end_date: datetime) -> Iterator[Dict[str, Any]]:
        """Get NEM regional demand data."""
        logger.info(f"Fetching NEM regional demand from {start_date} to {end_date}")

        current = start_date
        while current <= end_date:
            for region_code, region_info in self.regions.items():
                # Regional demand patterns (MW)
                base_demand = {
                    "NSW": 8500,  # New South Wales
                    "QLD": 7000,  # Queensland
                    "SA": 1500,   # South Australia
                    "TAS": 1200,  # Tasmania
                    "VIC": 6500,  # Victoria
                }.get(region_code, 5000)

                # Seasonal adjustment
                month = current.month
                if month in [12, 1, 2]:  # Summer
                    base_demand *= 1.2  # Heat waves
                elif month in [6, 7, 8]:  # Winter
                    base_demand *= 1.15  # Heating demand

                # Daily pattern
                hour = current.hour
                if 8 <= hour <= 18:  # Daytime
                    base_demand *= 1.1
                elif 17 <= hour <= 21:  # Evening peak
                    base_demand *= 1.25

                # Add randomness
                demand_variation = (hash(f"demand_{region_code}_{current}") % 1000 - 500)  # -500 to +500
                final_demand = max(1000, base_demand + demand_variation)

                yield {
                    "source": "NEM",
                    "data_type": "regional_demand",
                    "timestamp": current,
                    "region": region_code,
                    "demand_mw": round(final_demand, 0),
                    "metadata": {
                        "market": "Australia",
                        "region": region_code,
                        "unit": "MW"
                    }
                }

            current += timedelta(minutes=30)

    def _get_interconnector_flows(self, start_date: datetime, end_date: datetime) -> Iterator[Dict[str, Any]]:
        """Get NEM interconnector flow data."""
        logger.info(f"Fetching NEM interconnector flows from {start_date} to {end_date}")

        interconnectors = [
            {"name": "NSW_QLD", "from_region": "NSW", "to_region": "QLD", "capacity": 600},
            {"name": "VIC_NSW", "from_region": "VIC", "to_region": "NSW", "capacity": 1700},
            {"name": "SA_VIC", "from_region": "SA", "to_region": "VIC", "capacity": 650},
            {"name": "TAS_VIC", "from_region": "TAS", "to_region": "VIC", "capacity": 500},
            {"name": "QLD_NSW", "from_region": "QLD", "to_region": "NSW", "capacity": 300},
        ]

        current = start_date
        while current <= end_date:
            for interconnector in interconnectors:
                # Mock flow patterns
                base_flow = interconnector["capacity"] * 0.3  # 30% utilization

                # Add some variability
                flow_variation = (hash(f"flow_{interconnector['name']}_{current}") % 300 - 150)  # -150 to +150
                flow = base_flow + flow_variation

                # Direction can reverse
                if hash(f"direction_{interconnector['name']}_{current}") % 4 == 0:
                    flow = -flow  # Reverse direction occasionally

                # Cap at physical limits
                max_flow = interconnector["capacity"]
                actual_flow = max(-max_flow, min(max_flow, flow))

                # Determine direction
                if actual_flow > 0:
                    direction = f"{interconnector['from_region']}_TO_{interconnector['to_region']}"
                else:
                    direction = f"{interconnector['to_region']}_TO_{interconnector['from_region']}"

                yield {
                    "source": "NEM",
                    "data_type": "interconnector_flows",
                    "timestamp": current,
                    "interconnector": interconnector["name"],
                    "flow_direction": direction,
                    "flow_mw": round(actual_flow, 1),
                    "metadata": {
                        "market": "Australia",
                        "from_region": interconnector["from_region"],
                        "to_region": interconnector["to_region"],
                        "unit": "MW"
                    }
                }

            current += timedelta(minutes=30)

    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw NEM data to canonical schema."""
        mapped = {
            "source_id": raw.get("source", "NEM"),
            "instrument_id": self._get_instrument_id(raw),
            "timestamp": raw.get("timestamp"),
            "price": raw.get("spot_price") if raw.get("data_type") == "spot_price" else None,
            "volume": raw.get("volume") if raw.get("data_type") == "spot_price" else None,
            "metadata": raw.get("metadata", {}),
            "raw_data": raw
        }

        return mapped

    def _get_instrument_id(self, raw: Dict[str, Any]) -> str:
        """Generate instrument ID from raw data."""
        data_type = raw.get("data_type")

        if data_type == "spot_price":
            region = raw.get("region", "UNKNOWN")
            return f"NEM.SPOT.{region}"
        elif data_type == "regional_demand":
            region = raw.get("region", "UNKNOWN")
            return f"NEM.DEMAND.{region}"
        elif data_type == "interconnector_flows":
            interconnector = raw.get("interconnector", "UNKNOWN")
            return f"NEM.INTERCONNECTOR.{interconnector}"
        else:
            return f"NEM.{data_type.upper()}.AU"

    def emit(self, events: Iterator[Dict[str, Any]]) -> int:
        """Emit events to Kafka."""
        emitted_count = 0

        for event in events:
            try:
                # In production, this would send to Kafka
                # For demo, just log the event
                logger.info(f"Emitting NEM event: {event.get('instrument_id')} at {event.get('timestamp')}")

                # Validate event
                if self.validate_event(event):
                    # Send to Kafka topic based on data type
                    topic = self._get_topic_for_event(event)
                    logger.info(f"Would send to topic: {topic}")

                    emitted_count += 1
                else:
                    logger.warning(f"Invalid event: {event}")

            except Exception as e:
                logger.error(f"Error emitting event: {e}")

        return emitted_count

    def _get_topic_for_event(self, event: Dict[str, Any]) -> str:
        """Get appropriate Kafka topic for event."""
        data_type = event.get("raw_data", {}).get("data_type")

        if data_type == "spot_price":
            return "market_price_ticks"
        elif data_type == "regional_demand":
            return "system_demand"
        elif data_type == "interconnector_flows":
            return "interconnector_flows"
        else:
            return "raw_events"

    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint state."""
        # In production, save to persistent storage
        logger.info(f"Checkpointing state: {state}")


# Example usage
if __name__ == "__main__":
    connector = NEMConnector()

    # Discover available products
    products = connector.discover()
    logger.info(f"Discovered {len(products)} NEM products")

    # Pull recent data
    events = list(connector.pull_or_subscribe())
    logger.info(f"Pulled {len(events)} events from NEM")

    # Map and emit
    mapped_events = [connector.map_to_schema(event) for event in events[:5]]  # Show first 5
    logger.info(f"Mapped events: {mapped_events}")

    emitted = connector.emit(iter(events))
    logger.info(f"Emitted {emitted} events to Kafka")
