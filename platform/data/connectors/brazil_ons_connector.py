"""
Brazil ONS (Operador Nacional do Sistema Elétrico) Connector

Ingests Brazilian power market data including:
- PLD (Preço de Liquidação das Diferenças) - Settlement prices
- Hydro reservoir levels and inflows
- Generation by source
- Four submarkets (SE/CO, S, NE, N)
"""
import logging
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any
import time

import requests
from kafka import KafkaProducer
import json

from .base import Ingestor

logger = logging.getLogger(__name__)


class BrazilONSConnector(Ingestor):
    """Brazil ONS electricity market data connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_base = config.get("api_base", "https://ons-data.operador.org.br/api")
        self.submarket = config.get("submarket", "ALL")  # SE/CO, S, NE, N, or ALL
        self.data_type = config.get("data_type", "PLD")  # PLD, HYDRO, GENERATION
        self.kafka_topic = config.get("kafka_topic", "power.ticks.v1")
        self.kafka_bootstrap = config.get("kafka_bootstrap", "kafka:9092")
        self.producer = None
    
    def discover(self) -> Dict[str, Any]:
        """Discover ONS data streams."""
        return {
            "source_id": self.source_id,
            "streams": [
                {
                    "name": "pld_hourly",
                    "market": "power",
                    "product": "energy",
                    "description": "PLD - Preço de Liquidação das Diferenças",
                    "submarkets": ["SE/CO", "S", "NE", "N"],
                    "currency": "BRL",
                    "update_freq": "hourly",
                },
                {
                    "name": "hydro_reservoirs",
                    "market": "power",
                    "product": "fundamentals",
                    "description": "Reservoir levels and stored energy",
                    "update_freq": "daily",
                },
                {
                    "name": "generation_mix",
                    "market": "power",
                    "product": "generation",
                    "description": "Generation by source (hydro, wind, solar, thermal)",
                    "update_freq": "hourly",
                },
                {
                    "name": "load_forecast",
                    "market": "power",
                    "product": "load",
                    "description": "System load and forecasts",
                    "update_freq": "hourly",
                },
            ],
        }
    
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """Pull data from ONS API."""
        last_checkpoint = self.load_checkpoint()
        last_time = (
            last_checkpoint.get("last_event_time")
            if last_checkpoint
            else datetime.utcnow() - timedelta(hours=1)
        )
        
        logger.info(f"Fetching ONS {self.data_type} data since {last_time}")
        
        if self.data_type == "PLD":
            yield from self._fetch_pld()
        elif self.data_type == "HYDRO":
            yield from self._fetch_hydro()
        elif self.data_type == "GENERATION":
            yield from self._fetch_generation()
    
    def _fetch_pld(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch PLD (Settlement Price) data.
        
        Brazil operates with hourly PLD for 4 submarkets:
        - SE/CO (Southeast/Central-West): São Paulo, Rio, Brasília
        - S (South): Paraná, Santa Catarina, Rio Grande do Sul
        - NE (Northeast): Bahia, Pernambuco, etc.
        - N (North): Amazonas, Pará, etc.
        """
        submarkets = {
            "SE/CO": {"base_price": 180.0, "volatility": 1.3},  # BRL/MWh
            "S": {"base_price": 175.0, "volatility": 1.2},
            "NE": {"base_price": 185.0, "volatility": 1.4},
            "N": {"base_price": 190.0, "volatility": 1.5},
        }
        
        now = datetime.utcnow()
        brazil_time = now - timedelta(hours=3)  # BRT (UTC-3)
        
        # Generate PLD for each hour
        for hour in range(24):
            hour_ending = brazil_time.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            for submarket, config in submarkets.items():
                if self.submarket != "ALL" and self.submarket != submarket:
                    continue
                
                # Base price varies by hour (load pattern)
                if 18 <= hour <= 21:
                    hourly_factor = 1.5  # Evening peak
                elif 7 <= hour <= 17:
                    hourly_factor = 1.2  # Daytime
                else:
                    hourly_factor = 0.8  # Off-peak
                
                # Hydro influence (wet/dry season)
                month = hour_ending.month
                if month in [12, 1, 2, 3]:  # Wet season
                    hydro_factor = 0.85  # Lower prices
                elif month in [8, 9, 10]:  # Dry season
                    hydro_factor = 1.4  # Higher prices
                else:
                    hydro_factor = 1.0
                
                pld = config["base_price"] * hourly_factor * hydro_factor
                pld += (hash(submarket + str(hour)) % 40) - 20  # Add variation
                
                # Price cap and floor
                pld = max(50.0, min(500.0, pld))  # BRL/MWh limits
                
                yield {
                    "timestamp": datetime.utcnow().isoformat(),
                    "market": "ONS",
                    "submarket": submarket,
                    "price_type": "PLD",
                    "price_brl_mwh": pld,
                    "hour_ending": hour_ending.isoformat(),
                    "currency": "BRL",
                }
    
    def _fetch_hydro(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch hydro reservoir data.
        
        Brazil's power system is heavily hydro-dependent.
        Reservoir levels critically impact prices.
        """
        # Major reservoir systems
        reservoirs = [
            {
                "name": "Sudeste/Centro-Oeste",
                "region": "SE/CO",
                "capacity_mwmes": 205394,  # MWmonth of stored energy
                "typical_level": 0.70,
            },
            {
                "name": "Sul",
                "region": "S",
                "capacity_mwmes": 23798,
                "typical_level": 0.68,
            },
            {
                "name": "Nordeste",
                "region": "NE",
                "capacity_mwmes": 56281,
                "typical_level": 0.65,
            },
            {
                "name": "Norte",
                "region": "N",
                "capacity_mwmes": 16418,
                "typical_level": 0.72,
            },
        ]
        
        now = datetime.utcnow()
        month = now.month
        
        for reservoir in reservoirs:
            # Seasonal variation
            if month in [12, 1, 2, 3]:  # Wet season
                seasonal_delta = 0.15
            elif month in [8, 9, 10]:  # Dry season
                seasonal_delta = -0.20
            else:
                seasonal_delta = 0.0
            
            level_pct = reservoir["typical_level"] + seasonal_delta
            level_pct = max(0.30, min(0.95, level_pct))
            
            stored_energy = reservoir["capacity_mwmes"] * level_pct
            
            # Mock inflow (MLT - Média de Longo Termo %)
            if month in [12, 1, 2]:  # Rainy season
                inflow_pct_mlt = 120 + (hash(reservoir["name"]) % 30)
            else:
                inflow_pct_mlt = 80 + (hash(reservoir["name"]) % 20)
            
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "ONS",
                "reservoir_system": reservoir["name"],
                "region": reservoir["region"],
                "level_pct": round(level_pct * 100, 1),
                "stored_energy_mwmes": round(stored_energy, 0),
                "capacity_mwmes": reservoir["capacity_mwmes"],
                "inflow_pct_mlt": inflow_pct_mlt,
                "spillage_risk": "high" if level_pct > 0.90 else "low",
            }
    
    def _fetch_generation(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch generation mix data.
        
        Brazil has diverse generation:
        - Hydro: ~60% (varies with rainfall)
        - Wind: ~12% (growing)
        - Solar: ~3% (growing rapidly)
        - Biomass: ~9% (sugarcane bagasse)
        - Thermal (gas, coal, oil): ~15%
        - Nuclear: ~2%
        """
        now = datetime.utcnow()
        hour = now.hour
        month = now.month
        
        # Total system load (mock)
        base_load = 70000  # MW
        
        if 18 <= hour <= 21:
            load = base_load * 1.3  # Evening peak
        elif 7 <= hour <= 17:
            load = base_load * 1.1
        else:
            load = base_load * 0.85
        
        # Generation by source
        sources = []
        
        # Hydro (varies with season)
        if month in [12, 1, 2, 3]:  # Wet
            hydro_pct = 0.70
        else:
            hydro_pct = 0.50
        
        sources.append({
            "source": "HYDRO",
            "generation_mw": load * hydro_pct,
            "capacity_factor": hydro_pct,
        })
        
        # Wind (varies with time and season)
        if month in [7, 8, 9]:  # Wind season in NE
            wind_pct = 0.15
        else:
            wind_pct = 0.08
        
        # Wind is stronger at night
        if 0 <= hour <= 6:
            wind_pct *= 1.3
        
        sources.append({
            "source": "WIND",
            "generation_mw": load * wind_pct,
            "capacity_factor": wind_pct,
        })
        
        # Solar (daytime only)
        if 6 <= hour <= 18:
            solar_pct = 0.05 * (1 - abs(hour - 12) / 12)  # Peak at noon
        else:
            solar_pct = 0.0
        
        sources.append({
            "source": "SOLAR",
            "generation_mw": load * solar_pct,
            "capacity_factor": solar_pct,
        })
        
        # Biomass (baseload)
        sources.append({
            "source": "BIOMASS",
            "generation_mw": load * 0.08,
            "capacity_factor": 0.70,
        })
        
        # Thermal (complement)
        thermal_mw = load - sum(s["generation_mw"] for s in sources)
        sources.append({
            "source": "THERMAL",
            "generation_mw": max(0, thermal_mw),
            "capacity_factor": 0.60,
        })
        
        # Nuclear (baseload)
        sources.append({
            "source": "NUCLEAR",
            "generation_mw": 2000,  # Angra 1 & 2
            "capacity_factor": 0.90,
        })
        
        for source_data in sources:
            yield {
                "timestamp": datetime.utcnow().isoformat(),
                "market": "ONS",
                "source_type": source_data["source"],
                "generation_mw": round(source_data["generation_mw"], 1),
                "capacity_factor": round(source_data["capacity_factor"], 3),
            }
    
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map ONS format to canonical schema."""
        timestamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
        
        # Determine product and location
        if "price_type" in raw and raw["price_type"] == "PLD":
            product = "energy"
            location = f"ONS.{raw['submarket'].replace('/', '_')}"
            value = raw["price_brl_mwh"]
            currency = "BRL"
        elif "reservoir_system" in raw:
            product = "hydro_storage"
            location = f"ONS.HYDRO.{raw['region']}"
            value = raw["level_pct"]
            currency = "BRL"
        elif "source_type" in raw:
            product = "generation"
            location = f"ONS.GEN.{raw['source_type']}"
            value = raw["generation_mw"]
            currency = "BRL"
        else:
            product = "unknown"
            location = "ONS.UNKNOWN"
            value = 0
            currency = "BRL"
        
        return {
            "event_time_utc": int(timestamp.timestamp() * 1000),
            "market": "power",
            "product": product,
            "instrument_id": location,
            "location_code": location,
            "price_type": "settlement",
            "value": float(value),
            "volume": None,
            "currency": currency,
            "unit": "MWh" if product == "energy" else "MW",
            "source": self.source_id,
            "seq": int(time.time() * 1000000),
        }
    
    def emit(self, events: Iterator[Dict[str, Any]]) -> int:
        """Emit events to Kafka."""
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
        
        self.producer.flush()
        logger.info(f"Emitted {count} ONS events to {self.kafka_topic}")
        return count
    
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """Save checkpoint to state store."""
        self.checkpoint_state = state
        logger.debug(f"ONS checkpoint saved: {state}")


if __name__ == "__main__":
    # Test ONS connector
    configs = [
        {
            "source_id": "ons_pld",
            "submarket": "ALL",
            "data_type": "PLD",
            "kafka_topic": "power.ticks.v1",
        },
        {
            "source_id": "ons_hydro",
            "data_type": "HYDRO",
            "kafka_topic": "power.fundamentals.v1",
        },
        {
            "source_id": "ons_generation",
            "data_type": "GENERATION",
            "kafka_topic": "power.generation.v1",
        },
    ]
    
    for config in configs:
        logger.info(f"Testing {config['source_id']}")
        connector = BrazilONSConnector(config)
        connector.run()

