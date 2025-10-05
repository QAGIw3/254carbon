"""
Infrastructure connector main entry point.
"""

import sys
import logging
import yaml
import os
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load configuration from file or environment."""
    config_path = os.environ.get('CONFIG_PATH', 'config.yaml')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Fallback to environment variables
        config = {
            'source_id': os.environ.get('SOURCE_ID', 'infrastructure_connector'),
            'kafka': {
                'topic': os.environ.get('KAFKA_TOPIC', 'market.infrastructure'),
                'bootstrap_servers': os.environ.get('KAFKA_BOOTSTRAP', 'kafka:9092'),
            },
            'database': {
                'host': os.environ.get('POSTGRES_HOST', 'postgresql'),
                'port': int(os.environ.get('POSTGRES_PORT', '5432')),
                'database': os.environ.get('POSTGRES_DB', 'market_intelligence'),
                'user': os.environ.get('POSTGRES_USER', 'postgres'),
                'password': os.environ.get('POSTGRES_PASSWORD', 'postgres'),
            }
        }
    
    # Override with environment variables
    if 'POSTGRES_PASSWORD' in os.environ:
        config.setdefault('database', {})['password'] = os.environ['POSTGRES_PASSWORD']
    
    return config


def main():
    """Main entry point for running connectors."""
    if len(sys.argv) < 2:
        print("Usage: python -m data.connectors.external.infrastructure <connector_name>")
        print("Available connectors: alsi_lng, reexplorer, wri_powerplants, gem_transmission")
        sys.exit(1)
    
    connector_name = sys.argv[1]
    config = load_config()
    
    try:
        if connector_name == "alsi_lng":
            from .alsi_lng_connector import ALSILNGConnector
            config['api_key'] = os.environ.get('ALSI_API_KEY')
            connector = ALSILNGConnector(config)
            
        elif connector_name == "reexplorer":
            from .reexplorer_connector import REExplorerConnector
            config['api_key'] = os.environ.get('NREL_API_KEY')
            connector = REExplorerConnector(config)
            
        elif connector_name == "wri_powerplants":
            from .wri_powerplants_connector import WRIPowerPlantsConnector
            connector = WRIPowerPlantsConnector(config)
            
        elif connector_name == "gem_transmission":
            from .gem_transmission_connector import GEMTransmissionConnector
            config['api_key'] = os.environ.get('GEM_API_KEY')
            connector = GEMTransmissionConnector(config)
            
        else:
            logger.error(f"Unknown connector: {connector_name}")
            sys.exit(1)
        
        # Run the connector
        logger.info(f"Starting {connector_name} connector...")
        processed = connector.run()
        logger.info(f"Connector completed. Processed {processed} events.")
        
    except Exception as e:
        logger.error(f"Connector failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
