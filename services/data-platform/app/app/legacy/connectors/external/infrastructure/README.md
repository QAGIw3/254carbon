# Infrastructure Data Connectors

This module provides connectors for ingesting energy infrastructure data from various sources into the 254Carbon platform.

## Overview

The infrastructure connectors fetch and process data about:
- LNG terminals and storage facilities
- Power plants and generation assets
- Transmission lines and grid infrastructure
- Renewable energy resources and projects

## Data Sources

### 1. ALSI LNG Inventory (GIE)
- **Source**: Gas Infrastructure Europe (GIE) ALSI API
- **Data**: LNG terminal inventory, send-out, ship arrivals
- **Coverage**: European LNG terminals
- **Frequency**: Daily updates, fetched every 6 hours
- **API**: https://alsi.gie.eu/

### 2. REexplorer (NREL)
- **Source**: National Renewable Energy Laboratory
- **Data**: Solar/wind resource assessments, renewable projects
- **Coverage**: Global, with focus on US
- **Frequency**: Static resource data, weekly updates
- **API**: https://developer.nrel.gov/

### 3. Global Power Plant Database (WRI)
- **Source**: World Resources Institute
- **Data**: ~30,000 power plants worldwide
- **Coverage**: Global
- **Frequency**: Quarterly releases, monthly checks
- **URL**: https://datasets.wri.org/dataset/globalpowerplantdatabase

### 4. Global Energy Monitor Transmission
- **Source**: Global Energy Monitor
- **Data**: Transmission lines, substations, infrastructure projects
- **Coverage**: Global
- **Frequency**: Weekly updates
- **API**: https://api.globalenergymonitor.org/

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  External APIs  │────▶│  Connectors  │────▶│    Kafka    │
└─────────────────┘     └──────────────┘     └─────────────┘
                               │                      │
                               ▼                      ▼
                        ┌──────────────┐     ┌─────────────┐
                        │  PostgreSQL  │     │ ClickHouse  │
                        │ (Checkpoints)│     │(Time Series)│
                        └──────────────┘     └─────────────┘
```

## Configuration

Each connector is configured via YAML files or environment variables:

```yaml
source_id: alsi_lng_inventory
api_key: ${ALSI_API_KEY}
granularity: terminal  # terminal|country|eu
lookback_days: 30
kafka:
  topic: market.infrastructure
  bootstrap_servers: kafka:9092
database:
  host: postgresql
  port: 5432
  database: market_intelligence
```

## Usage

### Running Connectors

#### As Kubernetes CronJobs (Production)
```bash
kubectl apply -f infra/k8s/infrastructure/
```

#### Locally for Development
```bash
# Set environment variables
export ALSI_API_KEY=your_key
export POSTGRES_PASSWORD=postgres

# Run specific connector
python -m data.connectors.external.infrastructure alsi_lng

# Or run with config file
CONFIG_PATH=config/alsi.yaml python -m data.connectors.external.infrastructure alsi_lng
```

### Accessing Data

#### GraphQL API
```graphql
query {
  lngInventory(
    terminalId: "ES_BARCELONA"
    startTime: "2025-10-01T00:00:00Z"
    endTime: "2025-10-03T00:00:00Z"
  ) {
    ts
    inventoryGwh
    fullnessPct
    sendOutGwh
  }
}
```

#### REST API
```bash
# Get LNG terminals
curl http://api.254carbon.local/infrastructure/api/v1/infrastructure/lng-terminals?country=ES

# Get power plants
curl http://api.254carbon.local/infrastructure/api/v1/infrastructure/power-plants?fuel_type=wind&min_capacity_mw=100

# Get time series data
curl http://api.254carbon.local/infrastructure/api/v1/infrastructure/lng-inventory/ES_BARCELONA?start_date=2025-10-01
```

## Data Quality

The connectors implement comprehensive data quality checks:

- **Completeness**: Required fields validation
- **Validity**: Range checks for metrics
- **Consistency**: Cross-source reconciliation
- **Timeliness**: Data freshness monitoring
- **Accuracy**: Geographic and technical validation

Quality metrics are exposed via Prometheus:
```
connector_data_quality_score{source_id="alsi_lng_inventory"} 95.5
connector_validation_failures{source_id="wri_powerplants",validation_type="range"} 12
```

## Monitoring

### Prometheus Metrics
- `connector_up`: Connector health status
- `connector_records_processed`: Total records processed
- `connector_errors_total`: Error count by type
- `connector_last_event_time`: Timestamp of last processed event
- `infrastructure_assets_total`: Total assets by type and country

### Grafana Dashboard
Access the infrastructure monitoring dashboard at:
http://grafana.254carbon.local/d/infrastructure-monitoring

### Alerts
- Data staleness (>24h for daily sources)
- High error rates (>10% failure rate)
- API rate limits
- Low data quality scores (<80%)

## Development

### Adding a New Connector

1. Create connector class inheriting from `InfrastructureConnector`:
```python
class NewSourceConnector(InfrastructureConnector):
    def discover(self) -> Dict[str, Any]:
        # Return available data streams
        
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        # Fetch data from source
        
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        # Map to canonical schema
```

2. Add configuration in `infra/k8s/infrastructure/`

3. Register in `__main__.py`

4. Add tests in `tests/integration/`

### Testing

```bash
# Run unit tests
pytest tests/integration/test_infrastructure_connectors.py

# Run with coverage
pytest --cov=data.connectors.external.infrastructure tests/

# Test specific connector
pytest -k "TestALSILNGConnector"
```

## Troubleshooting

### Common Issues

1. **API Rate Limits**
   - Check `connector_api_rate_limit_hits` metric
   - Adjust request frequency in configuration
   - Implement exponential backoff

2. **Data Quality Issues**
   - Review `/api/v1/infrastructure/data-quality` endpoint
   - Check validation logs
   - Adjust tolerance thresholds if needed

3. **Checkpoint Failures**
   - Verify PostgreSQL connectivity
   - Check checkpoint table permissions
   - Review checkpoint history

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python -m data.connectors.external.infrastructure alsi_lng
```

## License

Copyright © 2025 254Carbon. All rights reserved.
