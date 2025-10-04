# 254Carbon Python SDK

Official Python client library for the 254Carbon Market Intelligence Platform.

## Installation

```bash
pip install carbon254
```

## Quick Start

```python
from carbon254 import CarbonClient
from datetime import datetime, timedelta

# Initialize client
client = CarbonClient(api_key="your_api_key")

# Get instruments
instruments = client.get_instruments(market="power")
print(f"Found {len(instruments)} power instruments")

# Get historical prices as DataFrame
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=7)

df = client.get_prices_dataframe(
    instrument_id="MISO.HUB.INDIANA",
    start_time=start_time,
    end_time=end_time,
)

print(df.head())

# Get forward curve
from datetime import date

curve_df = client.get_curve_dataframe(
    instrument_id="PJM.HUB.WEST",
    as_of_date=date.today(),
    scenario_id="BASE",
)

print(curve_df.head())
```

## Features

- **Type-safe API client** with Pydantic models
- **Pandas integration** for easy data analysis
- **Async support** for high-performance applications
- **WebSocket streaming** for real-time price updates
- **Advanced analytics** (VaR, correlation, forecasting)
- **Local development mode** with mock data fallbacks
- **Automatic retry** and error handling
- **Rate limiting** compliance

## Usage Examples

### Working with Instruments

```python
# Get all power instruments
instruments = client.get_instruments(market="power")

# Filter by product
lmp_instruments = client.get_instruments(
    market="power",
    product="lmp",
)

# Get instrument details
for inst in lmp_instruments[:5]:
    print(f"{inst.instrument_id}: {inst.location_code}")
```

### Historical Price Analysis

```python
import pandas as pd
from datetime import datetime, timedelta

# Get 30 days of hourly data
df = client.get_prices_dataframe(
    instrument_id="MISO.HUB.INDIANA",
    start_time=datetime.utcnow() - timedelta(days=30),
    end_time=datetime.utcnow(),
)

# Calculate statistics
print(f"Average price: ${df['price'].mean():.2f}/MWh")
print(f"Max price: ${df['price'].max():.2f}/MWh")
print(f"Std dev: ${df['price'].std():.2f}/MWh")

# Resample to daily
daily_avg = df['price'].resample('D').mean()
print(daily_avg)
```

### Forward Curve Analysis

```python
from datetime import date

# Get forward curve
curve = client.get_curve_dataframe(
    instrument_id="PJM.HUB.WEST",
    as_of_date=date.today(),
)

# Plot curve
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(curve.index, curve['price'])
plt.title("PJM West Hub Forward Curve")
plt.xlabel("Delivery Period")
plt.ylabel("Price ($/MWh)")
plt.grid(True)
plt.show()
```

### Scenario Modeling

```python
# Create custom scenario
scenario_id = client.create_scenario(
    title="High Load Growth",
    description="2.5% annual load growth scenario",
    assumptions={
        "as_of_date": "2025-10-03",
        "power": {
            "load_growth": {"PJM": 2.5, "MISO": 2.0}
        },
        "fuels": {
            "gas": {"curve": "HENRY", "shockbps": +50}
        }
    }
)

# Execute scenario
run_id = client.run_scenario(scenario_id)

# Wait for completion
import time
while True:
    status = client.get_run_status(scenario_id, run_id)
    if status['status'] in ['success', 'failed']:
        break
    time.sleep(10)

# Get scenario curves
curve = client.get_curve_dataframe(
    instrument_id="PJM.HUB.WEST",
    as_of_date=date.today(),
    scenario_id=scenario_id,
)
```

### Real-time Streaming

```python
import asyncio
from carbon254 import CarbonClient

def handle_price_update(price_tick):
    print(f"📈 {price_tick.instrument_id}: ${price_tick.value:.2f}")

async def stream_example():
    client = CarbonClient(local_dev=True)

    # Stream real-time prices
    instruments = ["MISO.HUB.INDIANA", "PJM.HUB.WEST"]
    await client.stream_prices(instruments, handle_price_update)

    client.close()

# Run streaming
asyncio.run(stream_example())
```

### Advanced Analytics

```python
# Portfolio Value at Risk
portfolio = [
    {"instrument_id": "MISO.HUB.INDIANA", "quantity": 1000},
    {"instrument_id": "PJM.HUB.WEST", "quantity": -500}
]

var_result = client.get_portfolio_var(portfolio, confidence_level=0.95)
print(f"Portfolio VaR: ${var_result['total_var']:.2f}")

# Correlation Analysis
instruments = ["MISO.HUB.INDIANA", "PJM.HUB.WEST", "CAISO.SP15"]
corr_matrix = client.get_correlation_matrix(
    instruments,
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now()
)
print(corr_matrix)

# Price Forecasting
forecast = client.get_price_forecast("MISO.HUB.INDIANA", horizon_days=30)
print(f"30-day forecast average: ${forecast['forecast_price'].mean():.2f}/MWh")
```

### Local Development Mode

```python
# Enable local development with mock data fallbacks
client = CarbonClient(local_dev=True)  # Uses localhost:8000 and mock data

# No API key required for basic functionality
# Automatic fallback to realistic mock data when API unavailable
```

### Async Usage

```python
import asyncio
from carbon254 import CarbonClient

async def fetch_multiple_instruments():
    client = CarbonClient(api_key="your_api_key")
    
    # Fetch instruments asynchronously
    instruments = await client.get_instruments_async(market="power")
    print(f"Found {len(instruments)} instruments")
    
    client.close()

asyncio.run(fetch_multiple_instruments())
```

### Context Manager

```python
# Automatic resource cleanup
with CarbonClient(api_key="your_api_key") as client:
    instruments = client.get_instruments(market="power")
    # Client automatically closed on exit
```

## Error Handling

```python
from carbon254 import (
    CarbonClient,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
)

client = CarbonClient(api_key="your_api_key")

try:
    instruments = client.get_instruments()
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded - slow down!")
except NotFoundError:
    print("Resource not found")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## API Reference

### CarbonClient

Main client class for interacting with the 254Carbon API.

**Core Methods:**

- `get_instruments(market=None, product=None)` - Get available instruments
- `get_prices(instrument_id, start_time, end_time, price_type="mid")` - Get price ticks
- `get_prices_dataframe(...)` - Get prices as pandas DataFrame
- `get_forward_curve(instrument_id, as_of_date, scenario_id="BASE")` - Get forward curve
- `get_curve_dataframe(...)` - Get curve as pandas DataFrame
- `create_scenario(title, description, assumptions)` - Create scenario
- `run_scenario(scenario_id)` - Execute scenario
- `get_run_status(scenario_id, run_id)` - Get run status

**Streaming Methods:**

- `stream_prices(instrument_ids, callback, reconnect=True)` - Stream real-time price updates
- `stream_prices_sync(instrument_ids, callback)` - Synchronous price streaming

**Advanced Analytics:**

- `get_portfolio_var(positions, confidence_level=0.95, method="historical")` - Portfolio VaR
- `get_correlation_matrix(instruments, start_date, end_date, window=30)` - Correlation matrix
- `get_price_forecast(instrument_id, horizon_days=30, model_type="ensemble")` - Price forecasts

## Configuration

### Environment Variables

```bash
# Production
export CARBON254_API_KEY="your_api_key"
export CARBON254_BASE_URL="https://api.254carbon.ai"  # Optional

# Local Development (default when no API key provided)
export CARBON254_LOCAL_DEV="true"  # Enable local dev mode
export CARBON254_API_URL="http://localhost:8000"  # Local API URL
export CARBON254_API_KEY="dev-key"  # Development key
```

### Client Configuration

```python
import os
from carbon254 import CarbonClient

# Production client
client = CarbonClient(api_key="your_api_key")

# Local development client (auto-detects localhost)
client = CarbonClient(local_dev=True)  # Uses localhost:8000

# Manual configuration
client = CarbonClient(
    api_key="your_key",
    base_url="https://api.254carbon.ai",
    local_dev=False
)

# Environment variable configuration
client = CarbonClient(api_key=os.getenv("CARBON254_API_KEY"))
```

## Examples

### Basic Examples
- `examples/streaming_example.py` - Real-time price streaming demo
- `examples/analytics_example.py` - Advanced analytics (VaR, correlation, forecasting)

Run examples:
```bash
cd examples/
python streaming_example.py
python analytics_example.py
```

## Development

### Running Tests

```bash
pip install -r requirements.txt
pytest tests/
```

### Code Formatting

```bash
black carbon254/
```

### Type Checking

```bash
mypy carbon254/
```

### Local Development Setup

1. **Start the 254Carbon platform:**
   ```bash
   cd ../platform
   ./scripts/dev-setup.sh
   ```

2. **Install SDK in development mode:**
   ```bash
   pip install -e .
   ```

3. **Run examples:**
   ```bash
   python examples/streaming_example.py
   python examples/analytics_example.py
   ```

## Support

- **Documentation**: https://docs.254carbon.ai
- **Email**: sdk@254carbon.ai
- **Issues**: https://github.com/254carbon/python-sdk/issues

## License

MIT License - see LICENSE file for details

## Exports and Streaming

- Exports
```python
from carbon254 import CarbonClient
from datetime import date
client = CarbonClient()
job = client.create_export_job(dataset_id="energy_prices", start_date=date(2024,1,1), end_date=date(2024,1,31), fmt="parquet")
status = client.get_export_status(job["job_id"])  # poll until completed
url = client.get_export_download_url(job["job_id"])  # presigned URL
```

- WebSocket streaming
```python
async for tick in client.stream_prices(["MISO.HUB.INDIANA"], commodities=["oil"], subscribe_all=False, callback=None):
    print(tick)
```

- SSE streaming
```python
for event in client.stream_prices_sse(instruments=["MISO.HUB.INDIANA"], commodities=["oil"]):
    print(event)
```

