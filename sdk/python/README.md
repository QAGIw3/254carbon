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
- **Automatic retry** and error handling
- **Rate limiting** compliance
- **Streaming support** via WebSocket (coming soon)

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

**Methods:**

- `get_instruments(market=None, product=None)` - Get available instruments
- `get_prices(instrument_id, start_time, end_time, price_type="mid")` - Get price ticks
- `get_prices_dataframe(...)` - Get prices as pandas DataFrame
- `get_forward_curve(instrument_id, as_of_date, scenario_id="BASE")` - Get forward curve
- `get_curve_dataframe(...)` - Get curve as pandas DataFrame
- `create_scenario(title, description, assumptions)` - Create scenario
- `run_scenario(scenario_id)` - Execute scenario
- `get_run_status(scenario_id, run_id)` - Get run status

## Configuration

### Environment Variables

```bash
export CARBON254_API_KEY="your_api_key"
export CARBON254_BASE_URL="https://api.254carbon.ai"  # Optional
```

```python
import os
from carbon254 import CarbonClient

# Client will read from environment
client = CarbonClient(api_key=os.getenv("CARBON254_API_KEY"))
```

## Development

### Running Tests

```bash
pip install carbon254[dev]
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

## Support

- **Documentation**: https://docs.254carbon.ai
- **Email**: sdk@254carbon.ai
- **Issues**: https://github.com/254carbon/python-sdk/issues

## License

MIT License - see LICENSE file for details

