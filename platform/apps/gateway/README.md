### API Gateway Extensions

- New commodity endpoints
  - GET `/api/v1/commodities/{commodity}/prices`
  - GET `/api/v1/commodities/{commodity}/curves`
  - GET `/api/v1/commodities/{commodity}/benchmarks`
- Research bulk export
  - POST `/api/v1/research/export/jobs`
  - GET `/api/v1/research/export/jobs/{job_id}`
  - GET `/api/v1/research/export/jobs/{job_id}/download`
  - GET `/api/v1/research/export/preview`
- Real-time streaming
  - WS: `/api/v1/stream` subscribe with `{type:"subscribe", instruments:[...], commodities:[...], all:false}`
  - SSE: GET `/api/v1/stream/sse?instruments=...&commodities=...&all=false`

Example (Python SDK):
```python
from carbon254 import CarbonClient
from datetime import datetime

client = CarbonClient(local_dev=True)

# Commodity prices
prices = client._client.get(
    "/api/v1/commodities/WTI/prices",
    params={"start_time": "2024-01-01T00:00:00Z", "end_time": "2024-01-02T00:00:00Z"}
).json()

# Export job
job = client.create_export_job(dataset_id="energy_prices", start_date=datetime(2024,1,1).date(), end_date=datetime(2024,1,2).date(), fmt="parquet")
status = client.get_export_status(job["job_id"])  # poll until completed
url = client.get_export_download_url(job["job_id"]) if status["status"] == "completed" else None

# WS streaming
async def run():
    async for tick in client.stream_prices(["MISO.HUB.INDIANA"], commodities=["oil"], subscribe_all=False, callback=None):
        print(tick)
```
