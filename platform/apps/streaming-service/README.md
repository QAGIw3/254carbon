# Streaming Service

Real-time data streaming service for the 254Carbon platform.

## Overview

The Streaming Service handles all stateful, real-time streaming connections including WebSocket and Server-Sent Events (SSE). It consumes data from Kafka and fans it out to connected clients with efficient connection management.

## Features

- **WebSocket Streaming**: Full-duplex real-time price updates
- **Server-Sent Events**: HTTP-based streaming for browsers
- **Kafka Integration**: Consumes market data from Kafka topics
- **Connection Management**: Efficient handling of thousands of concurrent connections
- **Subscription Filtering**: Subscribe by instrument, commodity, or all updates
- **Mock Data**: Development mode with simulated price streams
- **Authentication**: JWT validation via Auth Service
- **Authorization**: Entitlement checks via Entitlements Service
- **High Memory**: Optimized for connection state management

## Architecture

```
┌─────────────┐
│   Kafka     │
│  (Topics)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│ Streaming   │────▶│ Auth Service │
│   Service   │     └──────────────┘
│ (Port 8001) │
└──────┬──────┘     ┌──────────────────┐
       │            │Entitlements Svc  │
       │            └──────────────────┘
       │
       ├─────▶ WebSocket Clients
       └─────▶ SSE Clients
```

## API Endpoints

### `GET /health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-05T12:00:00Z",
  "active_websockets": 45,
  "active_sse": 12
}
```

### `WebSocket /ws/stream`
WebSocket endpoint for real-time streaming.

**Connection Flow**:

1. Client connects to `ws://streaming-service:8001/ws/stream`
2. Client sends subscription message:
   ```json
   {
     "type": "subscribe",
     "instruments": ["MISO.LMP.INDIANA.HUB", "CAISO.LMP.SP15"],
     "commodities": ["natural_gas", "crude_oil"],
     "all": false,
     "api_key": "Bearer eyJhbGc..."
   }
   ```
3. Server responds with confirmation:
   ```json
   {
     "type": "subscribed",
     "instruments": ["MISO.LMP.INDIANA.HUB", "CAISO.LMP.SP15"],
     "commodities": ["natural_gas", "crude_oil"],
     "all": false,
     "message": "Subscribed to 2 instruments"
   }
   ```
4. Server streams price updates:
   ```json
   {
     "type": "price_update",
     "data": {
       "instrument_id": "MISO.LMP.INDIANA.HUB",
       "value": 35.42,
       "timestamp": "2025-10-05T12:00:00Z",
       "source": "kafka",
       "market": "power",
       "product": "lmp"
     }
   }
   ```

### `GET /sse/stream`
Server-Sent Events endpoint for HTTP streaming.

**Query Parameters**:
- `instruments`: List of instrument IDs
- `commodities`: List of commodity types
- `all`: Subscribe to all updates (boolean)
- `api_key`: JWT token for authentication

**Example**:
```bash
curl -N "http://streaming-service:8001/sse/stream?instruments=MISO.LMP.INDIANA.HUB&api_key=Bearer..."
```

**Response** (event-stream):
```
event: subscribed
data: {"instruments":["MISO.LMP.INDIANA.HUB"],"commodities":[],"all":false}

data: {"type":"price_update","data":{"instrument_id":"MISO.LMP.INDIANA.HUB","value":35.42,...}}

:ka
```

## Configuration

Environment variables:

- `PORT`: Service port (default: `8001`)
- `DATABASE_URL`: PostgreSQL connection string
- `CLICKHOUSE_HOST`: ClickHouse hostname
- `KAFKA_BOOTSTRAP`: Kafka bootstrap servers
- `ENABLE_KAFKA`: Enable Kafka consumer (true/false)
- `AUTH_SERVICE_URL`: Auth Service URL
- `ENTITLEMENTS_SERVICE_URL`: Entitlements Service URL
- `METRICS_SERVICE_URL`: Metrics Service URL
- `LOCAL_DEV`: Development mode (true/false)

## Deployment

### Docker

```bash
docker build -t 254carbon/streaming-service:latest .
docker run -p 8001:8001 \
  -e KAFKA_BOOTSTRAP=kafka:9092 \
  -e ENABLE_KAFKA=true \
  254carbon/streaming-service:latest
```

### Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
```

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (mock mode)
LOCAL_DEV=true uvicorn main:app --reload --port 8001

# Test WebSocket
wscat -c ws://localhost:8001/ws/stream
> {"type":"subscribe","instruments":["MISO.LMP.INDIANA.HUB"],"api_key":"dev-key"}

# Test SSE
curl -N "http://localhost:8001/sse/stream?instruments=MISO.LMP.INDIANA.HUB"
```

## Connection Management

### WebSocket
- Bidirectional maps: `connection ↔ subscriptions`
- Automatic cleanup on disconnect
- Supports instrument, commodity, and wildcard subscriptions

### SSE
- Uses `asyncio.Queue` for each connection
- Automatic cleanup on disconnect
- HTTP/2 compatible for multiplexing

## Performance

- **Memory**: 2-8GB per replica (stores connection state)
- **Connections**: Thousands per replica
- **Latency**: < 100ms from Kafka to client
- **Scaling**: Horizontal scaling with sticky sessions (optional)

## Monitoring

Metrics exported to Metrics Service:
- Active WebSocket connections
- Active SSE connections
- Message fanout rate
- Connection duration
- Kafka consumer lag

## Mock Data Mode

In `LOCAL_DEV=true` mode:
- Generates random price updates
- Updates every 5 seconds
- No Kafka required
- Useful for frontend development

## Production Mode

In production:
- Connects to Kafka
- Consumes from `market.price.ticks` and `market.fundamentals` topics
- Deserializes Avro messages
- Fans out to subscribed clients

## Related Services

- **API Gateway**: REST API (port 8000)
- **Auth Service**: Authentication (port 8010)
- **Entitlements Service**: Authorization (port 8011)
- **Metrics Service**: Metrics collection (port 8012)

