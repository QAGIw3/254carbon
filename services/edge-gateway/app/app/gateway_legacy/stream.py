"""
WebSocket stream manager for real-time data distribution.
"""
import asyncio
import json
import logging
from typing import Dict, Set, Optional, List
from fastapi import WebSocket
import os
if os.getenv("ENABLE_KAFKA", "false").lower() == "true":
    try:
        from aiokafka import AIOKafkaConsumer  # type: ignore
    except Exception:  # pragma: no cover
        AIOKafkaConsumer = None
else:
    AIOKafkaConsumer = None
import avro.schema
import avro.io
import io

logger = logging.getLogger(__name__)


class StreamManager:
    """Manage real-time data streaming to WebSocket and SSE subscribers.

    Maintains bidirectional maps of connections/subscriptions and runs a
    Kafka consumer loop to fan out Avro-encoded messages to subscribers.
    Supports subscriptions by instrument, commodity, and wildcard (all).
    """

    def __init__(self):
        # WebSocket -> subscribed instrument IDs
        self.connections: Dict[WebSocket, Set[str]] = {}
        # Instrument ID -> subscribed WebSockets
        self.subscriptions: Dict[str, Set[WebSocket]] = {}
        # Commodity type -> subscribed WebSockets
        self.commodity_ws_subscriptions: Dict[str, Set[WebSocket]] = {}
        # Wildcard (all) WebSocket subscribers
        self.all_ws_subscribers: Set[WebSocket] = set()

        # SSE (HTTP) queue subscribers
        # Instrument ID -> set of asyncio.Queue objects
        self.instrument_queues: Dict[str, Set[asyncio.Queue]] = {}
        # Commodity type -> set of asyncio.Queue objects
        self.commodity_queues: Dict[str, Set[asyncio.Queue]] = {}
        # Wildcard (all) SSE queue subscribers
        self.all_queues: Set[asyncio.Queue] = set()
        self._kafka_task = None
        self._kafka_consumer: Optional[AIOKafkaConsumer] = None
        self._kafka_bootstrap_servers = "kafka:9092"
        self._kafka_topics = ["market.price.ticks", "market.fundamentals"]

        # Avro schema for deserializing messages
        self._avro_schema = avro.schema.parse("""
        {
            "type": "record",
            "name": "MarketTick",
            "fields": [
                {"name": "event_time", "type": "string"},
                {"name": "instrument_id", "type": "string"},
                {"name": "location_code", "type": "string"},
                {"name": "price_type", "type": "string"},
                {"name": "value", "type": "double"},
                {"name": "volume", "type": ["null", "double"]},
                {"name": "currency", "type": "string"},
                {"name": "unit", "type": "string"},
                {"name": "source", "type": "string"}
            ]
        }
        """)
    
    async def register(self, websocket: WebSocket, instrument_ids: List[str], commodity_types: Optional[List[str]] = None, subscribe_all: bool = False):
        """Register a WebSocket connection with subscriptions.

        Args:
            websocket: FastAPI WebSocket instance.
            instrument_ids: List of instrument IDs to subscribe to.
            commodity_types: Optional list of commodity types to subscribe to.
            subscribe_all: If True, receive all updates.
        """
        self.connections[websocket] = set(instrument_ids)

        for inst_id in instrument_ids:
            if inst_id not in self.subscriptions:
                self.subscriptions[inst_id] = set()
            self.subscriptions[inst_id].add(websocket)

        if commodity_types:
            for c in commodity_types:
                if c not in self.commodity_ws_subscriptions:
                    self.commodity_ws_subscriptions[c] = set()
                self.commodity_ws_subscriptions[c].add(websocket)

        if subscribe_all:
            self.all_ws_subscribers.add(websocket)

        logger.info(
            f"Registered WebSocket with {len(instrument_ids)} instrument subscriptions, "
            f"{len(commodity_types or [])} commodity subscriptions, all={subscribe_all}"
        )
        
        # Start Kafka consumer if not running
        if self._kafka_task is None:
            self._kafka_task = asyncio.create_task(self._consume_kafka())
    
    async def unregister(self, websocket: WebSocket):
        """Unregister a WebSocket connection and clean up subscriptions."""
        if websocket in self.connections:
            instrument_ids = self.connections[websocket]

            for inst_id in instrument_ids:
                if inst_id in self.subscriptions:
                    self.subscriptions[inst_id].discard(websocket)
                    if not self.subscriptions[inst_id]:
                        del self.subscriptions[inst_id]

            # Remove from commodity and all maps
            for cset in list(self.commodity_ws_subscriptions.values()):
                cset.discard(websocket)
            self.all_ws_subscribers.discard(websocket)

            del self.connections[websocket]
            logger.info("Unregistered WebSocket")
    
    async def broadcast(self, instrument_id: str, data: dict):
        """Broadcast data to all subscribers of an instrument.

        Args:
            instrument_id: Key for subscriber lookup.
            data: JSON-serializable payload to send.
        """
        # WebSocket by instrument
        if instrument_id in self.subscriptions:
            disconnected = []

            for websocket in self.subscriptions[instrument_id]:
                try:
                    await websocket.send_json(data)
                except Exception as e:
                    logger.error(f"Error sending to WebSocket: {e}")
                    disconnected.append(websocket)

            for websocket in disconnected:
                await self.unregister(websocket)

        # SSE queues by instrument
        if instrument_id in self.instrument_queues:
            for q in list(self.instrument_queues[instrument_id]):
                try:
                    q.put_nowait(data)
                except Exception:
                    # Drop if receiver is slow/full
                    pass

        # Wildcard recipients (WebSocket and SSE)
        for websocket in list(self.all_ws_subscribers):
            try:
                await websocket.send_json(data)
            except Exception:
                await self.unregister(websocket)
        for q in list(self.all_queues):
            try:
                q.put_nowait(data)
            except Exception:
                pass

    async def broadcast_commodity(self, commodity_type: Optional[str], data: dict):
        """Broadcast data to commodity-based subscribers if type provided."""
        if not commodity_type:
            return

        # WebSocket by commodity
        if commodity_type in self.commodity_ws_subscriptions:
            disconnected = []
            for websocket in self.commodity_ws_subscriptions[commodity_type]:
                try:
                    await websocket.send_json(data)
                except Exception as e:
                    logger.error(f"Error sending to WebSocket: {e}")
                    disconnected.append(websocket)
            for websocket in disconnected:
                await self.unregister(websocket)

        # SSE queues by commodity
        if commodity_type in self.commodity_queues:
            for q in list(self.commodity_queues[commodity_type]):
                try:
                    q.put_nowait(data)
                except Exception:
                    pass

    async def register_http(self, queue: asyncio.Queue, instrument_ids: List[str], commodity_types: Optional[List[str]] = None, subscribe_all: bool = False):
        """Register an HTTP (SSE) subscriber represented by an asyncio.Queue."""
        for inst_id in instrument_ids:
            if inst_id not in self.instrument_queues:
                self.instrument_queues[inst_id] = set()
            self.instrument_queues[inst_id].add(queue)

        if commodity_types:
            for c in commodity_types:
                if c not in self.commodity_queues:
                    self.commodity_queues[c] = set()
                self.commodity_queues[c].add(queue)

        if subscribe_all:
            self.all_queues.add(queue)

    async def unregister_http(self, queue: asyncio.Queue):
        """Unregister an HTTP (SSE) subscriber queue."""
        for inst_id in list(self.instrument_queues.keys()):
            self.instrument_queues[inst_id].discard(queue)
            if not self.instrument_queues[inst_id]:
                del self.instrument_queues[inst_id]
        for c in list(self.commodity_queues.keys()):
            self.commodity_queues[c].discard(queue)
            if not self.commodity_queues[c]:
                del self.commodity_queues[c]
        self.all_queues.discard(queue)
    
    async def _consume_kafka(self):
        """Consume messages from Kafka and broadcast to subscribers."""
        logger.info(f"Starting Kafka consumer for topics: {self._kafka_topics}")

        try:
            if os.getenv("ENABLE_KAFKA", "false").lower() != "true" or AIOKafkaConsumer is None:
                raise RuntimeError("Kafka disabled or aiokafka unavailable")

            self._kafka_consumer = AIOKafkaConsumer(
                *self._kafka_topics,
                bootstrap_servers=self._kafka_bootstrap_servers,
                group_id="gateway-stream-manager",
                auto_offset_reset="latest",
                value_deserializer=self._deserialize_avro_message,
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                max_poll_records=1000,
                fetch_max_bytes=134217728,  # 128 MiB
                max_partition_fetch_bytes=16777216,  # 16 MiB
                partition_assignment_strategy=("cooperative-sticky",)
            )

            await self._kafka_consumer.start()

            logger.info("Kafka consumer connected successfully")

            try:
                async for message in self._kafka_consumer:
                    try:
                        # Parse the deserialized Avro message
                        if message.value:
                            data = message.value

                            # Extract instrument_id for routing
                            instrument_id = data.get("instrument_id")
                            commodity_type = data.get("commodity_type")
                            if instrument_id:
                                # Broadcast to all subscribers of this instrument
                                await self.broadcast(instrument_id, {
                                    "type": "price_tick",
                                    "data": data,
                                    "timestamp": data.get("event_time")
                                })
                                # Also broadcast to commodity subscribers if available
                                await self.broadcast_commodity(commodity_type, {
                                    "type": "price_tick",
                                    "data": data,
                                    "timestamp": data.get("event_time")
                                })
                            else:
                                logger.warning(f"Received message without instrument_id: {data}")

                    except Exception as e:
                        logger.error(f"Error processing Kafka message: {e}")

            except asyncio.CancelledError:
                logger.info("Kafka consumer cancelled")
                raise
            except Exception as e:
                logger.error(f"Kafka consumer error: {e}")
                raise

        except Exception as e:
            logger.error(f"Failed to start Kafka consumer: {e}")
            # For development, continue without Kafka if it's not available
            logger.warning("Kafka consumer failed, continuing without real-time data")

        finally:
            if self._kafka_consumer:
                await self._kafka_consumer.stop()
                logger.info("Kafka consumer stopped")

    def _deserialize_avro_message(self, message_bytes):
        """Deserialize Avro message from Kafka."""
        try:
            if not message_bytes:
                return None

            # Create a BytesIO object from the message bytes
            bytes_io = io.BytesIO(message_bytes)

            # Create decoder
            decoder = avro.io.BinaryDecoder(bytes_io)

            # Create datum reader
            reader = avro.io.DatumReader(self._avro_schema)

            # Read the message
            message = reader.read(decoder)

            return message

        except Exception as e:
            logger.error(f"Error deserializing Avro message: {e}")
            return None
    
    async def shutdown(self):
        """Shutdown stream manager and close connections."""
        logger.info("Shutting down stream manager...")

        # Stop Kafka consumer
        if self._kafka_consumer:
            await self._kafka_consumer.stop()
            logger.info("Kafka consumer stopped")

        # Cancel Kafka task
        if self._kafka_task:
            self._kafka_task.cancel()
            try:
                await self._kafka_task
            except asyncio.CancelledError:
                pass
            logger.info("Kafka task cancelled")

        # Close all WebSocket connections
        for websocket in list(self.connections.keys()):
            try:
                await websocket.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")

        self.connections.clear()
        self.subscriptions.clear()
        logger.info("Stream manager shut down")
