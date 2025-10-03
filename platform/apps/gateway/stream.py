"""
WebSocket stream manager for real-time data distribution.
"""
import asyncio
import json
import logging
from typing import Dict, Set, Optional
from fastapi import WebSocket
from aiokafka import AIOKafkaConsumer
import avro.schema
import avro.io
import io

logger = logging.getLogger(__name__)


class StreamManager:
    """Manages WebSocket connections and real-time data streaming."""

    def __init__(self):
        # WebSocket -> subscribed instrument IDs
        self.connections: Dict[WebSocket, Set[str]] = {}
        # Instrument ID -> subscribed WebSockets
        self.subscriptions: Dict[str, Set[WebSocket]] = {}
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
    
    async def register(self, websocket: WebSocket, instrument_ids: list[str]):
        """Register a WebSocket connection with subscriptions."""
        self.connections[websocket] = set(instrument_ids)
        
        for inst_id in instrument_ids:
            if inst_id not in self.subscriptions:
                self.subscriptions[inst_id] = set()
            self.subscriptions[inst_id].add(websocket)
        
        logger.info(f"Registered WebSocket with {len(instrument_ids)} subscriptions")
        
        # Start Kafka consumer if not running
        if self._kafka_task is None:
            self._kafka_task = asyncio.create_task(self._consume_kafka())
    
    async def unregister(self, websocket: WebSocket):
        """Unregister a WebSocket connection."""
        if websocket in self.connections:
            instrument_ids = self.connections[websocket]
            
            for inst_id in instrument_ids:
                if inst_id in self.subscriptions:
                    self.subscriptions[inst_id].discard(websocket)
                    if not self.subscriptions[inst_id]:
                        del self.subscriptions[inst_id]
            
            del self.connections[websocket]
            logger.info("Unregistered WebSocket")
    
    async def broadcast(self, instrument_id: str, data: dict):
        """Broadcast data to all subscribers of an instrument."""
        if instrument_id in self.subscriptions:
            disconnected = []
            
            for websocket in self.subscriptions[instrument_id]:
                try:
                    await websocket.send_json(data)
                except Exception as e:
                    logger.error(f"Error sending to WebSocket: {e}")
                    disconnected.append(websocket)
            
            # Clean up disconnected sockets
            for websocket in disconnected:
                await self.unregister(websocket)
    
    async def _consume_kafka(self):
        """Consume messages from Kafka and broadcast to subscribers."""
        logger.info(f"Starting Kafka consumer for topics: {self._kafka_topics}")

        try:
            self._kafka_consumer = AIOKafkaConsumer(
                *self._kafka_topics,
                bootstrap_servers=self._kafka_bootstrap_servers,
                group_id="gateway-stream-manager",
                auto_offset_reset="latest",
                value_deserializer=self._deserialize_avro_message,
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                max_poll_records=100
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
                            if instrument_id:
                                # Broadcast to all subscribers of this instrument
                                await self.broadcast(instrument_id, {
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

