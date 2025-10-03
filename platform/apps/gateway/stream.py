"""
WebSocket stream manager for real-time data distribution.
"""
import asyncio
import logging
from typing import Dict, Set
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class StreamManager:
    """Manages WebSocket connections and real-time data streaming."""
    
    def __init__(self):
        # WebSocket -> subscribed instrument IDs
        self.connections: Dict[WebSocket, Set[str]] = {}
        # Instrument ID -> subscribed WebSockets
        self.subscriptions: Dict[str, Set[WebSocket]] = {}
        self._kafka_task = None
    
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
        # TODO: Implement Kafka consumer
        # This would connect to Kafka topics and broadcast ticks
        logger.info("Kafka consumer started (stub)")
        
        while True:
            await asyncio.sleep(1)
            # In production: consume from Kafka and call broadcast()
    
    async def shutdown(self):
        """Shutdown stream manager and close connections."""
        if self._kafka_task:
            self._kafka_task.cancel()
        
        for websocket in list(self.connections.keys()):
            try:
                await websocket.close()
            except Exception:
                pass
        
        self.connections.clear()
        self.subscriptions.clear()
        logger.info("Stream manager shut down")

