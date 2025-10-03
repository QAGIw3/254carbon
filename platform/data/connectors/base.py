"""
Base connector SDK for data ingestion.

All data source connectors implement the Ingestor interface.
"""
from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Ingestor(ABC):
    """
    Base class for all data source connectors.
    
    Connectors are plugins that pull data from external sources,
    map to canonical schema, and emit to Kafka.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize connector with configuration.
        
        Args:
            config: Connector-specific configuration dict
        """
        self.config = config
        self.source_id = config.get("source_id")
        self.checkpoint_state: Dict[str, Any] = {}
    
    @abstractmethod
    def discover(self) -> Dict[str, Any]:
        """
        Discover available data streams from the source.
        
        Returns:
            Dict with metadata about available streams:
            {
                "streams": [
                    {"name": "nodal_lmp", "fields": [...], "update_freq": "5min"},
                    ...
                ]
            }
        """
        pass
    
    @abstractmethod
    def pull_or_subscribe(self) -> Iterator[Dict[str, Any]]:
        """
        Pull or subscribe to data from source.
        
        For batch sources: pull data since last checkpoint.
        For streaming sources: subscribe and yield messages.
        
        Yields:
            Raw message dicts from the source
        """
        pass
    
    @abstractmethod
    def map_to_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map raw source data to canonical schema.
        
        Args:
            raw: Raw message from source
        
        Returns:
            Message dict conforming to canonical Avro schema
        """
        pass
    
    @abstractmethod
    def emit(self, events: Iterator[Dict[str, Any]]) -> int:
        """
        Emit events to Kafka topic.
        
        Args:
            events: Iterator of canonical schema messages
        
        Returns:
            Number of events emitted
        """
        pass
    
    @abstractmethod
    def checkpoint(self, state: Dict[str, Any]) -> None:
        """
        Save checkpoint state for resume/recovery.
        
        Args:
            state: State dict to persist (e.g., last_timestamp, last_offset)
        """
        pass
    
    def validate_event(self, event: Dict[str, Any]) -> bool:
        """
        Validate event against data quality rules.
        
        Args:
            event: Canonical schema event
        
        Returns:
            True if valid, False otherwise
        """
        # Basic validation
        required_fields = [
            "event_time_utc",
            "market",
            "product",
            "instrument_id",
            "value",
        ]
        
        for field in required_fields:
            if field not in event:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Value range check
        value = event.get("value")
        if value is not None and (value < -1000 or value > 100000):
            logger.warning(f"Value out of range: {value}")
            return False
        
        return True
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load last checkpoint state.
        
        Returns:
            Last checkpoint state dict or None
        """
        # TODO: Load from PostgreSQL checkpoint table
        return self.checkpoint_state
    
    def run(self) -> int:
        """
        Main execution loop: pull, map, validate, emit, checkpoint.
        
        Returns:
            Number of events processed
        """
        logger.info(f"Starting connector: {self.source_id}")
        
        # Load checkpoint
        last_state = self.load_checkpoint()
        if last_state:
            logger.info(f"Resuming from checkpoint: {last_state}")
        
        processed = 0
        batch = []
        batch_size = self.config.get("batch_size", 1000)
        
        try:
            for raw_event in self.pull_or_subscribe():
                # Map to canonical schema
                try:
                    canonical = self.map_to_schema(raw_event)
                except Exception as e:
                    logger.error(f"Mapping error: {e}")
                    continue
                
                # Validate
                if not self.validate_event(canonical):
                    continue
                
                batch.append(canonical)
                
                # Emit batch
                if len(batch) >= batch_size:
                    count = self.emit(iter(batch))
                    processed += count
                    
                    # Checkpoint
                    self.checkpoint({"last_event_time": canonical["event_time_utc"]})
                    
                    batch = []
            
            # Emit remaining
            if batch:
                count = self.emit(iter(batch))
                processed += count
        
        except Exception as e:
            logger.error(f"Connector error: {e}")
            raise
        
        logger.info(f"Connector finished: {processed} events processed")
        return processed

