"""
Infrastructure Connector Metrics
---------------------------------

Prometheus metrics for monitoring infrastructure data connectors.
"""

from prometheus_client import Counter, Gauge, Histogram, Info
import time
from functools import wraps
from typing import Any, Callable, Dict


# Connector health metrics
connector_up = Gauge(
    'connector_up',
    'Whether the connector is running (1) or not (0)',
    ['source_id']
)

connector_info = Info(
    'connector',
    'Connector information',
    ['source_id']
)

# Data ingestion metrics
records_processed = Counter(
    'connector_records_processed',
    'Total number of records processed',
    ['source_id', 'record_type']
)

events_emitted = Counter(
    'connector_events_emitted',
    'Total number of events emitted to Kafka',
    ['source_id', 'topic']
)

errors_total = Counter(
    'connector_errors_total',
    'Total number of errors encountered',
    ['source_id', 'error_type']
)

# Timing metrics
processing_duration = Histogram(
    'connector_processing_duration_seconds',
    'Time spent processing records',
    ['source_id', 'operation'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
)

api_request_duration = Histogram(
    'connector_api_request_duration_seconds',
    'Time spent making API requests',
    ['source_id', 'endpoint'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
)

# State metrics
last_successful_run = Gauge(
    'connector_last_successful_run',
    'Timestamp of last successful run',
    ['source_id']
)

last_event_time = Gauge(
    'connector_last_event_time',
    'Timestamp of last processed event',
    ['source_id']
)

checkpoint_lag_seconds = Gauge(
    'connector_checkpoint_lag_seconds',
    'Lag between current time and last checkpointed event',
    ['source_id']
)

# Data quality metrics
data_quality_score = Gauge(
    'connector_data_quality_score',
    'Data quality score (0-100)',
    ['source_id']
)

validation_failures = Counter(
    'connector_validation_failures',
    'Number of validation failures',
    ['source_id', 'validation_type']
)

missing_data_points = Counter(
    'connector_missing_data_points',
    'Number of missing or null data points',
    ['source_id', 'field']
)

# API rate limiting metrics
api_rate_limit_hits = Counter(
    'connector_api_rate_limit_hits',
    'Number of times API rate limit was hit',
    ['source_id', 'api']
)

api_requests_total = Counter(
    'connector_api_requests_total',
    'Total API requests made',
    ['source_id', 'api', 'status_code']
)

# Infrastructure-specific metrics
infrastructure_assets_total = Gauge(
    'infrastructure_assets_total',
    'Total number of infrastructure assets',
    ['asset_type', 'country', 'status']
)

lng_terminal_fullness_pct = Gauge(
    'lng_terminal_fullness_pct',
    'LNG terminal storage fullness percentage',
    ['terminal_id', 'country']
)

power_generation_mwh = Counter(
    'power_generation_mwh',
    'Power generation in MWh',
    ['plant_id', 'fuel_type', 'country']
)

transmission_utilization_pct = Gauge(
    'transmission_utilization_pct',
    'Transmission line utilization percentage',
    ['line_id', 'from_zone', 'to_zone']
)

renewable_resource_potential = Gauge(
    'renewable_resource_potential',
    'Renewable resource potential',
    ['resource_type', 'location', 'unit']
)

infrastructure_projects_total = Gauge(
    'infrastructure_projects_total',
    'Total infrastructure projects',
    ['project_type', 'status', 'country']
)


# Decorator for timing operations
def time_operation(operation: str):
    """Decorator to time an operation and record metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(self, *args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                processing_duration.labels(
                    source_id=getattr(self, 'source_id', 'unknown'),
                    operation=operation
                ).observe(duration)
        return wrapper
    return decorator


# Decorator for counting errors
def count_errors(error_type: str):
    """Decorator to count errors by type."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                errors_total.labels(
                    source_id=getattr(self, 'source_id', 'unknown'),
                    error_type=error_type
                ).inc()
                raise
        return wrapper
    return decorator


class MetricsCollector:
    """Helper class for collecting and updating metrics."""
    
    def __init__(self, source_id: str):
        self.source_id = source_id
        connector_up.labels(source_id=source_id).set(1)
        connector_info.labels(source_id=source_id).info({
            'source_id': source_id,
            'version': '1.0.0',
        })
    
    def record_processed(self, record_type: str, count: int = 1):
        """Record processed records."""
        records_processed.labels(
            source_id=self.source_id,
            record_type=record_type
        ).inc(count)
    
    def record_emitted(self, topic: str, count: int = 1):
        """Record emitted events."""
        events_emitted.labels(
            source_id=self.source_id,
            topic=topic
        ).inc(count)
    
    def record_error(self, error_type: str):
        """Record an error."""
        errors_total.labels(
            source_id=self.source_id,
            error_type=error_type
        ).inc()
    
    def update_last_successful_run(self):
        """Update last successful run timestamp."""
        last_successful_run.labels(
            source_id=self.source_id
        ).set_to_current_time()
    
    def update_last_event_time(self, timestamp: float):
        """Update last event timestamp."""
        last_event_time.labels(
            source_id=self.source_id
        ).set(timestamp)
        
        # Calculate and update lag
        current_time = time.time()
        lag = current_time - timestamp
        checkpoint_lag_seconds.labels(
            source_id=self.source_id
        ).set(lag)
    
    def update_data_quality(self, score: float):
        """Update data quality score."""
        data_quality_score.labels(
            source_id=self.source_id
        ).set(score)
    
    def record_validation_failure(self, validation_type: str):
        """Record a validation failure."""
        validation_failures.labels(
            source_id=self.source_id,
            validation_type=validation_type
        ).inc()
    
    def record_missing_data(self, field: str):
        """Record missing data point."""
        missing_data_points.labels(
            source_id=self.source_id,
            field=field
        ).inc()
    
    def record_api_request(self, api: str, status_code: int):
        """Record API request."""
        api_requests_total.labels(
            source_id=self.source_id,
            api=api,
            status_code=str(status_code)
        ).inc()
        
        if status_code == 429:
            api_rate_limit_hits.labels(
                source_id=self.source_id,
                api=api
            ).inc()
    
    def time_api_request(self, api: str, endpoint: str) -> 'ApiTimer':
        """Context manager for timing API requests."""
        return ApiTimer(self.source_id, api, endpoint)
    
    def update_infrastructure_metrics(self, metrics: Dict[str, Any]):
        """Update infrastructure-specific metrics."""
        
        # Update asset counts
        if 'asset_counts' in metrics:
            for key, count in metrics['asset_counts'].items():
                asset_type, country, status = key
                infrastructure_assets_total.labels(
                    asset_type=asset_type,
                    country=country,
                    status=status
                ).set(count)
        
        # Update LNG metrics
        if 'lng_fullness' in metrics:
            for terminal_id, data in metrics['lng_fullness'].items():
                lng_terminal_fullness_pct.labels(
                    terminal_id=terminal_id,
                    country=data['country']
                ).set(data['fullness_pct'])
        
        # Update power generation
        if 'power_generation' in metrics:
            for plant_id, data in metrics['power_generation'].items():
                power_generation_mwh.labels(
                    plant_id=plant_id,
                    fuel_type=data['fuel_type'],
                    country=data['country']
                ).inc(data['generation_mwh'])
        
        # Update transmission utilization
        if 'transmission_utilization' in metrics:
            for line_id, data in metrics['transmission_utilization'].items():
                transmission_utilization_pct.labels(
                    line_id=line_id,
                    from_zone=data['from_zone'],
                    to_zone=data['to_zone']
                ).set(data['utilization_pct'])
        
        # Update project counts
        if 'project_counts' in metrics:
            for key, count in metrics['project_counts'].items():
                project_type, status, country = key
                infrastructure_projects_total.labels(
                    project_type=project_type,
                    status=status,
                    country=country
                ).set(count)
    
    def close(self):
        """Clean up metrics on connector shutdown."""
        connector_up.labels(source_id=self.source_id).set(0)


class ApiTimer:
    """Context manager for timing API requests."""
    
    def __init__(self, source_id: str, api: str, endpoint: str):
        self.source_id = source_id
        self.api = api
        self.endpoint = endpoint
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            api_request_duration.labels(
                source_id=self.source_id,
                endpoint=f"{self.api}:{self.endpoint}"
            ).observe(duration)
