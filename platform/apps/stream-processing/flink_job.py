"""
Apache Flink Stream Processing Job

Real-time complex event processing, anomaly detection,
and sliding window aggregations for market data.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Iterator

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.window import TumblingEventTimeWindows, SlidingEventTimeWindows
from pyflink.common import Time, WatermarkStrategy
from pyflink.datastream.functions import MapFunction, FilterFunction, KeyedProcessFunction
from pyflink.common.typeinfo import Types
from pyflink.common.watermark_strategy import TimestampAssigner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceAnomalyDetector(KeyedProcessFunction):
    """
    Detect price anomalies using statistical methods.
    
    Identifies prices that deviate significantly from recent patterns.
    """
    
    def __init__(self, threshold_std: float = 3.0):
        self.threshold_std = threshold_std
        self.price_history = {}
    
    def process_element(self, value, ctx):
        """Process each price event."""
        key = ctx.get_current_key()
        price = value['value']
        
        # Initialize history
        if key not in self.price_history:
            self.price_history[key] = []
        
        # Add to history
        self.price_history[key].append(price)
        
        # Keep only recent prices (last 100)
        if len(self.price_history[key]) > 100:
            self.price_history[key].pop(0)
        
        # Calculate statistics
        if len(self.price_history[key]) >= 10:
            mean = sum(self.price_history[key]) / len(self.price_history[key])
            variance = sum((x - mean) ** 2 for x in self.price_history[key]) / len(self.price_history[key])
            std = variance ** 0.5
            
            # Check for anomaly
            z_score = abs(price - mean) / std if std > 0 else 0
            
            if z_score > self.threshold_std:
                # Emit anomaly
                anomaly = {
                    'timestamp': value['timestamp'],
                    'instrument_id': value['instrument_id'],
                    'price': price,
                    'mean': mean,
                    'std': std,
                    'z_score': z_score,
                    'anomaly_type': 'PRICE_SPIKE' if price > mean else 'PRICE_DROP',
                    'severity': 'HIGH' if z_score > 5 else 'MEDIUM',
                }
                yield anomaly


class PriceSpreadCalculator(KeyedProcessFunction):
    """
    Calculate spreads between markets in real-time.
    
    Monitors price differences for arbitrage opportunities.
    """
    
    def __init__(self, spread_threshold: float = 10.0):
        self.spread_threshold = spread_threshold
        self.latest_prices = {}
    
    def process_element(self, value, ctx):
        """Process each price event."""
        instrument_id = value['instrument_id']
        price = value['value']
        timestamp = value['timestamp']
        
        # Store latest price
        self.latest_prices[instrument_id] = {
            'price': price,
            'timestamp': timestamp
        }
        
        # Calculate spreads with other markets
        for other_id, other_data in self.latest_prices.items():
            if other_id != instrument_id:
                spread = price - other_data['price']
                
                if abs(spread) > self.spread_threshold:
                    # Emit spread alert
                    alert = {
                        'timestamp': timestamp,
                        'instrument_1': instrument_id,
                        'instrument_2': other_id,
                        'price_1': price,
                        'price_2': other_data['price'],
                        'spread': spread,
                        'spread_pct': (spread / other_data['price'] * 100) if other_data['price'] != 0 else 0,
                        'alert_type': 'SPREAD_ALERT',
                    }
                    yield alert


class MarketCorrelationTracker(KeyedProcessFunction):
    """
    Track correlations between markets in sliding windows.
    
    Identifies correlation regime changes.
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.price_windows = {}
    
    def process_element(self, value, ctx):
        """Process each price event."""
        market = value['market']
        price = value['value']
        
        # Initialize window
        if market not in self.price_windows:
            self.price_windows[market] = []
        
        # Add to window
        self.price_windows[market].append(price)
        
        # Keep only window size
        if len(self.price_windows[market]) > self.window_size:
            self.price_windows[market].pop(0)
        
        # Calculate correlations when we have enough data
        if len(self.price_windows) >= 2:
            for other_market in self.price_windows:
                if other_market != market and len(self.price_windows[other_market]) == self.window_size:
                    correlation = self._calculate_correlation(
                        self.price_windows[market],
                        self.price_windows[other_market]
                    )
                    
                    # Emit correlation update
                    yield {
                        'timestamp': value['timestamp'],
                        'market_1': market,
                        'market_2': other_market,
                        'correlation': correlation,
                        'window_size': self.window_size,
                    }
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        if n == 0:
            return 0.0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
        std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5
        
        if std_x == 0 or std_y == 0:
            return 0.0
        
        return covariance / (std_x * std_y)


def create_price_stream_job(env: StreamExecutionEnvironment):
    """
    Create Flink job for price stream processing.
    
    Includes:
    - Anomaly detection
    - Spread calculation
    - Sliding window aggregations
    - Real-time alerting
    """
    # Configure Kafka source
    from pyflink.datastream.connectors.kafka import KafkaSource
    
    kafka_source = KafkaSource.builder() \
        .set_bootstrap_servers("kafka:9092") \
        .set_topics("power.ticks.v1") \
        .set_group_id("flink-price-processor") \
        .set_value_only_deserializer(
            # JSON deserializer
            ...
        ) \
        .build()
    
    # Create stream from Kafka
    price_stream = env.from_source(
        kafka_source,
        WatermarkStrategy.for_monotonous_timestamps(),
        "kafka-source"
    )
    
    # 1. Anomaly Detection
    anomalies = price_stream \
        .key_by(lambda x: x['instrument_id']) \
        .process(PriceAnomalyDetector(threshold_std=3.0))
    
    # Write anomalies to sink
    anomalies.print()  # Or write to Kafka, database, etc.
    
    # 2. Spread Calculation
    spreads = price_stream \
        .key_by(lambda x: x['market']) \
        .process(PriceSpreadCalculator(spread_threshold=10.0))
    
    spreads.print()
    
    # 3. Sliding Window Aggregations (5-minute windows, 1-minute slide)
    windowed_aggregates = price_stream \
        .key_by(lambda x: x['instrument_id']) \
        .window(SlidingEventTimeWindows.of(Time.minutes(5), Time.minutes(1))) \
        .apply(lambda key, window, values: {
            'instrument_id': key,
            'window_start': window.start,
            'window_end': window.end,
            'avg_price': sum(v['value'] for v in values) / len(values),
            'max_price': max(v['value'] for v in values),
            'min_price': min(v['value'] for v in values),
            'volume': sum(v.get('volume', 0) for v in values),
        })
    
    windowed_aggregates.print()
    
    # 4. Tumbling Window Aggregations (1-hour non-overlapping windows)
    hourly_stats = price_stream \
        .key_by(lambda x: x['market']) \
        .window(TumblingEventTimeWindows.of(Time.hours(1))) \
        .apply(lambda key, window, values: {
            'market': key,
            'hour': datetime.fromtimestamp(window.start / 1000).strftime('%Y-%m-%d %H:00'),
            'avg_price': sum(v['value'] for v in values) / len(values),
            'weighted_avg_price': sum(v['value'] * v.get('volume', 1) for v in values) / sum(v.get('volume', 1) for v in values),
            'volatility': (sum((v['value'] - sum(x['value'] for x in values) / len(values)) ** 2 for v in values) / len(values)) ** 0.5,
            'events_count': len(values),
        })
    
    hourly_stats.print()
    
    return env


def create_cep_job(env: StreamExecutionEnvironment):
    """
    Create Complex Event Processing job.
    
    Detects patterns like:
    - Consecutive price increases (momentum)
    - Price reversals
    - Coordinated market movements
    """
    from pyflink.cep import CEP, Pattern
    
    # Define pattern: 3 consecutive price increases
    price_momentum_pattern = Pattern.begin("start").where(
        lambda event: True  # First event
    ).next("second").where(
        lambda event, ctx: event['value'] > ctx['start']['value']
    ).next("third").where(
        lambda event, ctx: event['value'] > ctx['second']['value']
    )
    
    # Apply pattern to stream
    # ... (CEP implementation)
    
    return env


def main():
    """Main entry point for Flink jobs."""
    logger.info("Starting Flink stream processing jobs")
    
    # Create execution environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(4)
    
    # Configure checkpointing
    env.enable_checkpointing(60000)  # Every 60 seconds
    
    # Create price stream job
    create_price_stream_job(env)
    
    # Execute
    env.execute("254Carbon Market Data Stream Processing")


if __name__ == "__main__":
    main()

