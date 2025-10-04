"""
Data Quality Framework

Comprehensive data quality validation, outlier detection, and lineage tracking for multi-commodity energy data:
- Cross-source validation for commodities
- Outlier detection by commodity type
- Missing data imputation strategies
- Data lineage tracking enhancement
- Quality scoring and alerting
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


COAL_METRICS = {
    "coal_stockpile_tonnes": {"value_min": 0, "value_max": 10_000_000},
    "coal_stockpile_change_7d_pct": {"value_min": -100, "value_max": 100},
}

LNG_METRICS = {
    "lng_vessels_at_berth_count": {"value_min": 0, "value_max": 500},
    "lng_vessels_waiting_count": {"value_min": 0, "value_max": 500},
    "lng_arrivals_24h_count": {"value_min": 0, "value_max": 500},
    "lng_departures_24h_count": {"value_min": 0, "value_max": 500},
}

GAS_STORAGE_METRICS = {
    "ng_storage_level_gwh": {"value_min": 0, "value_max": 2_000_000},
    "ng_storage_pct_full": {"value_min": 0, "value_max": 100},
    "ng_injection_gwh": {"value_min": 0, "value_max": 200_000},
    "ng_withdrawal_gwh": {"value_min": 0, "value_max": 200_000},
}

SUPPLY_CHAIN_METRICS = {
    "lng_cost_per_mmbtu_usd": {"value_min": 0, "value_max": 50},
    "lng_total_route_cost_usd": {"value_min": 0, "value_max": 50_000_000},
    "coal_total_route_cost_usd": {"value_min": 0, "value_max": 75_000_000},
    "pipeline_utilization_forecast_pct": {"value_min": 0, "value_max": 100},
    "pipeline_congestion_probability": {"value_min": 0, "value_max": 1},
    "seasonal_demand_forecast_mw": {"value_min": 0, "value_max": 15_000_000},
    "seasonal_peak_risk_score": {"value_min": 0, "value_max": 1},
}


class DataQualityFramework:
    """
    Comprehensive data quality framework for energy commodity data.

    Features:
    - Cross-source validation
    - Outlier detection algorithms
    - Missing data imputation
    - Data lineage tracking
    - Quality scoring and monitoring
    """

    def __init__(self):
        self.outlier_detectors = {}
        self.quality_scores = {}
        self.lineage_tracker = DataLineageTracker()
        self.metric_catalog = {
            'coal': COAL_METRICS,
            'lng': LNG_METRICS,
            'gas_storage': GAS_STORAGE_METRICS,
            'supply_chain': SUPPLY_CHAIN_METRICS,
        }

    def validate_cross_source_data(
        self,
        data_sources: Dict[str, pd.DataFrame],
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate data consistency across multiple sources.

        Args:
            data_sources: Dictionary of data by source
            validation_rules: Custom validation rules

        Returns:
            Cross-source validation results
        """
        if validation_rules is None:
            validation_rules = {
                'price_tolerance': 0.05,    # 5% price tolerance
                'volume_tolerance': 0.10,   # 10% volume tolerance
                'time_alignment': True,     # Check time alignment
                'completeness_threshold': 0.95  # 95% completeness required
            }

        validation_results = {
            'overall_score': 0,
            'source_scores': {},
            'validation_errors': [],
            'data_gaps': [],
            'consistency_issues': []
        }

        sources = list(data_sources.keys())
        if len(sources) < 2:
            return validation_results

        # Check data completeness
        for source, data in data_sources.items():
            completeness = self._calculate_completeness(data)
            validation_results['source_scores'][source] = completeness

            if completeness < validation_rules['completeness_threshold']:
                validation_results['validation_errors'].append({
                    'type': 'completeness',
                    'source': source,
                    'score': completeness,
                    'threshold': validation_rules['completeness_threshold']
                })

        # Check cross-source consistency
        consistency_score = self._check_cross_source_consistency(data_sources, validation_rules)
        validation_results['consistency_score'] = consistency_score

        if consistency_score < 0.8:  # 80% consistency threshold
            validation_results['consistency_issues'].append({
                'type': 'cross_source_consistency',
                'score': consistency_score,
                'sources': sources
            })

        # Calculate overall quality score
        validation_results['overall_score'] = np.mean([
            score for score in validation_results['source_scores'].values()
        ])

        return validation_results

    def get_metric_rules(self, domain: str) -> Dict[str, Dict[str, float]]:
        """Return configured metric bounds for a given data domain."""
        return self.metric_catalog.get(domain.lower(), {})

    def register_metric_rules(
        self,
        domain: str,
        rules: Dict[str, Dict[str, float]],
        overwrite: bool = False
    ) -> None:
        """Allow dynamic registration of metric rule dictionaries."""
        if domain.lower() in self.metric_catalog and not overwrite:
            raise ValueError(f"Metric rules for {domain!r} already exist; pass overwrite=True to replace.")
        self.metric_catalog[domain.lower()] = rules

    def _calculate_completeness(self, data: pd.DataFrame) -> float:
        """Calculate data completeness score."""
        total_cells = data.shape[0] * data.shape[1]
        non_null_cells = data.notna().sum().sum()

        return non_null_cells / total_cells if total_cells > 0 else 0

    def _check_cross_source_consistency(
        self,
        data_sources: Dict[str, pd.DataFrame],
        rules: Dict[str, Any]
    ) -> float:
        """Check consistency across data sources."""
        sources = list(data_sources.keys())
        consistency_scores = []

        # Compare each pair of sources
        for i, source1 in enumerate(sources[:-1]):
            for source2 in sources[i+1:]:
                data1 = data_sources[source1]
                data2 = data_sources[source2]

                # Align data by common columns and time
                common_data = self._align_data_sources(data1, data2)

                if common_data is None or len(common_data) < 10:
                    continue

                # Calculate price consistency
                if 'value' in common_data.columns:
                    price_corr = np.corrcoef(common_data['value_x'], common_data['value_y'])[0, 1]
                    consistency_scores.append(abs(price_corr))

                # Calculate volume consistency
                if 'volume' in common_data.columns:
                    vol_corr = np.corrcoef(common_data['volume_x'], common_data['volume_y'])[0, 1]
                    consistency_scores.append(abs(vol_corr))

        return np.mean(consistency_scores) if consistency_scores else 0

    def _align_data_sources(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Align two data sources for comparison."""
        # Find common time periods and instruments
        common_index = data1.index.intersection(data2.index)

        if len(common_index) < 10:
            return None

        aligned_data1 = data1.loc[common_index]
        aligned_data2 = data2.loc[common_index]

        # Merge on index
        merged = pd.merge(
            aligned_data1.reset_index(),
            aligned_data2.reset_index(),
            on='event_time',
            how='inner',
            suffixes=('_x', '_y')
        )

        return merged.set_index('event_time')

    def detect_outliers_by_commodity(
        self,
        data: pd.DataFrame,
        commodity_type: str,
        method: str = 'isolation_forest',
        contamination: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect outliers specific to commodity types.

        Args:
            data: Price/volume data
            commodity_type: Type of commodity
            method: Outlier detection method
            contamination: Expected proportion of outliers

        Returns:
            Outlier detection results
        """
        # Prepare features for outlier detection
        features = self._prepare_outlier_features(data, commodity_type)

        if len(features) < 50:  # Need minimum data for reliable detection
            return {'error': 'Insufficient data for outlier detection'}

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Detect outliers
        if method == 'isolation_forest':
            detector = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = detector.fit_predict(scaled_features)

            # Convert to boolean (1 = normal, -1 = outlier)
            outliers = outlier_labels == -1

        elif method == 'z_score':
            # Z-score method
            z_scores = np.abs(stats.zscore(scaled_features, axis=0))
            outliers = np.any(z_scores > 3, axis=1)  # 3-sigma rule

        elif method == 'iqr':
            # IQR method
            Q1 = np.percentile(scaled_features, 25, axis=0)
            Q3 = np.percentile(scaled_features, 75, axis=0)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = np.any((scaled_features < lower_bound) | (scaled_features > upper_bound), axis=1)

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        # Analyze outlier patterns
        outlier_data = data[outliers]
        normal_data = data[~outliers]

        outlier_analysis = {
            'outlier_count': outliers.sum(),
            'outlier_percentage': outliers.mean() * 100,
            'outlier_prices': outlier_data['value'].tolist() if 'value' in outlier_data.columns else [],
            'outlier_dates': outlier_data.index.tolist(),
            'normal_price_mean': normal_data['value'].mean() if 'value' in normal_data.columns else 0,
            'outlier_price_mean': outlier_data['value'].mean() if 'value' in outlier_data.columns else 0,
            'detection_method': method,
            'contamination_rate': contamination
        }

        # Store detector for future use
        self.outlier_detectors[commodity_type] = {
            'detector': detector if method == 'isolation_forest' else None,
            'scaler': scaler,
            'features': list(features.columns)
        }

        return outlier_analysis

    def _prepare_outlier_features(self, data: pd.DataFrame, commodity_type: str) -> pd.DataFrame:
        """Prepare features for outlier detection."""
        features = pd.DataFrame(index=data.index)

        # Price features
        if 'value' in data.columns:
            features['price'] = data['value']
            features['price_change'] = data['value'].pct_change()
            features['price_volatility'] = data['value'].rolling(20).std()

        # Volume features
        if 'volume' in data.columns:
            features['volume'] = data['volume']
            features['volume_change'] = data['volume'].pct_change()

        # Time-based features
        features['hour'] = data.index.hour
        features['day_of_week'] = data.index.dayofweek
        features['month'] = data.index.month

        # Commodity-specific features
        if commodity_type == 'power':
            features['is_peak_hour'] = ((data.index.hour >= 7) & (data.index.hour <= 22)).astype(int)
        elif commodity_type in ['oil', 'gas']:
            features['is_trading_hour'] = ((data.index.hour >= 6) & (data.index.hour <= 18)).astype(int)

        return features.dropna()

    def impute_missing_data(
        self,
        data: pd.DataFrame,
        commodity_type: str,
        method: str = 'interpolation',
        max_gap_hours: int = 24
    ) -> pd.DataFrame:
        """
        Impute missing data using commodity-specific strategies.

        Args:
            data: Data with missing values
            commodity_type: Type of commodity for strategy selection
            method: Imputation method
            max_gap_hours: Maximum gap size for interpolation

        Returns:
            Imputed data
        """
        imputed_data = data.copy()

        # Handle missing values by column
        for column in data.columns:
            if data[column].isna().sum() == 0:
                continue

            missing_mask = data[column].isna()

            if method == 'interpolation':
                # Linear interpolation for gaps smaller than max_gap
                imputed_data[column] = self._interpolate_gaps(
                    data[column], missing_mask, max_gap_hours
                )

            elif method == 'forward_fill':
                # Forward fill for short gaps
                imputed_data[column] = data[column].fillna(method='ffill', limit=max_gap_hours)

            elif method == 'mean_fill':
                # Fill with rolling mean
                rolling_mean = data[column].rolling(window=24, center=True).mean()
                imputed_data[column] = data[column].fillna(rolling_mean)

            elif method == 'commodity_specific':
                # Commodity-specific imputation strategies
                imputed_data[column] = self._commodity_specific_imputation(
                    data[column], commodity_type, missing_mask
                )

        return imputed_data

    def _interpolate_gaps(
        self,
        series: pd.Series,
        missing_mask: pd.Series,
        max_gap_hours: int
    ) -> pd.Series:
        """Interpolate gaps in time series data."""
        imputed = series.copy()

        # Find gap locations
        gap_starts = []
        gap_sizes = []

        in_gap = False
        for i, is_missing in enumerate(missing_mask):
            if is_missing and not in_gap:
                gap_starts.append(i)
                in_gap = True
            elif not is_missing and in_gap:
                gap_sizes.append(i - gap_starts[-1])
                in_gap = False

        # Handle ongoing gap
        if in_gap:
            gap_sizes.append(len(missing_mask) - gap_starts[-1])

        # Interpolate gaps smaller than max_gap
        for start_idx, size in zip(gap_starts, gap_sizes):
            if size <= max_gap_hours:
                # Linear interpolation
                if start_idx > 0 and start_idx + size < len(series):
                    start_val = series.iloc[start_idx - 1]
                    end_val = series.iloc[start_idx + size]

                    if not pd.isna(start_val) and not pd.isna(end_val):
                        # Create interpolated values
                        interp_values = np.linspace(start_val, end_val, size + 2)[1:-1]
                        imputed.iloc[start_idx:start_idx + size] = interp_values

        return imputed

    def _commodity_specific_imputation(
        self,
        series: pd.Series,
        commodity_type: str,
        missing_mask: pd.Series
    ) -> pd.Series:
        """Apply commodity-specific imputation strategies."""
        imputed = series.copy()

        if commodity_type == 'power':
            # Power prices often follow daily patterns
            hourly_mean = series.groupby(series.index.hour).mean()
            for idx in series[missing_mask].index:
                imputed.loc[idx] = hourly_mean.get(idx.hour, series.mean())

        elif commodity_type in ['oil', 'gas']:
            # Energy commodities often follow market hours patterns
            trading_hour_mean = series.groupby(
                (series.index.hour >= 6) & (series.index.hour <= 18)
            ).mean()
            for idx in series[missing_mask].index:
                is_trading_hour = (idx.hour >= 6) and (idx.hour <= 18)
                imputed.loc[idx] = trading_hour_mean.get(is_trading_hour, series.mean())

        else:
            # Default to overall mean
            imputed = series.fillna(series.mean())

        return imputed

    def track_data_lineage(
        self,
        data: pd.DataFrame,
        source: str,
        transformation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Track data lineage for audit and debugging.

        Args:
            data: Data being tracked
            source: Data source name
            transformation: Transformation applied
            metadata: Additional metadata

        Returns:
            Lineage ID for tracking
        """
        lineage_id = self.lineage_tracker.add_entry(
            source=source,
            transformation=transformation,
            timestamp=datetime.now(),
            metadata=metadata or {},
            data_shape=data.shape,
            data_hash=hash(str(data.values.tobytes()))
        )

        return lineage_id

    def get_data_quality_score(
        self,
        data: pd.DataFrame,
        commodity_type: str,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate overall data quality score.

        Args:
            data: Data to score
            commodity_type: Type of commodity
            weights: Weights for different quality dimensions

        Returns:
            Quality score (0-1)
        """
        if weights is None:
            weights = {
                'completeness': 0.3,
                'accuracy': 0.3,
                'consistency': 0.2,
                'timeliness': 0.2
            }

        # Completeness score
        completeness = self._calculate_completeness(data)

        # Accuracy score (inverse of outlier percentage)
        outlier_analysis = self.detect_outliers_by_commodity(data, commodity_type)
        if 'error' not in outlier_analysis:
            accuracy = 1 - (outlier_analysis['outlier_percentage'] / 100)
        else:
            accuracy = 0.5  # Default if outlier detection fails

        # Consistency score (placeholder - would need cross-source data)
        consistency = 0.8  # Placeholder

        # Timeliness score (based on data freshness)
        if not data.empty:
            max_age = (datetime.now() - data.index.max()).total_seconds() / 3600  # hours
            timeliness = max(0, 1 - (max_age / 24))  # 1 if <24h old, 0 if >24h old
        else:
            timeliness = 0

        # Weighted quality score
        quality_score = (
            completeness * weights['completeness'] +
            accuracy * weights['accuracy'] +
            consistency * weights['consistency'] +
            timeliness * weights['timeliness']
        )

        # Store quality score for monitoring
        self.quality_scores[commodity_type] = {
            'score': quality_score,
            'completeness': completeness,
            'accuracy': accuracy,
            'consistency': consistency,
            'timeliness': timeliness,
            'timestamp': datetime.now()
        }

        return quality_score

    def generate_quality_report(
        self,
        data_sources: Dict[str, pd.DataFrame],
        commodity_types: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.

        Args:
            data_sources: Dictionary of data by source
            commodity_types: Commodity types by source

        Returns:
            Quality report summary
        """
        report = {
            'timestamp': datetime.now(),
            'sources_analyzed': len(data_sources),
            'quality_scores': {},
            'outlier_summary': {},
            'completeness_summary': {},
            'recommendations': []
        }

        for source, data in data_sources.items():
            commodity_type = commodity_types.get(source, 'unknown')

            # Calculate quality score
            quality_score = self.get_data_quality_score(data, commodity_type)
            report['quality_scores'][source] = quality_score

            # Detect outliers
            outlier_analysis = self.detect_outliers_by_commodity(data, commodity_type)
            report['outlier_summary'][source] = outlier_analysis

            # Calculate completeness
            completeness = self._calculate_completeness(data)
            report['completeness_summary'][source] = completeness

        # Generate recommendations
        if any(score < 0.8 for score in report['quality_scores'].values()):
            report['recommendations'].append("Review data sources with quality scores below 0.8")

        avg_outliers = np.mean([
            analysis.get('outlier_percentage', 0)
            for analysis in report['outlier_summary'].values()
        ])

        if avg_outliers > 5:  # 5% outlier threshold
            report['recommendations'].append(f"High outlier rate detected ({avg_outliers:.1f}%). Review outlier detection parameters.")

        return report


class DataLineageTracker:
    """Track data lineage for audit and debugging purposes."""

    def __init__(self):
        self.lineage_entries = []
        self.entry_counter = 0

    def add_entry(
        self,
        source: str,
        transformation: str,
        timestamp: datetime,
        metadata: Dict[str, Any],
        data_shape: Tuple[int, int],
        data_hash: int
    ) -> str:
        """Add a lineage entry."""
        self.entry_counter += 1
        entry_id = f"LINEAGE_{self.entry_counter}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        entry = {
            'id': entry_id,
            'source': source,
            'transformation': transformation,
            'timestamp': timestamp,
            'metadata': metadata,
            'data_shape': data_shape,
            'data_hash': data_hash,
            'inputs': [],
            'outputs': []
        }

        self.lineage_entries.append(entry)
        return entry_id

    def link_entries(self, parent_id: str, child_id: str) -> None:
        """Link parent and child lineage entries."""
        for entry in self.lineage_entries:
            if entry['id'] == parent_id:
                entry['outputs'].append(child_id)
            elif entry['id'] == child_id:
                entry['inputs'].append(parent_id)

    def get_lineage_chain(self, entry_id: str) -> List[Dict[str, Any]]:
        """Get full lineage chain for an entry."""
        chain = []

        def find_ancestors(entry_id: str):
            for entry in self.lineage_entries:
                if entry['id'] == entry_id:
                    chain.append(entry)
                    for input_id in entry['inputs']:
                        find_ancestors(input_id)
                    break

        find_ancestors(entry_id)
        return chain

    def get_lineage_summary(self) -> Dict[str, Any]:
        """Get summary of data lineage."""
        return {
            'total_entries': len(self.lineage_entries),
            'sources': list(set(entry['source'] for entry in self.lineage_entries)),
            'transformations': list(set(entry['transformation'] for entry in self.lineage_entries)),
            'time_range': {
                'start': min(entry['timestamp'] for entry in self.lineage_entries),
                'end': max(entry['timestamp'] for entry in self.lineage_entries)
            } if self.lineage_entries else None
        }
