"""
Pipeline Congestion Prediction Model

Machine learning model for predicting natural gas pipeline congestion:
- Flow pattern analysis using time series forecasting
- Demand surge prediction based on weather and economic indicators
- Maintenance schedule impact assessment
- Capacity constraint identification
- Congestion risk scoring and alerts
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class PipelineCongestionModel:
    """
    Pipeline congestion prediction and analysis model.

    Features:
    - Time series forecasting of pipeline flows
    - Weather and demand impact modeling
    - Maintenance schedule optimization
    - Congestion risk assessment
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}

        # Feature importance tracking
        self.feature_importance = {}

    def train_congestion_model(
        self,
        historical_flows: pd.Series,
        weather_data: pd.DataFrame,
        demand_data: pd.Series,
        maintenance_schedule: Optional[pd.Series] = None,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train machine learning model for congestion prediction.

        Args:
            historical_flows: Historical pipeline flow data
            weather_data: Weather features (temperature, HDD, CDD)
            demand_data: Demand proxy data
            maintenance_schedule: Pipeline maintenance indicators
            test_size: Fraction of data for testing

        Returns:
            Model training results
        """
        logger.info("Training pipeline congestion prediction model")

        # Prepare features
        features = self._prepare_congestion_features(
            historical_flows, weather_data, demand_data, maintenance_schedule
        )

        # Target variable (flow as percentage of capacity)
        # Assume 10 Bcf/d capacity for this example
        capacity = 10.0  # Bcf/d
        utilization = historical_flows / capacity

        # Remove any NaN values
        valid_data = features.dropna()
        valid_target = utilization[valid_data.index]

        if len(valid_data) < 50:
            return {'error': 'Insufficient data for model training'}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            valid_data, valid_target, test_size=test_size, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        results = {}

        for model_name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)

            # Make predictions
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)

            # Calculate metrics
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

            results[model_name] = {
                'model': model,
                'scaler': scaler,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None
            }

            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': valid_data.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[model_name] = importance_df

        # Select best model (lowest test MAE)
        best_model_name = min(results.keys(), key=lambda x: results[x]['test_mae'])

        return {
            'best_model': best_model_name,
            'model_results': results,
            'training_data_size': len(X_train),
            'test_data_size': len(X_test),
            'capacity_assumption': capacity,
            'features_used': list(valid_data.columns),
            'feature_importance': self.feature_importance
        }

    def _prepare_congestion_features(
        self,
        flows: pd.Series,
        weather: pd.DataFrame,
        demand: pd.Series,
        maintenance: Optional[pd.Series]
    ) -> pd.DataFrame:
        """Prepare features for congestion modeling."""
        features = pd.DataFrame(index=flows.index)

        # Lag features for flows
        for lag in [1, 7, 14, 30]:
            features[f'flow_lag_{lag}'] = flows.shift(lag)

        # Rolling statistics for flows
        for window in [7, 14, 30]:
            features[f'flow_mean_{window}'] = flows.rolling(window).mean()
            features[f'flow_std_{window}'] = flows.rolling(window).std()
            features[f'flow_max_{window}'] = flows.rolling(window).max()

        # Weather features
        if 'temperature' in weather.columns:
            features['temperature'] = weather['temperature']

            # Temperature lags
            for lag in [1, 7]:
                features[f'temp_lag_{lag}'] = weather['temperature'].shift(lag)

        if 'heating_degree_days' in weather.columns:
            features['heating_degree_days'] = weather['heating_degree_days']

        if 'cooling_degree_days' in weather.columns:
            features['cooling_degree_days'] = weather['cooling_degree_days']

        # Demand features
        features['demand'] = demand

        # Demand lags
        for lag in [1, 7]:
            features[f'demand_lag_{lag}'] = demand.shift(lag)

        # Maintenance indicators
        if maintenance is not None:
            features['maintenance'] = maintenance.astype(int)

            # Maintenance lag effects
            for lag in [1, 7, 14]:
                features[f'maintenance_lag_{lag}'] = maintenance.shift(lag).astype(int)

        # Time-based features
        features['hour'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        features['month'] = features.index.month
        features['quarter'] = features.index.quarter

        # Seasonal indicators
        features['is_winter'] = features.index.month.isin([12, 1, 2]).astype(int)
        features['is_summer'] = features.index.month.isin([6, 7, 8]).astype(int)

        return features

    def predict_congestion(
        self,
        model_name: str,
        forecast_features: pd.DataFrame,
        forecast_horizon: int = 7
    ) -> pd.Series:
        """
        Predict pipeline congestion for future periods.

        Args:
            model_name: Name of trained model to use
            forecast_features: Features for prediction period
            forecast_horizon: Days to forecast

        Returns:
            Congestion predictions (utilization rates)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train model first.")

        model_info = self.models[model_name]
        model = model_info['model']
        scaler = model_info['scaler']

        # Scale features
        feature_cols = forecast_features.columns
        scaled_features = scaler.transform(forecast_features)

        # Make predictions
        predictions = model.predict(scaled_features)

        # Convert to Series
        prediction_series = pd.Series(predictions, index=forecast_features.index)

        return prediction_series

    def assess_congestion_risk(
        self,
        predicted_utilization: pd.Series,
        capacity: float = 10.0,  # Bcf/d
        risk_thresholds: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Assess congestion risk based on utilization predictions.

        Args:
            predicted_utilization: Predicted utilization rates
            capacity: Pipeline capacity (Bcf/d)
            risk_thresholds: Risk threshold definitions

        Returns:
            Congestion risk assessment
        """
        if risk_thresholds is None:
            risk_thresholds = {
                'low': 0.7,      # <70% = low risk
                'medium': 0.85,  # 70-85% = medium risk
                'high': 0.95,    # 85-95% = high risk
                'critical': 1.0  # >95% = critical risk
            }

        # Classify risk levels
        risk_classification = pd.Series(index=predicted_utilization.index)

        for date, utilization in predicted_utilization.items():
            if utilization >= risk_thresholds['critical']:
                risk_classification[date] = 'critical'
            elif utilization >= risk_thresholds['high']:
                risk_classification[date] = 'high'
            elif utilization >= risk_thresholds['medium']:
                risk_classification[date] = 'medium'
            else:
                risk_classification[date] = 'low'

        # Calculate risk statistics
        risk_counts = risk_classification.value_counts()

        # Identify high-risk periods
        high_risk_periods = risk_classification[risk_classification.isin(['high', 'critical'])]

        # Calculate expected congestion duration
        congestion_events = self._identify_congestion_events_from_risk(risk_classification)
        avg_congestion_duration = np.mean([event['duration'] for event in congestion_events]) if congestion_events else 0

        return {
            'risk_classification': risk_classification,
            'risk_distribution': risk_counts.to_dict(),
            'high_risk_days': len(high_risk_periods),
            'congestion_events': len(congestion_events),
            'avg_congestion_duration': avg_congestion_duration,
            'max_predicted_utilization': predicted_utilization.max(),
            'risk_thresholds': risk_thresholds,
            'capacity_bcfd': capacity
        }

    def _identify_congestion_events_from_risk(self, risk_series: pd.Series) -> List[Dict[str, Any]]:
        """Identify congestion events from risk classification."""
        events = []

        in_congestion = False
        start_date = None

        for date, risk in risk_series.items():
            if risk in ['high', 'critical'] and not in_congestion:
                # Start of congestion event
                in_congestion = True
                start_date = date
            elif risk not in ['high', 'critical'] and in_congestion:
                # End of congestion event
                in_congestion = False
                duration = (date - start_date).days
                events.append({
                    'start_date': start_date,
                    'end_date': date,
                    'duration': duration,
                    'risk_level': 'high'
                })

        # Handle ongoing event
        if in_congestion:
            duration = (risk_series.index[-1] - start_date).days
            events.append({
                'start_date': start_date,
                'end_date': risk_series.index[-1],
                'duration': duration,
                'risk_level': 'high'
            })

        return events

    def optimize_maintenance_scheduling(
        self,
        predicted_demand: pd.Series,
        maintenance_requirements: List[Dict[str, Any]],
        capacity: float = 10.0,
        min_operating_capacity: float = 7.0  # Minimum capacity during maintenance
    ) -> Dict[str, Any]:
        """
        Optimize maintenance scheduling to minimize congestion impact.

        Args:
            predicted_demand: Predicted demand/utilization
            maintenance_requirements: List of maintenance tasks with durations
            capacity: Total pipeline capacity
            min_operating_capacity: Minimum capacity needed during maintenance

        Returns:
            Optimal maintenance schedule
        """
        # Simple optimization: schedule maintenance during low-demand periods

        # Identify low-demand periods
        low_demand_threshold = predicted_demand.quantile(0.25)  # Bottom 25%
        low_demand_periods = predicted_demand[predicted_demand <= low_demand_threshold]

        # Sort maintenance tasks by duration (longest first)
        sorted_maintenance = sorted(
            maintenance_requirements,
            key=lambda x: x['duration_days'],
            reverse=True
        )

        scheduled_maintenance = []
        used_periods = set()

        for maintenance in sorted_maintenance:
            duration = maintenance['duration_days']

            # Find available low-demand period for this maintenance
            available_periods = []

            for start_date in low_demand_periods.index:
                if start_date in used_periods:
                    continue

                # Check if we have enough consecutive days
                end_date = start_date + pd.Timedelta(days=duration)

                # Check if all days in the period have low demand
                period_mask = (low_demand_periods.index >= start_date) & (low_demand_periods.index < end_date)
                period_demand = low_demand_periods[period_mask]

                if len(period_demand) >= duration:
                    # Check if utilization would stay above minimum
                    max_utilization_in_period = predicted_demand[start_date:end_date].max()

                    if max_utilization <= (min_operating_capacity / capacity):
                        available_periods.append({
                            'start_date': start_date,
                            'end_date': end_date,
                            'avg_demand': period_demand.mean(),
                            'max_utilization': max_utilization_in_period
                        })

            if available_periods:
                # Select best period (lowest average demand)
                best_period = min(available_periods, key=lambda x: x['avg_demand'])

                scheduled_maintenance.append({
                    'maintenance_task': maintenance['name'],
                    'start_date': best_period['start_date'],
                    'end_date': best_period['end_date'],
                    'duration_days': duration,
                    'expected_impact': best_period['avg_demand']
                })

                # Mark period as used
                for d in range(duration):
                    date = best_period['start_date'] + pd.Timedelta(days=d)
                    used_periods.add(date)

        # Calculate total impact
        total_impact = sum(maint['expected_impact'] for maint in scheduled_maintenance)

        return {
            'scheduled_maintenance': scheduled_maintenance,
            'total_maintenance_tasks': len(maintenance_requirements),
            'scheduled_tasks': len(scheduled_maintenance),
            'total_impact': total_impact,
            'unscheduled_tasks': len(maintenance_requirements) - len(scheduled_maintenance),
            'low_demand_periods_available': len(low_demand_periods),
            'capacity_bcfd': capacity,
            'min_operating_capacity_bcfd': min_operating_capacity
        }

    def generate_congestion_alerts(
        self,
        congestion_forecast: pd.Series,
        alert_thresholds: Dict[str, float] = None,
        lookback_days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Generate congestion alerts based on forecasts.

        Args:
            congestion_forecast: Predicted utilization rates
            alert_thresholds: Alert threshold levels
            lookback_days: Days to look ahead for alerts

        Returns:
            List of congestion alerts
        """
        if alert_thresholds is None:
            alert_thresholds = {
                'warning': 0.8,     # Warning at 80% utilization
                'critical': 0.95    # Critical at 95% utilization
            }

        alerts = []

        # Look ahead for congestion events
        forecast_window = congestion_forecast.head(lookback_days)

        for date, utilization in forecast_window.items():
            if utilization >= alert_thresholds['critical']:
                alerts.append({
                    'date': date,
                    'alert_type': 'critical_congestion',
                    'utilization': utilization,
                    'message': f'Critical congestion predicted: {utilization:.1%} utilization',
                    'severity': 'critical'
                })
            elif utilization >= alert_thresholds['warning']:
                alerts.append({
                    'date': date,
                    'alert_type': 'congestion_warning',
                    'utilization': utilization,
                    'message': f'High utilization warning: {utilization:.1%} utilization',
                    'severity': 'warning'
                })

        # Sort alerts by date and severity
        alerts.sort(key=lambda x: (x['date'], x['severity'] == 'critical'), reverse=True)

        return alerts

    def analyze_seasonal_congestion_patterns(
        self,
        historical_utilization: pd.Series,
        years_back: int = 3
    ) -> Dict[str, pd.Series]:
        """
        Analyze seasonal congestion patterns.

        Args:
            historical_utilization: Historical utilization data
            years_back: Years of historical data to analyze

        Returns:
            Seasonal congestion patterns
        """
        # Filter to analysis period
        end_date = historical_utilization.index.max()
        start_date = end_date - pd.DateOffset(years=years_back)

        analysis_data = historical_utilization[
            (historical_utilization.index >= start_date) &
            (historical_utilization.index <= end_date)
        ]

        if len(analysis_data) == 0:
            return {'error': 'No data in analysis period'}

        # Monthly congestion patterns
        monthly_congestion = analysis_data.groupby(analysis_data.index.month).agg(['mean', 'std', 'max'])

        # Weekly patterns
        weekly_congestion = analysis_data.groupby(analysis_data.index.weekday).agg(['mean', 'std', 'max'])

        # Seasonal risk scores
        seasonal_risk = monthly_congestion['mean'] / monthly_congestion['mean'].max()

        return {
            'monthly_patterns': monthly_congestion['mean'],
            'weekly_patterns': weekly_congestion['mean'],
            'monthly_volatility': monthly_congestion['std'],
            'seasonal_risk_scores': seasonal_risk,
            'analysis_period': f'{start_date.date()} to {end_date.date()}',
            'data_points': len(analysis_data)
        }
