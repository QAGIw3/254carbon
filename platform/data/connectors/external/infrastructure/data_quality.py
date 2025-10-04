"""
Infrastructure Data Quality Validation
---------------------------------------

Implements data quality checks and reconciliation for infrastructure data,
ensuring consistency, completeness, and accuracy across sources.

Quality Dimensions:
- Completeness: All required fields present
- Validity: Values within expected ranges
- Consistency: Cross-source agreement
- Timeliness: Data freshness
- Accuracy: Geographic and technical validation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

import numpy as np
from geopy.distance import geodesic

from .base import (
    InfrastructureAsset,
    PowerPlant,
    LNGTerminal,
    TransmissionLine,
    GeoLocation,
    FuelType,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QualityIssue:
    """Represents a data quality issue."""
    
    severity: str  # error, warning, info
    dimension: str  # completeness, validity, consistency, timeliness, accuracy
    field: str
    message: str
    asset_id: Optional[str] = None
    value: Optional[Any] = None
    expected: Optional[Any] = None
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class QualityReport:
    """Data quality assessment report."""
    
    total_records: int = 0
    valid_records: int = 0
    issues: List[QualityIssue] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def quality_score(self) -> float:
        """Calculate overall quality score (0-100)."""
        if self.total_records == 0:
            return 0.0
        
        # Base score from valid records ratio
        base_score = (self.valid_records / self.total_records) * 100
        
        # Apply penalties for issues
        error_penalty = sum(1 for i in self.issues if i.severity == "error") * 5
        warning_penalty = sum(1 for i in self.issues if i.severity == "warning") * 2
        
        return max(0, base_score - error_penalty - warning_penalty)
    
    def add_issue(self, issue: QualityIssue) -> None:
        """Add a quality issue to the report."""
        self.issues.append(issue)
        if issue.severity == "error":
            self.valid_records = max(0, self.valid_records - 1)


class InfrastructureDataValidator:
    """Validates infrastructure data quality."""
    
    # Expected value ranges by metric
    METRIC_RANGES = {
        # Power plant metrics
        "capacity_mw": (0.1, 25000),  # Largest is Three Gorges ~22,500 MW
        "capacity_factor": (0, 1.0),
        "efficiency_pct": (0, 65),  # Best combined cycle ~64%
        "emissions_rate_tco2_mwh": (0, 2.5),  # Worst coal ~2.2
        "heat_rate": (6000, 15000),  # BTU/kWh
        
        # LNG terminal metrics
        "lng_inventory_gwh": (0, 5000),  # Largest terminals
        "lng_fullness_pct": (0, 100),
        "lng_send_out_gwh": (0, 500),  # Daily
        "lng_ship_arrivals": (0, 50),  # Monthly
        
        # Transmission metrics
        "line_voltage_kv": (1, 1200),  # Up to UHVDC
        "line_capacity_mw": (10, 12000),  # Largest HVDC links
        "line_length_km": (0.1, 3000),  # Longest lines
        "utilization_pct": (-100, 100),  # Can flow both ways
        
        # Renewable resource metrics
        "solar_ghi_annual_kwh_m2": (500, 2800),  # Global range
        "wind_speed_100m_avg_ms": (0, 25),  # Hurricane-force excluded
    }
    
    # Known geographic bounds for countries
    COUNTRY_BOUNDS = {
        "US": {"lat": (24.5, 49.4), "lon": (-125.0, -66.9)},
        "DE": {"lat": (47.3, 55.1), "lon": (5.9, 15.0)},
        "FR": {"lat": (41.3, 51.1), "lon": (-5.1, 9.6)},
        "ES": {"lat": (35.9, 43.8), "lon": (-9.3, 3.3)},
        "UK": {"lat": (49.9, 60.9), "lon": (-8.6, 1.8)},
        "NO": {"lat": (57.9, 71.2), "lon": (4.5, 31.2)},
        "IT": {"lat": (35.5, 47.1), "lon": (6.6, 18.5)},
        "PL": {"lat": (49.0, 54.8), "lon": (14.1, 24.1)},
    }
    
    def __init__(self, tolerance: float = 0.1):
        """
        Initialize validator.
        
        Args:
            tolerance: Relative tolerance for cross-source comparison (10% default)
        """
        self.tolerance = tolerance
        self._reference_data: Dict[str, Any] = {}
        self._reconciliation_cache: Dict[str, List[Any]] = defaultdict(list)
    
    def validate_event(self, event: Dict[str, Any]) -> QualityReport:
        """
        Validate a single infrastructure event.
        
        Args:
            event: Event dictionary to validate
            
        Returns:
            QualityReport with validation results
        """
        report = QualityReport(total_records=1, valid_records=1)
        
        # Check completeness
        self._check_completeness(event, report)
        
        # Check validity
        self._check_validity(event, report)
        
        # Check geographic accuracy
        self._check_geographic_accuracy(event, report)
        
        # Check timeliness
        self._check_timeliness(event, report)
        
        return report
    
    def validate_asset(self, asset: InfrastructureAsset) -> QualityReport:
        """
        Validate an infrastructure asset.
        
        Args:
            asset: Infrastructure asset to validate
            
        Returns:
            QualityReport with validation results
        """
        report = QualityReport(total_records=1, valid_records=1)
        
        # Check basic asset properties
        if not asset.asset_id:
            report.add_issue(QualityIssue(
                severity="error",
                dimension="completeness",
                field="asset_id",
                message="Missing asset ID",
            ))
        
        if not asset.name:
            report.add_issue(QualityIssue(
                severity="warning",
                dimension="completeness",
                field="name",
                message="Missing asset name",
                asset_id=asset.asset_id,
            ))
        
        # Validate location
        try:
            if asset.location:
                self._validate_location(asset.location, asset.country, report, asset.asset_id)
        except Exception as e:
            report.add_issue(QualityIssue(
                severity="error",
                dimension="validity",
                field="location",
                message=f"Invalid location: {e}",
                asset_id=asset.asset_id,
            ))
        
        # Type-specific validation
        if isinstance(asset, PowerPlant):
            self._validate_power_plant(asset, report)
        elif isinstance(asset, LNGTerminal):
            self._validate_lng_terminal(asset, report)
        elif isinstance(asset, TransmissionLine):
            self._validate_transmission_line(asset, report)
        
        return report
    
    def reconcile_sources(
        self,
        primary_data: Dict[str, Any],
        secondary_data: Dict[str, Any],
        reconciliation_fields: List[str]
    ) -> Tuple[Dict[str, Any], QualityReport]:
        """
        Reconcile data from multiple sources.
        
        Args:
            primary_data: Primary/trusted source data
            secondary_data: Secondary source data to reconcile
            reconciliation_fields: Fields to compare and reconcile
            
        Returns:
            Tuple of (reconciled_data, quality_report)
        """
        report = QualityReport(total_records=2)
        reconciled = primary_data.copy()
        
        for field in reconciliation_fields:
            primary_val = primary_data.get(field)
            secondary_val = secondary_data.get(field)
            
            if primary_val is None and secondary_val is not None:
                # Fill missing data from secondary source
                reconciled[field] = secondary_val
                report.metrics[f"{field}_filled"] = 1
                
            elif primary_val is not None and secondary_val is not None:
                # Compare values
                if isinstance(primary_val, (int, float)) and isinstance(secondary_val, (int, float)):
                    # Numeric comparison with tolerance
                    rel_diff = abs(primary_val - secondary_val) / max(abs(primary_val), 1e-10)
                    
                    if rel_diff > self.tolerance:
                        report.add_issue(QualityIssue(
                            severity="warning",
                            dimension="consistency",
                            field=field,
                            message=f"Value mismatch: {rel_diff:.1%} difference",
                            value=secondary_val,
                            expected=primary_val,
                        ))
                        
                        # Use average for reconciliation
                        reconciled[field] = (primary_val + secondary_val) / 2
                        report.metrics[f"{field}_reconciled"] = 1
                
                elif primary_val != secondary_val:
                    # Non-numeric comparison
                    report.add_issue(QualityIssue(
                        severity="info",
                        dimension="consistency",
                        field=field,
                        message="Value mismatch between sources",
                        value=secondary_val,
                        expected=primary_val,
                    ))
        
        report.valid_records = 2 if report.quality_score > 80 else 1
        return reconciled, report
    
    def aggregate_quality_reports(self, reports: List[QualityReport]) -> QualityReport:
        """Aggregate multiple quality reports into a summary."""
        
        if not reports:
            return QualityReport()
        
        total = QualityReport()
        total.total_records = sum(r.total_records for r in reports)
        total.valid_records = sum(r.valid_records for r in reports)
        
        # Aggregate issues by severity
        for report in reports:
            total.issues.extend(report.issues)
        
        # Calculate aggregate metrics
        for report in reports:
            for metric, value in report.metrics.items():
                if metric in total.metrics:
                    total.metrics[metric] += value
                else:
                    total.metrics[metric] = value
        
        # Add summary metrics
        total.metrics["avg_quality_score"] = np.mean([r.quality_score for r in reports])
        total.metrics["min_quality_score"] = min(r.quality_score for r in reports)
        total.metrics["max_quality_score"] = max(r.quality_score for r in reports)
        
        return total
    
    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    
    def _check_completeness(self, event: Dict[str, Any], report: QualityReport) -> None:
        """Check if all required fields are present."""
        
        required_fields = [
            "event_time_utc", "market", "product", "instrument_id",
            "value", "unit", "source"
        ]
        
        for field in required_fields:
            if field not in event or event[field] is None:
                report.add_issue(QualityIssue(
                    severity="error",
                    dimension="completeness",
                    field=field,
                    message=f"Missing required field: {field}",
                ))
    
    def _check_validity(self, event: Dict[str, Any], report: QualityReport) -> None:
        """Check if values are within expected ranges."""
        
        metric = event.get("product", "")
        value = event.get("value")
        
        if value is None:
            return
        
        # Check against known ranges
        for pattern, (min_val, max_val) in self.METRIC_RANGES.items():
            if pattern in metric:
                if not min_val <= value <= max_val:
                    report.add_issue(QualityIssue(
                        severity="warning" if 0 <= value <= max_val * 1.5 else "error",
                        dimension="validity",
                        field="value",
                        message=f"Value out of expected range [{min_val}, {max_val}]",
                        value=value,
                        expected=f"{min_val}-{max_val}",
                    ))
                break
    
    def _check_geographic_accuracy(self, event: Dict[str, Any], report: QualityReport) -> None:
        """Validate geographic data."""
        
        metadata = event.get("metadata", {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                return
        
        coords = metadata.get("coordinates", {})
        if coords and "lat" in coords and "lon" in coords:
            try:
                location = GeoLocation(coords["lat"], coords["lon"])
                
                # Check against country bounds if available
                location_code = event.get("location_code", "")
                country = location_code.split("_")[0] if "_" in location_code else location_code
                
                if country in self.COUNTRY_BOUNDS:
                    bounds = self.COUNTRY_BOUNDS[country]
                    if not (bounds["lat"][0] <= location.lat <= bounds["lat"][1] and
                           bounds["lon"][0] <= location.lon <= bounds["lon"][1]):
                        report.add_issue(QualityIssue(
                            severity="warning",
                            dimension="accuracy",
                            field="coordinates",
                            message=f"Location outside expected bounds for {country}",
                            value=f"({location.lat}, {location.lon})",
                        ))
                        
            except ValueError as e:
                report.add_issue(QualityIssue(
                    severity="error",
                    dimension="validity",
                    field="coordinates",
                    message=f"Invalid coordinates: {e}",
                ))
    
    def _check_timeliness(self, event: Dict[str, Any], report: QualityReport) -> None:
        """Check data freshness."""
        
        event_time = event.get("event_time_utc")
        if not event_time:
            return
        
        try:
            if isinstance(event_time, int):
                event_dt = datetime.fromtimestamp(event_time / 1000, tz=timezone.utc)
            else:
                event_dt = datetime.fromisoformat(str(event_time))
            
            age = datetime.now(timezone.utc) - event_dt
            
            # Check if data is too old
            if age > timedelta(days=30):
                report.add_issue(QualityIssue(
                    severity="warning",
                    dimension="timeliness",
                    field="event_time_utc",
                    message=f"Data is {age.days} days old",
                    value=event_dt.isoformat(),
                ))
            
            # Check if data is from the future
            if age < timedelta(0):
                report.add_issue(QualityIssue(
                    severity="error",
                    dimension="timeliness",
                    field="event_time_utc",
                    message="Event time is in the future",
                    value=event_dt.isoformat(),
                ))
                
        except Exception as e:
            report.add_issue(QualityIssue(
                severity="error",
                dimension="validity",
                field="event_time_utc",
                message=f"Invalid timestamp: {e}",
            ))
    
    def _validate_location(
        self,
        location: GeoLocation,
        country: str,
        report: QualityReport,
        asset_id: Optional[str] = None
    ) -> None:
        """Validate geographic location."""
        
        # Check if location is in expected country bounds
        if country in self.COUNTRY_BOUNDS:
            bounds = self.COUNTRY_BOUNDS[country]
            if not (bounds["lat"][0] <= location.lat <= bounds["lat"][1] and
                   bounds["lon"][0] <= location.lon <= bounds["lon"][1]):
                report.add_issue(QualityIssue(
                    severity="warning",
                    dimension="accuracy",
                    field="location",
                    message=f"Location outside {country} bounds",
                    asset_id=asset_id,
                    value=f"({location.lat}, {location.lon})",
                ))
    
    def _validate_power_plant(self, plant: PowerPlant, report: QualityReport) -> None:
        """Validate power plant specific attributes."""
        
        # Check capacity
        if plant.capacity_mw <= 0:
            report.add_issue(QualityIssue(
                severity="error",
                dimension="validity",
                field="capacity_mw",
                message="Power plant capacity must be positive",
                asset_id=plant.asset_id,
                value=plant.capacity_mw,
            ))
        
        # Check capacity by fuel type
        fuel_capacity_limits = {
            FuelType.NUCLEAR: (100, 10000),
            FuelType.COAL: (1, 5000),
            FuelType.NATURAL_GAS: (1, 6000),
            FuelType.HYDRO: (0.1, 25000),
            FuelType.WIND: (0.1, 1000),
            FuelType.SOLAR: (0.1, 2500),
        }
        
        if plant.primary_fuel in fuel_capacity_limits:
            min_cap, max_cap = fuel_capacity_limits[plant.primary_fuel]
            if not min_cap <= plant.capacity_mw <= max_cap:
                report.add_issue(QualityIssue(
                    severity="warning",
                    dimension="validity",
                    field="capacity_mw",
                    message=f"Unusual capacity for {plant.primary_fuel.value} plant",
                    asset_id=plant.asset_id,
                    value=plant.capacity_mw,
                    expected=f"{min_cap}-{max_cap} MW",
                ))
        
        # Check capacity factor
        if plant.capacity_factor is not None:
            expected_cf_ranges = {
                FuelType.NUCLEAR: (0.8, 0.95),
                FuelType.COAL: (0.4, 0.8),
                FuelType.NATURAL_GAS: (0.3, 0.7),
                FuelType.HYDRO: (0.3, 0.6),
                FuelType.WIND: (0.2, 0.5),
                FuelType.SOLAR: (0.15, 0.35),
            }
            
            if plant.primary_fuel in expected_cf_ranges:
                min_cf, max_cf = expected_cf_ranges[plant.primary_fuel]
                if not min_cf <= plant.capacity_factor <= max_cf:
                    report.add_issue(QualityIssue(
                        severity="info",
                        dimension="validity",
                        field="capacity_factor",
                        message=f"Unusual capacity factor for {plant.primary_fuel.value}",
                        asset_id=plant.asset_id,
                        value=plant.capacity_factor,
                        expected=f"{min_cf}-{max_cf}",
                    ))
    
    def _validate_lng_terminal(self, terminal: LNGTerminal, report: QualityReport) -> None:
        """Validate LNG terminal specific attributes."""
        
        # Check storage capacity
        if terminal.storage_capacity_gwh <= 0:
            report.add_issue(QualityIssue(
                severity="error",
                dimension="validity",
                field="storage_capacity_gwh",
                message="LNG storage capacity must be positive",
                asset_id=terminal.asset_id,
                value=terminal.storage_capacity_gwh,
            ))
        
        # Check regasification capacity
        if terminal.regasification_capacity_gwh_d is not None:
            if terminal.regasification_capacity_gwh_d <= 0:
                report.add_issue(QualityIssue(
                    severity="error",
                    dimension="validity",
                    field="regasification_capacity_gwh_d",
                    message="Regasification capacity must be positive",
                    asset_id=terminal.asset_id,
                    value=terminal.regasification_capacity_gwh_d,
                ))
            
            # Check if send-out exceeds regasification capacity
            if terminal.send_out_capacity_gwh_d is not None:
                if terminal.send_out_capacity_gwh_d > terminal.regasification_capacity_gwh_d * 1.1:
                    report.add_issue(QualityIssue(
                        severity="warning",
                        dimension="consistency",
                        field="send_out_capacity_gwh_d",
                        message="Send-out capacity exceeds regasification capacity",
                        asset_id=terminal.asset_id,
                        value=terminal.send_out_capacity_gwh_d,
                        expected=f"<= {terminal.regasification_capacity_gwh_d}",
                    ))
    
    def _validate_transmission_line(self, line: TransmissionLine, report: QualityReport) -> None:
        """Validate transmission line specific attributes."""
        
        # Check voltage
        if line.voltage_kv <= 0:
            report.add_issue(QualityIssue(
                severity="error",
                dimension="validity",
                field="voltage_kv",
                message="Transmission voltage must be positive",
                asset_id=line.asset_id,
                value=line.voltage_kv,
            ))
        
        # Check capacity
        if line.capacity_mw <= 0:
            report.add_issue(QualityIssue(
                severity="error",
                dimension="validity",
                field="capacity_mw",
                message="Transmission capacity must be positive",
                asset_id=line.asset_id,
                value=line.capacity_mw,
            ))
        
        # Validate line length if calculated
        if line.length_km is not None:
            calculated_length = line.from_location.distance_to(line.to_location)
            if abs(line.length_km - calculated_length) / calculated_length > 0.2:
                report.add_issue(QualityIssue(
                    severity="info",
                    dimension="accuracy",
                    field="length_km",
                    message="Line length differs from geographic distance",
                    asset_id=line.asset_id,
                    value=line.length_km,
                    expected=f"~{calculated_length:.0f} km",
                ))
