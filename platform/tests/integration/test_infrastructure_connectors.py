"""
Integration tests for infrastructure data connectors.
"""

import pytest
import asyncio
import json
from datetime import datetime, date, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock

from data.connectors.external.infrastructure.base import (
    GeoLocation,
    PowerPlant,
    LNGTerminal,
    TransmissionLine,
    FuelType,
    OperationalStatus,
    InfrastructureConnector,
)
from data.connectors.external.infrastructure.alsi_lng_connector import (
    ALSILNGConnector,
    LNGStorageRecord,
)
from data.connectors.external.infrastructure.reexplorer_connector import (
    REExplorerConnector,
    ResourceType,
)
from data.connectors.external.infrastructure.wri_powerplants_connector import (
    WRIPowerPlantsConnector,
    WRIPowerPlant,
)
from data.connectors.external.infrastructure.gem_transmission_connector import (
    GEMTransmissionConnector,
    TransmissionProject,
    ProjectStatus,
)
from data.connectors.external.infrastructure.data_quality import (
    InfrastructureDataValidator,
    QualityReport,
)


class TestInfrastructureBase:
    """Test base infrastructure classes and utilities."""
    
    def test_geo_location_validation(self):
        """Test geographic location validation."""
        # Valid location
        loc = GeoLocation(40.7128, -74.0060)  # New York
        assert loc.lat == 40.7128
        assert loc.lon == -74.0060
        
        # Invalid latitude
        with pytest.raises(ValueError):
            GeoLocation(91, 0)
        
        # Invalid longitude
        with pytest.raises(ValueError):
            GeoLocation(0, 181)
    
    def test_geo_location_distance(self):
        """Test distance calculation between locations."""
        nyc = GeoLocation(40.7128, -74.0060)
        london = GeoLocation(51.5074, -0.1278)
        
        distance = nyc.distance_to(london)
        # Distance should be approximately 5570 km
        assert 5500 < distance < 5600
    
    def test_power_plant_creation(self):
        """Test power plant asset creation."""
        plant = PowerPlant(
            asset_id="TEST_PLANT_001",
            name="Test Solar Farm",
            location=GeoLocation(35.0, -120.0),
            country="US",
            capacity_mw=100.0,
            primary_fuel=FuelType.SOLAR,
            status=OperationalStatus.OPERATIONAL,
            commissioned_date=date(2020, 1, 1),
        )
        
        assert plant.asset_id == "TEST_PLANT_001"
        assert plant.capacity_mw == 100.0
        assert plant.primary_fuel == FuelType.SOLAR
        assert plant.is_operational()
        
        # Test invalid capacity
        with pytest.raises(ValueError):
            PowerPlant(
                asset_id="INVALID",
                name="Invalid Plant",
                location=GeoLocation(0, 0),
                country="XX",
                capacity_mw=-10.0,
                primary_fuel=FuelType.COAL,
            )
    
    def test_lng_terminal_creation(self):
        """Test LNG terminal asset creation."""
        terminal = LNGTerminal(
            asset_id="TEST_LNG_001",
            name="Test LNG Terminal",
            location=GeoLocation(41.0, 2.0),
            country="ES",
            storage_capacity_gwh=500.0,
            regasification_capacity_gwh_d=50.0,
            status=OperationalStatus.OPERATIONAL,
        )
        
        assert terminal.asset_id == "TEST_LNG_001"
        assert terminal.storage_capacity_gwh == 500.0
        assert terminal.to_instrument_id() == "LNG.ES.TEST_LNG_001"
    
    def test_transmission_line_creation(self):
        """Test transmission line asset creation."""
        line = TransmissionLine(
            asset_id="TEST_LINE_001",
            name="Test Transmission Line",
            location=GeoLocation(40.0, -100.0),
            country="US",
            from_location=GeoLocation(39.0, -101.0),
            to_location=GeoLocation(41.0, -99.0),
            voltage_kv=345.0,
            capacity_mw=1000.0,
            line_type="AC",
            status=OperationalStatus.OPERATIONAL,
        )
        
        assert line.asset_id == "TEST_LINE_001"
        assert line.voltage_kv == 345.0
        assert line.capacity_mw == 1000.0
        # Length should be calculated automatically
        assert line.length_km is not None
        assert 200 < line.length_km < 300  # Approximate distance


class TestALSILNGConnector:
    """Test ALSI LNG Inventory connector."""
    
    @pytest.fixture
    def connector_config(self):
        return {
            "source_id": "alsi_lng_inventory_test",
            "api_key": "test_api_key",
            "granularity": "terminal",
            "lookback_days": 7,
            "kafka": {
                "topic": "test.infrastructure",
                "bootstrap_servers": "localhost:9092",
            },
        }
    
    @pytest.fixture
    def mock_alsi_response(self):
        return {
            "data": [
                {
                    "gasDayStart": "2025-10-01",
                    "code": "ES_BARCELONA",
                    "name": "Barcelona",
                    "country": "ES",
                    "lngInventory": 2500.0,
                    "lngInventoryMcm": 1.5,
                    "full": 75.0,
                    "sendOut": 50.0,
                    "numberOfShipArrivals": 2,
                    "storageTankCapacity": 3333.0,
                },
                {
                    "gasDayStart": "2025-10-01",
                    "code": "FR_DUNKERQUE",
                    "name": "Dunkerque",
                    "country": "FR",
                    "lngInventory": 1800.0,
                    "lngInventoryMcm": 1.1,
                    "full": 60.0,
                    "sendOut": 40.0,
                    "numberOfShipArrivals": 1,
                    "storageTankCapacity": 3000.0,
                },
            ]
        }
    
    def test_connector_initialization(self, connector_config):
        """Test ALSI connector initialization."""
        connector = ALSILNGConnector(connector_config)
        
        assert connector.source_id == "alsi_lng_inventory_test"
        assert connector.api_key == "test_api_key"
        assert connector.granularity == "terminal"
        assert len(connector.assets) > 0  # Should have initialized terminals
    
    @patch('requests.Session.get')
    def test_fetch_records(self, mock_get, connector_config, mock_alsi_response):
        """Test fetching LNG records from API."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_alsi_response
        mock_get.return_value = mock_response
        
        connector = ALSILNGConnector(connector_config)
        
        # Test record fetching
        start = datetime.now(timezone.utc) - timedelta(days=1)
        end = datetime.now(timezone.utc)
        
        records = list(connector._fetch_records(start, end))
        
        assert len(records) == 2
        assert records[0].entity == "ES_BARCELONA"
        assert records[0].inventory_gwh == 2500.0
        assert records[0].fullness_pct == 75.0
        assert records[1].entity == "FR_DUNKERQUE"
    
    def test_record_to_events(self, connector_config):
        """Test converting LNG records to events."""
        connector = ALSILNGConnector(connector_config)
        
        record = LNGStorageRecord(
            as_of=datetime(2025, 10, 1, tzinfo=timezone.utc),
            entity="ES_BARCELONA",
            entity_type="terminal",
            inventory_gwh=2500.0,
            inventory_mcm=1.5,
            fullness_pct=75.0,
            send_out_gwh=50.0,
            ship_arrivals=2,
            capacity_gwh=3333.0,
            capacity_mcm=2.0,
            region="ES",
            raw={},
        )
        
        events = list(connector._record_to_events(record))
        
        # Should generate events for each non-null metric
        assert len(events) == 5  # inventory_gwh, inventory_mcm, fullness_pct, send_out_gwh, ship_arrivals
        
        # Check first event
        inventory_event = events[0]
        assert inventory_event["metric"] == "lng_inventory_gwh"
        assert inventory_event["value"] == 2500.0
        assert inventory_event["unit"] == "GWh"
    
    def test_data_aggregation(self, connector_config):
        """Test terminal data aggregation to country level."""
        connector = ALSILNGConnector(connector_config)
        
        records = [
            LNGStorageRecord(
                as_of=datetime(2025, 10, 1, tzinfo=timezone.utc),
                entity="ES_BARCELONA",
                entity_type="terminal",
                inventory_gwh=2500.0,
                fullness_pct=75.0,
                capacity_gwh=3333.0,
                region="ES",
                raw={},
            ),
            LNGStorageRecord(
                as_of=datetime(2025, 10, 1, tzinfo=timezone.utc),
                entity="ES_BILBAO",
                entity_type="terminal",
                inventory_gwh=1500.0,
                fullness_pct=60.0,
                capacity_gwh=2500.0,
                region="ES",
                raw={},
            ),
        ]
        
        aggregated = connector._aggregate_records(records, target="country")
        
        assert len(aggregated) == 1
        assert aggregated[0].entity == "ES"
        assert aggregated[0].inventory_gwh == 4000.0  # Sum of both terminals
        assert aggregated[0].capacity_gwh == 5833.0  # Sum of capacities
        assert aggregated[0].fullness_pct == 67.5  # Average of fullness


class TestREExplorerConnector:
    """Test REexplorer renewable resource connector."""
    
    @pytest.fixture
    def connector_config(self):
        return {
            "source_id": "reexplorer_test",
            "api_key": "test_api_key",
            "regions": ["US"],
            "resource_types": ["solar_ghi", "wind_speed_100m"],
            "include_projects": True,
            "grid_bounds": {
                "lat_min": 30.0,
                "lat_max": 35.0,
                "lon_min": -120.0,
                "lon_max": -115.0,
            },
        }
    
    def test_connector_initialization(self, connector_config):
        """Test REexplorer connector initialization."""
        connector = REExplorerConnector(connector_config)
        
        assert connector.source_id == "reexplorer_test"
        assert connector.regions == ["US"]
        assert "solar_ghi" in connector.resource_types
    
    def test_grid_point_generation(self, connector_config):
        """Test grid point generation for resource assessment."""
        connector = REExplorerConnector(connector_config)
        
        grid_points = connector._generate_grid_points()
        
        # Check grid covers the specified bounds
        assert len(grid_points) > 0
        
        # Check first and last points
        assert grid_points[0] == (30.0, -120.0)
        
        # Check grid spacing
        for i in range(1, len(grid_points)):
            lat_diff = abs(grid_points[i][0] - grid_points[i-1][0])
            lon_diff = abs(grid_points[i][1] - grid_points[i-1][1])
            # Either lat or lon should change by grid resolution
            assert lat_diff == 0 or lat_diff == connector.GRID_RESOLUTION or \
                   lon_diff == 0 or lon_diff == connector.GRID_RESOLUTION
    
    def test_resource_assessment_creation(self, connector_config):
        """Test creating resource assessment data."""
        connector = REExplorerConnector(connector_config)
        
        location = GeoLocation(33.0, -117.0)
        assessment = connector._fetch_solar_resource(location, "solar_ghi")
        
        assert assessment is not None
        assert assessment.resource_type == ResourceType.SOLAR_GHI
        assert assessment.annual_average > 0
        assert len(assessment.monthly_averages) == 12
        assert assessment.unit == "kWh/m2/year"
    
    def test_project_parsing(self, connector_config):
        """Test renewable project parsing."""
        connector = REExplorerConnector(connector_config)
        
        project_data = {
            "project_id": "US_SOLAR_TEST",
            "name": "Test Solar Project",
            "technology": "solar",
            "capacity_mw": 150.0,
            "lat": 34.0,
            "lon": -118.0,
            "country": "US",
            "status": "operational",
            "commissioned_date": "2023-01-01",
            "developer": "Test Developer",
            "capacity_factor": 0.25,
        }
        
        project = connector._parse_project(project_data)
        
        assert project is not None
        assert project.project_id == "US_SOLAR_TEST"
        assert project.capacity_mw == 150.0
        assert project.technology == FuelType.SOLAR
        assert project.status == OperationalStatus.OPERATIONAL
        
        # Check that power plant asset was created
        assert project.project_id in connector.assets


class TestWRIPowerPlantsConnector:
    """Test WRI Global Power Plant Database connector."""
    
    @pytest.fixture
    def connector_config(self):
        return {
            "source_id": "wri_powerplants_test",
            "countries": ["US", "DE"],
            "min_capacity_mw": 100.0,
            "fuel_types": ["wind", "solar"],
            "include_generation": True,
        }
    
    @pytest.fixture
    def sample_csv_row(self):
        return {
            "gppd_idnr": "USA0001234",
            "name": "Test Wind Farm",
            "country": "USA",
            "country_long": "United States",
            "latitude": "40.7128",
            "longitude": "-74.0060",
            "primary_fuel": "Wind",
            "other_fuel1": "",
            "capacity_mw": "250.5",
            "owner": "Test Energy Corp",
            "commissioning_year": "2018",
            "generation_gwh_2017": "650.2",
        }
    
    def test_connector_initialization(self, connector_config):
        """Test WRI connector initialization."""
        connector = WRIPowerPlantsConnector(connector_config)
        
        assert connector.source_id == "wri_powerplants_test"
        assert connector.countries == ["US", "DE"]
        assert connector.min_capacity_mw == 100.0
    
    def test_plant_parsing(self, connector_config, sample_csv_row):
        """Test parsing power plant from CSV row."""
        connector = WRIPowerPlantsConnector(connector_config)
        
        plant = connector._parse_plant_row(sample_csv_row)
        
        assert plant is not None
        assert plant.gppd_id == "USA0001234"
        assert plant.name == "Test Wind Farm"
        assert plant.capacity_mw == 250.5
        assert plant.primary_fuel == "Wind"
        assert plant.generation_gwh[2017] == 650.2
    
    def test_plant_filtering(self, connector_config):
        """Test plant filtering based on criteria."""
        connector = WRIPowerPlantsConnector(connector_config)
        
        # Plant that should be included
        plant1 = WRIPowerPlant(
            gppd_id="USA0001",
            name="Large Wind Farm",
            country="USA",
            country_long="United States",
            latitude=40.0,
            longitude=-100.0,
            primary_fuel="Wind",
            other_fuels=[],
            capacity_mw=200.0,
            owner=None,
            source=None,
            url=None,
            commissioning_year=2020,
            retirement_year=None,
            generation_gwh={},
            estimated_generation={},
            raw_data={},
        )
        
        # Plant that should be excluded (wrong country)
        plant2 = WRIPowerPlant(
            gppd_id="CHN0001",
            name="China Wind Farm",
            country="CHN",
            country_long="China",
            latitude=35.0,
            longitude=105.0,
            primary_fuel="Wind",
            other_fuels=[],
            capacity_mw=300.0,
            owner=None,
            source=None,
            url=None,
            commissioning_year=2020,
            retirement_year=None,
            generation_gwh={},
            estimated_generation={},
            raw_data={},
        )
        
        # Plant that should be excluded (too small)
        plant3 = WRIPowerPlant(
            gppd_id="USA0002",
            name="Small Solar",
            country="USA",
            country_long="United States",
            latitude=35.0,
            longitude=-110.0,
            primary_fuel="Solar",
            other_fuels=[],
            capacity_mw=50.0,
            owner=None,
            source=None,
            url=None,
            commissioning_year=2020,
            retirement_year=None,
            generation_gwh={},
            estimated_generation={},
            raw_data={},
        )
        
        assert connector._should_include_plant(plant1) is True
        assert connector._should_include_plant(plant2) is False
        assert connector._should_include_plant(plant3) is False


class TestGEMTransmissionConnector:
    """Test Global Energy Monitor transmission connector."""
    
    @pytest.fixture
    def connector_config(self):
        return {
            "source_id": "gem_transmission_test",
            "api_key": "test_api_key",
            "regions": ["US", "EU"],
            "min_voltage_kv": 230,
            "include_projects": True,
            "project_statuses": ["construction", "commissioned"],
        }
    
    def test_connector_initialization(self, connector_config):
        """Test GEM connector initialization."""
        connector = GEMTransmissionConnector(connector_config)
        
        assert connector.source_id == "gem_transmission_test"
        assert connector.min_voltage_kv == 230
        assert connector.include_projects is True
    
    def test_voltage_classification(self, connector_config):
        """Test voltage level classification."""
        connector = GEMTransmissionConnector(connector_config)
        
        assert connector._classify_voltage(138) == "HV"
        assert connector._classify_voltage(345) == "EHV"
        assert connector._classify_voltage(765) == "EHV"
        assert connector._classify_voltage(1000) == "UHV"
    
    def test_project_progress_calculation(self, connector_config):
        """Test infrastructure project progress calculation."""
        connector = GEMTransmissionConnector(connector_config)
        
        # Commissioned project
        project1 = TransmissionProject(
            project_id="PROJ_001",
            name="Test Project 1",
            project_type="line",
            countries=["US"],
            status=ProjectStatus.COMMISSIONED,
            voltage_kv=500.0,
            capacity_mw=2000.0,
            length_km=500.0,
            line_type="DC",
            developer="Test Corp",
            estimated_cost_million_usd=1000.0,
            start_year=2020,
            completion_year=2023,
            coordinates=[],
        )
        
        assert connector._calculate_project_progress(project1) == 100.0
        
        # Project under construction
        current_year = datetime.now().year
        project2 = TransmissionProject(
            project_id="PROJ_002",
            name="Test Project 2",
            project_type="line",
            countries=["US"],
            status=ProjectStatus.CONSTRUCTION,
            voltage_kv=345.0,
            capacity_mw=1500.0,
            length_km=300.0,
            line_type="AC",
            developer="Test Corp",
            estimated_cost_million_usd=500.0,
            start_year=current_year - 2,
            completion_year=current_year + 2,
            coordinates=[],
        )
        
        progress = connector._calculate_project_progress(project2)
        assert 0 < progress < 100


class TestDataQuality:
    """Test infrastructure data quality validation."""
    
    @pytest.fixture
    def validator(self):
        return InfrastructureDataValidator(tolerance=0.1)
    
    def test_event_validation(self, validator):
        """Test infrastructure event validation."""
        # Valid event
        valid_event = {
            "event_time_utc": int(datetime.now(timezone.utc).timestamp() * 1000),
            "market": "infra",
            "product": "lng_inventory_gwh",
            "instrument_id": "ALSI.ES_BARCELONA.LNG_TERMINAL",
            "value": 2500.0,
            "unit": "GWh",
            "source": "alsi_lng_inventory",
        }
        
        report = validator.validate_event(valid_event)
        assert report.quality_score > 90
        
        # Event with missing fields
        invalid_event = {
            "market": "infra",
            "product": "lng_inventory_gwh",
            # Missing required fields
        }
        
        report = validator.validate_event(invalid_event)
        assert report.quality_score < 50
        assert len(report.issues) > 0
    
    def test_asset_validation(self, validator):
        """Test infrastructure asset validation."""
        # Valid power plant
        plant = PowerPlant(
            asset_id="PLANT_001",
            name="Test Plant",
            location=GeoLocation(40.0, -100.0),
            country="US",
            capacity_mw=500.0,
            primary_fuel=FuelType.NATURAL_GAS,
            capacity_factor=0.5,
            status=OperationalStatus.OPERATIONAL,
        )
        
        report = validator.validate_asset(plant)
        assert report.quality_score == 100
        
        # Power plant with invalid capacity factor
        plant_invalid = PowerPlant(
            asset_id="PLANT_002",
            name="Invalid Plant",
            location=GeoLocation(40.0, -100.0),
            country="US",
            capacity_mw=500.0,
            primary_fuel=FuelType.SOLAR,
            capacity_factor=0.8,  # Too high for solar
            status=OperationalStatus.OPERATIONAL,
        )
        
        report = validator.validate_asset(plant_invalid)
        assert report.quality_score < 100
        assert any(issue.field == "capacity_factor" for issue in report.issues)
    
    def test_data_reconciliation(self, validator):
        """Test reconciling data from multiple sources."""
        primary_data = {
            "capacity_mw": 1000.0,
            "fuel_type": "wind",
            "location": "Texas",
        }
        
        secondary_data = {
            "capacity_mw": 1050.0,  # 5% difference - within tolerance
            "fuel_type": "wind",
            "location": "TX",
            "operator": "Wind Corp",  # Additional data
        }
        
        reconciled, report = validator.reconcile_sources(
            primary_data,
            secondary_data,
            ["capacity_mw", "fuel_type", "operator"]
        )
        
        # Should use average for numeric fields within tolerance
        assert reconciled["capacity_mw"] == 1025.0
        # Should fill missing data
        assert reconciled["operator"] == "Wind Corp"
        # Non-reconciled fields should match primary
        assert reconciled["location"] == "Texas"
        
        assert report.quality_score > 80


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for complete data flow."""
    
    async def test_alsi_to_kafka_flow(self):
        """Test complete flow from ALSI API to Kafka."""
        # This would require a test Kafka instance
        # For now, we'll mock the Kafka producer
        
        config = {
            "source_id": "alsi_lng_test",
            "api_key": "test_key",
            "granularity": "terminal",
            "kafka": {
                "topic": "test.infrastructure",
                "bootstrap_servers": "localhost:9092",
            },
        }
        
        with patch('data.connectors.base.KafkaProducer') as mock_kafka:
            mock_producer = MagicMock()
            mock_kafka.return_value = mock_producer
            
            connector = ALSILNGConnector(config)
            
            # Mock some events
            events = [
                {
                    "event_time_utc": int(datetime.now(timezone.utc).timestamp() * 1000),
                    "market": "infra",
                    "product": "lng_inventory_gwh",
                    "instrument_id": "ALSI.ES_BARCELONA.LNG_TERMINAL",
                    "value": 2500.0,
                    "unit": "GWh",
                    "source": "alsi_lng_test",
                    "commodity_type": "gas",
                    "metadata": json.dumps({"terminal": "ES_BARCELONA"}),
                }
            ]
            
            # Emit events
            count = connector.emit(events)
            
            assert count == 1
            assert mock_producer.send.called
            assert mock_producer.flush.called
