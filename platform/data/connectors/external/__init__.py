"""
External data connectors grouped by domain (infrastructure, weather, demographics, economics, geospatial).

Each connector implements the Ingestor interface and emits canonical events to Kafka.
These stubs provide safe mocked data with clear integration points for real APIs.
"""

__all__ = [
    # Infrastructure
    "eia_connector",
    "entsoe_connector",
    "oecd_energy_connector",
    "open_inframap_connector",
    "world_bank_energy_connector",
    # Weather/Climate
    "noaa_cdo_connector",
    "copernicus_cds_connector",
    "era5_connector",
    "nasa_power_connector",
    "openweathermap_connector",
    # Population/Demographics
    "us_census_connector",
    "un_data_connector",
    "worldpop_connector",
    "eurostat_connector",
    # Economics/Trade
    "fred_connector",
    "world_bank_econ_connector",
    "imf_connector",
    "oecd_data_connector",
    "un_comtrade_connector",
    "wto_connector",
    # Geospatial/Environment
    "openstreetmap_connector",
    "natural_earth_connector",
    "gbif_connector",
    "faostat_connector",
]

