Airflow Variables & Connections Templates

Use these variables to configure ingestion DAGs without code changes.

Variables (examples)
- NOAA_CDO_TOKEN: your CDO API token
- NOAA_CDO_DATASET: GHCND
- NOAA_CDO_LOCATIONID: FIPS:06
- NOAA_CDO_DATATYPES: TAVG,PRCP,AWND
- NOAA_CDO_STARTDATE / NOAA_CDO_ENDDATE: 2024-01-01 / 2024-01-31
- WB_COUNTRY: WLD (or USA, DEU, etc.)
- WB_ENERGY_INDICATORS: EG.USE.PCAP.KG.OE,EG.ELC.ACCS.ZS
- WB_ECON_INDICATORS: NY.GDP.PCAP.CD,FP.CPI.TOTL.ZG
- WB_START_YEAR / WB_END_YEAR: 2010 / 2025
- WB_BACKFILL_START_YEAR / WB_BACKFILL_END_YEAR / WB_BACKFILL_STEP: 2000 / 2025 / 10
- KAFKA_BOOTSTRAP: kafka:9092

EIA Open Data (variables)
- EIA_API_BASE: https://api.eia.gov/v2
- EIA_API_KEY: <your key>
- EIA_DATASETS: (optional) csv paths, e.g. electricity/retail-sales,petroleum/pri/gnd/data
- EIA_QUERIES: JSON array of advanced query specs
  - See general template: platform/data/ingestion-orch/templates/eia_queries.sample.json
- EIA_STORAGE_QUERIES: JSON array tailored for natural gas storage series (Total Lower 48 + regions, injections/withdrawals)
  - See storage template: platform/data/ingestion-orch/templates/eia_storage_queries.sample.json

Load EIA queries from file
- airflow variables set EIA_QUERIES "$(cat platform/data/ingestion-orch/templates/eia_queries.sample.json)"
- airflow variables set EIA_STORAGE_QUERIES "$(cat platform/data/ingestion-orch/templates/eia_storage_queries.sample.json)"

AGSI+ (GIE) Gas Storage (variables)
- AGSI_API_BASE: https://agsi.gie.eu/api/v1
- AGSI_API_KEY: <your key>
- AGSI_GRANULARITY: facility|country|eu (default: facility)
- AGSI_ENTITIES: optional CSV of facility or country codes
- AGSI_INCLUDE_ROLLUPS: true|false (facility-level roll-ups to country/EU)
- AGSI_SCHEDULE: cron expr (default: 0 6 * * *)

Coal Stockpile Monitoring
- SATELLITE_INTEL_BASE: http://satellite-intel:8025
- COAL_SITES_CSV: path to site registry CSV (default: platform/data/reference/coal_sites.csv)
- COAL_STOCKPILE_SCHEDULE: cron (default: 0 5 * * MON)

LNG Vessel Tracking (provider-agnostic stub)
- LNG_PROVIDER: adapter identifier (default: adapter_stub)
- LNG_LOOKBACK_HOURS: integer lookback window for arrivals/departures (default: 24)
- LNG_TERMINAL_FILE: reference GeoJSON for terminal polygons (default: platform/data/reference/lng_terminals.geojson)

ENTSO-E (variables)
- ENTSOE_API_BASE: https://web-api.tp.entsoe.eu
- ENTSOE_SECURITY_TOKEN: <your token>
- ENTSOE_AREA: 10Y1001A1001A83F (Germany-Lux)
- ENTSOE_OUT_AREA: (optional, for flows)
- ENTSOE_MODES: load,generation,flows,da_price
- ENTSOE_PERIOD_START / ENTSOE_PERIOD_END: 202501010000 / 202501022300 (UTC)
- ENTSOE_BACKFILL_START / ENTSOE_BACKFILL_END: 202501010000 / 202501070000
- ENTSOE_BACKFILL_STEP_HOURS: 24
- ENTSOE_AREA_NAMES: JSON mapping EIC→friendly (see templates/entsoe_area_names.sample.json)

Load ENTSO-E area names from template
- airflow variables set ENTSOE_AREA_NAMES "$(cat platform/data/ingestion-orch/templates/entsoe_area_names.sample.json)"

US Census (variables)
- CENSUS_API_BASE: https://api.census.gov/data
- CENSUS_API_KEY: <your key>
- CENSUS_QUERIES: JSON array of specs (dataset, variables, geo_for, geo_in, aliases, entity_field)
- CENSUS_SCHEDULE: @monthly
- Backfill (optional): CENSUS_BACKFILL_START_YEAR / CENSUS_BACKFILL_END_YEAR

Load Census queries from template
- airflow variables set CENSUS_QUERIES "$(cat platform/data/ingestion-orch/templates/census_queries.sample.json)"

UN Data (variables)
- UN_MODE: csv|sdmx (default csv)
- UN_DOWNLOADS: JSON array of CSV specs
- UN_SDMX_QUERIES: JSON array of SDMX specs
- UN_SCHEDULE: @monthly
- UN_BACKFILL_START_YEAR / UN_BACKFILL_END_YEAR / UN_BACKFILL_STEP_YEARS

Templates
- airflow variables set UN_DOWNLOADS "$(cat platform/data/ingestion-orch/templates/un_downloads.sample.json)"
- airflow variables set UN_SDMX_QUERIES "$(cat platform/data/ingestion-orch/templates/un_sdmx_queries.sample.json)"

Eurostat (variables)
- EUROSTAT_SDMX_QUERIES: JSON array [{"dataset":"demo_r_pjangr3","params":{"time":"2020","geo":"DE","sex":"T","age":"Y_GE65"},"variable":"population","unit":"people"}]
- EUROSTAT_BULK_DOWNLOADS: JSON array of TSV specs
- EUROSTAT_SCHEDULE: @monthly
- Backfill: EUROSTAT_BACKFILL_START_YEAR / EUROSTAT_BACKFILL_END_YEAR / EUROSTAT_BACKFILL_STEP_YEARS

Templates
- airflow variables set EUROSTAT_BULK_DOWNLOADS "$(cat platform/data/ingestion-orch/templates/eurostat_bulk_downloads.sample.json)"


CLI examples
airflow variables set NOAA_CDO_TOKEN <token>
airflow variables set KAFKA_BOOTSTRAP kafka:9092
airflow variables set WB_COUNTRY WLD

Connections
- If using HTTP hooks, define conn id with host and extra headers.
- Current DAGs read env vars directly; connections are optional.

Supply Chain Analytics (new DAGs)
- ML_SERVICE_URL: base URL for the ML service (default http://ml-service:8000)
- LNG_EXPORT_TERMINALS / LNG_IMPORT_TERMINALS: CSV lists for LNG optimisation
- COAL_ROUTES / COAL_MULTIMODAL_OPTIONS: CSV or JSON definitions for coal cost refreshes
- PIPELINE_CONGESTION_TARGETS: JSON array with pipeline_id, flow/weather configuration
- SEASONAL_DEMAND_TARGETS: JSON array specifying region, weather, and economic inputs
