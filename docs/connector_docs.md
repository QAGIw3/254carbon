# External Data Connectors — Official Docs & Endpoints

These references were gathered to align each connector with its official data sources. Where appropriate, the connector code is already calling the documented endpoint(s); otherwise, the code contains a safe mock with a clear path to wire up the production API.

Note: Some sources require API subscriptions or whitelisted IPs. If you want me to wire these in now, share keys (or stub them as env vars/secrets) and I’ll finish the integration end-to-end.

## MISO (Midcontinent ISO)
- Portal: https://www.misoenergy.org
- Real-Time Web Displays (Data Broker) index of endpoints: https://api.misoenergy.org/MISORTWDDataBroker/
- Real-time 5-minute consolidated LMP (JSON):
  - `https://api.misoenergy.org/MISORTWDDataBroker/DataBrokerServices.asmx?messageType=getlmpconsolidatedtable&returnType=json`
- Ex-ante (hub) LMP (JSON):
  - `https://api.misoenergy.org/MISORTWDDataBroker/DataBrokerServices.asmx?messageType=getexantelmp&returnType=json`
- Status: Implemented in `platform/data/connectors/miso_connector.py` using the above JSON endpoints with Eastern→UTC normalization.

## CAISO (California ISO)
- OASIS: https://oasis.caiso.com/
- SingleZip API (CSV-in-ZIP): `https://oasis.caiso.com/oasisapi/SingleZip`
  - RTM nodal LMP: `queryname=PRC_RTM_LMP` (e.g., `market_run_id=RTM`)
  - DAM nodal LMP: `queryname=PRC_LMP` (e.g., `market_run_id=DAM`)
- Result format: `resultformat=6` for CSV-in-ZIP
- Status: Implemented in `platform/data/connectors/caiso_connector.py` with entitlement filtering (pilot: hubs only).

## PJM Interconnection
- Data Miner 2: https://dataminer2.pjm.com
- API base: `https://api.pjm.com/api/v1/`
  - Real-time Hourly LMPs: `rt_hrl_lmps`
  - Day-ahead Hourly LMPs: `da_hrl_lmps`
- Auth: Subscription key header `Ocp-Apim-Subscription-Key: <key>`
- Notes: API supports pagination/rowCount; filtering by datetime is available. Provide a key to enable live calls in `platform/data/connectors/pjm_connector.py`.

## ERCOT (Texas)
- ERCOT Public Reports/API: https://api.ercot.com/api/public-reports
- Docs: Accessible via ERCOT’s portal; dataset names and schemas vary by report.
- Common data: SPP (Settlement Point Prices), hub prices, ORDC adders.
- Notes: Some endpoints require browsing dataset catalogs and/or authentication. The connector has mocks; share dataset slugs or docs to finalize live pulls.

## SPP (Southwest Power Pool)
- Marketplace/Portal: https://portal.spp.org/
- Public programmatic access is available, but documentation is hosted behind the portal UI. Typical feeds include RTBM/DAM LMPs and Operating Reserves. The connector currently ships with mocks.
- Notes: Provide API docs/keys (if required) to wire up live RTBM/DAM/OR endpoints.

## NYISO (New York ISO)
- Data & MIS: https://www.nyiso.com/energy-market-operational-data
- Public MIS CSVs (programmatic downloads) are commonly used for RT/DAM LBMPs (zone/gen); exact paths are date-based and documented in NYISO PDFs and web pages.
- Notes: Connector currently uses realistic mocks. If you want a robust MIS CSV ingestor, I can implement the daily path resolution + CSV parsing (RT/DAM zones and gens).

## IESO (Ontario)
- Data Directory: https://www.ieso.ca/en/Power-Data/Data-Directory
- Public reports APIs (JSON/CSV) for HOEP, pre-dispatch, interties, demand.
- Notes: Connector has mocks aligned to real series; with specific endpoints, I can switch to live pulls quickly.

## AESO (Alberta)
- Market/System Reporting: https://www.aeso.ca/market/market-and-system-reporting/
- ETS resources (historical/near-real-time): http://ets.aeso.ca/
- Live API (JSON, common base): https://api.aeso.ca/report/v1 (exact paths may vary by release)
- Auth: Use Authorization: Bearer <token> or x-api-key: <key>
- Connector config keys:
  - `api_base` (default `https://api.aeso.ca/report/v1`)
  - `bearer_token` or `api_key` (+ optional `api_key_header`, default `x-api-key`)
  - `use_live_pool`, `pool_price_endpoint` (default `/price/poolPrice`)
  - `use_live_ail`, `ail_endpoint` (default `/load/albertaInternalLoad`)
- Status: Live API supported in `platform/data/connectors/aeso_connector.py`. Fallback to mocks when disabled or if responses are unrecognized.
- Env vars: `AESO_BEARER_TOKEN` and/or `AESO_API_KEY` can be used instead of placing secrets in config.

## LATAM (CENACE Mexico, ONS Brazil, etc.)
- CENACE (MX): https://www.cenace.gob.mx
- ONS (BR): https://www.ons.org.br
- Notes: Public endpoints vary by ISO and often require scraping/downloads. Connectors currently simulate typical series; I can integrate live feeds with specific dataset URLs or APIs.

---

Implementation status highlights:
- MISO: done (RT nodal + DA ex-ante hub via JSON).
- CAISO: done (OASIS SingleZip CSV-in-ZIP with hub-only entitlement).
- PJM/NYISO/SPP/ERCOT/IESO/AESO: code scaffolds with strong mocks; live APIs ready to be wired once keys/endpoint details are provided.

---

New external connectors (infrastructure, weather, demographics, economics, geospatial):

Energy infrastructure
- US EIA Open Data: https://www.eia.gov/opendata/
  - Class: platform/data/connectors/external/infrastructure/eia_connector.py:EIAOpenDataConnector
  - Notes: Configure `api_key`, `datasets`; emits retail electricity sales and fuel prices.
- ENTSO-E Transparency Platform: https://transparency.entsoe.eu/
  - Class: platform/data/connectors/external/infrastructure/entsoe_connector.py:ENTSOETransparencyConnector
  - Notes: Configure `api_token`, `area` (EIC); emits load, generation, transmission, balancing price.
- OECD Energy Statistics: https://data.oecd.org/energy.htm
  - Class: platform/data/connectors/external/infrastructure/oecd_energy_connector.py:OECDEnergyStatsConnector
  - Notes: Download/SDMX in prod; emits energy balance, household electricity price.
- Open Infrastructure Map: https://openinframap.org/
  - Class: platform/data/connectors/external/infrastructure/open_inframap_connector.py:OpenInfrastructureMapConnector
  - Notes: Overpass/OSM-derived in prod; emits lines km, substation counts, pipelines km.
- World Bank Energy Data: https://datacatalog.worldbank.org/
  - Class: platform/data/connectors/external/infrastructure/world_bank_energy_connector.py:WorldBankEnergyConnector
  - Notes: Configure `indicators`, `country`; emits energy use per capita, electricity access.

Weather and climate
- NOAA Climate Data Online (CDO): https://www.ncdc.noaa.gov/cdo-web/
  - Class: platform/data/connectors/external/weather/noaa_cdo_connector.py:NOAACDOConnector
  - Notes: Configure `token`, `location`; emits temp, wind, precipitation.
- Copernicus Climate Data Store (CDS): https://cds.climate.copernicus.eu/
  - Class: platform/data/connectors/external/weather/copernicus_cds_connector.py:CopernicusCDSConnector
  - Notes: Configure `dataset`, `area`; emits t2m, precip, wind.
- ECMWF ERA5 Reanalysis: https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
  - Class: platform/data/connectors/external/weather/era5_connector.py:ERA5Connector
  - Notes: Hourly global reanalysis; emits t2m, wind, precip.
- NASA POWER: https://power.larc.nasa.gov/
  - Class: platform/data/connectors/external/weather/nasa_power_connector.py:NASAPowerConnector
  - Notes: Configure `site`; emits GHI, wind, temperature.
- OpenWeatherMap: https://openweathermap.org/api
  - Class: platform/data/connectors/external/weather/openweathermap_connector.py:OpenWeatherMapConnector
  - Notes: Configure `api_key`, `location`; emits current conditions.

Population and demographics
- US Census Bureau APIs: https://www.census.gov/data/developers/data-sets.html
  - Class: platform/data/connectors/external/demographics/us_census_connector.py:USCensusConnector
  - Notes: Configure `dataset`, `geo`; emits population, housing units.
- UN Data: https://data.un.org/
  - Class: platform/data/connectors/external/demographics/un_data_connector.py:UNDataConnector
  - Notes: Emits world population and life expectancy.
- WorldPop: https://www.worldpop.org/
  - Class: platform/data/connectors/external/demographics/worldpop_connector.py:WorldPopConnector
  - Notes: Emits total population and density summary.
- Eurostat: https://ec.europa.eu/eurostat/data/database
  - Class: platform/data/connectors/external/demographics/eurostat_connector.py:EurostatConnector
  - Notes: Emits EU population and net migration.

Economics and trade
- FRED: https://fred.stlouisfed.org/
  - Class: platform/data/connectors/external/economics/fred_connector.py:FREDConnector
  - Notes: Configure `api_key`; emits GDP, CPI, fed funds rate.
- World Bank Open Data: https://data.worldbank.org/
  - Class: platform/data/connectors/external/economics/world_bank_econ_connector.py:WorldBankEconomicsConnector
  - Notes: Emits GDP per capita and inflation.
- IMF Data: https://data.imf.org/
  - Class: platform/data/connectors/external/economics/imf_connector.py:IMFConnector
  - Notes: Emits current account and government debt ratios.
- OECD Data: https://data.oecd.org/
  - Class: platform/data/connectors/external/economics/oecd_data_connector.py:OECDDataConnector
  - Notes: Emits unemployment rate and trade balance.
- UN Comtrade: https://comtradeplus.un.org/
  - Class: platform/data/connectors/external/economics/un_comtrade_connector.py:UNComtradeConnector
  - Notes: Emits exports, imports, and trade balance.
- WTO Data Portal: https://timeseries.wto.org/
  - Class: platform/data/connectors/external/economics/wto_connector.py:WTODataConnector
  - Notes: Emits average tariff rate and trade policy index.

Geospatial and environment
- OpenStreetMap: https://www.openstreetmap.org/
  - Class: platform/data/connectors/external/geospatial/openstreetmap_connector.py:OpenStreetMapConnector
  - Notes: Overpass/planet in prod; emits road/rail/power line km.
- Natural Earth Data: https://www.naturalearthdata.com/
  - Class: platform/data/connectors/external/geospatial/natural_earth_connector.py:NaturalEarthConnector
  - Notes: Emits country and populated places counts by scale.
- GBIF: https://www.gbif.org/
  - Class: platform/data/connectors/external/geospatial/gbif_connector.py:GBIFConnector
  - Notes: Emits species occurrence counts and biodiversity index.
- FAO FAOSTAT: https://www.fao.org/faostat/en/
  - Class: platform/data/connectors/external/geospatial/faostat_connector.py:FAOSTATConnector
  - Notes: Emits cereal production and agricultural land share.

All new connectors default to emitting JSON records to Kafka topic `market.fundamentals`.
Swap to real APIs by providing credentials in config and replacing the mocked pull step.
