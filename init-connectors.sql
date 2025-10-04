-- Initialize MISO connector
INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json)
VALUES ('miso_rt_lmp', 'iso', 'https://api.misoenergy.org', 'active',
        '{"market_type": "RT", "kafka_topic": "power.ticks.v1"}');

-- Initialize CAISO connector with restrictions
INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json)
VALUES ('caiso_rt_lmp', 'iso', 'https://oasis.caiso.com', 'active',
        '{"market_type": "RT", "kafka_topic": "power.ticks.v1", "restricted_nodes": ["SP15", "NP15", "ZP26"]}');

-- External fundamentals connectors
INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json)
VALUES ('nasa_power', 'weather', 'https://power.larc.nasa.gov/api', 'active', '{"kafka_topic":"market.fundamentals"}');

INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json)
VALUES ('noaa_cdo', 'weather', 'https://www.ncdc.noaa.gov/cdo-web/api/v2', 'active', '{"kafka_topic":"market.fundamentals"}');

INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json)
VALUES ('eia_open_data', 'infra', 'https://api.eia.gov/v2', 'active', '{"kafka_topic":"market.fundamentals"}');

INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json)
VALUES ('entsoe_transparency', 'infra', 'https://web-api.tp.entsoe.eu', 'active', '{"kafka_topic":"market.fundamentals"}');

INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json)
VALUES ('world_bank_energy', 'infra', 'https://api.worldbank.org/v2', 'active', '{"kafka_topic":"market.fundamentals"}');

INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json)
VALUES ('world_bank_econ', 'economics', 'https://api.worldbank.org/v2', 'active', '{"kafka_topic":"market.fundamentals"}');

INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json)
VALUES ('fred', 'economics', 'https://api.stlouisfed.org/fred', 'active', '{"kafka_topic":"market.fundamentals"}');

INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json)
VALUES ('osm', 'geospatial', 'https://overpass-api.de/api/interpreter', 'active', '{"kafka_topic":"market.fundamentals"}');

INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json)
VALUES ('gbif', 'environmental', 'https://api.gbif.org/v1', 'active', '{"kafka_topic":"market.fundamentals"}');

INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json)
VALUES ('faostat', 'environmental', 'https://fenixservices.fao.org/faostat/api/v1/en', 'active', '{"kafka_topic":"market.fundamentals"}');

-- OECD Energy Statistics (download-driven; configure downloads or sdmx_queries at runtime)
INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json)
VALUES (
  'oecd_energy',
  'infra',
  'https://data.oecd.org/energy.htm',
  'active',
  '{"kafka_topic":"market.fundamentals","mode":"csv","downloads":[],"sdmx_queries":[]}'
);

-- Open Infrastructure Map (Overpass snapshot)
INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json)
VALUES (
  'open_inframap',
  'infra',
  'https://openinframap.org/',
  'active',
  '{"kafka_topic":"market.fundamentals","live":false,"region":"WORLD"}'
);

-- US Census Bureau APIs (population, housing, economics, geography)
INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json)
VALUES (
  'us_census',
  'demographics',
  'https://api.census.gov/data',
  'active',
  '{"kafka_topic":"market.fundamentals","dataset":"2020/dec/pl","variables":["NAME","P1_001N"],"geo_for":"state:*","entity_field":"NAME"}'
);

-- UN Data (download/SDMX driven; configure downloads or sdmx_queries at runtime)
INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json)
VALUES (
  'un_data',
  'demographics',
  'https://data.un.org/',
  'active',
  '{"kafka_topic":"market.fundamentals","mode":"csv","downloads":[],"sdmx_queries":[]}'
);

-- Initialize MISO pilot entitlement (full access)
INSERT INTO pg.entitlement_product (tenant_id, market, product, channels, seats)
VALUES ('pilot_miso', 'power', 'lmp',
        '{"hub": true, "api": true, "downloads": true}'::jsonb, 5);

-- Initialize CAISO pilot entitlement (hub + downloads only)
INSERT INTO pg.entitlement_product (tenant_id, market, product, channels, seats)
VALUES ('pilot_caiso', 'power', 'lmp',
        '{"hub": true, "api": false, "downloads": true}'::jsonb, 3);
