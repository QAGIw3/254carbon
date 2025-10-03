-- Initialize MISO connector
INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json)
VALUES ('miso_rt_lmp', 'iso', 'https://api.misoenergy.org', 'active',
        '{"market_type": "RT", "kafka_topic": "power.ticks.v1"}');

-- Initialize CAISO connector with restrictions
INSERT INTO pg.source_registry (source_id, kind, endpoint, status, cfg_json)
VALUES ('caiso_rt_lmp', 'iso', 'https://oasis.caiso.com', 'active',
        '{"market_type": "RT", "kafka_topic": "power.ticks.v1", "restricted_nodes": ["SP15", "NP15", "ZP26"]}');

-- Initialize MISO pilot entitlement (full access)
INSERT INTO pg.entitlement_product (tenant_id, market, product, channels, seats)
VALUES ('pilot_miso', 'power', 'lmp',
        '{"hub": true, "api": true, "downloads": true}'::jsonb, 5);

-- Initialize CAISO pilot entitlement (hub + downloads only)
INSERT INTO pg.entitlement_product (tenant_id, market, product, channels, seats)
VALUES ('pilot_caiso', 'power', 'lmp',
        '{"hub": true, "api": false, "downloads": true}'::jsonb, 3);
