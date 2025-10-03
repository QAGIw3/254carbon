-- PostgreSQL Schema Initialization
-- Market Intelligence Platform Metadata

CREATE SCHEMA IF NOT EXISTS pg;

-- Source registry for data connectors
CREATE TABLE IF NOT EXISTS pg.source_registry (
    source_id   TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,      -- iso, pra, broker, vendor, calc
    endpoint    TEXT NOT NULL,
    status      TEXT NOT NULL,      -- active, paused
    cfg_json    JSONB NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Instrument master data
CREATE TABLE IF NOT EXISTS pg.instrument (
    instrument_id  TEXT PRIMARY KEY,
    market         TEXT NOT NULL,
    product        TEXT NOT NULL,
    location_code  TEXT NOT NULL,
    timezone       TEXT NOT NULL DEFAULT 'UTC',
    unit           TEXT NOT NULL,
    currency       TEXT NOT NULL DEFAULT 'USD',
    attrs          JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ DEFAULT now()
);

-- Instrument aliases for vendor code mapping
CREATE TABLE IF NOT EXISTS pg.instrument_alias (
    source_id     TEXT NOT NULL REFERENCES pg.source_registry(source_id),
    vendor_code   TEXT NOT NULL,
    instrument_id TEXT NOT NULL REFERENCES pg.instrument(instrument_id),
    valid_from    DATE NOT NULL DEFAULT CURRENT_DATE,
    valid_to      DATE,
    PRIMARY KEY (source_id, vendor_code, valid_from)
);

-- Scenario definitions
CREATE TABLE IF NOT EXISTS pg.scenario (
    scenario_id   TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    description   TEXT NOT NULL,
    visibility    TEXT NOT NULL DEFAULT 'org',
    created_by    TEXT NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Scenario assumption sets
CREATE TABLE IF NOT EXISTS pg.assumption_set (
    assumption_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id   TEXT NOT NULL REFERENCES pg.scenario(scenario_id),
    as_of_date    DATE NOT NULL,
    payload       JSONB NOT NULL,
    version       INT  NOT NULL DEFAULT 1,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Scenario execution runs
CREATE TABLE IF NOT EXISTS pg.scenario_run (
    run_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id   TEXT NOT NULL REFERENCES pg.scenario(scenario_id),
    assumption_id UUID NOT NULL REFERENCES pg.assumption_set(assumption_id),
    status        TEXT NOT NULL,      -- queued, running, success, failed
    started_at    TIMESTAMPTZ,
    finished_at   TIMESTAMPTZ,
    notes         TEXT,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- Unit reference data
CREATE TABLE IF NOT EXISTS pg.unit_ref (
    unit          TEXT PRIMARY KEY,
    description   TEXT NOT NULL
);

-- Unit conversions
CREATE TABLE IF NOT EXISTS pg.unit_conv (
    from_unit TEXT,
    to_unit   TEXT,
    factor    NUMERIC,
    PRIMARY KEY(from_unit, to_unit)
);

-- Tenant/organization management
CREATE TABLE IF NOT EXISTS pg.tenant (
    tenant_id TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    status    TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Product entitlements by tenant
CREATE TABLE IF NOT EXISTS pg.entitlement_product (
    tenant_id   TEXT REFERENCES pg.tenant,
    market      TEXT,
    product     TEXT,
    instruments JSONB,
    from_date   DATE,
    to_date     DATE,
    seats       INT,
    channels    JSONB NOT NULL DEFAULT '{"hub":true,"api":true,"downloads":true}'::jsonb,
    PRIMARY KEY(tenant_id, product, market)
);

-- Audit log for data access
CREATE TABLE IF NOT EXISTS pg.audit_log (
    audit_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp     TIMESTAMPTZ DEFAULT now(),
    user_id       TEXT NOT NULL,
    tenant_id     TEXT,
    action        TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id   TEXT NOT NULL,
    ip_address    INET,
    user_agent    TEXT,
    success       BOOLEAN NOT NULL,
    details       JSONB
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_instrument_market ON pg.instrument(market, product);
CREATE INDEX IF NOT EXISTS idx_scenario_created ON pg.scenario(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_run_status ON pg.scenario_run(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON pg.audit_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_user ON pg.audit_log(user_id, timestamp DESC);

-- Insert reference data
INSERT INTO pg.unit_ref (unit, description) VALUES
    ('MWh', 'Megawatt-hour'),
    ('MMBtu', 'Million British Thermal Units'),
    ('short_ton', 'US short ton (2000 lbs)'),
    ('t', 'Metric tonne (1000 kg)'),
    ('MW', 'Megawatt')
ON CONFLICT (unit) DO NOTHING;

INSERT INTO pg.unit_conv (from_unit, to_unit, factor) VALUES
    ('MWh', 'MWh', 1.0),
    ('MMBtu', 'MMBtu', 1.0),
    ('short_ton', 't', 0.90718),
    ('t', 'short_ton', 1.10231)
ON CONFLICT (from_unit, to_unit) DO NOTHING;

-- Insert sample scenario
INSERT INTO pg.scenario (scenario_id, title, description, created_by) VALUES
    ('BASE', 'Base Case', 'Default baseline forecast with current policy and market assumptions', 'system')
ON CONFLICT (scenario_id) DO NOTHING;

-- Insert sample tenant for testing
INSERT INTO pg.tenant (tenant_id, name, status) VALUES
    ('pilot_miso', 'MISO Pilot Customer', 'active'),
    ('pilot_caiso', 'CAISO Pilot Customer', 'active')
ON CONFLICT (tenant_id) DO NOTHING;

-- MISO pilot: full access (hub + api + downloads)
INSERT INTO pg.entitlement_product (tenant_id, market, product, channels, seats) VALUES
    ('pilot_miso', 'power', 'lmp', '{"hub": true, "api": true, "downloads": true}'::jsonb, 5)
ON CONFLICT (tenant_id, product, market) DO NOTHING;

-- CAISO pilot: hub + downloads only (no API)
INSERT INTO pg.entitlement_product (tenant_id, market, product, channels, seats) VALUES
    ('pilot_caiso', 'power', 'lmp', '{"hub": true, "api": false, "downloads": true}'::jsonb, 3)
ON CONFLICT (tenant_id, product, market) DO NOTHING;

