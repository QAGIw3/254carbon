-- Infrastructure Data Schema Extensions
-- Adds tables and relationships for infrastructure assets

-- Infrastructure asset registry
CREATE TABLE IF NOT EXISTS pg.infrastructure_assets (
    asset_id            TEXT PRIMARY KEY,
    asset_name          TEXT NOT NULL,
    asset_type          TEXT NOT NULL CHECK (asset_type IN (
        'lng_terminal', 'gas_storage', 'power_plant', 
        'transmission_line', 'substation', 'renewable_resource'
    )),
    country             TEXT NOT NULL,
    region              TEXT,
    latitude            NUMERIC NOT NULL CHECK (latitude BETWEEN -90 AND 90),
    longitude           NUMERIC NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    status              TEXT NOT NULL DEFAULT 'unknown' CHECK (status IN (
        'operational', 'construction', 'planned', 
        'decommissioned', 'mothballed', 'unknown'
    )),
    operator            TEXT,
    owner               TEXT,
    commissioned_date   DATE,
    decommissioned_date DATE,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Power plant specific attributes
CREATE TABLE IF NOT EXISTS pg.power_plants (
    asset_id            TEXT PRIMARY KEY REFERENCES pg.infrastructure_assets(asset_id),
    capacity_mw         NUMERIC NOT NULL CHECK (capacity_mw > 0),
    primary_fuel        TEXT NOT NULL,
    secondary_fuel      TEXT,
    other_fuels         TEXT[],
    efficiency_pct      NUMERIC CHECK (efficiency_pct BETWEEN 0 AND 100),
    capacity_factor     NUMERIC CHECK (capacity_factor BETWEEN 0 AND 1),
    annual_generation_gwh NUMERIC,
    emissions_rate_tco2_mwh NUMERIC
);

-- LNG terminal specific attributes
CREATE TABLE IF NOT EXISTS pg.lng_terminals (
    asset_id                    TEXT PRIMARY KEY REFERENCES pg.infrastructure_assets(asset_id),
    storage_capacity_gwh        NUMERIC NOT NULL CHECK (storage_capacity_gwh > 0),
    storage_capacity_mcm        NUMERIC,
    regasification_capacity_gwh_d NUMERIC,
    send_out_capacity_gwh_d     NUMERIC,
    num_tanks                   INTEGER,
    berth_capacity              INTEGER,
    terminal_type               TEXT CHECK (terminal_type IN ('import', 'export', 'bidirectional'))
);

-- Transmission line specific attributes
CREATE TABLE IF NOT EXISTS pg.transmission_lines (
    asset_id        TEXT PRIMARY KEY REFERENCES pg.infrastructure_assets(asset_id),
    from_latitude   NUMERIC NOT NULL,
    from_longitude  NUMERIC NOT NULL,
    to_latitude     NUMERIC NOT NULL,
    to_longitude    NUMERIC NOT NULL,
    voltage_kv      NUMERIC NOT NULL CHECK (voltage_kv > 0),
    capacity_mw     NUMERIC NOT NULL CHECK (capacity_mw > 0),
    length_km       NUMERIC CHECK (length_km > 0),
    line_type       TEXT NOT NULL DEFAULT 'AC' CHECK (line_type IN ('AC', 'DC')),
    voltage_class   TEXT,
    circuits        INTEGER DEFAULT 1
);

-- Infrastructure projects tracking
CREATE TABLE IF NOT EXISTS pg.infrastructure_projects (
    project_id              TEXT PRIMARY KEY,
    project_name            TEXT NOT NULL,
    project_type            TEXT NOT NULL,
    countries               TEXT[] NOT NULL,
    status                  TEXT NOT NULL CHECK (status IN (
        'announced', 'pre_construction', 'construction', 
        'commissioned', 'suspended', 'cancelled'
    )),
    developer               TEXT,
    estimated_cost_million_usd NUMERIC,
    start_year              INTEGER,
    completion_year         INTEGER,
    capacity_mw             NUMERIC,
    voltage_kv              NUMERIC,
    length_km               NUMERIC,
    project_data            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

-- Renewable resource assessments
CREATE TABLE IF NOT EXISTS pg.renewable_resources (
    resource_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    latitude            NUMERIC NOT NULL,
    longitude           NUMERIC NOT NULL,
    resource_type       TEXT NOT NULL CHECK (resource_type IN (
        'solar_ghi', 'solar_dni', 'wind_speed_100m', 
        'wind_speed_50m', 'hydro_potential'
    )),
    annual_average      NUMERIC NOT NULL,
    monthly_averages    NUMERIC[12],
    unit                TEXT NOT NULL,
    data_year           INTEGER NOT NULL,
    resolution_km       NUMERIC NOT NULL,
    data_source         TEXT NOT NULL,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Network topology
CREATE TABLE IF NOT EXISTS pg.network_nodes (
    node_id         TEXT PRIMARY KEY,
    node_name       TEXT NOT NULL,
    node_type       TEXT NOT NULL CHECK (node_type IN (
        'substation', 'border_point', 'generator', 'load'
    )),
    country         TEXT NOT NULL,
    latitude        NUMERIC NOT NULL,
    longitude       NUMERIC NOT NULL,
    voltage_levels  NUMERIC[] NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Network connections
CREATE TABLE IF NOT EXISTS pg.network_connections (
    connection_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_node_id    TEXT REFERENCES pg.network_nodes(node_id),
    to_node_id      TEXT REFERENCES pg.network_nodes(node_id),
    line_id         TEXT REFERENCES pg.infrastructure_assets(asset_id),
    connection_type TEXT NOT NULL DEFAULT 'transmission',
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(from_node_id, to_node_id, line_id)
);

-- Infrastructure time series reference
CREATE TABLE IF NOT EXISTS pg.infrastructure_series_ref (
    series_id       TEXT PRIMARY KEY,
    asset_id        TEXT REFERENCES pg.infrastructure_assets(asset_id),
    metric          TEXT NOT NULL,
    unit            TEXT NOT NULL,
    frequency       TEXT NOT NULL,
    source          TEXT NOT NULL,
    description     TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_infra_assets_type ON pg.infrastructure_assets(asset_type);
CREATE INDEX IF NOT EXISTS idx_infra_assets_country ON pg.infrastructure_assets(country);
CREATE INDEX IF NOT EXISTS idx_infra_assets_location ON pg.infrastructure_assets USING GIST(point(longitude, latitude));
CREATE INDEX IF NOT EXISTS idx_infra_assets_status ON pg.infrastructure_assets(status);
CREATE INDEX IF NOT EXISTS idx_power_plants_fuel ON pg.power_plants(primary_fuel);
CREATE INDEX IF NOT EXISTS idx_transmission_voltage ON pg.transmission_lines(voltage_kv);
CREATE INDEX IF NOT EXISTS idx_projects_status ON pg.infrastructure_projects(status);
CREATE INDEX IF NOT EXISTS idx_renewable_type ON pg.renewable_resources(resource_type);
CREATE INDEX IF NOT EXISTS idx_renewable_location ON pg.renewable_resources USING GIST(point(longitude, latitude));
CREATE INDEX IF NOT EXISTS idx_network_nodes_country ON pg.network_nodes(country);

-- Create update trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_infrastructure_assets_updated_at 
    BEFORE UPDATE ON pg.infrastructure_assets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_infrastructure_projects_updated_at
    BEFORE UPDATE ON pg.infrastructure_projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
