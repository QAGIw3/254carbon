#!/bin/bash
# Seed local databases with sample data for development

# 254Carbon Platform - Sample Data Seeding Script
# Seeds sample market data for development and testing

set -e

echo "🌱 Seeding sample market data..."

# Function to execute SQL in PostgreSQL
execute_postgres() {
    docker-compose exec -T postgres psql -U postgres -d market_intelligence -c "$1"
}

# Function to execute SQL in ClickHouse
execute_clickhouse() {
    curl -s -X POST 'http://localhost:8123/' \
         --header 'Content-Type: application/octet-stream' \
         --data-binary "$1"
}

echo "📊 Seeding PostgreSQL reference data..."

# Insert sample instruments
execute_postgres "
INSERT INTO pg.instrument (instrument_id, market, product, location_code, timezone, unit, currency) VALUES
    ('MISO.HUB.INDIANA', 'power', 'lmp', 'INDIANA', 'America/Indiana/Indianapolis', 'MWh', 'USD'),
    ('PJM.HUB.WEST', 'power', 'lmp', 'WEST', 'America/New_York', 'MWh', 'USD'),
    ('CAISO.SP15', 'power', 'lmp', 'SP15', 'America/Los_Angeles', 'MWh', 'USD'),
    ('HENRY.HUB', 'gas', 'spot', 'HENRY', 'America/Chicago', 'MMBtu', 'USD')
ON CONFLICT (instrument_id) DO NOTHING;
"

# Insert sample scenarios
execute_postgres "
INSERT INTO pg.scenario (scenario_id, title, description, created_by) VALUES
    ('BASE', 'Base Case', 'Default baseline forecast with current policy and market assumptions', 'system'),
    ('HIGH_GROWTH', 'High Growth', 'Accelerated renewable deployment scenario', 'system'),
    ('CARBON_TAX', 'Carbon Tax', 'Carbon pricing scenario with $50/ton tax', 'system')
ON CONFLICT (scenario_id) DO NOTHING;
"

echo "📈 Seeding ClickHouse market data..."

# Create sample price data for the last 30 days
execute_clickhouse "
INSERT INTO market_price_ticks
SELECT
    toDateTime(now() - number * 3600) as timestamp,
    'power' as market,
    'lmp' as product,
    instrument_id,
    -- Generate realistic price patterns
    CASE
        WHEN instrument_id = 'MISO.HUB.INDIANA' THEN 35 + 15 * sin(number / 24) + randNormal(0, 3)
        WHEN instrument_id = 'PJM.HUB.WEST' THEN 40 + 12 * sin(number / 24) + randNormal(0, 2.5)
        WHEN instrument_id = 'CAISO.SP15' THEN 45 + 20 * sin(number / 24) + randNormal(0, 4)
        ELSE 3.5 + 0.8 * sin(number / 24) + randNormal(0, 0.2)
    END as price,
    CASE
        WHEN instrument_id LIKE '%HUB%' THEN 'hub'
        WHEN instrument_id LIKE '%SP15%' THEN 'hub'
        ELSE 'hub'
    END as channel,
    'system' as source_id,
    now() as created_at
FROM (
    SELECT arrayJoin(['MISO.HUB.INDIANA', 'PJM.HUB.WEST', 'CAISO.SP15', 'HENRY.HUB']) as instrument_id
) instruments
CROSS JOIN numbers(720)  -- 30 days * 24 hours
WHERE toDateTime(now() - number * 3600) >= now() - INTERVAL 30 DAY
"

echo "📊 Seeding forward curve data..."

# Create sample forward curve data
execute_clickhouse "
INSERT INTO forward_curve_points
SELECT
    instrument_id,
    delivery_period,
    -- Generate realistic forward curve shapes
    CASE
        WHEN instrument_id = 'MISO.HUB.INDIANA' THEN 38 + (month_num - 1) * 0.5 + randNormal(0, 1)
        WHEN instrument_id = 'PJM.HUB.WEST' THEN 42 + (month_num - 1) * 0.3 + randNormal(0, 0.8)
        WHEN instrument_id = 'CAISO.SP15' THEN 48 + (month_num - 1) * 0.7 + randNormal(0, 1.2)
        ELSE 3.8 + (month_num - 1) * 0.05 + randNormal(0, 0.1)
    END as price,
    'BASE' as scenario_id,
    today() as as_of_date,
    'system' as source_id,
    now() as created_at
FROM (
    SELECT arrayJoin(['MISO.HUB.INDIANA', 'PJM.HUB.WEST', 'CAISO.SP15', 'HENRY.HUB']) as instrument_id
) instruments
CROSS JOIN (
    SELECT
        toString(toYear(today()) + floor((number + 1) / 12)) || '-' ||
        toString((number % 12) + 1) as delivery_period,
        number + 1 as month_num
    FROM numbers(24)  -- 2 years of monthly data
) periods
"

echo "✅ Sample data seeding complete!"

echo ""
echo "📊 Sample Data Summary:"
echo "   • 4 instruments seeded"
echo "   • 30 days of hourly price data (~28,800 records)"
echo "   • 2 years of monthly forward curves (~96 records)"
echo "   • 3 scenarios available"
echo ""

echo "💡 You can now:"
echo "   • View data at http://localhost:8000/docs"
echo "   • Query ClickHouse at http://localhost:8123"
echo "   • Generate reports via API"
echo "   • Test scenarios and backtesting"
echo ""
