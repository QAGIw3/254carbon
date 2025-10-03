-- Configure CAISO Pilot Entitlement Restrictions
-- Hub + Downloads only (NO API access)

-- Ensure CAISO pilot tenant exists
INSERT INTO pg.tenant (tenant_id, name, status)
VALUES ('pilot_caiso', 'CAISO Pilot Customer', 'active')
ON CONFLICT (tenant_id) DO UPDATE SET status = 'active';

-- Set entitlements: Hub + Downloads, API = false
INSERT INTO pg.entitlement_product 
(tenant_id, market, product, channels, seats, from_date)
VALUES 
('pilot_caiso', 'power', 'lmp', 
 '{"hub": true, "api": false, "downloads": true}'::jsonb, 
 3, 
 CURRENT_DATE)
ON CONFLICT (tenant_id, product, market) 
DO UPDATE SET 
    channels = '{"hub": true, "api": false, "downloads": true}'::jsonb,
    seats = 3,
    from_date = CURRENT_DATE;

-- Verify configuration
SELECT 
    tenant_id,
    market,
    product,
    channels,
    seats,
    from_date,
    to_date
FROM pg.entitlement_product
WHERE tenant_id = 'pilot_caiso';

-- Expected output:
-- tenant_id    | market | product | channels                                       | seats | from_date  | to_date
-- pilot_caiso  | power  | lmp     | {"hub": true, "api": false, "downloads": true} | 3     | 2025-10-03 | null

-- Create sample CAISO users for UAT
-- NOTE: Actual users will be created in Keycloak and mapped to tenant_id

