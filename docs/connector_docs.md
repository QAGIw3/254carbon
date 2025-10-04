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
