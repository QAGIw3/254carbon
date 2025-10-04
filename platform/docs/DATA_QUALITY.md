# Data Quality Framework

This service and schema provide:
- Cross-source validation
- Commodity-aware outlier detection (batch + stream)
- Missing data imputation (hourly/daily tables)
- Lineage emission to Marquez (OpenLineage) and OpenMetadata mapping

## Components
- ClickHouse schema: `platform/data/schemas/clickhouse/dq.sql`
- Service: `platform/apps/data-quality-service/` (FastAPI + batch + stream)
- Airflow DAGs: `platform/data/ingestion-orch/dags/dq_*.py`
- Kafka topics: `dq.flags.v1`, `dq.reports.v1`
- Dashboards: Grafana `data-quality.json`

## Run locally
1. Apply CH schema.
2. Start service: `uvicorn main:app --port 8010`.
3. Trigger batch jobs via POST endpoints.
4. Run stream worker within the service container.


