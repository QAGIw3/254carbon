# Infra Ops

- Deploy locally:
  - `cd platform/infra/helmfile && helmfile -e dev sync`
- GitLab CI:
  - Push to `main` to trigger `infra:deploy:dev`

## Vault init/unseal (once)
1. `kubectl -n platform-security exec -it svc/vault -- vault operator init`
2. Save unseal keys and root token securely
3. Unseal 3 times: `vault operator unseal`
4. Enable Kubernetes auth and KV v2

## Access URLs
- Grafana: grafana.${BASE_DOMAIN}
- Prometheus: prometheus.${BASE_DOMAIN}
- Vault: vault.${BASE_DOMAIN}
- NiFi: nifi.${BASE_DOMAIN}
- Airflow: airflow.${BASE_DOMAIN}
- OpenMetadata: openmetadata.${BASE_DOMAIN}
- MLflow: mlflow.${BASE_DOMAIN}

## Shared Postgres

Use the existing cluster Postgres at `postgresql.market-intelligence.svc.cluster.local:5432`.

Create connection secrets:

- Airflow (full SQLAlchemy URI):
  - Secret: `airflow-postgres`, key: `connection`
  - Example: `postgresql+psycopg2://airflow:****@postgresql.market-intelligence.svc.cluster.local:5432/airflow`

- OpenMetadata (password only):
  - Secret: `openmetadata-postgres`, key: `password`

- MLflow (full DB URI or chart-specific connection):
  - Secret: `mlflow-postgres`, key: `connection`
  - Example: `postgresql://mlflow:****@postgresql.market-intelligence.svc.cluster.local:5432/mlflow`

Update `environments/dev/values-postgres.yaml` to adjust DB names/users as needed.
