"""Persistence utilities for research workflows."""

from __future__ import annotations

from typing import Any, Dict

from clickhouse_driver import Client


class ResearchPersistence:
    def __init__(self, *, ch_client: Client) -> None:
        self._client = ch_client

    def persist_experiment(self, payload: Dict[str, Any]) -> None:
        self._client.execute(
            """
            INSERT INTO market_intelligence.research_experiments
            (experiment_id, name, model_type, dataset, parameters, status, mlflow_run_id,
             started_at, completed_at, results)
            VALUES
            (%(experiment_id)s, %(name)s, %(model_type)s, %(dataset)s, %(parameters)s,
             %(status)s, %(mlflow_run_id)s, %(started_at)s, %(completed_at)s, %(results)s)
            """,
            payload,
        )
