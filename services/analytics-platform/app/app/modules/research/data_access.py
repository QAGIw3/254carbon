"""Data access layer for research analytics."""

from __future__ import annotations

from typing import Any, Dict

from clickhouse_driver import Client


class DataAccessLayer:
    def __init__(self, *, host: str = "clickhouse", port: int = 9000, database: str = "market_intelligence") -> None:
        self._client = Client(host=host, port=port, database=database)

    @property
    def client(self) -> Client:
        return self._client

    def fetch_dataframe(self, query: str, params: Dict[str, Any]) -> Any:
        return self._client.query_dataframe(query, params)
