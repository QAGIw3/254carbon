"""Utility helpers for ClickHouse interactions."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from clickhouse_driver import Client

logger = logging.getLogger(__name__)

_client: Optional[Client] = None


def get_clickhouse_client() -> Client:
    """Return a shared ClickHouse client instance."""
    global _client
    if _client is None:
        host = os.getenv("CLICKHOUSE_HOST", "clickhouse")
        port = int(os.getenv("CLICKHOUSE_PORT", "9000"))
        database = os.getenv("CLICKHOUSE_DB", "market_intelligence")
        _client = Client(
            host=host,
            port=port,
            database=database,
            send_receive_timeout=300,
            settings={
                "use_numpy": True,
                "strings_as_bytes": False,
                "async_insert": 1,
                "wait_for_async_insert": 0,
                "max_threads": 32,
                "max_execution_time": 120,
                "use_uncompressed_cache": 0,
            },
        )
        logger.info("Initialized ClickHouse client for gas-coal analytics")
    return _client


def _resolve_columns(column_types: Iterable[Tuple[str, str]]) -> List[str]:
    return [col for col, _ in column_types]


def query_dataframe(sql: str, parameters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Execute a ClickHouse query and return a DataFrame."""
    client = get_clickhouse_client()
    data, column_types = client.execute(sql, parameters or {}, with_column_types=True)
    columns = _resolve_columns(column_types)
    return pd.DataFrame(data, columns=columns)


def insert_rows(table: str, rows: List[Dict[str, Any]]) -> None:
    """Insert dictionaries into ClickHouse using column order from keys."""
    if not rows:
        return
    client = get_clickhouse_client()
    columns = list(rows[0].keys())
    payload = [[row.get(col) for col in columns] for row in rows]
    logger.info("Inserting %d rows into %s", len(rows), table)
    client.execute(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES",
        payload,
        types_check=True,
    )
