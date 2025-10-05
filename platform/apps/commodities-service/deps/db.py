"""
Database helpers for ClickHouse and Postgres.

This module centralizes connection setup and health checks for the service.
We keep the ClickHouse client as a process-wide singleton (threadsafe in our
usage) and expose a tiny async wrapper to execute queries off the event loop
using `run_in_executor` to avoid blocking.
"""
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import asyncpg
from clickhouse_driver import Client

logger = logging.getLogger(__name__)

_postgres_pool: Optional[asyncpg.Pool] = None
_clickhouse_client: Optional[Client] = None


def get_clickhouse_client() -> Client:
    """
    Return a singleton ClickHouse client.

    Notes
    - Client is synchronous; queries should be executed outside the event loop
      using `run_in_executor` (see `fetch_clickhouse`).
    - Settings tune performance and guard execution time to keep endpoints
      responsive.
    """

    global _clickhouse_client
    if _clickhouse_client is None:
        host = os.getenv("CLICKHOUSE_HOST", "clickhouse")
        port = int(os.getenv("CLICKHOUSE_PORT", "9000"))
        database = os.getenv("CLICKHOUSE_DATABASE", "market_intelligence")
        user = os.getenv("CLICKHOUSE_USER", "default")
        password = os.getenv("CLICKHOUSE_PASSWORD", "")

        logger.info(
            "Initializing ClickHouse client (host=%s port=%s database=%s)",
            host,
            port,
            database,
        )

        _clickhouse_client = Client(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password or None,
            send_receive_timeout=10,
            connect_timeout=5,
            settings={
                "use_numpy": 0,
                "strings_as_bytes": 0,
                "max_execution_time": 30,
                "max_threads": int(os.getenv("CLICKHOUSE_MAX_THREADS", "4")),
            },
        )
    return _clickhouse_client


async def fetch_clickhouse(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Execute a ClickHouse query asynchronously and return list of dicts.

    We capture column names from `with_column_types` so each row is a mapping
    suitable for Pydantic model construction without relying on tuple order.
    """

    client = get_clickhouse_client()
    loop = asyncio.get_running_loop()

    def _execute() -> List[Dict[str, Any]]:
        data, column_info = client.execute(
            query,
            parameters or {},
            with_column_types=True,
        )
        columns = [col[0] for col in column_info]
        return [dict(zip(columns, row)) for row in data]

    try:
        return await loop.run_in_executor(None, _execute)
    except Exception as exc:
        logger.error("ClickHouse query failed: %s", exc)
        raise


async def ping_clickhouse() -> bool:
    """Ping ClickHouse to verify readiness (best-effort)."""

    client = get_clickhouse_client()
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, client.ping)
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("ClickHouse ping failed: %s", exc)
        return False


async def get_postgres_pool() -> asyncpg.Pool:
    """
    Return a shared asyncpg pool.

    The service currently reads market data from ClickHouse; Postgres is kept
    to support entitlement checks or metadata in future expansions.
    """

    global _postgres_pool
    if _postgres_pool is None:
        dsn = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@postgres:5432/market_intelligence",
        )
        min_size = int(os.getenv("POSTGRES_MIN_POOL", "1"))
        max_size = int(os.getenv("POSTGRES_MAX_POOL", "5"))

        logger.info("Creating Postgres pool (min=%s max=%s)", min_size, max_size)
        _postgres_pool = await asyncpg.create_pool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            timeout=10,
        )
    return _postgres_pool


async def ping_postgres() -> bool:
    """Ping Postgres to verify readiness (best-effort)."""

    try:
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Postgres ping failed: %s", exc)
        return False


__all__ = [
    "fetch_clickhouse",
    "get_clickhouse_client",
    "ping_clickhouse",
    "get_postgres_pool",
    "ping_postgres",
]
