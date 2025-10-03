"""
Database connections for Curve Service.
"""
import os
import logging
from typing import Optional

import asyncpg
from clickhouse_driver import Client

logger = logging.getLogger(__name__)

_postgres_pool: Optional[asyncpg.Pool] = None
_clickhouse_client: Optional[Client] = None


async def get_postgres_pool() -> asyncpg.Pool:
    """Get or create PostgreSQL connection pool."""
    global _postgres_pool
    
    if _postgres_pool is None:
        DATABASE_URL = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@postgres:5432/market_intelligence",
        )
        
        _postgres_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
        logger.info("PostgreSQL pool created")
    
    return _postgres_pool


def get_clickhouse_client() -> Client:
    """Get or create ClickHouse client."""
    global _clickhouse_client
    
    if _clickhouse_client is None:
        CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
        CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "9000"))
        
        _clickhouse_client = Client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            database="default",
        )
        logger.info("ClickHouse client created")
    
    return _clickhouse_client

