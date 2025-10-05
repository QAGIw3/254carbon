"""
Database connection management for PostgreSQL.
"""
import logging
import os
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)

# Global connection pool
_pool: Optional[asyncpg.Pool] = None

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@postgresql:5432/market_intelligence"
)


async def init_db_pool():
    """Initialize PostgreSQL connection pool."""
    global _pool
    
    if _pool is not None:
        return
    
    try:
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=60,
        )
        logger.info("Database connection pool created")
    except Exception as e:
        logger.error(f"Failed to create database pool: {e}")
        raise


async def get_pool() -> asyncpg.Pool:
    """
    Get the database connection pool.
    
    Returns:
        asyncpg.Pool: The connection pool.
        
    Raises:
        RuntimeError: If pool is not initialized.
    """
    if _pool is None:
        await init_db_pool()
    
    return _pool


async def close_db_pool():
    """Close the database connection pool."""
    global _pool
    
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Database connection pool closed")

