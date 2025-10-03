"""Seed PostgreSQL with connector checkpoint and network topology metadata."""

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import asyncpg


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seed_postgres")


@dataclass
class ConnectorSeed:
    connector_id: str
    description: str
    metadata: Dict[str, Any]


NETWORK_TOPOLOGIES: Dict[str, Dict[str, Any]] = {
    "PJM": {
        "nodes": [
            {"id": "PJM.BUS1", "type": "generator"},
            {"id": "PJM.BUS2", "type": "load"},
            {"id": "PJM.BUS3", "type": "load"},
            {"id": "PJM.HUB.WEST", "type": "hub"},
        ],
        "lines": [
            {"from": "PJM.BUS1", "to": "PJM.BUS2", "reactance": 0.083, "limit_mw": 600},
            {"from": "PJM.BUS1", "to": "PJM.BUS3", "reactance": 0.092, "limit_mw": 550},
            {"from": "PJM.BUS2", "to": "PJM.BUS3", "reactance": 0.110, "limit_mw": 500},
            {"from": "PJM.BUS2", "to": "PJM.HUB.WEST", "reactance": 0.070, "limit_mw": 800},
        ],
        "reference_bus": "PJM.HUB.WEST",
        "voltage_level": "500kV",
    },
    "MISO": {
        "nodes": [
            {"id": "MISO.BUS1", "type": "generator"},
            {"id": "MISO.BUS2", "type": "load"},
            {"id": "MISO.BUS3", "type": "load"},
            {"id": "MISO.HUB.INDIANA", "type": "hub"},
        ],
        "lines": [
            {"from": "MISO.BUS1", "to": "MISO.BUS2", "reactance": 0.105, "limit_mw": 500},
            {"from": "MISO.BUS2", "to": "MISO.BUS3", "reactance": 0.118, "limit_mw": 450},
            {"from": "MISO.BUS3", "to": "MISO.HUB.INDIANA", "reactance": 0.094, "limit_mw": 650},
        ],
        "reference_bus": "MISO.HUB.INDIANA",
        "voltage_level": "345kV",
    },
    "CAISO": {
        "nodes": [
            {"id": "CAISO.SP15", "type": "hub"},
            {"id": "CAISO.NP15", "type": "hub"},
            {"id": "CAISO.ZP26", "type": "hub"},
            {"id": "CAISO.BUS1", "type": "generator"},
            {"id": "CAISO.BUS2", "type": "load"},
        ],
        "lines": [
            {"from": "CAISO.BUS1", "to": "CAISO.SP15", "reactance": 0.076, "limit_mw": 700},
            {"from": "CAISO.SP15", "to": "CAISO.NP15", "reactance": 0.081, "limit_mw": 600},
            {"from": "CAISO.SP15", "to": "CAISO.ZP26", "reactance": 0.089, "limit_mw": 580},
            {"from": "CAISO.NP15", "to": "CAISO.BUS2", "reactance": 0.097, "limit_mw": 520},
        ],
        "reference_bus": "CAISO.SP15",
        "voltage_level": "500kV",
    },
}


CONNECTOR_SEEDS: List[ConnectorSeed] = [
    ConnectorSeed(
        connector_id="miso_rt_lmp",
        description="MISO real-time nodal LMP ingestion",
        metadata={"market": "power", "product": "lmp", "interval": "5min"},
    ),
    ConnectorSeed(
        connector_id="caiso_rtm_lmp",
        description="CAISO real-time market nodal LMP ingestion",
        metadata={"market": "power", "product": "lmp", "interval": "5min"},
    ),
    ConnectorSeed(
        connector_id="pjm_rt_lmp",
        description="PJM real-time nodal LMP ingestion",
        metadata={"market": "power", "product": "lmp", "interval": "5min"},
    ),
]


async def ensure_tables(conn: asyncpg.Connection) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS connector_checkpoints (
            connector_id VARCHAR(255) PRIMARY KEY,
            last_event_time TIMESTAMPTZ,
            last_successful_run TIMESTAMPTZ,
            state JSONB,
            error_count INTEGER DEFAULT 0,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS connector_checkpoint_history (
            history_id BIGSERIAL PRIMARY KEY,
            connector_id VARCHAR(255) NOT NULL,
            state JSONB,
            metadata JSONB,
            status VARCHAR(20),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_connector_checkpoint_history__connector_time
            ON connector_checkpoint_history (connector_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS network_topology (
            iso VARCHAR(32) NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            nodes JSONB NOT NULL,
            lines JSONB NOT NULL,
            reference_bus VARCHAR(255) NOT NULL,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (iso, version)
        );
        """
    )


async def seed_connectors(conn: asyncpg.Connection) -> None:
    for seed in CONNECTOR_SEEDS:
        await conn.execute(
            """
            INSERT INTO connector_checkpoints (
                connector_id,
                state,
                metadata,
                error_count
            ) VALUES ($1, $2, $3, 0)
            ON CONFLICT (connector_id) DO NOTHING
            """,
            seed.connector_id,
            json.dumps({"status": "seeded"}),
            json.dumps({"description": seed.description, **seed.metadata}),
        )


async def seed_network_topology(conn: asyncpg.Connection) -> None:
    for iso, topology in NETWORK_TOPOLOGIES.items():
        await conn.execute(
            """
            INSERT INTO network_topology (
                iso,
                version,
                nodes,
                lines,
                reference_bus,
                metadata
            ) VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (iso, version)
            DO UPDATE SET
                nodes = EXCLUDED.nodes,
                lines = EXCLUDED.lines,
                reference_bus = EXCLUDED.reference_bus,
                metadata = EXCLUDED.metadata
            """,
            iso,
            1,
            json.dumps(topology["nodes"]),
            json.dumps(topology["lines"]),
            topology["reference_bus"],
            json.dumps({"voltage_level": topology["voltage_level"]}),
        )


async def verify_asyncpg(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await ensure_tables(conn)
        await seed_connectors(conn)
        await seed_network_topology(conn)

        version = await conn.fetchval("SELECT version()")
        logger.info("Connected to PostgreSQL: %s", version)

        populations = await conn.fetch(
            """
            SELECT iso, version, jsonb_array_length(nodes) AS nodes, jsonb_array_length(lines) AS lines
            FROM network_topology
            ORDER BY iso
            """
        )

        for row in populations:
            logger.info(
                "ISO %s v%s: %s nodes, %s lines",
                row["iso"],
                row["version"],
                row["nodes"],
                row["lines"],
            )


async def main(args: argparse.Namespace) -> None:
    pool = await asyncpg.create_pool(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
        min_size=1,
        max_size=5,
        command_timeout=60,
        server_settings={"application_name": "seed_postgres"},
    )

    try:
        await verify_asyncpg(pool)
    finally:
        await pool.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed PostgreSQL metadata for connectors and PTDF topology")
    parser.add_argument("--host", default="postgresql", help="PostgreSQL host")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port")
    parser.add_argument("--database", default="market_intelligence", help="Database name")
    parser.add_argument("--user", default="postgres", help="Database user")
    parser.add_argument("--password", default="postgres", help="Database password")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))

