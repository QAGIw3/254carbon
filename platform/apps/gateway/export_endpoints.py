"""
Research Bulk Export Endpoints

Implements dataset export job lifecycle backed by Postgres (job metadata),
ClickHouse (data source), and MinIO (object storage for large exports).

Endpoints (prefix: /api/v1/research/export):
- POST /jobs: Create an export job (dataset_id or SQL + params)
- GET  /jobs/{job_id}: Get job status and metadata
- GET  /jobs/{job_id}/download: Presigned URL for download
- GET  /preview: Sample first N rows to validate a query/config
"""
import asyncio
import csv
import gzip
import hashlib
import io
import json
import os
import tempfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional

import asyncpg
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from auth import verify_token
from entitlements import check_entitlement
from db import get_postgres_pool, get_clickhouse_client

try:
    from minio import Minio
    from minio.error import S3Error
except Exception:  # pragma: no cover - optional import during local dev
    Minio = None  # type: ignore
    S3Error = Exception  # type: ignore


router = APIRouter(
    prefix="/api/v1/research/export",
    tags=["research", "export"],
    dependencies=[Depends(verify_token)],
)


class ExportJobCreate(BaseModel):
    dataset_id: Optional[str] = Field(
        default=None, description="Predefined dataset id (maps to SQL template)"
    )
    sql: Optional[str] = Field(default=None, description="Custom ClickHouse SQL")
    params: Optional[Dict[str, Any]] = Field(default=None, description="SQL params")
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    fmt: str = Field(default="csv", description="csv|parquet|jsonl")
    compression: Optional[str] = Field(default=None, description="gzip|none")
    idempotency_key: Optional[str] = Field(default=None, description="Client-provided idempotency key")


class ExportJobStatus(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    fmt: str
    compression: Optional[str]
    object_key: Optional[str] = None
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    error: Optional[str] = None
    expires_at: Optional[datetime] = None


_DATASET_SQL_MAP: Dict[str, str] = {
    # Simple dataset templates; extend as needed
    "energy_prices": (
        """
        SELECT event_time, instrument_id, location_code, price_type, value, volume, currency, unit, source
        FROM market_intelligence.market_price_ticks
        WHERE event_time >= %(start)s AND event_time <= %(end)s
        ORDER BY event_time
        """
    ),
    "carbon_markets": (
        """
        SELECT event_time, instrument_id, value, source
        FROM market_intelligence.market_price_ticks
        WHERE commodity_type = 'emissions'
          AND event_time >= %(start)s AND event_time <= %(end)s
        ORDER BY event_time
        """
    ),
}


# Concurrency limits and quotas
_EXPORT_MAX_CONCURRENCY = int(os.getenv("EXPORT_MAX_CONCURRENCY", "4"))
_EXPORT_SEMAPHORE = asyncio.Semaphore(_EXPORT_MAX_CONCURRENCY)
_EXPORT_MAX_SIZE_BYTES = int(os.getenv("EXPORT_MAX_SIZE_BYTES", str(5 * 1024 * 1024 * 1024)))  # 5 GiB


async def _ensure_job_table(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS export_job (
                job_id          UUID PRIMARY KEY,
                idempotency_key TEXT,
                requester       TEXT,
                status          TEXT NOT NULL,
                fmt             TEXT NOT NULL,
                compression     TEXT,
                params          JSONB,
                object_key      TEXT,
                size_bytes      BIGINT,
                checksum        TEXT,
                error           TEXT,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                expires_at      TIMESTAMPTZ
            );
            CREATE INDEX IF NOT EXISTS export_job_idem_idx ON export_job (idempotency_key);
            """
        )


def _get_minio_client() -> Optional[Minio]:
    if Minio is None:
        return None
    endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


async def _update_job_status(
    pool: asyncpg.Pool, job_id: str, **updates: Any
) -> None:
    if not updates:
        return
    columns = ", ".join([f"{k} = ${i+2}" for i, k in enumerate(updates.keys())])
    values = list(updates.values())
    async with pool.acquire() as conn:
        await conn.execute(
            f"UPDATE export_job SET {columns}, updated_at = now() WHERE job_id = $1",
            job_id,
            *values,
        )


def _build_query(payload: ExportJobCreate) -> Dict[str, Any]:
    if payload.dataset_id:
        if payload.dataset_id not in _DATASET_SQL_MAP:
            raise HTTPException(status_code=400, detail="Unknown dataset_id")
        if not payload.start_date or not payload.end_date:
            raise HTTPException(status_code=400, detail="start_date and end_date required for dataset exports")
        sql = _DATASET_SQL_MAP[payload.dataset_id]
        params = {
            "start": datetime.combine(payload.start_date, datetime.min.time()),
            "end": datetime.combine(payload.end_date, datetime.max.time()),
        }
        return {"sql": sql, "params": params}
    if payload.sql:
        params = payload.params or {}
        if payload.start_date:
            params.setdefault("start", datetime.combine(payload.start_date, datetime.min.time()))
        if payload.end_date:
            params.setdefault("end", datetime.combine(payload.end_date, datetime.max.time()))
        return {"sql": payload.sql, "params": params}
    raise HTTPException(status_code=400, detail="Provide dataset_id or sql")


async def _run_export_job(job_id: str) -> None:
    async with _EXPORT_SEMAPHORE:
        pool = await get_postgres_pool()
        await _ensure_job_table(pool)
        ch = get_clickhouse_client()

    # Load job record
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM export_job WHERE job_id = $1", job_id)
        if row is None:
            return
        params = dict(row["params"] or {})
        fmt = row["fmt"]
        compression = row["compression"]

    # Reconstruct query and params from saved job params
    sql = params.get("__sql__")
    query_params = params.get("__params__", {})
    if not sql:
        await _update_job_status(pool, job_id, status="failed", error="Missing SQL in job params")
        return

    # Build local temp file
    suffix = ".csv"
    if fmt == "jsonl":
        suffix = ".jsonl"
    elif fmt == "parquet":  # placeholder, we export CSV for now, rename accordingly
        suffix = ".parquet.csv"

    use_gzip = (compression or "").lower() == "gzip"
    if use_gzip:
        suffix += ".gz"

    tmp_dir = os.getenv("EXPORT_TMP_DIR", "/tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = tempfile.mktemp(prefix="export_", suffix=suffix, dir=tmp_dir)

    try:
        await _update_job_status(pool, job_id, status="running")

        # Get column names via LIMIT 0
        metadata_query = f"SELECT * FROM ({sql}) LIMIT 0"
        _, colinfo = ch.execute(metadata_query, params=query_params, with_column_types=True)
        column_names = [c[0] for c in colinfo]

        # Prepare writers
        hasher = hashlib.sha256()
        total_bytes = 0
        raw_fh = open(tmp_path, "wb")
        try:
            fh: io.BufferedWriter
            if use_gzip:
                fh = gzip.GzipFile(fileobj=raw_fh, mode="wb")  # type: ignore
            else:
                fh = raw_fh  # type: ignore

            if fmt in ("csv", "parquet", "jsonl"):
                # CSV/JSONL streaming from ClickHouse rows
                if fmt == "csv":
                    text_buffer = io.TextIOWrapper(fh, encoding="utf-8", newline="")
                    writer = csv.writer(text_buffer)
                    writer.writerow(column_names)
                    for row in ch.execute_iter(sql, params=query_params):
                        writer.writerow(list(row))
                    text_buffer.flush()
                elif fmt == "jsonl":
                    for row in ch.execute_iter(sql, params=query_params):
                        obj = {column_names[i]: row[i] for i in range(len(column_names))}
                        line = (json.dumps(obj, default=str) + "\n").encode("utf-8")
                        fh.write(line)
                elif fmt == "parquet":
                    # Build Arrow table in chunks to control memory footprint
                    batch_size = 100_000
                    batches: List[pa.RecordBatch] = []
                    cols = {name: [] for name in column_names}
                    count = 0
                    for row in ch.execute_iter(sql, params=query_params):
                        for i, name in enumerate(column_names):
                            cols[name].append(row[i])
                        count += 1
                        if count % batch_size == 0:
                            arrays = [pa.array(cols[name]) for name in column_names]
                            batches.append(pa.RecordBatch.from_arrays(arrays, schema=pa.schema([(n, pa.null()) for n in column_names])))
                            cols = {name: [] for name in column_names}
                    if any(len(v) for v in cols.values()):
                        arrays = [pa.array(cols[name]) for name in column_names]
                        batches.append(pa.RecordBatch.from_arrays(arrays, schema=pa.schema([(n, pa.null()) for n in column_names])))
                    writer = pq.ParquetWriter(fh, schema=batches[0].schema) if batches else None
                    for b in batches:
                        if writer is None:
                            writer = pq.ParquetWriter(fh, schema=b.schema)
                        writer.write_batch(b)
                    if writer:
                        writer.close()
            else:
                raise HTTPException(status_code=400, detail="Unsupported format")
        finally:
            if isinstance(fh, gzip.GzipFile):  # type: ignore
                fh.close()  # type: ignore
            raw_fh.close()

        # Compute checksum/size and enforce size quota
        stat = os.stat(tmp_path)
        total_bytes = stat.st_size
        if total_bytes > _EXPORT_MAX_SIZE_BYTES:
            await _update_job_status(pool, job_id, status="failed", error=f"Export exceeds max size {_EXPORT_MAX_SIZE_BYTES} bytes")
            return
        with open(tmp_path, "rb") as fchk:
            for chunk in iter(lambda: fchk.read(1024 * 1024), b""):
                hasher.update(chunk)
        checksum = hasher.hexdigest()

        # Upload to MinIO
        object_key = f"exports/{job_id}{suffix}"
        bucket = os.getenv("EXPORT_BUCKET", "research-exports")
        minio_client = _get_minio_client()
        if minio_client is None:
            await _update_job_status(
                pool, job_id, status="failed", error="MinIO client unavailable on server"
            )
            return

        # Ensure bucket exists
        try:
            found = minio_client.bucket_exists(bucket)
            if not found:
                minio_client.make_bucket(bucket)
        except S3Error as e:
            await _update_job_status(pool, job_id, status="failed", error=str(e))
            return

        # Upload
        with open(tmp_path, "rb") as fup:
            minio_client.put_object(
                bucket, object_key, fup, length=total_bytes, content_type="text/csv"
            )

        expires_at = datetime.utcnow() + timedelta(days=int(os.getenv("EXPORT_TTL_DAYS", "7")))
        await _update_job_status(
            pool,
            job_id,
            status="completed",
            object_key=object_key,
            size_bytes=total_bytes,
            checksum=checksum,
            expires_at=expires_at,
        )
    except Exception as e:  # pragma: no cover - best effort
        await _update_job_status(pool, job_id, status="failed", error=str(e))
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


async def cleanup_expired_exports() -> None:
    """Delete expired export objects from MinIO and prune job records."""
    try:
        pool = await get_postgres_pool()
        await _ensure_job_table(pool)
        minio_client = _get_minio_client()
        bucket = os.getenv("EXPORT_BUCKET", "research-exports")
        # Find expired
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT job_id, object_key FROM export_job WHERE expires_at IS NOT NULL AND expires_at < now()"
            )
        # Delete objects first
        if minio_client:
            for row in rows:
                key = row["object_key"]
                if key:
                    try:
                        minio_client.remove_object(bucket, key)
                    except Exception:
                        pass
        # Delete job records
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM export_job WHERE expires_at IS NOT NULL AND expires_at < now()")
    except Exception:
        # Best-effort cleanup; do not raise
        pass


def schedule_export_cleanup(loop: asyncio.AbstractEventLoop) -> None:
    """Schedule periodic export cleanup task on the given event loop."""
    async def _runner():
        interval_minutes = int(os.getenv("EXPORT_CLEANUP_INTERVAL_MIN", "60"))
        while True:
            try:
                await cleanup_expired_exports()
            except Exception:
                pass
            await asyncio.sleep(interval_minutes * 60)

    loop.create_task(_runner())


@router.post("/jobs", response_model=ExportJobStatus)
async def create_export_job(
    payload: ExportJobCreate,
    background_tasks: BackgroundTasks,
):
    try:
        await check_entitlement("data_warehouse")
        await check_entitlement("research_export")

        pool = await get_postgres_pool()
        await _ensure_job_table(pool)

        # Idempotency check
        if payload.idempotency_key:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT job_id, status, created_at, updated_at, fmt, compression, object_key, size_bytes, checksum, error, expires_at "
                    "FROM export_job WHERE idempotency_key = $1 ORDER BY created_at DESC LIMIT 1",
                    payload.idempotency_key,
                )
                if row:
                    return ExportJobStatus(**dict(row))

        # Build SQL and params now and persist in job params
        built = _build_query(payload)
        job_params = {"__sql__": built["sql"], "__params__": built["params"]}

        # Insert job
        async with pool.acquire() as conn:
            rec = await conn.fetchrow(
                """
                INSERT INTO export_job (job_id, idempotency_key, requester, status, fmt, compression, params)
                VALUES (gen_random_uuid(), $1, $2, 'queued', $3, $4, $5)
                RETURNING job_id, status, created_at, updated_at, fmt, compression, object_key, size_bytes, checksum, error, expires_at
                """,
                payload.idempotency_key,
                "requester",  # TODO: derive from auth context when available
                payload.fmt,
                payload.compression,
                json.dumps(job_params),
            )

        job = ExportJobStatus(**dict(rec))

        # Enqueue background task
        background_tasks.add_task(_run_export_job, job.job_id)

        return job
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=ExportJobStatus)
async def get_export_job(job_id: str):
    try:
        await check_entitlement("data_warehouse")
        pool = await get_postgres_pool()
        await _ensure_job_table(pool)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT job_id, status, created_at, updated_at, fmt, compression, object_key, size_bytes, checksum, error, expires_at FROM export_job WHERE job_id = $1",
                job_id,
            )
        if not row:
            raise HTTPException(status_code=404, detail="Export job not found")
        return ExportJobStatus(**dict(row))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/download")
async def download_export(job_id: str):
    try:
        await check_entitlement("data_warehouse")
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT object_key, status FROM export_job WHERE job_id = $1",
                job_id,
            )
        if not row:
            raise HTTPException(status_code=404, detail="Export job not found")
        if row["status"] != "completed":
            raise HTTPException(status_code=400, detail="Job not completed")

        object_key = row["object_key"]
        if not object_key:
            raise HTTPException(status_code=500, detail="Missing object key")

        minio_client = _get_minio_client()
        if minio_client is None:
            raise HTTPException(status_code=500, detail="MinIO unavailable")
        bucket = os.getenv("EXPORT_BUCKET", "research-exports")
        url = minio_client.presigned_get_object(bucket, object_key, expires=timedelta(hours=1))
        return {"url": url, "expires_in": 3600}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preview")
async def preview_export(
    dataset_id: Optional[str] = Query(None),
    sql: Optional[str] = Query(None),
    params: Optional[str] = Query(None, description="JSON-encoded params"),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """Return a small preview sample for a given dataset or SQL."""
    try:
        await check_entitlement("data_warehouse")
        payload = ExportJobCreate(
            dataset_id=dataset_id,
            sql=sql,
            params=json.loads(params) if params else None,
            start_date=start_date,
            end_date=end_date,
        )
        built = _build_query(payload)
        ch = get_clickhouse_client()
        sample_sql = f"SELECT * FROM ({built['sql']}) LIMIT {limit}"
        rows = ch.execute(sample_sql, params=built["params"])
        return {"count": len(rows), "rows": rows}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


