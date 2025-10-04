"""
CAISO Backfill DAG

Manually-triggered backfill with chunked windows and checkpoint-driven progress.
RTM: 2-hour chunks; DAM: 1-day chunks.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.providers.postgres.hooks.postgres import PostgresHook

import sys
import os

# Make connectors importable
sys.path.append(os.path.join(os.path.dirname(__file__), '../../connectors'))

from caiso_connector import CAISOConnector


default_args = {
    'owner': 'market-intelligence',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'email': ['ops@254carbon.ai'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def parse_date(value: str) -> datetime:
    # Accept YYYY-MM-DD or full ISO; return UTC-aware midnight when date-only
    try:
        if len(value) == 10:
            dt = datetime.strptime(value, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        # Fallback to now-7d if parse fails
        return (datetime.now(timezone.utc) - timedelta(days=7))


def run_caiso_backfill(**context):
    """Run CAISO backfill over chunked windows with checkpoint resume."""
    dag_conf: Dict[str, Any] = (context.get('dag_run') and context['dag_run'].conf) or {}

    market_type = str(dag_conf.get('market_type', 'RTM')).upper()  # RTM or DAM
    start_date = parse_date(dag_conf.get('start_date', (datetime.now(timezone.utc) - timedelta(days=3)).strftime('%Y-%m-%d')))
    end_date = parse_date(dag_conf.get('end_date', datetime.now(timezone.utc).strftime('%Y-%m-%d')))

    # Normalize end_date to be inclusive upper bound
    if end_date < start_date:
        raise ValueError('end_date must be >= start_date')

    # Nodes to backfill: default to hub-only pilot nodes
    nodes: List[str] = dag_conf.get('nodes') or [
        'TH_SP15_GEN-APND',
        'TH_NP15_GEN-APND',
        'TH_ZP26_GEN-APND',
    ]

    kafka_topic = dag_conf.get('kafka_topic', 'power.ticks.v1')
    kafka_bootstrap = dag_conf.get('kafka_bootstrap', 'kafka:9092')

    # Chunk sizes
    chunk = timedelta(hours=2) if market_type == 'RTM' else timedelta(days=1)

    total_events = 0
    nodes_processed = 0

    for node in nodes:
        source_id = f"caiso_backfill_{market_type.lower()}_{node}"

        # Determine resume point from checkpoint, if any
        resume_from = start_date
        # Load previous checkpoint (if exists) to support resume
        checkpoint_probe = CAISOConnector({
            'source_id': source_id,
            'market_type': market_type,
            'hub_only': True,
            'entitlements_enabled': True,
            'kafka_topic': kafka_topic,
            'kafka_bootstrap': kafka_bootstrap,
        })
        ckpt = checkpoint_probe.load_checkpoint()
        if ckpt and isinstance(ckpt.get('last_event_time'), (int, float)):
            # Advance one second past last event time
            resume_from = max(start_date, datetime.fromtimestamp(ckpt['last_event_time'] / 1000, tz=timezone.utc) + timedelta(seconds=1))

        cursor = resume_from
        while cursor <= end_date:
            window_end = min(cursor + chunk, end_date)

            # Instantiate connector for this window chunk with overrides
            connector = CAISOConnector({
                'source_id': source_id,
                'market_type': market_type,
                'hub_only': True,
                'entitlements_enabled': True,
                'kafka_topic': kafka_topic,
                'kafka_bootstrap': kafka_bootstrap,
                'override_start': cursor,
                'override_end': window_end,
                'timeout_seconds': dag_conf.get('timeout_seconds', 30),
                'max_retries': dag_conf.get('max_retries', 3),
                'retry_backoff_base': dag_conf.get('retry_backoff_base', 1.0),
                'dev_mode': dag_conf.get('dev_mode', False),
            })

            events_processed = connector.run()
            total_events += events_processed or 0

            # Quality checks: bounds and staleness for this chunk
            try:
                _quality_check_and_metrics(
                    node=node,
                    market_type=market_type,
                    chunk_start=cursor,
                    chunk_end=window_end,
                    source_id=source_id,
                )
            except Exception as qc_err:
                # Surface as failure for the DAG
                raise

            # Advance cursor; if nothing processed, still move forward to avoid infinite loops
            cursor = (window_end + timedelta(seconds=1))

        nodes_processed += 1

    # Push summary metrics
    ti = context['task_instance']
    ti.xcom_push(key='nodes_processed', value=nodes_processed)
    ti.xcom_push(key='events_processed_total', value=total_events)

    return {
        'nodes_processed': nodes_processed,
        'events_processed_total': total_events,
        'market_type': market_type,
    }


def _quality_check_and_metrics(
    *,
    node: str,
    market_type: str,
    chunk_start: datetime,
    chunk_end: datetime,
    source_id: str,
) -> None:
    """Check event count, value bounds, and staleness; push metrics to Prometheus."""
    pg = PostgresHook(postgres_conn_id='market_intelligence_db')

    instr = f"CAISO.{node}"
    sql = """
        SELECT COUNT(*) AS cnt,
               MAX(event_time) AS latest,
               MIN(value) AS minv,
               MAX(value) AS maxv
        FROM ch.market_price_ticks
        WHERE instrument_id = %s
          AND product = 'lmp'
          AND source = %s
          AND event_time >= %s
          AND event_time <= %s
    """
    result = pg.get_first(sql, parameters=(instr, source_id, chunk_start, chunk_end))

    cnt = result[0] if result and result[0] is not None else 0
    latest = result[1] if result else None
    minv = result[2] if result else None
    maxv = result[3] if result else None

    # Thresholds
    min_cnt = 8 if market_type == 'RTM' else 12
    tol_seconds = 600 if market_type == 'RTM' else 7200

    quality_pass = True
    reasons: list[str] = []

    if cnt < min_cnt:
        quality_pass = False
        reasons.append(f"low_count:{cnt}")
    if minv is None or maxv is None:
        quality_pass = False
        reasons.append("no_values")
    else:
        if minv < -5000 or maxv > 5000:
            quality_pass = False
            reasons.append(f"bounds:{minv},{maxv}")

    lag_seconds = None
    if latest is None:
        quality_pass = False
        reasons.append("no_latest")
        lag_seconds = None
    else:
        # latest can be timezone-aware from CH; ensure both are UTC aware
        try:
            latest_utc = latest if latest.tzinfo else latest.replace(tzinfo=timezone.utc)
        except Exception:
            latest_utc = latest
        lag_seconds = (chunk_end - latest_utc).total_seconds()
        if lag_seconds > tol_seconds:
            quality_pass = False
            reasons.append(f"stale:{int(lag_seconds)}s")

    # Push metrics to Prometheus pushgateway
    try:
        import requests as _r
        labels = f'node="{node}",market="{market_type}"'
        lines = []
        lines.append(f"caiso_backfill_chunk_events_total{{{labels}}} {cnt}")
        if lag_seconds is not None:
            lines.append(f"caiso_backfill_chunk_lag_seconds{{{labels}}} {int(lag_seconds)}")
        if minv is not None:
            lines.append(f"caiso_backfill_chunk_min_value{{{labels}}} {float(minv)}")
        if maxv is not None:
            lines.append(f"caiso_backfill_chunk_max_value{{{labels}}} {float(maxv)}")
        lines.append(f"caiso_backfill_chunk_quality_pass{{{labels}}} {1 if quality_pass else 0}")
        body = "\n".join(lines) + "\n"
        _r.post(
            "http://prometheus:9091/metrics/job/caiso_backfill",
            data=body,
            timeout=5,
        )
    except Exception:
        # Ignore metrics push failures
        pass

    if not quality_pass:
        raise ValueError(
            f"Quality check failed for {instr} {market_type} [{chunk_start}..{chunk_end}]: "
            + ",".join(reasons)
        )


with DAG(
    'caiso_backfill',
    default_args=default_args,
    description='CAISO historical backfill with chunked windows and checkpoint resume',
    schedule_interval=None,  # Manual trigger
    start_date=days_ago(1),
    catchup=False,
    tags=['backfill', 'caiso', 'historical'],
) as dag:

    run_backfill = PythonOperator(
        task_id='run_caiso_backfill',
        python_callable=run_caiso_backfill,
        provide_context=True,
    )
