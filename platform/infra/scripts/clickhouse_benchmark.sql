-- Baseline metrics before changes
SELECT
  toStartOfMinute(event_time) AS minute,
  count() AS queries,
  sum(query_duration_ms) AS total_ms,
  quantilesExact(0.5, 0.95, 0.99)(query_duration_ms) AS p50_p95_p99_ms,
  sum(read_rows) AS read_rows,
  sum(read_bytes) AS read_bytes
FROM system.query_log
WHERE event_time >= now() - INTERVAL 15 MINUTE
  AND type = 'QueryFinish'
GROUP BY minute
ORDER BY minute DESC
LIMIT 15;

-- Top 10 slow queries (last 30 minutes)
SELECT
  query_id,
  user,
  query_duration_ms,
  read_rows,
  read_bytes,
  query
FROM system.query_log
WHERE event_time >= now() - INTERVAL 30 MINUTE
  AND type = 'QueryFinish'
ORDER BY query_duration_ms DESC
LIMIT 10;

-- Aggregated by query_kind
SELECT
  query_kind,
  count() AS cnt,
  quantileExact(0.95)(query_duration_ms) AS p95_ms,
  sum(read_rows) AS read_rows,
  sum(read_bytes) AS read_bytes
FROM system.query_log
WHERE event_time >= now() - INTERVAL 30 MINUTE
  AND type = 'QueryFinish'
GROUP BY query_kind
ORDER BY p95_ms DESC;

-- Table-level part and merge pressure
SELECT
  table,
  sum(bytes) AS bytes,
  count() AS parts
FROM system.parts
WHERE active
  AND database = 'market_intelligence'
GROUP BY table
ORDER BY parts DESC
LIMIT 20;


