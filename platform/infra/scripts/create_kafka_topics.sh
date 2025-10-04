#!/usr/bin/env bash
set -euo pipefail

# Usage: KAFKA_BROKER=broker:9092 ./create_kafka_topics.sh

BROKER=${KAFKA_BROKER:-localhost:9092}

create_topic() {
  local name=$1
  local partitions=$2
  local rf=$3
  shift 3
  local configs=$*

  echo "Creating topic ${name} (partitions=${partitions}, rf=${rf})"
  kafka-topics --bootstrap-server "$BROKER" \
    --create --if-not-exists \
    --topic "$name" \
    --partitions "$partitions" \
    --replication-factor "$rf"

  if [[ -n "$configs" ]]; then
    echo "Configuring topic ${name}: ${configs}"
    kafka-configs --bootstrap-server "$BROKER" \
      --alter --topic "$name" \
      --add-config "$configs"
  fi
}

# commodities.ticks.v1: 96 partitions, RF=3, 7d retention, zstd
create_topic "commodities.ticks.v1" 96 3 \
  "retention.ms=604800000,compression.type=zstd,min.insync.replicas=2,message.timestamp.type=CreateTime"

# commodities.futures.v1: 24 partitions, RF=3, 30d retention
create_topic "commodities.futures.v1" 24 3 \
  "retention.ms=2592000000,compression.type=zstd,min.insync.replicas=2,message.timestamp.type=CreateTime"

# market.fundamentals: 12 partitions, RF=3, 14d retention
create_topic "market.fundamentals" 12 3 \
  "retention.ms=1209600000,compression.type=zstd,min.insync.replicas=2,message.timestamp.type=CreateTime"

echo "Kafka topics created/configured."


