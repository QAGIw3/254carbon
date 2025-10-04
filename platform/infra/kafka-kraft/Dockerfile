FROM eclipse-temurin:17-jre-jammy

ENV KAFKA_VERSION=3.7.1 \
    SCALA_VERSION=2.13 \
    KAFKA_HOME=/opt/kafka \
    PATH=/opt/kafka/bin:$PATH \
    LOG_DIRS=/var/lib/kafka/data

WORKDIR /opt

# Download Kafka from Apache archive for deterministic availability
RUN set -eux; \
    curl -fsSL https://archive.apache.org/dist/kafka/${KAFKA_VERSION}/kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz -o kafka.tgz; \
    tar -xzf kafka.tgz; \
    mv kafka_${SCALA_VERSION}-${KAFKA_VERSION} ${KAFKA_HOME}; \
    rm -f kafka.tgz; \
    mkdir -p ${LOG_DIRS}

# Copy startup script
COPY start.sh /usr/local/bin/start-kafka.sh
RUN chmod +x /usr/local/bin/start-kafka.sh

EXPOSE 9092 9093
VOLUME ["/var/lib/kafka/data"]

ENTRYPOINT ["/usr/local/bin/start-kafka.sh"]

