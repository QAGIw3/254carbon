FROM python:3.11-slim

#
# Dockerfile for the commodities-service
# - Uses a root-context build to vendor `platform/shared` alongside the app
# - Sets PYTHONPATH=/app:/app/platform to import shared modules directly
# - Exposes port 8012 and includes a simple HTTP healthcheck
#

ARG SERVICE_PATH=platform/apps/commodities-service

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/platform

WORKDIR /app

# Install pinned dependencies
COPY ${SERVICE_PATH}/requirements.txt requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy service source and shared utilities needed at runtime
COPY ${SERVICE_PATH} /app
COPY platform/shared /app/platform/shared

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8012

# Basic container health check; relies on the app to be serving /health
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8012/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8012"]
