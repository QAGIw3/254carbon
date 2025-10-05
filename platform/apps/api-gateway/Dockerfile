FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY platform/apps/api-gateway/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy API Gateway application code
COPY platform/apps/api-gateway /app

# Copy legacy gateway modules that API Gateway imports (routers, shared code)
COPY platform/apps/gateway /app/gateway

# No data connectors included in API Gateway image

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

