FROM python:3.11-slim

# Base image with Python runtime. Use a slim image to reduce size.

WORKDIR /app

# Install minimal system toolchain required by some Python wheels.

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching for deps.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (kept last to optimize rebuilds during dev).
COPY . .

# Expose HTTP port used by uvicorn
EXPOSE 8020

# Run the application
# NOTE: Use `--reload` only in local dev; not recommended in prod images.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8020"]
