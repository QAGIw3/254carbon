FROM python:3.11-slim

# Base image for Analytics Service (ML + analytics). Use slim variant to keep
# image size small while still supporting common build tools.

WORKDIR /app

# Install system dependencies (C/C++ toolchain for scientific packages)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose REST port for FastAPI/uvicorn
EXPOSE 8008

# Run the application
# In production, prefer a process manager and proper logging configuration.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8008"]
