FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8007

# Run service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8007"]
