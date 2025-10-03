"""
Download Center Service
Signed URL generation, exports, and batch capabilities.
"""
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

import boto3
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Download Center",
    description="Data export and signed download service",
    version="1.0.0",
)

# MinIO/S3 client
s3_client = boto3.client(
    "s3",
    endpoint_url="http://minio:9000",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
)


class ExportRequest(BaseModel):
    instrument_ids: list[str]
    start_date: str
    end_date: str
    format: str = "parquet"  # csv, parquet
    include_fundamentals: bool = False


class ExportResponse(BaseModel):
    export_id: str
    status: str
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/v1/exports", response_model=ExportResponse)
async def create_export(request: ExportRequest):
    """Create a data export job."""
    export_id = str(uuid.uuid4())
    
    logger.info(f"Creating export {export_id} for {len(request.instrument_ids)} instruments")
    
    # TODO: Queue export job
    # For now, generate a mock download URL
    
    filename = f"export_{export_id}.{request.format}"
    bucket = "downloads"
    
    # Generate presigned URL (valid for 1 hour)
    try:
        download_url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": filename},
            ExpiresIn=3600,
        )
        
        return ExportResponse(
            export_id=export_id,
            status="completed",
            download_url=download_url,
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
    except Exception as e:
        logger.error(f"Error generating download URL: {e}")
        return ExportResponse(
            export_id=export_id,
            status="queued",
        )


@app.get("/api/v1/exports/{export_id}", response_model=ExportResponse)
async def get_export_status(export_id: str):
    """Get export job status."""
    # TODO: Query from database
    return ExportResponse(
        export_id=export_id,
        status="completed",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

