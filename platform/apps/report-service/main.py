"""
Report Service
HTML/PDF generation with charts and monthly market briefs.
"""
import logging
from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Report Service",
    description="Market report generation",
    version="1.0.0",
)


class ReportRequest(BaseModel):
    report_type: str  # monthly_brief, custom
    market: str
    as_of_date: date
    format: str = "pdf"  # html, pdf


class ReportResponse(BaseModel):
    report_id: str
    status: str
    download_url: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/v1/reports", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """Generate a market report."""
    import uuid
    
    report_id = str(uuid.uuid4())
    
    logger.info(
        f"Generating {request.report_type} report for {request.market}, "
        f"as of {request.as_of_date}"
    )
    
    # TODO: Implement report generation
    # 1. Query data from ClickHouse
    # 2. Generate charts
    # 3. Render HTML template
    # 4. Convert to PDF if requested
    # 5. Store in MinIO
    
    return ReportResponse(
        report_id=report_id,
        status="queued",
    )


@app.get("/api/v1/reports/{report_id}", response_model=ReportResponse)
async def get_report(report_id: str):
    """Get report status and download URL."""
    # TODO: Query from database
    return ReportResponse(
        report_id=report_id,
        status="completed",
        download_url=f"http://minio:9000/reports/{report_id}.pdf",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

