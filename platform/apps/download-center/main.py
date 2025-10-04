"""
Enhanced Download Center Service with Scheduled Exports and CAISO Support
Signed URL generation, exports, batch capabilities, and automated scheduling.
"""
import logging
import uuid
import asyncio
import json
from datetime import datetime, timedelta, time
from typing import Optional, List, Dict, Any
from enum import Enum

import boto3
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Download Center",
    description="Data export and signed download service",
    version="1.0.0",
)

class ExportFormat(str, Enum):
    """Supported export formats."""
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    XLSX = "xlsx"


class ExportStatus(str, Enum):
    """Export job status."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ScheduleFrequency(str, Enum):
    """Schedule frequencies for automated exports."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    HOURLY = "hourly"


class ExportType(str, Enum):
    """Types of exports."""
    PRICE_DATA = "price_data"
    CURVE_DATA = "curve_data"
    FUNDAMENTALS = "fundamentals"
    COMPLIANCE = "compliance"


# Enhanced models

class ScheduledExportRequest(BaseModel):
    """Request for creating a scheduled export."""
    name: str
    description: str
    export_type: ExportType
    instrument_ids: List[str]
    format: ExportFormat = ExportFormat.PARQUET
    frequency: ScheduleFrequency
    recipients: List[str]  # Email addresses
    start_date: str
    end_date: str
    include_fundamentals: bool = False
    caiso_compliance: bool = False  # CAISO-specific compliance format


class ExportJob(BaseModel):
    """Export job information."""
    export_id: str
    name: str
    status: ExportStatus
    export_type: ExportType
    format: ExportFormat
    created_at: datetime
    completed_at: Optional[datetime] = None
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None
    file_size: Optional[int] = None
    error_message: Optional[str] = None


class ScheduledExport(BaseModel):
    """Scheduled export configuration."""
    schedule_id: str
    name: str
    description: str
    export_type: ExportType
    instrument_ids: List[str]
    format: ExportFormat
    frequency: ScheduleFrequency
    recipients: List[str]
    start_date: str
    end_date: str
    include_fundamentals: bool
    caiso_compliance: bool
    next_run: datetime
    created_at: datetime
    active: bool = True


# MinIO/S3 client with CAISO-specific bucket
s3_client = boto3.client(
    "s3",
    endpoint_url="http://minio:9000",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
)

# Scheduler for automated exports
scheduler = AsyncIOScheduler()
scheduler.start()


class ExportManager:
    """Manage export jobs and scheduled exports."""

    def __init__(self):
        self.active_exports: Dict[str, ExportJob] = {}
        self.scheduled_exports: Dict[str, ScheduledExport] = {}
        self.export_history: List[Dict] = []

    async def create_export_job(self, request: ScheduledExportRequest) -> str:
        """Create a new export job."""
        export_id = str(uuid.uuid4())

        export_job = ExportJob(
            export_id=export_id,
            name=request.name,
            status=ExportStatus.QUEUED,
            export_type=request.export_type,
            format=request.format,
            created_at=datetime.utcnow(),
        )

        self.active_exports[export_id] = export_job

        # Process export in background
        asyncio.create_task(self.process_export(export_id, request))

        logger.info(f"Created export job {export_id} for {request.name}")
        return export_id

    async def process_export(self, export_id: str, request: ScheduledExportRequest):
        """Process an export job."""
        try:
            export_job = self.active_exports[export_id]
            export_job.status = ExportStatus.PROCESSING

            # Generate filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{request.export_type.value}_{request.name}_{timestamp}.{request.format.value}"
            bucket = "caiso-exports" if request.caiso_compliance else "downloads"

            # Generate export data based on type
            export_data = await self.generate_export_data(request)

            # Upload to MinIO/S3
            await self.upload_export_data(bucket, filename, export_data, request.format)

            # Generate signed URL
            download_url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": filename},
                ExpiresIn=3600 * 24 * 7,  # 7 days for scheduled exports
            )

            # Update export job
            export_job.status = ExportStatus.COMPLETED
            export_job.completed_at = datetime.utcnow()
            export_job.download_url = download_url
            export_job.expires_at = datetime.utcnow() + timedelta(days=7)

            # Send notifications
            if request.recipients:
                await self.send_export_notifications(request.recipients, export_job, download_url)

            logger.info(f"Export job {export_id} completed successfully")

        except Exception as e:
            logger.error(f"Export job {export_id} failed: {e}")
            export_job = self.active_exports.get(export_id)
            if export_job:
                export_job.status = ExportStatus.FAILED
                export_job.error_message = str(e)

    async def generate_export_data(self, request: ScheduledExportRequest) -> Any:
        """Generate export data based on request type."""
        # This would integrate with the actual data sources
        # For now, return mock data structure

        if request.export_type == ExportType.PRICE_DATA:
            return {
                "type": "price_data",
                "instruments": request.instrument_ids,
                "period": f"{request.start_date} to {request.end_date}",
                "data": [
                    {
                        "instrument_id": inst_id,
                        "timestamp": "2024-01-01T00:00:00Z",
                        "price": 50.0 + i * 0.1,
                        "volume": 100.0 + i * 5.0
                    }
                    for i, inst_id in enumerate(request.instrument_ids)
                ]
            }

        elif request.export_type == ExportType.CURVE_DATA:
            return {
                "type": "curve_data",
                "instruments": request.instrument_ids,
                "scenario": "BASE",
                "data": [
                    {
                        "instrument_id": inst_id,
                        "delivery_date": "2024-01-01",
                        "price": 45.0 + i * 0.05,
                        "tenor": "monthly"
                    }
                    for i, inst_id in enumerate(request.instrument_ids)
                ]
            }

        elif request.export_type == ExportType.COMPLIANCE:
            # CAISO-specific compliance format
            return {
                "type": "compliance",
                "market": "CAISO",
                "report_type": "settlement_data",
                "period": f"{request.start_date} to {request.end_date}",
                "data": {
                    "settlement_summary": {
                        "total_settled_volume": 1000000,
                        "total_settled_amount": 50000000,
                        "avg_price": 50.0
                    },
                    "node_details": [
                        {
                            "node_id": inst_id,
                            "settled_volume": 1000,
                            "settled_amount": 50000,
                            "avg_price": 50.0
                        }
                        for inst_id in request.instrument_ids
                    ]
                }
            }

        return {"type": "mock_data", "message": "Export data generated"}

    async def upload_export_data(self, bucket: str, filename: str, data: Any, format: ExportFormat):
        """Upload export data to MinIO/S3."""
        try:
            # Convert data to appropriate format
            if format == ExportFormat.CSV:
                # Convert to CSV format
                import csv
                import io
                output = io.StringIO()
                if isinstance(data, dict) and "data" in data:
                    writer = csv.DictWriter(output, fieldnames=data["data"][0].keys())
                    writer.writeheader()
                    writer.writerows(data["data"])
                content = output.getvalue()
            elif format == ExportFormat.JSON:
                content = json.dumps(data, indent=2, default=str)
            else:
                # For Parquet and other formats, would need additional libraries
                content = json.dumps(data, default=str)

            # Upload to S3
            s3_client.put_object(
                Bucket=bucket,
                Key=filename,
                Body=content,
                ContentType="application/octet-stream"
            )

            logger.info(f"Uploaded {filename} to {bucket}")

        except Exception as e:
            logger.error(f"Error uploading export data: {e}")
            raise

    async def send_export_notifications(self, recipients: List[str], export_job: ExportJob, download_url: str):
        """Send export completion notifications."""
        # This would integrate with email service
        for recipient in recipients:
            logger.info(f"Sending export notification to {recipient}: {export_job.name} - {download_url}")

    async def create_scheduled_export(self, request: ScheduledExportRequest) -> str:
        """Create a scheduled export."""
        schedule_id = str(uuid.uuid4())

        # Calculate next run time
        next_run = self._calculate_next_run(request.frequency)

        scheduled_export = ScheduledExport(
            schedule_id=schedule_id,
            name=request.name,
            description=request.description,
            export_type=request.export_type,
            instrument_ids=request.instrument_ids,
            format=request.format,
            frequency=request.frequency,
            recipients=request.recipients,
            start_date=request.start_date,
            end_date=request.end_date,
            include_fundamentals=request.include_fundamentals,
            caiso_compliance=request.caiso_compliance,
            next_run=next_run,
            created_at=datetime.utcnow(),
        )

        self.scheduled_exports[schedule_id] = scheduled_export

        # Schedule the job
        await self.schedule_export_job(scheduled_export)

        logger.info(f"Created scheduled export {schedule_id}: {request.name}")
        return schedule_id

    def _calculate_next_run(self, frequency: ScheduleFrequency) -> datetime:
        """Calculate next run time for a schedule."""
        now = datetime.utcnow()

        if frequency == ScheduleFrequency.HOURLY:
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif frequency == ScheduleFrequency.DAILY:
            return (now + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)  # 6 AM UTC
        elif frequency == ScheduleFrequency.WEEKLY:
            # Next Monday at 6 AM UTC
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            return (now + timedelta(days=days_until_monday)).replace(hour=6, minute=0, second=0, microsecond=0)
        elif frequency == ScheduleFrequency.MONTHLY:
            # First day of next month at 6 AM UTC
            if now.month == 12:
                next_month = now.replace(year=now.year + 1, month=1, day=1, hour=6, minute=0, second=0, microsecond=0)
            else:
                next_month = now.replace(month=now.month + 1, day=1, hour=6, minute=0, second=0, microsecond=0)
            return next_month

        return now + timedelta(hours=1)  # Default to hourly

    async def schedule_export_job(self, scheduled_export: ScheduledExport):
        """Schedule an export job with APScheduler."""
        try:
            # Create cron expression based on frequency
            if scheduled_export.frequency == ScheduleFrequency.HOURLY:
                cron_expr = "0 * * * *"  # Every hour
            elif scheduled_export.frequency == ScheduleFrequency.DAILY:
                cron_expr = "0 6 * * *"  # Daily at 6 AM UTC
            elif scheduled_export.frequency == ScheduleFrequency.WEEKLY:
                cron_expr = "0 6 * * 1"  # Weekly on Monday at 6 AM UTC
            elif scheduled_export.frequency == ScheduleFrequency.MONTHLY:
                cron_expr = "0 6 1 * *"  # Monthly on 1st at 6 AM UTC
            else:
                cron_expr = "0 6 * * *"  # Default daily

            # Schedule the job
            scheduler.add_job(
                self._run_scheduled_export,
                CronTrigger.from_crontab(cron_expr),
                id=scheduled_export.schedule_id,
                args=[scheduled_export.schedule_id],
                replace_existing=True
            )

            logger.info(f"Scheduled export job {scheduled_export.schedule_id} with cron: {cron_expr}")

        except Exception as e:
            logger.error(f"Error scheduling export job {scheduled_export.schedule_id}: {e}")

    async def _run_scheduled_export(self, schedule_id: str):
        """Run a scheduled export job."""
        try:
            scheduled_export = self.scheduled_exports.get(schedule_id)
            if not scheduled_export or not scheduled_export.active:
                return

            # Create export request from scheduled export
            export_request = ScheduledExportRequest(
                name=scheduled_export.name,
                description=scheduled_export.description,
                export_type=scheduled_export.export_type,
                instrument_ids=scheduled_export.instrument_ids,
                format=scheduled_export.format,
                frequency=scheduled_export.frequency,
                recipients=scheduled_export.recipients,
                start_date=scheduled_export.start_date,
                end_date=scheduled_export.end_date,
                include_fundamentals=scheduled_export.include_fundamentals,
                caiso_compliance=scheduled_export.caiso_compliance,
            )

            # Create and process export
            export_id = await self.create_export_job(export_request)

            # Update next run time
            scheduled_export.next_run = self._calculate_next_run(scheduled_export.frequency)

            logger.info(f"Completed scheduled export {schedule_id}, next run: {scheduled_export.next_run}")

        except Exception as e:
            logger.error(f"Error running scheduled export {schedule_id}: {e}")


# Global export manager
export_manager = ExportManager()


# Legacy models (for backward compatibility)
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


# Legacy API endpoints (for backward compatibility)

@app.post("/api/v1/exports", response_model=ExportResponse)
async def create_export(request: ExportRequest):
    """Create a data export job (legacy endpoint)."""
    # Convert legacy request to new format
    scheduled_request = ScheduledExportRequest(
        name=f"Export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        description="Legacy export request",
        export_type=ExportType.PRICE_DATA,
        instrument_ids=request.instrument_ids,
        format=ExportFormat(request.format),
        frequency=ScheduleFrequency.HOURLY,  # One-time export
        recipients=[],  # No email notifications for legacy
        start_date=request.start_date,
        end_date=request.end_date,
        include_fundamentals=request.include_fundamentals,
        caiso_compliance=False,
    )

    export_id = await export_manager.create_export_job(scheduled_request)

    # Return in legacy format
    export_job = export_manager.active_exports[export_id]
    return ExportResponse(
        export_id=export_id,
        status=export_job.status.value,
        download_url=export_job.download_url,
        expires_at=export_job.expires_at,
    )


@app.get("/api/v1/exports/{export_id}", response_model=ExportResponse)
async def get_export_status(export_id: str):
    """Get export job status (legacy endpoint)."""
    export_job = export_manager.active_exports.get(export_id)
    if not export_job:
        raise HTTPException(status_code=404, detail="Export not found")

    return ExportResponse(
        export_id=export_id,
        status=export_job.status.value,
        download_url=export_job.download_url,
        expires_at=export_job.expires_at,
    )


# New API endpoints for scheduled exports

@app.post("/api/v1/scheduled-exports", response_model=dict)
async def create_scheduled_export(request: ScheduledExportRequest):
    """Create a scheduled export."""
    schedule_id = await export_manager.create_scheduled_export(request)

    return {
        "schedule_id": schedule_id,
        "status": "created",
        "message": "Scheduled export created successfully"
    }


@app.get("/api/v1/scheduled-exports", response_model=List[ScheduledExport])
async def list_scheduled_exports():
    """List all scheduled exports."""
    return list(export_manager.scheduled_exports.values())


@app.get("/api/v1/scheduled-exports/{schedule_id}", response_model=ScheduledExport)
async def get_scheduled_export(schedule_id: str):
    """Get a specific scheduled export."""
    scheduled_export = export_manager.scheduled_exports.get(schedule_id)
    if not scheduled_export:
        raise HTTPException(status_code=404, detail="Scheduled export not found")

    return scheduled_export


@app.delete("/api/v1/scheduled-exports/{schedule_id}")
async def delete_scheduled_export(schedule_id: str):
    """Delete a scheduled export."""
    if schedule_id not in export_manager.scheduled_exports:
        raise HTTPException(status_code=404, detail="Scheduled export not found")

    # Remove from scheduler
    scheduler.remove_job(schedule_id)

    # Remove from memory
    del export_manager.scheduled_exports[schedule_id]

    return {"status": "deleted", "schedule_id": schedule_id}


@app.get("/api/v1/export-jobs", response_model=List[ExportJob])
async def list_export_jobs():
    """List all export jobs."""
    return list(export_manager.active_exports.values())


@app.get("/api/v1/export-jobs/{export_id}", response_model=ExportJob)
async def get_export_job(export_id: str):
    """Get a specific export job."""
    export_job = export_manager.active_exports.get(export_id)
    if not export_job:
        raise HTTPException(status_code=404, detail="Export job not found")

    return export_job


# CAISO-specific endpoints

@app.post("/api/v1/caiso/exports/compliance")
async def create_caiso_compliance_export(
    instrument_ids: List[str],
    start_date: str,
    end_date: str,
    recipients: List[str] = [],
):
    """Create CAISO compliance export (settlement data format)."""
    request = ScheduledExportRequest(
        name=f"CAISO_Compliance_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        description="CAISO regulatory compliance export",
        export_type=ExportType.COMPLIANCE,
        instrument_ids=instrument_ids,
        format=ExportFormat.CSV,  # Compliance typically requires CSV
        frequency=ScheduleFrequency.DAILY,
        recipients=recipients,
        start_date=start_date,
        end_date=end_date,
        caiso_compliance=True,
    )

    schedule_id = await export_manager.create_scheduled_export(request)

    return {
        "schedule_id": schedule_id,
        "status": "created",
        "message": "CAISO compliance export scheduled",
        "format": "CSV (CAISO compliance format)",
        "frequency": "Daily"
    }


@app.post("/api/v1/caiso/exports/curve-analysis")
async def create_caiso_curve_export(
    instrument_ids: List[str],
    scenario_id: str = "BASE",
    recipients: List[str] = [],
):
    """Create CAISO curve analysis export."""
    request = ScheduledExportRequest(
        name=f"CAISO_Curve_Analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        description="CAISO forward curve analysis export",
        export_type=ExportType.CURVE_DATA,
        instrument_ids=instrument_ids,
        format=ExportFormat.PARQUET,
        frequency=ScheduleFrequency.DAILY,
        recipients=recipients,
        start_date=(datetime.utcnow() - timedelta(days=1)).date().isoformat(),
        end_date=datetime.utcnow().date().isoformat(),
        include_fundamentals=True,
    )

    schedule_id = await export_manager.create_scheduled_export(request)

    return {
        "schedule_id": schedule_id,
        "status": "created",
        "message": "CAISO curve analysis export scheduled",
        "scenario": scenario_id,
        "includes_fundamentals": True
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

