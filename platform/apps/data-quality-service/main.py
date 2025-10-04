"""
Data Quality Service

Batch jobs: cross-source validation, outlier detection, imputation
Streaming worker: online outlier flags
"""
import os
import logging
from fastapi import FastAPI

from dq_batch import run_cross_source_validation_job, run_outliers_job, run_imputation_job
from dq_stream import start_stream_worker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dq-service")

app = FastAPI(title="Data Quality Service", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/jobs/cross-source")
async def trigger_cross_source_job():
    run_cross_source_validation_job()
    return {"status": "submitted"}


@app.post("/jobs/outliers")
async def trigger_outliers_job():
    run_outliers_job()
    return {"status": "submitted"}


@app.post("/jobs/imputation")
async def trigger_imputation_job():
    run_imputation_job()
    return {"status": "submitted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8010")))


