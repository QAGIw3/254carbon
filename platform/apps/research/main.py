"""Research service exposing notebook and experiment management APIs."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from clickhouse_driver import Client
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from jupyter_integration import JupyterIntegration

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Research Platform Service",
    description="Notebook orchestration and experiment tracking for research workflows",
    version="1.0.0",
)

_integration = JupyterIntegration()
_ch_client = Client(
    host=os.getenv("CLICKHOUSE_HOST", "clickhouse"),
    port=int(os.getenv("CLICKHOUSE_PORT", "9000")),
    database="market_intelligence",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, (set, tuple)):
        return list(value)
    return value


def _load_json(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _truncate(text: Optional[str], limit: int = 4000) -> Optional[str]:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[:limit] + "... (truncated)"


class NotebookCreateRequest(BaseModel):
    title: str
    description: Optional[str] = None
    template: str = Field("research_template", description="Template key registered in the integration")
    tags: List[str] = Field(default_factory=list)
    author: str = Field("researcher")
    metadata: Optional[Dict[str, Any]] = None


class NotebookExecuteRequest(BaseModel):
    execution_timeout: int = Field(600, ge=60, le=3600)
    max_memory: str = Field("2GB")


class ResearchNotebook(BaseModel):
    notebook_id: str
    title: str
    author: str
    status: str
    path: str
    tags: List[str]
    created_at: datetime
    executed_at: Optional[datetime]
    artifacts: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]


class ExperimentCreateRequest(BaseModel):
    name: str
    model_type: str
    dataset: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    mlflow_run_id: Optional[str] = None


class ExperimentStartRequest(BaseModel):
    mlflow_run_id: Optional[str] = None


class ExperimentCompleteRequest(BaseModel):
    results: Dict[str, Any] = Field(default_factory=dict)
    execution_time: Optional[float] = None
    status: str = Field("completed")
    mlflow_run_id: Optional[str] = None


class ResearchExperiment(BaseModel):
    experiment_id: str
    name: str
    model_type: str
    dataset: str
    parameters: Dict[str, Any]
    status: str
    mlflow_run_id: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]
    results: Optional[Dict[str, Any]]


def _fetch_notebook(notebook_id: str) -> Optional[ResearchNotebook]:
    rows = _ch_client.execute(
        """
        SELECT notebook_id, title, author, status, path, tags, created_at, executed_at, artifacts, metadata
        FROM market_intelligence.research_notebooks
        WHERE notebook_id = %(nid)s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        {"nid": notebook_id},
    )
    if not rows:
        return None
    row = rows[0]
    return ResearchNotebook(
        notebook_id=row[0],
        title=row[1],
        author=row[2],
        status=row[3],
        path=row[4],
        tags=row[5] or [],
        created_at=row[6],
        executed_at=row[7],
        artifacts=_load_json(row[8]),
        metadata=_load_json(row[9]),
    )


def _fetch_experiment(experiment_id: str) -> Optional[ResearchExperiment]:
    rows = _ch_client.execute(
        """
        SELECT experiment_id, name, model_type, dataset, parameters, status, mlflow_run_id,
               started_at, completed_at, results
        FROM market_intelligence.research_experiments
        WHERE experiment_id = %(eid)s
        ORDER BY started_at DESC
        LIMIT 1
        """,
        {"eid": experiment_id},
    )
    if not rows:
        return None
    row = rows[0]
    return ResearchExperiment(
        experiment_id=row[0],
        name=row[1],
        model_type=row[2],
        dataset=row[3],
        parameters=_load_json(row[4]) or {},
        status=row[5],
        mlflow_run_id=row[6],
        started_at=row[7],
        completed_at=row[8],
        results=_load_json(row[9]) or {},
    )


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "healthy", "notebooks_tracked": len(_integration.execution_history)}


@app.post("/api/v1/research/notebooks", response_model=ResearchNotebook)
async def create_notebook(request: NotebookCreateRequest) -> ResearchNotebook:
    notebook_id = _integration.create_research_notebook(
        title=request.title,
        description=request.description or "",
        template=request.template,
        tags=request.tags,
        author=request.author,
    )

    notebook_path = _integration.notebooks_dir / f"{notebook_id}.ipynb"
    artifacts_payload = {
        "artifact_path": str(notebook_path),
        "description": request.description,
    }
    now_ts = datetime.utcnow()

    try:
        _ch_client.execute(
            """
            INSERT INTO market_intelligence.research_notebooks
            (notebook_id, title, author, status, path, tags, created_at, executed_at, artifacts, metadata)
            VALUES
            """,
            [[
                notebook_id,
                request.title,
                request.author,
                "created",
                str(notebook_path),
                request.tags,
                now_ts,
                None,
                json.dumps(artifacts_payload, default=_json_default),
                json.dumps(request.metadata or {}, default=_json_default),
            ]],
            types_check=True,
        )
    except Exception as exc:
        logger.exception("Failed to persist notebook metadata")
        raise HTTPException(status_code=500, detail=f"Failed to record notebook: {exc}") from exc

    notebook = _fetch_notebook(notebook_id)
    if notebook is None:
        raise HTTPException(status_code=500, detail="Notebook metadata persistence failed")
    return notebook


@app.get("/api/v1/research/notebooks", response_model=List[ResearchNotebook])
async def list_notebooks(
    status: Optional[str] = None,
    author: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = 50,
) -> List[ResearchNotebook]:
    conditions = ["1 = 1"]
    params: Dict[str, Any] = {"limit": limit}
    if status:
        conditions.append("status = %(status)s")
        params["status"] = status
    if author:
        conditions.append("author = %(author)s")
        params["author"] = author
    if tag:
        conditions.append("has(tags, %(tag)s)")
        params["tag"] = tag

    query = f"""
        SELECT notebook_id, title, author, status, path, tags, created_at, executed_at, artifacts, metadata
        FROM market_intelligence.research_notebooks
        WHERE {' AND '.join(conditions)}
        ORDER BY created_at DESC
        LIMIT %(limit)s
    """
    rows = _ch_client.execute(query, params)
    notebooks: List[ResearchNotebook] = []
    for row in rows:
        notebooks.append(
            ResearchNotebook(
                notebook_id=row[0],
                title=row[1],
                author=row[2],
                status=row[3],
                path=row[4],
                tags=row[5] or [],
                created_at=row[6],
                executed_at=row[7],
                artifacts=_load_json(row[8]),
                metadata=_load_json(row[9]),
            )
        )
    return notebooks


@app.get("/api/v1/research/notebooks/{notebook_id}", response_model=ResearchNotebook)
async def get_notebook(notebook_id: str) -> ResearchNotebook:
    notebook = _fetch_notebook(notebook_id)
    if notebook is None:
        raise HTTPException(status_code=404, detail=f"Notebook {notebook_id} not found")
    return notebook


@app.post("/api/v1/research/notebooks/{notebook_id}/execute", response_model=ResearchNotebook)
async def execute_notebook(notebook_id: str, request: NotebookExecuteRequest) -> ResearchNotebook:
    result = _integration.execute_notebook(
        notebook_id=notebook_id,
        execution_timeout=request.execution_timeout,
        max_memory=request.max_memory,
    )

    status_map = {
        "completed": "executed",
        "failed": "failed",
        "timeout": "timeout",
        "error": "error",
    }
    new_status = status_map.get(result.get("status"), result.get("status", "executed"))
    executed_at = datetime.utcnow() if new_status == "executed" else None

    artifacts_payload = {
        "execution_id": result.get("execution_id"),
        "status": result.get("status"),
        "return_code": result.get("return_code"),
        "stdout": _truncate(result.get("stdout")),
        "stderr": _truncate(result.get("stderr")),
        "error": result.get("error"),
    }

    try:
        _ch_client.execute(
            """
            ALTER TABLE market_intelligence.research_notebooks
            UPDATE status = %(status)s, executed_at = %(executed_at)s, artifacts = %(artifacts)s
            WHERE notebook_id = %(notebook_id)s
            """,
            {
                "status": new_status,
                "executed_at": executed_at,
                "artifacts": json.dumps(artifacts_payload, default=_json_default),
                "notebook_id": notebook_id,
            },
        )
    except Exception as exc:
        logger.exception("Failed to update notebook execution state")
        raise HTTPException(status_code=500, detail=f"Failed to update notebook status: {exc}") from exc

    notebook = _fetch_notebook(notebook_id)
    if notebook is None:
        raise HTTPException(status_code=500, detail="Notebook state unavailable after execution")
    return notebook


@app.post("/api/v1/research/experiments", response_model=ResearchExperiment)
async def register_experiment(request: ExperimentCreateRequest) -> ResearchExperiment:
    experiment_id = _integration.experiment_registry.register_experiment(
        name=request.name,
        model_type=request.model_type,
        dataset=request.dataset,
        parameters=request.parameters,
        mlflow_run_id=request.mlflow_run_id,
    )

    now_ts = datetime.utcnow()
    try:
        _ch_client.execute(
            """
            INSERT INTO market_intelligence.research_experiments
            (experiment_id, name, model_type, dataset, parameters, status, mlflow_run_id, started_at, completed_at, results)
            VALUES
            """,
            [[
                experiment_id,
                request.name,
                request.model_type,
                request.dataset,
                json.dumps(request.parameters, default=_json_default),
                "registered",
                request.mlflow_run_id,
                now_ts,
                None,
                json.dumps({}, default=_json_default),
            ]],
            types_check=True,
        )
    except Exception as exc:
        logger.exception("Failed to persist experiment metadata")
        raise HTTPException(status_code=500, detail=f"Failed to record experiment: {exc}") from exc

    experiment = _fetch_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=500, detail="Experiment metadata persistence failed")
    return experiment


@app.post("/api/v1/research/experiments/{experiment_id}/start", response_model=ResearchExperiment)
async def start_experiment(experiment_id: str, request: ExperimentStartRequest) -> ResearchExperiment:
    _integration.experiment_registry.mark_experiment_running(
        experiment_id=experiment_id,
        mlflow_run_id=request.mlflow_run_id,
    )

    now_ts = datetime.utcnow()
    try:
        _ch_client.execute(
            """
            ALTER TABLE market_intelligence.research_experiments
            UPDATE status = 'running', mlflow_run_id = %(mlflow)s, started_at = %(started)s
            WHERE experiment_id = %(eid)s
            """,
            {
                "mlflow": request.mlflow_run_id,
                "started": now_ts,
                "eid": experiment_id,
            },
        )
    except Exception as exc:
        logger.exception("Failed to update experiment start state")
        raise HTTPException(status_code=500, detail=f"Failed to update experiment: {exc}") from exc

    experiment = _fetch_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found after update")
    return experiment


@app.post("/api/v1/research/experiments/{experiment_id}/complete", response_model=ResearchExperiment)
async def complete_experiment(experiment_id: str, request: ExperimentCompleteRequest) -> ResearchExperiment:
    _integration.experiment_registry.update_experiment_results(
        experiment_id=experiment_id,
        results=request.results,
        execution_time=request.execution_time,
        status=request.status,
        mlflow_run_id=request.mlflow_run_id,
    )

    now_ts = datetime.utcnow()
    try:
        _ch_client.execute(
            """
            ALTER TABLE market_intelligence.research_experiments
            UPDATE status = %(status)s, completed_at = %(completed)s, results = %(results)s, mlflow_run_id = ifNull(%(mlflow)s, mlflow_run_id)
            WHERE experiment_id = %(eid)s
            """,
            {
                "status": request.status,
                "completed": now_ts,
                "results": json.dumps(request.results, default=_json_default),
                "mlflow": request.mlflow_run_id,
                "eid": experiment_id,
            },
        )
    except Exception as exc:
        logger.exception("Failed to update experiment completion state")
        raise HTTPException(status_code=500, detail=f"Failed to persist experiment completion: {exc}") from exc

    experiment = _fetch_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found after completion")
    return experiment


@app.get("/api/v1/research/experiments", response_model=List[ResearchExperiment])
async def list_experiments(status: Optional[str] = None, model_type: Optional[str] = None, limit: int = 50) -> List[ResearchExperiment]:
    conditions = ["1 = 1"]
    params: Dict[str, Any] = {"limit": limit}
    if status:
        conditions.append("status = %(status)s")
        params["status"] = status
    if model_type:
        conditions.append("model_type = %(model_type)s")
        params["model_type"] = model_type

    query = f"""
        SELECT experiment_id, name, model_type, dataset, parameters, status, mlflow_run_id, started_at, completed_at, results
        FROM market_intelligence.research_experiments
        WHERE {' AND '.join(conditions)}
        ORDER BY started_at DESC
        LIMIT %(limit)s
    """
    rows = _ch_client.execute(query, params)
    experiments: List[ResearchExperiment] = []
    for row in rows:
        experiments.append(
            ResearchExperiment(
                experiment_id=row[0],
                name=row[1],
                model_type=row[2],
                dataset=row[3],
                parameters=_load_json(row[4]) or {},
                status=row[5],
                mlflow_run_id=row[6],
                started_at=row[7],
                completed_at=row[8],
                results=_load_json(row[9]) or {},
            )
        )
    return experiments


@app.get("/api/v1/research/experiments/{experiment_id}", response_model=ResearchExperiment)
async def get_experiment(experiment_id: str) -> ResearchExperiment:
    experiment = _fetch_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    return experiment
