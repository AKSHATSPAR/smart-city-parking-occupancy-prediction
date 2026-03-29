from __future__ import annotations

import subprocess
import sys
import uuid
from pathlib import Path
from time import perf_counter

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from .live_ops import LiveOpsStore
from .runtime import InMemoryRateLimiter, load_runtime_settings
from .service import (
    DemoPlaybookRequest,
    DemoResetRequest,
    DemoScenarioRequest,
    IngestObservationRequest,
    WhatIfRequest,
    get_service,
    reset_service_cache,
)


ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_runtime_settings()
RATE_LIMITER = InMemoryRateLimiter(limit=SETTINGS.rate_limit_per_minute)
API_LOG_STORE = LiveOpsStore()


app = FastAPI(
    title="Smart Parking Production API",
    version="1.0.0",
    description="Operational API for parking forecasts, recommendations, anomalies, and monitoring.",
)


def _is_exempt_path(path: str) -> bool:
    return path in {"/health", "/ready", "/openapi.json"} or path.startswith("/docs") or path.startswith("/redoc")


def _client_identity(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client is not None:
        return request.client.host
    return "unknown"


def _log_request(
    request_id: str,
    request: Request,
    status_code: int,
    duration_ms: float,
    metadata: dict | None = None,
) -> None:
    API_LOG_STORE.append_api_request_log(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=status_code,
        client_host=_client_identity(request),
        duration_ms=duration_ms,
        metadata=metadata,
    )


@app.middleware("http")
async def runtime_guard(request: Request, call_next):
    request_id = str(uuid.uuid4())
    started_at = perf_counter()
    rate_state = {"allowed": True, "remaining": SETTINGS.rate_limit_per_minute, "retry_after": 0}

    if not _is_exempt_path(request.url.path):
        if SETTINGS.api_key and request.headers.get("x-api-key") != SETTINGS.api_key:
            duration_ms = (perf_counter() - started_at) * 1000
            response = JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
            response.headers["X-API-Key-Required"] = "true"
            _log_request(
                request_id=request_id,
                request=request,
                status_code=401,
                duration_ms=duration_ms,
                metadata={"reason": "api_key_required"},
            )
            return response

        rate_state = RATE_LIMITER.check(_client_identity(request))
        if not rate_state["allowed"]:
            duration_ms = (perf_counter() - started_at) * 1000
            response = JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
            response.headers["X-RateLimit-Limit"] = str(SETTINGS.rate_limit_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(rate_state["remaining"])
            response.headers["Retry-After"] = str(rate_state["retry_after"])
            _log_request(
                request_id=request_id,
                request=request,
                status_code=429,
                duration_ms=duration_ms,
                metadata={"reason": "rate_limit_exceeded"},
            )
            return response

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as exc:
        duration_ms = (perf_counter() - started_at) * 1000
        _log_request(
            request_id=request_id,
            request=request,
            status_code=500,
            duration_ms=duration_ms,
            metadata={"error": type(exc).__name__},
        )
        raise

    duration_ms = (perf_counter() - started_at) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    response.headers["X-RateLimit-Limit"] = str(SETTINGS.rate_limit_per_minute)
    response.headers["X-RateLimit-Remaining"] = str(rate_state["remaining"])
    if SETTINGS.api_key:
        response.headers["X-API-Key-Required"] = "true"
    _log_request(
        request_id=request_id,
        request=request,
        status_code=status_code,
        duration_ms=duration_ms,
    )
    return response


@app.get("/health")
def health():
    return get_service().health()


@app.get("/ready")
def ready():
    return get_service().ready()


@app.get("/systems")
def systems():
    return {"systems": get_service().systems()}


@app.get("/forecast/latest")
def latest_forecast(limit: int = Query(default=20, ge=1, le=100)):
    return {"items": get_service().latest_forecasts(limit=limit)}


@app.post("/live/ingest")
def ingest_observation(request: IngestObservationRequest):
    try:
        return get_service().ingest_observation(request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/live/state")
def live_state():
    return get_service().live_state()


@app.get("/demo/scenarios")
def demo_scenarios():
    return {"items": get_service().demo_scenarios()}


@app.get("/demo/playbooks")
def demo_playbooks():
    return {"items": get_service().demo_playbooks()}


@app.post("/demo/reset")
def demo_reset(request: DemoResetRequest | None = None):
    payload = request or DemoResetRequest()
    return get_service().reset_live_state(clear_jobs=payload.clear_jobs)


@app.post("/demo/run")
def demo_run(request: DemoScenarioRequest):
    try:
        return get_service().run_demo_scenario(
            scenario_name=request.scenario_name,
            steps=request.steps,
            reset_first=request.reset_first,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/demo/playbook")
def demo_playbook(request: DemoPlaybookRequest):
    try:
        return get_service().run_demo_playbook(
            playbook_name=request.playbook_name,
            reset_first=request.reset_first,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/forecast/what-if")
def what_if(request: WhatIfRequest):
    try:
        return get_service().what_if(request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/recommendations")
def recommendations(objective: str = "Balanced", limit: int = Query(default=10, ge=1, le=100)):
    return {"items": get_service().recommendations(objective=objective, limit=limit)}


@app.get("/anomalies")
def anomalies(limit: int = Query(default=20, ge=1, le=100)):
    return {"items": get_service().anomalies(limit=limit)}


@app.get("/metrics")
def metrics():
    return get_service().metrics()


@app.get("/monitoring")
def monitoring():
    return get_service().monitoring()


@app.get("/registry")
def registry():
    return get_service().registry_payload()


def _run_retrain_job(job_id: str) -> None:
    service = get_service()
    service.store.update_retrain_job(job_id, "running", "Retraining pipeline started")
    service.store.append_audit_event("retrain_job_running", {"job_id": job_id}, actor="api")
    process = subprocess.run(
        [sys.executable, "scripts/run_pipeline.py"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if process.returncode == 0:
        service.store.update_retrain_job(job_id, "completed", "Pipeline retrained successfully")
        service.store.append_audit_event("retrain_job_completed", {"job_id": job_id}, actor="api")
        reset_service_cache()
    else:
        error_message = (process.stderr or process.stdout or "Unknown retrain failure")[:1000]
        service.store.update_retrain_job(job_id, "failed", error_message)
        service.store.append_audit_event(
            "retrain_job_failed",
            {"job_id": job_id, "message": error_message},
            actor="api",
        )


@app.post("/ops/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    service = get_service()
    job_id = service.store.create_retrain_job()
    service.store.append_audit_event("retrain_job_created", {"job_id": job_id}, actor="api")
    background_tasks.add_task(_run_retrain_job, job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/ops/alerts")
def ops_alerts():
    return {"items": get_service().ops_alerts()}


@app.get("/ops/summary")
def ops_summary():
    return get_service().ops_summary()


@app.get("/ops/activity")
def ops_activity(limit: int = Query(default=20, ge=1, le=100)):
    return {"items": get_service().recent_activity(limit=limit)}


@app.get("/ops/retrain/{job_id}")
def retrain_status(job_id: str):
    job = get_service().store.retrain_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Retrain job not found")
    return job
