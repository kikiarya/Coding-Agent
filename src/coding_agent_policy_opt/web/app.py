from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse, StreamingResponse

from .models import HealthResponse, RunRequest, RunResponse
from .service import TERMINAL_STATES, RunManager


def create_app(manager: RunManager | None = None) -> FastAPI:
    manager = manager or RunManager.from_environment()
    app = FastAPI(
        title="CAPO Coding Agent Control Plane",
        version="0.1.0",
        description="Local control plane for auditable repository-level Coding Agent runs.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        del request
        # Pydantic's default 422 payload includes the rejected raw input. Do
        # not reflect credentials or other sensitive values back to a client.
        detail = [
            {key: value for key, value in error.items() if key not in {"input", "ctx"}}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": detail})

    app.state.run_manager = manager

    @app.get("/api/health", response_model=HealthResponse)
    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        active = manager._active_run()
        return HealthResponse(status="ok", active_run=active.run_id if active else None)

    @app.post("/api/runs", response_model=RunResponse, status_code=202)
    @app.post("/runs", response_model=RunResponse, status_code=202)
    def start_run(request: RunRequest) -> RunResponse:
        try:
            run = manager.start(request)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409 if isinstance(exc, RuntimeError) else 400, detail=str(exc)) from exc
        return RunResponse(**run.public())

    @app.get("/api/runs/{run_id}", response_model=RunResponse)
    @app.get("/runs/{run_id}", response_model=RunResponse)
    def get_run(run_id: str) -> RunResponse:
        try:
            return RunResponse(**manager.get(run_id).public())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

    @app.post("/api/runs/{run_id}/cancel", response_model=RunResponse)
    @app.post("/runs/{run_id}/cancel", response_model=RunResponse)
    def cancel_run(run_id: str) -> RunResponse:
        try:
            return RunResponse(**manager.cancel(run_id).public())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

    @app.post("/api/runs/{run_id}/resume", response_model=RunResponse, status_code=202)
    @app.post("/runs/{run_id}/resume", response_model=RunResponse, status_code=202)
    def resume_run(run_id: str) -> RunResponse:
        try:
            return RunResponse(**manager.resume(run_id).public())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/trace")
    @app.get("/runs/{run_id}/trace")
    def get_trace(run_id: str) -> dict:
        try:
            return manager.trace(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

    @app.get("/api/runs/{run_id}/artifacts")
    @app.get("/runs/{run_id}/artifacts")
    def get_artifacts(run_id: str) -> dict:
        try:
            return manager.artifacts(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

    @app.get("/api/runs/{run_id}/events")
    @app.get("/runs/{run_id}/events")
    def stream_events(
        run_id: str,
        after: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        try:
            run = manager.get(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

        def generate() -> Iterator[str]:
            cursor = after
            while True:
                events = run.bus.wait_for(cursor, timeout=10.0)
                if events:
                    for event in events:
                        cursor = int(event["seq"])
                        yield f"id: {cursor}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                    if run.status in TERMINAL_STATES and cursor >= run.bus.latest_seq():
                        break
                elif run.status in TERMINAL_STATES and cursor >= run.bus.latest_seq():
                    break
                else:
                    yield ": heartbeat\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
