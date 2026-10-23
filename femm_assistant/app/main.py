from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hmac
import os
from pathlib import Path
import shutil
import threading
import time

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from . import ai
from .generate import preview_svg, source_zip
from .jobs import BusyError, JobStore, SolverWorker
from .model import ASSUMPTIONS, CCoreSpec


@dataclass
class Settings:
    data_dir: Path
    token: str = ""
    ai_key: str = ""
    ai_model: str = ""
    ai_base_url: str = "https://openrouter.ai/api/v1"
    femm_lua: str = "femm-lua"
    timeout: int = 180
    retention_hours: int = 24
    start_worker: bool = True

    @classmethod
    def from_env(cls):
        return cls(data_dir=Path(os.getenv("DATA_DIR", "./data")), token=os.getenv("ASSISTANT_TOKEN", ""),
                   ai_key=os.getenv("OPENROUTER_API_KEY", ""), ai_model=os.getenv("AI_MODEL", ""),
                   ai_base_url=os.getenv("AI_BASE_URL", "https://openrouter.ai/api/v1"),
                   femm_lua=os.getenv("FEMM_LUA", "femm-lua"), timeout=int(os.getenv("SOLVE_TIMEOUT_SECONDS", "180")),
                   retention_hours=int(os.getenv("RETENTION_HOURS", "24")))


class InterpretRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=5, max_length=3000)
    current: CCoreSpec = Field(default_factory=CCoreSpec)


class Budget:
    """One private pilot has one shared hourly budget, not a paid service tier."""
    def __init__(self):
        self.requests = {}
        self.lock = threading.Lock()

    def use(self, kind, limit):
        with self.lock:
            now = time.monotonic()
            events = self.requests.setdefault(kind, deque())
            while events and events[0] < now - 3600:
                events.popleft()
            if len(events) >= limit:
                raise HTTPException(429, "Hourly pilot limit reached. Try again later.")
            events.append(now)


def create_app(settings: Settings | None = None):
    settings = settings or Settings.from_env()
    store = JobStore(settings.data_dir)
    worker = SolverWorker(store, settings.femm_lua, settings.timeout, settings.retention_hours)
    budget = Budget()

    @asynccontextmanager
    async def lifespan(app):
        if settings.start_worker:
            worker.start()
        yield
        if settings.start_worker:
            worker.stop()

    app = FastAPI(title="FEMM Assistant", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.worker = store, worker

    async def authenticate(authorization: str = Header(default="")):
        # Empty token is explicitly for localhost development; Compose requires one.
        if settings.token and not hmac.compare_digest(authorization, "Bearer " + settings.token):
            raise HTTPException(401, "Enter the assistant access token.", headers={"WWW-Authenticate": "Bearer"})

    secured = [Depends(authenticate)]

    @app.middleware("http")
    async def headers(request, call_next):
        # Reject oversized bodies even when no Content-Length was supplied.
        if request.method == "POST":
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 16000:
                    return Response("Request is too large.", status_code=413)
            request._body = bytes(body)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/api/config")
    async def config():
        return {"auth_required": bool(settings.token), "ai_available": bool(settings.ai_key and settings.ai_model),
                "solver_detected": bool(shutil.which(settings.femm_lua)), "defaults": CCoreSpec().model_dump(),
                "assumptions": ASSUMPTIONS, "retention_hours": settings.retention_hours}

    @app.post("/api/preview", dependencies=secured)
    async def preview(spec: CCoreSpec):
        return Response(preview_svg(spec), media_type="image/svg+xml")

    @app.post("/api/export", dependencies=secured)
    async def export(spec: CCoreSpec):
        return Response(source_zip(spec), media_type="application/zip",
                                 headers={"Content-Disposition": 'attachment; filename="femm-c-core-source.zip"'})

    @app.post("/api/interpret", dependencies=secured)
    async def interpret(request: InterpretRequest):
        if not settings.ai_key or not settings.ai_model:
            raise HTTPException(503, "Natural-language configuration needs OPENROUTER_API_KEY and AI_MODEL on the server. You can edit parameters directly.")
        budget.use("ai", 20)
        try:
            return await ai.interpret(request.prompt, request.current, settings.ai_key, settings.ai_base_url, settings.ai_model)
        except httpx.TimeoutException:
            raise HTTPException(504, "The AI request timed out. Your model parameters have not changed.")
        except (httpx.HTTPError, ValidationError, KeyError, ValueError, IndexError):
            raise HTTPException(502, "The AI provider did not return a valid C-core specification. Refine the request or edit parameters directly.")

    @app.post("/api/jobs", dependencies=secured, status_code=202)
    async def submit(spec: CCoreSpec):
        if not shutil.which(settings.femm_lua):
            raise HTTPException(503, "FEMM 5 is not installed on this server. You can still preview and download the source bundle.")
        budget.use("solve", 20)
        try:
            return store.create(spec)
        except BusyError as exc:
            raise HTTPException(429, str(exc))

    def find_job(job_id):
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found or expired.")
        return job

    @app.get("/api/jobs/{job_id}", dependencies=secured)
    async def job(job_id: str):
        return find_job(job_id)

    @app.get("/api/jobs/{job_id}/{artifact}", dependencies=secured)
    async def download(job_id: str, artifact: str):
        job = find_job(job_id)
        if artifact not in {"download.zip", "field.svg", "solver.log"}:
            raise HTTPException(404, "Artifact not found.")
        if artifact != "solver.log" and job["status"] != "completed":
            raise HTTPException(409, "Simulation has not completed successfully.")
        path = store.root / job_id / artifact
        if not path.is_file():
            raise HTTPException(404, "Artifact not available.")
        media = {"download.zip": "application/zip", "field.svg": "image/svg+xml", "solver.log": "text/plain"}[artifact]
        return FileResponse(path, media_type=media, filename=artifact if artifact != "field.svg" else None)

    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="web")
    return app
