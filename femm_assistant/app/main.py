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
from .plan import GeometryPlan, wire_example
from .examples import axisymmetric_example
from . import plan_generate, planner, decisions, vision, cases
from .history import ModelRevisionStore
from . import replication, recovery_api, provenance, paper_constraints
from . import assembly_api, paper_assets, source_review
from .targets import TargetContract, model_fingerprint


@dataclass
class Settings:
    data_dir: Path
    token: str = ""
    ai_key: str = ""
    ai_model: str = "google/gemini-3.1-flash-lite"
    ai_base_url: str = "https://openrouter.ai/api/v1"
    decision_model: str = "typesafe/jev-1.14"
    decision_base_url: str = "https://openrouter.ai/api/alpha"
    decision_threshold: float = 0.90
    vision_model: str = "google/gemini-3.1-flash-lite"
    femm_lua: str = "femm-lua"
    timeout: int = 180
    retention_hours: int = 24
    start_worker: bool = True
    ai_timeout: float | None = None
    hierarchical_ai: bool = True

    @classmethod
    def from_env(cls):
        return cls(data_dir=Path(os.getenv("DATA_DIR", "./data")), token=os.getenv("ASSISTANT_TOKEN", ""),
                   ai_key=os.getenv("OPENROUTER_API_KEY", ""), ai_model=os.getenv("AI_MODEL") or "google/gemini-3.1-flash-lite",
                   ai_base_url=os.getenv("AI_BASE_URL", "https://openrouter.ai/api/v1"),
                   decision_model=os.getenv("DECISION_MODEL") or "typesafe/jev-1.14",
                   decision_base_url=os.getenv("DECISION_BASE_URL") or "https://openrouter.ai/api/alpha",
                   decision_threshold=float(os.getenv("DECISION_THRESHOLD") or "0.90"),
                   vision_model=os.getenv("VISION_MODEL", "google/gemini-3.1-flash-lite"),
                   femm_lua=os.getenv("FEMM_LUA", "femm-lua"), timeout=int(os.getenv("SOLVE_TIMEOUT_SECONDS", "180")),
                   retention_hours=int(os.getenv("RETENTION_HOURS", "24")),
                   ai_timeout=float(os.environ["AI_REQUEST_TIMEOUT_SECONDS"]) if os.getenv("AI_REQUEST_TIMEOUT_SECONDS") else None)


class InterpretRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=5, max_length=3000)
    current: CCoreSpec = Field(default_factory=CCoreSpec)


class PlanInterpretRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=5, max_length=3000)
    current: GeometryPlan | None = None
    allowed_paths: list[str] | None = Field(default=None, max_length=64)
    evidence_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    case_id: str = Field(default="baseline", pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,79}$")
    preserve_paper_excitation: bool = False
    hierarchical: bool = True
    model: str | None = Field(default=None, min_length=1, max_length=160)


class PlanSolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: GeometryPlan
    mesh_job_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    revision_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    target_contract_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class VisualReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=3000)
    spec: GeometryPlan
    mesh_job_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


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
    history = ModelRevisionStore(settings.data_dir)
    paper_store = replication.ReplicationStore(settings.data_dir)
    worker = SolverWorker(store, settings.femm_lua, settings.timeout, settings.retention_hours,
                          revision_store=history, replication_store=paper_store)
    budget = Budget()

    @asynccontextmanager
    async def lifespan(app):
        if settings.start_worker:
            worker.start()
        try:
            yield
        finally:
            if settings.start_worker:
                worker.stop()

    app = FastAPI(title="FEMM Assistant", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.worker, app.state.history = store, worker, history
    app.state.replication = paper_store

    async def authenticate(authorization: str = Header(default="")):
        # An empty token supports the owner's explicitly configured local trial.
        if settings.token and not hmac.compare_digest(authorization, "Bearer " + settings.token):
            raise HTTPException(401, "Enter the assistant access token.", headers={"WWW-Authenticate": "Bearer"})

    secured = [Depends(authenticate)]
    assembly_store = assembly_api.register_routes(app, secured, settings, budget, paper_store)

    @app.middleware("http")
    async def headers(request, call_next):
        # Reject oversized bodies even when no Content-Length was supplied.
        if request.method == "POST":
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                limit = (8 * 1024 * 1024 if request.url.path == "/api/papers" else
                         8 * 1024 * 1024 if request.url.path.startswith(("/api/plan/", "/api/revisions", "/api/papers/", "/api/targets", "/api/assemblies", "/api/source-review")) else 16000)
                if len(body) > limit:
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
                "assumptions": ASSUMPTIONS, "retention_hours": settings.retention_hours,
                "plan_example": wire_example().model_dump(), "plan_schema": GeometryPlan.model_json_schema(),
                "axisymmetric_example": axisymmetric_example().model_dump(),
                "drafting_model": settings.ai_model, "decision_model": settings.decision_model,
                "decision_threshold": None, "decision_role": "function_routing_and_error_diagnostics", "mesh_required": True,
                "argument_diagnosis_available": bool(settings.ai_key and settings.decision_model),
                "visual_review_available": bool(settings.ai_key and settings.vision_model),
                "visual_review_model": settings.vision_model,
                "revision_history_available": True,
                "paper_intake_available": True, "target_comparison_available": True,
                "hierarchical_geometry_available": True, "strict_response_schema": True,
                "source_assets_available": bool(shutil.which("pdftoppm")),
                "source_comparison_available": bool(settings.ai_key and settings.vision_model),
                "ai_output_token_limit": None,
                "suggested_models": ["~z-ai/glm-latest", "anthropic/claude-haiku-5.5", "deepseek/deepseek-v4-pro-0813"],
                "controlled_recovery_available": bool(settings.ai_key and settings.ai_model and settings.decision_model),
                "plan_capabilities": ["Planar and axisymmetric DC magnetostatics", "Circles, rectangles and simple polygons",
                                      "Nested or separate closed regions with linear materials",
                                      "DC circuits, an outer zero-potential boundary, field probes and optional force"]}

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
        except httpx.HTTPStatusError as exc:
            messages = {401: "The AI provider rejected the API key. Check the server's OpenRouter settings.",
                        402: "The AI account has insufficient credits. Check its balance and key spending limit.",
                        429: "The AI provider is rate limiting requests. Try again shortly."}
            code = exc.response.status_code
            raise HTTPException(429 if code == 429 else 502,
                                messages.get(code, "The AI provider could not complete this request. Try again shortly.")) from None
        except (httpx.HTTPError, ValidationError, KeyError, ValueError, IndexError):
            raise HTTPException(502, "The AI provider did not return a valid C-core specification. Refine the request or edit parameters directly.")

    @app.post("/api/plan/preview", dependencies=secured)
    async def plan_preview(spec: GeometryPlan):
        return Response(plan_generate.preview_svg(spec), media_type="image/svg+xml")

    @app.post("/api/plan/review", dependencies=secured)
    async def visual_review(request: VisualReviewRequest):
        if not settings.ai_key or not settings.vision_model:
            raise HTTPException(503, "Configure an API key and VISION_MODEL to enable optional visual review.")
        mesh_svg = None
        if request.mesh_job_id is not None:
            mesh_job = store.get(request.mesh_job_id)
            if (not mesh_job or mesh_job["kind"] != "mesh" or mesh_job["status"] != "completed"
                    or not mesh_job["result"] or mesh_job["result"].get("status") != "passed"
                    or mesh_job["spec"] != request.spec.model_dump(mode="json")):
                raise HTTPException(409, "Visual mesh review requires a passed mesh check for this exact unchanged model.")
            path = store.root / request.mesh_job_id / "mesh.svg"
            if not path.is_file():
                raise HTTPException(404, "The mesh image is unavailable. Review geometry only or check the mesh again.")
            if path.stat().st_size > vision.MAX_SVG_BYTES:
                raise HTTPException(422, "The mesh image is too detailed for visual review. Review geometry only.")
            mesh_svg = path.read_text(encoding="utf-8")
        budget.use("visual_review", 20)
        try:
            async with httpx.AsyncClient() as client:
                return await vision.review(client, settings.ai_key, settings.ai_base_url, settings.vision_model,
                                           request.prompt, request.spec, mesh_svg=mesh_svg)
        except vision.VisionError as exc:
            raise HTTPException(exc.status_code, exc.message) from None

    @app.post("/api/plan/export", dependencies=secured)
    async def plan_export(spec: GeometryPlan):
        return Response(plan_generate.source_zip(spec), media_type="application/zip",
                        headers={"Content-Disposition": 'attachment; filename="femm-geometry-source.zip"'})

    @app.post("/api/plan/interpret", dependencies=secured)
    async def plan_interpret(request: PlanInterpretRequest):
        if not settings.ai_key or not settings.ai_model:
            raise HTTPException(503, "Set OPENROUTER_API_KEY and AI_MODEL on the server to enable AI geometry requests. You can load or edit a model plan directly.")
        budget.use("ai", 20)
        try:
            context = (paper_store.planning_context(request.evidence_id, case_id=request.case_id,
                        instructions=request.prompt) if request.evidence_id else None)
            if context is not None:
                context["preserve_paper_excitation"] = request.preserve_paper_excitation
            proposal = await planner.interpret_plan(request.prompt, request.current, settings.ai_key,
                                                settings.ai_base_url, request.model or settings.ai_model,
                                                decision_model=settings.decision_model,
                                                decision_base_url=settings.decision_base_url,
                                                decision_threshold=settings.decision_threshold,
                                                allowed_paths=request.allowed_paths, evidence_context=context,
                                                hierarchical=request.hierarchical)
            result = proposal.model_dump(mode="json")
            if proposal.assembly is not None:
                record = assembly_store.put(proposal.assembly, instructions=request.prompt,
                    inventory=proposal.inventory, provider_usage=proposal.provider_usage,
                    provider_metadata=proposal.provider_metadata,
                    evidence_id=request.evidence_id, case_id=request.case_id)
                result["assembly_record"] = record
                result["assembly_id"] = record["assembly_id"]
            excitation_check = (paper_constraints.validate_excitation(proposal.spec, context)
                                if request.evidence_id and proposal.spec is not None else None)
            if excitation_check is not None:
                result["paper_excitation_check"] = excitation_check
            if request.evidence_id and proposal.spec is not None and not proposal.questions:
                binding = provenance.put_planning(paper_store, proposal.spec, request.evidence_id,
                                                   request.case_id, request.prompt,
                                                   preserve_paper_excitation=request.preserve_paper_excitation,
                                                   excitation_check=excitation_check)
                result["planning_id"] = binding["planning_id"]
            return result
        except planner.PlannerError as exc:
            raise HTTPException(exc.status_code, exc.message) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post("/api/plan/mesh", dependencies=secured, status_code=202)
    async def plan_mesh(spec: GeometryPlan, x_model_revision: str | None = Header(default=None)):
        if not shutil.which(settings.femm_lua):
            raise HTTPException(503, "FEMM 5 is not installed on this server. You can still preview and download the source bundle.")
        budget.use("mesh", 20)
        revision = cases.require_revision(history, x_model_revision, spec)
        try:
            return store.create(spec, kind="mesh", revision_id=revision["id"])
        except BusyError as exc:
            raise HTTPException(429, str(exc))

    @app.post("/api/plan/jobs", dependencies=secured, status_code=202)
    async def plan_submit(request: PlanSolveRequest):
        if not shutil.which(settings.femm_lua):
            raise HTTPException(503, "FEMM 5 is not installed on this server. You can still preview and download the source bundle.")
        mesh_job = store.get(request.mesh_job_id)
        if not mesh_job or mesh_job["kind"] != "mesh" or mesh_job["status"] != "completed" or not mesh_job["result"] or mesh_job["result"].get("status") != "passed":
            raise HTTPException(409, "Run a successful mesh check before solving this model.")
        if mesh_job["spec"] != request.spec.model_dump(mode="json"):
            raise HTTPException(409, "The model changed after its mesh check. Check the mesh again before solving.")
        revision_id = request.revision_id or mesh_job.get("revision_id")
        revision = cases.require_revision(history, revision_id, request.spec)
        if request.revision_id and mesh_job.get("revision_id") != request.revision_id:
            raise HTTPException(409, "This mesh belongs to a different model revision. Check the current revision's mesh.")
        if request.target_contract_id:
            record = paper_store.get_contract(request.target_contract_id)
            if (record is None or record["revision_id"] != revision["id"] or
                    TargetContract.model_validate(record["contract"]).model_sha256 != model_fingerprint(request.spec)):
                raise HTTPException(409, "Freeze targets for this exact model revision before solving.")
        budget.use("solve", 20)
        try:
            return store.create(request.spec, revision_id=revision["id"], target_contract_id=request.target_contract_id)
        except BusyError as exc:
            raise HTTPException(429, str(exc))

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

    @app.post("/api/jobs/{job_id}/diagnose", dependencies=secured)
    async def diagnose_job(job_id: str):
        failed = find_job(job_id)
        if failed["status"] != "failed":
            raise HTTPException(409, "Argument diagnostics are available only after a native function fails.")
        cached = (failed["result"] or {}).get("function_diagnosis")
        if cached:
            return cached
        if not settings.ai_key or not settings.decision_model:
            raise HTTPException(503, "Configure the API key and decision model to diagnose failed function calls.")
        function_name = ("mesh_plan" if failed["kind"] == "mesh" else
                         "solve_plan" if failed["spec"].get("kind") == "magnetostatic_plan" else "solve_ccore")
        error = failed["error"] or "The native function failed."
        log = store.root / job_id / "solver.log"
        log_tail = ""
        if log.is_file():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 6000))
                log_tail = stream.read(6000).decode("utf-8", errors="replace")
        budget.use("function_diagnostic", 20)
        try:
            async with httpx.AsyncClient() as client:
                diagnostic = await decisions.diagnose_call(client, settings.ai_key, settings.decision_base_url,
                    settings.decision_model, function_name, {"spec": failed["spec"]},
                    [{"message": error, "native_log_tail": log_tail}],
                    "Diagnose this failed native call against its supplied model arguments.")
        except decisions.DecisionError as exc:
            raise HTTPException(exc.status_code, exc.message) from None
        result = {"function_name": function_name, "arguments_match": diagnostic.arguments_match,
                  "trace": [trace.model_dump(mode="json") for trace in diagnostic.trace],
                  "advisory": True, "function_status": "failed", "original_error": error}
        # A diagnostic annotates the failure. It cannot approve a mesh or solve.
        store.update(job_id, "failed", result={**(failed["result"] or {}), "function_diagnosis": result}, error=failed["error"])
        return result

    @app.get("/api/jobs/{job_id}/{artifact}", dependencies=secured)
    async def download(job_id: str, artifact: str):
        job = find_job(job_id)
        if artifact not in {"download.zip", "field.svg", "mesh.svg", "solver.log"}:
            raise HTTPException(404, "Artifact not found.")
        if artifact != "solver.log" and job["status"] != "completed":
            raise HTTPException(409, "Simulation has not completed successfully.")
        path = store.root / job_id / artifact
        if not path.is_file():
            raise HTTPException(404, "Artifact not available.")
        media = {"download.zip": "application/zip", "field.svg": "image/svg+xml", "mesh.svg": "image/svg+xml", "solver.log": "text/plain"}[artifact]
        return FileResponse(path, media_type=media, filename=artifact if artifact not in {"field.svg", "mesh.svg"} else None)

    cases.register_routes(app, secured, history, paper_store)
    replication.register_routes(app, secured, settings, budget, history, store)
    paper_assets.register_routes(app, secured, paper_store)
    source_review.register_routes(app, secured, settings, budget, app.state.paper_assets, store,
                                 assembly_resolver=assembly_store.get)
    recovery_api.register_routes(app, secured, settings, budget, history, store, paper_store)
    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="web")
    return app
