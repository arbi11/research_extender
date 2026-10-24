"""One authenticated, cached recovery attempt per observed job failure.

This route starts only a new mesh job. The caller must observe its successful
completion before submitting one solve with the returned frozen target contract.
The original job, branch and failed checks are retained.
"""
from __future__ import annotations

import asyncio
from contextlib import closing
from copy import deepcopy
import hashlib
import json
import re
import shutil
import time
from typing import Annotated, Literal
import uuid

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .history import PATH_PATTERN, apply_feature_edits, plan_diff
from .plan import GeometryPlan
from . import recovery
from .targets import TargetContract, model_fingerprint


FeaturePath = Annotated[str, Field(pattern=PATH_PATTERN, max_length=250)]


class RecoverRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    instruction: str = Field(default="", max_length=3000)
    scope: Literal["mesh", "last_edit", "custom"] = "mesh"
    allowed_paths: list[FeaturePath] | None = Field(default=None, max_length=64)


def _within(path: str, scope: str) -> bool:
    return path == scope or path.startswith(scope + "/")


def _request_hash(request: RecoverRequest) -> str:
    value = request.model_dump(mode="json")
    if value["allowed_paths"] is not None:
        value["allowed_paths"] = sorted(set(value["allowed_paths"]))
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _tail(path) -> str:
    if not path.is_file() or path.is_symlink():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 6000))
        return stream.read(6000).decode("utf-8", errors="replace")


def _scopes(request: RecoverRequest, revision: dict) -> list[str]:
    if request.scope != "custom" and request.allowed_paths is not None:
        raise HTTPException(422, "Explicit allowed_paths require scope='custom'.")
    if request.scope == "mesh":
        return ["/mesh_mm"] + [f"/regions/{region['id']}/mesh_mm" for region in revision["plan"]["regions"]]
    if request.scope == "last_edit":
        action = revision["action"]
        paths = action.get("allowed_paths", []) if action.get("type") == "patch" else []
        if not paths:
            raise HTTPException(409, "This revision has no previous scoped edit to repair. Select explicit custom paths or mesh settings.")
    else:
        paths = request.allowed_paths or []
        if not paths:
            raise HTTPException(422, "Custom recovery requires at least one independently permitted path.")
    if len(paths) > 64 or any(not isinstance(path, str) or len(path) > 250 or
                            not re.fullmatch(PATH_PATTERN, path) or path.split("/")[-1] == "id" for path in paths):
        raise HTTPException(422, "Recovery requires bounded named-feature paths; feature ids cannot change.")
    return list(dict.fromkeys(paths))


def _context(history, jobstore, job, failure, revision):
    context = history.recovery_context(failure["id"])
    # Preserve the recorded check in storage; enrich only the provider context
    # with corroborating process evidence from this exact failed job.
    context["failure"] = deepcopy(context["failure"])
    context["failure"]["details"] = {**context["failure"].get("details", {}),
        "native_error": job.get("error"), "native_log_tail": _tail(jobstore.root / job["id"] / "solver.log")}
    context["candidates"] = context["candidates"][:8]
    context["history_truncated"] = len(history.lineage(revision["id"])) > 4
    ancestors = history.lineage(revision["id"])[:4]
    context["checkpoint_models"] = []
    context["checkpoints"] = []
    for ancestor in ancestors:
        snapshot = {"id": ancestor["id"], "plan": ancestor["plan"]}
        checkpoint = {"id": ancestor["id"], "plan_hash": ancestor["plan_hash"]}
        context["checkpoint_models"].append(snapshot)
        context["checkpoints"].append(checkpoint)
        if len(json.dumps(context, ensure_ascii=True, allow_nan=False).encode()) > recovery.MAX_CONTEXT_BYTES:
            context["checkpoint_models"].pop()
            context["checkpoints"].pop()
            context["history_truncated"] = True
            break
    if not context["checkpoint_models"]:
        raise HTTPException(422, "The failed model and evidence exceed the recovery input budget. Inspect the retained evidence and select a smaller case.")
    return context


def _claim(jobstore, job_id, failure_id, request_sha):
    """Atomically reserve paid work across requests/processes sharing this DB."""
    with closing(jobstore.connect()) as db, db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT evidence FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Job not found or expired.")
        evidence = json.loads(row[0]) if row[0] else {}
        if evidence.get("failure_id") != failure_id:
            raise HTTPException(409, "The job's recorded failure changed. Reload its evidence.")
        existing = evidence.get("recovery_attempt")
        if existing:
            return existing
        attempt = {"failure_id": failure_id, "request_sha256": request_sha, "status": "proposing"}
        evidence["recovery_attempt"] = attempt
        db.execute("UPDATE jobs SET evidence=? WHERE id=?", (json.dumps(evidence, allow_nan=False), job_id))
        return None


def _cached(attempt: dict, failure_id: str, request_sha: str):
    if attempt.get("failure_id") != failure_id or attempt.get("request_sha256") != request_sha:
        raise HTTPException(409, "A recovery was already attempted for this failure with different instructions or scopes. Inspect its retained result.")
    if attempt["status"] == "completed":
        return JSONResponse(attempt["response"], status_code=attempt.get("response_status", 202))
    if attempt["status"] == "failed":
        error = attempt["error"]
        raise HTTPException(error["status_code"], error["message"])
    if attempt["status"] == "ready_to_queue":
        return None
    raise HTTPException(409, "This failure already has a recovery proposal in progress or interrupted. No duplicate paid request was started.")


def register_routes(app, secured, settings, budget, history, jobstore, replicationstore):
    locks: dict[str, asyncio.Lock] = {}

    def save(job_id, failure_id, request_sha, status, **fields):
        attempt = {"failure_id": failure_id, "request_sha256": request_sha, "status": status, **fields}
        jobstore.annotate(job_id, recovery_attempt=attempt)
        return attempt

    def queue_response(job_id, failure_id, request_sha, response):
        # This state is durable before queueing: a temporarily full queue can be
        # retried without another model call, branch or feature patch.
        with closing(jobstore.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT evidence FROM jobs WHERE id=?", (job_id,)).fetchone()
            evidence = json.loads(row[0]) if row and row[0] else {}
            attempt = evidence.get("recovery_attempt", {})
            if attempt.get("status") == "completed":
                return _cached(attempt, failure_id, request_sha)
            if attempt.get("failure_id") != failure_id or attempt.get("request_sha256") != request_sha:
                raise HTTPException(409, "The recovery reservation changed before the mesh was queued.")
            if attempt.get("status") == "ready_to_queue":
                response = attempt["response"]
            elif attempt.get("status") != "proposing":
                raise HTTPException(409, "This recovery is not ready for a native mesh job.")
            evidence["recovery_attempt"] = {**attempt, "status": "ready_to_queue", "response": response}
            db.execute("UPDATE jobs SET evidence=? WHERE id=?", (json.dumps(evidence, allow_nan=False), job_id))
        budget.use("mesh", 20)
        # Publish the queued mesh and its cached response in ONE transaction.
        # A crash or another server process cannot queue the same repair twice.
        with closing(jobstore.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT evidence FROM jobs WHERE id=?", (job_id,)).fetchone()
            evidence = json.loads(row[0]) if row and row[0] else {}
            attempt = evidence.get("recovery_attempt", {})
            if attempt.get("status") == "completed":
                return _cached(attempt, failure_id, request_sha)
            if (attempt.get("status") != "ready_to_queue" or attempt.get("failure_id") != failure_id or
                    attempt.get("request_sha256") != request_sha):
                raise HTTPException(409, "The recovery reservation changed before native execution.")
            pending = db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if pending >= 5:
                raise HTTPException(429, "The recovery proposal is saved, but the native queue is full. Retry the same request after a job finishes.")
            response = attempt["response"]
            plan = GeometryPlan.model_validate(response["revision"]["plan"])
            mesh_id, now = uuid.uuid4().hex, time.time()
            db.execute("INSERT INTO jobs (id,spec,status,created,updated,result,error,kind,revision_id) VALUES (?,?,?,?,?,?,?,?,?)",
                (mesh_id, plan.model_dump_json(), "queued", now, now, None, None, "mesh", response["revision"]["id"]))
            mesh = dict(db.execute("SELECT * FROM jobs WHERE id=?", (mesh_id,)).fetchone())
            mesh["spec"] = json.loads(mesh["spec"])
            mesh["result"], mesh["evidence"] = None, {}
            response = {**response, "mesh_job": mesh}
            evidence["recovery_attempt"] = {**attempt, "status": "completed", "response": response,
                                             "response_status": 202}
            db.execute("UPDATE jobs SET evidence=? WHERE id=?", (json.dumps(evidence, allow_nan=False), job_id))
        return JSONResponse(response, status_code=202)

    @app.post("/api/jobs/{job_id}/recover", dependencies=secured)
    async def recover_job(job_id: str, request: RecoverRequest):
        if jobstore.get(job_id) is None:
            raise HTTPException(404, "Job not found or expired.")
        lock = locks.setdefault(job_id, asyncio.Lock())
        async with lock:
            job = jobstore.get(job_id)
            if job is None:
                raise HTTPException(404, "Job not found or expired.")
            failure_id = job["evidence"].get("failure_id")
            try:
                failure = history.get_check(failure_id) if isinstance(failure_id, str) else None
            except ValueError:
                raise HTTPException(409, "The recorded failure no longer matches its immutable evidence.") from None
            if (not failure or failure["status"] != "failed" or failure["job_id"] != job_id or
                    failure["revision_id"] != job.get("revision_id") or
                    failure["stage"] not in recovery.STAGE_FUNCTIONS or
                    (job["status"] != "failed" and not (job["status"] == "completed" and failure["stage"] == "target"))):
                raise HTTPException(409, "Recovery requires this job's actual recorded mesh, solver or target failure.")
            try:
                revision = history.get(job["revision_id"])
            except ValueError:
                raise HTTPException(409, "The failed model no longer matches its immutable revision.") from None
            if revision is None or revision["plan"] != job["spec"]:
                raise HTTPException(409, "The failed job does not match its immutable model revision.")
            request_sha = _request_hash(request)
            previous = job["evidence"].get("recovery_attempt")
            if previous:
                cached = _cached(previous, failure_id, request_sha)
                if cached is not None:
                    return cached
                return queue_response(job_id, failure_id, request_sha, previous["response"])
            if not shutil.which(settings.femm_lua):
                raise HTTPException(503, "FEMM 5 is unavailable. Resolve the native execution environment before requesting recovery.")
            if not settings.ai_key or not settings.ai_model:
                raise HTTPException(503, "Recovery proposals need OPENROUTER_API_KEY and AI_MODEL on the server.")
            scopes = _scopes(request, revision)
            try:
                context = _context(history, jobstore, job, failure, revision)
            except ValueError:
                raise HTTPException(409, "The recovery history no longer matches its immutable evidence.") from None
            target_record = None
            if job.get("target_contract_id"):
                try:
                    target_record = replicationstore.get_contract(job["target_contract_id"])
                except ValueError:
                    raise HTTPException(409, "The failed job's frozen target record no longer matches its stored digest.") from None
                if not target_record or target_record["revision_id"] != revision["id"]:
                    raise HTTPException(409, "The failed job's frozen target contract is unavailable or belongs to another revision.")
                try:
                    contract = TargetContract.model_validate(target_record["contract"])
                except ValueError:
                    raise HTTPException(409, "The failed job's numerical target contract is invalid.") from None
                if contract.model_sha256 != model_fingerprint(revision["plan"]):
                    raise HTTPException(409, "The frozen targets do not match the failed model.")
            with closing(jobstore.connect()) as db:
                pending = db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if pending >= 5:
                raise HTTPException(429, "The native queue is full. Wait for a job to finish before requesting a recovery proposal.")
            # A request rejected here has not started a paid recovery attempt.
            budget.use("recovery", 20)
            budget.use("function_diagnostic", 20)
            budget.use("ai", 20)
            claimed = _claim(jobstore, job_id, failure_id, request_sha)
            if claimed:
                cached = _cached(claimed, failure_id, request_sha)
                if cached is not None:
                    return cached
                return queue_response(job_id, failure_id, request_sha, claimed["response"])
            try:
                proposal = await recovery.propose_recovery(context, settings.ai_key, settings.ai_base_url,
                    settings.ai_model, decision_model=settings.decision_model,
                    decision_base_url=settings.decision_base_url,
                    instruction=request.instruction, allowed_paths=scopes)
                proposal_json = proposal.model_dump(mode="json")
                if proposal.questions or proposal.spec is None:
                    response = {"proposal": proposal_json, "revision": None, "mesh_job": None,
                                "target_contract_id": None}
                    save(job_id, failure_id, request_sha, "completed", response=response, response_status=200)
                    return JSONResponse(response, status_code=200)
                if (proposal.failure_id != failure_id or proposal.function_name != recovery.STAGE_FUNCTIONS[failure["stage"]] or
                        proposal.checkpoint_id not in {item["id"] for item in context["checkpoint_models"]}):
                    raise ValueError("The recovery proposal is not bound to this failure and supplied checkpoint evidence.")
                checkpoint = history.get(proposal.checkpoint_id)
                candidate = GeometryPlan.model_validate(checkpoint["plan"])
                if proposal.edits:
                    candidate, _ = apply_feature_edits(candidate, proposal.edits, scopes)
                changes = plan_diff(GeometryPlan.model_validate(revision["plan"]), candidate)
                if not changes or any(not any(_within(change["path"], scope) for scope in scopes) for change in changes):
                    raise ValueError("The proposed checkpoint or edits change fields outside the independently permitted recovery paths.")
                if candidate.model_dump(mode="json") != proposal.spec.model_dump(mode="json"):
                    raise ValueError("The recovery edits do not match their proposed model.")
                branch = "recovery-" + uuid.uuid4().hex[:20]
                recovered = history.branch_recovery(failure_id, proposal.checkpoint_id, branch=branch,
                    diagnosis=proposal.explanation, evidence=[{"source": "artifact",
                        "reference": "job:" + job_id, "detail": "Bounded recovery proposal after recorded failure " + failure_id}])
                if proposal.edits:
                    recovered = history.apply_patch(recovered["id"], proposal.edits, allowed_paths=scopes,
                        branch=branch, reason=proposal.explanation)
                recovered = {**recovered, "branch": branch}
                target_contract_id = None
                if target_record:
                    contract = TargetContract.model_validate(target_record["contract"])
                    rebound = replicationstore.put_contract(recovered["id"], contract.case_id,
                        contract.targets, GeometryPlan.model_validate(recovered["plan"]),
                        evidence_id=target_record.get("evidence_id"))
                    if rebound["contract"]["targets_sha256"] != contract.targets_sha256:
                        raise ValueError("Recovery cannot alter the frozen targets or tolerances.")
                    target_contract_id = rebound["contract_id"]
                response = {"proposal": proposal_json, "revision": recovered, "mesh_job": None,
                            "target_contract_id": target_contract_id}
                return queue_response(job_id, failure_id, request_sha, response)
            except HTTPException as error:
                # Queue pressure retains a durable proposal ready for a native
                # retry. It must never purchase a second proposal.
                pending_attempt = jobstore.get(job_id)["evidence"].get("recovery_attempt", {})
                if pending_attempt.get("status") != "ready_to_queue":
                    save(job_id, failure_id, request_sha, "failed",
                         error={"status_code": error.status_code, "message": str(error.detail)})
                raise
            except recovery.RecoveryError as error:
                save(job_id, failure_id, request_sha, "failed", error={"status_code": error.status_code,
                                                                       "message": error.message})
                raise HTTPException(error.status_code, error.message) from None
            except ValueError as error:
                message = str(error)
                save(job_id, failure_id, request_sha, "failed", error={"status_code": 409, "message": message})
                raise HTTPException(409, message) from None
            except Exception:
                message = "Recovery could not be completed. The original failure is retained and no successful check was recorded."
                save(job_id, failure_id, request_sha, "failed", error={"status_code": 500, "message": message})
                raise HTTPException(500, message) from None
