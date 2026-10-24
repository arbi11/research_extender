"""Authenticated model revision routes; native checks are recorded by the worker.

Clients can author models and scoped edits, but cannot submit a successful check.
The journal stores model state separately from expiring solver artifacts.
"""
from __future__ import annotations

import uuid
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .history import (EvidenceReference, FeatureEdit, HistoryConflict,
                      HistoryIntegrityError, ModelRevisionStore, plan_diff, apply_feature_edits)
from .plan import GeometryPlan
from . import provenance


class RevisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: GeometryPlan
    parent_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    branch: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    evidence: list[EvidenceReference] = Field(default_factory=list, max_length=128)
    edits: list[FeatureEdit] | None = Field(default=None, max_length=64)
    allowed_paths: list[str] | None = Field(default=None, max_length=64)
    reason: str = Field(default="", max_length=4000)
    planning_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    evidence_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    case_id: str = Field(default="baseline", pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,79}$")
    preserve_paper_excitation: bool = False


class PatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    edits: list[FeatureEdit] = Field(min_length=1, max_length=64)
    allowed_paths: list[str] = Field(min_length=1, max_length=64)
    branch: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    reason: str = Field(default="", max_length=4000)
    evidence: list[EvidenceReference] = Field(default_factory=list, max_length=128)


class RecoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checkpoint_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    diagnosis: str = Field(min_length=1, max_length=8000)


def record_revision(history: ModelRevisionStore, spec: GeometryPlan, *, parent_id=None,
                    branch=None, evidence=(), edits=None, allowed_paths=None, reason=""):
    """Record manual full-model edits or apply AI's explicit typed patch."""
    if parent_id is None:
        branch = branch or "case-" + uuid.uuid4().hex[:20]
        return {**history.create(spec, branch=branch, evidence=evidence), "branch": branch}
    if not branch:
        raise ValueError("An existing model revision requires its branch name.")
    parent = history.get(parent_id)
    if parent is None:
        raise ValueError("The parent model revision is unavailable.")
    changes = plan_diff(GeometryPlan.model_validate(parent["plan"]), spec)
    if not changes:
        if (head := history.head(branch)) is None or head["id"] != parent_id:
            raise HistoryConflict("This branch has changed. Reload its current revision.")
        return {**parent, "branch": branch}
    # Manual model editing explicitly supplies a complete new spec. AI edits
    # instead supply their concrete operations, and cannot replace collections.
    if edits is None:
        edits = [{"op": change.get("op", "replace"), "path": change["path"],
                  **({"value": change["after"]} if change.get("op") != "remove" else {})}
                 for change in changes]
        allowed_paths = [change["path"] for change in changes]
    candidate, _ = apply_feature_edits(GeometryPlan.model_validate(parent["plan"]), edits, allowed_paths)
    if candidate.model_dump(mode="json") != spec.model_dump(mode="json"):
        # Both plans should be generated from the same operations. A discrepancy
        # is an integration error, never permission to silently change a model.
        raise ValueError("The supplied patch differs from its proposed model.")
    revision = history.apply_patch(parent_id, edits, allowed_paths=allowed_paths,
                                   branch=branch, evidence=evidence, reason=reason)
    return {**revision, "branch": branch}


def require_revision(history: ModelRevisionStore, revision_id, spec: GeometryPlan):
    if revision_id is None:
        return record_revision(history, spec)
    revision = history.get(revision_id)
    if revision is None or revision["plan"] != spec.model_dump(mode="json"):
        raise HTTPException(409, "The model does not match its recorded revision. Validate and record it again.")
    return revision


def register_routes(app, secured, history: ModelRevisionStore, paper_store=None):
    @app.post("/api/revisions", dependencies=secured, status_code=201)
    async def create_revision(request: RevisionRequest):
        try:
            evidence = list(request.evidence)
            if request.planning_id is not None:
                if paper_store is None or request.evidence_id is None:
                    raise ValueError("Paper model recording requires its evidence and planning ids.")
                evidence.append(provenance.binding_evidence(paper_store, request.spec, request.planning_id,
                                                           request.evidence_id, request.case_id,
                                                           preserve_paper_excitation=request.preserve_paper_excitation))
            elif request.evidence_id is not None:
                raise ValueError("A paper evidence id requires the exact generated model's planning id.")
            provenance.validate_reserved_references(paper_store, request.spec, evidence)
            parent_id, branch = request.parent_id, request.branch
            if request.planning_id and parent_id:
                parent = history.get(parent_id)
                if parent and parent["plan"] == request.spec.model_dump(mode="json"):
                    old = provenance.revision_binding(paper_store, parent)
                    if old.get("planning_id") != request.planning_id:
                        # A different source binding is a new case even when its
                        # numerical model happens to be identical.
                        parent_id, branch = None, None
            revision = record_revision(history, request.spec, parent_id=parent_id,
                                   branch=branch, evidence=evidence, edits=request.edits,
                                   allowed_paths=request.allowed_paths, reason=request.reason)
            return {**revision, **(provenance.revision_binding(paper_store, revision) if paper_store else {})}
        except HistoryConflict as exc:
            raise HTTPException(409, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/revisions/{revision_id}", dependencies=secured)
    async def get_revision(revision_id: str):
        revision = history.get(revision_id)
        if revision is None:
            raise HTTPException(404, "Model revision not found.")
        lineage = history.lineage(revision_id)
        return {**revision, **(provenance.revision_binding(paper_store, revision) if paper_store else {}),
                "checks": history.checks(revision_id),
                "lineage": [{key: row[key] for key in ("id", "parent_id", "action", "changes", "created")}
                            for row in lineage[:100]]}

    @app.post("/api/revisions/{revision_id}/patch", dependencies=secured, status_code=201)
    async def patch_revision(revision_id: str, request: PatchRequest):
        try:
            parent = history.get(revision_id)
            if parent is None:
                raise ValueError("Unknown parent revision.")
            proposed, _ = apply_feature_edits(GeometryPlan.model_validate(parent["plan"]), request.edits, request.allowed_paths)
            provenance.validate_reserved_references(paper_store, proposed, request.evidence)
            return {**history.apply_patch(revision_id, request.edits, allowed_paths=request.allowed_paths,
                    branch=request.branch, evidence=request.evidence, reason=request.reason), "branch": request.branch}
        except HistoryConflict as exc:
            raise HTTPException(409, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.get("/api/recovery/{failure_id}", dependencies=secured)
    async def recovery_evidence(failure_id: str):
        try:
            return history.recovery_context(failure_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.post("/api/recovery/{failure_id}/restore", dependencies=secured, status_code=201)
    async def restore_checkpoint(failure_id: str, request: RecoveryRequest):
        branch = "recovery-" + uuid.uuid4().hex[:20]
        try:
            return {**history.branch_recovery(failure_id, request.checkpoint_id, branch=branch,
                                             diagnosis=request.diagnosis), "branch": branch}
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
