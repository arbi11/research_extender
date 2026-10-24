"""Immutable component assemblies, their compiler output and scoped edits.

The saved assembly is a source-linked program of geometric features. It can be
inspected even when its physics cannot yet be executed by the native harness.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from .assembly import MachineAssembly, AssemblyEdit, compile_assembly, preview_assembly, apply_assembly_edits
from . import assembly_provider
from .replication import ReplicationStore, ReplicationIntegrityError
from .targets import model_fingerprint

PREFIX = "assembly-planning:"


class AssemblyStore:
    def __init__(self, paper_store: ReplicationStore):
        self.store = paper_store
        path = self.store.root / "assemblies"
        path.mkdir(exist_ok=True)
        if path.is_symlink():
            raise ReplicationIntegrityError("Assembly storage must be a regular directory.")

    def put(self, assembly, *, instructions="", inventory=None, provider_usage=None,
            provider_metadata=None, evidence_id=None, case_id="baseline", parent_id=None, changes=None):
        parsed = assembly if isinstance(assembly, MachineAssembly) else MachineAssembly.model_validate(assembly)
        if evidence_id is None and any(fact.status == "paper" for fact in parsed.facts):
            raise ValueError("Paper-origin facts require verified paper evidence and its selected operating case.")
        if evidence_id is not None:
            # Unknown source ids/cases cannot be attached to a model after the fact.
            context = self.store.planning_context(evidence_id, case_id, instructions)
            assembly_provider.validate_paper_facts(parsed, context)
        if parent_id is not None and self.get(parent_id) is None:
            raise ValueError("The parent assembly is unavailable.")
        compilation = compile_assembly(parsed)
        record = {"version": 1, "assembly": parsed.model_dump(mode="json"),
                  "compilation": compilation.model_dump(mode="json"),
                  "preview_svg": preview_assembly(compilation), "feature_map": compilation.feature_map,
                  "model_sha256": model_fingerprint(compilation.plan) if compilation.plan is not None else None,
                  "instructions": instructions, "inventory": inventory.model_dump(mode="json") if isinstance(inventory, BaseModel) else inventory,
                  "provider_usage": provider_usage or [], "provider_metadata": provider_metadata or [],
                  "evidence_id": evidence_id, "case_id": case_id, "parent_id": parent_id,
                  "changes": changes or []}
        identity = self.store._put("assemblies", record)
        return {"assembly_id": identity, **record}

    def get(self, identity):
        record = self.store._get("assemblies", identity)
        if record is None:
            return None
        parsed = MachineAssembly.model_validate(record["assembly"])
        compilation = compile_assembly(parsed)
        if (compilation.model_dump(mode="json") != record["compilation"]
                or preview_assembly(compilation) != record["preview_svg"]
                or compilation.feature_map != record["feature_map"]
                or (model_fingerprint(compilation.plan) if compilation.plan is not None else None) != record["model_sha256"]):
            raise ReplicationIntegrityError("The assembly no longer agrees with its recorded compiler output.")
        try:
            if record["evidence_id"] is None:
                if any(fact.status == "paper" for fact in parsed.facts):
                    raise ValueError("Unverified paper-origin facts.")
            else:
                context = self.store.planning_context(record["evidence_id"], record["case_id"], record["instructions"])
                assembly_provider.validate_paper_facts(parsed, context)
        except ValueError:
            raise ReplicationIntegrityError("The assembly paper facts no longer agree with their verified source evidence.") from None
        return {"assembly_id": identity, **record}

    def evidence(self, identity, spec):
        record = self.get(identity)
        if record is None or record["model_sha256"] != model_fingerprint(spec):
            raise ValueError("The assembly reference does not compile to this exact model.")
        return {"source": "artifact", "reference": PREFIX + identity, "sha256": identity,
                "detail": "Source-linked component hierarchy and feature dependencies for this exact native model."}


class SaveAssemblyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assembly: MachineAssembly
    instructions: str = ""
    evidence_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    case_id: str = "baseline"
    parent_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class AssemblyEditsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    edits: list[AssemblyEdit] = Field(min_length=1)
    allowed_paths: list[str] | None = None
    instructions: str = ""


class DiscoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)
    model: str | None = None
    evidence_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    case_id: str = "baseline"


def register_routes(app, secured, settings, budget, paper_store):
    store = AssemblyStore(paper_store)
    app.state.assemblies = store

    def get(identity):
        record = store.get(identity)
        if record is None:
            raise HTTPException(404, "Assembly not found.")
        return record

    @app.post("/api/assemblies/discover", dependencies=secured)
    async def discover(request: DiscoveryRequest):
        budget.use("ai", 20)
        try:
            context = paper_store.planning_context(request.evidence_id, request.case_id, request.prompt) if request.evidence_id else None
            value = await assembly_provider.discover_assembly(request.prompt, settings.ai_key, settings.ai_base_url,
                request.model or settings.ai_model, evidence_context=context,
                decision_model=settings.decision_model, decision_base_url=settings.decision_base_url)
            return value.model_dump(mode="json")
        except assembly_provider.AssemblyProviderError as error:
            raise HTTPException(error.status_code, error.message) from None
        except (ValueError, ReplicationIntegrityError) as error:
            raise HTTPException(422, str(error)) from None

    @app.post("/api/assemblies", dependencies=secured, status_code=201)
    async def save(request: SaveAssemblyRequest):
        try:
            return await asyncio.to_thread(store.put, request.assembly, instructions=request.instructions,
                evidence_id=request.evidence_id, case_id=request.case_id, parent_id=request.parent_id)
        except ReplicationIntegrityError as error:
            raise HTTPException(409, str(error)) from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.get("/api/assemblies/{identity}", dependencies=secured)
    async def fetch(identity: str):
        try:
            return await asyncio.to_thread(get, identity)
        except ReplicationIntegrityError as error:
            raise HTTPException(409, str(error)) from None

    @app.get("/api/assemblies/{identity}/preview.svg", dependencies=secured)
    async def preview(identity: str):
        try:
            record = await asyncio.to_thread(get, identity)
            return Response(record["preview_svg"], media_type="image/svg+xml")
        except ReplicationIntegrityError as error:
            raise HTTPException(409, str(error)) from None

    @app.post("/api/assemblies/{identity}/edits", dependencies=secured, status_code=201)
    async def edit(identity: str, request: AssemblyEditsRequest):
        try:
            before = await asyncio.to_thread(get, identity)
            edited = apply_assembly_edits(before["assembly"], request.edits, allowed_paths=request.allowed_paths)
            # A manual/user variation cannot keep a false paper-fact claim.
            record = await asyncio.to_thread(store.put, edited["assembly"], parent_id=identity,
                instructions=request.instructions, evidence_id=before["evidence_id"],
                case_id=before["case_id"], changes=edited["changed_paths"])
            return {**record, "affected_features": edited["affected_feature_ids"]}
        except ReplicationIntegrityError as error:
            raise HTTPException(409, str(error)) from None
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise HTTPException(422, str(error)) from None

    return store
