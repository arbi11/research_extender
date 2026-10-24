"""Authenticated paper evidence and pre-solve immutable target contracts.

No endpoint accepts a claimed target result. Native solver workers alone compare
their actual results with a contract already frozen when a job was queued.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

from fastapi import HTTPException, Request
import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import papers, provenance, structured
from .plan import GeometryPlan
from .targets import NumericalTarget, TargetContract, contract_fingerprint, freeze_targets
from .assembly_provider import AssemblyProviderError


MAX_CONTEXT_BYTES = 8 * 1024 * 1024
MAX_PROVIDER_BYTES = 8 * 1024 * 1024
MAX_JSON_FILE_BYTES = 32 * 1024 * 1024
# The provider's native output/reasoning budget is used. Owners can explicitly
# configure a deadline; the application does not cut reasoning off at 90 seconds.
EXTRACTION_TIMEOUT_SECONDS = None
ID_PATTERN = r"^[a-f0-9]{64}$"


class ReplicationIntegrityError(ValueError):
    pass


def _encode(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      ensure_ascii=True).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_once(path: Path, data: bytes):
    """Atomically publish a complete file without replacing an existing record."""
    if path.is_symlink():
        raise ReplicationIntegrityError("A stored replication record is not a regular file.")
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ReplicationIntegrityError("An immutable replication record already has different content.")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
                raise ReplicationIntegrityError("An immutable replication record already has different content.") from None
    finally:
        Path(temporary).unlink(missing_ok=True)


class ReplicationStore:
    def __init__(self, root: Path):
        self.root = root.resolve() / "replication"
        for directory in ("papers", "evidence", "contracts", "extraction-audits"):
            path = self.root / directory
            path.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                raise ReplicationIntegrityError("Replication storage directories must be regular directories.")

    def _path(self, category: str, identity: str, suffix=".json") -> Path | None:
        if not isinstance(identity, str) or not re.fullmatch(ID_PATTERN, identity):
            return None
        return self.root / category / (identity + suffix)

    def _put(self, category: str, record: dict) -> str:
        data = _encode(record)
        if len(data) > MAX_JSON_FILE_BYTES:
            raise ValueError("The replication record exceeds the storage limit.")
        identity = _sha(data)
        _write_once(self._path(category, identity), data)
        return identity

    def _get(self, category: str, identity: str) -> dict | None:
        path = self._path(category, identity)
        if path is None or not path.exists():
            return None
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_FILE_BYTES:
            raise ReplicationIntegrityError("A stored replication record has an invalid file type or size.")
        data = path.read_bytes()
        if _sha(data) != identity:
            raise ReplicationIntegrityError("A replication record no longer matches its content hash.")
        try:
            return json.loads(data)
        except ValueError:
            raise ReplicationIntegrityError("A stored replication record is unreadable.") from None

    def put_paper(self, data: bytes, filename="paper.pdf") -> dict:
        document = papers.extract_pdf(data, filename)
        paper_id = document.sha256
        index = self._path("papers", paper_id)
        if index.exists():
            return {"paper_id": paper_id, "document": self.get_document(paper_id).model_dump(mode="json")}
        record = document.model_dump(mode="json")
        # Original bytes and extracted text are both retained for an audit. The
        # index records the text hash independently of the original PDF digest.
        _write_once(self._path("papers", paper_id, ".pdf"), data)
        _write_once(index, _encode({"paper_id": paper_id, "document_sha256": _sha(_encode(record)), "document": record}))
        return {"paper_id": paper_id, "document": record}

    def get_document(self, paper_id: str) -> papers.PaperDocument | None:
        index = self._path("papers", paper_id)
        if index is None or not index.exists():
            return None
        pdf = self._path("papers", paper_id, ".pdf")
        if (index.is_symlink() or pdf.is_symlink() or not index.is_file() or not pdf.is_file()
                or index.stat().st_size > MAX_JSON_FILE_BYTES or pdf.stat().st_size > papers.MAX_PDF_BYTES):
            raise ReplicationIntegrityError("The stored paper has an invalid file type or size.")
        try:
            record = json.loads(index.read_bytes())
            document = papers.PaperDocument.model_validate(record["document"])
            if (record["paper_id"] != paper_id or document.sha256 != paper_id
                    or _sha(_encode(record["document"])) != record["document_sha256"]
                    or _sha(pdf.read_bytes()) != paper_id):
                raise ValueError
            return document
        except (ValueError, KeyError, TypeError):
            raise ReplicationIntegrityError("The stored paper or extracted text no longer matches its recorded digest.") from None

    def put_evidence(self, paper_id: str, evidence, *, model="manual", usage=None,
                     extraction_status=None, validation_summary=None, audit_id=None, rejected_claim_count=0,
                     assembly_inventory=None) -> dict:
        document = self.get_document(paper_id)
        if document is None:
            raise ValueError("The uploaded paper is unavailable.")
        parsed = papers.validate_evidence(document, evidence)
        record = {"paper_id": paper_id, "evidence": parsed.model_dump(mode="json"),
                  "model": model, "usage": _usage({"usage": usage or {}})}
        if extraction_status is not None:
            record.update(extraction_status=extraction_status, validation_summary=validation_summary or [],
                          audit_id=audit_id, rejected_claim_count=rejected_claim_count)
        if assembly_inventory is not None:
            record["assembly_inventory"] = assembly_inventory
        identity = self._put("evidence", record)
        return {"evidence_id": identity, **record}

    def get_evidence(self, evidence_id: str) -> dict | None:
        record = self._get("evidence", evidence_id)
        if record is None:
            return None
        document = self.get_document(record["paper_id"])
        if document is None:
            raise ReplicationIntegrityError("The evidence source paper is unavailable.")
        try:
            papers.validate_evidence(document, record["evidence"])
        except (ValueError, ValidationError):
            raise ReplicationIntegrityError("The stored evidence no longer agrees with its source document.") from None
        return {"evidence_id": evidence_id, **record}

    def put_contract(self, revision_id: str, case_id: str, targets, plan: GeometryPlan, *, evidence_id=None) -> dict:
        if not re.fullmatch(ID_PATTERN, revision_id):
            raise ValueError("A target contract requires an immutable model revision id.")
        contract = freeze_targets(targets, plan, case_id=case_id)
        record = {"revision_id": revision_id, "contract": contract.model_dump(mode="json"),
                  "contract_sha256": contract_fingerprint(contract)}
        if evidence_id is not None:
            record["evidence_id"] = evidence_id
        identity = self._put("contracts", record)
        return {"contract_id": identity, **record}

    def get_contract(self, contract_id: str) -> dict | None:
        record = self._get("contracts", contract_id)
        if record is None:
            return None
        try:
            contract = TargetContract.model_validate(record["contract"])
            if contract_fingerprint(contract) != record["contract_sha256"]:
                raise ValueError
        except (ValueError, KeyError, TypeError):
            raise ReplicationIntegrityError("The stored numerical target contract is invalid.") from None
        return {"contract_id": contract_id, **record}

    def put_extraction_audit(self, record: dict) -> str:
        """Private bounded diagnostics; no route publishes raw candidates."""
        return self._put("extraction-audits", record)

    def get_extraction_audit(self, audit_id: str) -> dict | None:
        return self._get("extraction-audits", audit_id)

    def planning_context(self, evidence_id: str, case_id="baseline", instructions="") -> dict:
        record = self.get_evidence(evidence_id)
        if record is None:
            raise ValueError("Paper evidence not found.")
        evidence = papers.PaperEvidence.model_validate(record["evidence"])
        known = {parameter.case_id for parameter in evidence.parameters} | {target.case_id for target in evidence.targets}
        if case_id not in known:
            raise ValueError("Select an extracted operating case before creating its model.")
        # Baseline parameters are common facts; explicitly tagged parameters and
        # targets from other operating cases are excluded from this model input.
        parameters = tuple(parameter for parameter in evidence.parameters if parameter.case_id in {"baseline", case_id})
        targets = tuple(target for target in evidence.targets if target.case_id == case_id)
        selected = evidence.model_copy(update={"parameters": parameters, "targets": targets})
        context = json.loads(papers.build_planning_context(selected, instructions))
        if record.get("assembly_inventory") is not None:
            # Discovery ids are reused only for the same operating case. Another
            # selected case receives a new discovery instead of a stale assembly.
            inventory = record["assembly_inventory"]
            if inventory.get("selected_case_id") in {None, case_id}:
                context["assembly_inventory"] = inventory
        context.update(evidence_id=evidence_id, case_id=case_id,
            source_policy="Paper parameters retain their cited baseline values. Apply explicit user changes as a separate variation and describe every override; never rewrite the cited baseline or inherit targets whose operating conditions changed.")
        if len(_encode(context)) > MAX_CONTEXT_BYTES:
            raise ValueError("The selected paper case exceeds the 32 KiB planning limit; retain fewer cited parameters.")
        return context


def _usage(packet: dict) -> dict:
    if not isinstance(packet, dict):
        return {}
    raw = packet.get("usage")
    if not isinstance(raw, dict):
        return {}
    result = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens", "cost"):
        value = raw.get(name)
        if name == "cost":
            if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1_000_000:
                result[name] = value
        elif type(value) is int and 0 <= value <= 1_000_000_000_000:
            result[name] = value
    for name in ("completion_tokens_details", "prompt_tokens_details", "cost_details"):
        if isinstance(raw.get(name), dict):
            result[name] = {str(key)[:80]: value for key, value in raw[name].items()
                            if type(value) in (int, float) and math.isfinite(value) and value >= 0}
    return result


class ExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instructions: str = Field(default="", max_length=3000)
    model: str | None = Field(default=None, min_length=1, max_length=160)
    hierarchical: bool = True


class TargetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision_id: str = Field(pattern=ID_PATTERN)
    evidence_id: str | None = Field(default=None, pattern=ID_PATTERN)
    case_id: str = Field(default="baseline", pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,79}$")
    targets: tuple[NumericalTarget, ...] = Field(min_length=1, max_length=32)


def validate_paper_targets(evidence: papers.PaperEvidence, case_id: str, targets) -> None:
    """Allow measurement binding/tolerance choices, never rewriting the source."""
    originals = {target.id: target for target in evidence.targets if target.case_id == case_id}
    protected = ("id", "case_id", "value", "unit", "reference_kind", "evidence_kind",
                 "citations", "conditions", "derivation")
    for target in targets:
        original = originals.get(target.id)
        if original is None or any(getattr(target, field) != getattr(original, field) for field in protected):
            raise ValueError("A paper-linked target must preserve its extracted value, units, operating conditions and source evidence. Create a separately labelled manual reference for a different target.")
        if target.observable.quantity != original.observable.quantity:
            raise ValueError("A paper target must retain its measurement quantity; only circuit/probe binding and pre-solve tolerance may change.")


def _extraction_tolerances(packet: dict) -> dict:
    """Choose our declared 2% criterion before solving, never from output fit."""
    if not isinstance(packet, dict):
        raise ValueError("Evidence must be an object.")
    targets, questions = [], list(packet.get("questions", []))
    raw_targets = packet.get("targets", [])
    if not isinstance(raw_targets, list):
        # The top-level/collection validator produces an audited schema error.
        return packet
    for target in raw_targets:
        if not isinstance(target, dict):
            targets.append(target)
            continue
        value = target.get("value")
        if type(value) not in (int, float) or not math.isfinite(value):
            # Scope this malformed claim to the independent claim validator;
            # it must not discard unrelated valid geometry/source evidence.
            targets.append(target)
            continue
        if value == 0:
            questions.append("Select an absolute tolerance for the zero-valued target '" + str(target.get("id", "unnamed"))[:64] + "' before freezing it.")
            continue
        target = dict(target)
        target["tolerance"] = {"absolute": 0, "relative": .02}
        target["description"] = str(target.get("description", ""))[:480] + " Acceptance tolerance: 2%, selected by the application before any solve; not a paper accuracy claim."
        targets.append(target)
    return {**packet, "targets": targets, "questions": questions}


def _redact_audit(value, secrets, *, depth=0):
    if depth > 24:
        return "[NESTING LIMIT]"
    if isinstance(value, str):
        for secret in secrets:
            if len(secret) < 8:
                continue
            value = value.replace(secret, "[REDACTED]")
            if "\\u" in value:
                encoded = "".join("(?:" + re.escape(character) + "|\\\\u" + f"{ord(character):04x}" + ")" for character in secret)
                value = re.sub(encoded, "[REDACTED]", value, flags=re.I)
        return value
    if isinstance(value, dict):
        return {_redact_audit(str(key), secrets, depth=depth+1): _redact_audit(item, secrets, depth=depth+1)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_audit(item, secrets, depth=depth+1) for item in value]
    if type(value) is float and not math.isfinite(value):
        return None
    if value is None or type(value) in (int, float, bool):
        return value
    return "[UNSUPPORTED VALUE]"


def _validation_issues(error) -> list[dict]:
    if getattr(error, "issues", None):
        return list(error.issues)
    if isinstance(error, papers.PaperEvidenceError):
        return [error.issue]
    if isinstance(error, ValidationError):
        return [{"path": ".".join(str(item) for item in issue["loc"]), "code": issue["type"],
                 "message": issue["msg"][:1000]}
                for issue in error.errors(include_input=False, include_url=False)[:20]]
    return [{"path": "", "code": type(error).__name__, "message": str(error)[:1000]}]


@dataclass(frozen=True)
class ExtractionOutcome:
    evidence: papers.PaperEvidence
    usage: dict
    extraction_status: str = "complete"
    validation_summary: tuple = ()
    audit_id: str | None = None
    rejected_claim_count: int = 0
    inventory: dict | None = None

    def __iter__(self):
        # Preserve the existing evidence, usage = await extract(...) interface.
        yield self.evidence
        yield self.usage


async def extract_with_provider(document: papers.PaperDocument, instructions: str, settings) -> ExtractionOutcome:
    if not settings.ai_key or not settings.ai_model:
        raise HTTPException(503, "Paper extraction needs the server OpenRouter API key and model configuration.")
    response_model = papers.source_evidence_model()
    messages = papers.extraction_messages(document, instructions)
    # Eligibility depends on the actual execution contract, rather than an
    # undefined request for a 'supported' problem or a remembered machine shape.
    from .assembly import capability_manifest
    messages.insert(1, {"role": "system", "content": "Execution capabilities and representation requirements: "
                     + json.dumps(capability_manifest(), ensure_ascii=True)})
    payload = structured.request_payload(settings.ai_model, messages,
        response_model=response_model, name="paper_source_evidence", temperature=None)
    audit_store = ReplicationStore(settings.data_dir)
    secrets = [settings.ai_key, getattr(settings, "token", "")]
    packet, content, decoded_candidate, finish_reason = {}, None, None, None
    discovery = None
    stage, started = "provider_request", time.monotonic()

    def combined_usage():
        values = [_usage(packet)] + ([discovery.usage] if discovery is not None else [])
        result = {}
        for value in values:
            for key, entry in value.items():
                if type(entry) in (int, float):
                    result[key] = result.get(key, 0) + entry
                elif isinstance(entry, dict):
                    group = result.setdefault(key, {})
                    for name, number in entry.items():
                        if type(number) in (int, float):
                            group[name] = group.get(name, 0) + number
        return result

    def record_audit(status, issues):
        audit = {"version": 1, "status": status, "stage": stage,
            "paper_sha256": document.sha256, "source_id": document.source_id,
            "page_count": document.page_count, "requested_model": settings.ai_model,
            "instructions": instructions, "finish_reason": finish_reason,
            "returned_model": packet.get("model"), "provider": packet.get("provider"),
            "provider_request_id": packet.get("id"),
            "contract": structured.contract_metadata(response_model),
            "request": payload, "provider_response": packet,
            "assembly_discovery": discovery.model_dump(mode="json") if discovery is not None else None,
            "usage": combined_usage(), "issues": issues,
            "candidate": decoded_candidate if decoded_candidate is not None else content,
            "candidate_bytes": len(content.encode()) if isinstance(content, str) else None,
            "elapsed_seconds": round(time.monotonic()-started, 3)}
        audit = _redact_audit(audit, secrets)
        audit_id = audit_store.put_extraction_audit(audit)
        return audit_id, audit

    def public_issues(audit):
        return [{key: issue[key] for key in ("path", "code", "message", "claim_id") if key in issue}
                for issue in audit["issues"][:8]]

    def failure(status, message, error):
        audit_id, audit = record_audit("failed", _validation_issues(error))
        return HTTPException(status, {"message": message, "audit_id": audit_id,
            "validation_summary": public_issues(audit), "usage": audit["usage"], "model": audit["requested_model"]})

    try:
        if getattr(settings, "hierarchical_ai", True):
            from .assembly_provider import discover_assembly
            stage = "machine_and_case_discovery"
            discovery = await discover_assembly(instructions or "Identify the machine, its components and operating cases in this paper.",
                settings.ai_key, settings.ai_base_url, settings.ai_model,
                evidence_context={"input_mode": "combined" if instructions.strip() else "paper",
                                  "paper_source_inventory": papers.source_inventory(document)})
            messages.insert(2, {"role": "system", "content": "First-stage machine/component/case inventory: "
                + discovery.inventory.model_dump_json() + "\nUse selected_case_id exactly for the selected operating case. "
                "Extract its complete inputs and measurements; do not mix another variant's figures, values or targets. "
                "A discovered name is not evidence for a physical value. Preserve missing information explicitly."})
            payload = structured.request_payload(settings.ai_model, messages, response_model=response_model,
                name="paper_source_evidence", temperature=None)
        stage = "provider_request"
        deadline = getattr(settings, "ai_timeout", EXTRACTION_TIMEOUT_SECONDS)
        async with asyncio.timeout(deadline):
            async with httpx.AsyncClient(timeout=httpx.Timeout(deadline, connect=20), follow_redirects=False) as client:
                async with client.stream("POST", settings.ai_base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": "Bearer " + settings.ai_key}, json=payload) as response:
                    if response.status_code >= 300:
                        if response.status_code == 429:
                            raise HTTPException(429, "OpenRouter is rate limiting paper extraction. Try again later.")
                        if response.status_code in (401, 402, 403):
                            raise HTTPException(502, "OpenRouter rejected paper extraction access or credit. Check the server key and permitted models.")
                        raise HTTPException(502, "The paper-extraction provider rejected the request or is unavailable.")
                    body = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        body.extend(chunk)
                        if len(body) > MAX_PROVIDER_BYTES:
                            raise HTTPException(502, "The paper-extraction provider returned too much data.")
        stage = "provider_envelope"
        packet = json.loads(body)
        choice = packet["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
        if choice.get("finish_reason") == "length" or not isinstance(content, str) or not 1 <= len(content.encode()) <= papers.MAX_EVIDENCE_BYTES:
            raise ValueError("Truncated or missing paper evidence.")
        if any(len(secret) >= 8 and secret in content for secret in secrets):
            raise ValueError("Unexpected credential in provider content.")
        stage = "candidate_json"
        decoded_candidate = json.loads(content)
        wire = structured.validate_response(decoded_candidate, response_model)
        parsed = _extraction_tolerances(papers.resolve_source_evidence(document, wire))
        stage = "schema_and_source_validation"
        evidence, issues, rejected_count = papers.validate_partial_evidence(document, parsed)
        if any(len(secret) >= 8 and secret in evidence.model_dump_json() for secret in secrets):
            raise ValueError("Unexpected credential in decoded evidence.")
        if rejected_count:
            audit_id, audit = record_audit("partial", issues)
            return ExtractionOutcome(evidence, combined_usage(), "partial", tuple(public_issues(audit)), audit_id, rejected_count,
                discovery.inventory.model_dump(mode="json") if discovery is not None else None)
        audit_id, _ = record_audit("complete", [])
        return ExtractionOutcome(evidence, combined_usage(), audit_id=audit_id,
            inventory=discovery.inventory.model_dump(mode="json") if discovery is not None else None)
    except HTTPException as error:
        raise failure(error.status_code, str(error.detail), error) from None
    except AssemblyProviderError as error:
        raise failure(error.status_code, error.message, error) from None
    except (TimeoutError, httpx.TimeoutException):
        raise failure(504, "Paper extraction timed out. The uploaded paper remains available; no model or solve was created.", TimeoutError("The bounded extraction deadline expired.")) from None
    except httpx.HTTPError:
        raise failure(502, "Could not connect to OpenRouter for paper extraction.", ValueError("The provider connection failed; request headers are not recorded.")) from None
    except (ValidationError, ValueError, TypeError, KeyError, IndexError, AttributeError, RecursionError) as error:
        raise failure(502, "Paper evidence failed validation. Review the cited claim or schema field listed below; no geometry was applied.", error) from None


def register_routes(app, secured, settings, budget, history, jobstore):
    store = ReplicationStore(settings.data_dir)
    app.state.replication = store

    def found_document(paper_id):
        try:
            document = store.get_document(paper_id)
        except ReplicationIntegrityError as error:
            raise HTTPException(409, str(error)) from None
        if document is None:
            raise HTTPException(404, "Uploaded paper not found.")
        return document

    @app.post("/api/papers", dependencies=secured, status_code=201)
    async def upload_paper(request: Request):
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/pdf":
            raise HTTPException(415, "Upload the PDF as application/pdf.")
        budget.use("paper_upload", 20)
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > papers.MAX_PDF_BYTES:
                raise HTTPException(413, "The PDF exceeds the 8 MiB upload limit.")
        try:
            return await asyncio.to_thread(store.put_paper, bytes(data), request.headers.get("x-filename", "paper.pdf"))
        except ReplicationIntegrityError as error:
            raise HTTPException(409, str(error)) from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.get("/api/papers/{paper_id}", dependencies=secured)
    async def get_paper(paper_id: str):
        return {"paper_id": paper_id, "document": found_document(paper_id).model_dump(mode="json")}

    @app.post("/api/papers/{paper_id}/extract", dependencies=secured)
    async def extract_paper(paper_id: str, request: ExtractionRequest):
        document = found_document(paper_id)
        budget.use("ai", 20)
        active_settings = replace(settings, ai_model=request.model or settings.ai_model,
                                  hierarchical_ai=request.hierarchical)
        outcome = await extract_with_provider(document, request.instructions, active_settings)
        evidence, usage = outcome
        try:
            metadata = ({"extraction_status": outcome.extraction_status,
                         "validation_summary": list(outcome.validation_summary), "audit_id": outcome.audit_id,
                         "rejected_claim_count": outcome.rejected_claim_count,
                         "assembly_inventory": outcome.inventory} if isinstance(outcome, ExtractionOutcome) else {})
            return store.put_evidence(paper_id, evidence, model=active_settings.ai_model, usage=usage, **metadata)
        except ReplicationIntegrityError as error:
            raise HTTPException(409, str(error)) from None
        except ValueError:
            raise HTTPException(422, "Paper evidence could not be stored with its source citations.") from None

    @app.post("/api/papers/{paper_id}/evidence", dependencies=secured, status_code=201)
    async def put_manual_evidence(paper_id: str, evidence: papers.PaperEvidence):
        found_document(paper_id)
        budget.use("paper_evidence", 60)
        try:
            return store.put_evidence(paper_id, evidence)
        except ReplicationIntegrityError as error:
            raise HTTPException(409, str(error)) from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.get("/api/paper-evidence/{evidence_id}", dependencies=secured)
    async def get_paper_evidence(evidence_id: str):
        try:
            record = store.get_evidence(evidence_id)
        except ReplicationIntegrityError as error:
            raise HTTPException(409, str(error)) from None
        if record is None:
            raise HTTPException(404, "Paper evidence not found.")
        return record

    @app.post("/api/targets", dependencies=secured, status_code=201)
    async def put_targets(request: TargetRequest):
        revision = history.get(request.revision_id)
        if revision is None:
            raise HTTPException(404, "Model revision not found.")
        budget.use("target_contract", 60)
        try:
            if request.evidence_id is not None:
                provenance.require_paper_binding(store, history, revision, request.evidence_id, request.case_id)
                record = store.get_evidence(request.evidence_id)
                if record is None:
                    raise HTTPException(404, "Paper evidence not found.")
                validate_paper_targets(papers.PaperEvidence.model_validate(record["evidence"]),
                                       request.case_id, request.targets)
            return store.put_contract(request.revision_id, request.case_id, request.targets,
                                      GeometryPlan.model_validate(revision["plan"]), evidence_id=request.evidence_id)
        except ReplicationIntegrityError as error:
            raise HTTPException(409, str(error)) from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.get("/api/targets/{contract_id}", dependencies=secured)
    async def get_target_contract(contract_id: str):
        try:
            record = store.get_contract(contract_id)
        except ReplicationIntegrityError as error:
            raise HTTPException(409, str(error)) from None
        if record is None:
            raise HTTPException(404, "Numerical target contract not found.")
        return record

    return store
