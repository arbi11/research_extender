"""Opt-in, audited model matrix for paper/NL -> hierarchy -> native FEMM.

Cases contain requests and source locations, never an answer key. Each model and
case gets a fresh application/database. There are no external retries, physical
corrections, target fitting, or model-specific prompt changes.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import quote, urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.main import Settings, create_app
from app.paper_assets import CropBox
from app.targets import NumericalTarget


DEFAULT_MODELS = ("~z-ai/glm-latest", "anthropic/claude-haiku-5.5", "deepseek/deepseek-v4-pro-0813")
PROTOCOL_VERSION = 3
TARGET_SELECTION_POLICY = {
    "reference_kinds": ["reported_numerical", "analytical", "experimental"],
    "evidence_kind": "explicit", "nonzero_source_value_required": True,
    "single_selected_case": True, "formula_substitution": False,
    "verdict_scope": "Agreement with each original reference method; analytical agreement is not FE-result replication.",
}
EXTRACTION_PROMPT = (
    "Identify the earliest fully specified operating case supported by the supplied execution capabilities. "
    "Extract its complete physical geometry, materials, excitation, coordinate conventions and numerical "
    "comparison targets. Keep distinct cases and analytical/numerical/experimental references separate. "
    "Use immutable source references and preserve missing information as questions. Do not invent "
    "physical inputs, target values or figure dimensions."
)
PLANNING_PROMPT = (
    "Create the selected source case faithfully as a machine/component hierarchy and editable features. "
    "Preserve sourced geometry, materials, excitation, coordinate conventions and measurement regions. "
    "Disclose numerical assumptions for exterior air, mesh and label placement. Preserve unsupported "
    "intent and missing information; do not replace its physics or invent dimensions to make it executable. "
    "Never fit inputs to published targets or claim simulated results."
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _safe(value, secrets):
    if isinstance(value, str):
        for secret in secrets:
            if secret and len(secret) >= 8:
                value = value.replace(secret, "[REDACTED]").replace(quote(secret, safe=""), "[REDACTED]")
                value = value.replace(json.dumps(secret, ensure_ascii=True)[1:-1], "[REDACTED]")
                if "\\u" in value:
                    pattern = "".join("(?:" + re.escape(char) + "|\\\\{1,4}u" + f"{ord(char):04x}" + ")" for char in secret)
                    value = re.sub(pattern, "[REDACTED]", value, flags=re.I)
        return value
    if isinstance(value, dict):
        return {_safe(str(key), secrets): _safe(entry, secrets) for key, entry in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(entry, secrets) for entry in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json(value))
    path.chmod(0o600)


class CohortStopped(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__("The cohort stopped after a provider quota/budget response; no further provider request was sent.")


class CohortQuotaGate:
    """First confirmed provider limit wins, including across runner processes."""

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self._local_reason = None

    def reason(self):
        if not self.path.exists():
            return self._local_reason
        try:
            value = json.loads(self.path.read_bytes())
            if not isinstance(value, dict):
                raise ValueError("Expected an object")
            return value
        except (OSError, ValueError):
            return {"kind": "unreadable_cohort_stop_marker", "scientific_outcome": "unassessed",
                    "marker": str(self.path)}

    def check(self):
        reason = self.reason()
        if reason is not None:
            raise CohortStopped(reason)

    def trip(self, reason):
        if self._local_reason is None:
            self._local_reason = reason
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix=".quota-stop-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(_json(reason))
                stream.flush()
                os.fsync(stream.fileno())
            # link publishes a fully written file atomically and never replaces
            # another process's first observed provider limit.
            try:
                os.link(temporary, self.path)
            except FileExistsError:
                pass
        except OSError as error:
            # A bad shared mount must not permit this process to continue
            # spending, or replace the original provider/read failure.
            self._local_reason = {**self._local_reason, "shared_marker_publish_error": type(error).__name__}
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return self.reason()


def _provider_quota_reason(status, packet):
    if status not in (402, 403):
        return None
    error = packet.get("error", {}) if isinstance(packet, dict) else {}
    error = error if isinstance(error, dict) else {"message": str(error)}
    metadata = error.get("metadata") if isinstance(error.get("metadata"), dict) else {}
    message = str(error.get("message", ""))
    text = (message + " " + str(metadata.get("raw", ""))).lower()
    limit_source = metadata.get("limit_source")
    exhausted = any(word in text for word in ("exceeded", "exhausted", "insufficient", "reached"))
    quota = any(word in text for word in ("quota", "budget", "credit", "usage limit", "key limit", "spend limit"))
    if status != 402 and limit_source not in {"openrouter_key_limit", "account_credit_limit", "account_budget_limit"} and not (exhausted and quota):
        return None
    return {"kind": "provider_quota_or_budget_exhausted", "observed_utc": _now(),
            "http_status": status, "provider_error_code": error.get("code"),
            "message": message, "limit_source": limit_source,
            "limit_usd": metadata.get("limit_usd"), "spend_usd": metadata.get("spend_usd"),
            "resets_at": metadata.get("resets_at"), "scientific_outcome": "unassessed"}


class DecodedAudit:
    """Observe complete decoded bodies; aread() caches bytes for application use.

    HTTPX response hooks run before the caller reads a response. Reading through
    HTTPX performs gzip decoding exactly once, unlike decoding raw compressed
    chunks as UTF-8. The existing client/transport/routing remain in control.
    """

    def __init__(self, directory: Path, *, secrets=(), allowed_hosts=(), quota_gate=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.secrets = tuple(secrets)
        self.allowed_hosts = set(allowed_hosts)
        self.quota_gate = quota_gate
        self.exchanges, self.pending, self.errors = [], {}, []
        self.blocked_requests = []

    def body(self, prefix, content):
        result = {"decoded_sha256": _sha(content), "decoded_bytes": len(content)}
        try:
            text = content.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            redacted = content
            for secret in self.secrets:
                if secret and len(secret) >= 8:
                    redacted = redacted.replace(secret.encode(), b"[REDACTED]")
            result["non_utf8_body_base64"] = base64.b64encode(redacted).decode()
            stored = redacted
            suffix = ".bin"
        else:
            stored = _safe(text, self.secrets).encode()
            suffix = ".txt"
            try:
                result["json"] = _safe(json.loads(text), self.secrets)
            except ValueError:
                result["json"] = None
        filename = prefix + suffix
        (self.directory / filename).write_bytes(stored)
        (self.directory / filename).chmod(0o600)
        result.update(body_file=filename, retained_sha256=_sha(stored), retained_bytes=len(stored),
                      complete=True, credential_redaction=True)
        return result

    async def on_request(self, request):
        if request.url.host not in self.allowed_hosts:
            return
        if self.quota_gate is not None:
            try:
                self.quota_gate.check()
            except CohortStopped:
                self.blocked_requests.append({"blocked_utc": _now(), "method": request.method,
                    "url": f"{request.url.scheme}://{request.url.host}{request.url.path}", "http_request_sent": False})
                raise
        record = {"id": len(self.exchanges) + 1, "started_utc": _now(),
            "started_monotonic": time.monotonic(), "method": request.method,
            "url": f"{request.url.scheme}://{request.url.host}{request.url.path}",
            "headers_recorded": False}
        self.exchanges.append(record)
        self.pending[id(request)] = record
        try:
            record["request"] = self.body(f'{record["id"]:04d}.request', request.content)
        except Exception as error:
            self.errors.append({"stage": "request_observation", "type": type(error).__name__})
        _write(self.directory / f'{record["id"]:04d}.json', record)

    async def on_response(self, response):
        record = self.pending.pop(id(response.request), None)
        if record is None:
            return
        record["status_code"] = response.status_code
        chunks = []
        original_iterator = response.aiter_bytes
        async def observe_decoded(*args, **kwargs):
            async for chunk in original_iterator(*args, **kwargs):
                chunks.append(chunk)
                yield chunk
        response.aiter_bytes = observe_decoded
        try:
            # The cached decoded body is yielded unchanged to the application.
            content = await response.aread()
            record["response"] = self.body(f'{record["id"]:04d}.response', content)
        except (Exception, asyncio.CancelledError) as error:
            self.errors.append({"stage": "response_observation", "type": type(error).__name__})
            record["response_observation_error"] = type(error).__name__
            if response.status_code == 402 and self.quota_gate is not None:
                reason = _provider_quota_reason(402, None)
                reason.update(exchange_id=record["id"], response_body_incomplete=True,
                    audit_exchange=str((self.directory / f'{record["id"]:04d}.json').resolve()))
                self.quota_gate.trip(_safe(reason, self.secrets))
            if chunks:
                record["partial_response"] = self.body(f'{record["id"]:04d}.response-partial', b"".join(chunks))
                record["partial_response"].update(complete=False, hash_scope="Decoded prefix only; response was interrupted.")
            record["elapsed_seconds"] = round(time.monotonic() - record["started_monotonic"], 6)
            _write(self.directory / f'{record["id"]:04d}.json', record)
            # A partial read has consumed the underlying stream. Swallowing its
            # exception would make the caller attempt another read and replace
            # the real network failure with a misleading StreamConsumed error.
            raise
        finally:
            response.aiter_bytes = original_iterator
        record["elapsed_seconds"] = round(time.monotonic() - record["started_monotonic"], 6)
        _write(self.directory / f'{record["id"]:04d}.json', record)
        reason = _provider_quota_reason(response.status_code, record["response"].get("json"))
        if reason is not None and self.quota_gate is not None:
            reason.update(exchange_id=record["id"], response_decoded_sha256=record["response"]["decoded_sha256"],
                          audit_exchange=str((self.directory / f'{record["id"]:04d}.json').resolve()),
                          requested_model=(record.get("request", {}).get("json") or {}).get("model"))
            self.quota_gate.trip(_safe(reason, self.secrets))

    def client_class(self, original):
        owner = self
        class AuditedClient(original):
            def __init__(self, *args, **kwargs):
                hooks = {key: list(value) for key, value in (kwargs.get("event_hooks") or {}).items()}
                hooks["request"] = [*hooks.get("request", []), owner.on_request]
                hooks["response"] = [owner.on_response, *hooks.get("response", [])]
                kwargs["event_hooks"] = hooks
                super().__init__(*args, **kwargs)
        return AuditedClient

    def summary(self):
        calls, known_cost, unknown_costs = [], 0.0, 0
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for exchange in self.exchanges:
            request = exchange.get("request", {}).get("json") or {}
            packet = exchange.get("response", {}).get("json") or {}
            usage = packet.get("usage") if isinstance(packet.get("usage"), dict) else {}
            cost = usage.get("cost")
            if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0:
                known_cost += cost
            else:
                unknown_costs += 1
            for key in totals:
                amount = usage.get(key)
                if type(amount) is int and amount >= 0:
                    totals[key] += amount
            choices = packet.get("choices") or []
            calls.append({"exchange_id": exchange["id"], "requested_model": request.get("model"),
                "returned_model": packet.get("model"), "provider": packet.get("provider"),
                "request_id": packet.get("id"), "usage": usage,
                "finish_reason": choices[0].get("finish_reason") if choices else None,
                "elapsed_seconds": exchange.get("elapsed_seconds"), "status_code": exchange.get("status_code")})
        return {"calls": calls, "provider_call_count": len(calls), "reported_known_cost": known_cost,
                "calls_without_reported_cost": unknown_costs,
                "reported_cost_complete": unknown_costs == 0, "reported_token_totals": totals,
                "observer_errors": self.errors, "pending_without_response": len(self.pending),
                "blocked_provider_requests": self.blocked_requests,
                "cohort_stop": self.quota_gate.reason() if self.quota_gate is not None else None,
                "transcript_complete": not self.errors and not self.pending
                    and all(exchange.get("response", {}).get("complete") for exchange in self.exchanges),
                "hash_scope": "Complete decoded HTTP bodies; original compressed/TLS bytes are not claimed."}


class AssetCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page: int = Field(ge=1)
    bbox: CropBox | None = None
    purpose: str = "page"
    label: str = ""
    case_id: str | None = None


class TargetSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str | None = None
    target_ids: list[str] = Field(default_factory=list)


class LiveCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    prompt: str = Field(min_length=1)
    pdf: str | None = None
    source_assets: list[AssetCase] = Field(default_factory=list)
    target_selection: TargetSelection = Field(default_factory=TargetSelection)
    review_mesh: bool = False
    focused_checks: bool = False


def load_cases(path: Path) -> list[LiveCase]:
    raw = json.loads(path.read_bytes())
    if not isinstance(raw, list) or not raw:
        raise ValueError("A cases file must contain a nonempty array of independent requests.")
    cases = [LiveCase.model_validate(entry) for entry in raw]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Case ids must be unique.")
    return [case.model_copy(update={"pdf": str((path.parent / case.pdf).resolve())})
            if case.pdf else case for case in cases]


def select_targets(extracted, selection: TargetSelection):
    evidence = extracted["evidence"]
    available = [NumericalTarget.model_validate(target) for target in evidence.get("targets", [])]
    eligible = [target for target in available if target.reference_kind in TARGET_SELECTION_POLICY["reference_kinds"]
                and target.evidence_kind == "explicit" and target.value != 0]
    case_id = selection.case_id or (extracted.get("assembly_inventory") or {}).get("selected_case_id")
    if case_id is None and eligible:
        # Source location and extraction order only; no matching to an answer key.
        eligible.sort(key=lambda target: min((citation.page or 10**9 for citation in target.citations), default=10**9))
        case_id = eligible[0].case_id
    eligible = [target for target in eligible if target.case_id == case_id]
    if selection.target_ids:
        eligible = [target for target in eligible if target.id in selection.target_ids]
        if {target.id for target in eligible} != set(selection.target_ids):
            raise ValueError("The requested source target ids are not all eligible explicit published references in the selected case.")
    if not eligible:
        raise ValueError("No explicit nonzero published analytical, numerical or experimental target belongs to the selected source case.")
    return case_id, [target.model_dump(mode="json") for target in eligible]


def bind_targets(targets, spec):
    bound = json.loads(json.dumps(targets))
    bindings = []
    for target in bound:
        observable = target["observable"]
        quantity = observable["quantity"]
        if quantity in {"current", "flux_linkage", "inductance"}:
            circuits = [circuit["id"] for circuit in spec["circuits"]]
            if observable.get("circuit_id") not in circuits:
                if len(circuits) != 1:
                    raise ValueError("A target circuit cannot be mapped uniquely; manual selection is forbidden.")
                bindings.append({"target_id": target["id"], "original_circuit": observable.get("circuit_id"),
                                 "native_circuit": circuits[0], "reason": "Exactly one native circuit exists."})
                observable["circuit_id"] = circuits[0]
        if quantity.startswith("force") and not spec.get("force_region"):
            raise ValueError("A published force target has no explicit measurement region.")
        probe = observable.get("probe_index")
        if probe is not None and probe >= len(spec.get("probes", [])):
            raise ValueError("A source target's required field probe is missing.")
    return bound, bindings


class TrialStop(ValueError):
    pass


async def run_case(case: LiveCase, model: str, output: Path, *, live=False, settings=None,
                   vision_model=None, quota_stop_path=None) -> dict:
    if not live:
        raise ValueError("--live is required before reading credentials or making paid requests.")
    if case.source_assets and (not case.pdf or not vision_model):
        raise ValueError("Source assets require a PDF and an explicitly selected --vision-model.")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh trial directory; previous outcomes cannot be overwritten or resumed.")
    original = settings or Settings.from_env()
    if not original.ai_key:
        raise ValueError("Configure the existing OpenRouter key before this explicitly live trial.")
    output.mkdir(parents=True, exist_ok=True)
    active = replace(original, data_dir=output / "data", ai_model=model,
        vision_model=vision_model or original.vision_model, start_worker=True, ai_timeout=None, hierarchical_ai=True)
    secrets = [active.ai_key, active.token]
    hosts = {urlparse(active.ai_base_url).hostname, urlparse(active.decision_base_url).hostname}
    quota_gate = CohortQuotaGate(quota_stop_path or output / "cohort-stop.json")
    auditor = DecodedAudit(output / "provider", secrets=secrets, allowed_hosts=hosts, quota_gate=quota_gate)
    original_client = httpx.AsyncClient
    report = {"version": PROTOCOL_VERSION, "status": "running", "started_utc": _now(), "case": case.model_dump(mode="json"),
        "target_selection_policy": TARGET_SELECTION_POLICY,
        "requested_model": model, "decision_model": active.decision_model,
        "vision_model": vision_model, "focused_checks_model": active.decision_model if case.focused_checks else None,
        "focused_checks_experimental": case.focused_checks,
        "cohort_stop_marker": str(quota_gate.path),
        "execution_pipeline_passed": False, "source_fidelity": {"status": "pending_independent_audit"},
        "external_retries": 0, "human_corrections": 0, "target_tuning": 0,
        "stages": {}, "api_calls": [], "attempt_policy": "One independent application attempt; no external correction or retry."}
    started, stage, app = time.monotonic(), "initialization", None

    def persist():
        _write(output / "report.json", _safe(report, secrets))

    def record_cancellation():
        report["stages"][stage] = "cancelled_unassessed"
        report["execution_pipeline_passed"] = False
        report["termination"] = {"kind": "task_cancelled", "stage": stage,
            "scientific_outcome": "unassessed", "subsequent_cases_allowed": False,
            "unreported_provider_charges": "unknown"}
        report["source_fidelity"] = {"status": "unassessed", "reason": "task_cancelled"}
        report["error"] = {"type": "CancelledError", "message": "Trial was cancelled before completion; no scientific verdict."}

    async def request(client, name, method, path, *, body=None, content=None, extra=None):
        kwargs = {"headers": {"Authorization": "Bearer " + active.token, **(extra or {})}}
        if body is not None:
            kwargs["json"] = body
        if content is not None:
            kwargs["content"] = content
        tick = time.monotonic()
        response = await client.request(method, path, **kwargs)
        try:
            result = response.json()
        except ValueError:
            artifact = name + (".zip" if "zip" in response.headers.get("content-type", "") else ".bin")
            (output / artifact).write_bytes(response.content)
            result = {"artifact": artifact, "sha256": _sha(response.content), "bytes": len(response.content)}
        report["api_calls"].append({"name": name, "method": method, "path": path,
            "request": body, "content_sha256": _sha(content) if content is not None else None,
            "status_code": response.status_code, "elapsed_seconds": round(time.monotonic() - tick, 6),
            "response": result})
        persist()
        if response.status_code >= 300:
            raise TrialStop(name + " returned HTTP " + str(response.status_code))
        return result

    async def wait_job(client, job, name):
        # Native execution keeps the owner's solver timeout; there is no model
        # output deadline or token budget in this runner.
        deadline = time.monotonic() + active.timeout + 30
        while time.monotonic() < deadline:
            latest = await request(client, name, "GET", "/api/jobs/" + job["id"])
            if latest["status"] in {"completed", "failed"}:
                return latest
            await asyncio.sleep(.2)
        raise TrialStop("The native worker did not finish within its configured execution deadline.")

    persist()
    try:
        quota_gate.check()
        httpx.AsyncClient = auditor.client_class(original_client)
        app = create_app(active)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://hierarchy-trial", timeout=None) as client:
                extracted, selected_targets, evidence_id, paper_id = None, [], None, None
                case_id = "baseline"
                if case.pdf:
                    stage = "paper_upload"
                    source = Path(case.pdf).read_bytes()
                    report["pdf_sha256"] = _sha(source)
                    uploaded = await request(client, "upload_pdf", "POST", "/api/papers", content=source,
                        extra={"Content-Type": "application/pdf", "X-Filename": Path(case.pdf).name})
                    paper_id = uploaded["paper_id"]
                    report["stages"][stage] = "passed"
                    stage = "paper_extraction"
                    extracted = await request(client, "extract_evidence", "POST", f"/api/papers/{paper_id}/extract",
                        body={"instructions": EXTRACTION_PROMPT + "\nUser request: " + case.prompt,
                              "model": model, "hierarchical": True})
                    if (extracted.get("extraction_status") != "complete" or extracted.get("rejected_claim_count", 0)
                            or extracted["evidence"].get("questions")):
                        raise TrialStop("Extraction is partial or unresolved; no manual repair or substituted physical input is allowed.")
                    evidence_id = extracted["evidence_id"]
                    case_id, selected_targets = select_targets(extracted, case.target_selection)
                    report["selected_source_targets"] = selected_targets
                    report["stages"][stage] = "passed"
                stage = "hierarchical_planning"
                proposal = await request(client, "create_hierarchy", "POST", "/api/plan/interpret", body={
                    "prompt": (PLANNING_PROMPT + "\nUser request: " + case.prompt) if case.pdf else case.prompt,
                    "model": model, "hierarchical": True, "evidence_id": evidence_id,
                    "case_id": case_id, "preserve_paper_excitation": bool(case.pdf)})
                report["proposal"] = proposal
                assembly_id = None
                if proposal.get("assembly") is not None:
                    if proposal.get("assembly_id"):
                        saved = await request(client, "fetch_assembly", "GET", "/api/assemblies/" + proposal["assembly_id"])
                    else:
                        saved = await request(client, "save_assembly", "POST", "/api/assemblies", body={
                            "assembly": proposal["assembly"], "instructions": case.prompt,
                            "evidence_id": evidence_id, "case_id": case_id})
                    assembly_id = saved["assembly_id"]
                    report["assembly_id"] = assembly_id
                    await request(client, "assembly_preview", "GET", f"/api/assemblies/{assembly_id}/preview.svg")
                asset_ids = []
                if case.source_assets:
                    for index, source_asset in enumerate(case.source_assets):
                        asset_body = source_asset.model_dump(mode="json", exclude_none=True)
                        asset_body.setdefault("case_id", case_id)
                        asset = await request(client, f"source_asset_{index}", "POST", f"/api/papers/{paper_id}/assets", body=asset_body)
                        asset_ids.append(asset["asset_id"])
                    comparison_body = {"asset_ids": asset_ids, "case_id": case_id,
                        "model": vision_model, "focused_checks": case.focused_checks,
                        "prompt": "Compare source and generated objects, counts, orientation, adjacency and annotated dimensions. Report concrete discrepancies and uncertainty."}
                    if assembly_id:
                        comparison_body["assembly_id"] = assembly_id
                    elif proposal.get("spec"):
                        comparison_body["spec"] = proposal["spec"]
                    else:
                        raise TrialStop("Source images were retained, but no geometry exists for visual comparison.")
                    stage = "source_visual_review"
                    report["source_visual_review"] = await request(client, "source_visual_review", "POST", "/api/source-review", body=comparison_body)
                    report["stages"][stage] = "completed_observation"
                stage = "hierarchical_planning"
                if proposal.get("spec") is None or proposal.get("questions"):
                    raise TrialStop("The retained hierarchy is unsupported, unresolved or has no executable native model.")
                if evidence_id and not proposal.get("planning_id"):
                    raise TrialStop("The model is not immutably bound to the paper evidence.")
                spec = proposal["spec"]
                report["stages"][stage] = "passed"
                stage = "revision_and_frozen_targets"
                revision = await request(client, "record_revision", "POST", "/api/revisions", body={
                    "spec": spec, "planning_id": proposal.get("planning_id"), "evidence_id": evidence_id,
                    "case_id": case_id, "preserve_paper_excitation": bool(case.pdf),
                    "evidence": ([{"source": "artifact", "reference": "assembly-planning:" + assembly_id,
                                   "sha256": assembly_id, "detail": "Immutable source-linked hierarchy."}] if assembly_id else [])})
                contract_id = None
                if selected_targets:
                    bound, bindings = bind_targets(selected_targets, spec)
                    report["administrative_bindings"] = bindings
                    frozen = await request(client, "freeze_targets", "POST", "/api/targets", body={
                        "revision_id": revision["id"], "evidence_id": evidence_id, "case_id": case_id, "targets": bound})
                    contract_id = frozen["contract_id"]
                report["stages"][stage] = "passed"
                stage = "native_mesh"
                queued = await request(client, "queue_mesh", "POST", "/api/plan/mesh", body=spec,
                    extra={"X-Model-Revision": revision["id"]})
                mesh = await wait_job(client, queued, "mesh_status")
                if mesh["status"] != "completed" or (mesh.get("result") or {}).get("status") != "passed":
                    raise TrialStop("Native meshing failed; its actual model and failure are retained.")
                report["stages"][stage] = "passed"
                if case.review_mesh and asset_ids:
                    stage = "source_mesh_review"
                    report["source_mesh_review"] = await request(client, "source_mesh_review", "POST", "/api/source-review", body={
                        "asset_ids": asset_ids, "spec": spec, "mesh_job_id": mesh["id"], "case_id": case_id,
                        "model": vision_model, "focused_checks": case.focused_checks})
                    report["stages"][stage] = "completed_observation"
                stage = "native_solver"
                queued = await request(client, "queue_solve", "POST", "/api/plan/jobs", body={"spec": spec,
                    "mesh_job_id": mesh["id"], "revision_id": revision["id"], "target_contract_id": contract_id})
                solve = await wait_job(client, queued, "solver_status")
                if solve["status"] != "completed":
                    raise TrialStop("The native solver failed; no correction or external retry is allowed.")
                report["stages"][stage] = "passed"
                await request(client, "native_export", "GET", "/api/jobs/" + solve["id"] + "/download.zip")
                if contract_id:
                    stage = "published_target_comparison"
                    report["target_report"] = solve.get("evidence", {}).get("target_report")
                    if not report["target_report"] or report["target_report"].get("status") != "passed":
                        raise TrialStop("The original frozen numerical target did not pass.")
                    report["stages"][stage] = "passed"
                report["execution_pipeline_passed"] = True
    except asyncio.CancelledError:
        record_cancellation()
        raise
    except Exception as error:
        report["stages"][stage] = "failed_or_incomplete"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        httpx.AsyncClient = original_client
        # A failed API gate can leave an application lifespan through an
        # exception. Explicit cleanup also protects older app lifespan versions.
        try:
            if app is not None and getattr(app.state, "worker", None) is not None:
                await asyncio.to_thread(app.state.worker.stop)
        except asyncio.CancelledError:
            record_cancellation()
            raise
        finally:
            quota_reason = quota_gate.reason()
            if quota_reason is not None and not report["execution_pipeline_passed"]:
                previous = report.get("termination")
                report["termination"] = {"kind": "provider_quota_stop", "reason": quota_reason,
                    "scientific_outcome": "unassessed", "subsequent_cases_allowed": False}
                if previous is not None:
                    report["termination"]["interruption"] = previous
                report["source_fidelity"] = {"status": "unassessed", "reason": "provider_quota_stop"}
            report.update(status="locked", ended_utc=_now(), elapsed_seconds=round(time.monotonic() - started, 6),
                          provider_audit=auditor.summary())
            persist()
            manifest = {str(path.relative_to(output)): _sha(path.read_bytes()) for path in output.rglob("*")
                        if path.is_file() and path.name != "manifest.json"}
            _write(output / "manifest.json", manifest)
    return _safe(report, secrets)


async def run_matrix(cases, output: Path, *, live=False, models=DEFAULT_MODELS, settings=None,
                     vision_model=None, quota_stop_path=None):
    if not live:
        raise ValueError("--live is required before environment access or paid calls.")
    parsed = [case if isinstance(case, LiveCase) else LiveCase.model_validate(case) for case in cases]
    if not parsed or not models or len(set(case.id for case in parsed)) != len(parsed) or len(set(models)) != len(models):
        raise ValueError("Provide nonempty independent cases and unique model ids.")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("The model matrix requires a fresh output directory.")
    output.mkdir(parents=True, exist_ok=True)
    quota_gate = CohortQuotaGate(quota_stop_path or output / "cohort-stop.json")
    _write(output / "protocol.json", {"version": PROTOCOL_VERSION, "created_utc": _now(), "models": list(models),
        "target_selection_policy": TARGET_SELECTION_POLICY,
        "cohort_stop_marker": str(quota_gate.path),
        "quota_policy": "Stop after an actual provider quota/budget 402/403; no new request after observing the shared marker. Existing in-flight charges remain unknown unless reported.",
        "cases": [case.model_dump(mode="json") for case in parsed], "vision_model": vision_model,
        "external_retries": 0, "human_corrections": 0, "gold_inputs": False,
        "model_token_budget": "Provider native default; no artificial max_tokens.",
        "model_read_deadline": None, "source_fidelity": "Independent audit remains separate."})
    reports, scheduled, termination = [], [], None
    planned = [(model, case) for model in models for case in parsed]
    def summary(status):
        unstarted = [{"requested_model": model, "case_id": case.id,
                      "disposition": "not_started_after_cohort_stop" if termination or quota_gate.reason() else "not_started",
                      "scientific_outcome": "unassessed"}
                     for model, case in planned if (model, case.id) not in scheduled]
        return {"status": status, "version": PROTOCOL_VERSION, "trial_count": len(reports),
            "planned_trial_count": len(planned), "trials": reports, "not_started": unstarted,
            "termination": termination, "cohort_stop": quota_gate.reason(),
            "source_fidelity": "Pending independent source/intent audit; native success is not a reproduction verdict."}
    try:
        for model, case in planned:
            if quota_gate.reason() is not None:
                termination = {"kind": "provider_quota_stop", "scientific_outcome": "unassessed"}
                break
            slug = re.sub(r"[^A-Za-z0-9_-]", "_", model) + "-" + _sha(model.encode())[:8]
            trial_path = output / slug / case.id
            scheduled.append((model, case.id))
            try:
                reports.append(await run_case(case, model, trial_path,
                    live=True, settings=settings, vision_model=vision_model, quota_stop_path=quota_gate.path))
            except asyncio.CancelledError:
                if (trial_path / "report.json").exists():
                    reports.append(json.loads((trial_path / "report.json").read_bytes()))
                raise
            _write(output / "summary.json", summary("running"))
            if quota_gate.reason() is not None:
                termination = {"kind": "provider_quota_stop", "scientific_outcome": "unassessed"}
                break
    except asyncio.CancelledError:
        termination = {"kind": "task_cancelled", "scientific_outcome": "unassessed"}
        raise
    finally:
        _write(output / "summary.json", summary("locked"))
    return summary("locked")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vision-model", help="Explicit shared visual model when cases select source assets")
    parser.add_argument("--cohort-stop", type=Path, help="Shared writable quota-stop marker for independently launched model processes")
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required before reading credentials or making paid requests")
    summary = asyncio.run(run_matrix(load_cases(args.cases), args.output, live=True,
        models=args.models, vision_model=args.vision_model, quota_stop_path=args.cohort_stop))
    print(json.dumps({"status": summary["status"], "trial_count": summary["trial_count"],
        "trials": [{"case": report["case"]["id"], "model": report["requested_model"],
                    "execution_pipeline_passed": report["execution_pipeline_passed"],
                    "reported_known_cost": report["provider_audit"]["reported_known_cost"],
                    "error": report.get("error")} for report in summary["trials"]]}))
    raise SystemExit(0 if not summary["termination"] and not summary["not_started"]
                     and all(report["execution_pipeline_passed"] for report in summary["trials"]) else 1)


if __name__ == "__main__":
    main()
