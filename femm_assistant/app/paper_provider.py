"""Small strict paper-intake stages with immutable source-span references.

Each completed stage is a checkpoint. Provider/grammar failures stop intake;
there is no ordinary-JSON fallback, alternate model, or implicit retry. The
assembled claims still pass papers.resolve_source_evidence and its independent
per-claim scientific checks before becoming planning evidence.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Literal

import httpx
from pydantic import Field, ValidationError

from . import papers, structured
from .targets import FrozenModel, Observable


MAX_PROVIDER_BYTES = 8 * 1024 * 1024
CLAIM_ID = r"^[A-Za-z][A-Za-z0-9_-]{0,63}$"
CASE_ID = r"^[A-Za-z][A-Za-z0-9_-]{0,79}$"
ID_DESCRIPTION = "Opaque claim id: start with a letter; use letters, digits, underscore or hyphen; at most 64 characters."
CASE_DESCRIPTION = "Use an exact declared case id. baseline identifies common source inputs, not another operating variant."


class PaperCaseSelection(FrozenModel):
    source_id: str = Field(pattern=r"^paper_[a-f0-9]{24}$")
    title: str = Field(min_length=1, max_length=500)
    summary: str = Field(min_length=1, max_length=2000)
    case_ids: tuple[str, ...] = Field(min_length=1, max_length=32,
        description="Selected operating case id and, if common inputs exist, baseline. Preserve discovery ids exactly.")
    selected_case_id: str = Field(pattern=CASE_ID, description=CASE_DESCRIPTION)
    questions: tuple[str, ...] = Field(max_length=24)
    limitations: tuple[str, ...] = Field(max_length=24)


class ParameterDraft(FrozenModel):
    """Separate scalar slots avoid the former number/text/null value union."""

    id: str = Field(pattern=CLAIM_ID, description=ID_DESCRIPTION)
    case_id: str = Field(pattern=CASE_ID, description=CASE_DESCRIPTION)
    description: str = Field(min_length=1, max_length=600)
    value_number: float | None
    value_text: str | None
    unit: str | None
    evidence_kind: Literal["explicit", "derived", "assumption", "unknown"]
    source_refs: tuple[str, ...] = Field(max_length=12)
    value_refs: tuple[str, ...] = Field(max_length=12)
    unit_refs: tuple[str, ...] = Field(max_length=12)
    header_refs: tuple[str, ...] = Field(max_length=12)
    derivation: str | None = Field(max_length=1200)
    assumption_reason: str | None = Field(max_length=1000)
    unresolved_reason: str | None = Field(max_length=1200)

    def source_claim(self):
        result = self.model_dump(mode="json", exclude={"value_number", "value_text"})
        result["value"] = self.value_number if self.value_number is not None else self.value_text
        return result


class PaperParameters(FrozenModel):
    source_id: str = Field(pattern=r"^paper_[a-f0-9]{24}$")
    parameters: tuple[ParameterDraft, ...] = Field(max_length=120)
    questions: tuple[str, ...] = Field(max_length=24)
    limitations: tuple[str, ...] = Field(max_length=24)


class ReferenceDraft(FrozenModel):
    """Flat observable fields; acceptance tolerance is set by the application."""

    id: str = Field(pattern=CLAIM_ID, description=ID_DESCRIPTION)
    case_id: str = Field(pattern=CASE_ID, description=CASE_DESCRIPTION)
    description: str = Field(min_length=1, max_length=600)
    quantity: Observable.model_fields["quantity"].annotation
    circuit_id: str | None
    probe_index: int | None = Field(ge=0, le=15, strict=True)
    value: float = Field(strict=True)
    unit: str = Field(min_length=1, max_length=24)
    reference_kind: Literal["reported_numerical", "analytical", "experimental"]
    evidence_kind: Literal["explicit", "derived"]
    derivation: str | None = Field(max_length=1000)
    conditions: str = Field(min_length=1, max_length=1000)
    source_refs: tuple[str, ...] = Field(max_length=12)
    value_refs: tuple[str, ...] = Field(max_length=12)
    unit_refs: tuple[str, ...] = Field(max_length=12)
    header_refs: tuple[str, ...] = Field(max_length=12)

    def source_claim(self):
        result = self.model_dump(mode="json", exclude={"quantity", "circuit_id", "probe_index"})
        result["observable"] = {"quantity": self.quantity, "circuit_id": self.circuit_id,
                                "probe_index": self.probe_index}
        # A declared application choice, never a source accuracy claim. Zero
        # references remain explicit unresolved requests for an absolute tolerance.
        result["tolerance"] = {"absolute": 0, "relative": .02}
        return result


class PaperReferences(FrozenModel):
    source_id: str = Field(pattern=r"^paper_[a-f0-9]{24}$")
    targets: tuple[ReferenceDraft, ...] = Field(max_length=32)
    questions: tuple[str, ...] = Field(max_length=24)
    limitations: tuple[str, ...] = Field(max_length=24)


STAGES = (("paper_cases", PaperCaseSelection), ("paper_parameters", PaperParameters),
          ("paper_references", PaperReferences))


class PaperProviderError(ValueError):
    def __init__(self, status_code, message, *, stage, attempts, issues=()):
        super().__init__(message)
        self.status_code, self.message, self.stage = status_code, message, stage
        self.attempts, self.issues = list(attempts), list(issues)


@dataclass(frozen=True)
class PaperExtraction:
    candidate: dict
    attempts: tuple[dict, ...]
    cases: PaperCaseSelection


def _issues(error):
    if getattr(error, "issues", None):
        return list(error.issues)
    if isinstance(error, ValidationError):
        return [{"path": ".".join(map(str, item["loc"])), "code": item["type"],
                 "message": item["msg"][:1000]}
                for item in error.errors(include_input=False, include_url=False)[:20]]
    return [{"path": "", "code": type(error).__name__, "message": str(error)[:1000]}]


def _failure_message(status, packet):
    if status == 429:
        return "OpenRouter is rate limiting paper extraction. Try again later."
    if status in (401, 402, 403):
        return "OpenRouter rejected paper extraction access or credit. Check the server key and permitted models."
    message = packet.get("error", {}).get("message", "") if isinstance(packet, dict) and isinstance(packet.get("error"), dict) else ""
    if status in (400, 422) and isinstance(message, str) and "grammar" in message.lower():
        return "The selected provider could not compile this strict paper-extraction stage. No schema fallback or physical repair was attempted."
    return "The paper-extraction provider rejected the request or is unavailable."


async def _request(stage, response_model, messages, settings, attempts):
    # Reuse the established numeric billing allowlist; no raw provider fields
    # become public usage. Import lazily to keep this module independent of API wiring.
    from .replication import _usage

    payload = structured.request_payload(settings.ai_model, messages, response_model=response_model, name=stage)
    record = {"stage": stage, "requested_model": settings.ai_model,
              "contract": structured.contract_metadata(response_model), "request": payload,
              "usage": {}, "outcome": "request_started"}
    attempts.append(record)
    deadline = getattr(settings, "ai_timeout", None)
    try:
        async with asyncio.timeout(deadline):
            async with httpx.AsyncClient(timeout=httpx.Timeout(deadline, connect=20), follow_redirects=False) as client:
                async with client.stream("POST", settings.ai_base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": "Bearer " + settings.ai_key}, json=payload) as response:
                    record["http_status"] = response.status_code
                    body = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        body.extend(chunk)
                        if len(body) > MAX_PROVIDER_BYTES:
                            raise ValueError("The paper-extraction provider returned too much data.")
        try:
            packet = json.loads(body)
        except (ValueError, UnicodeError):
            record["provider_response_text"] = body.decode("utf-8", errors="replace")
            raise ValueError("The paper-extraction provider envelope is not JSON.") from None
        record["provider_response"] = packet
        record["usage"] = _usage(packet)
        if isinstance(packet, dict):
            for field in ("model", "provider", "id"):
                if isinstance(packet.get(field), str):
                    record[field] = packet[field]
        if record["http_status"] >= 300:
            record["outcome"] = "provider_rejected"
            raise PaperProviderError(429 if record["http_status"] == 429 else 502,
                _failure_message(record["http_status"], packet), stage=stage, attempts=attempts,
                issues=[{"path": "", "code": "provider_rejected", "message": "The provider rejected the strict stage request."}])
        choice = packet["choices"][0]
        record["finish_reason"] = choice.get("finish_reason")
        content = choice["message"]["content"]
        record["candidate"] = content
        if choice.get("finish_reason") in ("length", "content_filter") or choice["message"].get("refusal"):
            raise ValueError("Incomplete or refused paper evidence.")
        if not isinstance(content, str) or not 1 <= len(content.encode()) <= papers.MAX_EVIDENCE_BYTES:
            raise ValueError("Truncated, oversized or missing paper evidence.")
        record["candidate_bytes"] = len(content.encode())
        secrets = (settings.ai_key, getattr(settings, "token", ""))
        if any(len(secret) >= 8 and secret in content for secret in secrets):
            raise ValueError("Unexpected credential in provider content.")
        record["candidate_sha256"] = hashlib.sha256(content.encode()).hexdigest()
        candidate = json.loads(content)
        record["candidate"] = candidate
        parsed = response_model.model_validate(structured.validate_response(candidate, response_model))
        record["outcome"] = "validated"
        return parsed
    except PaperProviderError:
        raise
    except (TimeoutError, httpx.TimeoutException) as error:
        record["outcome"] = "timed_out"
        raise PaperProviderError(504, "Paper extraction timed out. The uploaded paper remains available; no model or solve was created.",
            stage=stage, attempts=attempts, issues=_issues(error)) from None
    except httpx.HTTPError:
        record["outcome"] = "connection_failed"
        raise PaperProviderError(502, "Could not connect to OpenRouter for paper extraction.", stage=stage, attempts=attempts,
            issues=[{"path": "", "code": "connection_failed", "message": "The provider connection failed; request headers are not recorded."}]) from None
    except (ValidationError, ValueError, TypeError, KeyError, IndexError, AttributeError, RecursionError) as error:
        record["outcome"] = "validation_failed"
        record["issues"] = _issues(error)
        raise PaperProviderError(502, "Paper evidence failed its strict stage contract; no geometry was applied.",
            stage=stage, attempts=attempts, issues=record["issues"]) from None


def _stage_issue(stage, attempts, path, message):
    issues = [{"path": path, "code": "stage_binding", "message": message}]
    attempts[-1].update(outcome="validation_failed", issues=issues)
    raise PaperProviderError(502, "Paper evidence failed case/source binding; no geometry was applied.",
                             stage=stage, attempts=attempts, issues=issues)


def _messages(base, *instructions):
    """Keep policy/checkpoints ahead of the untrusted user/source payload."""
    return [*base[:-1], *instructions, base[-1]]


async def extract_source_evidence(document, instructions, settings, *, inventory=None):
    """Run three strict checkpoints, retaining all responses even on later failure."""
    from .assembly import capability_manifest

    base = papers.extraction_messages(document, instructions)
    base.insert(1, {"role": "system", "content": "Execution capabilities and representation requirements: "
                   + json.dumps(capability_manifest(), ensure_ascii=True)})
    inventory_data = inventory.model_dump(mode="json") if inventory is not None else None
    if inventory_data is not None:
        base.insert(2, {"role": "system", "content": "First-stage machine/component/case inventory: "
            + json.dumps(inventory_data, separators=(",", ":"))
            + "\nPreserve selected_case_id exactly. Use baseline only for common source inputs. "
              "Do not mix other variants' inputs or targets. Discovery is not evidence for physical values."})
    attempts = []
    case_messages = _messages(base, {"role": "system", "content":
        "This checkpoint returns only paper identity, a concise summary, case_ids, selected_case_id, questions and limitations. "
        "case_ids lists the selected operating case and baseline if there are common source inputs. "
        "Do not return parameter values or reference quantities. With no specific variants, select baseline. "
        "Do not silently discard missing information or unsupported source physics."})
    cases = await _request("paper_cases", PaperCaseSelection, case_messages, settings, attempts)
    if cases.source_id != document.source_id:
        _stage_issue("paper_cases", attempts, "source_id", "The stage belongs to another uploaded paper.")
    if (len(set(cases.case_ids)) != len(cases.case_ids) or cases.selected_case_id not in cases.case_ids
            or any(not re.fullmatch(CASE_ID, case) for case in cases.case_ids)):
        _stage_issue("paper_cases", attempts, "case_ids", "Declare unique valid case ids including the selected case.")
    selected = inventory_data.get("selected_case_id") if inventory_data else None
    if selected is not None and cases.selected_case_id != selected:
        _stage_issue("paper_cases", attempts, "selected_case_id", "Preserve the selected discovery case id exactly.")
    if set(cases.case_ids) - {"baseline", cases.selected_case_id}:
        _stage_issue("paper_cases", attempts, "case_ids", "Extract only the selected operating case and common baseline inputs.")
    checkpoint = {"role": "system", "content": "Validated paper/case checkpoint: " + cases.model_dump_json()}
    parameters = await _request("paper_parameters", PaperParameters, _messages(base, checkpoint,
        {"role": "system", "content": "Return only source parameters, questions and limitations for the validated case. "
         "Use value_number for numeric values, value_text for categorical values, exactly one slot for a known value; "
         "both null only for unknown parameters with an unresolved_reason. Preserve common inputs under baseline and "
         "case-specific inputs under selected_case_id. Preserve all needed geometry, materials and excitation, "
         "including unresolved/derived claims. No target values or source quotations."}), settings, attempts)
    for index, parameter in enumerate(parameters.parameters):
        if parameter.value_number is not None and parameter.value_text is not None:
            _stage_issue("paper_parameters", attempts, f"parameters.{index}", "A source parameter cannot contain two different value types.")
    if parameters.source_id != document.source_id:
        _stage_issue("paper_parameters", attempts, "source_id", "The stage belongs to another uploaded paper.")
    for index, parameter in enumerate(parameters.parameters):
        if parameter.case_id not in cases.case_ids:
            _stage_issue("paper_parameters", attempts, f"parameters.{index}.case_id", "The parameter belongs to an unselected operating case.")
    references = await _request("paper_references", PaperReferences, _messages(base, checkpoint,
        {"role": "system", "content": "Return only exact published numerical references for the selected case, with "
         "questions and limitations. Flatten the observable into quantity, circuit_id and probe_index. Preserve "
         "reference_kind as reported_numerical, analytical or experimental; a numerical solver may compare any of these "
         "while disclosing its kind. Do not replace published values by formula evaluations or desired design setpoints. "
         "Application tolerance is selected separately before solving; do not invent a paper accuracy statement. "
         "Unavailable numeric targets remain questions; source values may never be fitted."}), settings, attempts)
    if references.source_id != document.source_id:
        _stage_issue("paper_references", attempts, "source_id", "The stage belongs to another uploaded paper.")
    for index, target in enumerate(references.targets):
        if target.case_id != cases.selected_case_id:
            _stage_issue("paper_references", attempts, f"targets.{index}.case_id", "A reference belongs to an unselected operating case.")
    # Preserve every message rather than truncating a combined list to its local
    # bound. Overflow is a visible final validation failure with all checkpoints retained.
    candidate = {"source_id": cases.source_id, "title": cases.title, "summary": cases.summary,
        "parameters": [parameter.source_claim() for parameter in parameters.parameters],
        "targets": [target.source_claim() for target in references.targets],
        "questions": [question for part in (cases, parameters, references) for question in part.questions],
        "limitations": [item for part in (cases, parameters, references) for item in part.limitations]}
    return PaperExtraction(candidate, tuple(attempts), cases)
