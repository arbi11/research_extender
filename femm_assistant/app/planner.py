"""Bounded AI planning for the typed magnetic-model harness.

The provider can propose a model, not executable Lua. Every candidate passes the
same geometry/physics validation used by manual plans before it leaves here.
"""
import asyncio
import json

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from . import decisions
from .decisions import DecisionTrace
from .plan import GeometryPlan


MAX_PROVIDER_BYTES = 128 * 1024
MAX_MODEL_CHARACTERS = 64 * 1024
MAX_ATTEMPTS = 2
TOTAL_TIMEOUT_SECONDS = 90


class PlannerError(Exception):
    """A safe, actionable error; provider bodies and credentials stay private."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class PlanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: GeometryPlan | None
    explanation: str = Field(min_length=1, max_length=2500)
    questions: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def clarification_or_model(self):
        if any(not q.strip() or len(q) > 600 for q in self.questions):
            raise ValueError("Questions must be nonempty and at most 600 characters each.")
        if self.spec is None and not self.questions:
            raise ValueError("A missing model requires clarification questions.")
        return self


class PlanProposal(PlanDraft):
    attempts: int = Field(ge=0, le=MAX_ATTEMPTS)
    decision_trace: list[DecisionTrace] = Field(default_factory=list, max_length=4)


SYSTEM = """You plan finite-element magnetic models for a FEMM 5 teaching tool.
Return only one JSON object matching the supplied response schema. The application
compiles and executes your validated plan. Never return Lua, Python, shell commands,
file paths, a function body, or invented numerical simulation results.

Supported physics: 2D PLANAR, DC magnetostatics, millimeters, linear isotropic
relative permeability. Model depth is out of plane. The outer domain has A=0 and
background air. Supported closed contours: exact circles, rectangles, and simple
polygons. The same schema supports NEW TOPOLOGIES; you are not restricted to a
C-core or any template. Coordinates can be positive or negative.
Convert explicit physical units to millimeters and amperes before returning the
plan, and explain any unit conversions. Do not interpret a diameter as a radius.

Workflow: define the physical problem; draw closed contours; define linear
materials and circuits; assign every face using a label; define the outer domain;
choose bounded meshing; specify optional measurements. Mesh and solve happen in
the application after validation, not in this response.

The built-in material id 'air' has relative permeability 1. Material and circuit
ids must be defined before reference. A circuit is DC current in amperes; a region
assigned to a circuit needs a nonzero signed integer turns count. Opposite coil
cross-sections use the same circuit with opposite turns. A circular wire carrying
current perpendicular to the plane is supported. Axisymmetric solenoids, full 3D,
AC, saturation, nonlinear B-H curves, permanent magnets and other physics are not
supported in this version: ask clear questions with spec=null rather than silently
substituting a C-core or a different physical problem.

Contours must not cross, overlap at edges, or touch. Nested contours are allowed.
Each region label is strictly inside its own contour and outside every contour
nested inside it. The background air label must be strictly inside the outer
domain and outside all region contours. A probe is strictly inside the domain and
away from contour edges. The outer domain must strictly enclose all regions.
Keep contour/node counts and mesh density modest for a 4 GB computer.

For a modification, use current_plan as the starting point and preserve every
physical property not requested to change. For creation, current_plan is null;
construct the requested geometry from scratch. Do not invent missing physical
dimensions, out-of-plane depth, material permeability or excitation. Ask concise
clarification questions when these are required. If the user explicitly permits
reasonable assumptions, state the assumed physical values in the explanation.
You may propose a surrounding air domain and mesh size as numerical choices,
explaining their finite-boundary and discretization limitations. For an ambiguous
or unsupported request return spec=null, an explanation, and questions. A valid
candidate with outstanding questions is allowed, but the user must resolve them
before solving. Never claim that a proposed model is an accurate engineering
result, already meshed, or already solved.
"""


def _provider_error(status: int) -> PlannerError:
    if status in (401, 403):
        return PlannerError(502, "OpenRouter rejected the API key or model access. Check the key and permitted models on the server.")
    if status == 402:
        return PlannerError(502, "OpenRouter has insufficient credit for this request. Check the account balance and API-key spending limit.")
    if status == 429:
        return PlannerError(429, "OpenRouter is rate limiting requests. Wait briefly before trying again.")
    if status in (400, 404, 422):
        return PlannerError(502, "OpenRouter rejected this model or request format. Choose a model supporting JSON-object responses and check its model id.")
    return PlannerError(502, "The AI provider is unavailable. Try again later; the current model has not been changed.")


async def _candidate(client, endpoint: str, key: str, model: str, messages: list[dict]) -> str:
    payload = {"model": model, "temperature": 0, "max_tokens": 6000,
               "messages": messages, "response_format": {"type": "json_object"}}
    async with client.stream("POST", endpoint, headers={"Authorization": "Bearer " + key}, json=payload) as response:
        if response.status_code >= 400:
            raise _provider_error(response.status_code)
        body = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=8192):
            body.extend(chunk)
            if len(body) > MAX_PROVIDER_BYTES:
                raise PlannerError(502, "The AI provider returned too much data. Request a smaller geometry.")
    try:
        packet = json.loads(body)
        content = packet["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Missing model content")
        if len(content) > MAX_MODEL_CHARACTERS:
            raise PlannerError(502, "The AI geometry plan is too large. Request fewer regions or polygon vertices.")
        return content
    except PlannerError:
        raise
    except (ValueError, TypeError, KeyError, IndexError):
        raise PlannerError(502, "The AI provider returned an unreadable response. Try again or choose another JSON-capable model.") from None


def _repair_feedback(error: ValidationError) -> str:
    # Input values and URLs are excluded. This feedback goes only to the same
    # provider; raw model output and validation details are never client errors.
    issues = [{"path": ".".join(str(part) for part in item["loc"]), "message": item["msg"]}
              for item in error.errors(include_input=False, include_url=False)[:12]]
    feedback = json.dumps(issues, ensure_ascii=True)[:6000]
    return ("The candidate failed the harness validation. Correct these issues and return one complete JSON object: "
            + feedback + ". Preserve the user's requested physics and dimensions. If a physically valid supported model "
            "cannot be created from the supplied information, return spec=null with clarification questions. "
            "Reduce unnecessary polygon vertices if the previous answer was truncated.")


async def interpret_plan(prompt: str, current: GeometryPlan | None, key: str,
                         base_url: str, model: str, *,
                         decision_model: str = decisions.DEFAULT_MODEL,
                         decision_base_url: str = decisions.DEFAULT_BASE_URL,
                         decision_threshold: float = decisions.DEFAULT_THRESHOLD) -> PlanProposal:
    """At most two generations plus two semantic checks, within 90 seconds.

    Jev gates bounded semantic judgments only. Every actual candidate must still
    pass the full deterministic schema, topology, references and resource limits.
    The threshold is provisional until evaluated on labeled CAD requests.
    """
    if not key or not model:
        raise PlannerError(503, "AI planning needs OPENROUTER_API_KEY and AI_MODEL configured on the server.")
    if not isinstance(prompt, str) or not 5 <= len(prompt) <= 3000:
        raise PlannerError(422, "Describe the requested model using 5 to 3000 characters.")
    endpoint = base_url.rstrip("/") + "/chat/completions"
    schema = json.dumps(PlanDraft.model_json_schema(), ensure_ascii=True)
    messages = [{"role": "system", "content": SYSTEM + "\nResponse JSON schema:\n" + schema},
                {"role": "user", "content": json.dumps({"operation": "modify" if current else "create",
                 "current_plan": current.model_dump(mode="json") if current else None, "request": prompt}, ensure_ascii=True)}]
    try:
        decisions.validate_configuration(decision_model, decision_base_url, decision_threshold)
        async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
            async with httpx.AsyncClient(timeout=httpx.Timeout(35, connect=10), follow_redirects=False) as client:
                preflight = await decisions.preflight(client, key, decision_base_url, decision_model,
                                                      decision_threshold, prompt, current)
                trace = list(preflight.trace)
                if not preflight.allowed:
                    return PlanProposal(spec=None,
                        explanation="The semantic decision checks need clarification before a geometry can be proposed.",
                        questions=preflight.questions, attempts=0, decision_trace=trace)
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    candidate = await _candidate(client, endpoint, key, model, messages)
                    try:
                        draft = PlanDraft.model_validate_json(candidate)
                    except ValidationError as error:
                        if attempt == MAX_ATTEMPTS:
                            raise PlannerError(502, "The AI could not produce a valid geometry after one repair attempt. Simplify the request or edit the plan; the current model has not changed.") from None
                        messages.extend([{"role": "assistant", "content": candidate},
                                         {"role": "user", "content": _repair_feedback(error)}])
                        continue
                    if draft.spec is not None:
                        verification = await decisions.fidelity(client, key, decision_base_url, decision_model,
                            decision_threshold, prompt, current, draft.spec, draft.explanation)
                        trace.extend(verification.trace)
                        if not verification.allowed:
                            return PlanProposal(spec=None,
                                explanation="The candidate passed geometry validation, but the semantic fidelity check needs clarification. No candidate has been applied.",
                                questions=verification.questions, attempts=attempt, decision_trace=trace)
                    return PlanProposal(**draft.model_dump(), attempts=attempt, decision_trace=trace)
    except PlannerError:
        raise
    except decisions.DecisionError as error:
        raise PlannerError(error.status_code, error.message) from None
    except (TimeoutError, httpx.TimeoutException):
        raise PlannerError(504, "The AI request timed out. Try a smaller geometry; the current model has not changed.") from None
    except httpx.HTTPError:
        raise PlannerError(502, "Could not connect to the AI provider. Check the server connection and AI_BASE_URL.") from None
    raise PlannerError(502, "The AI did not produce a geometry plan.")
