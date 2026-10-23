"""Bounded AI planning for the typed magnetic-model harness.

The provider can propose a model, not executable Lua. Every candidate passes the
same geometry/physics validation used by manual plans before it leaves here.
"""
import asyncio
import json
from typing import Literal

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

    @model_validator(mode="before")
    @classmethod
    def explicit_measurements(cls, value):
        if isinstance(value, dict) and isinstance(value.get("spec"), dict):
            missing = {"probes", "force_region"} - value["spec"].keys()
            if missing:
                raise ValueError("Draft model arguments must explicitly include probes (an empty list if none) and force_region (null if none).")
        return value

    @model_validator(mode="after")
    def clarification_or_model(self):
        if any(not q.strip() or len(q) > 600 for q in self.questions):
            raise ValueError("Questions must be nonempty and at most 600 characters each.")
        if self.spec is None and not self.questions:
            raise ValueError("A missing model requires clarification questions.")
        return self


class PlanProposal(PlanDraft):
    attempts: int = Field(ge=0, le=MAX_ATTEMPTS)
    function_name: Literal["create_plan", "modify_plan", "clarify"] = "create_plan"
    decision_trace: list[DecisionTrace] = Field(default_factory=list, max_length=2)
    diagnostic_error: str | None = None


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

The built-in material id 'air' has relative permeability 1. It is reserved: never
add it to materials. The outer domain and its background air are implicit; never
add a region duplicating the outer domain. Define named winding materials with
relative permeability 1 when the request describes windings; material names do
not supply conductivity or other unsupported properties. Other material and circuit
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

Before emitting arguments, check each region's physical values against the request
separately. A material name such as steel does not supply its permeability. A
missing or explicitly unknown material permeability is not 1: return spec=null
and ask for that value, unless the user explicitly authorizes assumptions. Do not
copy the known winding permeability into an unspecified core material.

Keep current and turns separate: current_a is the circuit current I in amperes,
and turns is the signed winding count N. For example, 7 turns carrying 3 A means
current_a=3 and turns=7, never current_a=7 or current_a=21. Opposite cross-sections
share the same current and use opposite signed turns. Do not replace current with
ampere-turns NI. Check each circuit and each region's turn count independently.

Every requested field measurement must be present in probes at its requested
coordinates. Retain unchanged probes during modifications. Verify the emitted
JSON includes all requested regions, physical assignments and measurements; an
explanation saying a value was applied does not substitute for that JSON field.
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
                         decision_threshold: float | None = None) -> PlanProposal:
    """Route a function, validate its arguments, and diagnose only actual errors.

    The legacy threshold argument is accepted for settings compatibility and has
    no role in accepting geometry. At most two drafts, one route and one error
    diagnostic are allowed within the overall 90 second deadline.
    """
    if not key or not model:
        raise PlannerError(503, "AI planning needs OPENROUTER_API_KEY and AI_MODEL configured on the server.")
    if not isinstance(prompt, str) or not 5 <= len(prompt) <= 3000:
        raise PlannerError(422, "Describe the requested model using 5 to 3000 characters.")
    endpoint = base_url.rstrip("/") + "/chat/completions"
    schema = json.dumps(_draft_schema(), ensure_ascii=True)
    try:
        decisions.validate_configuration(decision_model, decision_base_url)
        async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
            async with httpx.AsyncClient(timeout=httpx.Timeout(35, connect=10), follow_redirects=False) as client:
                route = await decisions.route(client, key, decision_base_url, decision_model, prompt, current)
                trace = list(route.trace)
                function_name = route.function_name
                if function_name == "clarify":
                    return PlanProposal(spec=None,
                        explanation="Describe a model to create, or provide a current model and the changes to make.",
                        questions=route.questions or ["What geometry would you like to create or modify?"],
                        attempts=0, function_name=function_name, decision_trace=trace)
                if function_name == "modify_plan" and current is None:
                    return PlanProposal(spec=None, explanation="The modify function requires a current model.",
                        questions=["Load or create the model you want to modify, then describe the changes."],
                        attempts=0, function_name=function_name, decision_trace=trace)
                context = current if function_name == "modify_plan" else None
                messages = [{"role": "system", "content": SYSTEM + "\nResponse JSON schema:\n" + schema},
                            {"role": "user", "content": json.dumps({"function": function_name,
                             "operation": "modify" if context is not None else "create",
                             "current_plan": context.model_dump(mode="json") if context is not None else None,
                             "request": prompt}, ensure_ascii=True)}]
                diagnostic_error = None
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    candidate = await _candidate(client, endpoint, key, model, messages)
                    try:
                        draft = (modify_plan(candidate, context) if function_name == "modify_plan" else create_plan(candidate))
                    except ValidationError as error:
                        if attempt == MAX_ATTEMPTS:
                            raise PlannerError(502, "The AI could not produce a valid geometry after one repair attempt. Simplify the request or edit the plan; the current model has not changed.") from None
                        issues = [{"path": ".".join(str(part) for part in item["loc"]),
                                   "type": item["type"], "message": item["msg"]}
                                  for item in error.errors(include_input=False, include_url=False)[:12]]
                        try:
                            arguments = json.loads(candidate)
                        except ValueError:
                            arguments = {"invalid_json": candidate}
                        feedback = _repair_feedback(error)
                        try:
                            diagnosis = await decisions.diagnose_call(client, key, decision_base_url, decision_model,
                                function_name, {"candidate": arguments, "schema": _draft_schema()}, issues, prompt)
                            trace.extend(diagnosis.trace)
                            feedback += (" The function-argument diagnostic reported "
                                         + ("a match" if diagnosis.arguments_match else "a mismatch")
                                         + "; correct the actual validation errors above and re-run validation.")
                        except decisions.DecisionError as exc:
                            # A diagnostic cannot turn a failed call into success
                            # or prevent a bounded deterministic-error repair.
                            diagnostic_error = exc.message
                        messages.extend([{"role": "assistant", "content": candidate},
                                         {"role": "user", "content": feedback}])
                        continue
                    return PlanProposal(**draft.model_dump(), attempts=attempt, function_name=function_name,
                                        decision_trace=trace, diagnostic_error=diagnostic_error)
    except PlannerError:
        raise
    except decisions.DecisionError as error:
        raise PlannerError(error.status_code, error.message) from None
    except (TimeoutError, httpx.TimeoutException):
        raise PlannerError(504, "The AI request timed out. Try a smaller geometry; the current model has not changed.") from None
    except httpx.HTTPError:
        raise PlannerError(502, "Could not connect to the AI provider. Check the server connection and AI_BASE_URL.") from None
    raise PlannerError(502, "The AI did not produce a geometry plan.")


def create_plan(arguments: str) -> PlanDraft:
    """Build a new typed model; deterministic validators reject invalid calls."""
    return PlanDraft.model_validate_json(arguments)


def _draft_schema() -> dict:
    """Require explicit measurement arguments in AI calls; manual plans keep defaults."""
    schema = PlanDraft.model_json_schema()
    required = schema["$defs"]["GeometryPlan"].setdefault("required", [])
    required.extend(name for name in ("probes", "force_region") if name not in required)
    return schema


def modify_plan(arguments: str, current: GeometryPlan) -> PlanDraft:
    """Validate a complete replacement proposal for an existing typed model."""
    if current is None:
        raise ValueError("modify_plan requires current_plan")
    return PlanDraft.model_validate_json(arguments)
