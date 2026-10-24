"""The model proposes parameters, never executable code."""
import json
import httpx
from pydantic import BaseModel, ConfigDict, Field
from .model import CCoreSpec
from . import structured


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: CCoreSpec
    explanation: str
    questions: list[str] = Field(default_factory=list)


SYSTEM = '''You configure a teaching model of a C-core electromagnetic actuator in FEMM 5.
Return only a JSON object matching the supplied schema. Never write or execute code.
Supported: 2D planar DC C-core, rectangular armature, linear permeability, paired coil cross-sections.
There is a gap at both pole faces. Length inputs are mm, current is A, depth is out-of-plane mm.
Core width/height must each be >= 4*limb_mm. Use the supplied current specification as the starting point.
Preserve parameters not explicitly changed. Convert explicit units correctly.
Explain all changes and assumptions. If geometry/physics/material behavior is unsupported, or the
request is ambiguous, keep a valid starting specification and put clear questions in questions.
Do not silently replace a requested solenoid, 3D model, nonlinear steel, or AC analysis with a C-core.
Even with questions, return a valid spec; users must resolve questions and review before solving.
Do not invent numerical simulation results or claim the model has been solved.
'''


async def interpret(prompt: str, current: CCoreSpec, key: str, base_url: str, model: str) -> Proposal:
    payload = structured.request_payload(model, response_model=Proposal, messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps({"current_spec": current.model_dump(), "request": prompt})},
    ])
    async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=20), follow_redirects=False) as client:
        response = await client.post(base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": "Bearer " + key}, json=payload)
        response.raise_for_status()
        choice = response.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("The provider truncated its parameter proposal.")
        candidate = json.loads(choice["message"]["content"])
        return Proposal.model_validate(structured.validate_response(candidate, Proposal))
