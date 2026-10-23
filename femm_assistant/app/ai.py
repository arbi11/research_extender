"""The model proposes parameters, never executable code."""
import json
import httpx
from pydantic import BaseModel, ConfigDict, Field
from .model import CCoreSpec


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: CCoreSpec
    explanation: str = Field(max_length=2500)
    questions: list[str] = Field(default_factory=list, max_length=8)


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
    schema = Proposal.model_json_schema()
    async with httpx.AsyncClient(timeout=40) as client:
        response = await client.post(base_url.rstrip("/") + "/chat/completions", headers={"Authorization": "Bearer " + key}, json={
            "model": model, "temperature": 0, "max_tokens": 1500,
            "messages": [{"role": "system", "content": SYSTEM + "\nJSON schema: " + json.dumps(schema)},
                         {"role": "user", "content": json.dumps({"current_spec": current.model_dump(), "request": prompt})}],
            "response_format": {"type": "json_object"},
        })
        response.raise_for_status()
        return Proposal.model_validate_json(response.json()["choices"][0]["message"]["content"])
