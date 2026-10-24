"""Live OpenRouter -> production API -> native Docker FEMM acceptance checks.

Run this module inside the development FEMM image. It creates its own temporary
database and jobs, never connects to the deployed service, and needs --live.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import zipfile
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import Settings, create_app
from app.plan import GeometryPlan
from app import planner
from .semantic import load_cases, save_report


class CheckFailed(Exception):
    pass


def require(condition, message):
    if not condition:
        raise CheckFailed(message)


def physical_wire(spec, current):
    """Independent assertions on the actual draft, beyond Jev's evidence."""
    plan = GeometryPlan.model_validate(spec)
    require(len(plan.regions) == 1, "Expected exactly one conductor region")
    region = plan.regions[0]
    shape = region.shape
    require(shape.type == "circle" and shape.center.x == shape.center.y == 0 and shape.radius == 1,
            "Conductor geometry differs from the requested centered 1 mm radius")
    require(plan.depth_mm == 20 and region.turns == 1, "Depth or turn direction differs from the request")
    material = next((item for item in plan.materials if item.id == region.material), None)
    require(material is not None and material.relative_permeability == 1, "Requested conductor permeability was changed")
    circuit = next((item for item in plan.circuits if item.id == region.circuit), None)
    require(circuit is not None and circuit.current_a == current, "Requested current was changed")
    require(plan.domain.type == "circle" and plan.domain.radius == 10
            and plan.domain.center.x == plan.domain.center.y == 0, "Requested air domain was changed")
    require(plan.mesh_mm == .5 and plan.air_label.x == 5 and plan.air_label.y == 0, "Requested numerical settings were changed")
    require(len(plan.probes) == 1 and plan.probes[0].x == 2 and plan.probes[0].y == 0,
            "Requested probe was changed")
    return plan


def wait_job(client, job_id, headers, timeout=240):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get("/api/jobs/" + job_id, headers=headers)
        require(response.status_code == 200, "Could not read temporary job")
        job = response.json()
        if job["status"] in {"completed", "failed"}:
            require(job["status"] == "completed", job.get("error") or "Native job failed")
            return job
        time.sleep(.1)
    raise CheckFailed("Temporary native job did not finish within the evaluation deadline")


def mesh_solve(client, spec, headers):
    # Exercise the actual HTTP gate, worker, mesh diagnostics and export path.
    response = client.post("/api/plan/mesh", json=spec, headers=headers)
    require(response.status_code == 202, "Native mesh submission failed")
    mesh = wait_job(client, response.json()["id"], headers)
    require(mesh["result"]["status"] == "passed", "Native mesh report did not pass")
    response = client.post("/api/plan/jobs", json={"spec": spec, "mesh_job_id": mesh["id"]}, headers=headers)
    require(response.status_code == 202, "Mesh-gated solve submission failed")
    solve = wait_job(client, response.json()["id"], headers)
    download = client.get("/api/jobs/" + solve["id"] + "/download.zip", headers=headers)
    require(download.status_code == 200, "Solved model download failed")
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        require({"model.py", "model.lua", "model.fem", "model.ans", "mesh-report.json", "results.json"}
                <= set(archive.namelist()), "Native export is missing required artifacts")
        require(json.loads(archive.read("mesh-report.json"))["status"] == "passed", "Exported mesh report failed")
    result = solve["result"]
    measured = math.hypot(result["bx_0"]["value"], result["by_0"]["value"])
    plan = GeometryPlan.model_validate(spec)
    current = abs(plan.circuits[0].current_a)
    expected = 2e-7 * current / .002
    require(abs(measured / expected - 1) <= .08, "Generated wire field differs from Ampere's law by more than 8%")
    require(abs(result["bx_0"]["value"]) <= expected * .05 and result["by_0"]["value"] > 0,
            "Generated wire field has the wrong direction")
    require(result["energy"]["value"] > 0, "Generated wire has nonpositive energy")
    return {"mesh": mesh["result"], "quantities": result, "analytic_probe_t": expected,
            "probe_relative_error": abs(measured / expected - 1), "export_verified": True}


def run(settings, checkpoint, names):
    cases = {case.id: case for case in load_cases()}
    rows = []
    headers = {"Authorization": "Bearer " + settings.token}
    current = None
    with TestClient(create_app(settings)) as client:
        for name in names:
            row = {"id": name}
            try:
                case = cases[name]
                body = {"prompt": case.prompt}
                if name == "modify_current":
                    require(current is not None, "Creation did not produce a model to modify")
                    # Keep the benchmark's current direction fixed for scaling.
                    body.update(prompt="Double only the current to +2 A. Preserve all other physical properties, geometry, probes and mesh settings.", current=current)
                drafts = []
                original_candidate = planner._candidate
                async def record_candidate(*arguments, **kwargs):
                    # Instrument real provider output for these public prompts;
                    # no request headers, environment or API key are recorded.
                    content = await original_candidate(*arguments, **kwargs)
                    try:
                        drafts.append(json.loads(content))
                    except ValueError:
                        drafts.append({"invalid_json": True})
                    return content
                with patch.object(planner, "_candidate", record_candidate):
                    response = client.post("/api/plan/interpret", json=body, headers=headers)
                row["provider_drafts"] = drafts
                require(response.status_code == 200, f"Live planner returned HTTP {response.status_code}: " + str(response.json().get("detail", "request failed")))
                proposal = response.json()
                row["proposal"] = proposal
                if name in {"wire_missing_depth", "wire_ac"}:
                    require(proposal["spec"] is None and proposal["questions"],
                            "Unsupported or incomplete request did not request clarification")
                else:
                    require(proposal["spec"] is not None and not proposal["questions"], "Complete wire request was not accepted")
                    current_a = 1 if name == "wire_complete" else 2
                    physical_wire(proposal["spec"], current_a)
                    if current is not None:
                        before, after = dict(current), dict(proposal["spec"])
                        before.pop("title"); after.pop("title")
                        before["circuits"] = after["circuits"]
                        require(before == after, "Modification changed an unrequested model property")
                    row["execution"] = mesh_solve(client, proposal["spec"], headers)
                    current = proposal["spec"]
                row["status"] = "passed"
            except CheckFailed as exc:
                row.update(status="failed", error=str(exc))
            rows.append(row)
            checkpoint(rows)
            print(name + ": " + row["status"], flush=True)
    if len(rows) >= 2 and [row["id"] for row in rows[:2]] == ["wire_complete", "modify_current"] and all(row["status"] == "passed" for row in rows[:2]):
        energy = [row["execution"]["quantities"]["energy"]["value"] for row in rows[:2]]
        ratio = energy[1] / energy[0]
        rows[1]["energy_scaling_ratio"] = ratio
        if abs(ratio / 4 - 1) > .08:
            rows[1].update(status="failed", error="Double-current energy did not scale quadratically within 8%")
    checkpoint(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--create-only", action="store_true", help="Diagnose only the complete creation request")
    args = parser.parse_args()
    settings = Settings.from_env()
    if not settings.ai_key:
        parser.error("Set OPENROUTER_API_KEY in the environment")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write("{}\n")
    report = {"schema_version": 1, "source": "live_openrouter_and_native_femm",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "drafting_model": settings.ai_model, "decision_model": settings.decision_model,
              "decision_role": "function_routing_and_error_diagnostics",
              "limits": "4 requests; at most 8 routing/error checks and 8 drafts, production time/token bounds apply",
              "reference": "https://openstax.org/books/university-physics-volume-2/pages/12-5-amperes-law"}
    def checkpoint(rows):
        report["rows"] = rows
        report["passed"] = sum(row["status"] == "passed" for row in rows)
        save_report(args.output, report)
    with tempfile.TemporaryDirectory(prefix="femm-cad-evaluation-") as directory:
        settings.data_dir = Path(directory)
        names = ("wire_complete",) if args.create_only else ("wire_complete", "modify_current", "wire_missing_depth", "wire_ac")
        rows = run(settings, checkpoint, names)
    return int(any(row["status"] != "passed" for row in rows))


if __name__ == "__main__":
    sys.exit(main())
