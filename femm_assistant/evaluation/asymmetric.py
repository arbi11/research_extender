"""Live regression for the user-tested concave polygon and signed windings."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

from fastapi.testclient import TestClient

from app.main import Settings, create_app
from app.plan import GeometryPlan
from .cad import CheckFailed, require, wait_job
from .semantic import load_cases, save_report


def physical_signature(data):
    """Compare requested geometry and physical values independently of names."""
    plan = GeometryPlan.model_validate(data)
    materials = {"air": 1.0, **{item.id: item.relative_permeability for item in plan.materials}}
    currents = {item.id: item.current_a for item in plan.circuits}
    regions = [{"shape": region.shape.model_dump(mode="json"), "label": region.label.model_dump(mode="json"),
                "permeability": materials[region.material], "current_a": currents.get(region.circuit),
                "turns": region.turns} for region in plan.regions]
    return {"physics": plan.physics, "units": plan.units, "depth_mm": plan.depth_mm,
            "mesh_mm": plan.mesh_mm, "domain": plan.domain.model_dump(mode="json"),
            "air_label": plan.air_label.model_dump(mode="json"), "circuits": sorted(currents.values()),
            "regions": sorted(regions, key=lambda item: json.dumps(item["shape"], sort_keys=True)),
            "probes": [point.model_dump(mode="json") for point in plan.probes], "force_region": plan.force_region}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = Settings.from_env()
    if not settings.ai_key:
        parser.error("Set OPENROUTER_API_KEY in the environment")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write("{}\n")
    case = next(case for case in load_cases() if case.id == "asymmetric_core_faithful")
    expected = case.candidate.model_dump(mode="json")
    report = {"source": "live_openrouter_and_native_femm", "decision_role": "function_routing_and_error_diagnostics",
              "drafting_model": settings.ai_model, "decision_model": settings.decision_model,
              "prompt": case.prompt, "rows": []}
    headers = {"Authorization": "Bearer " + settings.token}
    current = None
    with tempfile.TemporaryDirectory(prefix="femm-asymmetric-") as directory:
        settings.data_dir = Path(directory)
        with TestClient(create_app(settings)) as client:
            for stage in ("create", "extend_upper_arm"):
                row = {"stage": stage}
                try:
                    if stage == "create":
                        body = {"prompt": case.prompt}
                    else:
                        require(current is not None, "Creation did not produce an approved model to modify")
                        expected = copy.deepcopy(expected)
                        points = next(region for region in expected["regions"] if region["shape"]["type"] == "polygon")["shape"]["points"]
                        points[5]["x"] = points[6]["x"] = 14.0
                        body = {"prompt": "Extend the upper arm to x=14 mm by replacing vertices (6,6) and (6,12) with (14,6) and (14,12). Preserve everything else.", "current": current}
                    response = client.post("/api/plan/interpret", json=body, headers=headers)
                    row["proposal"] = response.json()
                    require(response.status_code == 200, "Planner returned HTTP " + str(response.status_code))
                    proposal = row["proposal"]
                    require(proposal["spec"] is not None and not proposal["questions"], "Planner did not return the requested model")
                    require(physical_signature(proposal["spec"]) == physical_signature(expected),
                            "Generated geometry or physical assignments differ from the explicit request")
                    row["physical_values_verified"] = True
                    mesh = client.post("/api/plan/mesh", json=proposal["spec"], headers=headers)
                    require(mesh.status_code == 202, "Native mesh submission failed")
                    mesh_job = wait_job(client, mesh.json()["id"], headers)
                    require(mesh_job["result"]["status"] == "passed", "Native mesh did not pass")
                    solve = client.post("/api/plan/jobs", json={"spec": proposal["spec"], "mesh_job_id": mesh_job["id"]}, headers=headers)
                    require(solve.status_code == 202, "Mesh-gated solve submission failed")
                    solved = wait_job(client, solve.json()["id"], headers)
                    exported = client.get("/api/jobs/" + solved["id"] + "/download.zip", headers=headers)
                    require(exported.status_code == 200, "Solved native export failed")
                    archive = args.output.with_name(args.output.stem + "-" + stage + ".zip")
                    archive.write_bytes(exported.content)
                    row.update(status="passed", mesh=mesh_job["result"], quantities=solved["result"], archive=archive.name)
                    current = proposal["spec"]
                except CheckFailed as exc:
                    row.update(status="failed", error=str(exc))
                report["rows"].append(row)
                save_report(args.output, report)
                print(stage + ": " + row["status"], flush=True)
    return int(any(row["status"] != "passed" for row in report["rows"]))


if __name__ == "__main__":
    sys.exit(main())
