"""Opt-in PDF -> OpenRouter -> typed model -> native FEMM reproduction trial.

This is one attempted paper case, without a whole-case retry. The independent
gold fixture is used only after model generation for auditing; its physical
numbers are never put into the extraction or geometry-generation prompts.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

import httpx

from app.main import Settings, create_app
from app import papers
from app.plan import GeometryPlan
from app.targets import NumericalTarget, convert_units
from evaluation.replication import load_fixture


EXTRACTION_INSTRUCTIONS = (
    "Extract the paper's centred iron-core finite-element reproduction case. "
    "Keep the finite-element material and references separate from the infinite-permeability "
    "semi-analytical model and the air-core case. Extract the dimensions, winding turns, "
    "excitation/current-density conversion, boundary treatment and the published centred "
    "inductance reference with precise citations. Preserve missing information as questions."
)

PLANNING_INSTRUCTIONS = (
    "Create the selected centred iron-core finite-element configuration from the cited paper evidence. "
    "Use its published finite relative permeability, full axisymmetric winding cross-section, "
    "and number of turns. Derive circuit current from published current density times winding "
    "cross-sectional area divided by turns; state this derivation. Retain the cited target "
    "circuit id when possible. Do not substitute the analytical infinite-permeability case. "
    "Translate the axial origin to the winding centre at z=0; preserve the physical dimensions. "
    "Use these explicit numerical choices: an outer rectangle r=0 to 750 mm, z=-1000 to 1000 mm, "
    "A=0 on its boundary, global mesh 25 mm. Add an internal air-only rectangle r=0 to 200 mm, "
    "z=-250 to 250 mm with mesh 2 mm; it is a mesh-control face with no added physical material. "
    "Use mesh 1 mm in the core and winding. Put labels inside their own faces. "
    "We need the selected paper inductance target from circuit flux linkage/current. "
    "Return a typed model, not code or simulated results. Ask about unknown source information "
    "instead of inventing physical parameters; do not mesh or solve in this planning response."
)


def select_centred_target(evidence: dict) -> NumericalTarget:
    """Select the paper's FE reference; never substitute its analytical result."""
    candidates = []
    expected = next(case for case in load_fixture()["cases"] if case["id"] == "iron_center")["target"]
    expected_h = convert_units(expected["value"], expected["unit"], "H")
    for value in evidence.get("targets", []):
        target = NumericalTarget.model_validate(value)
        if (target.reference_kind == "reported_numerical"
                and target.observable.quantity == "inductance"
                and math.isclose(convert_units(target.value, target.unit, "H"), expected_h, rel_tol=1e-9)):
            candidates.append(target)
    if len(candidates) != 1:
        raise ValueError("The verified extraction does not identify one unambiguous centred finite-element inductance reference. No model generation was requested.")
    return candidates[0]


def _rectangle(shape, expected) -> bool:
    """Accept reordered/reversed/collinearly split rectangle polygons too."""
    if shape.type == "rectangle":
        actual = (shape.x_min, shape.x_max, shape.y_min, shape.y_max)
        return all(math.isclose(a, b, abs_tol=1e-7) for a, b in zip(actual, expected))
    if shape.type != "polygon":
        return False
    points = [(point.x, point.y) for point in shape.points]
    xmin, xmax, ymin, ymax = expected
    if not all(xmin-1e-7 <= x <= xmax+1e-7 and ymin-1e-7 <= y <= ymax+1e-7
               and (math.isclose(x, xmin, abs_tol=1e-7) or math.isclose(x, xmax, abs_tol=1e-7)
                    or math.isclose(y, ymin, abs_tol=1e-7) or math.isclose(y, ymax, abs_tol=1e-7))
               for x, y in points):
        return False
    area = abs(sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(points, points[1:]+points[:1]))) / 2
    return math.isclose(area, (xmax-xmin)*(ymax-ymin), rel_tol=1e-9, abs_tol=1e-7)


def audit_generated_plan(plan: GeometryPlan) -> dict:
    """Independent factual comparison after generation; no identifiers assumed."""
    fixture = load_fixture()
    parameters = {name: entry["value"] for name, entry in fixture["parameters"].items()}
    mu = {"air": 1, **{material.id: material.relative_permeability for material in plan.materials}}
    circuits = {circuit.id: circuit.current_a for circuit in plan.circuits}
    checks = {"axisymmetric": plan.physics == "axisymmetric_dc_magnetics",
              "full_revolved_volume": plan.depth_mm is None,
              "millimeters": plan.units == "millimeters",
              "declared_outer_domain": _rectangle(plan.domain, (0, 750, -1000, 1000)),
              "declared_global_mesh": math.isclose(plan.mesh_mm, 25),
              "single_excitation_circuit": len(plan.circuits) == 1}
    expected_coil = (parameters["coil_inner_radius_mm"], parameters["coil_outer_radius_mm"],
                     parameters["coil_z_min_mm"], parameters["coil_z_max_mm"])
    expected_core = (0, parameters["core_radius_mm"], -parameters["core_half_length_mm"], parameters["core_half_length_mm"])
    winding = [region for region in plan.regions if _rectangle(region.shape, expected_coil)]
    core = [region for region in plan.regions if _rectangle(region.shape, expected_core)]
    air = [region for region in plan.regions if _rectangle(region.shape, (0, 200, -250, 250))]
    checks.update(published_winding_geometry=len(winding) == 1,
                  published_core_geometry_and_centring=len(core) == 1,
                  declared_air_mesh_control=len(air) == 1,
                  no_added_physical_regions=len(plan.regions) == 3)
    circuit_id = None
    if len(winding) == 1:
        region = winding[0]
        circuit_id = region.circuit
        checks.update(winding_relative_permeability=math.isclose(mu[region.material], 1),
                      published_turn_count=region.turns == parameters["turns"],
                      derived_current=region.circuit in circuits and math.isclose(circuits[region.circuit], parameters["current_a"], rel_tol=1e-9),
                      declared_winding_mesh=region.mesh_mm is not None and math.isclose(region.mesh_mm, 1))
    if len(core) == 1:
        region = core[0]
        checks.update(published_fe_permeability=math.isclose(mu[region.material], parameters["relative_permeability"]),
                      no_core_excitation=region.circuit is None,
                      declared_core_mesh=region.mesh_mm is not None and math.isclose(region.mesh_mm, 1))
    if len(air) == 1:
        region = air[0]
        checks.update(air_control_is_physically_air=math.isclose(mu[region.material], 1) and region.circuit is None,
                      declared_near_air_mesh=region.mesh_mm is not None and math.isclose(region.mesh_mm, 2))
    return {"passed": all(checks.values()), "checks": checks, "circuit_id": circuit_id,
            "verification_source": "Independent source fixture checked only after generation; region, material and circuit ids are irrelevant.",
            "source_doi": fixture["source"]["doi"]}


def _safe(value, secrets):
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {_safe(str(key), secrets): _safe(item, secrets) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item, secrets) for item in value]
    return value


def verified_resume_evidence(report_path: Path, source: bytes) -> dict:
    """Verify a real prior extraction and its source before spending again."""
    if report_path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("The prior paper report exceeds the resume size limit.")
    raw = report_path.read_bytes()
    previous = json.loads(raw)
    digest = hashlib.sha256(source).hexdigest()
    if previous.get("status") != "complete" or previous.get("passed") is not False:
        raise ValueError("Evidence continuation requires a completed, unsuccessful prior trial.")
    if previous.get("pdf_sha256") != digest:
        raise ValueError("The supplied PDF differs from the prior trial's source digest.")
    uploads = [step for step in previous.get("steps", []) if step.get("name") == "upload_pdf" and step.get("status_code") == 201]
    extractions = [step for step in previous.get("steps", []) if step.get("name") in {"extract_evidence", "reuse_verified_evidence"}
                   and step.get("status_code") in (200, 201)]
    proposals = [step for step in previous.get("steps", []) if step.get("name") == "create_model_from_evidence"]
    if len(uploads) != 1 or len(extractions) != 1 or len(proposals) != 1:
        raise ValueError("The prior report must contain one successful PDF upload/extraction and one attempted model proposal.")
    uploaded = uploads[0]["response"]
    extracted = extractions[0]["response"]
    document = papers.PaperDocument.model_validate(uploaded["document"])
    if uploaded["paper_id"] != digest or extracted.get("paper_id") != digest or document.sha256 != digest:
        raise ValueError("The prior extracted evidence does not belong to this PDF.")
    identity = extracted.get("evidence_id")
    record = {key: value for key, value in extracted.items() if key != "evidence_id"}
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    if not isinstance(identity, str) or hashlib.sha256(encoded).hexdigest() != identity:
        raise ValueError("The prior extraction record no longer matches its content hash.")
    if not isinstance(extracted.get("model"), str) or extracted["model"] in {"", "manual"}:
        raise ValueError("Resume requires the original AI extraction model and usage provenance.")
    evidence = papers.validate_evidence(document, extracted["evidence"])
    selected = select_centred_target(evidence.model_dump(mode="json"))
    previous_target = NumericalTarget.model_validate(previous["selected_original_target"])
    if selected != previous_target:
        raise ValueError("The selected paper target changed after the prior failed proposal.")
    return {"report_sha256": hashlib.sha256(raw).hexdigest(), "previous_error": previous.get("error"),
            "record": extracted, "document": document,
            "previous_proposal": proposals[0].get("response")}


async def run_paper_live(pdf_path: Path, output: Path, *, live=False, model=None, settings=None,
                       resume_evidence: Path | None = None) -> dict:
    if not live:
        raise ValueError("This runner makes paid OpenRouter requests. Pass --live explicitly to opt in.")
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory to preserve previous responses and failures.")
    output.mkdir(parents=True, exist_ok=True)
    original = settings or Settings.from_env()
    settings = replace(original, data_dir=output / "data", start_worker=False,
                       ai_model=model or original.ai_model)
    if not settings.ai_key:
        raise ValueError("Configure the existing OpenRouter key in the environment before a live trial.")
    secrets = [settings.ai_key, settings.token]
    source = pdf_path.read_bytes()
    resumed = verified_resume_evidence(resume_evidence, source) if resume_evidence is not None else None
    app = create_app(settings)
    headers = {"Authorization": "Bearer " + settings.token} if settings.token else {}
    report = {"status": "running", "passed": False, "requested_model": settings.ai_model,
        "requested_decision_model": settings.decision_model,
        "pdf_sha256": hashlib.sha256(source).hexdigest(), "pdf_size_bytes": len(source),
        "attempt_policy": "One paper extraction and one plan request; no whole-case retry. The normal planner may diagnose/repair an invalid function call within its own bound.",
        "usage": {"extraction": None, "planning": None,
                  "planning_usage_note": "The planner does not expose provider usage. Missing usage is unknown, not zero cost."},
        "steps": []}
    if resumed is not None:
        old_proposal = resumed["previous_proposal"] or {}
        old_spec = old_proposal.get("spec") or {}
        report["continuation"] = {"source_report_sha256": resumed["report_sha256"],
            "reason": "Continue model planning after the retained failed proposal; reuse the unchanged verified AI extraction without another PDF extraction request.",
            "source_evidence_id": resumed["record"]["evidence_id"], "source_extraction_model": resumed["record"]["model"],
            "historical_extraction_usage": resumed["record"].get("usage", {}),
            "previous_failure": resumed["previous_error"],
            "previous_circuit_arguments": old_spec.get("circuits", []),
            "previous_questions": old_proposal.get("questions", [])}
        report["attempt_policy"] = "Resume unchanged verified evidence; zero new extraction requests and one bounded normal plan request, followed by independent source checks and native gates if it succeeds."
        report["usage"]["extraction_usage_note"] = "No new extraction was requested. Historical extraction usage is listed separately in continuation and is not charged again."
    started = time.monotonic()

    def persist():
        (output / "report.json").write_text(json.dumps(_safe(report, secrets), indent=2, allow_nan=False) + "\n")

    async def request(client, name, method, path, *, json_body=None, content=None, extra_headers=None):
        kwargs = {"headers": {**headers, **(extra_headers or {})}}
        if json_body is not None:
            kwargs["json"] = json_body
        if content is not None:
            kwargs["content"] = content
        response = await client.request(method, path, **kwargs)
        try:
            body = response.json()
        except ValueError:
            body = {"text": response.text[:64 * 1024]}
        step = {"name": name, "method": method, "path": path, "status_code": response.status_code,
                "request": json_body if json_body is not None else ({"pdf_sha256": report["pdf_sha256"], "bytes": len(content)} if content is not None else None),
                "response": body}
        report["steps"].append(step)
        persist()
        if response.status_code >= 300:
            raise ValueError(name + " failed; its response is retained in the report.")
        return body

    persist()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://paper-trial", timeout=None) as client:
            uploaded = await request(client, "upload_pdf", "POST", "/api/papers", content=source,
                extra_headers={"Content-Type": "application/pdf", "X-Filename": pdf_path.name})
            if resumed is None:
                extracted = await request(client, "extract_evidence", "POST", f'/api/papers/{uploaded["paper_id"]}/extract',
                    json_body={"instructions": EXTRACTION_INSTRUCTIONS})
                report["usage"]["extraction"] = extracted.get("usage", {})
            else:
                actual_document = papers.PaperDocument.model_validate(uploaded["document"])
                if actual_document.pages != resumed["document"].pages:
                    raise ValueError("The actual PDF extraction differs from the source pages in the prior verified report.")
                old = resumed["record"]
                evidence = papers.validate_evidence(actual_document, old["evidence"])
                metadata = {key: old[key] for key in ("extraction_status", "validation_summary", "audit_id", "rejected_claim_count") if key in old}
                extracted = app.state.replication.put_evidence(uploaded["paper_id"], evidence,
                    model=old["model"], usage=old.get("usage", {}), **metadata)
                if extracted["evidence_id"] != old["evidence_id"]:
                    raise ValueError("Replayed evidence changed its immutable extraction record.")
                report["steps"].append({"name": "reuse_verified_evidence", "method": "LOCAL_REPLAY",
                    "path": "private_content_addressed_evidence", "status_code": 201,
                    "source_report_sha256": resumed["report_sha256"], "response": extracted})
                persist()
            selected = select_centred_target(extracted["evidence"])
            report["selected_original_target"] = selected.model_dump(mode="json")
            persist()
            proposal = await request(client, "create_model_from_evidence", "POST", "/api/plan/interpret",
                json_body={"prompt": PLANNING_INSTRUCTIONS, "evidence_id": extracted["evidence_id"],
                           "case_id": selected.case_id, "preserve_paper_excitation": True})
            if proposal.get("questions") or proposal.get("spec") is None:
                raise ValueError("The model proposal requires clarification. Questions are retained; no mesh or solve was requested.")
            plan = GeometryPlan.model_validate(proposal["spec"])
            audit = audit_generated_plan(plan)
            report["model_fidelity"] = audit
            persist()
            if not audit["passed"]:
                raise ValueError("The generated model does not match the paper or declared numerical choices. No gold values were sent back as a repair prompt, and no native solve was requested.")
            revision = await request(client, "record_revision", "POST", "/api/revisions", json_body={
                "spec": plan.model_dump(mode="json"), "branch": "paper-reproduction",
                "planning_id": proposal["planning_id"], "evidence_id": extracted["evidence_id"],
                "case_id": selected.case_id, "preserve_paper_excitation": True,
                "evidence": [{"source": "paper", "reference": extracted["evidence_id"],
                              "sha256": extracted["evidence_id"], "detail": "Verified uploaded-paper extraction; source target and physical conditions retained."}],
                "reason": "One attempted paper-to-model reproduction with declared numerical choices."})
            meshed = await request(client, "queue_mesh", "POST", "/api/plan/mesh", json_body=plan.model_dump(mode="json"),
                extra_headers={"X-Model-Revision": revision["id"]})
            await asyncio.to_thread(app.state.worker.run, app.state.store.get(meshed["id"]))
            mesh_job = await request(client, "native_mesh_result", "GET", "/api/jobs/" + meshed["id"])
            if mesh_job["status"] != "completed" or mesh_job["result"].get("status") != "passed":
                raise ValueError("The actual native mesh gate failed; no field solve was requested.")
            # Only bind the named native circuit. Published value, units,
            # conditions, citations and the preselected tolerance stay intact.
            bound = selected.model_dump(mode="json")
            bound["observable"]["circuit_id"] = audit["circuit_id"]
            frozen = await request(client, "freeze_paper_target", "POST", "/api/targets", json_body={
                "revision_id": revision["id"], "evidence_id": extracted["evidence_id"],
                "case_id": selected.case_id, "targets": [bound]})
            solved = await request(client, "queue_solve", "POST", "/api/plan/jobs", json_body={
                "spec": plan.model_dump(mode="json"), "mesh_job_id": meshed["id"], "revision_id": revision["id"],
                "target_contract_id": frozen["contract_id"]})
            await asyncio.to_thread(app.state.worker.run, app.state.store.get(solved["id"]))
            solve_job = await request(client, "native_solver_and_target_result", "GET", "/api/jobs/" + solved["id"])
            target_report = solve_job.get("evidence", {}).get("target_report")
            report["target_report"] = target_report
            report["passed"] = solve_job["status"] == "completed" and isinstance(target_report, dict) and target_report.get("status") == "passed"
            if not report["passed"]:
                report["error"] = "The native solver or original frozen paper target did not pass. Its actual results are retained."
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        report.update(status="complete", runtime_seconds=round(time.monotonic()-started, 3))
        persist()
    return _safe(report, secrets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly permit this paid OpenRouter trial")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", help="Optional model override for this isolated trial only")
    parser.add_argument("--resume-evidence", type=Path, help="Prior unsuccessful report containing an unchanged verified AI extraction; no new extraction call")
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required because this trial uses the OpenRouter API")
    report = asyncio.run(run_paper_live(args.pdf, args.output, live=args.live, model=args.model,
                                      resume_evidence=args.resume_evidence))
    print(json.dumps({"status": report["status"], "passed": report["passed"],
                      "runtime_seconds": report["runtime_seconds"], "usage": report["usage"],
                      "error": report.get("error")}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
