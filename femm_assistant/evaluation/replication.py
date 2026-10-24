"""Native paper-reproduction benchmarks with targets fixed before execution.

Run in the patched FEMM Docker image, without model/API credentials:
    python -m evaluation.replication --output /results/replication

The models are independently transcribed from cited source data, not read from
an author's FEMM file. This benchmark measures numerical reproduction, separate
from the later PDF-to-model LLM extraction test. Original failures are retained.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from app.jobs import JobStore, SolverWorker, read_results
from app.plan import GeometryPlan
from app.targets import (NumericalTarget, TargetContract, contract_fingerprint, evaluate_target_file,
                         freeze_targets)
from app.plan_generate import required_results
from evaluation.axisymmetric import rectangular_winding_axis_bz


FIXTURE = Path(__file__).parent / "fixtures" / "lubin2012.json"


def load_fixture(path: Path = FIXTURE) -> dict:
    data = json.loads(path.read_text())
    if data.get("version") != 1 or data.get("id") != "lubin2012":
        raise ValueError("Unsupported replication fixture.")
    for case in data["cases"]:
        target = NumericalTarget.model_validate(case["target"])
        if target.case_id != case["id"]:
            raise ValueError("A paper target belongs to a different operating case.")
    return data


def _protocol(data, protocol):
    if protocol == "initial":
        return {"profiles": data["numerical_choices"]["profiles"]}
    selected = data["numerical_choices"].get("protocols", {}).get(protocol)
    if selected is None:
        raise ValueError("Unknown predetermined numerical protocol.")
    return selected


def build_lubin_plan(case_id: str, *, profile: str = "base", protocol: str = "initial",
                     fixture: dict | None = None) -> GeometryPlan:
    data = fixture or load_fixture()
    case = next((case for case in data["cases"] if case["id"] == case_id), None)
    if case is None:
        raise ValueError("Unknown Lubin operating case.")
    protocol_data = _protocol(data, protocol)
    if profile not in protocol_data["profiles"]:
        raise ValueError("Unknown predetermined mesh/domain profile.")
    numerical = protocol_data["profiles"][profile]
    parameters = {name: entry["value"] for name, entry in data["parameters"].items()}
    radius, axial = numerical["radial_domain_mm"], parameters["axial_half_domain_mm"]
    mesh = numerical["region_mesh_mm"]
    regions = [{"id": "coil", "shape": {"type": "rectangle",
        "x_min": parameters["coil_inner_radius_mm"], "x_max": parameters["coil_outer_radius_mm"],
        "y_min": parameters["coil_z_min_mm"], "y_max": parameters["coil_z_max_mm"]},
        "material": "winding", "label": {"x": 60, "y": 0},
        "circuit": "coil", "turns": parameters["turns"], "mesh_mm": mesh}]
    materials = [{"id": "winding", "relative_permeability": 1}]
    if case["core_present"]:
        h, half_length = case["displacement_mm"], parameters["core_half_length_mm"]
        regions.append({"id": "core", "shape": {"type": "rectangle", "x_min": 0,
            "x_max": parameters["core_radius_mm"], "y_min": h-half_length, "y_max": h+half_length},
            "material": "iron", "label": {"x": 15, "y": h}, "mesh_mm": mesh})
        materials.append({"id": "iron", "relative_permeability": parameters["relative_permeability"]})
    if "near_air_rectangle_mm" in protocol_data:
        air = protocol_data["near_air_rectangle_mm"]
        regions.insert(0, {"id": "near_air", "shape": {"type": "rectangle",
            "x_min": air["r_min"], "x_max": air["r_max"], "y_min": air["z_min"], "y_max": air["z_max"]},
            "material": "air", "label": {"x": 150, "y": 200}, "mesh_mm": numerical["near_air_mesh_mm"]})
    return GeometryPlan.model_validate({"kind": "magnetostatic_plan",
        "title": f"Lubin 2012 independently reconstructed {case_id} {protocol} {profile}",
        "physics": "axisymmetric_dc_magnetics", "units": "millimeters", "depth_mm": None,
        "mesh_mm": numerical["global_mesh_mm"],
        "domain": {"type": "rectangle", "x_min": 0, "x_max": radius, "y_min": -axial, "y_max": axial},
        "air_label": {"x": .8*radius, "y": .8*axial}, "materials": materials,
        "circuits": [{"id": "coil", "current_a": parameters["current_a"]}], "regions": regions,
        "force_region": "core" if case["core_present"] else None,
        "probes": [{"x": 0, "y": 0}, {"x": 0, "y": 200}]})


def _write(path: Path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _point_rows(text, key):
    lines = text.splitlines()
    pattern = re.compile(r"^\[" + re.escape(key) + r"\]\s*=\s*(\d+)\s*$")
    for i, line in enumerate(lines):
        found = pattern.match(line)
        if found:
            count = int(found.group(1))
            if count > 2000:
                raise ValueError("Native geometry audit row limit exceeded.")
            rows = [[float(value) for value in row.split()] for row in lines[i+1:i+1+count]]
            if len(rows) != count or any(not all(math.isfinite(x) for x in row) for row in rows):
                raise ValueError("Incomplete/non-finite native geometry audit rows.")
            return rows
    raise ValueError("Missing native geometry section: " + key)


def _properties(text, kind):
    blocks = re.findall(r"<Begin" + kind + r">(.*?)<End" + kind + r">", text, re.S)
    return [{name: value.strip().strip('"') for name, value in re.findall(r"<([^>]+)>\s*=\s*([^\n]+)", block)}
            for block in blocks]


def _contour_edges(shape):
    if shape["type"] == "rectangle":
        a, b, c, d = (shape[name] for name in ("x_min", "x_max", "y_min", "y_max"))
        points = [(a, c), (b, c), (b, d), (a, d)]
    elif shape["type"] == "polygon":
        points = [(point["x"], point["y"]) for point in shape["points"]]
    else:
        raise ValueError("This independent benchmark audit expects straight contours.")
    return list(zip(points, points[1:] + points[:1]))


def _edge_interval(first, second, edge):
    a, b = edge
    dx, dy = b[0]-a[0], b[1]-a[1]
    squared = dx*dx + dy*dy
    intervals = []
    for point in (first, second):
        if abs((point[0]-a[0])*dy - (point[1]-a[1])*dx) > 1e-7*max(1, math.sqrt(squared)):
            return None
        t = ((point[0]-a[0])*dx + (point[1]-a[1])*dy) / squared
        if not -1e-8 <= t <= 1+1e-8:
            return None
        intervals.append(t)
    return tuple(sorted(intervals))


def audit_native_model(plan: GeometryPlan, path: Path) -> dict:
    """Read native coordinates/assignments independently of compiler output code.

    Shared axis edges may be split or deduplicated; full coverage is checked.
    Probe locations are checked by the typed postprocessor and native results,
    because they are not stored in a FEMM model file.
    """
    text = path.read_text()
    if len(text) > 4 * 1024 * 1024:
        raise ValueError("Native model is too large for this benchmark audit.")
    spec = plan.model_dump(mode="json")
    checks = {"axisymmetric": bool(re.search(r"\[ProblemType\]\s*=\s*axisymmetric", text)),
              "millimeters": bool(re.search(r"\[LengthUnits\]\s*=\s*millimeters", text)),
              "dc": bool(re.search(r"\[Frequency\]\s*=\s*0(?:\.0*)?\s*$", text, re.M))}
    points, rows = _point_rows(text, "NumPoints"), _point_rows(text, "NumSegments")
    segments = []
    for row in rows:
        first, second = int(row[0]), int(row[1])
        if first == second or min(first, second) < 0 or max(first, second) >= len(points):
            raise ValueError("Invalid segment endpoint in native model.")
        segments.append((tuple(points[first][:2]), tuple(points[second][:2]), row))
    declared_edges = _contour_edges(spec["domain"]) + [
        edge for region in spec["regions"] for edge in _contour_edges(region["shape"])]
    checks["no_unrequested_edges"] = all(any(_edge_interval(first, second, edge) is not None for edge in declared_edges)
                                         for first, second, _ in segments)
    coverage = []
    for edge in declared_edges:
        intervals = sorted(item for first, second, _ in segments if (item := _edge_interval(first, second, edge)) is not None)
        end = 0.
        for lo, hi in intervals:
            if lo > end + 1e-8:
                break
            end = max(end, hi)
        coverage.append(end >= 1-1e-8)
    checks["all_requested_edges_covered"] = all(coverage)
    checks["no_arc_geometry"] = len(_point_rows(text, "NumArcSegments")) == 0
    boundaries = _properties(text, "Bdry")
    a0 = [i for i, item in enumerate(boundaries, 1) if item.get("BdryName") == "outer_a_zero"
          and float(item.get("BdryType", -1)) == 0 and float(item.get("A_0", 1)) == 0]
    outer = _contour_edges(spec["domain"])
    checks["all_outer_edges_a_zero"] = bool(a0) and all(int(row[3]) in a0 for first, second, row in segments
        if any(_edge_interval(first, second, edge) is not None for edge in outer))
    material_records = _properties(text, "Block")
    material_names = [item["BlockName"] for item in material_records]
    expected_materials = {"air": 1, **{item["id"]: item["relative_permeability"] for item in spec["materials"]}}
    checks["materials"] = set(material_names) == set(expected_materials) and all(
        math.isclose(float(item["Mu_x"]), expected_materials[item["BlockName"]])
        and math.isclose(float(item["Mu_y"]), expected_materials[item["BlockName"]])
        and float(item.get("BHPoints", 0)) == 0 for item in material_records)
    circuits = _properties(text, "Circuit")
    circuit_names = [item["CircuitName"] for item in circuits]
    checks["circuit_current"] = len(circuits) == 1 and circuit_names == ["coil"] and math.isclose(
        float(circuits[0]["TotalAmps_re"]), spec["circuits"][0]["current_a"]) and float(circuits[0]["TotalAmps_im"]) == 0
    labels = _point_rows(text, "NumBlockLabels")
    expected_labels = [{"label": spec["air_label"], "material": "air", "turns": 0,
                        "circuit": None, "mesh_mm": spec["mesh_mm"]}, *spec["regions"]]
    label_checks = []
    for region in expected_labels:
        matching = [row for row in labels if math.isclose(row[0], region["label"]["x"], abs_tol=1e-8)
                    and math.isclose(row[1], region["label"]["y"], abs_tol=1e-8)]
        valid = len(matching) == 1
        if valid:
            row = matching[0]
            material_index, circuit_index = int(row[2]), int(row[4])
            material = material_names[material_index-1] if 1 <= material_index <= len(material_names) else None
            circuit = circuit_names[circuit_index-1] if 1 <= circuit_index <= len(circuit_names) else None
            # FEMM serializes a default turn count of 1 on a non-circuit block.
            # With no circuit this is not an excitation. Only circuit-assigned
            # labels have a physically meaningful signed winding count.
            valid = material == region["material"] and circuit == region.get("circuit")
            if region.get("circuit") is not None:
                valid = valid and row[7] == region["turns"]
            valid = valid and math.isclose(row[3], region.get("mesh_mm") or spec["mesh_mm"])
        label_checks.append(valid)
    checks["region_labels_materials_turns_mesh"] = len(labels) == len(expected_labels) and all(label_checks)
    return {"passed": all(checks.values()), "checks": checks,
            "native_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "point_count": len(points), "segment_count": len(segments)}


def _physical_invariants(case: dict, plan: GeometryPlan, result: dict, *, energy_relative_tolerance=1e-5) -> dict:
    value = lambda key: result[key]["value"]
    energy, current, linkage = (value(key) for key in ("energy", "current_coil", "flux_linkage_coil"))
    expected_energy = .5 * current * linkage
    checks = {"positive_energy": energy > 0,
              "linear_energy_equals_half_i_linkage": abs(energy-expected_energy) <= energy_relative_tolerance*max(abs(expected_energy), 1e-30),
              "current_20a": math.isclose(current, 20, rel_tol=1e-10)}
    detail = {"energy_j": energy, "half_i_linkage_j": expected_energy,
              "energy_relative_tolerance": energy_relative_tolerance,
              "energy_relative_difference": abs(energy-expected_energy)/max(abs(expected_energy), 1e-30)}
    if case["id"] == "iron_displaced":
        checks["force_toward_coil_centre"] = value("force_z") < 0
    if case["id"] == "iron_center":
        checks["symmetric_centred_force"] = abs(value("force_z")) <= .05
    if case["id"] == "air_core":
        errors = []
        for index, z in enumerate((0, 200)):
            reference = rectangular_winding_axis_bz(r_inner_mm=50, r_outer_mm=70,
                z_min_mm=-100, z_max_mm=100, turns=1000, current_a=20, z_mm=z)
            error = abs(value(f"bz_{index}")/reference-1)
            errors.append(error)
        checks["independent_biot_savart_axis_field"] = max(errors) <= .02
        detail["axis_bz_relative_errors"] = errors
    return {"passed": all(checks.values()), "checks": checks, **detail}


def _artifact_hashes(folder: Path) -> dict:
    names = {"model.lua", "spec.json", "model.fem", "model.ans", "results.csv", "results.json",
             "mesh-report.json", "model.node", "model.ele", "model.edge", "solver.log"}
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.iterdir() if path.is_file() and path.name in names}


def run_replication(output: Path, *, case_ids=None, profiles=None, protocol="initial", command="femm-lua", timeout=180) -> dict:
    fixture = load_fixture()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists() or (output / "preflight.json").exists():
        raise ValueError("Use a new output directory to preserve previous runs and their failures.")
    selected_cases = [case for case in fixture["cases"] if case_ids is None or case["id"] in case_ids]
    selected_profiles = list(profiles or _protocol(fixture, protocol)["profiles"])
    if not selected_cases or (case_ids and set(case_ids) != {case["id"] for case in selected_cases}):
        raise ValueError("Unknown or empty replication case selection.")
    prepared = []
    for case in selected_cases:
        for profile in selected_profiles:
            plan = build_lubin_plan(case["id"], profile=profile, protocol=protocol, fixture=fixture)
            contract = freeze_targets([case["target"]], plan, case_id=case["id"])
            prepared.append((case, profile, plan, contract))
    # Persist every physical input, numerical choice, target and tolerance before
    # any native execution. Subsequent measurements cannot influence this file.
    _write(output / "preflight.json", {"fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "source": fixture["source"], "protocol": protocol, "numerical_choices": fixture["numerical_choices"],
        "prepared_cases": [{"case_id": case["id"], "profile": profile,
            "plan": plan.model_dump(mode="json"), "contract": contract.model_dump(mode="json"),
            "contract_sha256": contract_fingerprint(contract)} for case, profile, plan, contract in prepared]})
    report = {"status": "running", "passed": False, "source": fixture["source"], "protocol": protocol,
              "limitations": fixture["limitations"], "runs": [], "convergence": []}
    _write(output / "report.json", report)
    started = time.monotonic()
    for case, profile, plan, contract in prepared:
        folder = output / case["id"] / profile
        folder.mkdir(parents=True)
        _write(folder / "target-contract.json", contract.model_dump(mode="json"))
        store = JobStore(folder / "jobs")
        worker = SolverWorker(store, command=command, timeout=timeout)
        run = {"case_id": case["id"], "profile": profile, "passed": False,
               "contract_sha256": contract_fingerprint(contract), "targets_sha256": contract.targets_sha256}
        try:
            mesh_job = store.create(plan, kind="mesh")
            worker.run(mesh_job)
            meshed = store.get(mesh_job["id"])
            run["mesh"] = meshed
            run["mesh_artifacts"] = _artifact_hashes(store.root / mesh_job["id"])
            if meshed["status"] != "completed" or meshed["result"].get("status") != "passed":
                raise RuntimeError("Mesh gate failed; no field solve was requested.")
            solve_job = store.create(plan)
            worker.run(solve_job)
            solved = store.get(solve_job["id"])
            run["solver"] = solved
            native_folder = store.root / solve_job["id"]
            run["solver_artifacts"] = _artifact_hashes(native_folder)
            if solved["status"] != "completed":
                raise RuntimeError("Native solver or its result parser failed.")
            run["native_model_audit"] = audit_native_model(plan, native_folder / "model.fem")
            run["target_check"] = evaluate_target_file(contract, native_folder, plan=plan,
                expected_contract_sha256=run["contract_sha256"])
            run["physical_invariants"] = _physical_invariants(case, plan, solved["result"],
                energy_relative_tolerance=_protocol(fixture, protocol).get("energy_consistency_relative_tolerance", 1e-5))
            run["passed"] = (run["native_model_audit"]["passed"] and run["target_check"]["status"] == "passed"
                             and run["physical_invariants"]["passed"])
        except Exception as exc:
            run["error"] = str(exc)
        report["runs"].append(run)
        _write(folder / "report.json", run)
        _write(output / "report.json", report)
    threshold = fixture["numerical_choices"]["convergence_relative_tolerance"]
    for case in selected_cases:
        runs = {run["profile"]: run for run in report["runs"] if run["case_id"] == case["id"]}
        convergence = {"case_id": case["id"], "relative_tolerance": threshold, "checks": []}
        for description, first, second in (("mesh_refinement", "base", "refined"), ("radial_domain", "refined", "far")):
            a, b = runs.get(first, {}), runs.get(second, {})
            try:
                original = a["target_check"]["checks"][0]["actual"]
                improved = b["target_check"]["checks"][0]["actual"]
                error = abs(original-improved) / max(abs(improved), 1e-30)
                check = {"name": description, "relative_change": error, "passed": error <= threshold}
            except (KeyError, TypeError):
                check = {"name": description, "passed": False, "error": "Required completed profile is absent."}
            convergence["checks"].append(check)
        convergence["passed"] = all(check["passed"] for check in convergence["checks"])
        report["convergence"].append(convergence)
    report["passed"] = all(run["passed"] for run in report["runs"]) and all(item["passed"] for item in report["convergence"])
    report.update(status="complete", runtime_seconds=round(time.monotonic()-started, 3),
                  native_solve_count=sum("solver" in run for run in report["runs"]),
                  native_mesh_count=sum("mesh" in run for run in report["runs"]))
    _write(output / "report.json", report)
    return report


def reaudit_existing_report(input_report: Path, output: Path, *, protocol="near_field_v3") -> dict:
    """Recheck unchanged native artifacts without remeshing or solving again.

    Verify every previously recorded native artifact hash first. Read numerical
    values again through the worker CSV parser, rather than trusting summary
    values from the old report. Both original failure reports remain intact.
    """
    fixture = load_fixture()
    protocol_data = _protocol(fixture, protocol)
    if protocol != "near_field_v3":
        raise ValueError("Artifact re-audit supports only the documented near_field_v3 diagnostic correction.")
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh re-audit directory to preserve the original report and failures.")
    raw = input_report.read_bytes()
    original = json.loads(raw)
    if original.get("protocol") != "near_field_v2" or original.get("status") != "complete":
        raise ValueError("Re-audit requires a completed near_field_v2 report and its unchanged artifacts.")
    source_root = input_report.resolve().parent
    preflight_bytes = (source_root / "preflight.json").read_bytes()
    preflight = json.loads(preflight_bytes)
    prepared = {(entry["case_id"], entry["profile"]): entry for entry in preflight["prepared_cases"]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "original-report.json").write_bytes(raw)
    (output / "original-preflight.json").write_bytes(preflight_bytes)
    report = {"status": "running", "passed": False, "protocol": protocol,
        "execution_mode": "artifact_reaudit", "native_execution_performed": False,
        "native_solve_count": 0, "native_mesh_count": 0,
        "source_native_solve_count": original.get("native_solve_count"),
        "source_native_mesh_count": original.get("native_mesh_count"),
        "original_report_path": str(input_report.resolve()), "original_report_sha256": hashlib.sha256(raw).hexdigest(),
        "original_preflight_sha256": hashlib.sha256(preflight_bytes).hexdigest(),
        "source": fixture["source"], "limitations": fixture["limitations"],
        "auxiliary_diagnostic_correction": protocol_data["energy_consistency_basis"],
        "runs": [], "convergence": []}
    _write(output / "report.json", report)
    started = time.monotonic()
    try:
        for original_run in original["runs"]:
            case_id, profile = original_run["case_id"], original_run["profile"]
            case = next((case for case in fixture["cases"] if case["id"] == case_id), None)
            if case is None or profile not in protocol_data["profiles"]:
                raise ValueError("The old report contains an unknown operating case or numerical profile.")
            folder = source_root / case_id / profile
            for stage in ("mesh", "solver"):
                job = original_run.get(stage)
                if not job or job.get("status") != "completed" or not re.fullmatch(r"[a-f0-9]{32}", job["id"]):
                    raise ValueError("A required original native job was incomplete.")
                native_folder = folder / "jobs" / job["id"]
                hashes = original_run.get(stage + "_artifacts", {})
                required = {"spec.json", "model.fem", "mesh-report.json", "solver.log"}
                if stage == "solver":
                    required |= {"model.ans", "results.csv"}
                if not required.issubset(hashes):
                    raise ValueError("The old report lacks required native artifact hashes.")
                for name, expected_hash in hashes.items():
                    if Path(name).name != name:
                        raise ValueError("Invalid native artifact name.")
                    path = native_folder / name
                    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64*1024*1024:
                        raise ValueError("A previously recorded native artifact is unavailable or invalid.")
                    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                        raise ValueError("A native artifact changed after the original run: " + name)
                if json.loads((native_folder / "mesh-report.json").read_text()).get("status") != "passed":
                    raise ValueError("An original native mesh report did not pass.")
            # Native solver artifacts include the actual compiled spec.
            plan = GeometryPlan.model_validate_json((native_folder / "spec.json").read_text())
            frozen = TargetContract.model_validate_json((folder / "target-contract.json").read_text())
            entry = prepared[(case_id, profile)]
            if (entry["plan"] != plan.model_dump(mode="json") or
                    contract_fingerprint(frozen) != original_run["contract_sha256"] or
                    entry["contract_sha256"] != original_run["contract_sha256"]):
                raise ValueError("The saved model or frozen paper target differs from the pre-solve contract.")
            expected_plan = build_lubin_plan(case_id, profile=profile, protocol=protocol).model_dump(mode="json")
            expected_plan["title"] = plan.title
            if expected_plan != plan.model_dump(mode="json"):
                raise ValueError("The original model differs from the selected re-audit protocol's physical/numerical inputs.")
            if frozen.targets_sha256 != freeze_targets([case["target"]], plan, case_id=case_id).targets_sha256:
                raise ValueError("The published target or its original acceptance tolerance changed.")
            results = read_results(native_folder, required_results(plan))
            run = copy.deepcopy(original_run)
            run["original_physical_invariants"] = run.get("physical_invariants")
            run["physical_invariants"] = _physical_invariants(case, plan, results,
                energy_relative_tolerance=protocol_data["energy_consistency_relative_tolerance"])
            run["native_model_audit"] = audit_native_model(plan, native_folder / "model.fem")
            run["target_check"] = evaluate_target_file(frozen, native_folder, plan=plan,
                expected_contract_sha256=original_run["contract_sha256"])
            run["artifact_hashes_verified"] = True
            run["native_execution_performed"] = False
            run["passed"] = run["native_model_audit"]["passed"] and run["physical_invariants"]["passed"] and run["target_check"]["status"] == "passed"
            report["runs"].append(run)
            _write(output / "report.json", report)
        threshold = fixture["numerical_choices"]["convergence_relative_tolerance"]
        for case in fixture["cases"]:
            runs = {run["profile"]: run for run in report["runs"] if run["case_id"] == case["id"]}
            if not runs:
                continue
            checks = []
            for name, first, second in (("mesh_refinement", "base", "refined"), ("radial_domain", "refined", "far")):
                a, b = (runs[profile]["target_check"]["checks"][0]["actual"] for profile in (first, second))
                change = abs(a-b) / max(abs(b), 1e-30)
                checks.append({"name": name, "relative_change": change, "passed": change <= threshold})
            report["convergence"].append({"case_id": case["id"], "relative_tolerance": threshold,
                "checks": checks, "passed": all(check["passed"] for check in checks)})
        report["passed"] = bool(report["runs"]) and all(run["passed"] for run in report["runs"]) and all(item["passed"] for item in report["convergence"])
    except Exception as exc:
        report["error"] = str(exc)
    report.update(status="complete", runtime_seconds=round(time.monotonic()-started, 3))
    _write(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--profile", action="append", dest="profiles")
    parser.add_argument("--protocol", choices=("initial", "near_field_v2", "near_field_v3"), default="initial")
    parser.add_argument("--audit-existing", type=Path, help="Re-read unchanged v2 native artifacts without new FEMM executions")
    parser.add_argument("--command", default=os.getenv("FEMM_LUA", "femm-lua"))
    args = parser.parse_args()
    report = (reaudit_existing_report(args.audit_existing, args.output, protocol="near_field_v3") if args.audit_existing else
              run_replication(args.output, case_ids=args.cases, profiles=args.profiles, protocol=args.protocol, command=args.command))
    print(json.dumps({"passed": report["passed"], "native_mesh_count": report["native_mesh_count"],
                      "native_solve_count": report["native_solve_count"], "runtime_seconds": report["runtime_seconds"]}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
