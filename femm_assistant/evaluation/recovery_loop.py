"""Native mesh→solver→target recovery benchmark with a known injected error.

Run in the patched FEMM Docker image:
    python -m evaluation.recovery_loop --output /results/recovery-loop

Default mode uses a deterministic restoration of the known injected current
error. It measures the execution/history mechanism, not Jev diagnosis quality.
``--live`` requests one Jev diagnosis and one OpenRouter repair proposal instead;
no provider credentials are used or required by the default mode.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

from app import decisions
from app.history import ModelRevisionStore
from app.jobs import JobStore, SolverWorker
from app.plan import GeometryPlan, wire_example
from app.recovery import propose_recovery
from app.targets import (NumericalTarget, contract_fingerprint, evaluate_target_file,
                         freeze_targets, rebind_targets)


CASE_ID = "wire_1a"
CURRENT_PATH = "/circuits/wire/current_a"


def reference_plan() -> GeometryPlan:
    """A 1 mm-radius, 1 A straight wire with an external probe at (2,0) mm."""
    value = wire_example().model_dump(mode="json")
    value["title"] = "Controlled-recovery straight-wire reference"
    value["mesh_mm"] = .2
    return GeometryPlan.model_validate(value)


def reference_target() -> NumericalTarget:
    """Independent Ampère-law target frozen for 1 A before fault injection."""
    expected = (4 * math.pi * 1e-7) * 1.0 / (2 * math.pi * .002)
    return NumericalTarget.model_validate({
        "id": "wire_by", "case_id": CASE_ID,
        "description": "External azimuthal magnetic field at x=2 mm, y=0 mm",
        "observable": {"quantity": "by", "probe_index": 0},
        "value": expected, "unit": "T", "tolerance": {"relative": .03},
        "reference_kind": "analytical", "evidence_kind": "derived",
        "citations": [{"source_id": "https://openstax.org/books/university-physics-volume-2/pages/12-6-amperes-law",
                       "section": "Ampere's law: magnetic field outside a long straight wire"}],
        "derivation": "B=mu0*I/(2*pi*r), with mu0=4*pi*1e-7 H/m, I=1 A and r=0.002 m; By is positive at x>0.",
        "conditions": "Planar, infinitely long straight conductor in air; radius 1 mm, current +1 A, one positive turn, probe (2,0) mm. The concentric circular A=0 boundary fixes the vector-potential reference."})


def _write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(folder: Path) -> dict:
    return {path.name: {"sha256": _sha(path), "bytes": path.stat().st_size}
            for path in sorted(folder.iterdir()) if path.is_file()}


def _execute_attempt(name: str, revision: dict, contract, *, output: Path,
                     jobs: JobStore, worker: SolverWorker, history: ModelRevisionStore) -> dict:
    """Run both native stages, then read actual .ans/CSV for the target check."""
    plan = GeometryPlan.model_validate(revision["plan"])
    frozen_sha = contract_fingerprint(contract)
    contract_path = output / (name + "-frozen-targets.json")
    _write(contract_path, contract.model_dump(mode="json"))
    started = time.monotonic()
    mesh = jobs.create(plan, kind="mesh", revision_id=revision["id"])
    worker.run(mesh)
    mesh = jobs.get(mesh["id"])
    if mesh["status"] != "completed" or mesh["result"].get("status") != "passed":
        raise RuntimeError(f"{name}: native mesh failed; inspect job {mesh['id']}: {mesh['error']}")
    if (jobs.root / mesh["id"] / "model.ans").exists():
        raise RuntimeError("A mesh-only attempt unexpectedly produced a field solution.")
    solve = jobs.create(plan, kind="solve", revision_id=revision["id"])
    # Persist the binding before native execution and retain its hash at queue time.
    jobs.annotate(solve["id"], target_contract_sha256=frozen_sha,
                  targets_sha256=contract.targets_sha256)
    worker.run(solve)
    solve = jobs.get(solve["id"])
    if solve["status"] != "completed":
        raise RuntimeError(f"{name}: native solve failed; inspect job {solve['id']}: {solve['error']}")
    folder = jobs.root / solve["id"]
    targets = evaluate_target_file(contract, folder, plan=plan,
        expected_contract_sha256=solve["evidence"]["target_contract_sha256"])
    report_path = folder / "target-report.json"
    _write(report_path, targets)
    manifest = {"job_id": solve["id"], "revision_id": revision["id"],
                "contract_sha256": frozen_sha, "targets_sha256": contract.targets_sha256,
                "target_report_sha256": _sha(report_path), "results_csv_sha256": _sha(folder / "results.csv"),
                "native_solution_sha256": _sha(folder / "model.ans")}
    manifest_path = folder / "target-execution.json"
    _write(manifest_path, manifest)
    check = history.record_check(revision["id"], stage="target",
        status="passed" if targets["status"] == "passed" else "failed", job_id=solve["id"],
        artifact_hash=_sha(manifest_path),
        details={"target_report": targets, "manifest": manifest},
        implicated_paths=[CURRENT_PATH] if targets["status"] == "failed" else [])
    checks = solve["evidence"].get("check_ids", [])
    jobs.annotate(solve["id"], check_ids=[*checks, check["id"]],
                  **({"failure_id": check["id"]} if targets["status"] != "passed" else {}))
    result = {"name": name, "revision_id": revision["id"], "plan_hash": revision["plan_hash"],
              "current_a": plan.circuits[0].current_a, "mesh_job_id": mesh["id"],
              "solve_job_id": solve["id"], "mesh_status": "passed", "solver_status": "passed",
              "target_status": targets["status"], "target_check_id": check["id"],
              "target_report": targets, "duration_seconds": time.monotonic()-started,
              "worker_runtime_seconds": mesh["result"]["runtime"]["value"] + solve["result"]["runtime"]["value"],
              "native_invocations": 2,
              "artifacts": {"mesh": _inventory(jobs.root / mesh["id"]), "solve": _inventory(folder)}}
    _write(output / (name + "-attempt.json"), result)
    return result


def verify_outcome(attempts: list[dict], baseline: dict, wrong: dict, recovered: dict,
                   history: ModelRevisionStore) -> dict:
    """Success requires observed failures/pass results and preserved evidence."""
    checks = {
        "baseline_three_checks_pass": all(attempts[0][key] == "passed" for key in ("mesh_status", "solver_status", "target_status")),
        "injected_model_mesh_solver_pass_but_target_fails": attempts[1]["mesh_status"] == attempts[1]["solver_status"] == "passed" and attempts[1]["target_status"] == "failed",
        "recovered_three_checks_pass": all(attempts[2][key] == "passed" for key in ("mesh_status", "solver_status", "target_status")),
        "frozen_target_definitions_unchanged": len({item["target_report"]["targets_sha256"] for item in attempts}) == 1,
        "restored_plan_matches_baseline": recovered["plan_hash"] == baseline["plan_hash"],
        "wrong_model_retained": history.get(wrong["id"])["plan"]["circuits"][0]["current_a"] == 2,
        "wrong_model_remains_main_head": history.head()["id"] == wrong["id"],
        "original_failure_retained": history.get_check(attempts[1]["target_check_id"])["status"] == "failed",
    }
    return {"passed": all(checks.values()), "checks": checks}


def run_benchmark(output: Path, *, command: str | None = None, timeout: int = 120,
                  live: bool = False) -> dict:
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory so previous recovery evidence is preserved.")
    output.mkdir(parents=True, exist_ok=True)
    launcher = command or os.getenv("FEMM_LUA", "femm-lua")
    if not shutil.which(launcher):
        raise RuntimeError("The native FEMM launcher is unavailable; run this benchmark in the patched Docker image.")
    started = time.monotonic()
    history = ModelRevisionStore(output / "history")
    jobs = JobStore(output / "jobs")
    worker = SolverWorker(jobs, command=launcher, timeout=timeout, revision_store=history)
    summary = {"benchmark": "straight_wire_controlled_recovery", "version": 1, "status": "running",
               "mode": "live_jev_repair_proposal" if live else "deterministic_known_injection",
               "attempts": [], "limitations": [
                   "The deliberately injected cause is known. Default restoration does not measure AI diagnosis quality.",
                   "No restart-from-scratch comparator is run; resource savings are not established.",
                   "This is an analytical reference case, not a journal-paper extraction benchmark."],
               "provider_usage": {"calls": 0 if not live else None,
                                  "maximum_calls": 0 if not live else 2,
                                  "cost_usd": 0.0 if not live else None,
                                  "note": "No provider is called in deterministic mode." if not live else
                                          "One diagnostic and one draft maximum; current planning transport does not expose provider usage/cost."}}
    summary_path = output / "summary.json"
    _write(summary_path, summary)
    try:
        baseline = history.create(reference_plan(), evidence=[{"source": "calculation",
            "reference": "Ampere's law outside the 1 A straight wire", "detail": reference_target().derivation}])
        # Freeze the target before the error is introduced or a native job runs.
        contract = freeze_targets([reference_target()], baseline["plan"], case_id=CASE_ID)
        summary["baseline_revision_id"] = baseline["id"]
        summary["frozen_targets_sha256"] = contract.targets_sha256
        summary["attempts"].append(_execute_attempt("baseline", baseline, contract,
            output=output, jobs=jobs, worker=worker, history=history))
        _write(summary_path, summary)
        wrong = history.apply_patch(baseline["id"], [{"op": "replace", "path": CURRENT_PATH, "value": 2.0}],
            allowed_paths=[CURRENT_PATH], reason="Deliberately inject a known current error for recovery measurement.",
            evidence=[{"source": "artifact", "reference": "controlled fault injection", "detail": "Change 1 A to 2 A; retain the 1 A target."}])
        wrong_contract = rebind_targets(contract, wrong["plan"])
        summary["wrong_revision_id"] = wrong["id"]
        summary["attempts"].append(_execute_attempt("injected_2a", wrong, wrong_contract,
            output=output, jobs=jobs, worker=worker, history=history))
        _write(summary_path, summary)
        if summary["attempts"][1]["target_status"] != "failed":
            raise RuntimeError("The known current injection did not fail the frozen analytical target.")
        failure_id = summary["attempts"][1]["target_check_id"]
        context = history.recovery_context(failure_id)
        _write(output / "failure-localization.json", context)
        if not any(candidate["revision_id"] == wrong["id"] and any(change["path"] == CURRENT_PATH
                   for change in candidate["changes"]) for candidate in context["candidates"]):
            raise RuntimeError("Failure localization lost the actual current modification.")
        diagnosis = "Known injected fixture: the current was changed from the 1 A reference to 2 A. Restore the ancestral 1 A checkpoint and rerun all checks."
        checkpoint_id, edits = baseline["id"], []
        if live:
            context["checkpoint_models"] = [{"id": revision["id"], "plan": revision["plan"]} for revision in [wrong, baseline]]
            proposal = asyncio.run(propose_recovery(context, os.getenv("OPENROUTER_API_KEY", ""),
                os.getenv("AI_BASE_URL") or "https://openrouter.ai/api/v1",
                os.getenv("AI_MODEL") or "google/gemini-3.1-flash-lite",
                decision_model=os.getenv("DECISION_MODEL") or decisions.DEFAULT_MODEL,
                decision_base_url=os.getenv("DECISION_BASE_URL") or decisions.DEFAULT_BASE_URL,
                instruction="The independent reference case requires 1 A. The native mesh and solver passed, but the frozen field target failed after the current modification. Identify and correct that implementation error; do not change geometry, targets or tolerances.",
                allowed_paths=[CURRENT_PATH]))
            _write(output / "recovery-proposal.json", proposal.model_dump(mode="json"))
            if proposal.questions or proposal.spec is None:
                raise RuntimeError("The bounded live proposal needs clarification; no speculative repair was executed.")
            checkpoint_id, edits = proposal.checkpoint_id, proposal.edits
            diagnosis = proposal.explanation
        recovered = history.branch_recovery(failure_id, checkpoint_id, branch="recovered", diagnosis=diagnosis,
            evidence=[{"source": "artifact", "reference": "recovery-proposal.json" if live else "known injected current error"}])
        if edits:
            recovered = history.apply_patch(recovered["id"], edits, allowed_paths=[CURRENT_PATH], branch="recovered",
                                            reason=diagnosis)
        summary["recovered_revision_id"] = recovered["id"]
        restored_contract = rebind_targets(contract, recovered["plan"])
        summary["attempts"].append(_execute_attempt("recovered_1a", recovered, restored_contract,
            output=output, jobs=jobs, worker=worker, history=history))
        summary["verification"] = verify_outcome(summary["attempts"], baseline, wrong, recovered, history)
        summary["status"] = "passed" if summary["verification"]["passed"] else "failed"
        summary["metrics"] = {"model_attempts": len(summary["attempts"]),
            "explicit_mesh_jobs": len(summary["attempts"]), "solve_jobs": len(summary["attempts"]),
            "native_invocations": sum(item["native_invocations"] for item in summary["attempts"]),
            "worker_runtime_seconds": sum(item["worker_runtime_seconds"] for item in summary["attempts"]),
            "wall_seconds": time.monotonic()-started}
        _write(output / "history.json", {"failed_main": history.lineage(wrong["id"]),
            "recovered_branch": history.lineage(recovered["id"]),
            "checks_by_revision": {revision["id"]: history.checks(revision["id"])
                                   for revision in [baseline, wrong, recovered]}})
        _write(summary_path, summary)
        return summary
    except Exception as error:
        summary.update(status="failed", error=str(error), wall_seconds=time.monotonic()-started)
        _write(summary_path, summary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Fresh output directory for all native and journal evidence")
    parser.add_argument("--live", action="store_true", help="Request one bounded Jev/OpenRouter repair proposal (requires provider key)")
    parser.add_argument("--timeout", type=int, default=120, help="Maximum seconds per native FEMM invocation")
    options = parser.parse_args()
    if not 1 <= options.timeout <= 180:
        parser.error("Native timeout must be between 1 and 180 seconds.")
    try:
        result = run_benchmark(options.output, timeout=options.timeout, live=options.live)
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error), "output": str(options.output)}))
        raise SystemExit(1) from None
    print(json.dumps({"status": result["status"], "mode": result["mode"], "metrics": result["metrics"],
                      "summary": str(options.output / "summary.json")}, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
