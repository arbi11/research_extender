#!/usr/bin/env python3
"""Isolated native probes of the unmodified FEMM Agent Suite.

Usage inside the evaluation Docker image (the package must be installed):
    FEMM_DIR=/opt/femm/bin python /evaluation/native_probes.py /testwork/native-probes.json

Uses registered tool functions directly rather than an LLM or MCP transport. Every
tool response is checkpointed verbatim after JSON-safe conversion. A confirmed
stale-solution finding is evidence, not a script infrastructure failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from typing import Any


CALL_TIMEOUT_S = 120
SESSION = "native_stale_solution_probe"
MU0 = 4e-7 * math.pi


class ProbeCallError(RuntimeError):
    pass


def json_safe(value: Any) -> Any:
    if isinstance(value, complex):
        return {"re": value.real, "im": value.imag}
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = output.parent / (output.stem + "-artifacts")
    scratch.mkdir(parents=True, exist_ok=True)
    # These only affect this process and child engines; no global client setup.
    os.environ["FEMM_AGENT_WORK"] = str(scratch / "sessions")
    os.environ["FEMM_AGENT_HOME"] = str(scratch / "suite-state")
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema": "femm-suite-native-probes/v1",
        "execution": "registered tool functions with a real native FEMM engine",
        "uses_llm": False,
        "upstream_modified": False,
        "call_timeout_seconds": CALL_TIMEOUT_S,
        "artifacts_dir": str(scratch),
        "calls": [],
        "checks": {},
        "errors": [],
        "status": "running",
    }

    def checkpoint() -> None:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)

    def alarm_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"registered tool exceeded {CALL_TIMEOUT_S} seconds")

    checkpoint()
    sessions = None
    exit_code = 0
    old_handler = signal.signal(signal.SIGALRM, alarm_handler)
    try:
        from femm_agent import __version__, catalog
        from femm_agent.config import find_femm, femm_version
        from femm_agent.server import build_server
        from femm_agent.sessions import SessionManager

        install = find_femm()
        if install is None:
            raise RuntimeError("FEMM installation not found; set FEMM_DIR to the native bin directory")
        sessions = SessionManager(install=install, default_work_dir=scratch / "sessions")
        _server, context = build_server(sessions)
        registry = context.extras["tools"]
        report["metadata"] = {
            "suite_version": __version__,
            "engine_version": femm_version(install),
            "engine_path": str(install.headless),
            "tool_count": len(registry),
            "catalog_count": len(catalog.CATALOG),
            "catalog_warning": catalog.CATALOG.warning,
            "tool_groups": context.extras.get("groups", {}),
            "skipped_tool_groups": context.extras.get("skipped_groups", {}),
            "excluded_action_tools": ["femm_star_repo"],
        }
        checkpoint()

        def call(tool_name: str, *, required: bool = True, **arguments: Any) -> Any:
            if tool_name == "femm_star_repo":
                raise AssertionError("star action is outside this evaluation")
            entry: dict[str, Any] = {"tool": tool_name, "arguments": arguments}
            report["calls"].append(entry)
            t0 = time.monotonic()
            try:
                signal.setitimer(signal.ITIMER_REAL, CALL_TIMEOUT_S)
                response = registry[tool_name](**arguments)
                entry["response"] = json_safe(response)
                if not isinstance(response, dict) or "ok" not in response:
                    raise ProbeCallError(f"{tool_name} returned an invalid tool envelope")
                if not response["ok"] and required:
                    raise ProbeCallError(f"{tool_name} failed: {response.get('error')}")
                return response.get("result") if response["ok"] else None
            except Exception as exc:
                entry["exception"] = {"type": type(exc).__name__, "message": str(exc)}
                if required:
                    raise
                return None
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                entry["elapsed_seconds"] = round(time.monotonic() - t0, 3)
                checkpoint()

        def magnitude(result: Any) -> float:
            value = result["values"][0]["|B|"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ProbeCallError(f"invalid native magnetic magnitude: {value!r}")
            return float(value)

        def session_state() -> dict[str, Any]:
            states = call("femm_session_list")
            return next(row for row in states if row["name"] == SESSION)

        probe = [10.0, 0.0]
        call("femm_session_open", name=SESSION, work_dir=str(scratch / "sessions" / SESSION), headless=True)
        call("femm_build_benchmark", kind="round_wire", path=str(scratch / "wire_1A.fem"),
             params={"frequency": 0.0, "current": 1.0, "radius": 5.0,
                     "boundary_radius": 50.0, "depth": 1000.0, "mesh_wire": 0.25}, session=SESSION)
        solved_one = call("femm_solve", timeout=CALL_TIMEOUT_S, session=SESSION)
        baseline_state = session_state()
        baseline = call("femm_point_values", points=[probe], session=SESSION)
        b_one = magnitude(baseline)
        if b_one <= 0:
            raise ProbeCallError("baseline field must be nonzero to test stale results and scaling")
        baseline_circuit = call("femm_circuit_results", name="wire", session=SESSION)

        # Deliberately request a result between the actual mutation and re-solve.
        # Nothing in this probe presents that old field as a valid 2 A result.
        call("femm_modify_circuit", name="wire", current=2.0, session=SESSION)
        mutated_state = session_state()
        before_resolve = call("femm_point_values", points=[probe], session=SESSION, required=False)
        before_resolve_circuit = call("femm_circuit_results", name="wire", session=SESSION, required=False)
        b_before = magnitude(before_resolve) if before_resolve is not None else None
        stale_accepted = (
            mutated_state.get("solved") is True
            and b_before is not None
            and math.isclose(b_before, b_one, rel_tol=1e-8, abs_tol=1e-15)
        )
        report["checks"]["stale_solution"] = {
            "finding": "confirmed" if stale_accepted else "not_reproduced",
            "mutation": "only circuit current changed from 1 A to 2 A",
            "probe_mm": probe,
            "baseline_B_T": b_one,
            "B_after_mutation_before_resolve_T": b_before,
            "ratio_to_baseline_before_resolve": b_before / b_one if b_before is not None else None,
            "baseline_session": baseline_state,
            "session_after_mutation": mutated_state,
            "baseline_circuit_result": baseline_circuit,
            "circuit_result_before_resolve": before_resolve_circuit,
            "interpretation": (
                "The registered point-values tool accepted the old loaded field after a current change; "
                "the session still advertised solved=True. Re-solving is required."
                if stale_accepted else
                "The tool either rejected the pre-resolve result, invalidated state, or returned a changed field; inspect raw calls."
            ),
        }
        checkpoint()

        solved_two = call("femm_solve", timeout=CALL_TIMEOUT_S, path=str(scratch / "wire_2A.fem"), session=SESSION)
        resolved = call("femm_point_values", points=[probe], session=SESSION)
        resolved_circuit = call("femm_circuit_results", name="wire", session=SESSION)
        b_two = magnitude(resolved)
        ratio = b_two / b_one
        expected_one = MU0 / (2 * math.pi * probe[0] * 1e-3)
        relative_error = abs(b_one - expected_one) / expected_one
        report["checks"]["native_linear_current_scaling"] = {
            "passed": math.isclose(ratio, 2.0, rel_tol=0.01),
            "current_A": [1.0, 2.0], "probe_mm": probe,
            "baseline_B_T": b_one, "resolved_B_T": b_two,
            "observed_ratio": ratio, "expected_ratio": 2.0,
            "ratio_relative_tolerance": 0.01,
            "circuit_result_after_resolve": resolved_circuit,
            "native_solve_one": solved_one, "native_solve_two": solved_two,
        }
        report["checks"]["wire_ampere_reference"] = {
            "passed": relative_error < 0.02,
            "formula": "B = mu0 I / (2 pi r), outside the concentric round wire",
            "I_A": 1.0, "r_m": probe[0] * 1e-3,
            "expected_B_T": expected_one, "observed_B_T": b_one,
            "relative_error": relative_error, "relative_tolerance": 0.02,
        }
        call("femm_journal", limit=0, save=True, session=SESSION)
        report["status"] = "completed"
        if not all(report["checks"][key]["passed"] for key in ("native_linear_current_scaling", "wire_ampere_reference")):
            report["status"] = "completed_with_physics_check_failure"
            exit_code = 1
    except Exception as exc:
        report["status"] = "error"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc), "trace": traceback.format_exc()})
        exit_code = 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if sessions is not None:
            try:
                signal.setitimer(signal.ITIMER_REAL, CALL_TIMEOUT_S)
                sessions.close_all()
            except Exception as exc:
                report["errors"].append({"stage": "session_cleanup", "type": type(exc).__name__, "message": str(exc)})
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        files = []
        for path in sorted(scratch.rglob("*")):
            if path.is_file():
                raw = path.read_bytes()
                files.append({"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        report["artifact_manifest"] = files
        checkpoint()
    print(json.dumps({"output": str(output), "status": report["status"], "checks": report["checks"]}, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
