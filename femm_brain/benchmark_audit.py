"""Audit native benchmark evidence without relabeling published-target failures.

An accepted reference discrepancy is a separate regression check against an
independent original-source control. It is never a published-output pass.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from .core import DATA, load_graph
from .simulation import assess_target, plan_simulation, simulation_catalog

REGISTRY = DATA / "verification" / "reference_discrepancies.json"


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def audit_summary(summary, registry=None, *, graph=None, plans=None):
    """Require every catalog baseline; narrowly audit reviewed source discrepancies.

    ``plans`` is an optional prebuilt baseline-plan mapping for offline callers
    and tests. Normal CI constructs these directly from the current catalog.
    Input reports are read only and their failure statuses remain in the audit.
    """
    graph = load_graph() if graph is None else graph
    registry = json.loads(REGISTRY.read_text()) if registry is None else registry
    audit = {"schema_version": 1, "checked_at": datetime.now(timezone.utc).isoformat(),
             "status": "failed", "all_published_targets_passed": False,
             "meaning": "Audit acceptance does not convert reference mismatches into published-output agreement.",
             "cases": [], "issues": [], "resolved_reference_discrepancies": [],
             "input_summary_status": summary.get("status"), "reports_modified": False}
    issues = audit["issues"]
    if graph.get("validation", {}).get("status") != "passed" or graph.get("source_changes"):
        issues.append("Knowledge graph is invalid or its curated source content changed.")
    if registry.get("schema_version") != 1:
        issues.append("Unsupported discrepancy registry schema.")
    rules = {}
    for rule in registry.get("discrepancies", []):
        identifier = rule.get("case_id")
        if not identifier or identifier in rules:
            issues.append("Duplicate or missing discrepancy case identity.")
        rules[identifier] = rule
    try:
        if plans is None:
            plans = {case["id"]: plan_simulation(case["id"], graph=graph)
                     for case in simulation_catalog(graph)}
    except (ValueError, KeyError, OSError) as exc:
        issues.append("Current baseline plans cannot be established: " + str(exc))
        return audit
    reports = summary.get("cases")
    if not isinstance(reports, list) or not reports:
        issues.append("A nonempty verify-solves cases list is required.")
        return audit
    identifiers = [report.get("case_id") for report in reports]
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(plans):
        issues.append("Native summary must contain every current catalog case exactly once.")
    if summary.get("status") != ("passed" if all(r.get("status") == "passed" for r in reports) else "failed"):
        issues.append("Summary status does not preserve its individual report statuses.")
    for report in reports:
        case_id = report.get("case_id")
        entry = {"case_id": case_id, "status": "failed", "report_status": report.get("status"), "issues": []}
        audit["cases"].append(entry)
        errors = entry["issues"]
        plan = plans.get(case_id)
        if plan is None:
            errors.append("Unregistered case.")
            continue
        checks = report.get("checks", {})
        if any(checks.get(stage, {}).get("status") != "passed" for stage in ("geometry", "mesh", "solve")):
            errors.append("Geometry, mesh and native solve must all pass.")
        if checks.get("geometry", {}).get("physical_assignments", {}).get("status") != "passed":
            errors.append("Saved native physical assignments were not verified.")
        if report.get("execution_error") or report.get("engine", {}).get("version_exit_code") != 0:
            errors.append("Native execution or engine inspection failed.")
        if report.get("parameters") != plan["parameters"]:
            errors.append("Audit requires the exact current catalog baseline parameters.")
        if report.get("plan_sha256") != plan["plan_sha256"]:
            errors.append("Native report is not bound to the current frozen baseline plan.")
        for key in ("knowledge_sources", "knowledge_artifacts", "frozen_reference", "frozen_references"):
            expected = plan.get(key, [])
            observed = report.get(key, [report.get("frozen_reference")] if key == "frozen_references" else [])
            if observed != expected:
                errors.append("Current baseline evidence differs: " + key)
        metrics = report.get("metrics", {})
        if not isinstance(metrics, dict) or not metrics or not all(_finite(v) for v in metrics.values()):
            errors.append("Native metrics must be nonempty finite numbers.")
        if errors:
            continue
        try:
            target = assess_target(plan, metrics)
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            errors.append("Cannot independently recompute native targets: " + str(exc))
            continue
        entry["published_target_status"] = target["status"]
        if target != checks.get("target"):
            errors.append("Reported target details differ from independently recomputed checks.")
        expected_status = "passed" if target["status"] == "passed" else "failed"
        if report.get("status") != expected_status:
            errors.append("Individual report status does not preserve its target result.")
        targets = target.get("targets", [target])
        failed = [item for item in targets if item.get("status") != "passed"]
        entry["failed_published_metrics"] = [item.get("metric") for item in failed]
        if target["status"] == "passed" and not errors:
            entry["status"] = "passed"
            if case_id in rules:
                audit["resolved_reference_discrepancies"].append({"case_id": case_id, "id": rules[case_id].get("id"),
                    "meaning": "All frozen published targets now pass; the historical discrepancy is not applied."})
            continue
        rule = rules.get(case_id)
        if not rule:
            errors.append("Unexpected published-target failure; no reviewed source-control discrepancy.")
            continue
        entry["reference_discrepancy"] = deepcopy(rule)
        if target.get("consistency", {}).get("status") == "failed" or not failed:
            errors.append("Physical consistency failure is not an allowed reference discrepancy.")
        if report["parameters"] != rule.get("parameters"):
            errors.append("Discrepancy control does not match these input parameters.")
        if not any(a.get("sha256") == rule.get("source_model_sha256") for a in report.get("knowledge_artifacts", [])):
            errors.append("Discrepancy is not bound to this exact original source model.")
        for key in ("protocol_sha256", "control"):
            if not rule.get(key):
                errors.append("Discrepancy record lacks " + key)
        if rule.get("protocol_targets_unchanged") is not True:
            errors.append("The original published-target protocol must remain unchanged.")
        control = rule.get("control", {})
        if control.get("source_model_sha256") != rule.get("source_model_sha256") or not control.get("evidence_sha256"):
            errors.append("Independent source-control provenance is incomplete.")
        allowed = {item["metric"]: item for item in rule.get("allowed_reference_failures", [])}
        checked = []
        for item in failed:
            metric = item.get("metric")
            allowance = allowed.get(metric)
            if not allowance:
                errors.append("Unreviewed failed reference metric: " + str(metric))
                continue
            reference = item["reference"]
            if any(reference.get(key) != allowance.get(key) for key in ("value", "relative_tolerance", "kind")):
                errors.append("Reviewed published reference changed: " + metric)
            baseline = control.get("metrics", {}).get(metric)
            tolerance = allowance.get("regression_relative_tolerance")
            if not _finite(baseline) or not baseline or not _finite(tolerance) or not 0 < tolerance <= .005:
                errors.append("Invalid independently established regression control: " + metric)
                continue
            difference = abs(metrics[metric] - baseline) / abs(baseline)
            checked.append({"metric": metric, "observed": metrics[metric], "original_source_control": baseline,
                            "regression_relative_difference": difference, "regression_relative_tolerance": tolerance,
                            "status": "passed" if difference <= tolerance else "failed",
                            "meaning": "Regression comparison only; published reference remains failed."})
            if difference > tolerance:
                errors.append("Native result differs from original-source control: " + metric)
        passed_metrics = {item.get("metric") for item in targets if item.get("status") == "passed"}
        if not set(rule.get("required_pass_metrics", [])).issubset(passed_metrics):
            errors.append("An independently required published target failed.")
        entry["original_source_regression_checks"] = checked
        if not errors:
            entry["status"] = "passed_with_reference_mismatch"
    if not issues and all(case["status"] != "failed" for case in audit["cases"]):
        audit["status"] = ("passed_with_reference_mismatch" if any(case["status"] == "passed_with_reference_mismatch"
                           for case in audit["cases"]) else "passed")
        audit["all_published_targets_passed"] = audit["status"] == "passed"
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    args = parser.parse_args(argv)
    try:
        result = audit_summary(json.loads(args.summary.read_text()), json.loads(args.registry.read_text()))
        result["summary_sha256"] = _digest(args.summary)
        result["discrepancy_registry_sha256"] = _digest(args.registry)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": result["status"], "all_published_targets_passed": result["all_published_targets_passed"],
                      "audit": str(args.output)}, indent=2))
    return 0 if result["status"] in {"passed", "passed_with_reference_mismatch"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
