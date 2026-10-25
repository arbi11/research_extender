"""A documented source discrepancy cannot hide a new physical or target failure."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from unittest.mock import patch

from femm_brain.benchmark_audit import REGISTRY, audit_summary, main
from femm_brain.simulation import assess_target, plan_simulation


class BenchmarkAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = plan_simulation("gapped-ei-core")
        cls.registry = json.loads(REGISTRY.read_text())

    def fixture(self):
        plan = deepcopy(self.plan)
        metrics = {"inductance_H": .0017560682796706, "energy_J": .00087803413983534,
                   "energy_inductance_H": .00175606827967068,
                   "force_I_x_N": -.0010695610743749, "force_I_y_N": 1.108276659203}
        report = {"case_id": plan["case_id"], "status": "failed", "parameters": plan["parameters"],
                  "plan_sha256": plan["plan_sha256"],
                  "engine": {"version_exit_code": 0}, "metrics": metrics,
                  "checks": {stage: {"status": "passed"} for stage in ("geometry", "mesh", "solve")}}
        for key in ("knowledge_sources", "knowledge_artifacts", "frozen_reference", "frozen_references"):
            report[key] = deepcopy(plan[key])
        report["checks"]["geometry"]["physical_assignments"] = {"status": "passed"}
        report["checks"]["target"] = assess_target(plan, metrics)
        return {"status": "failed", "cases": [report]}, {plan["case_id"]: plan}

    def audit(self, summary, plans, **kwargs):
        return audit_summary(summary, deepcopy(self.registry),
                             graph=kwargs.get("graph", {"validation": {"status": "passed"}, "source_changes": []}), plans=plans)

    def test_known_discrepancy_is_explicit_and_input_evidence_unchanged(self):
        summary, plans = self.fixture()
        before = deepcopy(summary)
        result = self.audit(summary, plans)
        self.assertEqual(result["status"], "passed_with_reference_mismatch")
        self.assertFalse(result["all_published_targets_passed"])
        self.assertEqual(result["cases"][0]["report_status"], "failed")
        self.assertEqual(summary, before)

    def test_bad_native_stage_cannot_be_waived(self):
        for stage in ("geometry", "mesh", "solve"):
            with self.subTest(stage=stage):
                summary, plans = self.fixture()
                summary["cases"][0]["checks"][stage]["status"] = "failed"
                self.assertEqual(self.audit(summary, plans)["status"], "failed")

    def test_old_compiler_plan_cannot_be_accepted_as_current_evidence(self):
        summary, plans = self.fixture()
        summary['cases'][0]['plan_sha256'] = 'old-compiler-plan'
        self.assertEqual(self.audit(summary, plans)['status'], 'failed')

    def test_force_failure_cannot_be_waived(self):
        summary, plans = self.fixture()
        report = summary["cases"][0]
        report["metrics"]["force_I_y_N"] = 1.0
        report["metrics"]["force_I_x_N"] = 0
        report["checks"]["target"] = assess_target(self.plan, report["metrics"])
        self.assertEqual(self.audit(summary, plans)["status"], "failed")

    def test_results_outside_original_source_control_are_regressions(self):
        summary, plans = self.fixture()
        report = summary["cases"][0]
        for metric in ("inductance_H", "energy_J", "energy_inductance_H"):
            report["metrics"][metric] *= 1.02
        report["checks"]["target"] = assess_target(self.plan, report["metrics"])
        self.assertEqual(self.audit(summary, plans)["status"], "failed")

    def test_future_published_reference_agreement_resolves_historical_exception(self):
        summary, plans = self.fixture()
        report = summary["cases"][0]
        report["metrics"].update(inductance_H=.001730084, energy_J=.000865042,
                                  energy_inductance_H=.001730084, force_I_y_N=1.11347, force_I_x_N=0)
        report["checks"]["target"] = assess_target(self.plan, report["metrics"])
        report["status"] = summary["status"] = "passed"
        result = self.audit(summary, plans)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["all_published_targets_passed"])
        self.assertEqual(len(result["resolved_reference_discrepancies"]), 1)

    def test_unexpected_case_missing_case_and_duplicate_case_are_rejected(self):
        for modification in ("missing", "duplicate", "unknown"):
            with self.subTest(modification=modification):
                summary, plans = self.fixture()
                if modification == "missing":
                    plans["another"] = deepcopy(self.plan)
                elif modification == "duplicate":
                    summary["cases"].append(deepcopy(summary["cases"][0]))
                else:
                    summary["cases"][0]["case_id"] = "unknown"
                self.assertEqual(self.audit(summary, plans)["status"], "failed")

    def test_changed_source_hash_parameters_or_tolerance_are_rejected(self):
        for field in ("source", "parameters", "tolerance"):
            with self.subTest(field=field):
                summary, plans = self.fixture()
                report = summary["cases"][0]
                if field == "source":
                    report["knowledge_artifacts"][0]["sha256"] = "different"
                elif field == "parameters":
                    report["parameters"]["current"] = 2
                else:
                    report["frozen_references"][0]["relative_tolerance"] = .1
                self.assertEqual(self.audit(summary, plans)["status"], "failed")

    def test_stale_source_and_report_relabeling_are_rejected(self):
        summary, plans = self.fixture()
        graph = {"validation": {"status": "passed"}, "source_changes": ["inductanceexample"]}
        self.assertEqual(self.audit(summary, plans, graph=graph)["status"], "failed")
        summary["cases"][0]["status"] = "passed"
        self.assertEqual(self.audit(summary, plans)["status"], "failed")

    def test_missing_physics_or_nonfinite_metrics_are_rejected(self):
        for issue in ("physics", "metrics"):
            summary, plans = self.fixture()
            report = summary["cases"][0]
            if issue == "physics":
                del report["checks"]["geometry"]["physical_assignments"]
            else:
                report["metrics"]["energy_J"] = float("nan")
            self.assertEqual(self.audit(summary, plans)["status"], "failed")

    def test_cli_preserves_existing_audit_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, plans = self.fixture()
            source = root / "summary.json"
            source.write_text(json.dumps(summary))
            output = root / "audit.json"
            output.write_text("existing evidence")
            with patch("femm_brain.benchmark_audit.audit_summary", return_value=self.audit(summary, plans)):
                with self.assertRaises(SystemExit) as stopped:
                    main(["--summary", str(source), "--output", str(output)])
            self.assertEqual(stopped.exception.code, 2)
            self.assertEqual(output.read_text(), "existing evidence")


if __name__ == "__main__":
    unittest.main()
