"""Frozen target and retained-error invariants; optional full native loop."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.history import ModelRevisionStore
from app.jobs import JobStore, SolverWorker
from app.targets import (contract_fingerprint, evaluate_targets, freeze_targets,
                         rebind_targets)
from evaluation.recovery_loop import (CASE_ID, CURRENT_PATH, reference_plan,
                                      reference_target, run_benchmark, verify_outcome)


class RecoveryLoopContractTests(unittest.TestCase):
    def test_analytical_target_is_frozen_before_current_injection_and_never_refitted(self):
        baseline = reference_plan()
        target = reference_target()
        self.assertAlmostEqual(target.value, .0001)
        self.assertEqual(target.tolerance.relative, .03)
        self.assertEqual(target.observable.quantity, "by")
        self.assertEqual(target.observable.probe_index, 0)
        self.assertEqual(baseline.mesh_mm, .2)
        contract = freeze_targets([target], baseline, case_id=CASE_ID)
        with tempfile.TemporaryDirectory() as directory:
            history = ModelRevisionStore(Path(directory))
            initial = history.create(baseline)
            wrong = history.apply_patch(initial["id"], [{"op": "replace", "path": CURRENT_PATH, "value": 2.0}],
                                        allowed_paths=[CURRENT_PATH])
            wrong_contract = rebind_targets(contract, wrong["plan"])
        self.assertNotEqual(contract.model_sha256, wrong_contract.model_sha256)
        self.assertEqual(contract.targets_sha256, wrong_contract.targets_sha256)
        self.assertEqual(contract.targets, wrong_contract.targets)
        self.assertNotEqual(contract_fingerprint(contract), contract_fingerprint(wrong_contract))
        # These are explicit synthetic quantities testing the comparison contract,
        # not a native FEMM result or a measurement of numerical accuracy.
        right_result = evaluate_targets(contract, {"by_0": {"value": .0001, "unit": "T"}}, plan=baseline)
        wrong_result = evaluate_targets(wrong_contract, {"by_0": {"value": .0002, "unit": "T"}}, plan=wrong["plan"])
        self.assertEqual(right_result["status"], "passed")
        self.assertEqual(wrong_result["status"], "failed")
        self.assertAlmostEqual(wrong_result["checks"][0]["target"], .0001)
        self.assertEqual(wrong_result["checks"][0]["tolerance"], {"absolute": 0.0, "relative": .03})
        with self.assertRaisesRegex(ValueError, "different model revision"):
            evaluate_targets(contract, {"by_0": {"value": .0001, "unit": "T"}}, plan=wrong["plan"])

    def test_outcome_requires_both_failed_target_and_preserved_wrong_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            history = ModelRevisionStore(Path(directory))
            baseline = history.create(reference_plan())
            wrong = history.apply_patch(baseline["id"], [{"op": "replace", "path": CURRENT_PATH, "value": 2.0}],
                                        allowed_paths=[CURRENT_PATH])
            failure = history.record_check(wrong["id"], stage="target", status="failed", job_id="fixture-target-failure",
                                           details={"synthetic_fixture": True}, implicated_paths=[CURRENT_PATH])
            recovered = history.branch_recovery(failure["id"], baseline["id"], branch="recovered",
                diagnosis="Restore the independently specified 1 A value after the known injection.")
            context = history.recovery_context(failure["id"])
            self.assertEqual(context["candidates"][0]["revision_id"], wrong["id"])
            attempts = [{"mesh_status": "passed", "solver_status": "passed", "target_status": status,
                         "target_check_id": failure["id"], "target_report": {"targets_sha256": "a" * 64}}
                        for status in ("passed", "failed", "passed")]
            self.assertTrue(verify_outcome(attempts, baseline, wrong, recovered, history)["passed"])
            attempts[1]["target_status"] = "passed"
            outcome = verify_outcome(attempts, baseline, wrong, recovered, history)
            self.assertFalse(outcome["passed"])
            self.assertFalse(outcome["checks"]["injected_model_mesh_solver_pass_but_target_fails"])
            attempts[1]["target_status"] = "failed"
            attempts[2]["target_report"]["targets_sha256"] = "b" * 64
            self.assertFalse(verify_outcome(attempts, baseline, wrong, recovered, history)["checks"]["frozen_target_definitions_unchanged"])

    def test_nonempty_output_preserves_prior_evidence_before_trying_native_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            evidence = output / "previous-evidence.json"
            evidence.write_text('{"preserve":true}')
            with self.assertRaisesRegex(ValueError, "empty output"):
                run_benchmark(output, command="nonexistent-native-launcher")
            self.assertEqual(evidence.read_text(), '{"preserve":true}')

    def test_full_solve_attributes_mesh_and_field_parser_failures_to_actual_stage(self):
        # /bin/true is an explicit harness fixture, not a FEMM numerical test.
        for failure_stage in ("mesh", "solver"):
            with self.subTest(stage=failure_stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                history = ModelRevisionStore(root)
                revision = history.create(reference_plan())
                store = JobStore(root)
                job = store.create(reference_plan(), kind="solve", revision_id=revision["id"])
                worker = SolverWorker(store, command="/bin/true", revision_store=history)
                mesh_failure = RuntimeError("Explicit mesh validation fixture") if failure_stage == "mesh" else None
                with patch("app.jobs.validate_mesh", side_effect=mesh_failure,
                           return_value={"status": "passed"}), \
                     patch("app.jobs.read_results", side_effect=RuntimeError("Explicit field-parser fixture")):
                    worker.run(job)
                finished = store.get(job["id"])
                self.assertEqual(finished["status"], "failed")
                observed = history.get_check(finished["evidence"]["failure_id"])
                self.assertEqual(observed["stage"], failure_stage)
                self.assertEqual(observed["status"], "failed")
                self.assertTrue((root / job["id"] / (failure_stage + "-execution.json")).is_file())


@unittest.skipUnless(os.getenv("FEMM_LIVE_TEST") == "1", "Set FEMM_LIVE_TEST=1 in the patched native FEMM Docker image")
class RecoveryLoopNativeTests(unittest.TestCase):
    def test_actual_mesh_solver_target_failure_and_checkpoint_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "native-recovery"
            result = run_benchmark(output)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["mode"], "deterministic_known_injection")
            self.assertEqual(result["metrics"]["native_invocations"], 6)
            self.assertEqual(result["provider_usage"]["calls"], 0)
            self.assertEqual([attempt["target_status"] for attempt in result["attempts"]], ["passed", "failed", "passed"])
            self.assertTrue(all(result["verification"]["checks"].values()))
            self.assertEqual(json.loads((output / "summary.json").read_text())["status"], "passed")
            for attempt in result["attempts"]:
                self.assertIn("model.ans", attempt["artifacts"]["solve"])
                self.assertIn("results.csv", attempt["artifacts"]["solve"])
                self.assertIn("mesh-report.json", attempt["artifacts"]["mesh"])
                self.assertEqual(attempt["target_report"]["targets_sha256"], result["frozen_targets_sha256"])


if __name__ == "__main__":
    unittest.main()
