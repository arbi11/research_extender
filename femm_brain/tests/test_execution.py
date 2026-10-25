"""Execution preserves frozen plans and failure evidence without a live Docker call."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from femm_brain.execution import SimulationJobs, run_case


class DockerExecutionTests(unittest.TestCase):
    def execute(self, output, side_effect):
        with patch("femm_brain.simulation.plan_simulation", return_value={"plan_sha256": "frozen"}), \
             patch("femm_brain.execution.shutil.which", return_value="/usr/bin/docker"), \
             patch("femm_brain.execution.subprocess.run", side_effect=side_effect) as invoked:
            return run_case("aircore-short", output), invoked

    def test_only_package_and_fresh_result_directory_are_mounted(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            calls = []
            def command(args, **kwargs):
                calls.append(args)
                if args[1] == "image":
                    return subprocess.CompletedProcess(args, 0, "sha256:pinned\n", "")
                (output / "report.json").write_text(json.dumps({"status": "passed", "plan_sha256": "frozen"}))
                return subprocess.CompletedProcess(args, 0, "native finished", "")
            report, _ = self.execute(output, command)
            invocation = calls[1]
            mounts = [invocation[index + 1] for index, arg in enumerate(invocation) if arg == "-v"]
            self.assertTrue(any(value.endswith("/femm_brain:/repo/femm_brain:ro") for value in mounts))
            self.assertFalse(any(value.endswith(":/repo:ro") for value in mounts))
            self.assertIn(str(output) + ":/reports", mounts)
            self.assertEqual(invocation[invocation.index("--network") + 1], "none")
            self.assertIn("--read-only", invocation)
            self.assertIn("sha256:pinned", invocation)
            self.assertNotIn("--env-file", invocation)
            self.assertFalse(report["execution"]["provider_credentials_passed"])

    def test_timeout_retains_partial_output_and_cleanup_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            calls = []
            def command(args, **kwargs):
                calls.append(args)
                if args[1] == "image":
                    return subprocess.CompletedProcess(args, 0, "sha256:pinned\n", "")
                if args[1] == "run":
                    (output / "plan.json").write_text("retained plan")
                    raise subprocess.TimeoutExpired(args, 240, output=b"partial native stdout", stderr=b"partial native stderr")
                return subprocess.CompletedProcess(args, 0, "removed container", "")
            with self.assertRaisesRegex(RuntimeError, "execution time limit"):
                self.execute(output, command)
            receipt = json.loads(output.with_name("run-execution.json").read_text())
            self.assertEqual(receipt["status"], "timed_out")
            self.assertEqual(receipt["stdout"], "partial native stdout")
            self.assertEqual(receipt["stderr"], "partial native stderr")
            self.assertEqual(receipt["cleanup"]["status"], "completed")
            self.assertEqual(calls[-1][1:3], ["rm", "-f"])
            self.assertEqual((output / "plan.json").read_text(), "retained plan")

    def test_failed_cleanup_does_not_erase_timeout_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            def command(args, **kwargs):
                if args[1] == "image":
                    return subprocess.CompletedProcess(args, 0, "sha256:pinned\n", "")
                raise subprocess.TimeoutExpired(args, 240, output="still retained", stderr="cleanup diagnostic")
            with self.assertRaises(RuntimeError):
                self.execute(output, command)
            receipt = json.loads(output.with_name("run-execution.json").read_text())
            self.assertEqual(receipt["stdout"], "still retained")
            self.assertEqual(receipt["cleanup"]["status"], "failed")
            self.assertEqual(receipt["cleanup"]["stderr"], "cleanup diagnostic")

    def test_existing_receipt_is_not_overwritten_even_with_an_empty_run_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            receipt = output.with_name("run-execution.json")
            receipt.write_text("previous attempt")
            with patch("femm_brain.simulation.plan_simulation", return_value={"plan_sha256": "frozen"}), \
                 patch("femm_brain.execution.subprocess.run") as command:
                with self.assertRaisesRegex(ValueError, "previous execution receipt"):
                    run_case("aircore-short", output)
                command.assert_not_called()
            self.assertEqual(receipt.read_text(), "previous attempt")

    def test_failed_container_exit_cannot_return_a_passing_overall_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            original = {"status": "passed", "plan_sha256": "frozen"}
            def command(args, **kwargs):
                if args[1] == "image":
                    return subprocess.CompletedProcess(args, 0, "sha256:pinned\n", "")
                (output / "report.json").write_text(json.dumps(original))
                return subprocess.CompletedProcess(args, 1, "", "process exit error")
            result, _ = self.execute(output, command)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["native_report_original_status"], "passed")
            self.assertEqual(json.loads((output / "report.json").read_text()), original)


class FrozenQueueTests(unittest.TestCase):
    def test_stale_displayed_plan_is_rejected_before_queueing(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("femm_brain.execution.ThreadPoolExecutor") as pool, \
             patch("femm_brain.simulation.plan_simulation", return_value={"plan_sha256": "current"}):
            jobs = SimulationJobs(directory)
            with self.assertRaisesRegex(ValueError, "displayed frozen plan has changed"):
                jobs.submit("aircore-short", {}, expected_plan_sha256="previous")
            self.assertEqual(jobs.jobs, {})
            pool.return_value.submit.assert_not_called()
            jobs.close()

    def test_executed_plan_hash_mismatch_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("femm_brain.execution.ThreadPoolExecutor"), \
             patch("femm_brain.simulation.plan_simulation", return_value={"plan_sha256": "submitted"}), \
             patch("femm_brain.execution.run_case", return_value={"status": "passed", "plan_sha256": "changed", "checks": {}}):
            jobs = SimulationJobs(directory)
            queued = jobs.submit("aircore-short", {}, expected_plan_sha256="submitted")
            jobs._run(queued["id"], "aircore-short", {})
            completed = jobs.get(queued["id"])
            self.assertEqual(completed["status"], "failed")
            self.assertEqual(completed["report"]["status"], "failed")
            self.assertEqual(completed["report"]["checks"]["submitted_plan_binding"]["executed_plan_sha256"], "changed")
            retained = json.loads(jobs.artifact(queued["id"], "job.json").read_text())
            self.assertEqual(retained, completed)
            self.assertEqual(retained["report"]["native_report_original_status"], "passed")
            self.assertEqual(retained["expected_displayed_plan_sha256"], "submitted")
            jobs.close()

    def test_matching_plan_hash_accepts_the_actual_native_report(self):
        report = {"status": "passed", "plan_sha256": "submitted", "checks": {}}
        with tempfile.TemporaryDirectory() as directory, \
             patch("femm_brain.execution.ThreadPoolExecutor"), \
             patch("femm_brain.simulation.plan_simulation", return_value={"plan_sha256": "submitted"}), \
             patch("femm_brain.execution.run_case", return_value=report):
            jobs = SimulationJobs(directory)
            queued = jobs.submit("aircore-short", {}, expected_plan_sha256="submitted")
            retained = json.loads(jobs.artifact(queued["id"], "job.json").read_text())
            self.assertEqual(retained["status"], "queued")
            jobs._run(queued["id"], "aircore-short", {})
            self.assertEqual(jobs.get(queued["id"])["report"], report)
            retained = json.loads(jobs.artifact(queued["id"], "job.json").read_text())
            self.assertEqual(retained["status"], "completed")
            self.assertIn("started_utc", retained)
            self.assertIn("completed_utc", retained)
            jobs.close()


class FrozenPlanCliTests(unittest.TestCase):
    def test_case_and_natural_language_cannot_silently_override_each_other(self):
        from femm_brain.__main__ import main
        with redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as exit_code:
                main(["plan", "uniform planar field of 1 T", "--case", "aircore-short"])
        self.assertEqual(exit_code.exception.code, 2)
        self.assertIn("do not combine", stderr.getvalue())

    def test_frozen_plan_file_is_created_exclusively(self):
        from femm_brain.__main__ import main
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["plan", "--case", "aircore-short", "--output", str(path)]), 0)
            first = path.read_bytes()
            with redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit):
                    main(["plan", "--case", "aircore-long", "--output", str(path)])
            self.assertIn("Refusing to overwrite", stderr.getvalue())
            self.assertEqual(path.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
