"""Public advisory benchmark fixtures/accounting, without keys or provider calls."""
import asyncio
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.plan import GeometryPlan
from app.vision import VisualReview, VisionError
from evaluation.vision_benchmark import DEFAULT_MODELS, build_cases, main, run_benchmark, summarize


def advisory(status="matches", usage=None):
    return {"advisory": True, "status": status, "summary": "Public fixture observation.",
            "findings": [], "requested_model": "test/model", "usage": usage or {}}


class FixtureTests(unittest.TestCase):
    def test_all_public_candidates_are_valid_and_independent(self):
        cases = build_cases()
        self.assertEqual(len(cases), 6)
        self.assertEqual(len({case.id for case in cases}), 6)
        self.assertEqual([case.expected for case in cases].count("matches"), 2)
        for case in cases:
            with self.subTest(case=case.id):
                self.assertEqual(GeometryPlan.model_validate(case.fixture()["candidate"]), case.candidate)
                self.assertGreater(len(case.prompt), 5)
        before = cases[0].candidate.model_dump_json()
        cases[1].candidate.regions[0].shape.radius = 1.8
        self.assertEqual(cases[0].candidate.model_dump_json(), before)

    def test_wrong_radius_changes_only_radius_and_keeps_probe_outside(self):
        cases = {case.id: case for case in build_cases()}
        good, wrong = cases["correct_wire"], cases["wrong_wire_radius"]
        self.assertEqual(good.prompt, wrong.prompt)
        self.assertIn("radius 1 mm", wrong.prompt)
        self.assertIn("(3,0)", wrong.prompt)
        data = wrong.candidate.model_dump(mode="json")
        self.assertEqual(data["regions"][0]["shape"]["radius"], 1.6)
        self.assertEqual(data["probes"][0], {"x": 3.0, "y": 0.0})
        data["regions"][0]["shape"]["radius"] = 1.0
        self.assertEqual(data, good.candidate.model_dump(mode="json"))
        self.assertEqual(wrong.expected, "issues")

    def test_missing_return_removes_one_negative_winding_and_keeps_circuit_used(self):
        cases = {case.id: case for case in build_cases()}
        good, missing = cases["correct_asymmetric_core"], cases["missing_return_winding"]
        self.assertEqual(good.prompt, missing.prompt)
        self.assertIn("-50 turns", missing.prompt)
        self.assertEqual(len(good.candidate.regions), 3)
        self.assertEqual(len(missing.candidate.regions), 2)
        self.assertEqual({region.id for region in good.candidate.regions} - {region.id for region in missing.candidate.regions}, {"coil2"})
        winding = next(region for region in missing.candidate.regions if region.circuit)
        self.assertEqual(winding.turns, 50)
        self.assertEqual(winding.circuit, missing.candidate.circuits[0].id)
        self.assertEqual(missing.candidate.circuits, good.candidate.circuits)
        self.assertEqual(missing.expected, "issues")

    def test_reversed_turns_and_complete_mirror_preserve_validity(self):
        cases = {case.id: case for case in build_cases()}
        good = cases["correct_asymmetric_core"].candidate
        wrong = cases["reversed_return_turns"].candidate
        self.assertEqual(next(region for region in wrong.regions if region.id == "coil2").turns, 50)
        mirrored = cases["mirrored_asymmetric_core"].candidate
        self.assertEqual(mirrored.domain.x_min, -good.domain.x_max)
        self.assertEqual(mirrored.domain.x_max, -good.domain.x_min)
        self.assertEqual(mirrored.air_label.x, -good.air_label.x)
        self.assertEqual(mirrored.probes[0].x, -good.probes[0].x)
        for original, reflected in zip(good.regions, mirrored.regions):
            self.assertEqual(reflected.label.x, -original.label.x)
            self.assertEqual(reflected.label.y, original.label.y)

    def test_cli_requires_live_and_bounds_before_key_or_provider(self):
        for args in (["--output", "never-created.json"], ["--live", "--output", "never-created.json", "--max-cases", "13"],
                     ["--live", "--output", "never-created.json", "--model", "a", "--model", "b", "--model", "c"]):
            with self.subTest(args=args), patch("evaluation.vision_benchmark.vision.review") as call, patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    main(args)
                call.assert_not_called()


class AccountingTests(unittest.TestCase):
    def test_status_counts_usage_uncertainty_errors_and_unmeasured_are_distinct(self):
        cases = build_cases()
        statuses = ["matches", "matches", "issues", "issues", "uncertain"]
        rows = [{"id": case.id, "requested_model": "a", "advisory": advisory(status),
                 "usage": {"cost": .01, "total_tokens": 10} if index == 0 else {}, "latency_seconds": index + 1}
                for index, (case, status) in enumerate(zip(cases, statuses))]
        rows.append({"id": cases[-1].id, "requested_model": "a", "error": {"status_code": 502, "message": "Safe error."}, "latency_seconds": 6})
        summary = summarize(cases, rows, ["a", "b"])
        self.assertEqual({key: summary["a"][key] for key in ("attempted", "evaluated", "matches", "false_flags", "missed_issues", "uncertain", "errors", "unmeasured")},
                         {"attempted": 6, "evaluated": 5, "matches": 2, "false_flags": 1, "missed_issues": 1, "uncertain": 1, "errors": 1, "unmeasured": 0})
        self.assertEqual(summary["a"]["cost"], .01)
        self.assertEqual(summary["a"]["cost_reported_rows"], 1)
        self.assertEqual(summary["a"]["cost_unknown_rows"], 5)
        self.assertEqual(summary["a"]["tokens"]["total_tokens"], 10)
        self.assertIsNone(summary["a"]["tokens"]["prompt_tokens"])
        self.assertEqual(summary["a"]["latency_seconds"], {"total": 21, "mean": 3.5, "max": 6})
        self.assertEqual(summary["b"]["unmeasured"], 6)
        self.assertIsNone(summary["b"]["cost"])
        self.assertIsNone(summary["b"]["latency_seconds"]["mean"])

    def test_empty_invalid_and_duplicate_evidence_is_never_counted_correct(self):
        cases = build_cases()[:1]
        for rows in ([{"id": cases[0].id, "requested_model": "a"}],
                     [{"id": "unknown", "requested_model": "a", "advisory": advisory()}],
                     [{"id": cases[0].id, "requested_model": "a", "advisory": advisory()}] * 2):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize(cases, rows, ["a"])
        result = summarize(cases, [{"id": cases[0].id, "requested_model": "a", "error": {"message": "Unavailable."}}], ["a"])["a"]
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["matches"], 0)
        self.assertEqual(result["evaluated"], 0)
        self.assertIsNone(result["cost"])


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_sequential_mock_calls_save_public_fixtures_actual_usage_and_safe_errors(self):
        cases = build_cases()[:2]
        calls = []
        active = 0

        async def fake_review(client, key, base_url, model, prompt, candidate):
            nonlocal active
            active += 1
            await asyncio.sleep(0)
            self.assertEqual(active, 1)
            calls.append((model, prompt, candidate))
            active -= 1
            if len(calls) == 2:
                raise VisionError(502, "The visual-review provider is unavailable.")
            status = "issues" if candidate.regions[0].shape.radius != 1 else "matches"
            return VisualReview(**{**advisory(status, {"total_tokens": 100, "cost": .0001}), "requested_model": model})

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.json"
            with patch("evaluation.vision_benchmark.vision.review", side_effect=fake_review), patch("evaluation.vision_benchmark.vision._render_png", return_value=b"\x89PNG\r\n\x1a\npublic-fixture"), patch("builtins.print"):
                report = await run_benchmark(cases, list(DEFAULT_MODELS), output, key="private-test-key")
            saved = json.loads(output.read_text())
            self.assertEqual(saved, report)
            self.assertEqual(len(calls), 4)
            self.assertEqual([call[0] for call in calls], [DEFAULT_MODELS[0]] * 2 + [DEFAULT_MODELS[1]] * 2)
            self.assertEqual(report["planned_calls"], 4)
            self.assertFalse(report["production_gate"])
            self.assertTrue(report["advisory_only"])
            self.assertEqual(report["summary"][DEFAULT_MODELS[0]]["errors"], 1)
            self.assertEqual(report["summary"][DEFAULT_MODELS[1]]["matches"], 2)
            self.assertNotIn("private-test-key", output.read_text())
            self.assertEqual(report["fixtures"][1]["candidate"], cases[1].candidate.model_dump(mode="json"))
            self.assertEqual(report["fixtures"][1]["prompt"], cases[1].prompt)
            for fixture in report["fixtures"]:
                self.assertTrue((output.parent / fixture["geometry_png"]).read_bytes().startswith(b"\x89PNG"))
            self.assertFalse(output.with_name(output.name + ".tmp").exists())
            with patch("evaluation.vision_benchmark.vision.review") as call:
                with self.assertRaises(FileExistsError):
                    await run_benchmark(cases, ["a"], output, key="private-test-key")
                call.assert_not_called()

    async def test_render_failure_stops_before_any_paid_call_and_is_saved(self):
        with tempfile.TemporaryDirectory() as directory, patch("evaluation.vision_benchmark.vision._render_png", side_effect=VisionError(503, "Rendering unavailable.")), patch("evaluation.vision_benchmark.vision.review") as call:
            output = Path(directory) / "render-error.json"
            report = await run_benchmark(build_cases()[:1], ["a"], output, key="private-test-key")
            self.assertEqual(report["setup_error"]["status_code"], 503)
            self.assertEqual(report["rows"], [])
            self.assertEqual(report["summary"]["a"]["unmeasured"], 1)
            self.assertEqual(json.loads(output.read_text()), report)
            call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
