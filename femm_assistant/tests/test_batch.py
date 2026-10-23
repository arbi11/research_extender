"""Golden batch fixtures and aggregation checks never call an AI provider."""
import copy
from contextlib import closing, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from fastapi.testclient import TestClient

from app.main import Settings, create_app
from app.plan import GeometryPlan, contains
from evaluation.batch import load_cases, main, physical_signature, save_report, summarize


class BatchFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()
        cls.by_id = {case.id: case for case in cls.cases}

    def test_all_cases_are_complete_isolated_and_have_two_phrasings(self):
        self.assertEqual(len(self.cases), 28)
        self.assertEqual(sum(case.category == "create" for case in self.cases), 18)
        self.assertEqual(sum(case.category == "modify" for case in self.cases), 5)
        self.assertEqual(sum(case.category == "clarify" for case in self.cases), 5)
        for case in self.cases:
            with self.subTest(case=case.id):
                self.assertEqual(len(case.prompts), 2)
                self.assertNotEqual(*case.prompts)
                if case.expected is None:
                    self.assertEqual(case.category, "clarify")
                    continue
                plan = GeometryPlan.model_validate(case.expected.model_dump())
                self.assertTrue(contains(plan.domain, plan.air_label))
                self.assertNotIn("air", {material.id for material in plan.materials})
                for region in plan.regions:
                    self.assertTrue(contains(region.shape, region.label))
                if case.category == "modify":
                    self.assertIsNotNone(case.current)
                    self.assertNotEqual(physical_signature(case.current), physical_signature(plan))

    def test_expected_materials_nested_labels_and_signed_excitation(self):
        coax = self.by_id["coax_nested_air"].expected
        by_radius = {region.shape.radius: region for region in coax.regions}
        self.assertEqual(by_radius[5].turns, -1)
        self.assertEqual(by_radius[1].turns, 1)
        self.assertEqual(by_radius[3].material, "air")
        self.assertIsNone(by_radius[3].circuit)
        self.assertEqual(by_radius[5].label.x, 4)
        self.assertEqual(by_radius[3].label.x, 2)
        self.assertEqual(coax.circuits[0].current_a, 2)
        opposed = self.by_id["opposed_rectangular_windings"].expected
        self.assertEqual(sorted(region.turns for region in opposed.regions), [-20, 0, 20])
        self.assertEqual(sorted(material.relative_permeability for material in opposed.materials), [1, 500])
        units = self.by_id["centimeter_milliamp_units"].expected
        self.assertEqual(units.regions[0].shape.radius, 1)
        self.assertEqual(units.depth_mm, 20)
        self.assertEqual(units.mesh_mm, .5)
        self.assertEqual(units.circuits[0].current_a, 1.5)
        self.assertEqual(self.by_id["modify_negative_current"].expected.circuits[0].current_a, -2)
        passive = self.by_id["passive_two_material_force"].expected
        self.assertEqual(passive.circuits, [])
        self.assertEqual(passive.force_region, "rect_core")

    def test_signature_ignores_ids_titles_and_resolves_force_target(self):
        original = self.by_id["passive_two_material_force"].expected.model_dump(mode="json")
        renamed = copy.deepcopy(original)
        renamed["title"] = "A different display title"
        for index, material in enumerate(renamed["materials"]):
            previous = material["id"]
            material["id"] = f"material_{index}"
            for region in renamed["regions"]:
                if region["material"] == previous:
                    region["material"] = material["id"]
        for index, region in enumerate(renamed["regions"]):
            if renamed["force_region"] == region["id"]:
                renamed["force_region"] = f"region_{index}"
            region["id"] = f"region_{index}"
        self.assertEqual(physical_signature(original), physical_signature(renamed))
        renamed["force_region"] = renamed["regions"][0]["id"]
        self.assertNotEqual(physical_signature(original), physical_signature(renamed))

    def test_signature_normalizes_polygon_direction_and_rectangle_contours(self):
        original = self.by_id["concave_l_core"].expected.model_dump(mode="json")
        reordered = copy.deepcopy(original)
        points = reordered["regions"][0]["shape"]["points"]
        reordered["regions"][0]["shape"]["points"] = list(reversed(points[2:] + points[:2]))
        rectangle = reordered["regions"][1]["shape"]
        reordered["regions"][1]["shape"] = {"type": "polygon", "points": [
            {"x": rectangle["x_min"], "y": rectangle["y_min"]}, {"x": rectangle["x_max"], "y": rectangle["y_min"]},
            {"x": rectangle["x_max"], "y": rectangle["y_max"]}, {"x": rectangle["x_min"], "y": rectangle["y_max"]}]}
        self.assertEqual(physical_signature(original), physical_signature(reordered))

    def test_signature_rejects_wrong_turns_material_current_and_circuit_membership(self):
        original = self.by_id["round_wire"].expected.model_dump(mode="json")
        for path, replacement in (("turns", -1), ("current_a", 2), ("relative_permeability", 100)):
            with self.subTest(path=path):
                changed = copy.deepcopy(original)
                target = changed["regions"][0] if path == "turns" else changed["circuits"][0] if path == "current_a" else changed["materials"][0]
                target[path] = replacement
                self.assertNotEqual(physical_signature(original), physical_signature(changed))
        separate = self.by_id["two_independent_circuits"].expected.model_dump(mode="json")
        for circuit in separate["circuits"]:
            circuit["current_a"] = 1
        merged = copy.deepcopy(separate)
        merged["circuits"] = [merged["circuits"][0]]
        for region in merged["regions"]:
            region["circuit"] = merged["circuits"][0]["id"]
        self.assertNotEqual(physical_signature(separate), physical_signature(merged))

    def test_golden_models_compile_through_real_preview_and_source_export_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(Settings(data_dir=Path(directory), start_worker=False, ai_key=""))
            with TestClient(app) as client:
                for case in self.cases:
                    if case.expected is None:
                        continue
                    with self.subTest(case=case.id):
                        data = case.expected.model_dump(mode="json")
                        preview = client.post("/api/plan/preview", json=data)
                        self.assertEqual(preview.status_code, 200, preview.text)
                        self.assertIn("<svg", preview.text)
                        response = client.post("/api/plan/export", json=data)
                        self.assertEqual(response.status_code, 200)
                        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                            self.assertEqual(physical_signature(json.loads(archive.read("spec.json"))), physical_signature(data))
                            source = archive.read("model.lua").decode()
                            self.assertEqual(source.count("mi_addblocklabel("), len(case.expected.regions) + 1)
                            for region in case.expected.regions:
                                self.assertIn(f'mi_addblocklabel({float(region.label.x)!r},{float(region.label.y)!r})', source)
                with closing(app.state.store.connect()) as database:
                    self.assertEqual(database.execute("SELECT count(*) FROM jobs").fetchone()[0], 0)


class BatchReportTests(unittest.TestCase):
    def test_aggregation_retains_failed_rounds_and_counts_repairs_and_stage_failures(self):
        rows = [{"id": "same_case", "round": 1, "status": "failed", "attempts": 2, "latency_seconds": 10,
                 "failures": [{"stage": "physical_signature", "error": "Wrong turns"}]},
                {"id": "same_case", "round": 2, "status": "passed", "attempts": 1, "latency_seconds": 20, "failures": []},
                {"id": "other", "round": 1, "status": "error", "attempts": 0, "latency_seconds": 30,
                 "failures": [{"stage": "planner", "error": "Unavailable"}]}]
        summary = summarize(rows, 4)
        self.assertEqual(summary["completed"], 3)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["remaining"], 1)
        self.assertEqual(summary["repair_attempts"], 1)
        self.assertEqual(summary["p50_latency_seconds"], 20)
        self.assertEqual(summary["p95_latency_seconds"], 29)
        self.assertEqual(summary["stage_failures"], {"physical_signature": 1, "planner": 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            save_report(path, {"rows": rows[:1]})
            save_report(path, {"rows": rows, "summary": summary})
            self.assertEqual(json.loads(path.read_text())["rows"], rows)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_cli_requires_explicit_live_flag_and_bounded_counts(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["--output", "/tmp/unused-femm-batch.json"])
        self.assertEqual(caught.exception.code, 2)
        for flag, value in (("--rounds", "6"), ("--workers", "4"), ("--max-cases", "81")):
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--live", "--output", "/tmp/unused-femm-batch.json", flag, value])
            self.assertEqual(caught.exception.code, 2)

if __name__ == "__main__":
    unittest.main()
