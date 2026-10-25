"""Protocol tests plus opt-in actual native FEMM construction/meshing."""
import os
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from femm_brain.geometry import construct, recipe_catalog
from femm_brain.native_check import MESH_RECIPES, geometry_signature, mesh_summary, native_script, run_checks


class NativeProtocolTests(unittest.TestCase):
    def test_geometry_comparison_detects_changed_arc_despite_equal_counts(self):
        intended = construct("circle")
        altered = deepcopy(intended)
        altered["arcs"][0]["angle"] = 180
        self.assertNotEqual(geometry_signature(intended), geometry_signature(altered))
        reordered = deepcopy(intended)
        reordered["nodes"] = list(reversed(reordered["nodes"]))
        maximum = len(reordered["nodes"])-1
        for item in reordered["segments"] + reordered["arcs"]:
            item["n0"], item["n1"] = maximum-item["n0"], maximum-item["n1"]
        self.assertEqual(geometry_signature(intended), geometry_signature(reordered))

    def test_native_protocol_uses_audited_recipe_calls_and_never_solves_fields(self):
        for recipe in recipe_catalog():
            geometry, script = native_script(recipe["id"])
            with self.subTest(recipe=recipe["id"]):
                self.assertTrue(geometry["operations"])
                self.assertIn('mi_saveas("geometry.fem")', script)
                self.assertIn("geometry-completed.txt", script)
                self.assertNotIn("mi_analyze", script)
                self.assertNotIn("mi_loadsolution", script)
                self.assertNotIn("mo_", script)
                self.assertEqual("mi_createmesh()" in script, recipe["id"] in MESH_RECIPES)

    def test_actual_mesh_artifact_reader_rejects_bad_connectivity_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "meshed.node").write_text("3 2 0 0\n0 0 0\n1 1 0\n2 0 1\n")
            elements = folder / "meshed.ele"
            elements.write_text("1 3 1\n0 0 1 2 1\n")
            report = mesh_summary(folder)
            self.assertEqual(report["element_count"], 1)
            self.assertEqual(report["total_element_area_mm2"], .5)
            for row in ("0 0 0 2 1", "0 0 1 8 1", "0 0 1 2 0", "0 0 1 2 nan"):
                elements.write_text("1 3 1\n" + row + "\n")
                with self.subTest(row=row), self.assertRaises(ValueError):
                    mesh_summary(folder)

    def test_missing_native_launcher_does_not_generate_a_passing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_checks(directory, command="/definitely/missing/femm-lua")
            self.assertFalse((Path(directory) / "native_geometry.json").exists())


@unittest.skipUnless(os.getenv("FEMM_BRAIN_NATIVE") == "1", "Set FEMM_BRAIN_NATIVE=1 with native FEMM 5 installed")
class NativeFemmTests(unittest.TestCase):
    def test_all_reconstructed_recipes_rebuild_in_native_femm(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_checks(directory)
            self.assertEqual(report["status"], "passed", report)
            self.assertEqual(len(report["recipes"]), len(recipe_catalog()))
            self.assertEqual(sum("mesh" in r for r in report["recipes"]), len(MESH_RECIPES))
            self.assertFalse(report["numerical_convergence_verified"])
            for result in report["recipes"]:
                self.assertEqual(result["expected_counts"], result["observed_counts"])
                self.assertFalse(result["field_solve_performed"])


if __name__ == "__main__":
    unittest.main()
