"""Source-input arithmetic catches a real current-density conversion mistake."""
from copy import deepcopy
import unittest

from pydantic import ValidationError

from app.paper_constraints import validate_excitation
from app.plan import GeometryPlan


def winding_plan(current=20.0, *, turns=1000):
    return GeometryPlan.model_validate({
        "title": "Uniform source winding", "physics": "axisymmetric_dc_magnetics",
        "depth_mm": None, "mesh_mm": 25.0,
        "domain": {"type": "rectangle", "x_min": 0.0, "x_max": 500.0, "y_min": -500.0, "y_max": 500.0},
        "air_label": {"x": 400.0, "y": 400.0},
        "materials": [{"id": "winding", "relative_permeability": 1.0}],
        "circuits": [{"id": "coil", "current_a": current}],
        "regions": [{"id": "coil", "shape": {"type": "rectangle", "x_min": 50.0, "x_max": 70.0,
            "y_min": -100.0, "y_max": 100.0}, "label": {"x": 60.0, "y": 80.0},
            "material": "winding", "circuit": "coil", "turns": turns}],
        "force_region": None, "probes": []})


def source_context(value=5.0, unit="A/mm2"):
    return {"preserve_paper_excitation": True, "source_id": "paper_fixture",
            "parameters": [{"id": "winding_density", "value": value, "unit": unit,
                "evidence_kind": "explicit", "citations": [{"source_id": "paper_fixture", "page": 15}]}]}


class PaperExcitationContractTests(unittest.TestCase):
    def test_mechanical_relation_rejects_10a_and_passes_20a_without_any_target(self):
        context = source_context()
        with self.assertRaises(ValidationError) as caught:
            validate_excitation(winding_plan(10.0), context)
        issue = caught.exception.errors(include_input=False, include_url=False)[0]
        self.assertEqual(issue["loc"], ("circuits", "coil", "current_a"))
        for evidence in ("/circuits/coil/current_a", "winding_density", "paper_fixture", "5 A/mm2",
                         "4000 mm2", "abs(turns)=1000", "requires 20 A", "10 A"):
            self.assertIn(evidence, issue["msg"])
        passed = validate_excitation(winding_plan(), context)
        self.assertEqual(passed["status"], "passed")
        self.assertTrue(passed["applicable"])
        self.assertEqual(passed["expected_current_a"], 20)
        self.assertEqual(passed["winding_area_mm2"], 4000)
        self.assertEqual(passed["absolute_turns"], 1000)
        self.assertIsInstance(passed["message"], str)

    def test_density_units_convert_to_amperes_per_square_millimeter(self):
        small = validate_excitation(winding_plan(), source_context())
        si = validate_excitation(winding_plan(), source_context(5_000_000.0, "A/m2"))
        self.assertEqual(si["current_density_a_per_mm2"], 5)
        self.assertEqual(si["expected_current_a"], small["expected_current_a"])

    def test_opt_in_is_explicit_and_does_not_override_instruction_variations(self):
        for context in (None, {}, {"preserve_paper_excitation": False},
                        {"preserve_paper_excitation": "true"}):
            report = validate_excitation(winding_plan(10.0), context)
            self.assertEqual(report["status"], "not_requested")
            self.assertFalse(report["applicable"])
            self.assertIsInstance(report["message"], str)

    def test_absent_ambiguous_unverified_and_nonfinite_density_are_not_guessed(self):
        absent = {"preserve_paper_excitation": True, "parameters": []}
        multiple = source_context()
        multiple["parameters"].append(deepcopy(multiple["parameters"][0]))
        assumed = source_context()
        assumed["parameters"][0]["evidence_kind"] = "assumption"
        uncited = source_context()
        uncited["parameters"][0]["citations"] = []
        for context in (absent, multiple, assumed, uncited, source_context(float("nan")),
                        source_context(float("inf")), source_context(True), source_context("5")):
            with self.subTest(context=context):
                report = validate_excitation(winding_plan(10.0), context)
                self.assertEqual(report["status"], "skipped")
                self.assertFalse(report["applicable"])

    def test_negative_density_or_current_does_not_invent_a_sign_convention(self):
        for plan, context in ((winding_plan(), source_context(-5.0)),
                              (winding_plan(-20.0), source_context())):
            report = validate_excitation(plan, context)
            self.assertEqual(report["status"], "skipped")
            self.assertIn("sign convention", report["message"])
        # Signed turns retain their orientation; the positive-current relation
        # uses abs(N) explicitly and does not change the sign in the model.
        plan = winding_plan(turns=-1000)
        self.assertEqual(validate_excitation(plan, source_context())["expected_current_a"], 20)
        self.assertEqual(plan.regions[0].turns, -1000)

    def test_zero_density_requires_zero_current_without_absolute_tolerance_slack(self):
        self.assertEqual(validate_excitation(winding_plan(0.0), source_context(0.0))["status"], "passed")
        with self.assertRaises(ValidationError):
            validate_excitation(winding_plan(1e-12), source_context(0.0))

    def test_relative_tolerance_is_strict_and_no_parameter_is_mutated(self):
        context, plan = source_context(), winding_plan(20.0 * (1 + 5e-10))
        original_context, original_plan = deepcopy(context), plan.model_dump(mode="json")
        self.assertEqual(validate_excitation(plan, context)["status"], "passed")
        with self.assertRaises(ValidationError):
            validate_excitation(winding_plan(20.0 * (1 + 2e-9)), context)
        self.assertEqual(context, original_context)
        self.assertEqual(plan.model_dump(mode="json"), original_plan)

    def test_multiwinding_models_are_not_summed_or_mapped_to_a_guessed_source(self):
        value = winding_plan(10.0).model_dump(mode="json")
        second = deepcopy(value["regions"][0])
        second.update(id="second", label={"x": 100.0, "y": 0.0})
        second["shape"].update(x_min=90.0, x_max=110.0)
        value["regions"].append(second)
        report = validate_excitation(GeometryPlan.model_validate(value), source_context())
        self.assertEqual(report["status"], "skipped")
        self.assertIn("exactly one wound region", report["message"])

    def test_nested_holes_are_skipped_but_an_enclosing_air_mesh_region_is_not_a_hole(self):
        value = winding_plan().model_dump(mode="json")
        value["regions"].append({"id": "near_air", "shape": {"type": "rectangle", "x_min": 0.0, "x_max": 150.0,
            "y_min": -150.0, "y_max": 150.0}, "label": {"x": 100.0, "y": 120.0}, "material": "air"})
        self.assertEqual(validate_excitation(GeometryPlan.model_validate(value), source_context())["status"], "passed")
        value["regions"].append({"id": "hole", "shape": {"type": "rectangle", "x_min": 55.0, "x_max": 65.0,
            "y_min": -20.0, "y_max": 20.0}, "label": {"x": 60.0, "y": 0.0}, "material": "air"})
        report = validate_excitation(GeometryPlan.model_validate(value), source_context())
        self.assertEqual(report["status"], "skipped")
        self.assertIn("Nested contours", report["message"])

    def test_targets_and_measurements_are_not_inputs_to_the_arithmetic_contract(self):
        context = source_context()
        context["published_targets"] = object()  # deliberately unusable by a target-fitting implementation
        self.assertEqual(validate_excitation(winding_plan(), context)["expected_current_a"], 20)


if __name__ == "__main__":
    unittest.main()
