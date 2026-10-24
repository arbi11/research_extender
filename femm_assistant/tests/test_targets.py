"""Numerical acceptance uses frozen source targets and actual result envelopes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from app.plan import wire_example
from app.targets import (NumericalTarget, TargetContract, Tolerance, contract_fingerprint,
                         convert_units, evaluate_target_file, evaluate_targets,
                         freeze_targets, rebind_targets)


def target(quantity="energy", value=.000865042, unit="J", **updates):
    observable = {"quantity": quantity}
    if quantity in {"inductance", "current", "flux_linkage"}:
        observable["circuit_id"] = "excitation"
    if quantity in {"bx", "by", "br", "bz", "b_magnitude"}:
        observable["probe_index"] = 0
    result = {"id": "reference", "description": "Synthetic reference with a preselected 2% tolerance.",
              "observable": observable, "value": value, "unit": unit,
              "tolerance": {"relative": .02}, "reference_kind": "reported_numerical",
              "conditions": "1 A, unchanged geometry and measurement definition.",
              "citations": [{"source_id": "synthetic", "section": "Reference table"}]}
    result.update(updates)
    return NumericalTarget.model_validate(result)


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.plan = wire_example().model_dump(mode="json")

    def contract(self, reference=None, plan=None):
        return freeze_targets([reference or target()], plan or self.plan, case_id="baseline")

    def test_units_are_dimensioned_not_suffix_guesses(self):
        self.assertAlmostEqual(convert_units(1.73, "mH", "H"), .00173)
        self.assertAlmostEqual(convert_units(865.042, "µJ", "J"), .000865042)
        self.assertAlmostEqual(convert_units(1, "mT", "uT"), 1000)
        for args in ((1, "mH", "J"), (1, "Wb", "Wb-turn"), (1, "tesla", "T"),
                     (True, "A", "A"), (float("nan"), "J", "J")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                convert_units(*args)

    def test_energy_pass_failure_and_wrong_unit(self):
        contract = self.contract()
        report = evaluate_targets(contract, {"energy": {"value": .865042, "unit": "mJ"}}, plan=self.plan)
        self.assertEqual(report["status"], "passed")
        failed = evaluate_targets(contract, {"energy": {"value": .0004, "unit": "J"}}, plan=self.plan)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["checks"][0]["target"], .000865042)
        wrong = evaluate_targets(contract, {"energy": {"value": .000865042, "unit": "H"}}, plan=self.plan)
        self.assertEqual(wrong["status"], "incomplete")
        self.assertEqual(wrong["checks"][0]["status"], "unavailable")

    def test_inductance_uses_native_linkage_over_native_current(self):
        self.plan["circuits"][0]["id"] = "excitation"
        contract = self.contract(target("inductance", 1.73, "mH"))
        native = {"current_excitation": {"value": 2, "unit": "A"},
                  "flux_linkage_excitation": {"value": .00346, "unit": "Wb-turn"},
                  "inductance": {"value": 999, "unit": "H"}}
        report = evaluate_targets(contract, native, plan=self.plan)
        self.assertEqual(report["status"], "passed")
        self.assertAlmostEqual(report["checks"][0]["actual"], 1.73)
        native["current_excitation"]["value"] = 0
        self.assertEqual(evaluate_targets(contract, native, plan=self.plan)["status"], "incomplete")

    def test_exact_revision_and_target_tolerances_are_frozen(self):
        contract = self.contract()
        with self.assertRaises(ValidationError):
            contract.targets[0].value = .0004
        packet = contract.model_dump(mode="json")
        packet["targets"][0]["tolerance"]["relative"] = .9
        with self.assertRaises(ValidationError):
            TargetContract.model_validate(packet)
        changed = copy.deepcopy(self.plan)
        changed["circuits"][0]["current_a"] = 2
        with self.assertRaisesRegex(ValueError, "different model revision"):
            evaluate_targets(contract, {}, plan=changed)
        rebound = rebind_targets(contract, changed)
        self.assertEqual(rebound.targets_sha256, contract.targets_sha256)
        self.assertEqual(rebound.targets, contract.targets)
        self.assertNotEqual(rebound.model_sha256, contract.model_sha256)
        with self.assertRaisesRegex(ValueError, "frozen before solving"):
            evaluate_targets(rebound, {}, plan=changed,
                             expected_contract_sha256=contract_fingerprint(contract))
        bypass = contract.model_copy(update={"targets": (target(value=.0004),)})
        with self.assertRaises(ValidationError):
            evaluate_targets(bypass, {}, plan=self.plan)

    def test_cases_cannot_mix_operating_points(self):
        centre = target(case_id="h_zero")
        displaced = target(case_id="h_75mm", id="displaced")
        with self.assertRaisesRegex(ValidationError, "different operating cases"):
            freeze_targets([centre, displaced], self.plan, case_id="h_zero")

    def test_tolerances_are_explicit_and_zero_targets_need_absolute(self):
        for packet in ({"absolute": 0, "relative": 0}, {"relative": 2}, {"relative": True}):
            with self.assertRaises(ValidationError):
                Tolerance.model_validate(packet)
        with self.assertRaisesRegex(ValidationError, "zero-valued"):
            target(value=0)
        zero = target(value=0, tolerance={"absolute": 1e-8})
        self.assertEqual(evaluate_targets(self.contract(zero), {"energy": {"value": 5e-9, "unit": "J"}},
                                          plan=self.plan)["status"], "passed")

    def test_assumed_reference_and_unit_dimension_mismatch_rejected(self):
        for updates in ({"evidence_kind": "assumption"}, {"unit": "mH"},
                        {"evidence_kind": "derived", "derivation": None},
                        {"citations": []}, {"value": True}, {"value": float("inf")}):
            with self.subTest(updates=updates), self.assertRaises(ValidationError):
                target(**updates)

    def test_results_must_have_finite_native_envelopes(self):
        for bad in ({"value": True, "unit": "J"}, {"value": float("nan"), "unit": "J"},
                    {"value": "0.000865042", "unit": "J"}, {"value": .000865042, "unit": "J", "explanation": "model says so"},
                    .000865042):
            report = evaluate_targets(self.contract(), {"energy": bad}, plan=self.plan)
            self.assertEqual(report["status"], "incomplete")

    def test_native_csv_reader_requires_solution_and_all_worker_quantities(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "results.csv").write_text("quantity,value,unit\nenergy,0.000865042,J\n")
            with self.assertRaisesRegex(RuntimeError, "solution file"):
                evaluate_target_file(self.contract(), folder, plan=wire_example())
            (folder / "model.ans").write_text("native solution placeholder")
            with self.assertRaisesRegex(RuntimeError, "all requested"):
                evaluate_target_file(self.contract(), folder, plan=wire_example())

    def test_axisymmetric_components_and_coordinate_mismatch(self):
        axis = {"kind": "magnetostatic_plan", "physics": "axisymmetric_dc_magnetics",
                "probes": [{"x": 0, "y": 0}], "force_region": "armature", "circuits": []}
        for quantity, native in (("bz", {"bz_0": {"value": .2, "unit": "T"}}),
                                 ("br", {"br_0": {"value": .2, "unit": "T"}}),
                                 ("b_magnitude", {"br_0": {"value": .12, "unit": "T"}, "bz_0": {"value": .16, "unit": "T"}})):
            with self.subTest(quantity=quantity):
                contract = freeze_targets([target(quantity, .2, "T")], axis, case_id="baseline")
                self.assertEqual(evaluate_targets(contract, native, plan=axis)["status"], "passed")
        force = freeze_targets([target("force_z", -77.9, "N")], axis, case_id="baseline")
        self.assertEqual(evaluate_targets(force, {"force_z": {"value": -77.9, "unit": "N"}}, plan=axis)["status"], "passed")
        for wrong in (target("bx", .2, "T"), target("force_y", 77.9, "N")):
            with self.assertRaisesRegex(ValueError, "coordinate system"):
                freeze_targets([wrong], axis, case_id="baseline")
        with self.assertRaisesRegex(ValueError, "coordinate system"):
            freeze_targets([target("bz", .2, "T")], self.plan, case_id="baseline")

    def test_missing_selector_is_rejected_before_solve(self):
        missing_probe = target("bx", .2, "T", observable={"quantity": "bx", "probe_index": 15})
        with self.assertRaisesRegex(ValueError, "probe absent"):
            self.contract(missing_probe)
        missing_circuit = target("inductance", 1, "mH")
        with self.assertRaisesRegex(ValueError, "existing circuit"):
            self.contract(missing_circuit)


if __name__ == "__main__":
    unittest.main()
