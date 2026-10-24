"""Public reference fixtures preserve physical data, operating cases and failures."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.targets import NumericalTarget, freeze_targets
from app.targets import contract_fingerprint
from app.plan_generate import required_results
from evaluation.axisymmetric import rectangular_winding_axis_bz
from evaluation import replication


def synthetic_native(plan):
    """Independent minimal FEMM-format fixture for parser-negative tests."""
    spec = plan.model_dump(mode="json")
    points, rows = [], []
    for shape in [spec["domain"]] + [region["shape"] for region in spec["regions"]]:
        for a, b in replication._contour_edges(shape):
            for point in (a, b):
                if point not in points:
                    points.append(point)
            is_boundary = shape is spec["domain"]
            rows.append([points.index(a), points.index(b), -1, 1 if is_boundary else 0, 0, 0])
    materials = [dict(id="air", relative_permeability=1), *spec["materials"]]
    text = "[ProblemType] = axisymmetric\n[LengthUnits] = millimeters\n[Frequency] = 0\n"
    text += '<BeginBdry>\n<BdryName> = "outer_a_zero"\n<BdryType> = 0\n<A_0> = 0\n<EndBdry>\n'
    for item in materials:
        text += f'<BeginBlock>\n<BlockName> = "{item["id"]}"\n<Mu_x> = {item["relative_permeability"]}\n<Mu_y> = {item["relative_permeability"]}\n<BHPoints> = 0\n<EndBlock>\n'
    text += '<BeginCircuit>\n<CircuitName> = "coil"\n<TotalAmps_re> = 20\n<TotalAmps_im> = 0\n<EndCircuit>\n'
    text += f"[NumPoints] = {len(points)}\n" + "".join(f"{x} {y} 0 0\n" for x, y in points)
    text += f"[NumSegments] = {len(rows)}\n" + "".join(" ".join(str(x) for x in row) + "\n" for row in rows)
    text += "[NumArcSegments] = 0\n"
    labels = [{"label": spec["air_label"], "material": "air", "turns": 0, "circuit": None, "mesh_mm": spec["mesh_mm"]}, *spec["regions"]]
    text += f"[NumBlockLabels] = {len(labels)}\n"
    for region in labels:
        material_index = next(i for i, material in enumerate(materials, 1) if material["id"] == region["material"])
        text += f'{region["label"]["x"]} {region["label"]["y"]} {material_index} {region["mesh_mm"]} {1 if region.get("circuit") else 0} 0 0 {region["turns"]} 0\n'
    return text


class ReplicationTests(unittest.TestCase):
    def test_journal_cases_have_distinct_native_and_analytical_references(self):
        fixture = replication.load_fixture()
        targets = {case["id"]: NumericalTarget.model_validate(case["target"]) for case in fixture["cases"]}
        self.assertEqual(targets["iron_center"].value, 87)
        self.assertEqual(targets["iron_displaced"].value, 77.9)
        self.assertEqual(targets["air_core"].value, 49)
        self.assertEqual({target.reference_kind for target in targets.values()}, {"reported_numerical"})
        self.assertEqual(fixture["parameters"]["relative_permeability"]["value"], 10000)
        self.assertEqual(fixture["parameters"]["current_a"]["value"], 20)
        self.assertIn("5 A/mm^2", fixture["parameters"]["current_a"]["derivation"])
        for case in fixture["cases"]:
            self.assertNotEqual(case["target"]["value"], case["separate_analytical_reference"]["value"])

    def test_translation_preserves_geometry_and_displacement(self):
        for case_id, displacement in (("iron_center", 0), ("iron_displaced", 75), ("air_core", 0)):
            plan = replication.build_lubin_plan(case_id)
            self.assertEqual(plan.physics, "axisymmetric_dc_magnetics")
            self.assertIsNone(plan.depth_mm)
            self.assertEqual((plan.domain.y_min, plan.domain.y_max), (-1000, 1000))
            self.assertEqual(plan.circuits[0].current_a, 20)
            winding = plan.regions[0]
            self.assertEqual((winding.shape.x_min, winding.shape.x_max), (50, 70))
            self.assertEqual((winding.shape.y_min, winding.shape.y_max), (-100, 100))
            self.assertEqual(winding.turns, 1000)
            if case_id != "air_core":
                core = plan.regions[1]
                self.assertEqual((core.shape.y_min, core.shape.y_max), (displacement-50, displacement+50))
                self.assertEqual((core.shape.x_min, core.shape.x_max), (0, 30))
            else:
                self.assertEqual(len(plan.regions), 1)
                self.assertIsNone(plan.force_region)

    def test_numerical_refinement_does_not_change_frozen_targets_or_physics(self):
        fixture = replication.load_fixture()
        for case in fixture["cases"]:
            hashes = set()
            for profile in fixture["numerical_choices"]["profiles"]:
                plan = replication.build_lubin_plan(case["id"], profile=profile)
                contract = freeze_targets([case["target"]], plan, case_id=case["id"])
                hashes.add(contract.targets_sha256)
                self.assertEqual(plan.circuits[0].current_a, 20)
                self.assertEqual(plan.materials[-1].relative_permeability, 10000 if case["core_present"] else 1)
            self.assertEqual(len(hashes), 1)
        with self.assertRaises(ValueError):
            replication.build_lubin_plan("unknown")

    def test_near_field_protocol_changes_only_air_mesh_control(self):
        fixture = replication.load_fixture()
        for case in fixture["cases"]:
            initial = replication.build_lubin_plan(case["id"], profile="refined")
            refined = replication.build_lubin_plan(case["id"], profile="refined", protocol="near_field_v2")
            before = {region.id: region.model_dump(mode="json") for region in initial.regions}
            after = {region.id: region.model_dump(mode="json") for region in refined.regions}
            self.assertEqual(after["near_air"]["material"], "air")
            for name, original in before.items():
                updated = after[name]
                self.assertEqual({k: v for k, v in updated.items() if k != "mesh_mm"},
                                 {k: v for k, v in original.items() if k != "mesh_mm"})
            self.assertEqual(initial.materials, refined.materials)
            self.assertEqual(initial.circuits, refined.circuits)
            old = freeze_targets([case["target"]], initial, case_id=case["id"])
            new = freeze_targets([case["target"]], refined, case_id=case["id"])
            self.assertEqual(old.targets_sha256, new.targets_sha256)

    def test_native_geometry_assignment_and_boundary_audit_catches_actual_mutations(self):
        plan = replication.build_lubin_plan("air_core")
        original = synthetic_native(plan)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.fem"
            path.write_text(original)
            self.assertTrue(replication.audit_native_model(plan, path)["passed"])
            for name, changed in (
                ("axisymmetric", original.replace("= axisymmetric", "= planar")),
                ("circuit_current", original.replace("TotalAmps_re> = 20", "TotalAmps_re> = 21")),
                ("materials", original.replace("<Mu_x> = 1", "<Mu_x> = 2", 1)),
                ("all_outer_edges_a_zero", original.replace("0 1 -1 1 0 0", "0 1 -1 0 0 0", 1)),
                ("region_labels_materials_turns_mesh", original.replace("1 0 0 1000 0", "1 0 0 999 0")),
                ("all_requested_edges_covered", original.replace("50.0 -100.0 0 0", "51.0 -100.0 0 0")),
            ):
                path.write_text(changed)
                audit = replication.audit_native_model(plan, path)
                with self.subTest(name=name):
                    self.assertFalse(audit["passed"])
                    self.assertFalse(audit["checks"][name])

    def test_failed_native_attempt_is_preserved_and_previous_output_never_overwritten(self):
        class FailedWorker:
            def __init__(self, store, **kwargs):
                self.store = store
            def run(self, job):
                folder = self.store.root / job["id"]
                folder.mkdir()
                (folder / "solver.log").write_text("Deliberate test-native failure")
                self.store.update(job["id"], "failed", error="Deliberate test-native failure")
        with tempfile.TemporaryDirectory() as directory, patch.object(replication, "SolverWorker", FailedWorker):
            output = Path(directory) / "attempt"
            report = replication.run_replication(output, case_ids=["air_core"], profiles=["base"])
            self.assertEqual(report["status"], "complete")
            self.assertFalse(report["passed"])
            self.assertEqual(report["native_solve_count"], 0)
            self.assertEqual(report["runs"][0]["mesh"]["error"], "Deliberate test-native failure")
            before = (output / "preflight.json").read_bytes()
            self.assertIn(b'"relative": 0.02', before)
            with self.assertRaisesRegex(ValueError, "preserve previous runs"):
                replication.run_replication(output, case_ids=["air_core"], profiles=["base"])
            self.assertEqual((output / "preflight.json").read_bytes(), before)

    def test_meeker_shared_boundary_limit_is_explicit_not_a_simulated_success(self):
        path = replication.FIXTURE.with_name("meeker_inductance_note.json")
        fixture = json.loads(path.read_text())
        self.assertEqual(fixture["status"], "deferred_unsupported_shared_interfaces")
        self.assertIn("No mesh, solve or target pass", fixture["result"])

    def test_v3_corrects_only_auxiliary_quadrature_criterion(self):
        fixture = replication.load_fixture()
        before = fixture["numerical_choices"]["protocols"]["near_field_v2"]
        after = fixture["numerical_choices"]["protocols"]["near_field_v3"]
        self.assertEqual(before["profiles"], after["profiles"])
        self.assertEqual(before["near_air_rectangle_mm"], after["near_air_rectangle_mm"])
        self.assertEqual(after["energy_consistency_relative_tolerance"], 1e-4)
        self.assertEqual(len(after["energy_consistency_basis"]["file_sha256"]), 64)

    def test_artifact_reaudit_reads_csv_verifies_hashes_and_never_executes_native(self):
        fixture = replication.load_fixture()
        case = next(case for case in fixture["cases"] if case["id"] == "air_core")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            runs, prepared = [], []
            csv_paths = []
            for index, profile in enumerate(("base", "refined", "far")):
                plan = replication.build_lubin_plan("air_core", profile=profile, protocol="near_field_v2")
                contract = freeze_targets([case["target"]], plan, case_id="air_core")
                folder = source / "air_core" / profile
                folder.mkdir(parents=True)
                replication._write(folder / "target-contract.json", contract.model_dump(mode="json"))
                run = {"case_id": "air_core", "profile": profile, "contract_sha256": contract_fingerprint(contract),
                       "physical_invariants": {"passed": False}, "target_check": {"checks": [{"actual": 999}]}}
                for offset, stage in enumerate(("mesh", "solver")):
                    identity = f"{index*2+offset+1:032x}"
                    artifact = folder / "jobs" / identity
                    artifact.mkdir(parents=True)
                    replication._write(artifact / "spec.json", plan.model_dump(mode="json"))
                    (artifact / "model.fem").write_text("Synthetic native-format fixture for unit test")
                    (artifact / "solver.log").write_text("Synthetic completed unit test")
                    replication._write(artifact / "mesh-report.json", {"status": "passed"})
                    if stage == "solver":
                        (artifact / "model.ans").write_text("Synthetic nonempty solution fixture")
                        values = {"energy": 9.80049, "current_coil": 20, "flux_linkage_coil": .98}
                        for i, z in enumerate((0, 200)):
                            values[f"br_{i}"] = 0
                            values[f"bz_{i}"] = rectangular_winding_axis_bz(r_inner_mm=50, r_outer_mm=70,
                                z_min_mm=-100, z_max_mm=100, turns=1000, current_a=20, z_mm=z)
                        csv_path = artifact / "results.csv"
                        csv_path.write_text("quantity,value,unit\n" + "".join(
                            f"{name},{values[name]},{unit}\n" for name, unit in required_results(plan).items()))
                        csv_paths.append(csv_path)
                    run[stage] = {"id": identity, "status": "completed"}
                    run[stage + "_artifacts"] = replication._artifact_hashes(artifact)
                prepared.append({"case_id": "air_core", "profile": profile, "plan": plan.model_dump(mode="json"),
                    "contract": contract.model_dump(mode="json"), "contract_sha256": contract_fingerprint(contract)})
                runs.append(run)
            replication._write(source / "preflight.json", {"prepared_cases": prepared})
            source_report = source / "report.json"
            replication._write(source_report, {"protocol": "near_field_v2", "status": "complete", "passed": False,
                "runs": runs, "native_mesh_count": 3, "native_solve_count": 3})
            original_hash = hashlib.sha256(source_report.read_bytes()).hexdigest()
            with patch.object(replication, "audit_native_model", return_value={"passed": True}), \
                 patch.object(replication.SolverWorker, "run", side_effect=AssertionError("No native execution allowed")):
                result = replication.reaudit_existing_report(source_report, root / "reaudit")
            self.assertTrue(result["passed"], result)
            self.assertFalse(result["native_execution_performed"])
            self.assertEqual(result["native_solve_count"], 0)
            self.assertEqual(result["source_native_solve_count"], 3)
            self.assertEqual(result["runs"][0]["target_check"]["checks"][0]["actual"], 49)
            self.assertEqual(result["original_report_sha256"], original_hash)
            self.assertEqual(hashlib.sha256(source_report.read_bytes()).hexdigest(), original_hash)
            csv_paths[0].write_text(csv_paths[0].read_text().replace("9.80049", "9.9"))
            tampered = replication.reaudit_existing_report(source_report, root / "tampered")
            self.assertFalse(tampered["passed"])
            self.assertIn("artifact changed", tampered["error"])


if __name__ == "__main__":
    unittest.main()
