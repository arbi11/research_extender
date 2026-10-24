"""Feature edits preserve unrelated inputs and failed evidence survives recovery."""
from copy import deepcopy
import sqlite3
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from app.history import (EvidenceReference, FeatureEdit, HistoryConflict, ModelRevisionStore,
                         affected_stages, apply_feature_edits, plan_diff)
from app.plan import GeometryPlan, wire_example


class FeatureEditTests(unittest.TestCase):
    def setUp(self):
        self.plan = wire_example()

    def test_radius_edit_preserves_excitation_labels_materials_and_probes(self):
        before = self.plan.model_dump(mode="json")
        candidate, changes = apply_feature_edits(self.plan, [
            {"op": "replace", "path": "/regions/wire/shape/radius", "value": 1.25}])
        expected = deepcopy(before)
        expected["regions"][0]["shape"]["radius"] = 1.25
        self.assertEqual(candidate.model_dump(mode="json"), expected)
        self.assertEqual(self.plan.model_dump(mode="json"), before)
        self.assertEqual(changes, [{"path": "/regions/wire/shape/radius", "before": 1.0, "after": 1.25}])

    def test_explicit_scope_blocks_an_unrequested_current_change(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            apply_feature_edits(self.plan, [
                {"op": "replace", "path": "/regions/wire/shape/radius", "value": 1.25},
                {"op": "replace", "path": "/circuits/wire/current_a", "value": 4.0}],
                allowed_paths=["/regions/wire/shape"])

    def test_feature_ids_are_stable_and_collections_cannot_be_replaced(self):
        for edit in [
            {"op": "replace", "path": "/regions/wire/id", "value": "renamed"},
            {"op": "replace", "path": "/regions", "value": []},
            {"op": "replace", "path": "/regions/wire", "value": {**self.plan.regions[0].model_dump(), "id": "renamed"}},
        ]:
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                apply_feature_edits(self.plan, [edit])

    def test_related_atomic_edits_validate_together_without_exposing_intermediate_plan(self):
        candidate, changes = apply_feature_edits(self.plan, [
            {"op": "replace", "path": "/regions/wire/shape/center/x", "value": -2.0},
            {"op": "replace", "path": "/regions/wire/label/x", "value": -2.0}])
        self.assertEqual(candidate.regions[0].label.x, -2)
        self.assertEqual(len(changes), 2)
        with self.assertRaisesRegex(ValidationError, "Label"):
            apply_feature_edits(self.plan, [
                {"op": "replace", "path": "/regions/wire/shape/center/x", "value": -2.0}])

    def test_overlapping_paths_rejected_regardless_of_edit_order(self):
        edits = [
            {"op": "replace", "path": "/regions/wire/shape", "value": self.plan.regions[0].shape.model_dump()},
            {"op": "replace", "path": "/regions/wire/shape/radius", "value": 1.25}]
        for order in [edits, list(reversed(edits))]:
            with self.assertRaisesRegex(ValueError, "overlap"):
                apply_feature_edits(self.plan, order)

    def test_new_named_feature_requires_matching_id_and_valid_final_references(self):
        region = {"id": "second", "shape": {"type": "circle", "center": {"x": -3.0, "y": 0.0}, "radius": .5},
                  "material": "conductor", "label": {"x": -3.0, "y": 0.0}}
        candidate, changes = apply_feature_edits(self.plan, [{"op": "add", "path": "/regions/second", "value": region}])
        self.assertEqual([item.id for item in candidate.regions], ["wire", "second"])
        self.assertEqual(changes[0]["op"], "add")
        for bad in [{**region, "id": "different"}, {**region, "material": "undefined"}]:
            with self.assertRaises(ValueError):
                apply_feature_edits(self.plan, [{"op": "add", "path": "/regions/second", "value": bad}])
        with self.assertRaisesRegex(ValidationError, "undefined material"):
            apply_feature_edits(self.plan, [{"op": "remove", "path": "/materials/conductor"}])

    def test_shape_list_and_unknown_paths_are_bounded(self):
        for path in ["/regions/0/shape/radius", "/probes/01/x", "/probes/999/x", "/probes/-1/x",
                     "/probes/0/z", "/../secrets", "/regions/wire/lua"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                apply_feature_edits(self.plan, [{"op": "replace", "path": path, "value": 2.0}])

    def test_no_op_missing_values_nonfinite_and_oversized_patches_fail(self):
        for edit in [
            {"op": "replace", "path": "/mesh_mm", "value": self.plan.mesh_mm},
            {"op": "replace", "path": "/mesh_mm"},
            {"op": "replace", "path": "/mesh_mm", "value": float("nan")},
            {"op": "remove", "path": "/mesh_mm"},
        ]:
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                apply_feature_edits(self.plan, [edit])
        with self.assertRaises(ValueError):
            apply_feature_edits(self.plan, [{"op": "replace", "path": "/mesh_mm", "value": 1.0}] * 65)

    def test_feature_reordering_is_not_a_change_and_probe_edits_invalidate_solution(self):
        value = self.plan.model_dump()
        value["materials"].append({"id": "spare", "relative_permeability": 2.0})
        first = GeometryPlan.model_validate(value)
        value["materials"].reverse()
        self.assertEqual(plan_diff(first, GeometryPlan.model_validate(value)), [])
        self.assertEqual(affected_stages([{"path": "/probes/0/x"}]), ["solver", "target"])
        self.assertEqual(affected_stages([{"path": "/regions/wire/shape/radius"}]), ["mesh", "solver", "target"])
        self.assertEqual(affected_stages([{"path": "/circuits/wire/current_a"}]), ["mesh", "solver", "target"])
        self.assertEqual(affected_stages([{"path": "/title"}]), [])


class RevisionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = ModelRevisionStore(Path(self.temp.name))
        self.initial = self.store.create(wire_example(), evidence=[
            EvidenceReference(source="user", reference="Original conductor request")])

    def tearDown(self):
        self.temp.cleanup()

    def edit(self, parent=None, **kwargs):
        return self.store.apply_patch((parent or self.initial)["id"], [
            FeatureEdit(op="replace", path="/regions/wire/shape/radius", value=1.25)],
            allowed_paths=["/regions/wire/shape/radius"], **kwargs)

    def test_checkpoint_hashes_and_before_after_evidence_survive_reload(self):
        revision = self.edit(evidence=[{"source": "paper", "reference": "Figure 2, radius", "detail": "1.25 mm"}])
        self.assertEqual(revision["parent_id"], self.initial["id"])
        self.assertNotEqual(revision["plan_hash"], self.initial["plan_hash"])
        self.assertEqual(revision["changes"][0]["before"], 1)
        self.assertEqual(revision["changes"][0]["after"], 1.25)
        reloaded = ModelRevisionStore(Path(self.temp.name))
        self.assertEqual(reloaded.head(), revision)
        self.assertEqual(reloaded.get(self.initial["id"]), self.initial)
        self.assertEqual([item["id"] for item in reloaded.lineage(revision["id"])], [revision["id"], self.initial["id"]])

    def test_stale_parent_and_accidental_reinitialization_cannot_overwrite(self):
        revision = self.edit()
        with self.assertRaises(HistoryConflict):
            self.edit()
        with self.assertRaises(HistoryConflict):
            self.store.create(wire_example())
        self.assertEqual(self.store.head(), revision)

    def test_invalid_patch_rolls_back_without_advancing_head(self):
        with self.assertRaises(ValidationError):
            self.store.apply_patch(self.initial["id"], [
                {"op": "replace", "path": "/regions/wire/shape/radius", "value": 11.0}],
                allowed_paths=["/regions/wire/shape/radius"])
        self.assertEqual(self.store.head(), self.initial)

    def test_sql_updates_and_deletes_cannot_rewrite_immutable_records(self):
        check = self.store.record_check(self.initial["id"], stage="mesh", status="passed", job_id="native-mesh-1")
        with self.store.connect() as db:
            for statement in ["UPDATE revisions SET record_json='{}'", "DELETE FROM revisions",
                              "UPDATE checks SET record_json='{}'", "DELETE FROM checks"]:
                with self.subTest(statement=statement), self.assertRaises(sqlite3.IntegrityError):
                    db.execute(statement)
        self.assertEqual(self.store.get_check(check["id"]), check)

    def test_recovery_requires_recorded_failure_and_ancestor_checkpoint(self):
        passed = self.store.record_check(self.initial["id"], stage="mesh", status="passed", job_id="native-mesh-1")
        for failure_id in ["missing", passed["id"]]:
            with self.assertRaisesRegex(ValueError, "failed check"):
                self.store.branch_recovery(failure_id, self.initial["id"], branch="repair", diagnosis="Inspect mesh")
        separate = self.store.create(wire_example(), branch="separate")
        failed = self.store.record_check(self.initial["id"], stage="mesh", status="failed", job_id="native-mesh-2")
        with self.assertRaisesRegex(ValueError, "ancestor"):
            self.store.branch_recovery(failed["id"], separate["id"], branch="repair", diagnosis="Inspect mesh")

    def test_recovery_forks_checkpoint_and_retains_original_failure_and_main(self):
        modified = self.edit()
        failed = self.store.record_check(modified["id"], stage="mesh", status="failed", job_id="native-mesh-2",
            artifact_hash="f" * 64, details={"error": "Injected native failure fixture"},
            implicated_paths=["/regions/wire/shape"])
        recovery = self.store.branch_recovery(failed["id"], self.initial["id"], branch="repair",
            diagnosis="Restore the prior feature before evaluating a smaller radius edit.")
        self.assertEqual(self.store.head(), modified)
        self.assertEqual(self.store.head("repair"), recovery)
        self.assertEqual(recovery["plan_hash"], self.initial["plan_hash"])
        self.assertEqual(recovery["parent_id"], self.initial["id"])
        self.assertEqual(recovery["recovery"]["failed_revision_id"], modified["id"])
        self.assertEqual(self.store.get_check(failed["id"]), failed)
        self.assertEqual(self.store.checks(recovery["id"]), [])
        repaired = self.store.apply_patch(recovery["id"], [{"op": "replace", "path": "/regions/wire/shape/radius", "value": 1.1}],
                                         allowed_paths=["/regions/wire/shape/radius"], branch="repair")
        self.assertEqual(repaired["plan"]["regions"][0]["shape"]["radius"], 1.1)
        self.assertEqual(self.store.head(), modified)

    def test_localization_follows_dependencies_without_claiming_last_writer_is_root_cause(self):
        changed_current = self.store.apply_patch(self.initial["id"], [
            {"op": "replace", "path": "/circuits/wire/current_a", "value": 2.0}],
            allowed_paths=["/circuits/wire/current_a"])
        radius = self.edit(changed_current)
        failure = self.store.record_check(radius["id"], stage="solver", status="failed", job_id="native-solve-3",
            implicated_paths=["/regions/wire/label"])
        context = self.store.recovery_context(failure["id"])
        self.assertIn("/regions/wire/shape", context["related_paths"])
        self.assertIn("/circuits/wire", context["related_paths"])
        self.assertEqual(context["candidates"][0]["revision_id"], radius["id"])
        self.assertIn(changed_current["id"], [item["revision_id"] for item in context["candidates"]])
        self.assertIn("not a proven root cause", context["causation"])
        self.assertEqual(context["required_rerun"], ["solver", "target"])
        restored = self.store.branch_recovery(failure["id"], self.initial["id"], branch="restore",
                                             diagnosis="Investigate the geometry and excitation edits.")
        self.assertEqual(restored["recovery"]["required_rerun"], ["mesh", "solver", "target"])

    def test_check_validation_unknown_revisions_and_no_nan_evidence(self):
        for arguments in [
            {"stage": "solve", "status": "failed", "job_id": "test"},
            {"stage": "mesh", "status": "pending", "job_id": "test"},
            {"stage": "mesh", "status": "failed", "job_id": ""},
            {"stage": "mesh", "status": "failed", "job_id": "test", "artifact_hash": "invalid"},
            {"stage": "mesh", "status": "failed", "job_id": "test", "details": {"value": float("nan")}},
        ]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.store.record_check(self.initial["id"], **arguments)
        with self.assertRaises(ValueError):
            self.store.record_check("missing", stage="mesh", status="failed", job_id="native-test")
        self.assertIsNone(self.store.get("missing"))


if __name__ == "__main__":
    unittest.main()
