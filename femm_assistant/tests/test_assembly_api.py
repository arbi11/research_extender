"""Immutable assembly APIs, declared variations and exact native provenance."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from app.assembly_api import AssemblyStore
from app.main import Settings, create_app
from app.plan import GeometryPlan
from app.replication import ReplicationIntegrityError, ReplicationStore
from app.targets import model_fingerprint
from tests.test_assembly import coil_assembly, custom_assembly
from tests.test_papers import text_pdf


class AssemblyAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.app = create_app(Settings(data_dir=self.root, token="assembly-test", start_worker=False))
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://local")
        self.headers = {"Authorization": "Bearer assembly-test"}
        self.store = self.app.state.assemblies

    async def asyncTearDown(self):
        await self.client.aclose()
        self.temporary.cleanup()

    async def save(self, assembly=None, **extra):
        raw = (assembly or coil_assembly())
        raw = raw.model_dump(mode="json") if hasattr(raw, "model_dump") else raw
        response = await self.client.post("/api/assemblies", headers=self.headers, json={"assembly": raw, **extra})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def revision_count(self):
        with self.app.state.history.connect() as db:
            return db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0]

    def assembly_count(self):
        return len(list((self.app.state.replication.root / "assemblies").glob("*.json")))

    def paper_coil(self):
        store = self.app.state.replication
        uploaded = store.put_paper(text_pdf("Inner radius 10 mm. Current 1 A."))
        document = store.get_document(uploaded["paper_id"])
        evidence = store.put_evidence(uploaded["paper_id"], {
            "source_id": document.source_id, "title": "Synthetic winding reference", "summary": "API provenance fixture",
            "parameters": [{"id": "inner_radius", "description": "Winding inner radius", "value": 10., "unit": "mm",
                "evidence_kind": "explicit", "case_id": "baseline", "citations": [{"source_id": document.source_id,
                    "page": 1, "quote": "Inner radius 10 mm."}]}], "targets": [], "questions": [], "limitations": []})
        assembly = coil_assembly().model_dump(mode="json")
        assembly["input_mode"] = "combined"
        assembly["facts"][0].update(status="paper", source_parameter_id="inner_radius", source_refs=[document.source_id])
        return assembly, evidence

    async def test_authentication_save_get_preview_and_exact_artifact_binding(self):
        missing = "a" * 64
        for method, path, body in [("post", "/api/assemblies", {"assembly": coil_assembly().model_dump(mode="json")}),
            ("get", "/api/assemblies/" + missing, None), ("get", "/api/assemblies/" + missing + "/preview.svg", None),
            ("post", "/api/assemblies/" + missing + "/edits", {"edits": []})]:
            kwargs = {"json": body} if body is not None else {}
            self.assertEqual((await getattr(self.client, method)(path, **kwargs)).status_code, 401)
        record = await self.save()
        self.assertEqual(record, await self.save())
        self.assertEqual(self.assembly_count(), 1)
        path = "/api/assemblies/" + record["assembly_id"]
        self.assertEqual((await self.client.get(path, headers=self.headers)).json(), record)
        preview = await self.client.get(path + "/preview.svg", headers=self.headers)
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.text, record["preview_svg"])
        self.assertIn("image/svg+xml", preview.headers["content-type"])
        plan = GeometryPlan.model_validate(record["compilation"]["plan"])
        evidence = self.store.evidence(record["assembly_id"], plan)
        self.assertEqual(evidence["sha256"], record["assembly_id"])
        revision = await self.client.post("/api/revisions", headers=self.headers,
            json={"spec": plan.model_dump(mode="json"), "evidence": [evidence]})
        self.assertEqual(revision.status_code, 201, revision.text)
        self.assertEqual(revision.json()["evidence"][0]["reference"], "assembly-planning:" + record["assembly_id"])
        self.assertIsNone(self.app.state.store.next())

    async def test_reserved_references_reject_forgery_before_recording_revision(self):
        record = await self.save()
        plan = GeometryPlan.model_validate(record["compilation"]["plan"])
        correct = self.store.evidence(record["assembly_id"], plan)
        changed = plan.model_dump(mode="json"); changed["circuits"][0]["current_a"] = 2.
        for spec, evidence in [(plan.model_dump(mode="json"), {**correct, "sha256": "b" * 64}),
            (changed, correct), (plan.model_dump(mode="json"), {**correct, "reference": "assembly-planning:" + "b" * 64, "sha256": "b" * 64})]:
            before = self.revision_count()
            response = await self.client.post("/api/revisions", headers=self.headers,
                json={"spec": spec, "evidence": [evidence]})
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(self.revision_count(), before)

    async def test_scoped_stable_id_edit_creates_child_without_mutating_original(self):
        original = await self.save()
        response = await self.client.post("/api/assemblies/" + original["assembly_id"] + "/edits", headers=self.headers,
            json={"edits": [{"op": "replace", "path": "/facts/amps/value", "value": 2.}],
                  "allowed_paths": ["/facts/amps/value"], "instructions": "Explicit current variation"})
        self.assertEqual(response.status_code, 201, response.text)
        child = response.json()
        self.assertEqual(child["parent_id"], original["assembly_id"])
        self.assertNotEqual(child["assembly_id"], original["assembly_id"])
        self.assertEqual(child["compilation"]["plan"]["circuits"][0]["current_a"], 2.)
        self.assertIn("coil.body", child["affected_features"])
        self.assertEqual(child["changes"], ["/facts/amps/value"])
        self.assertEqual(self.store.get(original["assembly_id"]), original)
        before = self.assembly_count()
        for edits, paths in [([{"op": "replace", "path": "/facts/ri/value", "value": .5}], ["/facts/amps/value"]),
            ([{"op": "replace", "path": "/components", "value": []}], ["/components"]),
            ([{"op": "replace", "path": "/id", "value": "another"}], ["/id"])]:
            response = await self.client.post("/api/assemblies/" + original["assembly_id"] + "/edits", headers=self.headers,
                json={"edits": edits, "allowed_paths": paths})
            self.assertEqual(response.status_code, 422, response.text)
            self.assertEqual(self.assembly_count(), before)

    async def test_unsupported_physics_retains_hierarchy_and_preview_without_native_plan(self):
        raw = custom_assembly().model_dump(mode="json"); raw["required_physics"] = ["permanent_magnets"]
        record = await self.save(raw)
        self.assertFalse(record["compilation"]["execution_supported"])
        self.assertIsNone(record["compilation"]["plan"])
        self.assertIsNone(record["model_sha256"])
        self.assertEqual(len(record["feature_map"]), 4)
        response = await self.client.get("/api/assemblies/" + record["assembly_id"] + "/preview.svg", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("native execution unavailable", response.text)
        self.assertIsNone(self.app.state.store.next())

    async def test_paper_facts_require_verified_case_and_edits_must_declare_variations(self):
        raw, evidence = self.paper_coil()
        before = self.assembly_count()
        rejected = await self.client.post("/api/assemblies", headers=self.headers, json={"assembly": raw})
        self.assertEqual(rejected.status_code, 422)
        self.assertEqual(self.assembly_count(), before)
        record = await self.save(raw, evidence_id=evidence["evidence_id"])
        self.assertEqual(record["compilation"]["plan"]["regions"][0]["shape"]["x_min"], 10.)
        for extra in [{"case_id": "different_case"}, {"evidence_id": "b" * 64}]:
            payload = {"assembly": raw, "evidence_id": evidence["evidence_id"], **extra}
            self.assertEqual((await self.client.post("/api/assemblies", headers=self.headers, json=payload)).status_code, 422)
        before = self.assembly_count()
        invalid = await self.client.post("/api/assemblies/" + record["assembly_id"] + "/edits", headers=self.headers,
            json={"edits": [{"op": "replace", "path": "/facts/ri/value", "value": .8}]})
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(self.assembly_count(), before)
        changed = copy.deepcopy(raw)
        changed["facts"][0].update(value=.8, status="user", source_parameter_id=None,
            source_refs=["user:manual-assembly-edit"], explanation="User explicitly changed winding inner radius.")
        variation = await self.save(changed, evidence_id=evidence["evidence_id"], parent_id=record["assembly_id"],
                                    instructions="User variation of radius")
        self.assertEqual(variation["compilation"]["plan"]["regions"][0]["shape"]["x_min"], 8.)
        self.assertEqual(self.store.get(record["assembly_id"])["assembly"]["facts"][0]["status"], "paper")

    async def test_tampered_record_preview_and_model_hash_are_rejected_consistently(self):
        record = await self.save()
        path = self.app.state.replication.root / "assemblies" / (record["assembly_id"] + ".json")
        original = path.read_bytes(); path.write_bytes(original + b" ")
        for suffix in ["", "/preview.svg"]:
            response = await self.client.get("/api/assemblies/" + record["assembly_id"] + suffix, headers=self.headers)
            self.assertEqual(response.status_code, 409)
        path.write_bytes(original)
        forged = {k: v for k, v in record.items() if k != "assembly_id"}; forged["model_sha256"] = "b" * 64
        forged_id = self.app.state.replication._put("assemblies", forged)
        with self.assertRaises(ReplicationIntegrityError):
            self.store.get(forged_id)
        self.assertEqual((await self.client.get("/api/assemblies/" + "c" * 64, headers=self.headers)).status_code, 404)
        response = await self.client.post("/api/assemblies", headers=self.headers,
            json={"assembly": coil_assembly().model_dump(mode="json"), "parent_id": "c" * 64})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
