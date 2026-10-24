"""HTTP revision/checkpoint integration with real worker-observed fixture outcomes.

The tiny executable fixtures stand in for FEMM only to exercise process failure,
mesh-artifact validation, journal hashes and API permissions without paid calls.
Native FEMM accuracy is covered separately by the live numerical references.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import httpx

from app.main import Settings, create_app
from app.plan import GeometryPlan, wire_example


def grid_plan():
    return GeometryPlan.model_validate({
        "title": "Two faces for worker journal integration", "depth_mm": 1., "mesh_mm": .5,
        "domain": {"type": "rectangle", "x_min": 0., "x_max": 3., "y_min": 0., "y_max": 3.},
        "air_label": {"x": .3, "y": .5}, "materials": [], "circuits": [],
        "regions": [{"id": "inner", "material": "air", "label": {"x": 1.3, "y": 1.5},
                     "shape": {"type": "rectangle", "x_min": 1., "x_max": 2., "y_min": 1., "y_max": 2.}}]})


def fixture_mesh_launcher(path: Path):
    """Write real indexed mesh artifacts, subsequently checked by validate_mesh."""
    nodes = "16 2 0 0\n" + "".join(f"{y*4+x} {x} {y}\n" for y in range(4) for x in range(4))
    triangles = []
    for y in range(3):
        for x in range(3):
            index = y*4+x
            attribute = 2 if (x, y) == (1, 1) else 1
            triangles.extend([(index, index+1, index+5, attribute), (index, index+5, index+4, attribute)])
    elements = "18 3 1\n" + "".join(f"{index} " + " ".join(map(str, row)) + "\n" for index, row in enumerate(triangles))
    path.write_text("#!/usr/bin/env python3\nfrom pathlib import Path\n"
                    + f"Path('model.node').write_text({nodes!r})\n"
                    + f"Path('model.ele').write_text({elements!r})\n"
                    + "Path('model.fem').write_text('Explicit mesh fixture, not a native FEMM solution')\n"
                    + "print('Indexed two-face mesh fixture emitted')\n")
    path.chmod(0o700)
    return str(path)


class RevisionApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.launcher = fixture_mesh_launcher(self.root / "fixture-mesher")
        self.settings = Settings(data_dir=self.root / "jobs", token="revision-test-token",
                                 femm_lua=self.launcher, start_worker=False)
        self.app = create_app(self.settings)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")
        self.headers = {"Authorization": "Bearer revision-test-token"}

    async def asyncTearDown(self):
        await self.client.aclose()
        self.temp.cleanup()

    async def revision(self, spec=None, **extra):
        response = await self.client.post("/api/revisions", headers=self.headers,
                                          json={"spec": (spec or grid_plan()).model_dump(mode="json"), **extra})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    async def get_revision(self, revision_id):
        response = await self.client.get("/api/revisions/"+revision_id, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    async def mesh(self, spec, revision):
        response = await self.client.post("/api/plan/mesh", headers={**self.headers, "X-Model-Revision": revision["id"]},
                                          json=spec.model_dump(mode="json"))
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["revision_id"], revision["id"])
        self.app.state.worker.run(response.json())
        return self.app.state.store.get(response.json()["id"])

    async def test_authentication_and_no_public_success_check_injection(self):
        missing = "a"*64
        for method, path, payload in [
            ("post", "/api/revisions", {}),
            ("get", "/api/revisions/"+missing, None),
            ("post", "/api/revisions/"+missing+"/patch", {}),
            ("get", "/api/recovery/"+missing, None),
            ("post", "/api/recovery/"+missing+"/restore", {}),
        ]:
            kwargs = {"json": payload} if payload is not None else {}
            response = await getattr(self.client, method)(path, **kwargs)
            self.assertEqual(response.status_code, 401, (path, response.text))
        revision = await self.revision()
        for path in ("/api/revisions/"+revision["id"]+"/checks", "/api/checks"):
            response = await self.client.post(path, headers=self.headers,
                json={"stage": "solver", "status": "passed", "revision_id": revision["id"]})
            self.assertIn(response.status_code, (404, 405))
        response = await self.client.post("/api/revisions", headers=self.headers,
            json={"spec": grid_plan().model_dump(mode="json"), "checks": [{"status": "passed"}]})
        self.assertEqual(response.status_code, 422)
        self.assertEqual((await self.get_revision(revision["id"]))["checks"], [])
        self.assertIsNone(self.app.state.store.next())

    async def test_independent_cases_do_not_share_an_implicit_branch(self):
        first = await self.revision()
        second = await self.revision()
        self.assertRegex(first["id"], r"^[a-f0-9]{64}$")
        self.assertNotEqual(first["id"], second["id"])
        self.assertNotEqual(first["branch"], second["branch"])
        self.assertIsNone(self.app.state.history.head("main"))
        for revision in (first, second):
            self.assertIsNone(revision["parent_id"])
            self.assertEqual(self.app.state.history.head(revision["branch"])["id"], revision["id"])
            self.assertEqual(len((await self.get_revision(revision["id"]))["lineage"]), 1)
        explicit = await self.revision(branch="explicit-case")
        conflict = await self.client.post("/api/revisions", headers=self.headers,
            json={"spec": grid_plan().model_dump(mode="json"), "branch": explicit["branch"]})
        self.assertEqual(conflict.status_code, 409)

    async def test_scoped_patch_preserves_unrelated_fields_and_rejects_stale_parent(self):
        plan = wire_example()
        parent = await self.revision(plan)
        evidence = [{"source": "user", "reference": "Increase only the circuit current"}]
        request = {"edits": [{"op": "replace", "path": "/circuits/wire/current_a", "value": 2.}],
                   "allowed_paths": ["/circuits/wire/current_a"], "branch": parent["branch"],
                   "reason": "User-requested current change", "evidence": evidence}
        response = await self.client.post("/api/revisions/"+parent["id"]+"/patch", headers=self.headers, json=request)
        self.assertEqual(response.status_code, 201, response.text)
        child = response.json()
        expected = plan.model_dump(mode="json")
        expected["circuits"][0]["current_a"] = 2.
        self.assertEqual(child["plan"], expected)
        self.assertEqual(child["parent_id"], parent["id"])
        self.assertEqual(child["invalidated_stages"], ["mesh", "solver", "target"])
        self.assertEqual(child["evidence"][0]["reference"], evidence[0]["reference"])
        self.assertEqual((await self.get_revision(parent["id"]))["plan"], plan.model_dump(mode="json"))
        self.assertEqual([item["id"] for item in (await self.get_revision(child["id"]))["lineage"]],
                         [child["id"], parent["id"]])
        stale = await self.client.post("/api/revisions/"+parent["id"]+"/patch", headers=self.headers, json=request)
        self.assertEqual(stale.status_code, 409)
        unrelated = dict(request, edits=[{"op": "replace", "path": "/regions/wire/shape/radius", "value": 1.2}])
        denied = await self.client.post("/api/revisions/"+child["id"]+"/patch", headers=self.headers, json=unrelated)
        self.assertEqual(denied.status_code, 422)
        self.assertIn("outside", denied.text)
        self.assertEqual(self.app.state.history.head(parent["branch"])["id"], child["id"])

    async def test_manual_revision_records_concrete_edits_and_rejects_a_stale_save(self):
        parent = await self.revision()
        changed = grid_plan().model_dump(mode="json")
        changed["depth_mm"] = 2.
        request = {"spec": changed, "parent_id": parent["id"], "branch": parent["branch"]}
        response = await self.client.post("/api/revisions", headers=self.headers, json=request)
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["changes"], [{"path": "/depth_mm", "before": 1., "after": 2.}])
        stale = await self.client.post("/api/revisions", headers=self.headers, json=request)
        self.assertEqual(stale.status_code, 409)

    async def test_mesh_rejects_changed_spec_before_queuing(self):
        revision = await self.revision()
        data = grid_plan().model_dump(mode="json")
        data["depth_mm"] = 2.
        response = await self.client.post("/api/plan/mesh", headers={**self.headers, "X-Model-Revision": revision["id"]}, json=data)
        self.assertEqual(response.status_code, 409)
        self.assertIsNone(self.app.state.store.next())
        self.assertEqual((await self.get_revision(revision["id"]))["checks"], [])

    async def test_completed_mesh_is_bound_to_both_exact_spec_and_revision(self):
        plan = grid_plan()
        revision = await self.revision(plan)
        mesh = await self.mesh(plan, revision)
        self.assertEqual(mesh["status"], "completed", mesh["error"])
        observed = await self.get_revision(revision["id"])
        self.assertEqual([(check["stage"], check["status"]) for check in observed["checks"]], [("mesh", "passed")])
        other = await self.revision(plan)
        request = {"spec": plan.model_dump(mode="json"), "mesh_job_id": mesh["id"], "revision_id": other["id"]}
        cross_case = await self.client.post("/api/plan/jobs", headers=self.headers, json=request)
        self.assertEqual(cross_case.status_code, 409)
        self.assertIn("different model revision", cross_case.text)
        changed = plan.model_dump(mode="json")
        changed["depth_mm"] = 2.
        stale_spec = await self.client.post("/api/plan/jobs", headers=self.headers,
            json={"spec": changed, "mesh_job_id": mesh["id"], "revision_id": revision["id"]})
        self.assertEqual(stale_spec.status_code, 409)
        self.assertIn("model changed", stale_spec.text)
        request["revision_id"] = revision["id"]
        queued = await self.client.post("/api/plan/jobs", headers=self.headers, json=request)
        self.assertEqual(queued.status_code, 202, queued.text)
        self.assertEqual(queued.json()["revision_id"], revision["id"])

    async def test_mesh_without_header_creates_an_isolated_revision(self):
        response = await self.client.post("/api/plan/mesh", headers=self.headers, json=grid_plan().model_dump(mode="json"))
        self.assertEqual(response.status_code, 202, response.text)
        mesh = response.json()
        revision = await self.get_revision(mesh["revision_id"])
        self.assertEqual(revision["plan"], mesh["spec"])
        self.assertEqual(revision["checks"], [])
        self.assertIsNone(revision["parent_id"])

    async def test_worker_failure_keeps_evidence_and_restore_forks_without_success(self):
        parent = await self.revision()
        child_response = await self.client.post("/api/revisions/"+parent["id"]+"/patch", headers=self.headers,
            json={"edits": [{"op": "replace", "path": "/depth_mm", "value": 2.}],
                  "allowed_paths": ["/depth_mm"], "branch": parent["branch"]})
        self.assertEqual(child_response.status_code, 201, child_response.text)
        child = child_response.json()
        # Executing an actual nonzero-exit process records a server-observed
        # failure; no API caller can mark this job or revision successful.
        self.app.state.worker.command = shutil.which("false")
        self.assertIsNotNone(self.app.state.worker.command)
        failed = await self.mesh(GeometryPlan.model_validate(child["plan"]), child)
        self.assertEqual(failed["status"], "failed")
        failure_id = failed["evidence"]["failure_id"]
        original = (await self.get_revision(child["id"]))["checks"]
        self.assertEqual(len(original), 1)
        self.assertEqual((original[0]["id"], original[0]["stage"], original[0]["status"]), (failure_id, "mesh", "failed"))
        folder = self.app.state.store.root / failed["id"]
        manifest_bytes = (folder / "mesh-execution.json").read_bytes()
        self.assertEqual(original[0]["artifact_hash"], hashlib.sha256(manifest_bytes).hexdigest())
        manifest = json.loads(manifest_bytes)
        self.assertEqual(manifest["revision_id"], child["id"])
        for name, expected_hash in manifest["artifacts"].items():
            self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), expected_hash)
        evidence = await self.client.get("/api/recovery/"+failure_id, headers=self.headers)
        self.assertEqual(evidence.status_code, 200, evidence.text)
        self.assertEqual(evidence.json()["failure"]["job_id"], failed["id"])
        self.assertEqual({item["id"] for item in evidence.json()["checkpoints"]}, {parent["id"], child["id"]})
        self.assertIn("not a proven root cause", evidence.json()["causation"])
        unrelated = await self.revision()
        denied = await self.client.post("/api/recovery/"+failure_id+"/restore", headers=self.headers,
            json={"checkpoint_id": unrelated["id"], "diagnosis": "Try unrelated case"})
        self.assertEqual(denied.status_code, 409)
        restored = await self.client.post("/api/recovery/"+failure_id+"/restore", headers=self.headers,
            json={"checkpoint_id": parent["id"], "diagnosis": "Restore prior model; investigate native execution separately"})
        self.assertEqual(restored.status_code, 201, restored.text)
        checkpoint = restored.json()
        self.assertNotEqual(checkpoint["branch"], parent["branch"])
        self.assertEqual(checkpoint["plan"], parent["plan"])
        self.assertEqual(checkpoint["recovery"]["failure_id"], failure_id)
        self.assertEqual(checkpoint["recovery"]["required_rerun"], ["mesh", "solver", "target"])
        self.assertEqual((await self.get_revision(checkpoint["id"]))["checks"], [])
        self.assertEqual((await self.get_revision(child["id"]))["checks"], original)
        self.assertEqual(self.app.state.store.get(failed["id"])["status"], "failed")
        self.assertEqual(self.app.state.history.head(parent["branch"])["id"], child["id"])
        self.assertIsNone(self.app.state.store.next())

    async def test_passed_check_and_unknown_failure_cannot_trigger_recovery(self):
        revision = await self.revision()
        mesh = await self.mesh(grid_plan(), revision)
        self.assertEqual(mesh["status"], "completed", mesh["error"])
        check_id = (await self.get_revision(revision["id"]))["checks"][0]["id"]
        for failure_id in (check_id, "a"*64):
            response = await self.client.get("/api/recovery/"+failure_id, headers=self.headers)
            self.assertEqual(response.status_code, 409)
            restore = await self.client.post("/api/recovery/"+failure_id+"/restore", headers=self.headers,
                json={"checkpoint_id": revision["id"], "diagnosis": "No actual failure"})
            self.assertEqual(restore.status_code, 409)
        self.assertEqual(self.app.state.history.head(revision["branch"])["id"], revision["id"])

    async def test_supplied_edits_must_match_spec_before_branch_is_advanced(self):
        parent = await self.revision()
        proposed = grid_plan().model_dump(mode="json")
        proposed["depth_mm"] = 2.
        request = {"spec": proposed, "parent_id": parent["id"], "branch": parent["branch"],
                   "edits": [{"op": "replace", "path": "/depth_mm", "value": 3.}],
                   "allowed_paths": ["/depth_mm"], "reason": "Change the depth to two millimeters"}
        rejected = await self.client.post("/api/revisions", headers=self.headers, json=request)
        self.assertEqual(rejected.status_code, 422, rejected.text)
        self.assertEqual(self.app.state.history.head(parent["branch"])["id"], parent["id"])
        self.assertEqual((await self.get_revision(parent["id"]))["plan"], parent["plan"])
        self.assertEqual(len((await self.get_revision(parent["id"]))["lineage"]), 1)
        # The same parent remains usable after rejection; no hidden three-mm
        # revision was committed or made the next valid save stale.
        request["edits"][0]["value"] = 2.
        accepted = await self.client.post("/api/revisions", headers=self.headers, json=request)
        self.assertEqual(accepted.status_code, 201, accepted.text)
        self.assertEqual(accepted.json()["plan"], proposed)
        self.assertEqual(accepted.json()["parent_id"], parent["id"])
        self.assertEqual(len((await self.get_revision(accepted.json()["id"]))["lineage"]), 2)

    async def test_title_revision_cannot_reuse_previous_revision_mesh(self):
        plan = grid_plan()
        parent = await self.revision(plan)
        mesh = await self.mesh(plan, parent)
        self.assertEqual(mesh["status"], "completed", mesh["error"])
        changed = await self.client.post("/api/revisions/"+parent["id"]+"/patch", headers=self.headers,
            json={"edits": [{"op": "replace", "path": "/title", "value": "Same geometry, revised title"}],
                  "allowed_paths": ["/title"], "branch": parent["branch"]})
        self.assertEqual(changed.status_code, 201, changed.text)
        child = changed.json()
        self.assertNotEqual(child["id"], parent["id"])
        self.assertEqual(child["invalidated_stages"], [])
        self.assertEqual((await self.get_revision(child["id"]))["checks"], [])
        # Even a presentation-only revision needs evidence bound to its own
        # exact spec. The API deliberately does not infer equivalence here.
        response = await self.client.post("/api/plan/jobs", headers=self.headers,
            json={"spec": child["plan"], "mesh_job_id": mesh["id"], "revision_id": child["id"]})
        self.assertEqual(response.status_code, 409, response.text)
        self.assertIsNone(self.app.state.store.next())
        original = await self.get_revision(parent["id"])
        self.assertEqual([(check["stage"], check["status"]) for check in original["checks"]], [("mesh", "passed")])
