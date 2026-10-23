"""General models share authenticated exports and the existing bounded job queue."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
import zipfile

import httpx

from app.main import Settings, create_app
from app.model import CCoreSpec
from app.plan import wire_example
from app.planner import PlanProposal, PlannerError


class PlanApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = Settings(data_dir=Path(self.temp.name), token="pilot", start_worker=False,
                                 ai_key="test-key-never-returned", ai_model="test-model")
        self.app = create_app(self.settings)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")
        self.headers = {"Authorization": "Bearer pilot"}

    async def asyncTearDown(self):
        await self.client.aclose()
        self.temp.cleanup()

    async def test_auth_preview_and_portable_export_for_new_topology(self):
        for endpoint in ("preview", "export", "mesh", "jobs", "interpret"):
            response = await self.client.post("/api/plan/" + endpoint, json={})
            self.assertEqual(response.status_code, 401)
        config = (await self.client.get("/api/config")).json()
        self.assertEqual(config["plan_example"]["kind"], "magnetostatic_plan")
        self.assertIn("$defs", config["plan_schema"])
        self.assertNotIn(self.settings.ai_key, json.dumps(config))
        spec = wire_example().model_dump()
        preview = await self.client.post("/api/plan/preview", headers=self.headers, json=spec)
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertIn("<svg", preview.text)
        exported = await self.client.post("/api/plan/export", headers=self.headers, json=spec)
        self.assertEqual(exported.status_code, 200, exported.text)
        with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
            self.assertEqual(json.loads(archive.read("spec.json"))["kind"], "magnetostatic_plan")
            self.assertIn("mi_addarc", archive.read("model.lua").decode())
            self.assertNotIn("model.ans", archive.namelist())
            self.assertNotIn("results.json", archive.namelist())
            self.assertNotIn(self.settings.ai_key, archive.read("README.txt").decode())

    async def test_create_modify_and_clarify_do_not_auto_solve(self):
        spec = wire_example()
        proposal = PlanProposal(spec=spec, explanation="Created a circular conductor.", questions=[], attempts=1)
        with patch("app.planner.interpret_plan", new=AsyncMock(return_value=proposal)) as mock:
            created = await self.client.post("/api/plan/interpret", headers=self.headers,
                                             json={"prompt": "Create a circular conductor."})
            self.assertEqual(created.status_code, 200, created.text)
            self.assertIsNone(mock.call_args.args[1])
            modified = await self.client.post("/api/plan/interpret", headers=self.headers,
                                              json={"prompt": "Double the conductor current.", "current": spec.model_dump()})
            self.assertEqual(modified.status_code, 200, modified.text)
            self.assertEqual(mock.call_args.args[1], spec)
        clarification = PlanProposal(spec=None, explanation="Need dimensions.",
                                     questions=["What is the conductor radius?"], attempts=1)
        with patch("app.planner.interpret_plan", new=AsyncMock(return_value=clarification)):
            response = await self.client.post("/api/plan/interpret", headers=self.headers,
                                               json={"prompt": "Create a conductor."})
            self.assertIsNone(response.json()["spec"])
            self.assertTrue(response.json()["questions"])
        self.assertIsNone(self.app.state.store.next())

    async def test_both_model_types_share_queue_and_result_endpoints(self):
        with patch("app.main.shutil.which", return_value="femm-lua"):
            first = await self.client.post("/api/plan/mesh", headers=self.headers, json=wire_example().model_dump())
            self.assertEqual(first.status_code, 202, first.text)
            self.assertEqual(first.json()["spec"]["kind"], "magnetostatic_plan")
            for _ in range(4):
                response = await self.client.post("/api/jobs", headers=self.headers, json=CCoreSpec().model_dump())
                self.assertEqual(response.status_code, 202, response.text)
            full = await self.client.post("/api/plan/mesh", headers=self.headers, json=wire_example().model_dump())
            self.assertEqual(full.status_code, 429)
        read = await self.client.get("/api/jobs/" + first.json()["id"], headers=self.headers)
        self.assertEqual(read.status_code, 200)
        self.assertEqual(read.json()["status"], "queued")
        self.assertEqual((await self.client.get("/api/jobs/" + first.json()["id"] + "/download.zip", headers=self.headers)).status_code, 409)

    async def test_solve_requires_successful_mesh_of_the_unchanged_plan(self):
        spec = wire_example().model_dump()
        with patch("app.main.shutil.which", return_value="femm-lua"):
            request = {"spec": spec, "mesh_job_id": "a" * 32}
            response = await self.client.post("/api/plan/jobs", headers=self.headers, json=request)
            self.assertEqual(response.status_code, 409)
            mesh = (await self.client.post("/api/plan/mesh", headers=self.headers, json=spec)).json()
            self.assertEqual(mesh["kind"], "mesh")
            request["mesh_job_id"] = mesh["id"]
            response = await self.client.post("/api/plan/jobs", headers=self.headers, json=request)
            self.assertEqual(response.status_code, 409)
            self.app.state.store.update(mesh["id"], "completed", result={"status": "passed"})
            changed = wire_example().model_dump()
            changed["circuits"][0]["current_a"] = 2
            response = await self.client.post("/api/plan/jobs", headers=self.headers,
                                              json={"spec": changed, "mesh_job_id": mesh["id"]})
            self.assertEqual(response.status_code, 409)
            response = await self.client.post("/api/plan/jobs", headers=self.headers, json=request)
            self.assertEqual(response.status_code, 202, response.text)
            self.assertEqual(response.json()["kind"], "solve")
            request["mesh_job_id"] = response.json()["id"]
            response = await self.client.post("/api/plan/jobs", headers=self.headers, json=request)
            self.assertEqual(response.status_code, 409)
            self.app.state.store.update(mesh["id"], "failed", error="invalid triangles")
            request["mesh_job_id"] = mesh["id"]
            response = await self.client.post("/api/plan/jobs", headers=self.headers, json=request)
            self.assertEqual(response.status_code, 409)

    async def test_credential_errors_and_body_limits_are_actionable(self):
        error = PlannerError(402, "The AI account has insufficient credits.")
        with patch("app.planner.interpret_plan", new=AsyncMock(side_effect=error)):
            response = await self.client.post("/api/plan/interpret", headers=self.headers,
                                               json={"prompt": "Create a round conductor."})
        self.assertEqual(response.status_code, 402)
        self.assertIn("credits", response.json()["detail"])
        self.assertNotIn(self.settings.ai_key, response.text)
        oversized = await self.client.post("/api/plan/preview", headers=self.headers, content=b"x" * 128001)
        self.assertEqual(oversized.status_code, 413)
        existing_limit = await self.client.post("/api/preview", headers=self.headers, content=b"x" * 17000)
        self.assertEqual(existing_limit.status_code, 413)


if __name__ == "__main__":
    unittest.main()
