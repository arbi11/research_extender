import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
import zipfile

import httpx
from pydantic import ValidationError
from app.ai import Proposal
from app.generate import lua_script, source_zip
from app.jobs import BusyError, JobStore, SolverWorker
from app.main import Settings, create_app
from app.model import CCoreSpec


class ModelTests(unittest.TestCase):
    def test_rejects_unsafe_or_infeasible_input(self):
        for change in [{"gap_mm": 0}, {"width_mm": 30}, {"turns": 2.5}, {"current_a": float("nan")},
                       {"current_a": float("inf")}, {"geometry": "3d_motor"}, {"command": "os.system('anything')"},
                       {"gap_mm": '1); print("injection")'}]:
            with self.subTest(change=change), self.assertRaises(ValidationError):
                CCoreSpec.model_validate(change)

    def test_bundle_runs_without_repository_imports(self):
        spec = CCoreSpec(current_a=2, turns=600, gap_mm=1.5)
        with zipfile.ZipFile(io.BytesIO(source_zip(spec))) as archive:
            self.assertEqual(set(archive.namelist()), {"model.lua", "run.py", "spec.json", "README.txt", "geometry.svg"})
            self.assertEqual(json.loads(archive.read("spec.json"))["gap_mm"], 1.5)
            self.assertEqual(archive.read("model.lua").decode(), lua_script(spec))
            compile(archive.read("run.py"), "run.py", "exec")
            self.assertNotIn("app.", archive.read("run.py").decode())
            # No simulated results can appear in a source-only download.
            self.assertNotIn("model.ans", archive.namelist())


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = Settings(data_dir=Path(self.temp.name), token="test-access", start_worker=False)
        self.app = create_app(self.settings)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")
        self.headers = {"Authorization": "Bearer test-access"}

    async def asyncTearDown(self):
        await self.client.aclose()
        self.temp.cleanup()

    async def test_every_model_job_and_artifact_endpoint_requires_auth(self):
        for method, url, body in [("POST", "/api/preview", {}), ("POST", "/api/export", {}),
                                   ("POST", "/api/jobs", {}), ("POST", "/api/interpret", {"prompt": "change current"}),
                                   ("GET", "/api/jobs/" + "a"*32, None),
                                   ("GET", "/api/jobs/" + "a"*32 + "/download.zip", None)]:
            with self.subTest(url=url):
                response = await self.client.request(method, url, json=body)
                self.assertEqual(response.status_code, 401)

    async def test_form_preview_and_export_work_without_ai_or_solver(self):
        response = await self.client.post("/api/preview", headers=self.headers, json={})
        self.assertEqual(response.status_code, 200)
        self.assertIn("<svg", response.text)
        exported = await self.client.post("/api/export", headers=self.headers, json={})
        self.assertEqual(exported.status_code, 200)
        self.assertTrue(zipfile.is_zipfile(io.BytesIO(exported.content)))
        self.assertEqual((await self.client.post("/api/interpret", headers=self.headers, json={"prompt": "double the current"})).status_code, 503)
        with patch("app.main.shutil.which", return_value=None):
            self.assertEqual((await self.client.post("/api/jobs", headers=self.headers, json={})).status_code, 503)

    async def test_ai_output_is_validated_and_never_auto_submitted(self):
        self.settings.ai_key, self.settings.ai_model = "server-secret", "configured-model"
        proposal = Proposal(spec=CCoreSpec(current_a=2), explanation="Changed current to 2 A.", questions=[])
        with patch("app.ai.interpret", new=AsyncMock(return_value=proposal)):
            response = await self.client.post("/api/interpret", headers=self.headers, json={"prompt": "double the current"})
        self.assertEqual(response.json()["spec"]["current_a"], 2)
        self.assertNotIn("server-secret", response.text)
        self.assertIsNone(self.app.state.store.next())

    async def test_bad_spec_oversize_body_and_incomplete_download(self):
        self.assertEqual((await self.client.post("/api/preview", headers=self.headers, json={"width_mm": 10})).status_code, 422)
        self.assertEqual((await self.client.post("/api/preview", headers=self.headers, content=b"x"*17000)).status_code, 413)
        job = self.app.state.store.create(CCoreSpec())
        response = await self.client.get(f'/api/jobs/{job["id"]}/download.zip', headers=self.headers)
        self.assertEqual(response.status_code, 409)
        self.assertEqual((await self.client.get("/api/jobs/not-a-job", headers=self.headers)).status_code, 404)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = JobStore(self.root / "data")

    def tearDown(self):
        self.temp.cleanup()

    def test_queue_limit_recovery_and_expiry(self):
        job = self.store.create(CCoreSpec(), max_pending=1)
        with self.assertRaises(BusyError):
            self.store.create(CCoreSpec(), max_pending=1)
        self.store.update(job["id"], "running")
        self.store.recover()
        self.assertEqual(self.store.get(job["id"])["status"], "failed")
        folder = self.store.root / job["id"]
        folder.mkdir()
        (folder / "private-file").write_text("data")
        self.store.expire(0)
        self.assertIsNone(self.store.get(job["id"]))
        self.assertFalse(folder.exists())

    def test_missing_solver_is_failed_never_completed(self):
        job = self.store.create(CCoreSpec())
        SolverWorker(self.store, command="/definitely/missing/femm-lua").run(job)
        result = self.store.get(job["id"])
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["result"])
        self.assertFalse((self.store.root / job["id"] / "download.zip").exists())

    def test_successful_process_without_physics_output_is_failed(self):
        job = self.store.create(CCoreSpec())
        with patch("app.jobs.socket.socket") as socket_mock:
            socket_mock.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 48015)
            SolverWorker(self.store, command="/bin/true").run(job)
        self.assertEqual(self.store.get(job["id"])["status"], "failed")
        self.assertIn("solution file", self.store.get(job["id"])["error"])

    def test_timeout_terminates_process_and_records_failure(self):
        launcher = self.root / "slow-solver"
        launcher.write_text("#!/bin/sh\nsleep 30\n")
        launcher.chmod(0o700)
        job = self.store.create(CCoreSpec())
        worker = SolverWorker(self.store, command=str(launcher), timeout=.1)
        with patch("app.jobs.socket.socket") as socket_mock:
            socket_mock.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 48015)
            worker.run(job)
        self.assertEqual(self.store.get(job["id"])["status"], "failed")
        self.assertIn("limit", self.store.get(job["id"])["error"])
        self.assertIsNone(worker.process)


if __name__ == "__main__":
    unittest.main()
