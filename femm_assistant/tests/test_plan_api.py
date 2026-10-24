"""General models share authenticated exports and the existing bounded job queue."""
import io
import json
import os
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
from app import decisions, vision


class SettingsTests(unittest.TestCase):
    def test_model_defaults_preserve_explicit_overrides_and_vision_disable(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
            self.assertEqual(settings.ai_model, "google/gemini-3.1-flash-lite")
            self.assertEqual(settings.vision_model, settings.ai_model)
        with patch.dict(os.environ, {"AI_MODEL": "existing-model", "VISION_MODEL": ""}, clear=True):
            settings = Settings.from_env()
            self.assertEqual(settings.ai_model, "existing-model")
            self.assertEqual(settings.vision_model, "")


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
        for endpoint in ("preview", "export", "mesh", "jobs", "interpret", "review"):
            response = await self.client.post("/api/plan/" + endpoint, json={})
            self.assertEqual(response.status_code, 401)
        config = (await self.client.get("/api/config")).json()
        self.assertEqual(config["plan_example"]["kind"], "magnetostatic_plan")
        self.assertIn("$defs", config["plan_schema"])
        self.assertIsInstance(config["argument_diagnosis_available"], bool)
        self.assertIsInstance(config["visual_review_available"], bool)
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

    async def test_valid_geometry_is_previewable_with_low_confidence_function_route(self):
        spec = wire_example()
        trace = decisions.DecisionTrace(stage="routing", question="route", answer="create_plan",
            probability=.34, confidence=0, probabilities={"create_plan": .34, "modify_plan": .33, "clarify": .33},
            passed=True, model="fixture")
        proposal = PlanProposal(spec=spec, explanation="Created the requested geometry.",
            attempts=1, function_name="create_plan", decision_trace=[trace])
        with patch("app.planner.interpret_plan", new=AsyncMock(return_value=proposal)):
            response = await self.client.post("/api/plan/interpret", headers=self.headers,
                json={"prompt": "Create a circular conductor."})
        self.assertEqual(response.status_code, 200, response.text)
        packet = response.json()
        self.assertEqual(packet["spec"], spec.model_dump(mode="json"))
        self.assertNotIn("review_spec", packet)
        self.assertIsNone(packet["decision_trace"][0]["threshold"])
        preview = await self.client.post("/api/plan/preview", headers=self.headers, json=packet["spec"])
        self.assertEqual(preview.status_code, 200)
        self.assertIsNone(self.app.state.store.next())

    async def test_failed_job_diagnosis_requires_authentication_and_an_actual_failed_job(self):
        path = "/api/jobs/" + "a" * 32 + "/diagnose"
        self.assertEqual((await self.client.post(path, json={})).status_code, 401)
        with patch("app.decisions.diagnose_call", new=AsyncMock()) as diagnose:
            response = await self.client.post(path, headers=self.headers, json={})
            self.assertEqual(response.status_code, 404)
            for status in ["queued", "running", "completed"]:
                job = self.app.state.store.create(wire_example(), kind="mesh")
                self.app.state.store.update(job["id"], status, result={"status": "passed"})
                response = await self.client.post("/api/jobs/" + job["id"] + "/diagnose",
                    headers=self.headers, json={})
                self.assertEqual(response.status_code, 409, response.text)
            diagnose.assert_not_awaited()

    async def test_failed_job_diagnosis_is_cached_advice_and_preserves_failure(self):
        for spec, kind, function in [(wire_example(), "mesh", "mesh_plan"),
                                     (wire_example(), "solve", "solve_plan"),
                                     (CCoreSpec(), "solve", "solve_ccore")]:
            with self.subTest(kind=kind):
                job = self.app.state.store.create(spec, kind=kind)
                self.app.state.store.update(job["id"], "failed", result={"native_detail": "unchanged"},
                    error="Native FEMM fixture failure")
                folder = self.app.state.store.root / job["id"]
                folder.mkdir()
                (folder / "solver.log").write_text("Native mesh fixture failed at region wire.\n")
                diagnosis = decisions.CallDiagnosis(arguments_match=True, trace=[decisions.DecisionTrace(
                    stage="function_error", question="arguments_match", answer="yes", probability=.9,
                    probabilities={"yes": .9, "no": .1}, passed=True, model="fixture")])
                with patch("app.decisions.diagnose_call", new=AsyncMock(return_value=diagnosis)) as mock:
                    response = await self.client.post("/api/jobs/" + job["id"] + "/diagnose",
                        headers=self.headers, json={})
                    self.assertEqual(response.status_code, 200, response.text)
                    packet = response.json()
                    self.assertEqual(packet["function_name"], function)
                    self.assertTrue(packet["arguments_match"])
                    self.assertTrue(packet["advisory"])
                    self.assertEqual(packet["function_status"], "failed")
                    self.assertEqual(packet["original_error"], "Native FEMM fixture failure")
                    self.assertEqual(mock.call_args.args[4], function)
                    self.assertIn(spec.model_dump(mode="json"), mock.call_args.args[5].values())
                    self.assertIn("Native FEMM fixture failure", json.dumps(mock.call_args.args[6]))
                    second = await self.client.post("/api/jobs/" + job["id"] + "/diagnose",
                        headers=self.headers, json={})
                    self.assertEqual(second.status_code, 200, second.text)
                    self.assertEqual(second.json(), packet)
                    self.assertEqual(mock.await_count, 1)
                stored = self.app.state.store.get(job["id"])
                self.assertEqual(stored["status"], "failed")
                self.assertEqual(stored["error"], "Native FEMM fixture failure")
                self.assertEqual(stored["spec"], spec.model_dump(mode="json"))
                self.assertEqual(stored["result"]["native_detail"], "unchanged")
                self.assertEqual(stored["result"]["function_diagnosis"], packet)
                self.assertNotIn(self.settings.ai_key, response.text)
        self.assertIsNone(self.app.state.store.next())

    async def test_diagnosis_failure_does_not_mutate_failed_job_or_allow_solving(self):
        job = self.app.state.store.create(wire_example(), kind="mesh")
        self.app.state.store.update(job["id"], "failed", error="Native fixture failure")
        before = self.app.state.store.get(job["id"])
        with patch("app.decisions.diagnose_call", new=AsyncMock(side_effect=decisions.DecisionError(502, "Decision provider unavailable."))):
            response = await self.client.post("/api/jobs/" + job["id"] + "/diagnose",
                headers=self.headers, json={})
        self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(self.app.state.store.get(job["id"]), before)
        with patch("app.main.shutil.which", return_value="femm-lua"):
            response = await self.client.post("/api/plan/jobs", headers=self.headers,
                json={"spec": wire_example().model_dump(), "mesh_job_id": job["id"]})
        self.assertEqual(response.status_code, 409)

    async def test_visual_review_is_advisory_and_does_not_mutate_geometry_or_jobs(self):
        spec = wire_example()
        job = self.app.state.store.create(spec, kind="mesh")
        before = self.app.state.store.get(job["id"])
        review = vision.VisualReview(advisory=True, status="issues", summary="Inspect the wire position.",
            findings=[vision.VisualFinding(location="wire", issue="A visible fixture discrepancy.",
                suggestion="Compare the coordinates with your request.")], requested_model="fixture-vision")
        with patch("app.vision.review", new=AsyncMock(return_value=review)) as mock:
            response = await self.client.post("/api/plan/review", headers=self.headers,
                json={"prompt": "Create a circular conductor.", "spec": spec.model_dump()})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["advisory"])
        self.assertEqual(response.json()["status"], "issues")
        self.assertEqual(mock.call_args.args[-1], spec)
        self.assertEqual(self.app.state.store.get(job["id"]), before)
        self.assertNotIn(self.settings.ai_key, response.text)
        preview = await self.client.post("/api/plan/preview", headers=self.headers, json=spec.model_dump())
        self.assertEqual(preview.status_code, 200)

    async def test_visual_review_rejects_invalid_request_before_provider(self):
        with patch("app.vision.review", new=AsyncMock()) as mock:
            for payload in [
                {"prompt": "Create a conductor."},
                {"prompt": "", "spec": wire_example().model_dump()},
                {"prompt": "Create a conductor.", "spec": {"kind": "invalid"}},
                {"prompt": "Create a conductor.", "spec": wire_example().model_dump(), "extra": True},
                {"prompt": "Create a conductor.", "spec": wire_example().model_dump(), "mesh_job_id": "../mesh"},
            ]:
                with self.subTest(payload=payload):
                    response = await self.client.post("/api/plan/review", headers=self.headers, json=payload)
                    self.assertEqual(response.status_code, 422, response.text)
            mock.assert_not_awaited()

    async def test_visual_review_mesh_requires_completed_passed_exact_model_evidence(self):
        spec = wire_example()
        job = self.app.state.store.create(spec, kind="mesh")
        body = {"prompt": "Inspect the requested wire mesh.", "spec": spec.model_dump(), "mesh_job_id": job["id"]}
        with patch("app.vision.review", new=AsyncMock()) as mock:
            response = await self.client.post("/api/plan/review", headers=self.headers, json=body)
            self.assertEqual(response.status_code, 409)
            self.app.state.store.update(job["id"], "completed", result={"status": "failed"})
            response = await self.client.post("/api/plan/review", headers=self.headers, json=body)
            self.assertEqual(response.status_code, 409)
            self.app.state.store.update(job["id"], "completed", result={"status": "passed"})
            changed = spec.model_dump()
            changed["circuits"][0]["current_a"] = 2
            response = await self.client.post("/api/plan/review", headers=self.headers, json={**body, "spec": changed})
            self.assertEqual(response.status_code, 409)
            response = await self.client.post("/api/plan/review", headers=self.headers, json=body)
            self.assertIn(response.status_code, (404, 409))
            mock.assert_not_awaited()
        folder = self.app.state.store.root / job["id"]
        folder.mkdir()
        mesh_svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path d="M 0 0 L 10 10"/></svg>'
        (folder / "mesh.svg").write_text(mesh_svg)
        before = self.app.state.store.get(job["id"])
        review = vision.VisualReview(advisory=True, status="uncertain", summary="Fixture mesh view.",
            findings=[], requested_model="fixture-vision")
        with patch("app.vision.review", new=AsyncMock(return_value=review)) as mock:
            response = await self.client.post("/api/plan/review", headers=self.headers, json=body)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(mock.call_args.kwargs["mesh_svg"], mesh_svg)
        self.assertEqual(self.app.state.store.get(job["id"]), before)

    async def test_visual_provider_failure_leaves_preview_and_job_state_available(self):
        spec = wire_example()
        job = self.app.state.store.create(spec, kind="mesh")
        before = self.app.state.store.get(job["id"])
        with patch("app.vision.review", new=AsyncMock(side_effect=vision.VisionError(502, "Visual provider unavailable."))):
            response = await self.client.post("/api/plan/review", headers=self.headers,
                json={"prompt": "Inspect the requested wire.", "spec": spec.model_dump()})
        self.assertEqual(response.status_code, 502, response.text)
        self.assertNotIn(self.settings.ai_key, response.text)
        self.assertEqual(self.app.state.store.get(job["id"]), before)
        preview = await self.client.post("/api/plan/preview", headers=self.headers, json=spec.model_dump())
        self.assertEqual(preview.status_code, 200)

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
