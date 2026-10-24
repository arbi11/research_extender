"""Authenticated scoped repair queues once and preserves failed native evidence."""
import asyncio
from collections import Counter
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import Depends, FastAPI, Header, HTTPException
import httpx

from app.history import ModelRevisionStore, apply_feature_edits, plan_diff
from app.jobs import JobStore
from app.plan import GeometryPlan, wire_example
from app.recovery import RecoveryError, RecoveryProposal, STAGE_FUNCTIONS
from app.recovery_api import register_routes
from app.replication import ReplicationStore
from evaluation.recovery_loop import CASE_ID, CURRENT_PATH, reference_target


class TestBudget:
    def __init__(self):
        self.counts = Counter()
        self.reject = None

    def use(self, kind, limit):
        if self.reject == kind:
            raise HTTPException(429, "Fixture hourly limit reached.")
        self.counts[kind] += 1


class RecoveryApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.history = ModelRevisionStore(root)
        self.jobs = JobStore(root)
        self.replication = ReplicationStore(root)
        self.budget = TestBudget()
        self.settings = SimpleNamespace(femm_lua=sys.executable, ai_key="private-test-key",
            ai_model="test/json-model", ai_base_url="https://provider.test/api/v1",
            decision_model="typesafe/jev-1.14", decision_base_url="https://provider.test/api/alpha")
        self.baseline = self.history.create(wire_example())
        self.wrong = self.history.apply_patch(self.baseline["id"], [
            {"op": "replace", "path": CURRENT_PATH, "value": 2.0}], allowed_paths=[CURRENT_PATH])
        self.job, self.failure = self.make_failure()
        self.app = FastAPI()

        async def authenticate(authorization: str = Header(default="")):
            if authorization != "Bearer pilot":
                raise HTTPException(401, "Token required.")

        register_routes(self.app, [Depends(authenticate)], self.settings, self.budget,
                        self.history, self.jobs, self.replication)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")
        self.headers = {"Authorization": "Bearer pilot"}

    async def asyncTearDown(self):
        await self.client.aclose()
        self.temp.cleanup()

    def make_failure(self, *, stage="mesh", contract_id=None):
        job = self.jobs.create(GeometryPlan.model_validate(self.wrong["plan"]),
            kind="mesh" if stage == "mesh" else "solve", revision_id=self.wrong["id"],
            target_contract_id=contract_id)
        self.jobs.update(job["id"], "completed" if stage == "target" else "failed",
                         result={"fixture": True}, error=None if stage == "target" else "Native mesh fixture failed.")
        check = self.history.record_check(self.wrong["id"], stage=stage, status="failed", job_id=job["id"],
            details={"error": "Observed failure fixture", "target_contract_id": contract_id})
        self.jobs.annotate(job["id"], failure_id=check["id"])
        return self.jobs.get(job["id"]), check

    def proposal(self, context, *, edits=None, checkpoint=None, questions=None, allowed_paths=None):
        checkpoint = checkpoint or self.wrong
        edits = ([{"op": "replace", "path": "/mesh_mm", "value": .25}] if edits is None else edits)
        spec = GeometryPlan.model_validate(checkpoint["plan"])
        if edits:
            spec, changes = apply_feature_edits(spec, edits)
        else:
            changes = []
        return RecoveryProposal(failure_id=context["failure"]["id"],
            function_name=STAGE_FUNCTIONS[context["failure"]["stage"]], checkpoint_id=checkpoint["id"],
            edits=edits, explanation="Repair only the permitted implementation inputs.", questions=questions or [],
            spec=spec, changes=plan_diff(GeometryPlan.model_validate(self.wrong["plan"]), spec),
            patch_changes=changes, allowed_paths=allowed_paths or [])

    async def post(self, body=None, job=None):
        return await self.client.post(f"/api/jobs/{(job or self.job)['id']}/recover",
                                      headers=self.headers, json=body or {})

    async def test_authenticated_mesh_recovery_queues_new_revision_once_and_caches_full_response(self):
        path = f"/api/jobs/{self.job['id']}/recover"
        self.assertEqual((await self.client.post(path, json={})).status_code, 401)
        captured = []

        async def propose(context, *args, **kwargs):
            captured.append((context, kwargs))
            return self.proposal(context, allowed_paths=kwargs["allowed_paths"])

        with patch("app.recovery_api.recovery.propose_recovery", new=AsyncMock(side_effect=propose)) as provider:
            first = await self.post()
            self.assertEqual(first.status_code, 202, first.text)
            repeated = await self.post({"scope": "mesh"})
            self.assertEqual(repeated.status_code, 202, repeated.text)
            self.assertEqual(first.json(), repeated.json())
            provider.assert_awaited_once()
        response = first.json()
        self.assertEqual(response["mesh_job"]["kind"], "mesh")
        self.assertEqual(response["mesh_job"]["revision_id"], response["revision"]["id"])
        self.assertEqual(response["revision"]["plan"]["mesh_mm"], .25)
        self.assertEqual(response["revision"]["plan"]["circuits"][0]["current_a"], 2)
        self.assertEqual(self.history.head(), self.wrong)
        self.assertEqual(self.history.get_check(self.failure["id"]), self.failure)
        self.assertEqual(self.history.checks(response["revision"]["id"]), [])
        self.assertEqual(self.jobs.get(self.job["id"])["status"], "failed")
        self.assertEqual(self.jobs.get(self.job["id"])["evidence"]["recovery_attempt"]["response"], response)
        self.assertEqual(self.budget.counts, {"ai": 1, "function_diagnostic": 1, "recovery": 1, "mesh": 1})
        self.assertEqual(captured[0][1]["allowed_paths"], ["/mesh_mm", "/regions/wire/mesh_mm"])

    async def test_completed_target_failure_rebinds_identical_targets_and_tolerances(self):
        original_contract = self.replication.put_contract(self.wrong["id"], CASE_ID,
            [reference_target()], GeometryPlan.model_validate(self.wrong["plan"]), evidence_id="e" * 64)
        job, failure = self.make_failure(stage="target", contract_id=original_contract["contract_id"])

        async def propose(context, *args, **kwargs):
            return self.proposal(context, edits=[], checkpoint=self.baseline, allowed_paths=kwargs["allowed_paths"])

        with patch("app.recovery_api.recovery.propose_recovery", new=AsyncMock(side_effect=propose)):
            response = await self.post({"scope": "last_edit"}, job=job)
        self.assertEqual(response.status_code, 202, response.text)
        body = response.json()
        rebound = self.replication.get_contract(body["target_contract_id"])
        self.assertEqual(rebound["revision_id"], body["revision"]["id"])
        self.assertEqual(rebound["contract"]["targets_sha256"], original_contract["contract"]["targets_sha256"])
        self.assertEqual(rebound["contract"]["targets"], original_contract["contract"]["targets"])
        self.assertEqual(rebound["evidence_id"], original_contract["evidence_id"])
        self.assertNotEqual(rebound["contract"]["model_sha256"], original_contract["contract"]["model_sha256"])
        self.assertEqual(body["revision"]["plan"], self.baseline["plan"])
        self.assertEqual(self.jobs.get(job["id"])["status"], "completed")
        self.assertEqual(self.history.get_check(failure["id"])["status"], "failed")

    async def test_no_failure_foreign_failure_or_wrong_revision_cannot_call_provider(self):
        queued = self.jobs.create(wire_example(), kind="mesh", revision_id=self.baseline["id"])
        self.jobs.annotate(queued["id"], failure_id=self.failure["id"])
        with patch("app.recovery_api.recovery.propose_recovery", new=AsyncMock()) as provider:
            self.assertEqual((await self.post(job=queued)).status_code, 409)
            self.jobs.annotate(self.job["id"], failure_id="missing")
            self.assertEqual((await self.post()).status_code, 409)
            self.assertEqual((await self.client.post("/api/jobs/" + "a" * 32 + "/recover", headers=self.headers, json={})).status_code, 404)
            provider.assert_not_awaited()

    async def test_scope_requests_and_original_native_log_tail_are_bounded(self):
        folder = self.jobs.root / self.job["id"]
        folder.mkdir()
        (folder / "solver.log").write_text("x" * 9000 + "\nRetained native tail.\n")
        captured = []

        async def propose(context, *args, **kwargs):
            captured.append((context, kwargs))
            return self.proposal(context, edits=[{"op": "replace", "path": CURRENT_PATH, "value": 1.0}],
                                 allowed_paths=kwargs["allowed_paths"])

        with patch("app.recovery_api.recovery.propose_recovery", new=AsyncMock(side_effect=propose)):
            response = await self.post({"scope": "custom", "allowed_paths": [CURRENT_PATH]})
        self.assertEqual(response.status_code, 202, response.text)
        details = captured[0][0]["failure"]["details"]
        self.assertLessEqual(len(details["native_log_tail"]), 6000)
        self.assertTrue(details["native_log_tail"].endswith("Retained native tail.\n"))
        self.assertEqual(details["native_error"], self.job["error"])
        self.assertNotIn("native_log_tail", self.history.get_check(self.failure["id"])["details"])
        self.assertEqual(len(captured[0][0]["checkpoint_models"]), 2)
        self.assertEqual(captured[0][1]["allowed_paths"], [CURRENT_PATH])

    async def test_bad_custom_scope_and_last_edit_without_history_do_not_buy_a_proposal(self):
        for body in [{"scope": "custom"}, {"scope": "custom", "allowed_paths": ["/regions/wire/id"]},
                     {"scope": "mesh", "allowed_paths": [CURRENT_PATH]}, {"scope": "custom", "allowed_paths": ["/../secret"]},
                     {"scope": "arbitrary"}, {"instruction": "x" * 3001}]:
            with self.subTest(body=body), patch("app.recovery_api.recovery.propose_recovery", new=AsyncMock()) as provider:
                self.assertEqual((await self.post(body)).status_code, 422)
                provider.assert_not_awaited()
        baseline_job = self.jobs.create(wire_example(), kind="mesh", revision_id=self.baseline["id"])
        self.jobs.update(baseline_job["id"], "failed", error="Native failure")
        failure = self.history.record_check(self.baseline["id"], stage="mesh", status="failed", job_id=baseline_job["id"])
        self.jobs.annotate(baseline_job["id"], failure_id=failure["id"])
        with patch("app.recovery_api.recovery.propose_recovery", new=AsyncMock()) as provider:
            self.assertEqual((await self.post({"scope": "last_edit"}, job=baseline_job)).status_code, 409)
            provider.assert_not_awaited()

    async def test_malicious_proposal_cannot_change_unpermitted_fields_and_is_not_retried(self):
        async def propose(context, *args, **kwargs):
            return self.proposal(context, edits=[{"op": "replace", "path": CURRENT_PATH, "value": 9.0}],
                                 allowed_paths=[CURRENT_PATH])

        with patch("app.recovery_api.recovery.propose_recovery", new=AsyncMock(side_effect=propose)) as provider:
            first = await self.post()
            second = await self.post()
            self.assertEqual(first.status_code, 409, first.text)
            self.assertEqual(second.status_code, 409)
            provider.assert_awaited_once()
        self.assertEqual(self.history.head(), self.wrong)
        self.assertEqual(self.jobs.get(self.job["id"])["evidence"]["recovery_attempt"]["status"], "failed")

    async def test_rollback_cannot_bypass_mesh_only_scope(self):
        async def propose(context, *args, **kwargs):
            return self.proposal(context, checkpoint=self.baseline, allowed_paths=kwargs["allowed_paths"])

        with patch("app.recovery_api.recovery.propose_recovery", new=AsyncMock(side_effect=propose)):
            response = await self.post()
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.history.head(), self.wrong)
        with self.history.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM revisions").fetchone()[0], 2)

    async def test_questions_are_cached_without_model_mutation_or_queueing(self):
        proposal = RecoveryProposal(failure_id=self.failure["id"], function_name="mesh_plan", checkpoint_id=None,
            edits=[], explanation="Need more failure evidence.", questions=["Which region failed?"], allowed_paths=[])
        with patch("app.recovery_api.recovery.propose_recovery", new=AsyncMock(return_value=proposal)) as provider:
            first, second = await self.post(), await self.post()
            self.assertEqual(first.status_code, 200, first.text)
            self.assertEqual(first.json(), second.json())
            provider.assert_awaited_once()
        self.assertIsNone(first.json()["revision"])
        self.assertIsNone(first.json()["mesh_job"])
        self.assertNotIn("mesh", self.budget.counts)

    async def test_changed_instructions_after_attempt_conflict_without_new_paid_work(self):
        with patch("app.recovery_api.recovery.propose_recovery", new=AsyncMock(
                side_effect=lambda context, *args, **kwargs: self.proposal(context))) as provider:
            self.assertEqual((await self.post()).status_code, 202)
            self.assertEqual((await self.post({"instruction": "Try another repair"})).status_code, 409)
            self.assertEqual((await self.post({"scope": "last_edit"})).status_code, 409)
            provider.assert_awaited_once()

    async def test_budget_rejection_does_not_start_or_cache_a_paid_attempt(self):
        self.budget.reject = "ai"
        with patch("app.recovery_api.recovery.propose_recovery", new=AsyncMock()) as provider:
            response = await self.post()
            self.assertEqual(response.status_code, 429)
            provider.assert_not_awaited()
        self.assertNotIn("recovery_attempt", self.jobs.get(self.job["id"])["evidence"])

    async def test_queue_becomes_full_after_proposal_then_retries_without_new_proposal_or_branch(self):
        queued = []

        async def propose(context, *args, **kwargs):
            for _ in range(5):
                queued.append(self.jobs.create(wire_example(), kind="mesh", revision_id=self.baseline["id"]))
            return self.proposal(context)

        with patch("app.recovery_api.recovery.propose_recovery", new=AsyncMock(side_effect=propose)) as provider:
            full = await self.post()
            self.assertEqual(full.status_code, 429, full.text)
            attempt = self.jobs.get(self.job["id"])["evidence"]["recovery_attempt"]
            self.assertEqual(attempt["status"], "ready_to_queue")
            revision_id = attempt["response"]["revision"]["id"]
            for job in queued:
                self.jobs.update(job["id"], "failed", error="Queue fixture released.")
            retried = await self.post()
            self.assertEqual(retried.status_code, 202, retried.text)
            self.assertEqual(retried.json()["revision"]["id"], revision_id)
            provider.assert_awaited_once()

    async def test_concurrent_same_request_buys_one_proposal_and_one_mesh_job(self):
        async def propose(context, *args, **kwargs):
            await asyncio.sleep(.02)
            return self.proposal(context)

        with patch("app.recovery_api.recovery.propose_recovery", new=AsyncMock(side_effect=propose)) as provider:
            first, second = await asyncio.gather(self.post(), self.post())
            self.assertEqual(first.status_code, 202, first.text)
            self.assertEqual(first.json(), second.json())
            provider.assert_awaited_once()
        with self.jobs.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM jobs WHERE status='queued'").fetchone()[0], 1)

    async def test_separate_server_instances_share_atomic_paid_work_reservation(self):
        another_app = FastAPI()
        register_routes(another_app, [], self.settings, self.budget, self.history, self.jobs, self.replication)
        other_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=another_app), base_url="http://other")

        async def propose(context, *args, **kwargs):
            await asyncio.sleep(.03)
            return self.proposal(context)

        try:
            with patch("app.recovery_api.recovery.propose_recovery", new=AsyncMock(side_effect=propose)) as provider:
                first, other = await asyncio.gather(self.post(), other_client.post(
                    f"/api/jobs/{self.job['id']}/recover", json={}))
                self.assertEqual(sorted([first.status_code, other.status_code]), [202, 409])
                provider.assert_awaited_once()
            with self.jobs.connect() as db:
                self.assertEqual(db.execute("SELECT count(*) FROM jobs WHERE status='queued'").fetchone()[0], 1)
        finally:
            await other_client.aclose()

    async def test_mismatching_proposal_spec_cannot_bypass_its_actual_feature_edits(self):
        async def propose(context, *args, **kwargs):
            proposal = self.proposal(context)
            value = proposal.spec.model_dump(mode="json")
            value["circuits"][0]["current_a"] = 9.0
            return proposal.model_copy(update={"spec": GeometryPlan.model_validate(value)})

        with patch("app.recovery_api.recovery.propose_recovery", new=AsyncMock(side_effect=propose)):
            response = await self.post()
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.history.head(), self.wrong)
        with self.jobs.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM jobs WHERE status='queued'").fetchone()[0], 0)

    async def test_invalid_frozen_target_record_is_rejected_before_paid_work(self):
        contract = self.replication.put_contract(self.wrong["id"], CASE_ID,
            [reference_target()], GeometryPlan.model_validate(self.wrong["plan"]))
        job, _ = self.make_failure(stage="target", contract_id=contract["contract_id"])
        with patch.object(self.replication, "get_contract", side_effect=ValueError("Broken stored digest")):
            with patch("app.recovery_api.recovery.propose_recovery", new=AsyncMock()) as provider:
                self.assertEqual((await self.post(job=job)).status_code, 409)
                provider.assert_not_awaited()

    async def test_provider_error_is_cached_and_unexpected_exception_does_not_expose_secrets(self):
        with patch("app.recovery_api.recovery.propose_recovery", new=AsyncMock(
                side_effect=RecoveryError(502, "Safe provider failure."))) as provider:
            first, second = await self.post(), await self.post()
            self.assertEqual(first.status_code, 502)
            self.assertEqual(first.json(), second.json())
            provider.assert_awaited_once()
        another, _ = self.make_failure()
        with patch("app.recovery_api.recovery.propose_recovery", new=AsyncMock(
                side_effect=RuntimeError("private-test-key upstream-secret"))):
            response = await self.post(job=another)
            self.assertEqual(response.status_code, 500)
            self.assertNotIn("private-test-key", response.text)
            self.assertNotIn("upstream-secret", response.text)


if __name__ == "__main__":
    unittest.main()
