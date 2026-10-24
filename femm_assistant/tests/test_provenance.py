"""A selected paper case cannot silently adopt an unrelated or edited draft."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app import papers, provenance
from app.main import Settings, create_app
from app.plan import GeometryPlan, wire_example
from app.planner import PlanProposal
from tests.test_papers import evidence, text_pdf


class PaperModelBindingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = create_app(Settings(data_dir=Path(self.temp.name), ai_key="test-only", start_worker=False))
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")
        self.store = self.app.state.replication
        uploaded = self.store.put_paper(text_pdf())
        doc = papers.PaperDocument.model_validate(uploaded["document"])
        packet = evidence(doc)
        extra = copy.deepcopy(packet["targets"][0])
        extra.update(id="displaced", case_id="displaced")
        packet["targets"].append(extra)
        self.record = self.store.put_evidence(uploaded["paper_id"], packet)
        self.plan = wire_example()
        self.binding = provenance.put_planning(self.store, self.plan, self.record["evidence_id"], "baseline", "Original centred case")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.temp.cleanup()

    def payload(self, **overrides):
        return {"spec": self.plan.model_dump(mode="json"), "planning_id": self.binding["planning_id"],
                "evidence_id": self.record["evidence_id"], "case_id": "baseline", **overrides}

    async def test_planning_route_binds_exact_generated_model(self):
        proposal = PlanProposal(spec=self.plan, explanation="Generated a typed model", attempts=1)
        with patch("app.main.planner.interpret_plan", AsyncMock(return_value=proposal)):
            response = await self.client.post("/api/plan/interpret", json={"prompt": "Use the original paper case", "evidence_id": self.record["evidence_id"]})
        self.assertEqual(response.status_code, 200, response.text)
        planned = provenance.get_planning(self.store, response.json()["planning_id"])
        self.assertEqual(planned["evidence_id"], self.record["evidence_id"])
        self.assertEqual(planned["status"], "source_bound_unverified")

    async def test_exact_binding_and_case_are_required_before_linked_targets(self):
        response = await self.client.post("/api/revisions", json=self.payload())
        self.assertEqual(response.status_code, 201, response.text)
        revision = response.json()
        self.assertEqual(revision["planning_id"], self.binding["planning_id"])
        reread = await self.client.get("/api/revisions/" + revision["id"])
        self.assertEqual(reread.json()["evidence_id"], self.record["evidence_id"])
        baseline = copy.deepcopy(self.record["evidence"]["targets"][0])
        baseline["observable"]["circuit_id"] = "wire"
        request = {"revision_id": revision["id"], "evidence_id": self.record["evidence_id"], "targets": [baseline]}
        accepted = await self.client.post("/api/targets", json=request)
        self.assertEqual(accepted.status_code, 201, accepted.text)
        request["case_id"] = "displaced"
        request["targets"][0].update(id="displaced", case_id="displaced")
        rejected = await self.client.post("/api/targets", json=request)
        self.assertEqual(rejected.status_code, 422, rejected.text)
        self.assertIn("operating case", rejected.text)

    async def test_edited_model_cannot_reuse_planning_id_or_forge_source_reference(self):
        changed = self.plan.model_dump(mode="json")
        changed["circuits"][0]["current_a"] = 2
        for override in ({"spec": changed}, {"case_id": "displaced"}, {"planning_id": "a" * 64}):
            response = await self.client.post("/api/revisions", json=self.payload(**override))
            self.assertEqual(response.status_code, 422, response.text)
        forged = self.app.state.history.create(GeometryPlan.model_validate(changed), branch="forged", evidence=[{
            "source": "artifact", "reference": provenance.PREFIX + self.binding["planning_id"], "sha256": self.binding["planning_id"]}])
        with self.assertRaisesRegex(ValueError, "operating case"):
            provenance.require_paper_binding(self.store, self.app.state.history, forged, self.record["evidence_id"], "baseline")

    async def test_same_geometry_for_new_source_case_gets_new_revision(self):
        first = (await self.client.post("/api/revisions", json=self.payload())).json()
        other = provenance.put_planning(self.store, self.plan, self.record["evidence_id"], "displaced", "Different case with equal geometry")
        response = await self.client.post("/api/revisions", json=self.payload(planning_id=other["planning_id"], case_id="displaced", parent_id=first["id"], branch=first["branch"]))
        self.assertEqual(response.status_code, 201, response.text)
        self.assertNotEqual(response.json()["id"], first["id"])
        self.assertEqual(response.json()["case_id"], "displaced")

    async def test_invalid_reserved_evidence_does_not_create_or_break_a_revision(self):
        bad = {"source": "artifact", "reference": provenance.PREFIX + "a" * 64, "sha256": "a" * 64}
        response = await self.client.post("/api/revisions", json={"spec": self.plan.model_dump(mode="json"), "branch": "invalid-reference", "evidence": [bad]})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIsNone(self.app.state.history.head("invalid-reference"))


if __name__ == "__main__":
    unittest.main()
