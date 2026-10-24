"""Authenticated, immutable paper intake and pre-solve numerical contracts."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import Depends, FastAPI, Header, HTTPException
import httpx

from app.history import ModelRevisionStore
from app.jobs import JobStore
from app.plan import wire_example
from app import papers, replication, provenance
from tests.test_papers import document, evidence, text_pdf
from tests.test_targets import target


class _Budget:
    def __init__(self):
        self.calls = []
    def use(self, kind, limit):
        self.calls.append((kind, limit))


class ReplicationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = replication.ReplicationStore(self.root)
    def tearDown(self):
        self.temporary.cleanup()

    def test_contract_is_content_addressed_and_detects_disk_tampering(self):
        plan = wire_example()
        record = self.store.put_contract("a" * 64, "baseline", [target()], plan)
        same = self.store.put_contract("a" * 64, "baseline", [target()], plan)
        self.assertEqual(record, same)
        loaded = self.store.get_contract(record["contract_id"])
        self.assertEqual(loaded, record)
        path = self.store.root / "contracts" / (record["contract_id"] + ".json")
        content = json.loads(path.read_text())
        content["contract"]["targets"][0]["value"] = 100
        path.write_text(json.dumps(content))
        with self.assertRaisesRegex(replication.ReplicationIntegrityError, "content hash"):
            self.store.get_contract(record["contract_id"])
        self.assertIsNone(self.store.get_contract("../../escape"))

    def test_atomic_storage_never_overwrites_records(self):
        path = self.root / "immutable"
        replication._write_once(path, b"original")
        replication._write_once(path, b"original")
        with self.assertRaises(replication.ReplicationIntegrityError):
            replication._write_once(path, b"replacement")
        self.assertEqual(path.read_bytes(), b"original")
        link = self.root / "symlink"
        link.symlink_to(path)
        with self.assertRaises(replication.ReplicationIntegrityError):
            replication._write_once(link, b"original")

    def test_usage_allows_numeric_known_fields_only(self):
        parsed = replication._usage({"usage": {"prompt_tokens": 10, "completion_tokens": 2,
            "total_tokens": 12, "cost": .001, "secret": "credential", "output_tokens": True,
            "input_tokens": "10", "other": 30}})
        self.assertEqual(parsed, {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "cost": .001})

    def test_extraction_tolerance_is_fixed_and_zero_target_requires_user_choice(self):
        packet = evidence(document())
        packet["targets"][0]["tolerance"] = {"relative": .99}
        result = replication._extraction_tolerances(packet)
        self.assertEqual(result["targets"][0]["tolerance"], {"absolute": 0, "relative": .02})
        self.assertIn("not a paper accuracy claim", result["targets"][0]["description"])
        packet["targets"][0]["value"] = 0
        zero = replication._extraction_tolerances(packet)
        self.assertFalse(zero["targets"])
        self.assertIn("absolute tolerance", zero["questions"][-1])

    def test_malformed_target_is_scoped_to_claim_without_losing_valid_parameters(self):
        doc = document()
        for bad_target in (None, 2, "invalid", {**evidence(doc)["targets"][0], "value": float("nan")}):
            packet = evidence(doc)
            packet["targets"] = [bad_target]
            prepared = replication._extraction_tolerances(packet)
            result, issues, count = papers.validate_partial_evidence(doc, prepared)
            with self.subTest(target=bad_target):
                self.assertEqual(count, 1)
                self.assertTrue(issues)
                self.assertEqual(result.parameters[0].id, "current")
                self.assertFalse(result.targets)

    def test_exact_unit_notation_aliases_do_not_change_values_or_weaken_citations(self):
        doc = document("Turns 1000. Current density 5 A/mm 2. Relative permeability 10000.")
        citation = {"source_id": doc.source_id, "page": 1, "quote": doc.pages[0].text}
        packet = {"source_id": doc.source_id, "title": "Unit aliases", "summary": "Synthetic source",
                  "parameters": [{"id": "turns", "description": "Turns", "value": 1000,
                    "unit": "dimensionless", "evidence_kind": "explicit", "citations": [citation]},
                    {"id": "current_density", "description": "Uniform current density", "value": 5,
                    "unit": "A/mm^2", "evidence_kind": "explicit", "citations": [citation]}]}
        result = papers.validate_evidence(doc, packet)
        self.assertEqual(result.parameters[0].unit, "1")
        self.assertEqual(result.parameters[1].unit, "A/mm2")
        self.assertEqual(result.parameters[1].value, 5)
        self.assertEqual(result.parameters[1].citations[0].quote, doc.pages[0].text)
        packet["parameters"][1]["value"] = 50
        with self.assertRaises(papers.PaperEvidenceError):
            papers.validate_evidence(doc, packet)

    def test_reference_category_conflict_is_rejected_without_changing_target(self):
        doc = document()
        packet = evidence(doc)
        packet["targets"][0].update(reference_kind="experimental", conditions="h=0 cm, FE simulation result")
        with self.assertRaises(papers.PaperEvidenceError) as caught:
            papers.validate_evidence(doc, packet)
        self.assertEqual(caught.exception.issue["code"], "reference_kind_conflict")
        self.assertEqual(packet["targets"][0]["value"], 1.73)
        self.assertEqual(packet["targets"][0]["reference_kind"], "experimental")

    def test_boundary_condition_is_text_without_an_invented_numeric_unit(self):
        doc = document("Homogeneous Dirichlet boundary conditions impose A=0.")
        citation = {"source_id": doc.source_id, "page": 1, "quote": doc.pages[0].text}
        packet = {"source_id": doc.source_id, "title": "Boundary source", "summary": "Synthetic boundary evidence",
                  "parameters": [{"id": "boundary", "description": "Boundary condition", "value": 0,
                    "unit": "Wb", "evidence_kind": "explicit", "citations": [citation]}]}
        with self.assertRaises(papers.PaperEvidenceError) as rejected:
            papers.validate_evidence(doc, packet)
        self.assertEqual(rejected.exception.issue["code"], "numeric_value_or_unit_unverified")
        packet["parameters"][0].update(value="A=0", unit="text")
        self.assertEqual(papers.validate_evidence(doc, packet).parameters[0].value, "A=0")


class ReplicationApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.settings = SimpleNamespace(data_dir=self.root, token="assistant-secret-token",
            ai_key="openrouter-test-secret-key", ai_model="test-model", ai_base_url="https://provider.test/api/v1")
        self.history = ModelRevisionStore(self.root)
        self.jobs = JobStore(self.root)
        self.budget = _Budget()
        self.app = FastAPI()
        async def authenticate(authorization: str = Header(default="")):
            if authorization != "Bearer assistant-secret-token":
                raise HTTPException(401)
        self.store = replication.register_routes(self.app, [Depends(authenticate)], self.settings,
                                                  self.budget, self.history, self.jobs)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")
        self.headers = {"Authorization": "Bearer assistant-secret-token"}
    async def asyncTearDown(self):
        await self.client.aclose()
        self.temporary.cleanup()

    async def upload(self):
        if importlib.util.find_spec("pypdf") is None:
            self.skipTest("pypdf runtime dependency is not installed")
        response = await self.client.post("/api/papers", content=text_pdf(),
            headers={**self.headers, "Content-Type": "application/pdf", "X-Filename": "case.pdf"})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    async def test_routes_are_authenticated_and_do_not_accept_reported_target_success(self):
        for path in ("/api/papers", "/api/papers/" + "a"*64 + "/extract", "/api/targets"):
            response = await self.client.post(path, json={})
            self.assertEqual(response.status_code, 401)
        response = await self.client.post("/api/targets/check", headers=self.headers, json={"passed": True})
        self.assertEqual(response.status_code, 405)
        wrong = await self.client.post("/api/papers", headers=self.headers, json={"pdf": "text"})
        self.assertEqual(wrong.status_code, 415)
        missing = await self.client.get("/api/papers/" + "a"*64, headers=self.headers)
        self.assertEqual(missing.status_code, 404)

    async def test_upload_manual_citations_and_selected_case_context(self):
        uploaded = await self.upload()
        identity = uploaded["paper_id"]
        doc = papers.PaperDocument.model_validate(uploaded["document"])
        packet = evidence(doc)
        other = copy.deepcopy(packet["parameters"][0])
        other.update(id="displaced_current", case_id="displaced")
        packet["parameters"].append(other)
        other_target = copy.deepcopy(packet["targets"][0])
        other_target.update(id="displaced_inductance", case_id="displaced")
        packet["targets"].append(other_target)
        response = await self.client.post(f"/api/papers/{identity}/evidence", headers=self.headers, json=packet)
        self.assertEqual(response.status_code, 201, response.text)
        record = response.json()
        fetched = await self.client.get("/api/paper-evidence/" + record["evidence_id"], headers=self.headers)
        self.assertEqual(fetched.json(), record)
        base = self.store.planning_context(record["evidence_id"], "baseline", "Use the original case")
        self.assertEqual(len(base["parameters"]), 1)
        self.assertEqual(len(base["published_targets"]), 1)
        displaced = self.store.planning_context(record["evidence_id"], "displaced", "Move the core")
        self.assertEqual(len(displaced["parameters"]), 2)
        self.assertEqual([target["case_id"] for target in displaced["published_targets"]], ["displaced"])
        self.assertIn("never rewrite", displaced["source_policy"])
        with self.assertRaisesRegex(ValueError, "Select an extracted"):
            self.store.planning_context(record["evidence_id"], "unknown")
        invalid = copy.deepcopy(packet)
        invalid["parameters"][0]["citations"][0]["quote"] = "Invented current 1 A"
        rejected = await self.client.post(f"/api/papers/{identity}/evidence", headers=self.headers, json=invalid)
        self.assertEqual(rejected.status_code, 422)
        # Original upload remains readable after a rejected extraction/edit.
        fetched = await self.client.get(f"/api/papers/{identity}", headers=self.headers)
        self.assertEqual(fetched.json(), uploaded)

    async def test_target_contract_requires_existing_exact_revision_and_is_immutable(self):
        missing = await self.client.post("/api/targets", headers=self.headers,
            json={"revision_id": "a" * 64, "targets": [target().model_dump(mode="json")]})
        self.assertEqual(missing.status_code, 404)
        revision = self.history.create(wire_example(), branch="reference")
        response = await self.client.post("/api/targets", headers=self.headers,
            json={"revision_id": revision["id"], "targets": [target().model_dump(mode="json")]})
        self.assertEqual(response.status_code, 201, response.text)
        record = response.json()
        fetched = await self.client.get("/api/targets/" + record["contract_id"], headers=self.headers)
        self.assertEqual(fetched.json(), record)
        altered = await self.client.post("/api/targets", headers=self.headers,
            json={"revision_id": revision["id"], "targets": [target(value=.0001).model_dump(mode="json")]})
        self.assertEqual(altered.status_code, 201)
        self.assertNotEqual(altered.json()["contract_id"], record["contract_id"])
        self.assertEqual(self.store.get_contract(record["contract_id"]), record)

    async def test_paper_target_provenance_cannot_change_but_binding_and_tolerance_can(self):
        uploaded = await self.upload()
        doc = papers.PaperDocument.model_validate(uploaded["document"])
        record = self.store.put_evidence(uploaded["paper_id"], evidence(doc))
        planning = provenance.put_planning(self.store, wire_example(), record["evidence_id"], "baseline", "Original case")
        binding = provenance.binding_evidence(self.store, wire_example(), planning["planning_id"], record["evidence_id"], "baseline")
        revision = self.history.create(wire_example(), branch="paper-reference", evidence=[binding])
        numerical = copy.deepcopy(record["evidence"]["targets"][0])
        numerical["observable"]["circuit_id"] = "wire"
        numerical["tolerance"] = {"relative": .03}
        body = {"revision_id": revision["id"], "evidence_id": record["evidence_id"], "targets": [numerical]}
        accepted = await self.client.post("/api/targets", headers=self.headers, json=body)
        self.assertEqual(accepted.status_code, 201, accepted.text)
        self.assertEqual(accepted.json()["evidence_id"], record["evidence_id"])
        self.assertEqual(accepted.json()["contract"]["targets"][0]["tolerance"]["relative"], .03)
        for field, value in (("value", 1.74), ("unit", "H"), ("conditions", "Changed current"),
                             ("reference_kind", "analytical"), ("case_id", "another_case")):
            changed = copy.deepcopy(body)
            changed["targets"][0][field] = value
            rejected = await self.client.post("/api/targets", headers=self.headers, json=changed)
            with self.subTest(field=field):
                self.assertEqual(rejected.status_code, 422, rejected.text)
        changed = copy.deepcopy(body)
        changed["targets"][0]["citations"][0]["quote"] = "Invented paper reference"
        rejected = await self.client.post("/api/targets", headers=self.headers, json=changed)
        self.assertEqual(rejected.status_code, 422)
        self.assertEqual(self.store.get_evidence(record["evidence_id"]), record)

    async def test_single_bounded_provider_call_usage_and_credential_protection(self):
        uploaded = await self.upload()
        doc = papers.PaperDocument.model_validate(uploaded["document"])
        real_client = httpx.AsyncClient
        calls = []
        def handler(request):
            calls.append(request)
            payload = json.loads(request.content)
            self.assertEqual(payload["max_tokens"], 6000)
            self.assertEqual(payload["model"], "test-model")
            packet = evidence(doc)
            packet["targets"][0]["tolerance"] = {"relative": .99}
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(packet)}}],
                "usage": {"cost": .002, "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                          "secret": self.settings.ai_key}, "provider_secret": self.settings.ai_key})
        with patch("app.replication.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)):
            response = await self.client.post(f'/api/papers/{uploaded["paper_id"]}/extract', headers=self.headers,
                                              json={"instructions": "Reproduce the original case"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(response.json()["usage"]["cost"], .002)
        self.assertEqual(response.json()["evidence"]["targets"][0]["tolerance"]["relative"], .02)
        self.assertNotIn(self.settings.ai_key, response.text)
        def leaked(request):
            packet = evidence(doc)
            packet["summary"] = self.settings.ai_key
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(packet)}}]})
        with patch("app.replication.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(leaked), **kwargs)):
            rejected = await self.client.post(f'/api/papers/{uploaded["paper_id"]}/extract', headers=self.headers, json={})
        self.assertEqual(rejected.status_code, 502)
        self.assertNotIn(self.settings.ai_key, rejected.text)
        audit_id = rejected.json()["detail"]["audit_id"]
        private = self.store.get_extraction_audit(audit_id)
        self.assertNotIn(self.settings.ai_key, json.dumps(private))
        self.assertEqual(private["requested_model"], "test-model")
        self.assertIn("[REDACTED]", json.dumps(private["candidate"]))

    async def test_invalid_provider_claim_is_excluded_without_retry_or_geometry(self):
        uploaded = await self.upload()
        doc = papers.PaperDocument.model_validate(uploaded["document"])
        real_client, calls = httpx.AsyncClient, []
        def handler(request):
            calls.append(request)
            packet = evidence(doc)
            packet["targets"][0]["value"] = 1.9
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(packet)}}],
                "usage": {"prompt_tokens": 33, "completion_tokens": 10, "total_tokens": 43, "cost": .001}})
        with patch("app.replication.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)):
            rejected = await self.client.post(f'/api/papers/{uploaded["paper_id"]}/extract', headers=self.headers, json={})
        self.assertEqual(rejected.status_code, 200)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(self.jobs.next())
        detail = rejected.json()
        self.assertEqual(detail["extraction_status"], "partial")
        self.assertEqual(detail["rejected_claim_count"], 1)
        self.assertFalse(detail["evidence"]["targets"])
        self.assertEqual(detail["evidence"]["parameters"][0]["id"], "current")
        self.assertEqual(detail["usage"]["cost"], .001)
        self.assertEqual(detail["validation_summary"][0]["path"], "targets.0.value")
        self.assertEqual(detail["validation_summary"][0]["code"], "numeric_value_or_unit_unverified")
        private = self.store.get_extraction_audit(detail["audit_id"])
        self.assertEqual(private["candidate"]["targets"][0]["value"], 1.9)
        self.assertEqual(private["usage"]["total_tokens"], 43)
        self.assertNotIn("candidate", detail)

    async def test_schema_failure_audit_has_field_path_and_no_provider_headers(self):
        uploaded = await self.upload()
        doc = papers.PaperDocument.model_validate(uploaded["document"])
        real_client = httpx.AsyncClient
        def handler(request):
            packet = evidence(doc)
            packet["parameters"][0]["value"] = None
            return httpx.Response(200, headers={"provider-private-header": self.settings.ai_key},
                json={"choices": [{"message": {"content": json.dumps(packet)}}], "usage": {"cost": .002}})
        with patch("app.replication.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)):
            response = await self.client.post(f'/api/papers/{uploaded["paper_id"]}/extract', headers=self.headers, json={})
        self.assertEqual(response.status_code, 200)
        detail = response.json()
        self.assertEqual(detail["extraction_status"], "partial")
        self.assertFalse(detail["evidence"]["parameters"])
        self.assertEqual(detail["validation_summary"][0]["path"], "parameters.0.value")
        private = self.store.get_extraction_audit(detail["audit_id"])
        self.assertEqual(private["usage"]["cost"], .002)
        self.assertNotIn("provider-private-header", json.dumps(private))
        self.assertNotIn(self.settings.ai_key, json.dumps(private))

    async def test_partial_boundary_claim_never_enters_planning_context(self):
        uploaded = await self.upload()
        doc = papers.PaperDocument.model_validate(uploaded["document"])
        real_client = httpx.AsyncClient
        def handler(request):
            packet = evidence(doc)
            packet["parameters"].append({"id": "bad_boundary", "description": "Invalid invented boundary unit",
                "value": 0, "unit": "Wb", "evidence_kind": "explicit", "citations": packet["parameters"][0]["citations"]})
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(packet)}}], "usage": {"cost": .002}})
        with patch("app.replication.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)):
            response = await self.client.post(f'/api/papers/{uploaded["paper_id"]}/extract', headers=self.headers, json={})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["extraction_status"], "partial")
        self.assertEqual(result["rejected_claim_count"], 1)
        self.assertEqual([p["id"] for p in result["evidence"]["parameters"]], ["current"])
        self.assertTrue(any("bad_boundary" in q for q in result["evidence"]["questions"]))
        context = self.store.planning_context(result["evidence_id"])
        self.assertEqual([p["id"] for p in context["parameters"]], ["current"])
        saved = self.store.get_evidence(result["evidence_id"])
        self.assertEqual(saved["extraction_status"], "partial")
        self.assertEqual(self.store.get_extraction_audit(result["audit_id"])["status"], "partial")

    async def test_all_invalid_claims_and_wrong_top_level_source_remain_failures(self):
        uploaded = await self.upload()
        doc = papers.PaperDocument.model_validate(uploaded["document"])
        real_client = httpx.AsyncClient
        for mode in ("all_invalid", "wrong_source"):
            def handler(request):
                packet = evidence(doc)
                if mode == "all_invalid":
                    packet["parameters"][0]["value"] = None
                    packet["targets"][0]["value"] = 999
                else:
                    packet["source_id"] = "paper_" + "a" * 24
                return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(packet)}}]})
            with patch("app.replication.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)):
                response = await self.client.post(f'/api/papers/{uploaded["paper_id"]}/extract', headers=self.headers, json={})
            with self.subTest(mode=mode):
                self.assertEqual(response.status_code, 502, response.text)
                self.assertIn("audit_id", response.json()["detail"])

    async def test_unbound_existing_model_cannot_inherit_paper_target(self):
        uploaded = await self.upload()
        doc = papers.PaperDocument.model_validate(uploaded["document"])
        record = self.store.put_evidence(uploaded["paper_id"], evidence(doc))
        revision = self.history.create(wire_example(), branch="unrelated-model")
        numerical = copy.deepcopy(record["evidence"]["targets"][0])
        numerical["observable"]["circuit_id"] = "wire"
        response = await self.client.post("/api/targets", headers=self.headers, json={
            "revision_id": revision["id"], "evidence_id": record["evidence_id"], "targets": [numerical]})
        self.assertEqual(response.status_code, 422)
        self.assertIn("not generated", response.text)


if __name__ == "__main__":
    unittest.main()
