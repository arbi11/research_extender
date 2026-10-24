"""Authenticated, immutable paper intake and pre-solve numerical contracts."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import Depends, FastAPI, Header, HTTPException
import httpx

from app.history import ModelRevisionStore
from app.jobs import JobStore
from app.main import Settings
from app.plan import wire_example
from app import papers, replication, provenance
from tests.test_papers import document, evidence, text_pdf
from tests.test_targets import target


def wire_evidence(doc, legacy=None):
    """Fill the real strict wire contract with existing immutable source spans."""
    packet = copy.deepcopy(legacy if legacy is not None else evidence(doc))
    spans = papers.source_spans(doc)
    for section in ("parameters", "targets"):
        for claim in packet.get(section, []):
            citations = claim.pop("citations", [])
            pages = {citation["page"] for citation in citations}
            claim["source_refs"] = [span.span_id for span in spans if span.page in pages]
    return papers.SourcePaperEvidence.model_validate(packet).model_dump(mode="json")


def staged_reply(packet, request):
    """Split existing synthetic evidence across the actual small wire contracts."""
    payload = json.loads(request.content) if isinstance(request, httpx.Request) else request
    name = payload["response_format"]["json_schema"]["name"]
    source = {"source_id": packet["source_id"]}
    for field in ("questions", "limitations"):
        if field in packet:
            # Full-evidence fixtures keep their messages in one checkpoint.
            source[field] = packet[field] if name == "paper_cases" else []
    if name == "paper_cases":
        ids = list(dict.fromkeys(claim.get("case_id", "baseline")
            for field in ("parameters", "targets") for claim in packet.get(field, []))) or ["baseline"]
        selected = next((case for case in ids if case != "baseline"), "baseline")
        return {**source, "title": packet["title"], "summary": packet["summary"],
                "case_ids": ids, "selected_case_id": selected}
    if name == "paper_parameters":
        values = []
        for parameter in packet.get("parameters", []):
            entry = copy.deepcopy(parameter)
            value = entry.pop("value")
            entry["value_number"] = value if not isinstance(value, str) else None
            entry["value_text"] = value if isinstance(value, str) else None
            values.append(entry)
        return {**source, "parameters": values}
    if name == "paper_references":
        values = []
        for target in packet.get("targets", []):
            entry = copy.deepcopy(target)
            observable = entry.pop("observable")
            entry.update(quantity=observable["quantity"], circuit_id=observable.get("circuit_id"),
                         probe_index=observable.get("probe_index"))
            entry.pop("tolerance")
            values.append(entry)
        return {**source, "targets": values}
    raise AssertionError("Unexpected strict stage " + name)


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
        self.settings = Settings(data_dir=self.root, token="assistant-secret-token",
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

    async def extract(self, uploaded, **body):
        return await self.client.post(f'/api/papers/{uploaded["paper_id"]}/extract', headers=self.headers,
                                      json={"hierarchical": False, **body})

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

    async def test_three_strict_provider_stages_have_no_artificial_cap_and_retain_private_audit(self):
        uploaded = await self.upload()
        doc = papers.PaperDocument.model_validate(uploaded["document"])
        real_client = httpx.AsyncClient
        calls = []
        def handler(request):
            calls.append(request)
            payload = json.loads(request.content)
            self.assertNotIn("max_tokens", payload)
            self.assertNotIn("max_completion_tokens", payload)
            self.assertNotIn("temperature", payload)
            self.assertEqual(payload["response_format"]["type"], "json_schema")
            self.assertTrue(payload["response_format"]["json_schema"]["strict"])
            self.assertEqual(payload["provider"], {"require_parameters": True})
            self.assertEqual(payload["model"], "test-model")
            packet = wire_evidence(doc)
            packet["targets"][0]["tolerance"]["relative"] = .99
            return httpx.Response(200, json={"id": "generation-test", "model": "returned-test-model", "provider": "provider-test",
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(staged_reply(packet, request))}}],
                "usage": {"cost": .002, "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                          "completion_tokens_details": {"reasoning_tokens": 12, "secret": self.settings.ai_key},
                          "secret": self.settings.ai_key}, "provider_secret": self.settings.ai_key})
        with patch("app.paper_provider.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)):
            response = await self.extract(uploaded, instructions="Reproduce the original case")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(calls), 3)
        self.assertAlmostEqual(response.json()["usage"]["cost"], .006)
        self.assertEqual(response.json()["evidence"]["targets"][0]["tolerance"]["relative"], .02)
        self.assertNotIn(self.settings.ai_key, response.text)
        private = self.store.get_extraction_audit(response.json()["audit_id"])
        self.assertEqual(private["usage"]["completion_tokens_details"], {"reasoning_tokens": 36})
        self.assertEqual(private["returned_model"], "returned-test-model")
        self.assertEqual(private["provider"], "provider-test")
        self.assertEqual(private["provider_request_id"], "generation-test")
        self.assertEqual(private["provider_response"]["provider_secret"], "[REDACTED]")
        self.assertEqual(private["request"]["model"], "test-model")
        self.assertNotIn(self.settings.ai_key, json.dumps(private))
        self.assertNotIn("provider_response", response.json())
        def leaked(request):
            packet = wire_evidence(doc)
            packet["summary"] = self.settings.ai_key
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(staged_reply(packet, request))}}]})
        with patch("app.paper_provider.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(leaked), **kwargs)):
            rejected = await self.extract(uploaded)
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
            packet = wire_evidence(doc)
            packet["targets"][0]["value"] = 1.9
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(staged_reply(packet, request))}}],
                "usage": {"prompt_tokens": 33, "completion_tokens": 10, "total_tokens": 43, "cost": .001}})
        with patch("app.paper_provider.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)):
            rejected = await self.extract(uploaded)
        self.assertEqual(rejected.status_code, 200)
        self.assertEqual(len(calls), 3)
        self.assertIsNone(self.jobs.next())
        detail = rejected.json()
        self.assertEqual(detail["extraction_status"], "partial")
        self.assertEqual(detail["rejected_claim_count"], 1)
        self.assertFalse(detail["evidence"]["targets"])
        self.assertEqual(detail["evidence"]["parameters"][0]["id"], "current")
        self.assertAlmostEqual(detail["usage"]["cost"], .003)
        self.assertEqual(detail["validation_summary"][0]["path"], "targets.0.value")
        self.assertEqual(detail["validation_summary"][0]["code"], "numeric_value_or_unit_unverified")
        private = self.store.get_extraction_audit(detail["audit_id"])
        self.assertEqual(private["candidate"]["targets"][0]["value"], 1.9)
        self.assertEqual(private["usage"]["total_tokens"], 129)
        self.assertNotIn("candidate", detail)

    async def test_schema_failure_audit_has_field_path_and_no_provider_headers(self):
        uploaded = await self.upload()
        doc = papers.PaperDocument.model_validate(uploaded["document"])
        real_client = httpx.AsyncClient
        def handler(request):
            packet = wire_evidence(doc)
            packet["parameters"][0]["value"] = None
            return httpx.Response(200, headers={"provider-private-header": self.settings.ai_key},
                json={"choices": [{"message": {"content": json.dumps(staged_reply(packet, request))}}], "usage": {"cost": .002}})
        with patch("app.paper_provider.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)):
            response = await self.extract(uploaded)
        self.assertEqual(response.status_code, 200)
        detail = response.json()
        self.assertEqual(detail["extraction_status"], "partial")
        self.assertFalse(detail["evidence"]["parameters"])
        self.assertEqual(detail["validation_summary"][0]["path"], "parameters.0.value")
        private = self.store.get_extraction_audit(detail["audit_id"])
        self.assertAlmostEqual(private["usage"]["cost"], .006)
        self.assertNotIn("provider-private-header", json.dumps(private))
        self.assertNotIn(self.settings.ai_key, json.dumps(private))

    async def test_partial_boundary_claim_never_enters_planning_context(self):
        uploaded = await self.upload()
        doc = papers.PaperDocument.model_validate(uploaded["document"])
        real_client = httpx.AsyncClient
        def handler(request):
            packet = wire_evidence(doc)
            bad_boundary = copy.deepcopy(packet["parameters"][0])
            bad_boundary.update(id="bad_boundary", description="Invalid invented boundary unit", value=0, unit="Wb")
            packet["parameters"].append(bad_boundary)
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(staged_reply(packet, request))}}], "usage": {"cost": .002}})
        with patch("app.paper_provider.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)):
            response = await self.extract(uploaded)
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
                packet = wire_evidence(doc)
                if mode == "all_invalid":
                    packet["parameters"][0]["value"] = None
                    packet["targets"][0]["value"] = 999
                else:
                    packet["source_id"] = "paper_" + "a" * 24
                return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(staged_reply(packet, request))}}]})
            with patch("app.paper_provider.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)):
                response = await self.extract(uploaded)
            with self.subTest(mode=mode):
                self.assertEqual(response.status_code, 502, response.text)
                self.assertIn("audit_id", response.json()["detail"])
                private = self.store.get_extraction_audit(response.json()["detail"]["audit_id"])
                if mode == "all_invalid":
                    self.assertEqual({issue["claim_id"] for issue in private["issues"]}, {"current", "inductance"})
                    self.assertEqual(private["candidate"]["targets"][0]["value"], 999)

    async def test_missing_wire_fields_are_not_filled_from_local_defaults(self):
        uploaded = await self.upload()
        doc = papers.PaperDocument.model_validate(uploaded["document"])
        real_client, calls = httpx.AsyncClient, []
        def handler(request):
            calls.append(request)
            packet = wire_evidence(doc)
            packet.pop("questions")
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(staged_reply(packet, request))}}]})
        with patch("app.paper_provider.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)):
            response = await self.extract(uploaded)
        self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(len(calls), 1)
        private = self.store.get_extraction_audit(response.json()["detail"]["audit_id"])
        self.assertTrue(any(issue["code"] == "wire_required" and issue["path"] == "questions" for issue in private["issues"]))
        self.assertNotIn("questions", private["candidate"])
        self.assertIsNone(self.jobs.next())

    async def test_source_span_values_support_scientific_notation_and_word_units(self):
        source = "All lengths are in centimeters. Radius 15. Force 5.30 x 10^-2 newtons."
        uploaded_response = await self.client.post("/api/papers", content=text_pdf(source),
            headers={**self.headers, "Content-Type": "application/pdf", "X-Filename": "notation.pdf"})
        self.assertEqual(uploaded_response.status_code, 201, uploaded_response.text)
        uploaded = uploaded_response.json()
        doc = papers.PaperDocument.model_validate(uploaded["document"])
        span = papers.source_spans(doc)[0]
        packet = papers.SourcePaperEvidence(source_id=doc.source_id, title="Synthetic notation", summary="Test source",
            parameters=(papers.SourceEvidenceParameter(id="radius", description="Radius", value=15,
                unit="cm", evidence_kind="explicit", value_refs=(span.span_id,), unit_refs=(span.span_id,)),),
            targets=(papers.SourceNumericalTarget(id="force", description="Published numerical reference", value=.053,
                unit="N", observable={"quantity": "force_magnitude"}, tolerance={"relative": .02},
                reference_kind="reported_numerical", conditions="Original source operating condition",
                value_refs=(span.span_id,), unit_refs=(span.span_id,)),)).model_dump(mode="json")
        real_client, calls = httpx.AsyncClient, []
        def handler(request):
            calls.append(request)
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(staged_reply(packet, request))}}]})
        with patch("app.paper_provider.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)):
            response = await self.extract(uploaded)
            packet["targets"][0]["value"] = .53
            wrong = await self.extract(uploaded)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["extraction_status"], "complete")
        self.assertEqual(response.json()["evidence"]["parameters"][0]["value"], 15)
        self.assertEqual(response.json()["evidence"]["targets"][0]["value"], .053)
        self.assertEqual(response.json()["evidence"]["targets"][0]["citations"][0]["quote"], span.text)
        self.assertEqual(wrong.status_code, 200, wrong.text)
        self.assertEqual(wrong.json()["extraction_status"], "partial")
        self.assertFalse(wrong.json()["evidence"]["targets"])
        self.assertEqual(len(calls), 6)  # Two explicit API requests, three stages each; no implicit repair.

    async def test_hierarchical_discovery_and_extraction_use_four_strict_provider_responses(self):
        from app.assembly_provider import MachineInventory
        uploaded = await self.upload()
        doc = papers.PaperDocument.model_validate(uploaded["document"])
        span = papers.source_spans(doc)[0]
        inventory = MachineInventory(id="study", title="Synthetic two-stage study", input_mode="paper",
            family="air_core_coils", physics="axisymmetric_dc_magnetics",
            required_physics=["axisymmetric_dc_magnetics"], components=[{
                "id": "coil", "name": "Coil", "role": "winding", "family": "air_core_coil",
                "mode": "custom", "scope": "region", "source_refs": [span.span_id], "intent": "Preserve the source winding"}],
            cases=[{"id": "thin", "label": "Thin-wall case", "source_refs": [span.span_id],
                    "required_physics": ["zero_thickness_winding"], "geometry_requirements": ["zero radial width"]},
                   {"id": "finite", "label": "Finite-volume case", "source_refs": [span.span_id],
                    "required_physics": ["axisymmetric_dc_magnetics"], "geometry_requirements": ["positive winding area"]}],
            selected_case_id="finite", explanation="Select the eligible source case.").model_dump(mode="json", exclude={"identity_map"})
        extraction = wire_evidence(doc)
        for section in ("parameters", "targets"):
            for claim in extraction[section]:
                claim["case_id"] = "finite"
        real_client, requests = httpx.AsyncClient, []
        def handler(request):
            payload = json.loads(request.content)
            requests.append(payload)
            self.assertTrue(payload["response_format"]["json_schema"]["strict"])
            self.assertTrue(payload["provider"]["require_parameters"])
            self.assertNotIn("max_tokens", payload)
            self.assertNotIn("temperature", payload)
            name = payload["response_format"]["json_schema"]["name"]
            if name == "MachineInventory":
                supplied = json.loads(payload["messages"][-1]["content"])["evidence_context"]["paper_source_inventory"]
                self.assertEqual(supplied["source_id"], doc.source_id)
                self.assertEqual(supplied["pages"][0]["spans"][0]["span_id"], span.span_id)
                candidate = inventory
            else:
                self.assertIn(name, {"paper_cases", "paper_parameters", "paper_references"})
                self.assertTrue(any('"selected_case_id":"finite"' in message["content"] for message in payload["messages"]))
                candidate = staged_reply(extraction, payload)
            return httpx.Response(200, json={"id": name + "-generation", "model": "test-model",
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(candidate)}}],
                "usage": {"cost": .001, "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
        with patch("app.paper_provider.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs)):
            response = await self.extract(uploaded, hierarchical=True)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(requests), 4)
        result = response.json()
        self.assertEqual(result["assembly_inventory"]["selected_case_id"], "finite")
        self.assertEqual({claim["case_id"] for claim in result["evidence"]["parameters"]}, {"finite"})
        private = self.store.get_extraction_audit(result["audit_id"])
        self.assertEqual(private["assembly_discovery"]["inventory"]["selected_case_id"], "finite")
        self.assertIsNone(self.jobs.next())

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
