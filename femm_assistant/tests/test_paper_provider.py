"""Offline contracts and accounting; these mocks do not prove live grammar acceptance."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
import httpx

from app import paper_provider, papers, replication, structured
from app.assembly_provider import AssemblyProviderError, MachineInventory
from tests.test_papers import document
from tests.test_replication_api import staged_reply, wire_evidence


class PaperProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.settings = SimpleNamespace(ai_key="private-test-provider-key", ai_model="test-model",
            ai_base_url="https://provider.test/api/v1", ai_timeout=None, token="private-assistant-token",
            hierarchical_ai=False, data_dir=Path(self.temporary.name), decision_model="test-decision",
            decision_base_url="https://decision.test")
        self.doc = document()
        self.real_client = httpx.AsyncClient

    def tearDown(self):
        self.temporary.cleanup()

    def transport(self, handler):
        return patch("app.paper_provider.httpx.AsyncClient", side_effect=lambda **kwargs:
            self.real_client(transport=httpx.MockTransport(handler), **kwargs))

    def response(self, candidate, *, cost=.001):
        return httpx.Response(200, json={"id": "test-generation", "model": "test-returned",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(candidate)}}],
            "usage": {"cost": cost, "prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
                      "completion_tokens_details": {"reasoning_tokens": 2}}})

    def private_audit(self, error):
        return replication.ReplicationStore(self.settings.data_dir).get_extraction_audit(error.detail["audit_id"])

    async def test_contracts_have_one_claim_type_and_no_nested_observable_grammar(self):
        old = structured.strict_json_schema(papers.SourcePaperEvidence)
        for name, model in paper_provider.STAGES:
            schema = structured.strict_json_schema(model)
            self.assertLess(len(json.dumps(schema)), len(json.dumps(old)))
            self.assertLessEqual(len(schema.get("$defs", {})), 1)
            self.assertTrue(schema["required"])
            self.assertFalse(schema["additionalProperties"])
        target_schema = structured.strict_json_schema(paper_provider.PaperReferences)
        self.assertNotIn("Observable", target_schema.get("$defs", {}))
        self.assertNotIn("Tolerance", target_schema.get("$defs", {}))

    async def test_complete_three_checkpoints_preserve_common_case_inputs_and_reference_kind(self):
        doc = document("Current 1 A. Radius 12 mm. Boundary A=0. Inductance 1.73 mH. Energy 0.000865042 J.")
        span = papers.source_spans(doc)[0].span_id
        packet = papers.SourcePaperEvidence(source_id=doc.source_id, title="Synthetic staged source",
            summary="Offline generic fixture", parameters=(
                papers.SourceEvidenceParameter(id="radius", case_id="baseline", description="Common radius", value=12,
                    unit="mm", evidence_kind="explicit", value_refs=(span,), unit_refs=(span,)),
                papers.SourceEvidenceParameter(id="current", case_id="case_b", description="Selected current", value=1,
                    unit="A", evidence_kind="explicit", source_refs=(span,)),
                papers.SourceEvidenceParameter(id="boundary", case_id="baseline", description="Outer boundary", value="A=0",
                    unit="text", evidence_kind="explicit", source_refs=(span,)),
                papers.SourceEvidenceParameter(id="unavailable", case_id="case_b", description="Missing material law", value=None,
                    unit=None, evidence_kind="unknown", unresolved_reason="Source does not specify its material law."),
                papers.SourceEvidenceParameter(id="derived_radius", case_id="case_b", description="Derived input", value=24,
                    unit="mm", evidence_kind="derived", derivation="Twice the cited radius; independently verify.", source_refs=(span,))),
            targets=tuple(papers.SourceNumericalTarget(id="reference_" + str(i), case_id="case_b",
                description="Published reference", value=1.73, unit="mH", observable={"quantity": "inductance", "circuit_id": "coil"},
                tolerance={"relative": .02}, reference_kind=kind, conditions="Original source condition", source_refs=(span,))
                for i, kind in enumerate(("reported_numerical", "analytical", "experimental"))),
            questions=("Fixture question",), limitations=("Fixture limitation",)).model_dump(mode="json")
        requests = []
        def handler(request):
            payload = json.loads(request.content)
            requests.append(payload)
            self.assertEqual(payload["provider"], {"require_parameters": True})
            self.assertEqual(payload["response_format"]["type"], "json_schema")
            self.assertTrue(payload["response_format"]["json_schema"]["strict"])
            self.assertEqual(payload["messages"][-1]["role"], "user")
            for cap in ("max_tokens", "max_completion_tokens", "reasoning", "temperature"):
                self.assertNotIn(cap, payload)
            candidate = staged_reply(packet, payload)
            if payload["response_format"]["json_schema"]["name"] == "paper_parameters":
                candidate["questions"] = ["Parameter checkpoint question"]
            if payload["response_format"]["json_schema"]["name"] == "paper_references":
                candidate["limitations"] = ["Reference checkpoint limitation"]
            return self.response(candidate)
        with self.transport(handler):
            outcome = await replication.extract_with_provider(doc, "Select case_b", self.settings)
        self.assertEqual(len(requests), 3)
        self.assertEqual([r["response_format"]["json_schema"]["name"] for r in requests],
                         ["paper_cases", "paper_parameters", "paper_references"])
        self.assertEqual([p.case_id for p in outcome.evidence.parameters], ["baseline", "case_b", "baseline", "case_b"])
        self.assertEqual({t.reference_kind for t in outcome.evidence.targets}, {"reported_numerical", "analytical", "experimental"})
        self.assertTrue(any("unavailable" in q and "material law" in q for q in outcome.evidence.questions))
        self.assertTrue(any("derived_radius" in q for q in outcome.evidence.questions))
        self.assertIn("Parameter checkpoint question", outcome.evidence.questions)
        self.assertIn("Reference checkpoint limitation", outcome.evidence.limitations)
        private = replication.ReplicationStore(self.settings.data_dir).get_extraction_audit(outcome.audit_id)
        self.assertEqual(len(private["paper_attempts"]), 3)
        self.assertEqual(private["candidate"]["parameters"][0]["value_refs"], [span])
        self.assertEqual(private["candidate"]["parameters"][0]["unit_refs"], [span])
        self.assertAlmostEqual(outcome.usage["cost"], .003)

    async def test_provider_rejection_retains_prior_checkpoint_and_error_usage_without_retry(self):
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                return self.response(staged_reply(wire_evidence(self.doc), request), cost=.01)
            return httpx.Response(400, headers={"private-header": self.settings.ai_key}, json={
                "error": {"message": "Compiled grammar rejected " + self.settings.ai_key},
                "id": "rejected-generation", "usage": {"cost": .02, "prompt_tokens": 9}})
        with self.transport(handler), self.assertRaises(HTTPException) as caught:
            await replication.extract_with_provider(self.doc, "", self.settings)
        self.assertEqual(len(requests), 2)
        error = caught.exception
        self.assertIn("strict paper-extraction stage", error.detail["message"])
        private = self.private_audit(error)
        self.assertEqual(private["stage"], "paper_parameters")
        self.assertEqual([a["outcome"] for a in private["paper_attempts"]], ["validated", "provider_rejected"])
        self.assertEqual(private["paper_attempts"][0]["candidate"]["source_id"], self.doc.source_id)
        self.assertEqual(private["paper_attempts"][1]["provider_response"]["id"], "rejected-generation")
        self.assertAlmostEqual(error.detail["usage"]["cost"], .03)
        self.assertEqual(error.detail["usage"]["prompt_tokens"], 19)
        self.assertNotIn(self.settings.ai_key, json.dumps(private))
        self.assertNotIn("private-header", json.dumps(private))

    async def test_missing_second_stage_wire_key_stops_without_fallback(self):
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            candidate = staged_reply(wire_evidence(self.doc), request)
            if len(requests) == 2:
                candidate["parameters"][0].pop("value_number")
            return self.response(candidate)
        with self.transport(handler), self.assertRaises(HTTPException) as caught:
            await replication.extract_with_provider(self.doc, "", self.settings)
        self.assertEqual(len(requests), 2)
        private = self.private_audit(caught.exception)
        self.assertEqual(private["stage"], "paper_parameters")
        self.assertTrue(any(i["path"] == "parameters.0.value_number" and i["code"] == "wire_required"
                            for i in private["issues"]))
        self.assertAlmostEqual(private["usage"]["cost"], .002)

    async def test_unknown_span_is_excluded_explicitly_after_all_checkpoints(self):
        packet = wire_evidence(self.doc)
        packet["targets"][0]["source_refs"] = ["span_does_not_exist"]
        def handler(request):
            return self.response(staged_reply(packet, request))
        with self.transport(handler):
            outcome = await replication.extract_with_provider(self.doc, "", self.settings)
        self.assertEqual(outcome.extraction_status, "partial")
        self.assertEqual(outcome.rejected_claim_count, 1)
        self.assertTrue(outcome.evidence.parameters)
        self.assertFalse(outcome.evidence.targets)
        self.assertEqual(outcome.validation_summary[0]["code"], "source_span_not_found")
        private = replication.ReplicationStore(self.settings.data_dir).get_extraction_audit(outcome.audit_id)
        self.assertEqual(private["candidate"]["targets"][0]["source_refs"], ["span_does_not_exist"])

    async def test_discovery_failure_retains_all_usage_attempts_and_safe_metadata(self):
        self.settings.hierarchical_ai = True
        attempts = [{"stage": "discovery", "usage": {"cost": .1, "total_tokens": 7}, "outcome": "validation_failed"},
                    {"stage": "diagnostic:discovery", "usage": {"cost": .02, "total_tokens": 3}, "outcome": "diagnostic_completed"},
                    {"stage": "discovery", "usage": {"cost": .3, "total_tokens": 11,
                        "completion_tokens_details": {"reasoning_tokens": 9}}, "outcome": "validation_failed",
                     "validation_issues": [{"path": "/id", "message": self.settings.ai_key}]}]
        error = AssemblyProviderError(502, "Synthetic discovery failure", attempts=attempts)
        with patch("app.assembly_provider.discover_assembly", new=AsyncMock(side_effect=error)), \
             patch("app.paper_provider._request", new=AsyncMock()) as requester, self.assertRaises(HTTPException) as caught:
            await replication.extract_with_provider(self.doc, "", self.settings)
        requester.assert_not_called()
        private = self.private_audit(caught.exception)
        self.assertEqual(private["stage"], "machine_and_case_discovery")
        self.assertEqual(private["assembly_discovery_failure"]["provider_calls"], 3)
        self.assertEqual(len(private["assembly_discovery_failure"]["attempts"]), 3)
        self.assertEqual(private["usage"]["total_tokens"], 21)
        self.assertAlmostEqual(private["usage"]["cost"], .42)
        self.assertEqual(private["usage"]["completion_tokens_details"]["reasoning_tokens"], 9)
        self.assertEqual(caught.exception.detail["usage"], private["usage"])
        self.assertNotIn(self.settings.ai_key, json.dumps(private))

    async def test_canonical_discovery_case_is_preserved_without_changing_span_ids(self):
        inventory = MachineInventory(id="source model", title="Fixture", input_mode="paper", family="unknown",
            physics="axisymmetric_dc_magnetics", components=[{"id": "coil", "name": "Coil", "role": "winding",
                "family": "winding", "scope": "region", "mode": "custom", "intent": "Use supplied source spans"}],
            cases=[{"id": "source-case-with-spaces and punctuation", "label": "Selected source variant"}],
            selected_case_id="source-case-with-spaces and punctuation", explanation="Generic fixture")
        packet = wire_evidence(self.doc)
        for section in ("parameters", "targets"):
            for claim in packet[section]:
                claim["case_id"] = inventory.selected_case_id
        def handler(request):
            payload = json.loads(request.content)
            self.assertIn(inventory.selected_case_id, json.dumps(payload["messages"]))
            return self.response(staged_reply(packet, request))
        with self.transport(handler):
            result = await paper_provider.extract_source_evidence(self.doc, "", self.settings, inventory=inventory)
        self.assertEqual(result.cases.selected_case_id, inventory.selected_case_id)
        self.assertEqual(result.candidate["parameters"][0]["case_id"], inventory.selected_case_id)
        self.assertEqual(result.candidate["parameters"][0]["source_refs"], packet["parameters"][0]["source_refs"])

    async def test_unselected_case_and_ambiguous_value_slots_are_not_silently_dropped(self):
        for mode in ("case", "two_values"):
            requests = []
            def handler(request):
                requests.append(json.loads(request.content))
                candidate = staged_reply(wire_evidence(self.doc), request)
                if len(requests) == 2:
                    if mode == "case":
                        candidate["parameters"][0]["case_id"] = "another_case"
                    else:
                        candidate["parameters"][0]["value_text"] = "contradicting text"
                return self.response(candidate)
            with self.subTest(mode=mode), self.transport(handler), self.assertRaises(HTTPException) as caught:
                await replication.extract_with_provider(self.doc, "", self.settings)
            self.assertEqual(len(requests), 2)
            private = self.private_audit(caught.exception)
            self.assertEqual(private["issues"][0]["code"], "stage_binding")
            self.assertTrue(private["paper_attempts"][1]["candidate"]["parameters"])

    async def test_messages_are_never_truncated_to_hide_overflow(self):
        def handler(request):
            candidate = staged_reply(wire_evidence(self.doc), request)
            candidate["questions"] = ["Unresolved question " + str(i) for i in range(10)]
            return self.response(candidate)
        with self.transport(handler), self.assertRaises(HTTPException) as caught:
            await replication.extract_with_provider(self.doc, "", self.settings)
        private = self.private_audit(caught.exception)
        self.assertEqual(private["stage"], "schema_and_source_validation")
        self.assertEqual(len(private["candidate"]["questions"]), 30)
        self.assertEqual(len(private["paper_attempts"]), 3)
        self.assertTrue(any(i["path"] == "questions" for i in private["issues"]))

    async def test_late_connection_failure_keeps_completed_stage_usage(self):
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            if len(requests) == 3:
                raise httpx.ConnectError("Synthetic network failure", request=request)
            return self.response(staged_reply(wire_evidence(self.doc), request))
        with self.transport(handler), self.assertRaises(HTTPException) as caught:
            await replication.extract_with_provider(self.doc, "", self.settings)
        private = self.private_audit(caught.exception)
        self.assertEqual(private["stage"], "paper_references")
        self.assertEqual(len(private["paper_attempts"]), 3)
        self.assertEqual(private["paper_attempts"][-1]["outcome"], "connection_failed")
        self.assertAlmostEqual(private["usage"]["cost"], .002)
        self.assertEqual(private["usage"]["total_tokens"], 28)

    async def test_nonfinite_rejected_candidate_still_has_audited_usage_and_original_content(self):
        def handler(request):
            candidate = staged_reply(wire_evidence(self.doc), request)
            if "parameters" in candidate:
                candidate["parameters"][0]["value_number"] = float("nan")
            return self.response(candidate)
        with self.transport(handler), self.assertRaises(HTTPException) as caught:
            await replication.extract_with_provider(self.doc, "", self.settings)
        private = self.private_audit(caught.exception)
        self.assertEqual(private["stage"], "paper_parameters")
        self.assertIsNone(private["candidate"]["parameters"][0]["value_number"])
        self.assertIn("NaN", private["provider_response"]["choices"][0]["message"]["content"])
        self.assertGreater(private["candidate_bytes"], 0)
        self.assertAlmostEqual(private["usage"]["cost"], .002)


if __name__ == "__main__":
    unittest.main()
