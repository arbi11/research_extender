"""The live matrix's audit and flow are tested without credits or native binaries."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
import zipfile

import httpx

from app import papers, replication
from app.jobs import SolverWorker
from app.main import Settings
from app.plan import wire_example
from app.planner import PlanProposal
from app.targets import TargetContract, evaluate_targets
from evaluation import hierarchy_live
from tests.test_assembly import custom_assembly
from tests.test_papers import text_pdf


class HierarchyHelpersTests(unittest.TestCase):
    def test_default_models_and_generic_cases_do_not_contain_an_answer_key(self):
        self.assertEqual(len(hierarchy_live.DEFAULT_MODELS), 3)
        self.assertIn("~z-ai/glm-latest", hierarchy_live.DEFAULT_MODELS)
        self.assertIn("anthropic/claude-haiku-5.5", hierarchy_live.DEFAULT_MODELS)
        self.assertIn("deepseek/deepseek-v4-pro-0813", hierarchy_live.DEFAULT_MODELS)
        with self.assertRaises(ValueError):
            hierarchy_live.LiveCase.model_validate({"id": "source", "prompt": "Reproduce it", "gold": {"value": 42}})

    def test_load_cases_resolves_pdfs_relative_to_manifest_and_preserves_requests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = [{"id": "paper", "prompt": "Use the original case", "pdf": "source.pdf",
                       "source_assets": [{"page": 2, "purpose": "figure"}]}]
            path = root / "cases.json"
            path.write_text(json.dumps(source))
            cases = hierarchy_live.load_cases(path)
            self.assertEqual(cases[0].pdf, str(root / "source.pdf"))
            self.assertEqual(cases[0].prompt, source[0]["prompt"])
            self.assertIsNone(cases[0].source_assets[0].case_id)

    def test_target_selection_uses_categories_and_case_not_reference_values(self):
        from tests.test_targets import target
        numerical = target(case_id="selected").model_dump(mode="json")
        analytical = copy.deepcopy(numerical)
        analytical.update(id="analytical", reference_kind="analytical", value=10000)
        other = copy.deepcopy(numerical)
        other.update(id="other_case", case_id="other")
        extracted = {"evidence": {"targets": [other, analytical, numerical]},
                     "assembly_inventory": {"selected_case_id": "selected"}}
        case, targets = hierarchy_live.select_targets(extracted, hierarchy_live.TargetSelection())
        self.assertEqual(case, "selected")
        self.assertEqual([value["id"] for value in targets], ["reference"])
        with self.assertRaises(ValueError):
            hierarchy_live.select_targets(extracted, hierarchy_live.TargetSelection(target_ids=["analytical"]))

    def test_administrative_circuit_binding_does_not_change_source_target(self):
        from tests.test_targets import target
        source = [target(quantity="inductance", value=1, unit="mH").model_dump(mode="json")]
        original = copy.deepcopy(source)
        bound, bindings = hierarchy_live.bind_targets(source, wire_example().model_dump(mode="json"))
        self.assertEqual(source, original)
        self.assertEqual(bound[0]["observable"]["circuit_id"], "wire")
        self.assertEqual(bound[0]["value"], 1)
        self.assertEqual(len(bindings), 1)
        plan = wire_example().model_dump(mode="json")
        plan["circuits"].append({"id": "second", "current_a": 1})
        with self.assertRaisesRegex(ValueError, "uniquely"):
            hierarchy_live.bind_targets(source, plan)


class DecodedAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_gzip_body_is_saved_losslessly_after_decoding_and_application_can_still_read(self):
        with tempfile.TemporaryDirectory() as temp:
            packet = {"id": "generation", "model": "actual-model", "provider": "actual-provider",
                "choices": [{"finish_reason": "stop", "message": {"content": '{"value":2.05e-5}'}}],
                "usage": {"cost": .001, "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                          "completion_tokens_details": {"reasoning_tokens": 2}}}
            decoded = json.dumps(packet, ensure_ascii=False).encode()
            audit = hierarchy_live.DecodedAudit(Path(temp), allowed_hosts={"provider.test"})
            seen = []
            async def original_request_hook(request):
                seen.append("request")
            async def original_response_hook(response):
                seen.append("response")
            def handler(request):
                return httpx.Response(200, headers={"Content-Encoding": "gzip", "Content-Type": "application/json"},
                                      content=gzip.compress(decoded))
            client_type = audit.client_class(httpx.AsyncClient)
            async with client_type(transport=httpx.MockTransport(handler), event_hooks={
                    "request": [original_request_hook], "response": [original_response_hook]}) as client:
                async with client.stream("POST", "https://provider.test/chat/completions", json={"model": "requested-model"}) as response:
                    chunks = [chunk async for chunk in response.aiter_bytes()]
                    self.assertEqual(b"".join(chunks), decoded)
                    self.assertEqual(response.json(), packet)
            self.assertEqual(seen, ["request", "response"])
            record = audit.exchanges[0]["response"]
            self.assertEqual(record["decoded_sha256"], hashlib.sha256(decoded).hexdigest())
            self.assertEqual((Path(temp) / record["body_file"]).read_bytes(), decoded)
            summary = audit.summary()
            self.assertTrue(summary["transcript_complete"])
            self.assertEqual(summary["reported_known_cost"], .001)
            self.assertEqual(summary["calls"][0]["returned_model"], "actual-model")
            self.assertEqual(summary["calls"][0]["usage"]["completion_tokens_details"]["reasoning_tokens"], 2)

    async def test_credentials_are_redacted_and_unreported_cost_is_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            secret = "private-provider-credential"
            escaped = "".join("\\u" + f"{ord(char):04x}" for char in secret)
            packet = {"message": secret, "encoded": escaped}
            audit = hierarchy_live.DecodedAudit(Path(temp), secrets=[secret], allowed_hosts={"provider.test"})
            client_type = audit.client_class(httpx.AsyncClient)
            async with client_type(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=packet))) as client:
                response = await client.post("https://provider.test/complete", json={"model": "model"})
            self.assertEqual(response.json(), packet, "The application receives unchanged content.")
            saved = json.dumps(audit.exchanges)
            self.assertNotIn(secret, saved)
            self.assertNotIn(escaped, (Path(temp) / "0001.response.txt").read_text())
            self.assertEqual(audit.exchanges[0]["response"]["json"]["encoded"], "[REDACTED]")
            self.assertEqual(audit.summary()["calls_without_reported_cost"], 1)
            self.assertFalse(audit.summary()["reported_cost_complete"])


class HierarchyFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_guard_precedes_environment_source_and_output_access(self):
        with patch.object(hierarchy_live.Settings, "from_env") as settings:
            with self.assertRaisesRegex(ValueError, "--live"):
                await hierarchy_live.run_case(hierarchy_live.LiveCase(id="case", prompt="Build it", pdf="missing.pdf"),
                                               "model", Path("unused"))
            with self.assertRaisesRegex(ValueError, "--live"):
                await hierarchy_live.run_matrix([], Path("unused"))
            settings.assert_not_called()

    async def run_mocked(self, *, paper=False, unsupported=False, target_fail=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root / "source.pdf"
        source.write_bytes(text_pdf("Energy 0.000865042 J. Current 1 A."))
        case = hierarchy_live.LiveCase(id="synthetic", prompt="Build the specified model.", pdf=str(source) if paper else None)
        settings = Settings(data_dir=root / "unchanged", token="test-assistant-credential", ai_key="test-provider-credential")
        assembly = custom_assembly().model_dump(mode="json")
        if unsupported:
            assembly["required_physics"] = ["permanent_magnets"]
        proposal = PlanProposal(spec=None if unsupported else wire_example(), explanation="Mocked typed model.", attempts=1,
            assembly=assembly if unsupported else None, questions=["Unsupported permanent magnets."] if unsupported else [])
        native = []
        def run_native(worker, job):
            native.append(job["kind"])
            folder = worker.store.root / job["id"]
            folder.mkdir(parents=True, exist_ok=True)
            if job["kind"] == "mesh":
                self.assertTrue(worker.replication_store.get_contract(
                    next(path.stem for path in (worker.replication_store.root / "contracts").glob("*.json")))) if paper else None
                worker.store.update(job["id"], "completed", result={"status": "passed", "elements": 100})
            else:
                if paper:
                    record = worker.replication_store.get_contract(job["target_contract_id"])
                    result = evaluate_targets(TargetContract.model_validate(record["contract"]),
                        {"energy": {"value": .0004 if target_fail else .000865042, "unit": "J"}},
                        plan=job["spec"], expected_contract_sha256=record["contract_sha256"])
                    worker.store.annotate(job["id"], target_report=result)
                with zipfile.ZipFile(folder / "download.zip", "w") as archive:
                    archive.writestr("test-fixture.txt", "Mocked native fixture; no actual FEMM claim.")
                worker.store.update(job["id"], "completed", result={"energy": {"value": .000865042, "unit": "J"}})
        async def extract(document, instructions, supplied_settings):
            citation = {"source_id": document.source_id, "page": 1, "quote": "Energy 0.000865042 J."}
            evidence = papers.PaperEvidence.model_validate({"source_id": document.source_id,
                "title": "Synthetic flow source", "summary": "Mocked extraction; no scientific claim.",
                "parameters": [], "targets": [{"id": "reference", "description": "Fixed synthetic target",
                    "observable": {"quantity": "energy"}, "value": .000865042, "unit": "J",
                    "tolerance": {"relative": .02}, "reference_kind": "reported_numerical",
                    "conditions": "Source case", "citations": [citation]}]})
            return replication.ExtractionOutcome(evidence, {"cost": .001})
        with patch("app.planner.interpret_plan", new=AsyncMock(return_value=proposal)) as planner, \
             patch("app.replication.extract_with_provider", new=AsyncMock(side_effect=extract)) as extractor, \
             patch.object(SolverWorker, "run", new=run_native), \
             patch("app.main.shutil.which", return_value="/mocked/femm-lua"):
            report = await hierarchy_live.run_case(case, "explicit-model", root / "trial", live=True, settings=settings)
        self.assertEqual(planner.await_count, 1)
        self.assertTrue(planner.call_args.kwargs["hierarchical"])
        self.assertEqual(extractor.await_count, 1 if paper else 0)
        self.assertFalse(settings.data_dir.exists())
        self.assertNotIn(settings.ai_key, (root / "trial" / "report.json").read_text())
        manifest = json.loads((root / "trial" / "manifest.json").read_bytes())
        self.assertEqual(manifest["report.json"], hashlib.sha256((root / "trial" / "report.json").read_bytes()).hexdigest())
        self.assertEqual(report["status"], "locked")
        self.assertEqual(report["source_fidelity"]["status"], "pending_independent_audit")
        return report, native

    async def test_natural_language_uses_actual_api_sequence_and_isolated_worker(self):
        report, native = await self.run_mocked()
        self.assertTrue(report["execution_pipeline_passed"], report)
        self.assertEqual(native, ["mesh", "solve"])
        self.assertNotIn("published_target_comparison", report["stages"])

    async def test_paper_target_is_frozen_before_both_native_mesh_and_solver(self):
        report, native = await self.run_mocked(paper=True)
        self.assertTrue(report["execution_pipeline_passed"], report)
        self.assertEqual(native, ["mesh", "solve"])
        names = [entry["name"] for entry in report["api_calls"]]
        self.assertLess(names.index("freeze_targets"), names.index("queue_mesh"))
        self.assertLess(names.index("freeze_targets"), names.index("queue_solve"))
        self.assertEqual(report["target_report"]["checks"][0]["target"], .000865042)

    async def test_native_success_with_target_failure_remains_failed_without_repair(self):
        report, native = await self.run_mocked(paper=True, target_fail=True)
        self.assertFalse(report["execution_pipeline_passed"])
        self.assertEqual(native, ["mesh", "solve"])
        self.assertEqual(report["target_report"]["status"], "failed")
        self.assertEqual(report["external_retries"], 0)

    async def test_unsupported_hierarchy_and_preview_survive_without_native_execution(self):
        report, native = await self.run_mocked(unsupported=True)
        self.assertFalse(report["execution_pipeline_passed"])
        self.assertEqual(native, [])
        self.assertTrue(report["assembly_id"])
        self.assertTrue(any(call["name"] == "assembly_preview" for call in report["api_calls"]))
        self.assertIn("permanent_magnets", json.dumps(report["proposal"]["assembly"]))

    async def test_existing_outputs_are_never_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "report.json").write_text("previous failure")
            with self.assertRaisesRegex(ValueError, "overwritten"):
                await hierarchy_live.run_case(hierarchy_live.LiveCase(id="case", prompt="Build it"), "model", root,
                                               live=True, settings=Settings(data_dir=root, ai_key="configured"))
            self.assertEqual((root / "report.json").read_text(), "previous failure")


if __name__ == "__main__":
    unittest.main()
