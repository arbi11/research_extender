"""The live matrix's audit and flow are tested without credits or native binaries."""
import copy
import asyncio
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
        experimental = copy.deepcopy(numerical)
        experimental.update(id="experimental", reference_kind="experimental", value=4321)
        other = copy.deepcopy(numerical)
        other.update(id="other_case", case_id="other")
        extracted = {"evidence": {"targets": [other, analytical, experimental, numerical]},
                     "assembly_inventory": {"selected_case_id": "selected"}}
        original = copy.deepcopy(extracted)
        case, targets = hierarchy_live.select_targets(extracted, hierarchy_live.TargetSelection())
        self.assertEqual(case, "selected")
        self.assertEqual([value["id"] for value in targets], ["analytical", "experimental", "reference"])
        self.assertEqual(targets, [analytical, experimental, numerical])
        self.assertEqual(extracted, original)
        _, selected = hierarchy_live.select_targets(extracted, hierarchy_live.TargetSelection(target_ids=["analytical"]))
        self.assertEqual(selected, [analytical])
        with self.assertRaisesRegex(ValueError, "selected case"):
            hierarchy_live.select_targets(extracted, hierarchy_live.TargetSelection(target_ids=["other_case"]))

    def test_target_selection_rejects_unknown_method_derived_and_zero_values(self):
        from tests.test_targets import target
        source = target(case_id="selected").model_dump(mode="json")
        for updates in ({"reference_kind": "unknown"},
                        {"evidence_kind": "derived", "derivation": "Compute a formula at the requested excitation."},
                        {"value": 0, "tolerance": {"absolute": .01, "relative": 0}}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                hierarchy_live.select_targets({"evidence": {"targets": [{**source, **updates}]},
                    "assembly_inventory": {"selected_case_id": "selected"}}, hierarchy_live.TargetSelection())

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
    async def test_mid_response_read_failure_is_retained_and_reraised_without_masking(self):
        failure = httpx.ReadError("Connection interrupted in the response body")
        class PartialStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'{"choices":'
                raise failure
        with tempfile.TemporaryDirectory() as temp:
            audit = hierarchy_live.DecodedAudit(Path(temp), allowed_hosts={"provider.test"})
            client_type = audit.client_class(httpx.AsyncClient)
            async with client_type(transport=httpx.MockTransport(lambda request:
                    httpx.Response(200, stream=PartialStream()))) as client:
                with self.assertRaises(httpx.ReadError) as caught:
                    await client.post("https://provider.test/complete", json={"model": "model"})
            self.assertIs(caught.exception, failure)
            record = json.loads((Path(temp) / "0001.json").read_bytes())
            self.assertEqual(record["status_code"], 200)
            self.assertEqual(record["response_observation_error"], "ReadError")
            self.assertIn("elapsed_seconds", record)
            self.assertNotIn("response", record)
            partial = record["partial_response"]
            self.assertFalse(partial["complete"])
            self.assertEqual((Path(temp) / partial["body_file"]).read_bytes(), b'{"choices":')
            summary = audit.summary()
            self.assertFalse(summary["transcript_complete"])
            self.assertEqual(summary["calls_without_reported_cost"], 1)
            self.assertFalse(summary["reported_cost_complete"])
            self.assertEqual(summary["observer_errors"], [{"stage": "response_observation", "type": "ReadError"}])

    async def test_confirmed_quota_stops_next_http_and_is_shared_without_replacing_first_reason(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gate = hierarchy_live.CohortQuotaGate(root / "cohort-stop.json")
            sent = []
            def provider(request):
                sent.append(request)
                return httpx.Response(403, json={"error": {"code": 403, "message": "Key weekly limit exceeded",
                    "metadata": {"limit_source": "openrouter_key_limit", "limit_usd": 2.5, "spend_usd": 2.76}}})
            audit = hierarchy_live.DecodedAudit(root / "first", allowed_hosts={"provider.test"}, quota_gate=gate)
            async with audit.client_class(httpx.AsyncClient)(transport=httpx.MockTransport(provider)) as client:
                response = await client.post("https://provider.test/complete", json={"model": "model"})
                self.assertEqual(response.status_code, 403)
                first_marker = gate.path.read_bytes()
                with self.assertRaises(hierarchy_live.CohortStopped):
                    await client.post("https://provider.test/complete", json={"model": "fallback"})
            # A separately created observer uses the same on-disk marker, as
            # independently launched model processes do.
            other = hierarchy_live.CohortQuotaGate(gate.path)
            other.trip({"kind": "later_limit_must_not_replace_first"})
            self.assertEqual(gate.path.read_bytes(), first_marker)
            second = hierarchy_live.DecodedAudit(root / "second", allowed_hosts={"provider.test"}, quota_gate=other)
            async with second.client_class(httpx.AsyncClient)(transport=httpx.MockTransport(provider)) as client:
                with self.assertRaises(hierarchy_live.CohortStopped):
                    await client.post("https://provider.test/complete", json={"model": "other-process"})
            self.assertEqual(len(sent), 1)
            self.assertEqual(audit.summary()["provider_call_count"], 1)
            self.assertEqual(audit.summary()["calls_without_reported_cost"], 1)
            self.assertFalse(audit.summary()["blocked_provider_requests"][0]["http_request_sent"])
            self.assertEqual(second.summary()["provider_call_count"], 0)

    async def test_payment_required_stops_but_unrelated_forbidden_is_not_a_quota(self):
        self.assertIsNotNone(hierarchy_live._provider_quota_reason(402, {"error": {"message": "Insufficient credits"}}))
        self.assertIsNone(hierarchy_live._provider_quota_reason(403, {"error": {"message": "Model access forbidden"}}))
        self.assertIsNone(hierarchy_live._provider_quota_reason(400, {"error": {"message": "Output schema budget exceeded"}}))

    async def test_payment_required_with_interrupted_body_still_stops_and_preserves_read_error(self):
        failure = httpx.ReadError("Interrupted payment error body")
        class PartialStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'{"error":'
                raise failure
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gate = hierarchy_live.CohortQuotaGate(root / "stop.json")
            sent = []
            def provider(request):
                sent.append(request)
                return httpx.Response(402, stream=PartialStream())
            audit = hierarchy_live.DecodedAudit(root / "audit", allowed_hosts={"provider.test"}, quota_gate=gate)
            async with audit.client_class(httpx.AsyncClient)(transport=httpx.MockTransport(provider)) as client:
                with self.assertRaises(httpx.ReadError) as caught:
                    await client.post("https://provider.test/complete", json={"model": "mock"})
                self.assertIs(caught.exception, failure)
                with self.assertRaises(hierarchy_live.CohortStopped):
                    await client.post("https://provider.test/complete", json={"model": "mock"})
            self.assertEqual(len(sent), 1)
            self.assertEqual(gate.reason()["http_status"], 402)
            self.assertTrue(gate.reason()["response_body_incomplete"])
            self.assertFalse(audit.summary()["reported_cost_complete"])

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


class HierarchyStopTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_provider_quota_stops_same_case_fallback_and_remaining_matrix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sent = []
            real_client = httpx.AsyncClient
            def provider(request):
                sent.append(request)
                return httpx.Response(403, json={"error": {"code": 403, "message": "Key weekly limit exceeded",
                    "metadata": {"limit_source": "openrouter_key_limit"}}})
            class RoutedClient(real_client):
                def __init__(self, *args, **kwargs):
                    kwargs.setdefault("transport", httpx.MockTransport(provider))
                    super().__init__(*args, **kwargs)
            async def planner(*args, **kwargs):
                async with httpx.AsyncClient() as client:
                    await client.post("https://provider.test/complete", json={"model": "draft"})
                    # Even an application fallback is blocked before transport.
                    await client.post("https://provider.test/complete", json={"model": "fallback"})
                raise AssertionError("A provider call escaped the quota gate")
            settings = Settings(data_dir=root / "unchanged", ai_key="mock-provider-credential",
                                ai_base_url="https://provider.test/v1")
            with patch.object(httpx, "AsyncClient", RoutedClient), \
                 patch("app.planner.interpret_plan", new=AsyncMock(side_effect=planner)) as interpret, \
                 patch("app.main.shutil.which", return_value="/mocked/femm-lua"):
                summary = await hierarchy_live.run_matrix([
                    {"id": "first", "prompt": "First request"}, {"id": "second", "prompt": "Second request"}],
                    root / "matrix", live=True, models=["model-a", "model-b"], settings=settings,
                    quota_stop_path=root / "shared-stop.json")
            self.assertEqual(len(sent), 1)
            self.assertEqual(interpret.await_count, 1)
            self.assertEqual(summary["trial_count"], 1)
            self.assertEqual(len(summary["not_started"]), 3)
            self.assertEqual(summary["termination"]["kind"], "provider_quota_stop")
            trial = summary["trials"][0]
            self.assertEqual(trial["termination"]["scientific_outcome"], "unassessed")
            self.assertEqual(trial["source_fidelity"]["status"], "unassessed")
            self.assertEqual(trial["provider_audit"]["provider_call_count"], 1)
            self.assertEqual(trial["provider_audit"]["calls_without_reported_cost"], 1)
            self.assertFalse((root / "unchanged").exists())

    async def test_cancellation_locks_partial_audit_and_reraises_without_scheduling_next_case(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            partial_seen, wait_forever = asyncio.Event(), asyncio.Event()
            sent = []
            class PartialStream(httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield b'{"choices":'
                    partial_seen.set()
                    await wait_forever.wait()
            def provider(request):
                sent.append(request)
                if len(sent) == 1:
                    return httpx.Response(200, json={"model": "mock", "usage": {"cost": .01,
                        "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
                return httpx.Response(200, stream=PartialStream())
            real_client = httpx.AsyncClient
            class RoutedClient(real_client):
                def __init__(self, *args, **kwargs):
                    kwargs.setdefault("transport", httpx.MockTransport(provider))
                    super().__init__(*args, **kwargs)
            async def planner(*args, **kwargs):
                async with httpx.AsyncClient() as client:
                    for _ in range(2):
                        await client.post("https://provider.test/complete", json={"model": "mock"})
                raise AssertionError("Interrupted provider call unexpectedly completed")
            settings = Settings(data_dir=root / "unchanged", ai_key="mock-provider-credential",
                                ai_base_url="https://provider.test/v1")
            with patch.object(httpx, "AsyncClient", RoutedClient), \
                 patch("app.planner.interpret_plan", new=AsyncMock(side_effect=planner)) as interpret, \
                 patch("app.main.shutil.which", return_value="/mocked/femm-lua"):
                task = asyncio.create_task(hierarchy_live.run_matrix([
                    {"id": "first", "prompt": "First request"}, {"id": "second", "prompt": "Second request"}],
                    root / "matrix", live=True, models=["mock"], settings=settings))
                await asyncio.wait_for(partial_seen.wait(), timeout=3)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertEqual(len(sent), 2)
            self.assertEqual(interpret.await_count, 1)
            summary = json.loads((root / "matrix" / "summary.json").read_bytes())
            self.assertEqual(summary["termination"]["kind"], "task_cancelled")
            self.assertEqual(len(summary["not_started"]), 1)
            report_path = next((root / "matrix").glob('*/*/report.json'))
            report = json.loads(report_path.read_bytes())
            self.assertEqual(report["status"], "locked")
            self.assertEqual(report["termination"]["kind"], "task_cancelled")
            self.assertEqual(report["termination"]["scientific_outcome"], "unassessed")
            self.assertFalse(report["execution_pipeline_passed"])
            self.assertEqual(report["provider_audit"]["reported_known_cost"], .01)
            self.assertEqual(report["provider_audit"]["calls_without_reported_cost"], 1)
            self.assertFalse(report["provider_audit"]["transcript_complete"])
            exchange = json.loads((report_path.parent / "provider" / "0002.json").read_bytes())
            self.assertEqual(exchange["response_observation_error"], "CancelledError")
            self.assertFalse(exchange["partial_response"]["complete"])
            self.assertEqual((report_path.parent / "provider" / exchange["partial_response"]["body_file"]).read_bytes(), b'{"choices":')
            manifest = json.loads((report_path.parent / "manifest.json").read_bytes())
            self.assertTrue(all(hashlib.sha256((report_path.parent / name).read_bytes()).hexdigest() == digest
                                for name, digest in manifest.items()))


class HierarchyFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_hierarchical_planner_consumes_small_strict_stages_without_monolithic_fallback(self):
        from tests.test_assembly import coil_assembly
        from tests.test_assembly_provider import PROMPT, staged_replies
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            replies = staged_replies(coil_assembly(), include_discovery=True)
            requests, native = [], []
            real_client = httpx.AsyncClient
            def provider(request):
                payload = json.loads(request.content)
                if request.url.path.endswith("/decisions"):
                    return httpx.Response(200, json={"id": "routing-test", "model": "typesafe/jev-1.14",
                        "answers": {"route": {"type": "choice", "choice": "create_plan", "confidence": 1,
                            "probabilities": {"create_plan": 1, "modify_plan": 0, "clarify": 0}}},
                        "usage": {"cost": .0001, "input_tokens": 10, "output_tokens": 2}})
                requests.append(payload)
                reply = replies[len(requests) - 1]
                return httpx.Response(200, json={"id": "stage-" + str(len(requests)), "model": "stage-model",
                    "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(reply)}}],
                    "usage": {"cost": .001, "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
            class RoutedClient(real_client):
                def __init__(self, *args, **kwargs):
                    if "transport" not in kwargs:
                        kwargs["transport"] = httpx.MockTransport(provider)
                    super().__init__(*args, **kwargs)
            def native_run(worker, job):
                native.append(job["kind"])
                folder = worker.store.root / job["id"]
                folder.mkdir(parents=True, exist_ok=True)
                if job["kind"] == "solve":
                    with zipfile.ZipFile(folder / "download.zip", "w") as archive:
                        archive.writestr("fixture.txt", "Offline native fixture")
                worker.store.update(job["id"], "completed", result={"status": "passed"} if job["kind"] == "mesh" else {})
            settings = Settings(data_dir=root / "unchanged", ai_key="provider-test-credential",
                                ai_base_url="https://provider.test/v1")
            case = hierarchy_live.LiveCase(id="staged_coil", prompt=PROMPT)
            with patch.object(httpx, "AsyncClient", RoutedClient), patch.object(SolverWorker, "run", native_run), \
                 patch("app.main.shutil.which", return_value="/mocked/femm-lua"):
                report = await hierarchy_live.run_case(case, "stage-model", root / "trial", live=True, settings=settings)
            self.assertTrue(report["execution_pipeline_passed"], report)
            self.assertEqual(native, ["mesh", "solve"])
            names = [request["response_format"]["json_schema"]["name"] for request in requests]
            self.assertEqual(names, ["MachineInventory", "AssemblySettingsDraft", "Rectangle", "ComponentDraft"])
            self.assertFalse(any("AssemblyDraft" == name or "MachineAssembly" == name for name in names))
            self.assertEqual(report["proposal"]["assembly"]["components"][0]["id"], "coil")
            self.assertEqual(report["provider_audit"]["provider_call_count"], 5)
            self.assertTrue(report["provider_audit"]["transcript_complete"])
            for request in requests:
                self.assertTrue(request["provider"]["require_parameters"])
                self.assertNotIn("max_tokens", request)

    async def test_live_guard_precedes_environment_source_and_output_access(self):
        with patch.object(hierarchy_live.Settings, "from_env") as settings:
            with self.assertRaisesRegex(ValueError, "--live"):
                await hierarchy_live.run_case(hierarchy_live.LiveCase(id="case", prompt="Build it", pdf="missing.pdf"),
                                               "model", Path("unused"))
            with self.assertRaisesRegex(ValueError, "--live"):
                await hierarchy_live.run_matrix([], Path("unused"))
            settings.assert_not_called()

    async def run_mocked(self, *, paper=False, unsupported=False, target_fail=False,
                         reference_kind="reported_numerical"):
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
                    "tolerance": {"relative": .02}, "reference_kind": reference_kind,
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
        self.assertEqual(report["version"], hierarchy_live.PROTOCOL_VERSION)
        self.assertEqual(report["target_selection_policy"]["reference_kinds"],
                         ["reported_numerical", "analytical", "experimental"])
        self.assertFalse(report["target_selection_policy"]["formula_substitution"])
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

    async def test_analytical_published_value_remains_analytical_through_frozen_comparison(self):
        report, native = await self.run_mocked(paper=True, reference_kind="analytical")
        self.assertTrue(report["execution_pipeline_passed"], report)
        self.assertEqual(native, ["mesh", "solve"])
        self.assertEqual(report["selected_source_targets"][0]["reference_kind"], "analytical")
        frozen = next(call for call in report["api_calls"] if call["name"] == "freeze_targets")
        self.assertEqual(frozen["request"]["targets"][0]["reference_kind"], "analytical")
        check = report["target_report"]["checks"][0]
        self.assertEqual(check["reference_kind"], "analytical")
        self.assertEqual(check["target"], .000865042)
        self.assertEqual(check["status"], "passed")

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
