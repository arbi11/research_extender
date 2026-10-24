"""Opt-in paper runner tests use mocked providers/native worker, never credits."""
import copy
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from app import papers
from app.jobs import SolverWorker
from app.main import Settings
from app.plan import GeometryPlan
from app.planner import PlanProposal
from app.targets import TargetContract, evaluate_targets
from evaluation import paper_live
from evaluation.replication import build_lubin_plan, load_fixture
from tests.test_papers import text_pdf


def renamed_plan():
    value = build_lubin_plan("iron_center", protocol="near_field_v2").model_dump(mode="json")
    value["circuits"][0]["id"] = "excitation"
    for region in value["regions"]:
        if region["circuit"] == "coil":
            region["circuit"] = "excitation"
        region["id"] = "named_" + region["id"]
    value["force_region"] = "named_core"
    return GeometryPlan.model_validate(value)


class PaperLiveHelpersTests(unittest.TestCase):
    def test_target_selection_does_not_substitute_analytical_or_air_reference(self):
        fixture = load_fixture()
        target = copy.deepcopy(fixture["cases"][0]["target"])
        analytical = copy.deepcopy(target)
        analytical.update(id="analytical", reference_kind="analytical", value=85.8)
        air = copy.deepcopy(target)
        air.update(id="air", case_id="air_core", value=49)
        selected = paper_live.select_centred_target({"targets": [air, analytical, target]})
        self.assertEqual(selected.value, 87)
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            paper_live.select_centred_target({"targets": [air, analytical]})
        target.update(value=.087, unit="H")
        self.assertEqual(paper_live.select_centred_target({"targets": [target]}).value, .087)

    def test_gold_is_only_an_audit_and_identifiers_do_not_determine_matching(self):
        plan = renamed_plan()
        audit = paper_live.audit_generated_plan(plan)
        self.assertTrue(audit["passed"], audit)
        self.assertEqual(audit["circuit_id"], "excitation")
        for change in ("turns", "permeability", "current", "geometry"):
            value = plan.model_dump(mode="json")
            if change == "turns":
                next(region for region in value["regions"] if region["circuit"])["turns"] = 999
            elif change == "permeability":
                next(material for material in value["materials"] if material["id"] == "iron")["relative_permeability"] = 9000
            elif change == "current":
                value["circuits"][0]["current_a"] = 19
            else:
                core = next(region for region in value["regions"] if region["material"] == "iron")
                core["shape"]["x_max"] = 31
            with self.subTest(change=change):
                self.assertFalse(paper_live.audit_generated_plan(GeometryPlan.model_validate(value))["passed"])

    def test_prompts_contain_numerical_choices_but_not_gold_physical_values(self):
        prompt = paper_live.EXTRACTION_INSTRUCTIONS + paper_live.PLANNING_INSTRUCTIONS
        for source_value in ("87 mH", "85.8", "10000", "20 A", "1000 turns", "30 mm", "50 mm", "70 mm"):
            self.assertNotRegex(prompt, r"(?<![0-9])" + re.escape(source_value))
        self.assertIn("750 mm", prompt)
        self.assertIn("current density", prompt)
        self.assertIn("Ask about unknown", prompt)

    def test_safe_report_redacts_credentials_recursively(self):
        value = {"text": "provider-key assistant-token", "nested": [{"secret": "provider-key"}]}
        self.assertNotIn("provider-key", json.dumps(paper_live._safe(value, ["provider-key", "assistant-token"])))


class PaperLiveFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_opt_in_is_required_before_environment_or_network_access(self):
        with patch.object(paper_live.Settings, "from_env") as settings:
            with self.assertRaisesRegex(ValueError, "--live"):
                await paper_live.run_paper_live(Path("missing.pdf"), Path("unused"))
            settings.assert_not_called()

    async def run_mocked(self, *, wrong_current=False, questions=False, resume_report=None):
        if importlib.util.find_spec("pypdf") is None:
            self.skipTest("pypdf runtime dependency is not installed")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        pdf = root / "paper.pdf"
        pdf.write_bytes(text_pdf("Centred finite-element inductance 87 mH."))
        resume = None
        if resume_report is not None:
            resume = root / "previous-report.json"
            resume.write_text(json.dumps(resume_report))
        settings = Settings(data_dir=root / "untouched", token="assistant-test-secret", ai_key="provider-test-secret", start_worker=False)
        seen_native = []
        async def extract(document, instructions, supplied_settings):
            citation = {"source_id": document.source_id, "page": 1, "quote": "inductance 87 mH"}
            value = papers.PaperEvidence.model_validate({"source_id": document.source_id, "title": "Synthetic test source",
                "summary": "Mocked extraction for flow verification; not a live reproduction claim.",
                "parameters": [], "targets": [{"id": "reference", "case_id": "centred", "description": "Fixed test tolerance",
                    "observable": {"quantity": "inductance", "circuit_id": "paper_circuit"}, "value": 87, "unit": "mH",
                    "tolerance": {"relative": .02}, "reference_kind": "reported_numerical",
                    "conditions": "Centred iron core", "citations": [citation]}]})
            return value, {"cost": .001, "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        plan = renamed_plan()
        if wrong_current:
            packet = plan.model_dump(mode="json")
            packet["circuits"][0]["current_a"] = 19
            plan = GeometryPlan.model_validate(packet)
        proposal = PlanProposal(spec=plan, explanation="Mocked model for flow test.",
            questions=["Missing source information"] if questions else [], attempts=1)
        def native(worker, job):
            seen_native.append(job["kind"])
            if job["kind"] == "mesh":
                worker.store.update(job["id"], "completed", result={"status": "passed", "elements": 100})
            else:
                record = worker.replication_store.get_contract(job["target_contract_id"])
                actual = {"energy": {"value": 17.4, "unit": "J"}, "current_excitation": {"value": 20, "unit": "A"},
                          "flux_linkage_excitation": {"value": 1.74, "unit": "Wb-turn"}}
                checked = evaluate_targets(TargetContract.model_validate(record["contract"]), actual,
                    plan=job["spec"], expected_contract_sha256=record["contract_sha256"])
                worker.store.annotate(job["id"], target_report=checked)
                worker.store.update(job["id"], "completed", result=actual)
        with patch("app.replication.extract_with_provider", new=AsyncMock(side_effect=extract)) as extractor, \
             patch("app.planner.interpret_plan", new=AsyncMock(return_value=proposal)) as planner, \
             patch.object(SolverWorker, "run", new=native), \
             patch("app.main.shutil.which", return_value="/fake/femm-lua"):
            report = await paper_live.run_paper_live(pdf, root / "attempt", live=True, settings=settings,
                                                     resume_evidence=resume)
        self.assertFalse(settings.data_dir.exists(), "The original configured job directory must stay unchanged.")
        self.assertEqual(extractor.await_count, 0 if resume_report is not None else 1)
        self.assertEqual(planner.await_count, 1)
        self.assertTrue(planner.call_args.kwargs["evidence_context"]["preserve_paper_excitation"])
        self.assertIsNone(report["usage"]["planning"])
        if resume_report is None:
            self.assertEqual(report["usage"]["extraction"]["cost"], .001)
        else:
            self.assertIsNone(report["usage"]["extraction"])
            self.assertEqual(report["continuation"]["historical_extraction_usage"]["cost"], .001)
        stored = (root / "attempt" / "report.json").read_text()
        self.assertNotIn(settings.ai_key, stored)
        self.assertNotIn(settings.token, stored)
        self.assertEqual(report["status"], "complete")
        return report, seen_native

    async def test_real_asgi_flow_freezes_original_target_before_native_solve(self):
        report, native = await self.run_mocked()
        self.assertTrue(report["passed"], report)
        self.assertEqual(native, ["mesh", "solve"])
        names = [step["name"] for step in report["steps"]]
        self.assertLess(names.index("freeze_paper_target"), names.index("queue_solve"))
        self.assertEqual(report["target_report"]["checks"][0]["target"], 87)
        self.assertEqual(report["selected_original_target"]["observable"]["circuit_id"], "paper_circuit")
        recorded = next(step for step in report["steps"] if step["name"] == "record_revision")
        self.assertTrue(recorded["request"]["preserve_paper_excitation"])

    async def test_wrong_physics_is_retained_without_native_calls_or_gold_repair(self):
        report, native = await self.run_mocked(wrong_current=True)
        self.assertFalse(report["passed"])
        self.assertFalse(report["model_fidelity"]["checks"]["derived_current"])
        self.assertEqual(native, [])
        self.assertIn("No gold values", report["error"])

    async def test_clarification_retains_questions_without_native_calls(self):
        report, native = await self.run_mocked(questions=True)
        self.assertFalse(report["passed"])
        self.assertIn("clarification", report["error"])
        self.assertEqual(native, [])

    async def test_resume_reuses_verified_ai_extraction_without_another_provider_call(self):
        previous, _ = await self.run_mocked(wrong_current=True)
        report, native = await self.run_mocked(resume_report=previous)
        self.assertTrue(report["passed"], report)
        self.assertEqual(native, ["mesh", "solve"])
        self.assertFalse(any(step["name"] == "extract_evidence" for step in report["steps"]))
        original = next(step["response"] for step in previous["steps"] if step["name"] == "extract_evidence")
        replay = next(step["response"] for step in report["steps"] if step["name"] == "reuse_verified_evidence")
        self.assertEqual(replay, original)
        self.assertEqual(report["continuation"]["previous_circuit_arguments"][0]["current_a"], 19)
        expected = hashlib.sha256(json.dumps(previous).encode()).hexdigest()
        self.assertEqual(report["continuation"]["source_report_sha256"], expected)
        self.assertEqual(report["selected_original_target"], previous["selected_original_target"])

    async def test_resume_rejects_changed_source_record_or_reference_before_network(self):
        previous, _ = await self.run_mocked(wrong_current=True)
        source = text_pdf("Centred finite-element inductance 87 mH.")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "previous.json"
        for change, message in (("source", "PDF differs"), ("record", "content hash"),
                                ("target", "target changed"), ("incomplete", "successful PDF")):
            value = copy.deepcopy(previous)
            pdf_bytes = source
            if change == "source":
                pdf_bytes = source + b"\n"
            elif change == "record":
                extraction = next(step for step in value["steps"] if step["name"] == "extract_evidence")
                extraction["response"]["model"] += "-tampered"
            elif change == "target":
                value["selected_original_target"]["value"] = 88
            else:
                next(step for step in value["steps"] if step["name"] == "extract_evidence")["status_code"] = 502
            path.write_text(json.dumps(value))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                paper_live.verified_resume_evidence(path, pdf_bytes)


if __name__ == "__main__":
    unittest.main()
