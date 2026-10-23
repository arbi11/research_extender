"""Scoring must expose false accepts, rejections and missing provider evidence."""
import unittest

from evaluation.semantic import gate, load_cases, summarize
from evaluation.cad import CheckFailed, physical_wire
from app.decisions import DecisionTrace
from app.plan import wire_example


def traces(support=.99, sufficient=.99, confidence=.99):
    return [DecisionTrace(stage="preflight", question="route", answer="create", probability=.98,
                          confidence=confidence, probabilities={"create": .98, "modify": .01, "clarify": .01},
                          threshold=.9, passed=confidence >= .9, model="fixture"),
            *[DecisionTrace(stage="preflight", question=name, answer="yes", probability=p,
                            confidence=None, probabilities={"yes": p, "no": 1-p}, threshold=.9,
                            passed=p >= .9, model="fixture") for name, p in (("support", support), ("sufficiency", sufficient))]]


class EvaluationTests(unittest.TestCase):
    def test_seed_cases_have_valid_geometry_and_consistent_labels(self):
        cases = load_cases()
        self.assertEqual(len(cases), 28)
        self.assertTrue({"supported", "unsupported", "ambiguous", "faithful", "unfaithful"}
                        <= {case.category for case in cases})

    def test_metrics_distinguish_false_accepts_rejections_and_provider_errors(self):
        cases = load_cases()[:3]
        rows = [{"id": cases[0].id, "trace": [t.model_dump() for t in traces(support=.89)]},
                {"id": cases[1].id, "trace": [t.model_dump() for t in traces()]},
                {"id": cases[2].id, "error": {"status_code": 429, "message": "fixture"}}]
        summary = summarize(cases, rows, .9)
        self.assertEqual((summary["evaluated"], summary["false_accepts"], summary["false_rejects"],
                          summary["provider_errors"], summary["correct"]), (2, 1, 1, 1, 0))
        self.assertEqual(summary["questions"]["support"]["count"], 2)
        self.assertAlmostEqual(summary["questions"]["support"]["brier_score"], ((.89-1)**2 + .99**2)/2)
        self.assertEqual(summary["threshold_sweep"]["0.8"]["false_rejects"], 0)

    def test_missing_cases_are_unmeasured_and_route_confidence_is_required(self):
        cases = load_cases()
        evidence = traces(confidence=.89)
        self.assertFalse(gate(cases[0], evidence, .9))
        summary = summarize(cases, [{"id": cases[0].id, "trace": [t.model_dump() for t in evidence]}], .9)
        self.assertEqual(summary["unmeasured"], 27)
        with self.assertRaises(ValueError):
            summarize(cases, [{"id": cases[0].id, "trace": [evidence[0].model_dump()]}], .9)

    def test_native_acceptance_assertions_reject_semantically_wrong_valid_plan(self):
        data = wire_example().model_dump(mode="json")
        physical_wire(data, 1)
        data["depth_mm"] = 40
        with self.assertRaisesRegex(CheckFailed, "Depth"):
            physical_wire(data, 1)


if __name__ == "__main__":
    unittest.main()
