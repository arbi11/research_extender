"""Paper replications: the index, case rules, the skill files and model drawings."""
from copy import deepcopy
import json
import tempfile
from pathlib import Path
import unittest

from femm_brain import papers
from femm_brain.reproduce import load_papers

SKILL = Path(__file__).parents[1] / "skills" / "replicate-paper"
ROOT = Path(__file__).parents[2]


def minimal_case(**changes):
    """A frozen two-region case: a magnet disk in air, with one parameter of each provenance."""
    case = {
        "schema_version": 1, "id": "test-2026-disk", "set": "development", "status": "frozen", "title": "Disk",
        "source": {"citation": "Test", "url": "https://example.org", "access": "test"},
        "machine": {"family": "other", "problem": "planar"},
        "article": {
            "parameters": {
                "radius_in": {"value": 1.0, "unit": "in", "provenance": "quoted", "quote": "a 1 in disk", "location": "p. 1"},
                "radius_mm": {"value": 25.4, "unit": "mm", "provenance": "derived", "expression": "radius_in*25.4",
                              "inputs": ["radius_in"]},
                "domain_mm": {"value": 80.0, "unit": "mm", "provenance": "figure", "figure": "Fig. 1",
                              "scale_reference": "the stated radius", "uncertainty": 2.0},
                "mu_r": {"value": 1.05, "unit": "ratio", "provenance": "assumed", "reason": "not stated",
                         "sensitivity_control": "mu-high"},
            },
            "assembly": {"problem": {"physics": "magnetics", "units": "millimeters", "type": "planar", "frequency": 0, "depth": 1},
                         "steps": [{"use": "material", "name": "Air"},
                                   {"use": "material", "name": "Magnet", "values": {"Mu_x": "$mu_r", "Mu_y": "$mu_r", "H_c": 900000}},
                                   {"use": "fixed_potential_boundary", "name": "A0"},
                                   {"use": "circle", "radius": "$radius_mm"},
                                   {"use": "circle", "radius": "$domain_mm", "boundary": "A0"},
                                   {"use": "block_label", "at": [0, 0], "material": "Magnet"},
                                   {"use": "block_label", "at": [0, "=(radius_mm+domain_mm)/2"], "material": "Air"}]}},
        "controls": [{"id": "mu-high", "level": "article", "parameters": {"mu_r": 1.1}, "purpose": "Magnet permeability"}],
        "gaps": [{"item": "magnet permeability", "affects": "field", "decision": "assumed"}],
        "measurements": [{"id": "centre", "op": "point_values", "points": [[0, 0]], "fields": ["B1"]}],
        "metrics": [{"name": "bx_T", "value": ["centre:B1", 0], "unit": "T"}],
        "targets": [{"metric": "bx_T", "value": 1.0, "kind": "analytic", "relative_tolerance": 0.01, "quote": "B = 1 T",
                     "location": "Eq. 1", "used_for_acceptance": True}],
        "coverage": {"scope": "core"},
    }
    case.update(changes)
    return case


class IndexTests(unittest.TestCase):
    def test_index_and_cases_agree_and_sets_are_balanced(self):
        self.assertEqual(papers.check_index(), [])
        entries = papers.load_index()["papers"]
        sets = [entry["set"] for entry in entries]
        # The development set grew from 6 to 8 on 2026-10-26 (the magnetic gear and the double-cage bar, promoted from the
        # unassigned candidates); the held-out set stays at the same four papers.
        self.assertEqual((sets.count("development"), sets.count("held_out")), (8, 4))
        self.assertTrue(all(entry["set"] in ("development", "held_out", "unassigned") for entry in entries))

    def test_index_entries_need_planning_fields_and_the_screening_vocabulary(self):
        entry = dict(papers.load_index()["papers"][-1], id="test-2026-entry", status="candidate",
                     geometry="mostly", target_sources=["measured", "rumour"])
        del entry["teaches"]
        found = [p for p in papers.check_index({"papers": [entry]}) if p.startswith("test-2026-entry")]
        self.assertTrue(any("missing teaches" in p for p in found))
        self.assertTrue(any("geometry must be one of" in p for p in found))
        self.assertTrue(any("unknown target sources rumour" in p for p in found))

    def test_every_case_file_passes_its_checks(self):
        for case in load_papers():
            with self.subTest(case["id"]):
                self.assertEqual(papers.check(case)["problems"], [])


class CaseRuleTests(unittest.TestCase):
    def problems(self, case):
        return papers.check(case)["problems"]

    def test_the_minimal_case_passes(self):
        self.assertEqual(self.problems(minimal_case()), [])

    def test_derived_values_must_follow_their_expressions(self):
        case = minimal_case()
        case["article"]["parameters"]["radius_mm"]["value"] = 25.0
        self.assertTrue(any("differs from its expression" in p for p in self.problems(case)))

    def test_assumptions_need_a_control_that_varies_them(self):
        case = minimal_case(controls=[])
        self.assertTrue(any("needs a control" in p for p in self.problems(case)))

    def test_provenance_fields_are_required(self):
        case = minimal_case()
        del case["article"]["parameters"]["radius_in"]["location"]
        case["article"]["parameters"]["domain_mm"]["uncertainty"] = 0
        found = self.problems(case)
        self.assertTrue(any("radius_in: quoted needs location" in p for p in found))
        self.assertTrue(any("positive uncertainty" in p for p in found))

    def test_weak_targets_cannot_decide_acceptance(self):
        digitized = dict(minimal_case()["targets"][0], kind="digitized")
        measured = dict(minimal_case()["targets"][0], kind="measured")
        self.assertTrue(any("twice" in p for p in self.problems(minimal_case(targets=[digitized]))))
        self.assertTrue(any("allowance" in p for p in self.problems(minimal_case(targets=[measured]))))
        self.assertEqual(self.problems(minimal_case(targets=[dict(digitized, digitization_uncertainty=0.005)])), [])
        # An absolute uncertainty needs an absolute (or a wide enough relative) tolerance of at least twice it.
        absolute = dict(digitized, absolute_digitization_uncertainty=0.02)
        self.assertTrue(any("twice" in p for p in self.problems(minimal_case(targets=[absolute]))))
        self.assertEqual(self.problems(minimal_case(targets=[dict(absolute, absolute_tolerance=0.04)])), [])
        self.assertEqual(self.problems(minimal_case(targets=[dict(absolute, relative_tolerance=0.04)])), [])

    def test_quotes_may_come_from_a_cached_related_source(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder)
            for sha, text in (("a" * 64, "a 1 in disk and B = 1 T"), ("b" * 64, "the companion says mu is 1.05")):
                (cache / sha).mkdir()
                (cache / sha / "text.txt").write_text(text)
            case = minimal_case()
            case["source"].update(pdf_sha256="a" * 64, related=[{"id": "companion", "pdf_sha256": "b" * 64}])
            case["article"]["parameters"]["mu_quoted"] = {"value": 1.05, "unit": "ratio", "provenance": "quoted",
                                                          "quote": "mu is 1.05", "location": "p. 2", "source": "companion"}
            self.assertEqual(papers.check(case, cache=cache, build=False)["problems"], [])
            case["article"]["parameters"]["mu_quoted"]["source"] = "main"
            self.assertTrue(any("not found" in p for p in papers.check(case, cache=cache, build=False)["problems"]))
            case["article"]["parameters"]["mu_quoted"]["source"] = "elsewhere"
            self.assertTrue(any("unknown source elsewhere" in p for p in papers.check(case, cache=cache, build=False)["problems"]))

    def test_open_questions_block_freezing_and_regions_are_checked(self):
        case = minimal_case(gaps=[{"item": "turns", "affects": "flux", "decision": "ask"}])
        self.assertTrue(any("resolve the question" in p for p in self.problems(case)))
        case = minimal_case()
        case["article"]["assembly"]["steps"].append({"use": "block_label", "at": [1, 1], "material": "Air"})
        self.assertTrue(any(p.startswith("region:") for p in self.problems(case)))

    def test_held_out_cases_record_the_skill_version(self):
        self.assertTrue(any("skill_version" in p for p in self.problems(minimal_case(set="held_out"))))

    def test_a_frozen_case_needs_a_measurement_plan(self):
        case = minimal_case(targets=[])
        del case["measurements"]
        found = self.problems(case)
        self.assertIn("a frozen case needs at least one acceptance target or assessor", found)
        self.assertIn("a frozen case needs measurements", found)


class SkillTests(unittest.TestCase):
    def test_the_claude_wrapper_shares_the_skill_front_matter(self):
        skill = (SKILL / "SKILL.md").read_text().split("---")[1]
        wrapper = (ROOT / ".claude" / "skills" / "replicate-paper" / "SKILL.md").read_text().split("---")[1]
        self.assertEqual(skill, wrapper)
        self.assertEqual((ROOT / "CLAUDE.md").read_text().strip(), "@AGENTS.md")

    def test_schema_and_checker_agree(self):
        schema = json.loads((SKILL / "spec.schema.json").read_text())
        provenance = schema["$defs"]["parameter"]["properties"]["provenance"]["enum"]
        kinds = schema["$defs"]["target"]["properties"]["kind"]["enum"]
        statuses = schema["properties"]["status"]["enum"]
        self.assertEqual((provenance, kinds, statuses), (list(papers.PROVENANCE), list(papers.TARGET_KINDS), list(papers.STATUSES)))

    def test_skill_files_reference_each_other(self):
        text = (SKILL / "SKILL.md").read_text()
        for name in ("lessons.md", "spec.schema.json", "brain-update.md"):
            self.assertIn(name, text)
            self.assertTrue((SKILL / name).is_file())


class PaperAuditTests(unittest.TestCase):
    """The committed paper evidence passes its audit; damaged copies do not."""
    def record(self):
        return json.loads((Path(__file__).parents[1] / "data" / "verification" / "paper_replications.json").read_text())

    def audit(self, record):
        from femm_brain.reproduction_report import DATA, audit
        return audit(record, DATA / "verification" / "reviewed_paper_failures.json", "papers")

    def test_committed_paper_evidence_is_complete_and_within_bounds(self):
        result = self.audit(self.record())
        self.assertEqual(result["status"], "passed", result["problems"])
        self.assertIn(["elrefaie-2006-fscw-spm", "article", "inductance_H@ind"], result["accepted_reviewed_failures"])

    def test_a_reviewed_paper_failure_cannot_drift_and_cases_cannot_vanish(self):
        record = self.record()
        case = next(c for c in record["cases"] if c["case_id"] == "elrefaie-2006-fscw-spm")
        target = next(t for t in case["levels"]["article"]["targets"] if t.get("metric") == "inductance_H@ind" and t["acceptance"])
        target["observed"] = 1.17e-3
        self.assertTrue(any("left the reviewed value" in p for p in self.audit(record)["problems"]))
        record = self.record()
        record["cases"] = [c for c in record["cases"] if c["case_id"] != "elrefaie-2006-fscw-spm"]
        self.assertIn("elrefaie-2006-fscw-spm: missing from the record.", self.audit(record)["problems"])

    def test_a_reviewed_failure_is_bound_to_its_frozen_tolerances(self):
        registry = json.loads((Path(__file__).parents[1] / "data" / "verification" / "reviewed_paper_failures.json").read_text())
        registry["failures"][0]["metrics"][0]["absolute_tolerance"] = 1e-5
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reviewed.json"
            path.write_text(json.dumps(registry))
            from femm_brain.reproduction_report import audit
            problems = audit(self.record(), path, "papers")["problems"]
        self.assertTrue(any("frozen target differs from the reviewed one" in p for p in problems))

    def test_a_case_that_newly_passes_or_fails_differently_is_caught(self):
        record = self.record()
        case = next(c for c in record["cases"] if c["case_id"] == "elrefaie-2006-fscw-spm")
        flux = next(t for t in case["levels"]["article"]["targets"] if t.get("metric") == "flux_A_Wb:fundamental_rms")
        flux["status"] = "failed"
        self.assertTrue(any("unreviewed failure" in p for p in self.audit(record)["problems"]))


class RenderTests(unittest.TestCase):
    def test_render_draws_contours_labels_and_magnet_directions(self):
        svg = papers.render_case(minimal_case())
        self.assertTrue(svg.startswith("<svg") and svg.endswith("</svg>"))
        self.assertEqual(svg.count("<path"), 4)
        self.assertIn(">Air<", svg)
        self.assertIn('stroke="#c62828"', svg)


if __name__ == "__main__":
    unittest.main()
