"""Official-example reproduction: typed models, frozen catalog and closed-form references."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from femm_brain import collect
from femm_brain.article_recipes import build as build_article
from femm_brain.core import load_graph
from femm_brain.geometry import parse_fem
from femm_brain.reference_models import bessel_i, bessel_k, run_assessor, tube_solution
from femm_brain.reproduce import (LEVELS, assess, compare_levels, compare_models, compile_model, compute_metrics,
                                  load_cases, normalize, plan_level, solve_lines, _source_record)

DATA = Path(__file__).parents[1] / "data"
TUTORIAL = "4242c4212516f11219b56dc797306780b8aa033d1813969c1864baea651711e7"
CAPACITOR = "e30650fb10a04caaa5f5e18fb0d4f20b8c0dd572e4edbde8273346bd43f77d0e"


def stored(identifier):
    return json.loads((DATA / "models" / (identifier + ".json")).read_text())


class TypedModelTests(unittest.TestCase):
    def test_every_supported_stored_model_normalizes_and_compiles(self):
        compiled, unsupported = 0, []
        for path in sorted((DATA / "models").glob("*.json")):
            record = json.loads(path.read_text())
            try:
                model = normalize(record)
            except ValueError as exc:
                unsupported.append(str(exc))
                continue
            script = "\n".join(compile_model(model))
            self.assertNotIn("os.execute", script)
            self.assertEqual(compare_models(model, deepcopy(model)), [])
            compiled += 1
        # Unsupported features must be named, never silently dropped.
        self.assertGreaterEqual(compiled, 60)
        self.assertTrue(all("not supported" in u or "Unsupported" in u for u in unsupported), unsupported)

    def test_stored_scalar_models_keep_conductors_and_magnetics_records_are_unchanged(self):
        capacitor = stored(CAPACITOR)
        self.assertEqual([c["name"] for c in capacitor["conductors"]], ["zero", "one"])
        model = normalize(capacitor)
        self.assertEqual({s["conductor"] for s in model["segments"]}, {"zero", "one", None})
        self.assertNotIn("conductors", stored(TUTORIAL))
        script = "\n".join(compile_model(model))
        self.assertIn('ei_addconductorprop("one",1,0,1)', script)

    def test_source_operations_are_ignored_and_names_cannot_inject_lua(self):
        record = stored(TUTORIAL)
        record["operations"] = [{"function": "os.execute", "arguments": ["rm -rf /"]}]
        self.assertNotIn("os.execute", "\n".join(compile_model(normalize(record))))
        record["materials"][0]["name"] = 'Air") os.execute("x'
        with self.assertRaisesRegex(ValueError, "Unsupported property name"):
            compile_model(normalize(record))

    def test_model_comparison_detects_physical_and_topological_changes(self):
        model = normalize(stored(TUTORIAL))
        changes = {
            "materials": lambda m: m["materials"][1]["props"].__setitem__("Sigma", 57),
            "circuit": lambda m: next(l for l in m["labels"] if l["circuit"]).__setitem__("circuit", None),
            "turns": lambda m: next(l for l in m["labels"] if l["turns"] == 1000).__setitem__("turns", 999),
            "boundary": lambda m: next(a for a in m["arcs"] if a["boundary"]).__setitem__("boundary", None),
            "node": lambda m: m["nodes"][0].__setitem__("x", m["nodes"][0]["x"] + 1e-3),
        }
        for name, change in changes.items():
            altered = deepcopy(model)
            change(altered)
            self.assertTrue(compare_models(model, altered), name)

    def test_heat_radiation_ambient_follows_the_engine_backward_compatibility_rule(self):
        text = Path(DATA / "models" / "c2df355f3cf84ff210e435991b4e65915e6837a7678546ba7da4ddfc3f56c856.json").read_text()
        model = normalize(json.loads(text))
        for boundary in model["boundaries"]:
            self.assertEqual(boundary["props"]["TinfRad"], boundary["props"]["Tinf"])


class CatalogTests(unittest.TestCase):
    def test_every_case_is_bound_to_graph_sources_and_frozen_targets(self):
        graph = load_graph()
        examples = {e["id"] for e in graph["examples"]}
        sources = {s["id"] for s in graph["sources"]}
        for case in load_cases():
            self.assertIn(case["example_id"], examples, case["id"])
            self.assertTrue(set(case["source_ids"]) <= sources, case["id"])
            ids = {m["id"] for m in case["measurements"]}
            for metric in case["metrics"]:
                self.assertIn(metric["value"][0].split(":")[0], ids, case["id"])
            for target in case["targets"]:
                self.assertTrue(target["quote"] and target["kind"], target)
                self.assertTrue(target.get("relative_tolerance") or target.get("absolute_tolerance"), target)
            for parameter in (case.get("article") or {}).get("parameters", {}).values():
                self.assertTrue(parameter["quote"], case["id"])

    @unittest.skipUnless((collect.DEFAULT_CACHE / "texts").is_dir(), "requires a restored source cache")
    def test_every_quote_is_verbatim_in_the_cited_official_text(self):
        squash = lambda text: " ".join(text.replace(" ", " ").split()).lower()
        for case in load_cases():
            texts = []
            for source in case["source_ids"]:
                path = collect.DEFAULT_CACHE / "texts" / (source + ".txt")
                if path.is_file():
                    texts.append(squash(path.read_text(errors="replace")))
            quoted = [t for t in case["targets"] if t.get("kind") != "analytic" and not t.get("quote_file")]
            quoted += [p for p in (case.get("article") or {}).get("parameters", {}).values() if not p["quote"].startswith(("Not stated", "FEMM "))]
            for item in quoted:
                quote = item["quote"].replace("...", "…")
                parts = [squash(part) for part in quote.split("…") if part.strip()]
                self.assertTrue(any(all(part in text for part in parts) for text in texts), (case["id"], item["quote"]))

    def test_every_level_plans_without_execution(self):
        for case in load_cases():
            for level in LEVELS:
                plan = plan_level(case, level)
                if plan.get("status"):
                    self.assertIn(plan["status"], ("not_applicable", "not_attempted"))
                    self.assertTrue(plan["reason"])
                    continue
                self.assertEqual(len(plan["plan_sha256"]), 64)
                self.assertIn("raw.csv", plan["lua"])
                self.assertFalse(plan["requires_ai"])

    def test_steps_measure_after_each_modification_and_metrics_are_suffixed(self):
        case = next(c for c in load_cases() if c["id"] == "electrostatics-two-spheres")
        lines = solve_lines(case, "electrostatics", normalize(stored(case["source_model_sha256"]))["labels"])
        self.assertLess(lines.index('ei_modifyboundprop("open_bc",5,0)'), max(i for i, l in enumerate(lines) if "grounded" in l))
        raw = {(s, i, k): complex(k + 1, 0) for s in ("open", "grounded") for i in ("sphere", "force", "mesh_nodes", "mesh_elements") for k in range(3)}
        metrics = compute_metrics(case, raw)
        self.assertEqual(set(metrics) >= {"charge_C@open", "charge_C@grounded", "force_z_N@open"}, True)
        self.assertEqual(assess({"metric": "charge_C@open", "value": 2.0, "relative_tolerance": .01}, metrics)["status"], "passed")
        self.assertEqual(assess({"metric": "charge_C@open", "value": 3.0, "relative_tolerance": .01}, metrics)["status"], "failed")

    def test_rebuild_comparison_covers_every_raw_value(self):
        case = load_cases()[0]
        original = {"status": "passed", "metrics": {"x": 1.0}, "raw": {"base|a|0": [1.0, 0.0], "base|a|1": [1e-9, 0.0]}}
        same = deepcopy(original)
        self.assertEqual(compare_levels(case, {"original": original, "rebuild": same})["rebuild"]["status"], "passed")
        changed = deepcopy(original); changed["raw"]["base|a|1"] = [2e-6, 0.0]
        self.assertEqual(compare_levels(case, {"original": original, "rebuild": changed})["rebuild"]["status"], "failed")
        missing = deepcopy(original); del missing["raw"]["base|a|1"]
        self.assertEqual(compare_levels(case, {"original": original, "rebuild": missing})["rebuild"]["status"], "failed")


class ArticleRecipeTests(unittest.TestCase):
    def case(self, identifier):
        return next(c for c in load_cases() if c["id"] == identifier)

    def test_open_boundary_reproduces_femm_builder_shells_in_the_official_tutorial(self):
        article = build_article(self.case("magneticstutorial-coil"))["model"]
        original = normalize(stored(TUTORIAL))
        shells = lambda m: {x["name"]: x["props"]["Mu_x"] for x in m["materials"] if x["name"].startswith("u")}
        self.assertEqual(shells(article), shells(original))
        radii = lambda m: sorted({round(math.hypot(m["nodes"][a["n0"]]["x"], m["nodes"][a["n0"]]["y"]), 9) for a in m["arcs"]})
        self.assertEqual(radii(article), radii(original))

    def test_article_builds_never_read_the_official_model(self):
        with mock.patch("femm_brain.reproduce._source_record", side_effect=AssertionError("model file read")):
            for case in load_cases():
                result = build_article(case)
                self.assertIn(result["status"], ("built", "not_attempted"))

    def test_article_builds_stay_close_to_the_official_files_they_never_read(self):
        for identifier, limit in (("electrostatics-square-capacitor", 2), ("electrostatics-two-spheres", 2)):
            case = self.case(identifier)
            differences = compare_models(_source_record(case)[1], build_article(case)["model"])
            self.assertLessEqual(len(differences), limit, differences)


class ReferenceModelTests(unittest.TestCase):
    def test_modified_bessel_functions_match_tabulated_values(self):
        # Reference values from an independent implementation (scipy.special.iv/kv).
        z = 3.5 + 3.5j
        cases = [(bessel_i(0, z), complex(-6.036104722145139, 0.328136627303097)),
                 (bessel_i(1, z), complex(-5.629757409306275, -0.15234524269064542)),
                 (bessel_k(0, z), complex(-0.012394298561465279, 0.01121404845651636)),
                 (bessel_k(1, z), complex(-0.012522171517131277, 0.012840068774915312))]
        for value, expected in cases:
            self.assertLess(abs(value - expected) / abs(expected), 1e-9)

    def test_tube_solution_matches_article_constants(self):
        s = tube_solution(.5 * .0254, .0254, 1000, 1e7, 1, 100)
        self.assertAlmostEqual(s["skin_depth_m"], .00503292, places=8)
        self.assertLess(abs(s["c0"] - complex(-2.548, 30.431)) / abs(s["c0"]), 1e-4)
        self.assertAlmostEqual(abs(s["c1"]), abs(complex(22724.439, 2858.776)), delta=1e-3)
        # Ampere's law: the enclosed current at both pipe surfaces is the wire current.
        for r in (.5 * .0254, .0254):
            self.assertAlmostEqual(abs(s["c0"] * bessel_i(1, s["alpha"] * r) + s["c1"] * bessel_k(1, s["alpha"] * r)),
                                   100 / (2 * math.pi * r), places=6)

    def test_dipole_assessor_uses_the_measured_applied_field_and_rotor_convention(self):
        case = next(c for c in load_cases() if c["id"] == "torque-benchmark")
        assessor = case["assessors"][0]
        model = {"labels": [{"material": "magnet", "magdir": 0.0}]}
        for field_angle in (0, 180):
            bx, by = math.cos(math.radians(field_angle)), math.sin(math.radians(field_angle))
            metrics = {"center_bx_T@field": bx, "center_by_T@field": by}
            for step, angle in assessor["rotation_steps"].items():
                metrics["torque_Nm@" + step] = math.sin(math.radians(field_angle - angle))
            self.assertEqual(run_assessor(assessor, case, {}, dict(metrics), model)["status"], "passed")
            metrics["torque_Nm@deg30"] *= -1
            self.assertEqual(run_assessor(assessor, case, {}, dict(metrics), model)["status"], "failed")


class RestoreTests(unittest.TestCase):
    def test_restore_accepts_only_recorded_bytes_and_ignores_taskrunner_timestamps(self):
        page = b'<html><body>Example text<img src="/doku/lib/exe/taskrunner.php?id=x&amp;1792914100"></body></html>'
        later = page.replace(b"1792914100", b"1792914999")
        self.assertEqual(collect.content_digest(page), collect.content_digest(later))
        good, bad = b"model bytes", b"changed bytes"
        import hashlib
        manifest = {"sources": [{"id": "x", "url": "https://femm.info/doku/doku.php?id=x", "status": "fetched",
                                 "sha256": hashlib.sha256(page).hexdigest(), "content_sha256": collect.content_digest(page),
                                 "text_cache_key": "texts/x.txt"}],
                    "artifacts": [{"id": "asset:a", "url": "https://femm.info/a.fem", "status": "fetched", "filename": "a.fem",
                                   "sha256": hashlib.sha256(good).hexdigest(), "source_ids": ["x"]},
                                  {"id": "asset:b", "url": "https://femm.info/b.fem", "status": "fetched", "filename": "b.fem",
                                   "sha256": hashlib.sha256(good + b"!").hexdigest(), "source_ids": ["x"]}]}
        responses = {"https://femm.info/doku/doku.php?id=x": later, "https://femm.info/a.fem": good, "https://femm.info/b.fem": bad}
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
                collect, "fetch", side_effect=lambda url, **_: (responses[url], "text/html" if "doku" in url else "application/octet-stream", url)):
            result = collect.restore(manifest, cache=temp, workers=1)
            self.assertEqual(result["pages"], {"restored": 1})
            self.assertEqual(result["artifacts"], {"changed": 1, "restored": 1})
            self.assertTrue((Path(temp) / "objects" / manifest["artifacts"][0]["sha256"]).is_file())
            self.assertFalse((Path(temp) / "objects" / hashlib.sha256(bad).hexdigest()).exists())


if __name__ == "__main__":
    unittest.main()
