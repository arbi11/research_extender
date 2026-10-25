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
            # A component parameter is either quoted or names the model entity it was identified from.
            for name, parameter in (case.get("components") or {}).get("parameters", {}).items():
                self.assertTrue(parameter.get("quote") or parameter.get("identified_from"), (case["id"], name))

    def test_quotes_cite_portable_cache_entries(self):
        for case in load_cases():
            text = json.dumps(case)
            self.assertNotIn("quote_file", text, case["id"])
            self.assertNotIn("/home/", text, case["id"])

    @unittest.skipUnless((collect.DEFAULT_CACHE / "texts").is_dir(), "requires a restored source cache")
    def test_every_quote_is_verbatim_in_the_cited_official_text(self):
        squash = lambda text: " ".join(text.replace("\xa0", " ").split()).lower()
        read = lambda path: squash(path.read_text(errors="replace"))
        for case in load_cases():
            texts = [read(path) for source in case["source_ids"]
                     if (path := collect.DEFAULT_CACHE / "texts" / (source + ".txt")).is_file()]
            quoted = [t for t in case["targets"] if t.get("kind") != "analytic"]
            quoted += [p for p in (case.get("article") or {}).get("parameters", {}).values() if not p["quote"].startswith(("Not stated", "FEMM "))]
            quoted += [p for p in (case.get("components") or {}).get("parameters", {}).values() if p.get("quote")]
            for item in quoted:
                cited = texts
                if item.get("quote_artifact"):
                    cited = [read(collect.DEFAULT_CACHE / "artifact_texts" / (item["quote_artifact"] + ".txt"))]
                elif item.get("quote_source"):
                    cited = [read(collect.DEFAULT_CACHE / "texts" / (item["quote_source"] + ".txt"))]
                quote = item["quote"].replace("...", "…")
                parts = [squash(part) for part in quote.split("…") if part.strip()]
                self.assertTrue(any(all(part in text for part in parts) for text in cited), (case["id"], item["quote"]))

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

    def test_frequency_steps_complex_values_and_circle_samples(self):
        case = {"id": "toy", "steps": [
                    {"label": "a", "modify": [{"function": "modifycircprop", "name": "C", "property": 1, "value": [3, -4]}]},
                    {"label": "b", "modify": [{"function": "set_frequency", "value": 120}]}],
                "measurements": [{"id": "ring", "op": "circle_values", "radius": 2, "count": 4, "fields": ["A", "B1"]},
                                 {"id": "core", "op": "circle_values", "radius": 1, "count": 3, "fields": ["Je"], "steps": ["b"]}]}
        lines = solve_lines(case, "magnetics", [])
        self.assertIn('mi_modifycircprop("C",1,(3+I*(-4)))', lines)
        self.assertIn("mi_probdef(120)", lines)
        tables = [line for line in lines if line.startswith("circle_x_0=")]
        self.assertEqual(len(tables), 1)
        self.assertEqual([float(v) for v in tables[0][len("circle_x_0={"):-1].split(",")][:2], [2.0, 1.2246467991473532e-16])
        loops = [line for line in lines if line.startswith("for k=1,")]
        # The ring is read at both steps, the core only at the step it names.
        self.assertEqual(len(loops), 3)
        self.assertTrue(any('emit("b","core:Je",k-1,p7)' in line for line in loops))
        self.assertFalse(any('"a","core:Je"' in line for line in loops))
        for bad in ([{"function": "set_frequency", "value": -1}], [{"function": "modifycircprop", "name": "C", "property": 1, "value": [1]}]):
            with self.assertRaises(ValueError):
                solve_lines(dict(case, steps=[{"label": "x", "modify": bad}]), "magnetics", [])
        with self.assertRaises(ValueError):
            solve_lines(dict(case, steps=[{"label": "x", "modify": [{"function": "set_frequency", "value": 5}]}]), "electrostatics", [])

    def test_step_values_can_follow_the_level_parameters(self):
        case = {"id": "toy", "measurements": [], "steps": [{"label": "a", "modify": [
            {"function": "set_frequency", "value": "=abs(f-3*w/(2*pi))"},
            {"function": "modifycircprop", "name": "C", "property": 1, "value": ["=i0*cos(radians(-120))", "=i0*sin(radians(-120))"]}]}]}
        lines = solve_lines(case, "magnetics", [], {"f": 60, "w": 2 * math.pi * 10, "i0": 2})
        self.assertTrue(any(line.startswith("mi_probdef(30.0000000000000") for line in lines))
        self.assertTrue(any(line.startswith('mi_modifycircprop("C",1,(-0.99999999999999') and "I*(-1.73205080756887" in line for line in lines))
        with self.assertRaisesRegex(ValueError, "parameter values"):
            solve_lines(case, "magnetics", [])

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

    def test_dipole_assessor_uses_the_predicted_applied_field_and_rotor_convention(self):
        case = next(c for c in load_cases() if c["id"] == "torque-benchmark")
        assessor = case["assessors"][0]
        for field_angle in (0, 180):
            # The exterior magnetized at -angle predicts a field at +angle (mirror about x).
            model = {"labels": [{"material": "magnet", "magdir": 0.0}, {"material": "Ext", "magdir": -field_angle}],
                     "materials": [{"name": "Ext", "props": {"H_c": 2 / (4e-7 * math.pi)}}]}
            bx, by = math.cos(math.radians(field_angle)), math.sin(math.radians(field_angle))
            metrics = {"center_bx_T@field": bx, "center_by_T@field": by}
            for step, angle in assessor["rotation_steps"].items():
                metrics["torque_Nm@" + step] = math.sin(math.radians(field_angle - angle))
            self.assertEqual(run_assessor(assessor, case, {}, dict(metrics), model)["status"], "passed")
            metrics["torque_Nm@deg30"] *= -1
            self.assertEqual(run_assessor(assessor, case, {}, dict(metrics), model)["status"], "failed")


class AuditTests(unittest.TestCase):
    """Runs the audit on the committed evidence, then on deliberately damaged copies."""
    registry = DATA / "verification" / "reviewed_reproduction_failures.json"

    def record(self, name="official_reproductions.json"):
        return json.loads((DATA / "verification" / name).read_text())

    def audit(self, record, catalog="reproductions", registry=None):
        from femm_brain.reproduction_report import audit
        default = self.registry if catalog == "reproductions" else DATA / "verification" / "reviewed_design_failures.json"
        return audit(record, registry or default, catalog)

    def test_committed_evidence_is_complete_current_and_within_bounds(self):
        self.assertEqual(self.audit(self.record())["status"], "passed", self.audit(self.record())["problems"])
        designs = self.record("component_designs.json")
        self.assertEqual(self.audit(designs, "designs")["status"], "passed")

    def test_missing_cases_levels_or_an_empty_record_fail(self):
        record = self.record()
        record["cases"] = [c for c in record["cases"] if c["case_id"] != "magneticstutorial-coil"]
        self.assertIn("magneticstutorial-coil: missing from the record.", self.audit(record)["problems"])
        record = self.record()
        del next(c for c in record["cases"] if c["case_id"] == "tube-eddy-current")["levels"]["article"]
        self.assertEqual(self.audit(record)["status"], "failed")
        empty = dict(self.record("component_designs.json"), cases=[])
        self.assertEqual(self.audit(empty, "designs")["status"], "failed")

    def test_reviewed_failures_are_bounded_and_bound(self):
        record = self.record()
        ei = next(c for c in record["cases"] if c["case_id"] == "inductanceexample-dc")
        target = next(t for t in ei["levels"]["original"]["targets"] if t["metric"] == "inductance_H" and t["acceptance"])
        target["observed"] = 1000.0
        self.assertTrue(any("left the reviewed value" in p for p in self.audit(record)["problems"]))
        registry = json.loads(self.registry.read_text())
        registry["failures"][0]["metrics"][0]["target"] = 0.00175
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "registry.json"
            path.write_text(json.dumps(registry))
            self.assertTrue(any("frozen target differs" in p for p in self.audit(self.record(), registry=path)["problems"]))

    def test_rebuild_agreement_and_current_catalog_are_mandatory(self):
        record = self.record()
        next(c for c in record["cases"] if c["case_id"] == "woofer-kelvin-centered")["rebuild_matches_original"]["status"] = "failed"
        self.assertTrue(any("rebuild does not reproduce" in p for p in self.audit(record)["problems"]))
        record = self.record()
        record["catalog_sha256"] = "0" * 64
        self.assertEqual(self.audit(record)["status"], "failed")

    def test_component_rebuilds_must_match_the_official_model(self):
        record = self.record()
        next(c for c in record["cases"] if c["case_id"] == "lrkanalysis-flux-sweep")["components_match_original"]["status"] = "failed"
        self.assertTrue(any("does not agree with the original" in p for p in self.audit(record)["problems"]))
        record = self.record()
        next(c for c in record["cases"] if c["case_id"] == "lrkanalysis-flux-sweep")["levels"]["components"]["reference"] = "failed"
        self.assertTrue(any("region by region" in p for p in self.audit(record)["problems"]))
        record = self.record()
        del next(c for c in record["cases"] if c["case_id"] == "lrkanalysis-flux-sweep")["levels"]["components"]
        self.assertIn("lrkanalysis-flux-sweep/components: missing level.", self.audit(record)["problems"])

    def test_unreviewed_failures_and_resolved_reviews_fail(self):
        record = self.record()
        tube = next(c for c in record["cases"] if c["case_id"] == "heatflow-chimney")
        tube["levels"]["article"]["targets"][0]["status"] = "failed"
        self.assertTrue(any("unreviewed failure" in p for p in self.audit(record)["problems"]))
        record = self.record()
        coilgun = next(c for c in record["cases"] if c["case_id"] == "coilgun-z1p5")
        for level in ("original", "rebuild"):
            for target in coilgun["levels"][level]["targets"]:
                target["status"] = "passed"
        self.assertTrue(any("no longer fails" in p for p in self.audit(record)["problems"]))


class ComponentAgreementTests(unittest.TestCase):
    def report(self, flux, torque):
        raw = {}
        for k in range(61):
            for ident in ("phA", "phB", "phC"):
                raw[f"p{k:02d}|{ident}|2"] = [1e-3 * math.sin(k / 9) * flux, 0.0]
            raw[f"p{k:02d}|torque|0"] = [1e-3 * math.cos(k / 3) * torque, 0.0]
        return {"raw": raw, "metrics": {"flux_A_Wb@p00": 0.0}}

    def test_agreement_binds_acceptance_measurements_and_reports_comparators(self):
        case = next(c for c in load_cases() if c["id"] == "lrkanalysis-flux-sweep")
        original = self.report(1, 1)
        noisy_torque = compare_levels(case, {"original": original, "components": self.report(1.005, 1.5)})["components"]
        self.assertEqual(noisy_torque["status"], "passed")
        self.assertEqual(noisy_torque["bounded_measurements"], ["phA", "phB", "phC"])
        self.assertAlmostEqual(noisy_torque["comparator_differences"]["torque"], 0.5)
        wrong_flux = compare_levels(case, {"original": original, "components": self.report(1.02, 1)})["components"]
        self.assertEqual(wrong_flux["status"], "failed")
        missing = self.report(1, 1)
        del missing["raw"]["p30|phB|2"]
        self.assertEqual(compare_levels(case, {"original": original, "components": missing})["components"]["status"], "failed")


class SourceChangeTests(unittest.TestCase):
    def test_refreshed_markup_is_not_a_change_but_edited_text_is(self):
        import shutil
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "data"
            shutil.copytree(DATA, data)
            baseline = json.loads((data / "curation_manifest.json").read_text())
            manifest = json.loads((data / "source_manifest.json").read_text())
            page = next(s for s in manifest["sources"] if s["id"] == "tubeexample")
            page["sha256"] = "0" * 64  # a refreshed crawl always sees new markup
            page["text_sha256"] = baseline["text_hashes"]["tubeexample"]
            (data / "source_manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(load_graph(data)["source_changes"], [])
            page["text_sha256"] = "1" * 64
            (data / "source_manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(load_graph(data)["source_changes"], ["tubeexample"])


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
