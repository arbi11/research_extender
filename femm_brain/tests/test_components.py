"""Reusable components: catalog/implementation agreement, safe assemblies and evidence binding."""
import hashlib
import inspect
import json
import unittest

from femm_brain.article_recipes import build
from femm_brain.components import IMPLEMENTATIONS, assemble, catalog, evaluate
from femm_brain.core import load_graph
from femm_brain.reproduce import load_cases, load_designs, plan_level

# Typed-model digests of the article builds that passed natively. A change here
# changes the solved model: update only deliberately and re-run the evidence.
ARTICLE_MODELS = {
    "magneticstutorial-coil": "ee4039ed7e118c55710dfb5c9d9bbd4b2cdc7451da252c8e8054bb2b625a7f42",
    "electrostatics-square-capacitor": "20c87fef8a2538a5c8a4f2261c98f2e87173ef46ce4fc3c5006fc3c996643f60",
    "electrostatics-two-spheres": "1a101c8229e4b97f65bf5ad829df92852a7a21221fb1576f172288c705d274c3",
    "tube-eddy-current": "58a6ba688d110f4568c88f8dbdd0f6e162d297d013bfebc0bd7c4755353dfe54",
    "torque-benchmark": "855b800fbb386e24c3aae766706fefd37099f2ed72158d6c71eca7d1247259b1",
    "heatflow-chimney": "9386bc6448f6436d22f9ccfe02ba9027851436ee9491637fbf25fe8dfc85287d",
}
CONTOUR = {"boundary", "conductor", "hidden", "group"}


class CatalogTests(unittest.TestCase):
    def test_every_record_has_an_implementation_with_matching_parameters(self):
        records = catalog()
        self.assertEqual(set(records), set(IMPLEMENTATIONS))
        for identifier, record in records.items():
            signature = inspect.signature(IMPLEMENTATIONS[identifier]).parameters
            accepts_contour = any(p.kind is p.VAR_KEYWORD for p in signature.values())
            declared = {p["name"] for p in record["parameters"]}
            explicit = set(signature) - {"m", "options"}
            self.assertEqual(declared - (CONTOUR if accepts_contour else set()), explicit, identifier)
            self.assertTrue(record["source_ids"] and record["physics"] and record["problem_types"], identifier)
            for used in record["uses_components"]:
                self.assertIn(used, records, identifier)

    def test_methods_bind_to_existing_graph_nodes_and_operations_get_their_own(self):
        graph = load_graph()
        nodes = {n["id"]: n for n in graph["nodes"]}
        for identifier, record in catalog().items():
            node = nodes[record["node_id"] or "component:" + identifier]
            self.assertEqual(node["executable_component"]["id"], identifier)
        self.assertEqual(nodes["method:methods_iabc_open_domain"]["executable_component"]["id"], "iabc_open_boundary")


class ExpressionTests(unittest.TestCase):
    def test_arithmetic_and_whitelisted_functions_only(self):
        self.assertAlmostEqual(evaluate("sqrt(a**2+b**2)/2", {"a": 3, "b": 4}), 2.5)
        self.assertAlmostEqual(evaluate("2*eps0/150", {}), 2 * 8.8541878188e-12 / 150)
        for hostile in ("__import__('os').system('x')", "a.__class__", "open('x')", "[1][0]", "lambda: 1",
                        "a if a else b", "'text'", "True", "x"):
            with self.assertRaises((ValueError, SyntaxError)):
                evaluate(hostile, {"a": 1, "b": 2})


class AssemblyTests(unittest.TestCase):
    problem = {"physics": "electrostatics", "units": "millimeters", "type": "planar", "depth": 1}

    def test_assemblies_validate_component_use_before_building(self):
        cases = {
            "unknown component": [{"use": "teleport"}],
            "wrong physics": [{"use": "circuit", "name": "c", "current": 1}],
            "unknown parameter": [{"use": "circle", "radius": 1, "colour": "red"}],
            "missing parameter": [{"use": "circle"}],
            "out of range": [{"use": "arc", "start": [0, 0], "end": [1, 0], "angle": 270}],
            "unknown reference": [{"use": "circle", "radius": "$nope"}],
            "duplicate property": [{"use": "material", "name": "a"}, {"use": "material", "name": "a"}],
        }
        for name, steps in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                assemble({"problem": self.problem, "steps": steps}, {})

    def test_methods_enforce_their_applicability(self):
        magnetic = {"physics": "magnetics", "units": "centimeters", "type": "planar", "frequency": 0, "depth": 1}
        with self.assertRaisesRegex(ValueError, "overlap"):
            assemble({"problem": magnetic, "steps": [{"use": "kelvin_periodic_exterior", "domain_radius": 1,
                                                        "exterior_center": [1.5, 0], "applied_field": 1}]}, {})
        with self.assertRaisesRegex(ValueError, "does not apply"):
            assemble({"problem": magnetic, "steps": [{"use": "axisymmetric_coil_section", "inner_radius": 1, "outer_radius": 2,
                                                        "z_min": 0, "z_max": 1, "material": "a", "circuit": "c", "turns": 1}]}, {})

    def test_article_builds_are_data_only_and_frozen(self):
        built = {}
        for case in load_cases():
            if "assembly" in (case.get("article") or {}):
                model = build(case)["model"]
                built[case["id"]] = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
        self.assertEqual(built, ARTICLE_MODELS)

    def test_designs_plan_and_record_the_components_they_use(self):
        for design in load_designs():
            plan = plan_level(design, "article")
            self.assertTrue(plan["components_used"])
            self.assertTrue(design["targets"] or design.get("assessors"))
            self.assertTrue(all(t["kind"] == "analytic" for t in design["targets"]))


class EvidenceTests(unittest.TestCase):
    def test_components_are_checked_only_by_passing_assemblies(self):
        graph = load_graph()
        nodes = {n["id"]: n for n in graph["nodes"]}
        for identifier, record in catalog().items():
            verification = nodes[record["node_id"] or "component:" + identifier]["verification"]
            if verification["status"] == "numerically_checked" and "case_ids" in verification:
                self.assertTrue(verification["reports"])
                self.assertTrue(all(report.startswith("verification/") for report in verification["reports"]))


if __name__ == "__main__":
    unittest.main()
