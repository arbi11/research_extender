"""Reusable components: catalog/implementation agreement, safe assemblies and evidence binding."""
import cmath
from copy import deepcopy
import hashlib
import inspect
import json
import math
import shutil
import tempfile
from pathlib import Path
import unittest

from femm_brain.article_recipes import build
from femm_brain.components import DATA as COMPONENT_DATA, IMPLEMENTATIONS, _check_curve, assemble, catalog, evaluate, winding_layout
from femm_brain.core import load_graph
from femm_brain.geometry import arc_geometry
from femm_brain.reference_models import assessor_metric, linkage_prediction, run_assessor
from femm_brain.reproduce import (_source_record, check_regions, compare_models, compare_regions, jitter_coordinates, load_cases,
                                  load_designs, permute_entities, plan_level)

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


class ContractTests(unittest.TestCase):
    magnetic = {"physics": "magnetics", "units": "centimeters", "type": "planar", "frequency": 0, "depth": 1}
    axisymmetric = dict(magnetic, type="axisymmetric")

    def rejects(self, steps, problem=None, pattern=None):
        with self.assertRaisesRegex(ValueError, pattern or ".") as caught:
            assemble({"problem": problem or self.magnetic, "steps": steps}, {})
        return caught

    def test_types_bounds_and_finiteness_are_enforced(self):
        coil = {"use": "axisymmetric_coil_section", "inner_radius": 1, "outer_radius": 2, "z_min": 0, "z_max": 1,
                "material": "a", "circuit": "c", "turns": 10}
        self.rejects([dict(coil, turns=2.5)], self.axisymmetric, "integer")
        self.rejects([dict(coil, turns=0), {"use": "circuit", "name": "c", "current": 1}], self.axisymmetric, "nonzero")
        self.rejects([{"use": "circle", "radius": True}], pattern="must be a number")
        self.rejects([{"use": "circle", "radius": float("nan")}], pattern="must be a number")
        self.rejects([{"use": "circle", "radius": 0}], pattern="range")
        self.rejects([{"use": "circle", "radius": 1, "center": [0, float("inf")]}], pattern="point")
        self.rejects([{"use": "block_label", "at": [0, 0], "material": "air", "turns": 5}], pattern="circuit")
        self.rejects([{"use": "conductor", "name": "v", "prescribed": "both"}],
                     {"physics": "electrostatics", "units": "meters", "type": "planar", "depth": 1}, "one of")

    def test_incompatible_libraries_and_unphysical_materials_are_rejected(self):
        self.rejects([{"use": "library_material", "name": "Brick, Common", "library": "heatlib.dat"}], pattern="heatflow")
        self.rejects([{"use": "material", "name": "x", "values": {"Mu_x": 0}}], pattern="not physical")
        self.rejects([{"use": "material", "name": "x", "values": {"Sigma": -1}}], pattern="not physical")

    def test_degenerate_and_duplicate_geometry_is_rejected(self):
        self.rejects([{"use": "segment", "start": [1, 1], "end": [1, 1]}], pattern="Degenerate")
        self.rejects([{"use": "segment", "start": [0, 0], "end": [1, 0]}, {"use": "segment", "start": [1, 0], "end": [0, 0]}], pattern="Duplicate")
        self.rejects([{"use": "polyline", "points": [[0, 0], [1, 0]]}], pattern="three points")
        self.rejects([{"use": "block_label", "at": [0, 0], "material": "a"}, {"use": "block_label", "at": [0, 0], "material": "a"}], pattern="Duplicate")
        self.rejects([{"use": "circle", "radius": 1}], dict(self.magnetic, depth=0), "depth")


class KelvinRuleTests(unittest.TestCase):
    def model(self, magdir, coercivity):
        return {"labels": [{"material": "Ext", "magdir": magdir}, {"material": "magnet", "magdir": 0.0}],
                "materials": [{"name": "Ext", "props": {"H_c": coercivity}}]}

    def test_predicted_field_mirrors_the_exterior_magnetization(self):
        from femm_brain.reference_models import MU0, kelvin_applied_field
        rule = {"expected_field": {"rule": "kelvin_mirror", "exterior_material": "Ext", "direction_tolerance_deg": 0.5}}
        bx, by, direction = kelvin_applied_field(rule, self.model(70, 2 * 0.8 / MU0))
        self.assertAlmostEqual(direction, -70)
        self.assertAlmostEqual(math.hypot(bx, by), 0.8)

    def test_torque_is_judged_against_the_predicted_field_not_the_measured_one(self):
        from femm_brain.reference_models import MU0, run_assessor
        case = next(c for c in load_designs() if c["id"] == "design-dipole-torque-variant")
        assessor = dict(case["assessors"][0], moment_A_m2=1.0, applied_field_T=1.0)
        model = self.model(30, 2 / MU0)
        wrong = math.radians(30)  # a measured field at +30 instead of the predicted -30 degrees
        metrics = {"center_bx_T@field": math.cos(wrong), "center_by_T@field": math.sin(wrong)}
        for step, angle in assessor["rotation_steps"].items():
            metrics["torque_Nm@" + step] = math.sin(wrong - math.radians(angle))
        result = run_assessor(assessor, case, {}, metrics, model)
        self.assertEqual((result["status"], result["field_direction_check"]), ("failed", False))


def _lrk():
    return next(c for c in load_cases() if c["id"] == "lrkanalysis-flux-sweep")


def _signs(layout):
    return " ".join(f"{phase}{'+' if polarity > 0 else '-'}" for _, (phase, polarity) in sorted(layout.items()))


class WindingTests(unittest.TestCase):
    def test_star_of_slots_reproduces_reference_windings(self):
        # LRK as the official model winds it, and the textbook concentrated windings.
        self.assertEqual(_signs(winding_layout(12, 7, ["A", "B", "C"], phase_sequence="reverse")), "A+ B- C+ A- B+ C-")
        self.assertEqual(_signs(winding_layout(12, 5, ["A", "B", "C"], layers="double")), "A+ A- B- B+ C+ C- A- A+ B+ B- C- C+")
        self.assertEqual(_signs(winding_layout(12, 4, ["A", "B", "C"], layers="double")), " ".join(["A+ B+ C+"] * 4))

    def test_every_accepted_winding_is_balanced_and_in_sequence(self):
        accepted = 0
        for slots in range(3, 49):
            for pairs in range(1, 25):
                for layout, layers in (("tooth_coils", "single"), ("tooth_coils", "double"), ("slot_star", "single")):
                    for sequence, direction in (("forward", 1), ("reverse", -1)):
                        try:
                            winding = winding_layout(slots, pairs, ["A", "B", "C"], layout, layers, sequence)
                        except ValueError:
                            continue
                        accepted += 1
                        phasors = {name: sum(polarity * cmath.exp(1j * math.radians(k * pairs * 360 / slots))
                                             for k, (phase, polarity) in winding.items() if phase == name) for name in "ABC"}
                        magnitudes = [abs(value) for value in phasors.values()]
                        self.assertGreater(min(magnitudes), 1e-9, (slots, pairs, layout, layers))
                        self.assertAlmostEqual(max(magnitudes), min(magnitudes), places=9)
                        for k, name in enumerate("ABC"):
                            angle = math.degrees(cmath.phase(phasors[name] / phasors["A"]))
                            self.assertAlmostEqual((angle - direction * k * 120 + 180) % 360 - 180, 0, places=6,
                                                   msg=(slots, pairs, layout, layers, sequence))
        self.assertGreater(accepted, 500)

    def test_unbalanced_and_ill_posed_windings_are_refused(self):
        with self.assertRaisesRegex(ValueError, "Unbalanced"):
            winding_layout(12, 6, ["A", "B", "C"], layers="double")
        with self.assertRaisesRegex(ValueError, "odd number"):
            winding_layout(12, 5, ["A", "B"])
        with self.assertRaisesRegex(ValueError, "even number of slots"):
            winding_layout(9, 4, ["A", "B", "C"])
        with self.assertRaisesRegex(ValueError, "single-layer"):
            winding_layout(24, 2, ["A", "B", "C"], layout="slot_star", layers="double")
        with self.assertRaisesRegex(ValueError, "forward or reverse"):
            winding_layout(12, 5, ["A", "B", "C"], layers="double", phase_sequence="backward")


class MachineGeometryTests(unittest.TestCase):
    problem = {"physics": "magnetics", "units": "millimeters", "type": "planar", "frequency": 0, "depth": 1}
    stator = {"use": "slotted_stator", "slot_count": 12, "slot_bottom_radius": 45, "tip_radius": 30, "tooth_width": 6,
              "tip_arc_length": 13, "shoe_axial_distance": 33}

    def build(self, steps):
        return assemble({"problem": self.problem, "steps": steps}, {})[0]

    def test_stator_teeth_are_symmetric_and_lie_on_their_radii(self):
        lrk = dict(self.stator, slot_bottom_radius=8.25, tip_radius=12.8, tooth_width=2.6, tip_arc_length=5, shoe_axial_distance=11.8)
        for step, radii in ((self.stator, (30, 45)), (lrk, (12.8, 8.25))):
            model = self.build([step])
            points = [(n["x"], n["y"]) for n in model["nodes"]]
            pitch = math.radians(30)
            for x, y in points:
                turned = (x * math.cos(pitch) - y * math.sin(pitch), x * math.sin(pitch) + y * math.cos(pitch))
                self.assertTrue(any(math.dist(turned, other) < 1e-9 for other in points), (x, y))
            for arc in model["arcs"]:
                a, b = model["nodes"][arc["n0"]], model["nodes"][arc["n1"]]
                radius = arc_geometry((a["x"], a["y"]), (b["x"], b["y"]), arc["angle"])["radius"]
                self.assertTrue(any(math.isclose(radius, r, rel_tol=1e-9) for r in radii), radius)
            half = step["tooth_width"] / 2
            self.assertEqual(sum(1 for x, y in points if math.isclose(abs(y), half) and x > 0), 4)

    profile = {"use": "trapezoidal_slot_stator", "slot_count": 36, "bore_radius": 102.6, "opening_width": 2, "opening_height": 3,
               "slot_width_top": 10, "slot_width_bottom": 14.6, "slot_height": 25.4}

    def test_slot_profiles_are_built_to_their_tabulated_widths(self):
        for step in (self.profile, dict(self.profile, rotor_side="outer", bore_radius=140, slot_bottom="arc", slot_divider=True)):
            model = self.build([step])
            points = [(n["x"], n["y"]) for n in model["nodes"]]
            pitch = math.radians(10)
            for x, y in points:
                turned = (x * math.cos(pitch) - y * math.sin(pitch), x * math.sin(pitch) + y * math.cos(pitch))
                self.assertTrue(any(math.dist(turned, other) < 1e-9 for other in points), (x, y))
            sign = 1 if step.get("rotor_side", "inner") == "inner" else -1
            centre = math.radians(5)  # slot 0, half a pitch after tooth 0
            local = [(x * math.cos(centre) + y * math.sin(centre), -x * math.sin(centre) + y * math.cos(centre)) for x, y in points]
            for x_end, width in ((step["bore_radius"] + sign * 3, 10), (step["bore_radius"] + sign * 28.4, 14.6), (None, 2)):
                if x_end is None:
                    near = [p for p in local if abs(math.hypot(*p) - step["bore_radius"]) < 1e-9 and abs(p[1]) < 5]
                else:
                    near = [p for p in local if abs(p[0] - x_end) < 1e-9 and abs(p[1]) < 10]
                self.assertAlmostEqual(max(p[1] for p in near) - min(p[1] for p in near), width, places=9)

    def test_profiled_slots_hold_a_winding_in_closed_regions(self):
        steps = [{"use": "material", "name": "Iron", "values": {"Mu_x": 1000, "Mu_y": 1000}}, {"use": "material", "name": "Cu"},
                 *({"use": "circuit", "name": name, "current": 0} for name in "ABC"),
                 dict(self.profile, iron_material="Iron"), {"use": "circle", "radius": 140},
                 {"use": "tooth_coil_winding", "slot_count": 36, "pole_pairs": 15, "turns": 18, "material": "Cu",
                  "label_radius": 118.3, "layers": "single"},
                 {"use": "block_label", "at": [102.1, 0], "material": "Cu"}]
        model = self.build(steps)
        self.assertEqual(check_regions(model), [])
        self.assertEqual(sum(1 for label in model["labels"] if label.get("circuit")), 36)

    def test_impossible_slot_profiles_are_refused(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.build([dict(self.profile, slot_width_bottom=30)])
        with self.assertRaisesRegex(ValueError, "not be wider"):
            self.build([dict(self.profile, opening_width=12)])
        with self.assertRaisesRegex(ValueError, "divider"):
            self.build([dict(self.profile, slot_divider=True, close_slot_openings=False)])
        with self.assertRaisesRegex(ValueError, "axis"):
            self.build([dict(self.profile, rotor_side="outer", bore_radius=20)])

    def test_impossible_stators_are_refused(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.build([dict(self.stator, tooth_width=40)])
        with self.assertRaisesRegex(ValueError, "shoe"):
            self.build([dict(self.stator, shoe_axial_distance=29)])
        with self.assertRaisesRegex(ValueError, "divider"):
            self.build([dict(self.stator, slot_divider=True, close_slot_openings=False)])

    def test_double_layers_need_divided_slots_and_labels_need_closed_regions(self):
        steps = [{"use": "material", "name": "Iron", "values": {"Mu_x": 1000, "Mu_y": 1000}}, {"use": "material", "name": "Cu"},
                 *({"use": "circuit", "name": name, "current": 0} for name in "ABC"),
                 dict(self.stator, iron_material="Iron"), {"use": "circle", "radius": 55},
                 {"use": "tooth_coil_winding", "slot_count": 12, "pole_pairs": 5, "turns": 10, "material": "Cu",
                  "label_radius": 39, "layers": "double"}]
        problems = check_regions(self.build(steps))
        self.assertEqual(len(problems), 12)
        self.assertTrue(all("several labels" in problem for problem in problems))
        steps[5] = dict(steps[5], slot_divider=True)
        self.assertEqual(check_regions(self.build(steps)), [])
        outside = check_regions(self.build(steps + [{"use": "block_label", "at": [60, 0], "material": "Cu"}]))
        self.assertEqual(len(outside), 1)
        self.assertIn("outside every closed contour", outside[0])

    def test_surface_magnets_alternate_and_seat_arcs_close_the_core(self):
        magnet = {"use": "material", "name": "N", "values": {"Mu_x": 1.05, "Mu_y": 1.05, "H_c": 900000}}
        rotor = {"use": "surface_magnet_rotor", "pole_count": 10, "magnet_material": "N", "thickness": 3, "magnet_shape": "arc",
                 "inner_radius": 25, "arc_angle": 30, "air_gap_side": "outer", "seat_arcs": True}
        model = self.build([magnet, rotor])
        labels = sorted(model["labels"], key=lambda label: math.atan2(label["y"], label["x"]) % (2 * math.pi))
        self.assertEqual([label["magdir_function"] for label in labels], ["theta", "theta+180"] * 5)
        seat = 0.0
        for arc in model["arcs"]:
            a, b = model["nodes"][arc["n0"]], model["nodes"][arc["n1"]]
            if math.isclose(arc_geometry((a["x"], a["y"]), (b["x"], b["y"]), arc["angle"])["radius"], 25):
                seat += arc["angle"]
        self.assertAlmostEqual(seat, 360)
        parallel = self.build([magnet, dict(rotor, magnetization="parallel", first_pole_polarity="inward")])
        directions = sorted((math.degrees(math.atan2(l["y"], l["x"])) % 360, l["magdir"]) for l in parallel["labels"])
        self.assertEqual([round((d - a) % 360, 9) for a, d in directions], [180.0, 0.0] * 5)
        with self.assertRaisesRegex(ValueError, "Seat arcs"):
            self.build([magnet, dict(rotor, magnet_shape="block", width=5, near_face_distance=25)])
        with self.assertRaisesRegex(ValueError, "below the pole pitch"):
            self.build([magnet, dict(rotor, arc_angle=36)])


class SalientPoleTests(unittest.TestCase):
    problem = {"physics": "magnetics", "units": "millimeters", "type": "planar", "frequency": 0, "depth": 1}
    stator = {"use": "salient_poles", "pole_count": 8, "tip_radius": 48.75, "root_radius": 77.65, "pole_arc": 20.45,
              "coil_start": 49.4, "coil_end": 72.9, "coil_width": 8.8}
    rotor = {"use": "salient_poles", "pole_count": 6, "tip_radius": 48.35, "root_radius": 30.85, "pole_arc": 21.5,
             "root_width": 19.5, "group": 1}

    def build(self, steps):
        return assemble({"problem": self.problem, "steps": steps}, {})[0]

    def machine(self, stator=None, rotor=None, winding=None):
        """An 8/6 switched-reluctance cross-section with four phases."""
        half = 48.75 * math.sin(math.radians(10.225))
        return [{"use": "material", "name": "Iron", "values": {"Mu_x": 1000, "Mu_y": 1000}}, {"use": "material", "name": "Air"},
                *({"use": "circuit", "name": name, "current": 0} for name in "ABCD"),
                dict(self.stator, iron_material="Iron", **(stator or {})), {"use": "circle", "radius": 89.75},
                dict({"use": "salient_pole_winding", "pole_count": 8, "phases": ["A", "B", "C", "D"], "turns": 88, "material": "Air",
                      "label_axial": 61, "label_offset": half + 4.4}, **(winding or {})),
                {"use": "block_label", "at": [60, 24], "material": "Air"},
                dict(self.rotor, iron_material="Iron", **(rotor or {})), {"use": "circle", "radius": 18, "group": 1},
                {"use": "block_label", "at": [0, 0], "material": "Air", "group": 1}]

    def test_poles_are_symmetric_with_faces_and_roots_on_their_radii(self):
        for step, pitch, radii in ((self.stator, 45, (48.75, 77.65)), (self.rotor, 60, (48.35, 30.85))):
            model = self.build([step])
            points = [(n["x"], n["y"]) for n in model["nodes"]]
            turn = math.radians(pitch)
            for x, y in points:
                turned = (x * math.cos(turn) - y * math.sin(turn), x * math.sin(turn) + y * math.cos(turn))
                self.assertTrue(any(math.dist(turned, other) < 1e-9 for other in points), (x, y))
            for arc in model["arcs"]:
                a, b = model["nodes"][arc["n0"]], model["nodes"][arc["n1"]]
                radius = arc_geometry((a["x"], a["y"]), (b["x"], b["y"]), arc["angle"])["radius"]
                self.assertTrue(any(math.isclose(radius, r, rel_tol=1e-9) for r in radii), radius)
            faces = sum(arc["angle"] for arc in model["arcs"] if math.isclose(arc["angle"], step["pole_arc"]))
            self.assertAlmostEqual(faces, step["pole_arc"] * step["pole_count"])
        # Parallel stator sides keep the face's half-width; the tapered rotor reaches the root width at its root corners.
        half = 48.75 * math.sin(math.radians(10.225))
        stator = [(n["x"], n["y"]) for n in self.build([self.stator])["nodes"]]
        self.assertEqual(sum(1 for x, y in stator if x > 0 and math.isclose(abs(y), half, abs_tol=1e-9)), 8)
        rotor = [(n["x"], n["y"]) for n in self.build([self.rotor])["nodes"]]
        self.assertEqual(sum(1 for x, y in rotor if x > 0 and math.isclose(abs(y), 9.75, abs_tol=1e-9)
                             and math.isclose(math.hypot(x, y), 30.85)), 2)

    def test_coil_blocks_share_the_pole_sides_and_every_region_is_closed(self):
        for variant in ({}, {"rotor": {"root_shape": "flat"}}, {"stator": {"root_shape": "flat"}}):
            model = self.build(self.machine(**variant))
            self.assertEqual(check_regions(model), [], variant)
        model = self.build(self.machine())
        coils = [label for label in model["labels"] if label.get("circuit")]
        self.assertEqual(len(coils), 16)
        # Pole 0 (phase A) is positive: +turns clockwise of it; its opposite pole 4 is negative.
        signs = {(label["circuit"], round(math.degrees(math.atan2(label["y"], label["x"])) % 360)): label["turns"] for label in coils}
        self.assertEqual(signs[("A", 348)], 88)
        self.assertEqual(signs[("A", 12)], -88)
        self.assertEqual(signs[("A", 168)], -88)
        self.assertEqual(signs[("A", 192)], 88)
        self.assertEqual({phase for phase, _ in signs}, set("ABCD"))
        for phase in "ABCD":
            self.assertEqual(sum(turns for (name, _), turns in signs.items() if name == phase), 0)
        explicit = self.build(self.machine(winding={"polarities": [1, -1] * 4}))
        flipped = [l for l in explicit["labels"] if l.get("circuit") == "A" and l["x"] < 0 and l["y"] > 0]
        self.assertEqual(flipped[0]["turns"], 88)
        field = self.build([{"use": "material", "name": "Cu"}, {"use": "circuit", "name": "F", "current": 0},
                            {"use": "salient_pole_winding", "pole_count": 4, "phases": ["F"], "turns": 10, "material": "Cu",
                             "label_axial": 20, "label_offset": 5}])
        clockwise = sorted((math.degrees(math.atan2(l["y"], l["x"])) % 360, l["turns"]) for l in field["labels"])
        # Alternate poles are north and south, so both coil sides in each gap carry the same sign.
        self.assertEqual([turns for _, turns in clockwise], [-10, -10, 10, 10, -10, -10, 10, 10])

    def test_impossible_poles_coils_and_windings_are_refused(self):
        with self.assertRaisesRegex(ValueError, "faces overlap"):
            self.build([dict(self.stator, pole_arc=46)])
        with self.assertRaisesRegex(ValueError, "roots overlap"):
            self.build([dict(self.rotor, root_width=40)])
        with self.assertRaisesRegex(ValueError, "between the face and root"):
            self.build([dict(self.stator, coil_end=80)])
        with self.assertRaisesRegex(ValueError, "half of the gap"):
            self.build([dict(self.stator, coil_width=13)])
        with self.assertRaisesRegex(ValueError, "half of the gap"):
            self.build([dict(self.stator, coil_end=77, coil_width=9)])
        with self.assertRaisesRegex(ValueError, "start, an end and a width"):
            self.build([{key: value for key, value in self.stator.items() if key != "coil_width"}])
        with self.assertRaisesRegex(ValueError, "multiple of the phase count"):
            self.build(self.machine(winding={"phases": ["A", "B", "C"]}))
        with self.assertRaisesRegex(ValueError, "one polarity"):
            self.build(self.machine(winding={"polarities": [1, -1]}))
        with self.assertRaisesRegex(ValueError, "middle of the gap"):
            self.build(self.machine(winding={"label_offset": 40}))


class BeltWindingTests(unittest.TestCase):
    problem = {"physics": "magnetics", "units": "meters", "type": "planar", "frequency": 60, "depth": 1}

    def build(self, belts):
        steps = [{"use": "material", "name": "Air"}, *({"use": "circuit", "name": n, "current": 0} for n in "ABC"),
                 {"use": "belt_winding", "inner_radius": 0.032, "outer_radius": 0.052, "belts": belts, "material": "Air",
                  "gap_material": "Air"}]
        return assemble({"problem": self.problem, "steps": steps}, {})[0]

    def test_belts_and_gaps_get_one_closed_region_each(self):
        three = [[-22.5 + 60 * k, 22.5 + 60 * k, "ACB"[k % 3], 1 if k % 2 == 0 else -1] for k in range(6)]
        model = self.build(three)
        self.assertEqual(check_regions(model), [])
        self.assertEqual((len(model["arcs"]), len(model["segments"]), len(model["labels"])), (24, 12, 12))
        signed = sorted((round(math.degrees(math.atan2(l["y"], l["x"])) % 360, 6) % 360, l["circuit"], l["turns"]) for l in model["labels"])
        self.assertEqual([item for item in signed if item[1]], [(0.0, "A", 1), (60.0, "C", -1), (120.0, "B", 1),
                                                                (180.0, "A", -1), (240.0, "C", 1), (300.0, "B", -1)])
        # Two belts leave stretches of 135 degrees: arcs stay below 180 degrees and each gap has one label.
        single = self.build([[-22.5, 22.5, "A", 1], [157.5, 202.5, "A", -1]])
        self.assertEqual(check_regions(single), [])
        self.assertEqual(len(single["labels"]), 4)
        self.assertTrue(all(arc["angle"] <= 180 for arc in single["arcs"]))
        lone = self.build([[0, 45, "A", 1]])
        self.assertEqual(check_regions(lone), [])
        self.assertTrue(all(arc["angle"] <= 180 for arc in lone["arcs"]))

    def test_overlapping_or_empty_belts_are_refused(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.build([[0, 50, "A", 1], [40, 90, "B", 1]])
        with self.assertRaisesRegex(ValueError, "more than zero"):
            self.build([[10, 10, "A", 1]])


class HarmonicSuperpositionTests(unittest.TestCase):
    """Synthetic fields with known answers: one order extracted from a mixture, solved directly or conjugated."""
    model = {"problem": {"units": "meters", "depth": 1}}

    def case(self, conjugate):
        rings = [{"id": f"gap{k}", "op": "circle_values", "radius": r, "count": 64, "fields": ["A"]} for k, r in enumerate((0.0305, 0.0315))]
        rings += [{"id": "wind", "op": "circle_values", "radius": 0.042, "count": 64, "fields": ["A"]},
                  {"id": "core", "op": "circle_values", "radius": 0.01, "count": 64, "fields": ["Je"]}]
        return {"measurements": rings, "steps": [{"label": "s", "conjugate": conjugate,
                                                  "harmonics": [{"sweep": "t", "speed_rad_s": 100, "order": -5}]}]}

    def raw(self, case, conjugate):
        """Order -5 in a Laplace gap plus an unrelated order 3; a conjugated solve stores conj(a) exp(+j m theta)."""
        alpha, beta, order = 0.3 + 0.1j, 2e-9 - 1e-9j, -5
        a = lambda r: alpha * r ** 5 + beta * r ** -5
        raw = {}
        for m in case["measurements"]:
            for k in range(64):
                theta = 2 * math.pi * k / 64
                value = {"wind": 0.002 - 0.001j, "core": 0.5 + 0.25j}.get(m["id"], a(m["radius"]))
                field = "Je" if m["id"] == "core" else "A"
                wave = value * cmath.exp(-1j * order * theta)
                if conjugate:
                    wave = (value * cmath.exp(-1j * order * theta)).conjugate()
                raw[("s", f"{m['id']}:{field}", k)] = wave + 0.7 * cmath.exp(-3j * theta)
        return raw, alpha, beta

    def test_torque_loss_and_voltage_of_one_order(self):
        for conjugate in (False, True):
            case = self.case(conjugate)
            raw, alpha, beta = self.raw(case, conjugate)
            base = {"kind": "harmonic_superposition", "sweep": "t", "speed_rad_s": 100, "supply_frequency_hz": 60, "value": 0,
                    "absolute_tolerance": 1e30}
            r, n = 0.0305, 5
            a = alpha * r ** n + beta * r ** -n
            da = n * (alpha * r ** (n - 1) - beta * r ** (-n - 1))
            expected = math.pi * r ** 2 / (4e-7 * math.pi) * ((5j * a / r) * (-da).conjugate()).real
            torque = run_assessor(dict(base, quantity="torque", gap_circles=["gap0", "gap1"]), case, raw, {}, self.model)
            # The gap fit spans r^5 to r^-5, so agreement is limited to about 1e-8 by rounding.
            self.assertAlmostEqual(torque["observed"] / expected, 1, places=7)
            loss = run_assessor(dict(base, quantity="moving_loss", gap_circles=["gap0", "gap1"]), case, raw, {}, self.model)
            slip = 2 * math.pi * 60 + 5 * 100
            self.assertAlmostEqual(loss["observed"] / (expected * slip / -5), 1, places=7)
            ring = run_assessor(dict(base, quantity="conductor_loss", rings=[{"measurement": "core", "weight": 2e-4}],
                                     conductivity_S_per_m=1e6), case, raw, {}, self.model)
            self.assertAlmostEqual(ring["observed"] / (math.pi / 1e6 * 2e-4 * abs((0.5 + 0.25j) * 1e6) ** 2), 1, places=9)
            winding = {"rings": [{"measurement": "wind", "weight": (0.052 ** 2 - 0.032 ** 2) / 2}], "inner_radius": 0.032,
                       "outer_radius": 0.052, "belts": [[-22.5, 22.5, 1], [157.5, 202.5, -1]]}
            voltage = run_assessor(dict(base, quantity="voltage", winding=winding), case, raw, {}, self.model)
            belt = lambda a0, a1: (cmath.exp(5j * math.radians(a1)) - cmath.exp(5j * math.radians(a0))) / (5j)
            flux = (0.002 - 0.001j) * (belt(-22.5, 22.5) - belt(157.5, 202.5)) / math.radians(45)
            self.assertAlmostEqual(voltage["observed"] / (2 * math.pi * 60 * abs(flux)), 1, places=9)

    def test_missing_orders_are_reported(self):
        case = self.case(False)
        result = run_assessor({"kind": "harmonic_superposition", "sweep": "t", "speed_rad_s": 200, "supply_frequency_hz": 60,
                               "quantity": "torque", "gap_circles": ["gap0", "gap1"], "value": 1}, case, {}, {}, self.model)
        self.assertEqual(result["status"], "failed")
        self.assertIn("No steps serve", result["error"])


class DqLinkageTests(unittest.TestCase):
    def test_park_transform_recovers_dq_flux_and_torque_at_any_angle(self):
        psi_d, psi_q, i_d, i_q = 0.12, -0.05, -100.0, 150.0
        steps, raw = [], {}
        for k, angle in enumerate((0, 37, 200)):
            theta = math.radians(angle)
            label = f"s{k}"
            steps.append({"label": label, "dq": {"point": "P", "id": i_d, "iq": i_q, "electrical_angle_deg": angle}})
            for phase in range(3):
                shift = theta - 2 * math.pi * phase / 3
                raw[(label, "ph" + "ABC"[phase], 2)] = complex(psi_d * math.cos(shift) - psi_q * math.sin(shift), 0)
            raw[(label, "torque", 0)] = complex(-1.5 * 4 * (psi_d * i_q - psi_q * i_d), 0)
        case = {"steps": steps + [{"label": "other", "dq": {"point": "Q", "id": 0, "iq": 0, "electrical_angle_deg": 0}}]}
        base = {"kind": "dq_linkage", "point": "P", "phases": ["phA", "phB", "phC"], "pole_pairs": 4, "value": 0, "absolute_tolerance": 1e9}
        for quantity, expected in (("psi_d", psi_d), ("psi_q", psi_q), ("torque", 1.5 * 4 * (psi_d * i_q - psi_q * i_d))):
            result = run_assessor(dict(base, quantity=quantity), case, raw, {}, {})
            self.assertAlmostEqual(result["observed"], expected, places=12)
            self.assertEqual(result["samples"], 3)
        stress = run_assessor(dict(base, quantity="stress_torque", torque_measurement="torque", sequence_direction=-1), case, raw, {}, {})
        self.assertAlmostEqual(stress["observed"], 1.5 * 4 * (psi_d * i_q - psi_q * i_d), places=9)
        missing = run_assessor(dict(base, point="Z", quantity="psi_d"), case, raw, {}, {})
        self.assertIn("No steps carry", missing["error"])


class PartialModelAndSymmetryTests(unittest.TestCase):
    def test_dq_scales_take_a_sector_to_the_machine(self):
        steps = [{"label": "s0", "dq": {"point": "P", "id": 0, "iq": 10, "electrical_angle_deg": 0}}]
        raw = {("s0", "phA", 2): 0.01 + 0j, ("s0", "phB", 2): -0.005 + 0j, ("s0", "phC", 2): -0.005 + 0j, ("s0", "gap", 0): 7 + 0j}
        base = {"kind": "dq_linkage", "point": "P", "phases": ["phA", "phB", "phC"], "pole_pairs": 4, "value": 0, "absolute_tolerance": 1e9}
        whole = run_assessor(dict(base, quantity="psi_d"), {"steps": steps}, raw, {}, {})["observed"]
        self.assertAlmostEqual(run_assessor(dict(base, quantity="psi_d", flux_scale=8), {"steps": steps}, raw, {}, {})["observed"], 8 * whole)
        stress = dict(base, quantity="stress_torque", torque_measurement="gap", sequence_direction=1)
        self.assertAlmostEqual(run_assessor(dict(stress, torque_scale=1), {"steps": steps}, raw, {}, {})["observed"], 7)

    def test_rotor_angles_from_a_sliding_band_are_absolute(self):
        from femm_brain.reference_models import _rotor_samples
        band = lambda angle: {"function": "modifyboundprop", "name": "AGE", "property": 10, "value": angle}
        steps = [{"label": f"p{k:02d}", "modify": [band(7.5 * k)]} for k in range(24)]
        steps.append({"label": "other", "modify": [{"function": "modifycircprop", "name": "A", "property": 1, "value": 1}]})
        samples, span = _rotor_samples({"rotor_group": 1, "rotor_band": "AGE", "pole_pairs": 2, "step_prefix": "p"}, {"steps": steps})
        self.assertEqual(span, 180)
        self.assertEqual([a for a, _ in samples], [7.5 * k for k in range(24)])
        # rotate_group increments are ignored when a band is named, and summed when not.
        turned = [{"label": f"p{k:02d}", "modify": [{"function": "rotate_group", "group": 1, "angle": 7.5 if k else 0}]} for k in range(24)]
        self.assertEqual([a for a, _ in _rotor_samples({"rotor_group": 1, "pole_pairs": 2}, {"steps": turned})[0]], [7.5 * k for k in range(24)])
        with self.assertRaises(ValueError):
            _rotor_samples({"rotor_group": 1, "rotor_band": "AGE", "pole_pairs": 2}, {"steps": turned})

    def test_metric_relation_reports_the_largest_relative_deviation(self):
        metrics = {"a@0": 2.0, "b@1": 2.002, "a@1": -1.0, "b@2": 1.0}
        assessor = {"kind": "metric_relation", "name": "shift", "value": 0, "absolute_tolerance": 0.002,
                    "pairs": [{"metric": "b@1", "reference": "a@0"}, {"metric": "b@2", "reference": "a@1", "factor": -1}]}
        result = run_assessor(assessor, {}, {}, metrics, {})
        self.assertEqual(result["metric"], "shift")
        self.assertAlmostEqual(result["observed"], 0.001)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(run_assessor(dict(assessor, absolute_tolerance=0.0005), {}, {}, metrics, {})["status"], "failed")
        wrong_sign = dict(assessor, pairs=[{"metric": "b@2", "reference": "a@1"}])
        self.assertAlmostEqual(run_assessor(wrong_sign, {}, {}, metrics, {})["observed"], 2.0)

    def test_bessel_series_satisfies_its_recurrence_at_high_order(self):
        from femm_brain.reference_models import bessel_i
        for z in (0.7, 4 * cmath.exp(0.25j * math.pi), 18 * cmath.exp(-0.25j * math.pi)):
            for n in (1, 5, 13):
                lhs = bessel_i(n - 1, z) - bessel_i(n + 1, z)
                self.assertLess(abs(lhs - 2 * n / z * bessel_i(n, z)), 1e-12 * abs(lhs))
        with self.assertRaises(ValueError):
            bessel_i(-1, 1.0)

    def test_rotating_cylinder_torque_balances_slip_power_and_ohmic_loss(self):
        from femm_brain.reference_models import MU0, bessel_i, rotating_cylinder_torque
        J, a, b, R, sigma, depth = 3.1e6 * cmath.exp(0.3j), 0.032, 0.052, 0.030, 3.72e7, 0.5
        for order in (1, -5, 2):
            n = abs(order)
            spread = math.log(b / a) if n == 2 else (b ** (2 - n) - a ** (2 - n)) / (2 - n)
            alpha = MU0 * J * spread / (2 * n)
            # Low slip: the induced current is -j sigma s alpha r^n, so T = pi L order |alpha|^2 sigma s R^(2n+2) / (2(n+1)).
            low = math.pi * depth * order * abs(alpha) ** 2 * sigma * 1e-3 * R ** (2 * n + 2) / (2 * (n + 1))
            self.assertAlmostEqual(rotating_cylinder_torque(order, J, a, b, R, sigma, 1e-3, depth) / low, 1, places=8)
            for slip in (377.0, -1023.0):
                torque = rotating_cylinder_torque(order, J, a, b, R, sigma, slip, depth)
                self.assertAlmostEqual(rotating_cylinder_torque(order, J, a, b, R, sigma, -slip, depth), -torque, places=12)
                k = cmath.sqrt(1j * MU0 * sigma * slip)
                f = n + k * R * bessel_i(n + 1, k * R) / bessel_i(n, k * R)
                inside = 2 * n * alpha * R ** n / ((n + f) * bessel_i(n, k * R))
                count = 2000
                loss = depth * sum(math.pi / sigma * abs(sigma * slip * inside * bessel_i(n, k * r)) ** 2 * r * R / count
                                   for r in ((i + 0.5) * R / count for i in range(count)))
                self.assertAlmostEqual(torque * slip / order / loss, 1, places=5)
        self.assertEqual(rotating_cylinder_torque(1, J, a, b, R, sigma, 0, depth), 0)


def _dxf(*entities, blocks=()):
    """A minimal ASCII DXF: BLOCKS then ENTITIES, each entity as group-code pairs."""
    lines = ["0", "SECTION", "2", "BLOCKS"]
    for name, items in blocks:
        lines += ["0", "BLOCK", "2", name]
        for item in items:
            lines += [str(v) for v in item]
        lines += ["0", "ENDBLK"]
    lines += ["0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]
    for item in entities:
        lines += [str(v) for v in item]
    return "\n".join(lines + ["0", "ENDSEC", "0", "EOF"]) + "\n"


class DrawingTests(unittest.TestCase):
    def test_dxf_entities_become_primitives(self):
        from femm_brain import dxf
        line = ["0", "LINE", "8", "0", "10", 0, "20", 0, "11", 10, "21", 0]
        arc = ["0", "ARC", "10", 0, "20", 0, "40", 10, "50", 0, "51", 270]
        bulged = ["0", "LWPOLYLINE", "90", 2, "70", 0, "10", 0, "20", 0, "42", 1, "10", 2, "20", 0]
        insert = ["0", "INSERT", "2", "B", "10", 5, "20", 5, "50", 90]
        text = _dxf(line, arc, bulged, insert, blocks=[("B", [["0", "LINE", "10", 0, "20", 0, "11", 1, "21", 0]])])
        prims = dxf.primitives(dxf.read(text), scale=2)
        segments = [p for p in prims if p[0] == "segment"]
        arcs = [p for p in prims if p[0] == "arc"]
        self.assertEqual(segments[0], ["segment", 0, 0, 20, 0])
        # A 270 degree arc is split into two arcs no wider than 180 degrees.
        self.assertEqual([round(a[5], 9) for a in arcs[:2]], [135.0, 135.0])
        # A bulge of 1 is a counterclockwise half circle on the chord from (0, 0) to (2, 0): centre (1, 0).
        half = arcs[2]
        self.assertAlmostEqual(half[5], 180)
        self.assertAlmostEqual(math.hypot(half[1] / 2 - 1, half[2] / 2), 1)
        # The block's line, turned 90 degrees and moved to (5, 5), then scaled by 2.
        self.assertEqual([round(v, 9) for v in segments[-1][1:]], [10, 10, 10, 12])

    def test_patterns_join_copies_snap_noise_and_split_t_junctions(self):
        problem = {"physics": "magnetics", "units": "millimeters", "type": "planar", "frequency": 0, "depth": 1}
        # One quarter: an arc from 0 to 90 degrees, two radial edges, and a stub whose end rests 1e-6 off the middle
        # of the lower edge (a T-junction drawn with noise).
        quarter = [["arc", 10, 0, 0, 10, 90], ["segment", 0, 0, 10, 0], ["segment", 0, 0, 0, 10],
                   ["segment", 5, 1e-6, 5, 3], ["segment", 5, 3, 7, 3], ["segment", 7, 3, 7, 0]]
        model = assemble({"problem": problem, "steps": [
            {"use": "entity_pattern", "entities": quarter, "copies": 4, "angle_step": 90, "snap": 1e-4}]}, {})[0]
        # Shared radial edges are drawn once, and each lower edge is split at the stub's two feet.
        self.assertEqual(len(model["arcs"]), 4)
        lengths = sorted(round(math.dist((model["nodes"][s["n0"]]["x"], model["nodes"][s["n0"]]["y"]),
                                         (model["nodes"][s["n1"]]["x"], model["nodes"][s["n1"]]["y"])), 6) for s in model["segments"])
        self.assertEqual(lengths.count(5.0), 4)
        self.assertEqual(lengths.count(3.0), 8)
        self.assertEqual(sum(1 for length in lengths if length == 2.0), 8)
        # Every stub rectangle and the rest of each quarter close into regions of their own.
        turn = lambda x, y, k: [x * math.cos(math.pi * k / 2) - y * math.sin(math.pi * k / 2),
                                x * math.sin(math.pi * k / 2) + y * math.cos(math.pi * k / 2)]
        labels = [{"use": "block_label", "at": turn(x, y, k), "material": "Air"} for k in range(4) for x, y in ((6, 1.5), (3, 6))]
        labelled = assemble({"problem": problem, "steps": [{"use": "material", "name": "Air"},
            {"use": "entity_pattern", "entities": quarter, "copies": 4, "angle_step": 90, "snap": 1e-4}] + labels}, {})[0]
        self.assertEqual(check_regions(labelled), [])
        bands = assemble({"problem": problem, "steps": [
            {"use": "entity_pattern", "entities": quarter, "copies": 4, "angle_step": 90, "snap": 1e-4,
             "max_segment_bands": [[9, 11, 0.25]]}]}, {})[0]
        self.assertTrue(all(arc["max_segment"] == 0.25 for arc in bands["arcs"]))
        # A minimum piece length coarsens small arcs only: 1 mm on a 10 mm radius is 5.73 degrees.
        coarse = assemble({"problem": problem, "steps": [
            {"use": "entity_pattern", "entities": quarter, "copies": 1, "snap": 1e-4, "max_segment": 1, "min_segment_length": 1}]}, {})[0]
        self.assertAlmostEqual(coarse["arcs"][0]["max_segment"], math.degrees(0.1))
        # The default snap distance (1e-9) on a drawing of millimetres: the split search stays on occupied cells.
        fine = assemble({"problem": problem, "steps": [{"use": "entity_pattern", "entities": quarter, "copies": 4, "angle_step": 90}]}, {})[0]
        self.assertEqual(len(fine["arcs"]), 4)


class AssessorMetricTests(unittest.TestCase):
    def test_names_are_those_the_assessors_report(self):
        self.assertEqual(assessor_metric({"kind": "waveform_statistic", "metric": "flux", "statistic": "mean"}), "flux:mean")
        self.assertEqual(assessor_metric({"kind": "spatial_harmonic", "measurement": "line", "field": "Bx", "order": 3}), "line:Bx:h3")
        self.assertEqual(assessor_metric({"kind": "harmonic_superposition", "sweep": "s", "speed_rad_s": 39.7935, "quantity": "voltage"}),
                         "s@39.7935:voltage:abs")
        self.assertEqual(assessor_metric({"kind": "harmonic_superposition", "sweep": "s", "speed_rad_s": 200, "quantity": "point_field",
                                          "point": {"label": "x0.05"}, "part": "im"}), "s@200:point_field:x0.05:im")
        self.assertEqual(assessor_metric({"kind": "dq_linkage", "point": "p", "quantity": "psi_d"}), "p:psi_d")
        self.assertEqual(assessor_metric({"kind": "dq_linkage", "point": "p", "quantity": "psi_d", "average": "first"}), "p:psi_d:first_position")
        self.assertIsNone(assessor_metric({"kind": "tube_bessel"}))


class SectorAirGapTests(unittest.TestCase):
    problem = {"physics": "magnetics", "units": "millimeters", "type": "planar", "frequency": 0, "depth": 1}

    def sector(self, **overrides):
        step = dict({"use": "sector_air_gap", "rotor_radius": 10, "stator_radius": 11, "start_angle": 0, "end_angle": 45,
                     "band_inner_radius": 10.3, "band_outer_radius": 10.7, "mesh": 0.1, "rotor_group": 1}, **overrides)
        return assemble({"problem": self.problem, "steps": [{"use": "material", "name": "Air"}, step]}, {})[0]

    def test_band_ends_labels_and_hole(self):
        model = self.sector()
        types = {b["name"]: b["props"]["BdryType"] for b in model["boundaries"]}
        self.assertEqual(types, {"AGE": 7, "AGE_rotor_ends": 5, "AGE_stator_ends": 5})
        radius = lambda n: math.hypot(model["nodes"][n]["x"], model["nodes"][n]["y"])
        band = [a for a in model["arcs"] if a["boundary"] == "AGE"]
        self.assertEqual(sorted(round(radius(a["n0"]), 9) for a in band), [10.3, 10.7])
        self.assertTrue(all(a["angle"] == 45 for a in band))
        # Each end: rotor side (in the rotor's group), the band's own end, stator side.
        ends = {s["boundary"]: s for s in model["segments"]}
        self.assertEqual(sum(s["boundary"] == "AGE_rotor_ends" for s in model["segments"]), 2)
        self.assertEqual(ends["AGE_rotor_ends"]["group"], 1)
        self.assertEqual(sum(s["boundary"] is None for s in model["segments"]), 2)
        labels = sorted((math.hypot(l["x"], l["y"]), l["group"]) for l in model["labels"])
        self.assertEqual([(round(r, 9), g) for r, g in labels], [(10.15, 1), (10.85, 0)])
        hole = model["holes"][0]
        self.assertAlmostEqual(math.hypot(hole["x"], hole["y"]), 10.5)
        self.assertAlmostEqual(math.degrees(math.atan2(hole["y"], hole["x"])), 22.5)

    def test_periodic_and_invalid_sectors(self):
        types = {b["name"]: b["props"]["BdryType"] for b in self.sector(antiperiodic=False)["boundaries"]}
        self.assertEqual(types, {"AGE": 6, "AGE_rotor_ends": 4, "AGE_stator_ends": 4})
        for bad in ({"band_inner_radius": 9.9}, {"band_outer_radius": 11.2}, {"end_angle": 200}, {"end_angle": 0}):
            with self.assertRaises(ValueError):
                self.sector(**bad)

    def test_arc_boundary_picks_one_arc(self):
        steps = [{"use": "fixed_potential_boundary", "name": "A=0"},
                 {"use": "entity_pattern", "entities": [["arc", 10, 0, 0, 10, 90], ["arc", 0, 10, -10, 0, 90]], "snap": 1e-6}]
        on_second = [[10 * math.cos(math.radians(135)), 10 * math.sin(math.radians(135))]]
        model = assemble({"problem": self.problem, "steps": steps + [{"use": "arc_boundary", "name": "A=0", "at": on_second}]}, {})[0]
        self.assertEqual([a["boundary"] for a in model["arcs"]], [None, "A=0"])
        for at, name in (([[10, 10]], "A=0"), ([[0, 10]], "A=0"), (on_second, "undefined")):
            with self.assertRaises(ValueError):
                assemble({"problem": self.problem, "steps": steps + [{"use": "arc_boundary", "name": name, "at": at}]}, {})


class LinearMachineTests(unittest.TestCase):
    problem = {"physics": "magnetics", "units": "millimeters", "type": "axisymmetric", "frequency": 0, "depth": 1}
    array = {"use": "magnet_array", "start": 0, "inner": 19.5, "outer": 24.5, "magnet_material": "M",
             "segments": [[2, 90], [6, 0], [4, 270], [6, 180], [2, 90]]}

    def build(self, steps):
        base = [{"use": "material", "name": "M", "values": {"Mu_x": 1.05, "Mu_y": 1.05, "H_c": 871576}},
                {"use": "boundary", "name": "per", "values": {"BdryType": 4}}]
        return assemble({"problem": self.problem, "steps": base + steps}, {})[0]

    def test_magnet_blocks_share_edges_and_keep_their_directions(self):
        model = self.build([self.array])
        self.assertEqual([label["magdir"] for label in model["labels"]], [90, 0, 270, 180, 90])
        self.assertEqual(len(model["segments"]), 2 * 5 + 6)
        self.assertEqual(len(self.build([dict(self.array, draw_ends=False)])["segments"]), 2 * 5 + 4)
        planar = assemble({"problem": {"physics": "magnetics", "units": "millimeters", "type": "planar", "frequency": 0, "depth": 1},
                           "steps": [{"use": "material", "name": "M"}, dict(self.array, axis="x")]}, {})[0]
        self.assertEqual(sorted({round(n["y"], 9) for n in planar["nodes"]}), [19.5, 24.5])
        with self.assertRaisesRegex(ValueError, "positive length"):
            self.build([dict(self.array, segments=[[0, 90]])])
        with self.assertRaisesRegex(ValueError, "nonnegative radii"):
            self.build([dict(self.array, inner=-1)])

    def test_boundaries_go_to_the_segment_through_each_point(self):
        model = self.build([self.array, {"use": "segment_boundary", "name": "per", "at": [[22, 0], [22, 20]]}])
        ends = sorted((round(model["nodes"][s["n0"]]["y"], 9), s["boundary"]) for s in model["segments"] if s["boundary"])
        self.assertEqual(ends, [(0.0, "per"), (20.0, "per")])
        for at, pattern in (([[19.5, 0]], "exactly one"), ([[30, 30]], "exactly one")):
            with self.assertRaisesRegex(ValueError, pattern):
                self.build([self.array, {"use": "segment_boundary", "name": "per", "at": at}])
        with self.assertRaisesRegex(ValueError, "Define the boundary"):
            self.build([self.array, {"use": "segment_boundary", "name": "missing", "at": [[22, 0]]}])

    def test_a_fixed_point_potential_holds_one_node(self):
        model = self.build([self.array, {"use": "fixed_point_potential", "at": [19.5, 2], "name": "gauge"}])
        held = [(n["x"], n["y"]) for n in model["nodes"] if n["point"] == "gauge"]
        self.assertEqual(held, [(19.5, 2.0)])
        self.assertEqual(model["point_properties"][0]["props"]["A_re"], 0.0)


class SpatialHarmonicTests(unittest.TestCase):
    def test_known_harmonics_are_recovered_and_uneven_sampling_refused(self):
        from femm_brain.reference_models import run_assessor
        period, count = 20.0, 160
        points = [[25.0, (k + 0.5) * period / count] for k in range(count)]
        field = lambda z: 1.164 * math.sin(2 * math.pi * z / period) + 0.128 * math.sin(6 * math.pi * z / period + 0.3)
        raw = {("base", "probe:B1", k): complex(field(p[1])) for k, p in enumerate(points)}
        case = {"measurements": [{"id": "probe", "op": "point_values", "points": points, "fields": ["B1"]}]}
        for order, value in ((1, 1.164), (3, 0.128), (5, 0.0)):
            assessor = {"kind": "spatial_harmonic", "measurement": "probe", "field": "B1", "axis": 1, "period": period,
                        "order": order, "value": value, "absolute_tolerance": 1e-9}
            result = run_assessor(assessor, case, raw, {}, None)
            self.assertEqual(result["status"], "passed", result)
        case["measurements"][0]["points"] = points[:-1] + [[25.0, 30.0]]
        result = run_assessor(dict(assessor, order=1, value=1.164), case, raw, {}, None)
        self.assertEqual(result["status"], "failed")
        self.assertIn("evenly", result["error"])


class LrkReconstructionTests(unittest.TestCase):
    def test_components_rebuild_the_official_model_region_by_region(self):
        plan = plan_level(_lrk(), "components")
        self.assertEqual(compare_regions(deepcopy(plan["reference_model"]), deepcopy(plan["expected_model"])), [])
        self.assertEqual(check_regions(plan["expected_model"]), [])

    def test_region_comparison_sees_winding_sign_phase_and_magnet_changes(self):
        case = _lrk()
        reference = plan_level(case, "components")["reference_model"]
        parameters = {name: item["value"] for name, item in case["components"]["parameters"].items()}

        def differences(use, **changes):
            assembly = deepcopy(case["components"]["assembly"])
            for step in assembly["steps"]:
                if step["use"] == use:
                    step.update(changes)
            return {d["where"] for d in compare_regions(deepcopy(reference), assemble(assembly, parameters)[0])}

        self.assertIn("regions.AWG22.circuit", differences("tooth_coil_winding", phase_sequence="forward"))
        self.assertIn("regions.AWG22.turns", differences("tooth_coil_winding", first_tooth_angle=30))
        self.assertIn("regions.NdFeB 40 MGOe.magdir", differences("surface_magnet_rotor", first_pole_polarity="outward"))
        self.assertTrue(differences("surface_magnet_rotor", thickness=1.0))

    def test_rotor_positions_rotate_the_rotor_group_before_each_solve(self):
        lua = plan_level(_lrk(), "components")["lua"].splitlines()
        rotations = [i for i, line in enumerate(lua) if line.startswith("mi_moverotate(0,0,")]
        self.assertEqual(len(rotations), 60)
        for i in rotations:
            self.assertEqual((lua[i - 1], lua[i + 1]), ("mi_selectgroup(2)", "mi_clearselected()"))
            self.assertTrue(any(line == "mi_createmesh()" for line in lua[i:i + 6]))
        self.assertIn("mo_groupselectblock(2)", lua)

    def test_entity_order_control_changes_order_only(self):
        model = _source_record(_lrk())[1]
        permuted = permute_entities(model, 1)
        self.assertEqual(compare_models(model, permuted), [])
        self.assertNotEqual([n["x"] for n in model["nodes"]], [n["x"] for n in permuted["nodes"]])

    def test_coordinate_jitter_control_stays_below_a_nanometre(self):
        model = _source_record(_lrk())[1]
        jittered = jitter_coordinates(model, 1, 1e-12)
        self.assertEqual(compare_models(model, jittered), [])
        self.assertNotEqual(model["nodes"], jittered["nodes"])
        self.assertTrue(all(abs(a["x"] - b["x"]) <= 1e-12 and abs(a["y"] - b["y"]) <= 1e-12
                            for a, b in zip(model["nodes"], jittered["nodes"])))
        with self.assertRaisesRegex(ValueError, "at most"):
            jitter_coordinates(model, 1, 1e-6)


class PhaseLinkageTests(unittest.TestCase):
    def published(self):
        case = _lrk()
        metrics = {t["metric"]: t["value"] for t in case["targets"] if t["metric"].startswith("flux_")}
        return case, metrics

    def test_construction_predicts_the_published_lrk_phase_linkage(self):
        case, metrics = self.published()
        for model in (_source_record(case)[1], plan_level(case, "components")["expected_model"]):
            result = run_assessor(case["assessors"][0], case, {}, metrics, model)
            self.assertEqual(result["status"], "passed", result)
            self.assertLess(max(row["phase_error_deg"] for row in result["rows"]), 0.05)

    def test_swapped_phases_or_a_reversed_coil_fail(self):
        case, metrics = self.published()
        model = _source_record(case)[1]
        swapped = {name.replace("flux_B", "flux_X").replace("flux_C", "flux_B").replace("flux_X", "flux_C"): value
                   for name, value in metrics.items()}
        result = run_assessor(case["assessors"][0], case, {}, swapped, model)
        self.assertEqual((result["status"], result["measured_sequence_check"]), ("failed", False))
        reversed_coil = deepcopy(model)
        coil = next(label for label in reversed_coil["labels"] if label.get("circuit") == "A")
        coil["turns"] = -coil["turns"]
        result = run_assessor(case["assessors"][0], case, {}, metrics, reversed_coil)
        self.assertEqual(result["status"], "failed")
        self.assertGreater(result["winding_imbalance"], 0.1)

    def test_prediction_follows_rotor_polarity(self):
        model = _source_record(_lrk())[1]
        windings, rotor = linkage_prediction(model, 7, ["A", "B", "C"], 2)
        flipped = deepcopy(model)
        for label in flipped["labels"]:
            if label.get("group") == 2 and label["material"] == "NdFeB 40 MGOe":
                label["magdir"] = (label["magdir"] + 180) % 360
        _, reversed_rotor = linkage_prediction(flipped, 7, ["A", "B", "C"], 2)
        self.assertAlmostEqual(abs(cmath.phase(reversed_rotor / rotor)), math.pi)


class LibraryTests(unittest.TestCase):
    def test_bundled_curves_are_valid_and_bad_curves_are_refused(self):
        physics = {"matlib.dat": "magnetics", "heatlib.dat": "heatflow", "statlib.dat": "electrostatics"}
        entries = json.loads((COMPONENT_DATA / "library_materials.json").read_text())["materials"]
        curves = [e for e in entries if e.get("curve")]
        self.assertGreater(len(curves), 200)
        for entry in curves:
            _check_curve(entry["name"], entry["curve"], physics[entry["library"]])
        problem = {"physics": "magnetics", "units": "millimeters", "type": "planar", "frequency": 0, "depth": 1}
        for curve, pattern in (([[0, 0], [1, 100], [0.9, 200]], "strictly increasing"), ([[0, 0], [1, 100]], "at least 3"),
                               ([[0, 0], [1, float("nan")], [2, 300]], "points")):
            with self.subTest(curve=curve), self.assertRaisesRegex(ValueError, pattern):
                assemble({"problem": problem, "steps": [{"use": "material", "name": "x", "curve": curve}]}, {})

    def test_library_names_are_unambiguous_and_renamable(self):
        heat = {"physics": "heatflow", "units": "meters", "type": "planar", "depth": 1}
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            assemble({"problem": heat, "steps": [{"use": "library_material", "name": "Ammonia", "library": "heatlib.dat"}]}, {})
        model, _ = assemble({"problem": heat, "steps": [{"use": "library_material", "name": "Ammonia", "library": "heatlib.dat",
                                                         "folder": "Gases at 1 atm (stagnant only)"}]}, {})
        self.assertAlmostEqual(model["materials"][0]["props"]["Kx"], 0.0153)
        magnetic = {"physics": "magnetics", "units": "millimeters", "type": "planar", "frequency": 0, "depth": 1}
        model, _ = assemble({"problem": magnetic, "steps": [{"use": "library_material", "name": "22 AWG", "library": "matlib.dat",
                                                             "as_name": "AWG22"}]}, {})
        self.assertEqual([m["name"] for m in model["materials"]], ["AWG22"])


class EvidenceTests(unittest.TestCase):
    def test_partial_coverage_is_not_reported_as_reproduced(self):
        graph = load_graph()
        examples = {e["id"]: e for e in graph["examples"]}
        nodes = {n["id"]: n for n in graph["nodes"]}
        status = lambda example: nodes[examples[example]["workflow_id"]]["verification"]["status"]
        self.assertEqual(status("transientheat"), "partially_checked")
        self.assertEqual(status("heatflowtutorial"), "numerically_checked")
        self.assertTrue(nodes[examples["transientheat"]["workflow_id"]]["verification"]["not_reproduced"])

    def test_evidence_goes_stale_when_a_component_or_case_changes(self):
        source = Path(__file__).parents[1] / "data"
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "data"
            shutil.copytree(source, data)
            record = json.loads((data / "verification" / "component_designs.json").read_text())
            for case in record["cases"]:
                digests = case["levels"]["article"].get("component_digests") or {}
                if "iabc_open_boundary" in digests:
                    digests["iabc_open_boundary"] = "0" * 64
            (data / "verification" / "component_designs.json").write_text(json.dumps(record))
            catalog_path = data / "reproductions.json"
            cases = json.loads(catalog_path.read_text())
            tutorial = next(c for c in cases["cases"] if c["id"] == "magneticstutorial-coil")
            tutorial["targets"][0]["value"] = 0.0230
            catalog_path.write_text(json.dumps(cases))
            graph = load_graph(data)
            nodes = {n["id"]: n for n in graph["nodes"]}
            iabc = nodes["method:methods_iabc_open_domain"]["verification"]
            self.assertNotEqual(iabc["status"], "numerically_checked")
            self.assertTrue(iabc["stale_evidence_case_ids"])
            examples = {e["id"]: e for e in graph["examples"]}
            self.assertFalse(examples["magneticstutorial"]["reproductions"][0]["fresh"])

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
