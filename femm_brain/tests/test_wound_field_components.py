"""Salient-pole wound-field rotors (pole shoes, polygon or round cores, field coils, damper bars), trapezoidal slots with
closed wedges and stacked layers, the field winding as the rotor's source in linkage predictions, and dq terminal
quantities."""
import cmath
import math
import unittest

from femm_brain.components import assemble
from femm_brain.geometry import arc_geometry
from femm_brain.reference_models import linkage_prediction, run_assessor
from femm_brain.reproduce import check_regions, overlapping_arcs

PROBLEM = {"physics": "magnetics", "units": "millimeters", "type": "planar", "frequency": 0, "depth": 100}
SHOE = 2 * 125 * math.sin(math.radians(20))
ROTOR = {"use": "salient_pole_rotor", "pole_count": 6, "outer_radius": 125, "shoe_width": SHOE, "shoe_tip_height": 6.25,
         "body_width": 31.09, "root_distance": SHOE / 2 / math.tan(math.radians(30)), "coil_width": (SHOE - 31.09) / 2,
         "damper_bars": 4, "damper_pitch": 8, "damper_radius": 122, "damper_diameter": 4.64, "damper_material": "Cu",
         "shaft_radius": 28, "shaft_material": "Air", "interpolar_material": "Air", "iron_material": "Iron", "group": 1}
MATERIALS = [{"use": "material", "name": n} for n in ("Iron", "Cu", "Air")]


def without(step, *keys):
    return {k: v for k, v in step.items() if k not in keys}


def build(steps):
    return assemble({"problem": PROBLEM, "steps": steps}, {})[0]


def field(**extra):
    axial = (ROTOR["root_distance"] + math.sqrt(125 ** 2 - (SHOE / 2) ** 2) - 6.25) / 2
    return dict({"use": "salient_pole_winding", "pole_count": 6, "phases": ["F"], "turns": 100, "material": "Cu",
                 "label_axial": axial, "label_offset": 31.09 / 2 + (SHOE - 31.09) / 4, "polarities": [-1, 1, -1, 1, -1, 1],
                 "group": 1}, **extra)


def local(model, degrees):
    c, s = math.cos(math.radians(-degrees)), math.sin(math.radians(-degrees))
    return [(n["x"] * c - n["y"] * s, n["x"] * s + n["y"] * c) for n in model["nodes"]]


def has(points, x, y, tol=1e-9):
    return any(math.isclose(px, x, abs_tol=tol) and math.isclose(py, y, abs_tol=tol) for px, py in points)


class SalientPoleRotorTests(unittest.TestCase):
    def test_shoes_bodies_and_a_hexagonal_core_whose_corners_the_coils_share(self):
        model = build(MATERIALS + [{"use": "circuit", "name": "F", "current": 1}, ROTOR, field(),
                                   {"use": "circle", "radius": 127}, {"use": "block_label", "at": [126, 0], "material": "Air"}])
        self.assertEqual(check_regions(model), [])
        self.assertEqual(overlapping_arcs(model), [])
        points = local(model, 60)
        corner_u = math.sqrt(125 ** 2 - (SHOE / 2) ** 2)
        root = ROTOR["root_distance"]
        for x, y in ((corner_u, SHOE / 2), (corner_u - 6.25, SHOE / 2), (corner_u - 6.25, 31.09 / 2), (root, 31.09 / 2)):
            self.assertTrue(has(points, x, y) and has(points, x, -y), (x, y))
        # The core's corner lies on the axis between poles, at the apothem over cos(30), and is one node.
        corners = [n for n in model["nodes"] if math.isclose(math.hypot(n["x"], n["y"]), root / math.cos(math.radians(30)), abs_tol=1e-9)]
        self.assertEqual(len(corners), 6)
        # Six faces, six interpolar closures on the face-corner circle, 24 bars of two halves and the shaft.
        radii = sorted(round(arc_geometry((model["nodes"][a["n0"]]["x"], model["nodes"][a["n0"]]["y"]),
                                          (model["nodes"][a["n1"]]["x"], model["nodes"][a["n1"]]["y"]), a["angle"])["radius"], 6)
                       for a in model["arcs"])
        self.assertEqual(radii.count(125.0), 12)
        self.assertEqual(radii.count(2.32), 48)
        labels = {}
        for label in model["labels"]:
            labels[label["material"]] = labels.get(label["material"], 0) + 1
        self.assertEqual(labels, {"Cu": 24 + 12, "Air": 6 + 1 + 1, "Iron": 1})

    def test_bars_sit_symmetrically_about_each_pole_axis(self):
        model = build(MATERIALS + [ROTOR])
        bars = [(l["x"], l["y"]) for l in model["labels"] if l["material"] == "Cu"]
        angles = sorted(round(math.degrees(math.atan2(y, x)) % 360, 9) for x, y in bars)
        self.assertEqual(angles[:4], [4.0, 12.0, 48.0, 56.0])
        self.assertTrue(all(math.isclose(math.hypot(x, y), 122, abs_tol=1e-9) for x, y in bars))

    def test_an_eccentric_face_widens_the_gap_toward_the_tips(self):
        model = build(MATERIALS + [dict(without(ROTOR, "interpolar_material"), face_radius=100, damper_bars=0)])
        faces = []
        for a in model["arcs"]:
            p0, p1 = model["nodes"][a["n0"]], model["nodes"][a["n1"]]
            info = arc_geometry((p0["x"], p0["y"]), (p1["x"], p1["y"]), a["angle"])
            if math.isclose(info["radius"], 100, abs_tol=1e-9):
                faces.append(info["center"])
        self.assertEqual(len(faces), 6)
        self.assertTrue(all(math.isclose(math.hypot(*c), 25, abs_tol=1e-9) for c in faces))
        corner = 25 + math.sqrt(100 ** 2 - (SHOE / 2) ** 2)
        self.assertTrue(has(local(model, 0), corner, SHOE / 2))
        self.assertLess(math.hypot(corner, SHOE / 2), 125)

    def test_a_round_core_and_coils_short_of_the_shoe(self):
        step = dict(ROTOR, root_shape="arc", root_distance=80, coil_width=20, coil_start=82, coil_end=105, damper_bars=0)
        model = build(MATERIALS + [{"use": "circuit", "name": "F", "current": 1}, step,
                                   field(label_axial=93.5, label_offset=25), {"use": "circle", "radius": 127},
                                   {"use": "block_label", "at": [126, 0], "material": "Air"}])
        self.assertEqual(check_regions(model), [])
        points = local(model, 0)
        for x, y in ((82, 15.545), (82, 35.545), (105, 35.545), (105, 15.545), (math.sqrt(80 ** 2 - 15.545 ** 2), 15.545)):
            self.assertTrue(has(points, x, y), (x, y))

    def test_impossible_rotors_are_refused(self):
        refusals = (
            (dict(ROTOR, pole_count=5), "even number of poles"),
            (dict(ROTOR, shoe_width=140), "overlap"),
            (dict(ROTOR, body_width=90), "narrower than the shoe"),
            (dict(ROTOR, face_radius=130), "face radius"),
            (dict(ROTOR, root_distance=115), "beyond the core"),
            (dict(ROTOR, coil_width=30), "half of the gap|beyond the shoe"),
            (dict(ROTOR, coil_start=60), "between the core and the shoe underside"),
            (dict(ROTOR, damper_radius=124), "inside the pole shoes"),
            (dict(ROTOR, damper_pitch=2), "overlap"),
            (without(ROTOR, "damper_material"), "need a material"),
            (without(ROTOR, "shaft_material"), "needs a material"),
            (dict(ROTOR, shaft_radius=80), "inside the core"),
        )
        for step, message in refusals:
            with self.assertRaisesRegex(ValueError, message):
                build(MATERIALS + [step])


STATOR = {"use": "trapezoidal_slot_stator", "slot_count": 36, "bore_radius": 128.5, "opening_width": 2, "opening_height": 2,
          "slot_width_top": 11.59, "slot_width_bottom": 14.6, "slot_height": 17.2, "wedge_height": 2.24, "close_wedges": True,
          "wedge_material": "Air", "layer_divider_height": 9.0, "iron_material": "Iron"}


class LayeredTrapezoidalSlotTests(unittest.TestCase):
    def test_the_wedge_and_two_stacked_layers_are_regions_of_their_own(self):
        winding = {"use": "tooth_coil_winding", "slot_count": 36, "pole_pairs": 3, "turns": 13, "material": "Cu",
                   "label_radius": 137.2, "layout": "slot_star", "layers": "double", "coil_pitch": 5, "return_layer_radius": 145.9}
        model = build(MATERIALS + [{"use": "circuit", "name": n, "current": 0} for n in "ABC"] +
                      [STATOR, winding, {"use": "circle", "radius": 175}, {"use": "circle", "radius": 127},
                       {"use": "block_label", "at": [0, 0], "material": "Air"}, {"use": "block_label", "at": [128, 0], "material": "Air"}])
        self.assertEqual(check_regions(model), [])
        points = local(model, 5)
        side = 128.5 + 2 + 2.24
        half = 11.59 / 2 + 9.0 * (14.6 - 11.59) / (2 * 17.2)
        for x, y in ((side, 11.59 / 2), (side + 9.0, half), (side + 17.2, 14.6 / 2)):
            self.assertTrue(has(points, x, y) and has(points, x, -y), (x, y))
        # The wedge chord and the layer divider are chords across each slot (36 of each).
        turned = local(model, 0)
        across = [(turned[s["n0"]], turned[s["n1"]]) for s in model["segments"]]
        for x, width in ((side, 11.59), (side + 9.0, 2 * half)):
            count = sum(1 for a, b in across if math.isclose(math.dist(a, b), width, abs_tol=1e-9)
                        and math.isclose(math.hypot(*a), math.hypot(*b), abs_tol=1e-9)
                        and math.isclose(math.hypot(*a), math.hypot(x, width / 2), abs_tol=1e-9))
            self.assertEqual(count, 36)
        self.assertEqual(sum(1 for l in model["labels"] if l["material"] == "Air"), 36 + 2)

    def test_impossible_wedges_and_layers_are_refused(self):
        refusals = (
            (dict(STATOR, wedge_height=0), "wedge height"),
            (dict(STATOR, layer_divider_height=17.2), "inside the slot body"),
            (dict(STATOR, slot_divider=True), "excludes a side-by-side"),
            (dict(STATOR, close_slot_openings=False), "closed slot openings"),
        )
        for step, message in refusals:
            with self.assertRaisesRegex(ValueError, message):
                build(MATERIALS + [step])


class FieldWindingLinkageTests(unittest.TestCase):
    def model(self, current):
        return build(MATERIALS + [{"use": "circuit", "name": n, "current": 0} for n in "ABC"] +
                     [{"use": "circuit", "name": "F", "current": current}, ROTOR, field(),
                      {"use": "tooth_coil_winding", "slot_count": 36, "pole_pairs": 3, "turns": 13, "material": "Cu",
                       "label_radius": 137.2, "layout": "slot_star", "layers": "double", "coil_pitch": 5,
                       "return_layer_radius": 145.9}, STATOR])

    def test_a_north_pole_on_the_x_axis_is_a_real_positive_rotor_phasor(self):
        windings, rotor = linkage_prediction(self.model(2.5), 3, ["A", "B", "C"], 1)
        self.assertLess(abs(cmath.phase(rotor)), 1e-9)
        # A reversed field current reverses the rotor, and with it every linkage.
        _, reversed_rotor = linkage_prediction(self.model(-2.5), 3, ["A", "B", "C"], 1)
        self.assertTrue(cmath.isclose(reversed_rotor, -rotor, rel_tol=1e-12))
        # The forward winding's phases are 120 electrical degrees apart in the order A, B, C.
        gamma = {n: math.degrees(cmath.phase(windings[n] / rotor)) for n in "ABC"}
        self.assertAlmostEqual((gamma["B"] - gamma["A"]) % 360, 120, places=6)
        self.assertAlmostEqual((gamma["C"] - gamma["A"]) % 360, 240, places=6)

    def test_a_rotor_without_field_current_or_magnets_is_refused(self):
        with self.assertRaisesRegex(ValueError, "field-current"):
            linkage_prediction(self.model(0), 3, ["A", "B", "C"], 1)


class TerminalQuantityTests(unittest.TestCase):
    def test_voltages_power_and_power_factor_follow_the_flux_linkages(self):
        psi_d, psi_q, i_d, i_q, theta = 0.9, -0.4, -20.0, -15.0, 30.0
        t = math.radians(theta)
        # Phase flux linkages and currents whose amplitude-invariant Park transform is (psi_d, psi_q) and (i_d, i_q).
        psi = [psi_d * math.cos(t - k * 2 * math.pi / 3) - psi_q * math.sin(t - k * 2 * math.pi / 3) for k in range(3)]
        amps = [i_d * math.cos(t - k * 2 * math.pi / 3) - i_q * math.sin(t - k * 2 * math.pi / 3) for k in range(3)]
        raw = {}
        for k, name in enumerate(("phA", "phB", "phC")):
            raw[("s0", name, 2)], raw[("s0", name, 0)] = complex(psi[k]), complex(amps[k])
        case = {"steps": [{"label": "s0", "dq": {"point": "p", "id": i_d, "iq": i_q, "electrical_angle_deg": theta}}]}
        base = {"kind": "dq_linkage", "phases": ["phA", "phB", "phC"], "pole_pairs": 3, "point": "p", "frequency_hz": 50,
                "resistance": 0.5, "end_inductance": 1e-3, "value": 1, "relative_tolerance": 1}
        w = 2 * math.pi * 50
        vd = 0.5 * i_d - w * (psi_q + 1e-3 * i_q)
        vq = 0.5 * i_q + w * (psi_d + 1e-3 * i_d)
        expected = {"v_d": vd, "v_q": vq, "line_voltage": math.sqrt(1.5) * math.hypot(vd, vq),
                    "electrical_power": 1.5 * (vd * i_d + vq * i_q),
                    "terminal_power_factor": (vd * i_d + vq * i_q) / (math.hypot(vd, vq) * math.hypot(i_d, i_q))}
        for quantity, value in expected.items():
            result = run_assessor(dict(base, quantity=quantity), case, raw, {}, None)
            self.assertAlmostEqual(result["observed"], value, places=9, msg=quantity)
        # scale multiplies the result; the currents the solve carried can replace the step records.
        result = run_assessor(dict(base, quantity="electrical_power", scale=-1, currents_from="circuits"), case, raw, {}, None)
        self.assertAlmostEqual(result["observed"], -expected["electrical_power"], places=9)
        refused = run_assessor(dict(without(base, "frequency_hz"), quantity="line_voltage"), case, raw, {}, None)
        self.assertIn("frequency_hz", refused["error"])
        refused = run_assessor(dict(base, quantity="v_d", transform="power_invariant"), case, raw, {}, None)
        self.assertIn("amplitude-invariant", refused["error"])


if __name__ == "__main__":
    unittest.main()
