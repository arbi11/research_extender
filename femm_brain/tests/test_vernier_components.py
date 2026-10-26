"""Tapered shoes, shoe-root closures, radial layer dividers and distributed double layers (Vernier and other
outer-rotor machines), an outer rotor turned by a sliding band, and dq currents and power factor."""
import math
import unittest

from femm_brain.components import assemble, winding_layout
from femm_brain.geometry import arc_geometry
from femm_brain.reference_models import _rotor_samples, run_assessor
from femm_brain.reproduce import check_regions, overlapping_arcs

PROBLEM = {"physics": "magnetics", "units": "millimeters", "type": "planar", "frequency": 0, "depth": 1}
# An outer-rotor stator of 36 slots with tapered shoes, closed at the shoe roots and split into two layers.
STATOR = {"use": "slotted_stator", "slot_count": 36, "slot_bottom_radius": 169.54, "tip_radius": 197.5, "tooth_width": 7.69,
          "tip_arc_length": 12.53, "shoe_axial_distance": 195.46, "shoe_corner_axial_distance": 196.40,
          "slot_closure": "shoe_roots", "layer_divider_radius": 183.3, "closure_max_segment": 0.5}


def build(steps):
    return assemble({"problem": PROBLEM, "steps": steps}, {})[0]


def local(model, degrees):
    """Node coordinates turned so that the tooth at ``degrees`` lies on the +x axis."""
    c, s = math.cos(math.radians(-degrees)), math.sin(math.radians(-degrees))
    return [(n["x"] * c - n["y"] * s, n["x"] * s + n["y"] * c) for n in model["nodes"]]


class TaperedShoeTests(unittest.TestCase):
    def test_the_underside_runs_from_the_root_to_a_corner_on_the_tip_radial(self):
        model = build([STATOR])
        points = local(model, 10)
        tip_half = math.degrees(12.53 / (2 * 197.5))
        corner_y = 196.40 * math.tan(math.radians(tip_half))
        for x, y in ((195.46, 7.69 / 2), (196.40, corner_y)):
            self.assertTrue(any(math.isclose(px, x, abs_tol=1e-9) and math.isclose(abs(py), y, abs_tol=1e-9) for px, py in points))
        # The corner and the tip end lie on one radial line.
        tip = [p for p in points if math.isclose(math.hypot(*p), 197.5, abs_tol=1e-9) and abs(p[1]) < 7 and p[0] > 0]
        self.assertEqual(len(tip), 2)
        self.assertTrue(all(math.isclose(math.degrees(math.atan2(abs(p[1]), p[0])), tip_half, abs_tol=1e-9) for p in tip))
        # A flat shoe is the default: without a corner distance the corner sits at the underside's axial distance.
        flat = local(build([{k: v for k, v in STATOR.items() if k != "shoe_corner_axial_distance"}]), 0)
        self.assertTrue(any(math.isclose(px, 195.46, abs_tol=1e-9) and abs(py) > 6 for px, py in flat))

    def test_closures_and_layer_dividers_are_arcs_centred_on_the_axis(self):
        model = build([STATOR])
        radii = set()
        for arc in model["arcs"]:
            a, b = model["nodes"][arc["n0"]], model["nodes"][arc["n1"]]
            info = arc_geometry((a["x"], a["y"]), (b["x"], b["y"]), arc["angle"])
            self.assertLess(math.hypot(*info["center"]), 1e-9)
            radii.add(round(info["radius"], 6))
        root = round(math.hypot(195.46, 7.69 / 2), 6)
        self.assertEqual(radii, {169.54, 183.3, root, 197.5})
        self.assertEqual(overlapping_arcs(model), [])

    def test_impossible_shoes_and_dividers_are_refused(self):
        refusals = (
            (dict(STATOR, shoe_corner_axial_distance=195.0), "corner lies between"),
            (dict(STATOR, slot_closure="middle"), "must be one of"),
            (dict(STATOR, layer_divider_radius=196.0), "between the slot bottom and the shoe roots"),
            (dict(STATOR, slot_divider=True), "excludes a layer divider"),
            (dict(STATOR, close_slot_openings=False, slot_closure="corners", slot_divider=False), "needs closed slots"),
        )
        for step, message in refusals:
            with self.assertRaisesRegex(ValueError, message):
                build([step])


class DistributedDoubleLayerTests(unittest.TestCase):
    def winding(self, **extra):
        return dict({"use": "tooth_coil_winding", "slot_count": 36, "pole_pairs": 3, "turns": 1, "material": "Cu",
                     "label_radius": 189.4, "layout": "slot_star", "layers": "double", "coil_pitch": 5,
                     "return_layer_radius": 176.4}, **extra)

    def model(self):
        steps = [{"use": "material", "name": "Iron", "values": {"Mu_x": 1000, "Mu_y": 1000}}, {"use": "material", "name": "Cu"},
                 {"use": "material", "name": "Air"}, *({"use": "circuit", "name": n, "current": 0} for n in "ABC"),
                 dict(STATOR, iron_material="Iron"), {"use": "circle", "radius": 156.5}, {"use": "block_label", "at": [0, 0], "material": "Air"},
                 self.winding(), {"use": "block_label", "at": [197.0, 15.0], "material": "Air"}, {"use": "circle", "radius": 199.0}]
        return build(steps)

    def test_each_slot_holds_a_go_side_above_a_return_side(self):
        model = self.model()
        self.assertEqual(check_regions(model), [])
        coils = [l for l in model["labels"] if l.get("circuit")]
        self.assertEqual(len(coils), 72)
        top = {round(math.degrees(math.atan2(l["y"], l["x"])) % 360, 6): (l["circuit"], l["turns"]) for l in coils
               if math.isclose(math.hypot(l["x"], l["y"]), 189.4)}
        bottom = {round(math.degrees(math.atan2(l["y"], l["x"])) % 360, 6): (l["circuit"], l["turns"]) for l in coils
                  if math.isclose(math.hypot(l["x"], l["y"]), 176.4)}
        for k in range(36):
            go = top[round((10 * k + 5) % 360, 6)]
            back = bottom[round((10 * (k + 5) + 5) % 360, 6)]
            self.assertEqual((back[0], back[1]), (go[0], -go[1]))
        # Short pitch by one slot: the return layer repeats the go layer one slot clockwise.
        for k in range(36):
            self.assertEqual(bottom[round((10 * k + 5) % 360, 6)], top[round((10 * (k + 1) + 5) % 360, 6)])

    def test_a_vernier_winding_links_the_rotor_harmonic_in_reverse_sequence(self):
        # 66 rotor poles over a 3-pole-pair winding: the 33rd harmonic of the slots' turns is the reverse of the 3rd.
        model = self.model()
        import cmath
        coils = [l for l in model["labels"] if l.get("circuit")]
        S = {n: sum(l["turns"] * cmath.exp(33j * math.atan2(l["y"], l["x"])) for l in coils if l["circuit"] == n) for n in "ABC"}
        three = {n: sum(l["turns"] * cmath.exp(3j * math.atan2(l["y"], l["x"])) for l in coils if l["circuit"] == n) for n in "ABC"}
        for n in "ABC":
            self.assertAlmostEqual(abs(S[n]), abs(three[n]), places=9)
        self.assertAlmostEqual((math.degrees(cmath.phase(S["B"] / S["A"])) + 360) % 360, 240, places=6)
        self.assertAlmostEqual((math.degrees(cmath.phase(three["B"] / three["A"])) + 360) % 360, 120, places=6)

    def test_contract_refusals(self):
        for extra, message in ((dict(coil_pitch=None), "coil pitch and a return-layer radius"),
                               (dict(coil_pitch=36), "between 1 and the slot count"),
                               (dict(return_layer_radius=189.4), "own positive radius"),
                               (dict(layout="tooth_coils"), "belong to a distributed double layer")):
            with self.assertRaisesRegex(ValueError, message):
                build([{"use": "material", "name": "Cu"}, *({"use": "circuit", "name": n, "current": 0} for n in "ABC"),
                       {k: v for k, v in self.winding(**extra).items() if v is not None}])
        # The layout itself stays single-layer for the slot star; the double layer is the winding's placement.
        with self.assertRaisesRegex(ValueError, "single-layer"):
            winding_layout(36, 3, ["A", "B", "C"], layout="slot_star", layers="double")


class OuterRotorBandAndPowerFactorTests(unittest.TestCase):
    def test_an_outer_rotor_turns_opposite_to_the_stator_side(self):
        band = lambda angle: {"function": "modifyboundprop", "name": "AGE", "property": 10, "value": angle}
        steps = [{"label": f"p{k:02d}", "modify": [band(-0.5 * k)]} for k in range(12)]
        samples, span = _rotor_samples({"rotor_group": 1, "rotor_band": "AGE", "rotor_band_side": "outer", "pole_pairs": 60},
                                       {"steps": steps})
        self.assertEqual(span, 6)
        self.assertEqual([a for a, _ in samples], [0.5 * k for k in range(12)])
        with self.assertRaises(ValueError):
            _rotor_samples({"rotor_group": 1, "rotor_band": "AGE", "pole_pairs": 60}, {"steps": steps})
        with self.assertRaises(ValueError):
            _rotor_samples({"rotor_group": 1, "rotor_band_side": "outer", "pole_pairs": 60}, {"steps": steps})

    def test_power_factor_and_measured_currents(self):
        psi_d, psi_q, current, beta = 0.12, 0.09, 150.0, 100.0
        i_d, i_q = current * math.cos(math.radians(beta)), current * math.sin(math.radians(beta))
        steps, raw = [], {}
        for k, angle in enumerate((0, 37, 200)):
            label = f"s{k}"
            steps.append({"label": label, "dq": {"point": "P", "id": 1.0, "iq": 2.0, "electrical_angle_deg": angle}})
            for phase in range(3):
                shift = math.radians(angle) - 2 * math.pi * phase / 3
                raw[(label, "ph" + "ABC"[phase], 2)] = complex(psi_d * math.cos(shift) - psi_q * math.sin(shift), 0)
                raw[(label, "ph" + "ABC"[phase], 0)] = complex(i_d * math.cos(shift) - i_q * math.sin(shift), 0)
        base = {"kind": "dq_linkage", "point": "P", "phases": ["phA", "phB", "phC"], "pole_pairs": 33, "value": 0,
                "absolute_tolerance": 1e9, "currents_from": "circuits"}
        expected = (psi_d * i_q - psi_q * i_d) / (math.hypot(psi_d, psi_q) * current)
        self.assertAlmostEqual(run_assessor(dict(base, quantity="power_factor"), {"steps": steps}, raw, {}, {})["observed"], expected, places=12)
        torque = run_assessor(dict(base, quantity="torque"), {"steps": steps}, raw, {}, {})["observed"]
        self.assertAlmostEqual(torque, 1.5 * 33 * (psi_d * i_q - psi_q * i_d), places=9)
        # From the step records instead: id = 1, iq = 2.
        recorded = run_assessor(dict(base, quantity="power_factor", currents_from="steps"), {"steps": steps}, raw, {}, {})["observed"]
        self.assertAlmostEqual(recorded, (psi_d * 2 - psi_q * 1) / (math.hypot(psi_d, psi_q) * math.hypot(1, 2)), places=12)
        self.assertIn("error", run_assessor(dict(base, quantity="power_factor", currents_from="guess"), {"steps": steps}, raw, {}, {}))


if __name__ == "__main__":
    unittest.main()
