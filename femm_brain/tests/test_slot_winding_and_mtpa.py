"""Windings given slot by slot (a combined star-delta winding read from a drawing), library steels with an
overridden lamination fill, and the torque at the optimal current angle from a sweep of current angles."""
import math
import unittest

from femm_brain.components import assemble, winding_layout
from femm_brain.reference_models import assessor_metric, run_assessor
from femm_brain.reproduce import check_regions

PROBLEM = {"physics": "magnetics", "units": "millimeters", "type": "planar", "frequency": 0, "depth": 1}
STATOR = {"use": "slotted_stator", "slot_count": 36, "slot_bottom_radius": 72.5, "tip_radius": 55, "tooth_width": 5.14,
          "tip_arc_length": 6.78, "shoe_axial_distance": 57.5, "shoe_corner_axial_distance": 55.4, "first_tooth_angle": -5,
          "iron_material": "Iron"}
STAR = [["A", -26], ["A", -26], ["C", 26], ["C", 26], ["C", 26], ["B", -26], ["B", -26], ["B", -26], ["A", 26]]
STAR_DELTA = [["A", -26], ["CA", 45], ["CA", 45], ["C", 26], ["BC", -45], ["BC", -45], ["B", -26], ["AB", 45], ["AB", 45]]


def build(steps):
    return assemble({"problem": PROBLEM, "steps": steps}, {})[0]


def stator(pattern, **extra):
    circuits = sorted({name for name, _ in pattern})
    return build([{"use": "material", "name": "Iron", "values": {"Mu_x": 1000, "Mu_y": 1000}}, {"use": "material", "name": "Cu"},
                  {"use": "material", "name": "Air"}, *({"use": "circuit", "name": n, "current": 0} for n in circuits), STATOR,
                  {"use": "circle", "radius": 50}, {"use": "block_label", "at": [0, 0], "material": "Air"},
                  {"use": "circle", "radius": 90}, {"use": "block_label", "at": [54.9, 0.5], "material": "Air"},
                  dict({"use": "slot_winding", "slot_count": 36, "pattern": pattern, "label_radius": 64, "material": "Cu"}, **extra)])


def slots(model):
    return {round(math.degrees(math.atan2(l["y"], l["x"])) % 360, 6): (l["circuit"], l["turns"]) for l in model["labels"] if l.get("circuit")}


class SlotWindingTests(unittest.TestCase):
    def test_a_pattern_over_one_pole_repeats_with_alternating_signs(self):
        model = stator(STAR)
        self.assertEqual(check_regions(model), [])
        found = slots(model)
        self.assertEqual(len(found), 36)
        for k in range(36):
            name, turns = STAR[k % 9]
            self.assertEqual(found[round(10.0 * k % 360, 6)], (name, turns if (k // 9) % 2 == 0 else -turns))

    def test_the_star_pattern_is_the_star_of_slots_with_every_sign_reversed(self):
        found = slots(stator(STAR))
        layout = winding_layout(36, 2, ["A", "B", "C"], "slot_star", "single")
        for k, (phase, polarity) in layout.items():
            self.assertEqual(found[round(10.0 * k % 360, 6)], (phase, -26 * polarity))

    def test_combined_star_delta_raises_the_winding_factor(self):
        # Delta currents (i_a - i_b)/3 and so on: 45 delta turns at 1/sqrt(3) of the line current give a star slot's
        # ampere-turns, and each delta pair's phasor lies between its star neighbours', so every slot's phasor is within
        # 10 electrical degrees of its position (20 for the star winding's 60-degree belts): kd 0.9899 against 0.9598.
        a = {"A": 1, "B": complex(-0.5, -math.sqrt(3) / 2), "C": complex(-0.5, math.sqrt(3) / 2)}
        current = dict(a, AB=(a["A"] - a["B"]) / 3, BC=(a["B"] - a["C"]) / 3, CA=(a["C"] - a["A"]) / 3)
        factors = {}
        for name, pattern in (("star", STAR), ("star-delta", STAR_DELTA)):
            found = slots(stator(pattern))
            phasors = [found[round(10.0 * k, 6)][1] * current[found[round(10.0 * k, 6)][0]] for k in range(36)]
            for k, phasor in enumerate(phasors):
                self.assertAlmostEqual(abs(phasor), 26, delta=0.02)
                deviation = (math.degrees(math.atan2(phasor.imag, phasor.real)) + 20 * k - 180 + 180) % 360 - 180
                self.assertLessEqual(abs(deviation), (20 if name == "star" else 10) + 1e-9)
            total = sum(p * complex(math.cos(math.radians(20 * k)), math.sin(math.radians(20 * k))) for k, p in enumerate(phasors))
            factors[name] = abs(total) / sum(abs(p) for p in phasors)
        self.assertAlmostEqual(factors["star"], (1 + 2 * math.cos(math.radians(20))) / 3, places=3)
        self.assertAlmostEqual(factors["star-delta"], (1 + 2 * math.cos(math.radians(10))) / 3, places=3)

    def test_contract_refusals(self):
        for pattern, extra, message in (([], {}, "lists"), ([["A", 0]], {}, "nonzero"), ([["A", 2.5]], {}, "whole number"),
                                        (STAR[:7], {}, "whole number of times"), (STAR, {"slot_count": 30}, "whole number of times"),
                                        (STAR * 2, {"slot_count": 54}, "even number")):
            with self.assertRaisesRegex(ValueError, message):
                build([{"use": "material", "name": "Cu"}, {"use": "circuit", "name": "A", "current": 0},
                       dict({"use": "slot_winding", "slot_count": 36, "pattern": pattern, "label_radius": 64, "material": "Cu"}, **extra)])


class LibraryOverrideTests(unittest.TestCase):
    def test_a_lamination_fill_overrides_the_entry_and_keeps_its_curve(self):
        step = {"use": "library_material", "name": "M-19 14 mil", "library": "matlib.dat", "as_name": "Steel",
                "folder": "Soft Magnetic Materials/Silicon Iron"}
        plain = build([step])["materials"][0]
        solid = build([dict(step, values={"LamFill": 1.0})])["materials"][0]
        self.assertEqual(plain["props"]["LamFill"], 0.955)
        self.assertEqual(solid["props"]["LamFill"], 1.0)
        self.assertEqual({k: v for k, v in plain["props"].items() if k != "LamFill"}, {k: v for k, v in solid["props"].items() if k != "LamFill"})
        self.assertEqual(plain["curve"], solid["curve"])
        for values, message in (({"Fill": 1.0}, "no properties"), ({"LamFill": float("nan")}, "finite")):
            with self.assertRaisesRegex(ValueError, message):
                build([dict(step, values=values)])


class OptimalCurrentAngleTests(unittest.TestCase):
    def sweep(self, angles, peak=52.0, positions=3):
        steps, raw = [], {}
        for alpha in angles:
            for j in range(positions):
                label = f"a{alpha}_{j}"
                steps.append({"label": label, "dq": {"point": f"a{alpha}", "id": 10 * math.cos(math.radians(alpha)),
                                                     "iq": 10 * math.sin(math.radians(alpha)), "electrical_angle_deg": 0}})
                raw[(label, "gap", 0)] = complex(20 * math.cos(math.radians(2 * (alpha - peak))) + 0.3 * (j - 1), 0)
                for phase in ("phA", "phB", "phC"):
                    raw[(label, phase, 2)] = 0j
        return {"steps": steps}, raw

    def assessor(self, angles, **extra):
        return dict({"kind": "dq_linkage", "points": [f"a{a}" for a in angles], "over_points": "maximum", "phases": ["phA", "phB", "phC"],
                     "pole_pairs": 2, "quantity": "stress_torque", "torque_measurement": "gap", "sequence_direction": 1,
                     "value": 20, "relative_tolerance": 0.001}, **extra)

    def test_the_largest_point_and_its_parabolic_refinement(self):
        angles = (40, 45, 50, 55, 60)
        case, raw = self.sweep(angles)
        coarse = run_assessor(self.assessor(angles), case, raw, {}, {})
        self.assertAlmostEqual(coarse["observed"], 20 * math.cos(math.radians(4)), places=9)
        self.assertAlmostEqual(coarse["optimal_current_angle_deg"], 50, places=9)
        fine = run_assessor(self.assessor(angles, refine="parabolic"), case, raw, {}, {})
        self.assertEqual(fine["status"], "passed")
        self.assertAlmostEqual(fine["observed"], 20, delta=0.01)
        self.assertAlmostEqual(fine["optimal_current_angle_deg"], 52, delta=0.1)
        self.assertEqual(fine["samples"], 15)
        self.assertEqual(assessor_metric(self.assessor(angles)), "max[a40..a60]:stress_torque")

    def test_a_maximum_at_the_sweep_end_is_not_extrapolated(self):
        angles = (30, 35, 40)
        case, raw = self.sweep(angles)
        result = run_assessor(self.assessor(angles, refine="parabolic"), case, raw, {}, {})
        self.assertAlmostEqual(result["observed"], 20 * math.cos(math.radians(24)), places=9)
        self.assertIn("largest point", result["statistic"])

    def test_ill_posed_sweeps_are_reported(self):
        case, raw = self.sweep((40, 45))
        for extra in ({"over_points": "mean"}, {"points": ["a40"]}, {"point": "a40"}, {"quantity": "L_d"}, {"refine": "cubic"}):
            self.assertIn("error", run_assessor(self.assessor((40, 45), **extra), case, raw, {}, {}))


if __name__ == "__main__":
    unittest.main()
