"""Straight slotted cores and their windings (the primaries of linear machines): geometry, regions and refusals."""
import unittest

from femm_brain.components import assemble, linear_slots
from femm_brain.reproduce import _face_cycles, check_regions

PROBLEM = {"physics": "magnetics", "units": "millimeters", "type": "planar", "frequency": 0, "depth": 50}
GEOMETRY = {"start": -133, "face": 0, "slot_count": 16, "slot_width": 10, "tooth_width": 6, "slot_depth": 20,
            "end_tooth_width": 10, "end_slot_width": 8}
SEQUENCE = [["A", 1], ["C", -1], ["B", 1], ["A", -1], ["C", 1], ["B", -1]]
PROPERTIES = [{"use": "material", "name": "Air"}, {"use": "material", "name": "Iron", "values": {"Mu_x": 1000, "Mu_y": 1000}},
              *({"use": "circuit", "name": n, "current": 0} for n in "ABC")]


def core(**extra):
    return dict({"use": "linear_slotted_core", "yoke_height": 6.5, "layers": 2, "iron_material": "Iron", "group": 1}, **GEOMETRY, **extra)


def coils(**extra):
    return dict({"use": "linear_winding", "material": "Air", "layers": 2, "sequence": SEQUENCE, "coil_pitch": 2, "first_slot": 1,
                 "coil_count": 12, "turns": 57, "go_layer": 1, "return_layer": 0, "empty_material": "Air", "group": 1}, **GEOMETRY, **extra)


def build(steps, box=((-200, -10), (200, 60))):
    (x0, y0), (x1, y1) = box
    around = [{"use": "polyline", "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]},
              {"use": "block_label", "at": [x0 + 1, y0 + 1], "material": "Air"}]
    return assemble({"problem": PROBLEM, "steps": PROPERTIES + steps + around}, {})[0]


def area(polygon):
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1]))) / 2


def sides(model):
    """{(slot centre x, layer centre y): (circuit, turns)} of the winding labels."""
    return {(round(l["x"], 9), round(l["y"], 9)): (l["circuit"], l["turns"]) for l in model["labels"] if l.get("circuit")}


class LinearSlotsTests(unittest.TestCase):
    def test_slot_edges_follow_the_end_teeth_and_end_slots(self):
        slots, length = linear_slots(-133, 16, 10, 6, 10, 8)
        self.assertEqual(length, 266)
        self.assertEqual(slots[0], (-123, -115))
        self.assertEqual(slots[1], (-109, -99))
        self.assertEqual(slots[-1], (115, 123))
        self.assertTrue(all(b - a == 10 for a, b in slots[1:-1]))
        # A [left, right] pair gives each end its own width; the default is a regular tooth and slot.
        slots, length = linear_slots(0, 3, 10, 6, [2, 4], [8, 9])
        self.assertEqual((slots, length), ([(2, 10), (16, 26), (32, 41)], 45))
        self.assertEqual(linear_slots(0, 2, 10, 6)[1], 6 + 10 + 6 + 10 + 6)

    def test_impossible_layouts_are_refused(self):
        for args, message in (((0, 0, 10, 6), "at least one"), ((0, 2.5, 10, 6), "whole number"), ((0, 3, 0, 6), "positive"),
                              ((0, 3, 10, 6, [1, 2, 3]), "end tooth width"), ((0, 3, 10, 6, None, -1), "end slot width"),
                              ((0, 1, 10, 6, None, [8, 9]), "single slot has one width")):
            with self.assertRaisesRegex(ValueError, message):
                linear_slots(*args)


class LinearCoreTests(unittest.TestCase):
    def test_regions_iron_area_and_layer_dividers(self):
        model = build([core(), coils()])
        self.assertEqual(check_regions(model), [])
        faces = _face_cycles(model)
        # The iron region: the whole block less sixteen slots, 20 mm deep.
        iron = 266 * 26.5 - (14 * 10 + 2 * 8) * 20
        self.assertTrue(any(abs(area(face) - iron) < 1e-6 for face in faces))
        # Each slot is split into two 10 mm layers; every layer region is 10 (or 8) by 10 mm.
        layer_areas = sorted(round(area(face), 6) for face in faces if area(face) < 120)
        self.assertEqual(layer_areas, [80.0] * 4 + [100.0] * 28)
        labels = [l for l in model["labels"] if l["material"] == "Iron"]
        self.assertEqual(len(labels), 1)
        self.assertEqual((labels[0]["x"], labels[0]["y"]), (0, 23.25))
        self.assertTrue(all(n["group"] == 1 for n in model["nodes"] if -133 <= n["x"] <= 133 and n["y"] >= 0))

    def test_shoes_close_the_coil_area_and_label_the_opening(self):
        model = build([core(opening_width=4, shoe_height=2, opening_material="Air"),
                       coils(shoe_height=2)])
        self.assertEqual(check_regions(model), [])
        openings = [l for l in model["labels"] if l["material"] == "Air" and abs(l["y"] - 1) < 1e-12]
        self.assertEqual(len(openings), 16)
        # The layers now share the 18 mm above the shoes: 9 mm each.
        self.assertEqual(sorted({round(y, 9) for _, y in sides(model)}), [6.5, 15.5])
        self.assertTrue(any(abs(n["x"] - (-119 - 2)) < 1e-12 and abs(n["y"] - 2) < 1e-12 for n in model["nodes"]))

    def test_an_upward_core_mirrors_the_downward_one(self):
        down, up = build([core(), coils()]), build([core(facing="up"), coils(facing="up")], box=((-200, -60), (200, 10)))
        self.assertEqual(check_regions(up), [])
        self.assertEqual({(x, -y): side for (x, y), side in sides(down).items()}, sides(up))

    def test_impossible_cores_are_refused(self):
        for extra, message in (({"opening_width": 4}, "both an opening width and a shoe height"),
                               ({"shoe_height": 2}, "both an opening width and a shoe height"),
                               ({"opening_width": 9, "shoe_height": 2, "opening_material": "Air"}, "narrower than every slot"),
                               ({"opening_width": 4, "shoe_height": 2}, "opening material"),
                               ({"shoe_height": 20, "opening_width": 4, "opening_material": "Air"}, "0 <= shoe height < slot depth"),
                               ({"close_slots": False}, "Layer dividers need closed slots"),
                               ({"facing": "left"}, "must be one of"),
                               ({"yoke_height": 0}, "outside its declared range")):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                build([core(**extra)])


class LinearWindingTests(unittest.TestCase):
    def test_coils_put_returns_a_pitch_along_with_the_opposite_sign(self):
        model = build([core(), coils()])
        placed = sides(model)
        centres = [(a + b) / 2 for a, b in linear_slots(-133, 16, 10, 6, 10, 8)[0]]
        for k in range(12):
            circuit, polarity = SEQUENCE[k % 6]
            self.assertEqual(placed[(centres[1 + k], 15.0)], (circuit, 57 * polarity))
            self.assertEqual(placed[(centres[3 + k], 5.0)], (circuit, -57 * polarity))
        # Half-filled end slots: the yoke-side layer of slots 13 and 14 and the gap-side layer of slots 1 and 2 are empty.
        empty = {(round(l["x"], 9), l["y"]) for l in model["labels"] if l["material"] == "Air" and not l.get("circuit") and l["y"] in (5, 15)}
        self.assertEqual(empty, {(centres[0], 5.0), (centres[0], 15.0), (centres[1], 5.0), (centres[2], 5.0),
                                 (centres[13], 15.0), (centres[14], 15.0), (centres[15], 5.0), (centres[15], 15.0)})
        # Every phase's turns cancel, so the winding carries no net current.
        for name in "ABC":
            self.assertEqual(sum(t for c, t in placed.values() if c == name), 0)

    def test_a_pattern_places_each_layer_as_given(self):
        pattern = [[None, None]] + [[["B", -3], ["A", 3]]] * 14 + [[None, None]]
        model = build([core(), {k: v for k, v in coils(pattern=pattern).items()
                                if k not in ("sequence", "coil_pitch", "first_slot", "coil_count", "turns", "go_layer", "return_layer")}])
        self.assertEqual(check_regions(model), [])
        self.assertEqual(sorted(set(sides(model).values())), [("A", 3), ("B", -3)])

    def test_a_periodic_core_wraps_coil_sides(self):
        geometry = dict(GEOMETRY, start=0, slot_count=12, end_tooth_width=3, end_slot_width=10)
        belts = [["A", 1], ["A", 1], ["C", -1], ["C", -1], ["B", 1], ["B", 1]]
        sequence = belts + [[circuit, -polarity] for circuit, polarity in belts]
        step = dict(coils(), **geometry, sequence=sequence, first_slot=0, coil_count=12, coil_pitch=5, wrap=True)
        step.pop("empty_material")
        model = build([dict(core(), **geometry), step], box=((-1, -10), (193, 60)))
        self.assertEqual(check_regions(model), [])
        self.assertEqual(len(sides(model)), 24)
        with self.assertRaisesRegex(ValueError, "outside the core"):
            build([dict(core(), **geometry), dict(step, wrap=False)], box=((-1, -10), (193, 60)))

    def test_impossible_windings_are_refused(self):
        bare = {k: v for k, v in coils().items() if k not in ("sequence", "coil_pitch", "first_slot", "coil_count", "turns", "go_layer",
                                                               "return_layer")}
        for step, message in (
                (coils(pattern=[[None, None]] * 16), "excludes the coil options"),
                (bare, "either a slot pattern or coils"),
                (dict(bare, pattern=[[None, None]] * 15), "every slot: 16"),
                (dict(bare, pattern=[[None]] * 16), "lists 2 layer"),
                (dict(bare, pattern=[[["A", 1.5], None]] * 16), "whole number of signed turns"),
                (coils(coil_pitch=0), "other than zero"),
                (coils(go_layer=2), "layer index from 0 to 1"),
                (coils(sequence=[["A", 2]]), "polarity"),
                (coils(coil_pitch=1, return_layer=1), "share layer 1"),
                (coils(first_slot=3), "outside the core"),
                ({k: v for k, v in coils().items() if k != "empty_material"}, "give an empty material")):
            with self.subTest(step={k: v for k, v in step.items() if k in ("pattern", "coil_pitch", "go_layer", "first_slot")}), \
                    self.assertRaisesRegex(ValueError, message):
                build([core(), step])


if __name__ == "__main__":
    unittest.main()
