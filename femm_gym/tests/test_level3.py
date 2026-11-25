"""Level 3 host tests: sampling, the paper and its quotes, flag credit, case files (no FEMM needed)."""
from __future__ import annotations

import math
import os
from pathlib import Path
import random
import tempfile
import unittest

from femm_gym import bench, level3, level3_score, level3_set


def _key(seed):
    """A key as make_key builds it, with made-up reference values (no solve)."""
    rng = random.Random(seed)
    machine = level3.sample(rng)
    plan = level3.presentation(machine, rng)
    published = level3.publish(machine, {"E01": 101.234, "T01": 5.4321, "T02": 10.5}, plan)
    text, quotes = level3.write_paper(published, plan)
    return {"id": f"t{seed}", "family": level3.FAMILY, "level": 3, "machine": published, "plan": plan,
            "expectations": level3.expectations(published, plan), "quotes": {k: list(v) for k, v in quotes.items()},
            "paper": text}


class Sampling(unittest.TestCase):
    def test_same_seed_same_machine(self):
        self.assertEqual(level3.sample(random.Random(5)), level3.sample(random.Random(5)))

    def test_radial_build_adds_up(self):
        for seed in range(20):
            m = level3.sample(random.Random(seed))
            self.assertAlmostEqual(m["rotor_core_radius"] + m["magnet_thickness"] + m["air_gap"], m["stator_inner_radius"])
            self.assertAlmostEqual(m["slot_bottom_radius"] + m["yoke_thickness"], m["stator_outer_radius"])
            self.assertIn((m["slots"], m["poles"]), level3.COMBOS)
            self.assertLess(m["magnet_arc"], 360 / m["poles"])

    def test_every_combination_draws(self):
        for slots, poles in level3.COMBOS:
            machine = dict(level3.sample(random.Random(0)), slots=slots, poles=poles, pole_pairs=poles // 2)
            machine["magnet_arc"] = 0.8 * 360 / poles
            model = level3.build(machine, level=0)
            self.assertEqual([c["name"] for c in model["circuits"]], ["A", "B", "C"])
            self.assertEqual(len(level3.coil_sequence(machine)), slots)

    def test_parallel_paths_only_when_coils_split_evenly(self):
        for seed in range(40):
            m = level3.sample(random.Random(seed))
            self.assertEqual((m["slots"] // 3) % m["parallel_paths"], 0)

    def test_slot_area_is_the_drawn_slot(self):
        # The two coil sides of the slot after tooth 1, measured on the drawing itself.
        for slots, poles in level3.COMBOS:
            machine = dict(level3.sample(random.Random(0)), slots=slots, poles=poles, pole_pairs=poles // 2)
            machine["magnet_arc"] = 0.8 * 360 / poles
            model = level3.build(machine, level=0)
            pitch = 2 * math.pi / slots
            sides = [l for l in model["labels"] if l["material"] == "Copper"
                     and 0 < math.atan2(l["y"], l["x"]) % (2 * math.pi) < pitch]
            self.assertEqual(len(sides), 2)
            drawn = sum(_region_area(model, (l["x"], l["y"])) for l in sides)
            self.assertAlmostEqual(level3._slot_area(machine) / drawn, 1, places=9, msg=f"{slots}/{poles}")

    def test_printed_fill_and_density_give_the_drawn_slot(self):
        for seed in range(30):
            m = level3.sample(random.Random(seed))
            implied = 2 * m["turns_per_coil"] * m["rated_current"] / m["parallel_paths"] / (
                m["current_density"] * m["fill_factor"])
            # Rounding: current to 3 figures, density to 0.1 A/mm2 (>= 4), fill to 0.01 (>= 0.40).
            self.assertLess(abs(implied / level3._slot_area(m) - 1), 0.031, msg=f"seed {seed}")


def _region_area(model, point):
    """Area of the drawn region holding `point`: the smallest bounded face of the drawing around it.

    The faces are traced from the model's segments and arcs (an arc runs counter-clockwise from n0
    to n1); each arc adds its circular segment to the polygon of its chord. The region has no holes.
    """
    nodes = [(n["x"], n["y"]) for n in model["nodes"]]
    edges = [(s["n0"], s["n1"], 0.0) for s in model["segments"]] + [(a["n0"], a["n1"], a["angle"]) for a in model["arcs"]]
    out = {}  # node -> [(departure angle, half-edge)]
    for k, (a, b, angle) in enumerate(edges):
        (xa, ya), (xb, yb) = nodes[a], nodes[b]
        chord = math.atan2(yb - ya, xb - xa)
        turn = math.radians(angle) / 2  # an arc leaves n0 turned right of its chord by half its angle
        out.setdefault(a, []).append((chord - turn, (k, 1)))
        out.setdefault(b, []).append((chord + math.pi + turn, (k, -1)))
    for v in out:
        out[v].sort(key=lambda t: t[0] % (2 * math.pi))

    def ends(h):
        a, b, _ = edges[h[0]]
        return (a, b) if h[1] == 1 else (b, a)

    def follow(h):
        """The next half-edge with the face on the left: clockwise after the reverse of h at its end."""
        v = ends(h)[1]
        ring = [e for _, e in out[v]]
        return ring[(ring.index((h[0], -h[1])) - 1) % len(ring)]

    def polyline(h):
        a, b, angle = edges[h[0]]
        (xa, ya), (xb, yb) = nodes[a], nodes[b]
        if not angle:
            pts = [(xa, ya), (xb, yb)]
        else:
            t = math.radians(angle)
            half = math.hypot(xb - xa, yb - ya) / 2
            r, mx, my = half / math.sin(t / 2), (xa + xb) / 2, (ya + yb) / 2
            d = half / math.tan(t / 2)
            ux, uy = (xb - xa) / (2 * half), (yb - ya) / (2 * half)
            cx, cy = mx - uy * d, my + ux * d
            a0 = math.atan2(ya - cy, xa - cx)
            pts = [(cx + r * math.cos(a0 + t * i / 64), cy + r * math.sin(a0 + t * i / 64)) for i in range(65)]
        return pts if h[1] == 1 else pts[::-1]

    best, seen = None, set()
    for start in [(k, s) for k in range(len(edges)) for s in (1, -1)]:
        if start in seen:
            continue
        face, h = [], start
        while h not in seen:
            seen.add(h)
            face.append(h)
            h = follow(h)
        area = 0.0
        for h in face:
            a, b = ends(h)
            (x1, y1), (x2, y2) = nodes[a], nodes[b]
            area += (x1 * y2 - x2 * y1) / 2
            angle = edges[h[0]][2]
            if angle:
                t = math.radians(angle)
                r = math.hypot(x2 - x1, y2 - y1) / 2 / math.sin(t / 2)
                area += h[1] * r * r / 2 * (t - math.sin(t))
        if area <= 0:
            continue
        ring = [p for h in face for p in polyline(h)]
        inside = False
        for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
            if (y1 > point[1]) != (y2 > point[1]) and point[0] < x1 + (point[1] - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
        if inside and (best is None or area < best):
            best = area
    return best


class Paper(unittest.TestCase):
    def test_reference_inputs_earn_full_flag_credit(self):
        for seed in range(25):
            key = _key(seed)
            credit, rows = level3_score.flag_credit(level3_score.reference_inputs(key), key["expectations"], key["paper"])
            self.assertEqual(credit, 1.0, {k: v for k, v in rows.items() if v["credit"] < 1})

    def test_every_vocabulary_name_is_known(self):
        names = {n for n, _, _ in level3.VOCABULARY}
        for seed in range(10):
            self.assertLessEqual(set(_key(seed)["expectations"]), names)

    def test_stated_values_are_printed_and_hidden_ones_are_not_quoted(self):
        for seed in range(25):
            key = _key(seed)
            for name, exp in key["expectations"].items():
                if exp["expect"] == "quoted":
                    quote, _ = key["quotes"][name]
                    self.assertIn(level3_score.normalize(quote), level3_score.normalize(key["paper"]), name)
                else:
                    self.assertNotIn(name, key["quotes"], name)

    def test_task_text_lists_every_input(self):
        text = level3.task_md(level3.sample(random.Random(3)))
        for name, _, _ in level3.VOCABULARY:
            self.assertIn(f"`{name}`", text)


class FlagCredit(unittest.TestCase):
    def setUp(self):
        self.key = _key(11)
        self.honest = level3_score.reference_inputs(self.key)

    def credit(self, doc):
        return level3_score.flag_credit(doc, self.key["expectations"], self.key["paper"])[0]

    def test_marking_everything_assumed_does_not_pay(self):
        doc = {"inputs": [dict(i, provenance="assumed", reason="x") for i in self.honest["inputs"]]}
        self.assertLess(self.credit(doc), 0.5)

    def test_invented_quotes_do_not_pay(self):
        doc = {"inputs": [dict(i, provenance="quoted", quote=f"{i['name']} = {i['value']}", location="x")
                          for i in self.honest["inputs"]]}
        self.assertLess(self.credit(doc), 0.5)

    def test_a_real_quote_with_the_wrong_value_fails(self):
        items = [dict(i) for i in self.honest["inputs"]]
        stack = next(i for i in items if i["name"] == "stack_length")
        stack["value"] = stack["value"] + 5
        rows = level3_score.flag_credit({"inputs": items}, self.key["expectations"], self.key["paper"])[1]
        self.assertEqual(rows["stack_length"]["credit"], 0.0)

    def test_missing_inputs_cost_credit(self):
        doc = {"inputs": self.honest["inputs"][:-3]}
        self.assertLess(self.credit(doc), 1.0)
        self.assertEqual(self.credit(None), 0.0)

    def test_quotes_survive_table_marks_and_case(self):
        item = {"value": 65, "quote": "STACK LENGTH 65 mm"}
        self.assertTrue(level3_score.quote_holds(item, "| Stack length | 65 mm |"))
        self.assertFalse(level3_score.quote_holds(item, "| Stack length | 60 mm |"))
        self.assertFalse(level3_score.quote_holds({"value": 32.5, "quote": "Stator outer diameter | 65 mm"},
                                                  "| Stator outer diameter | 65 mm |"))

    def test_coil_sequences_compare_by_signed_coils(self):
        self.assertTrue(level3_score.value_matches("+A, -A, -B, +B", "+A -A -B +B", "coil_sequence"))
        self.assertFalse(level3_score.value_matches("+A -A +B -B", "+A -A -B +B", "coil_sequence"))
        self.assertTrue(level3_score.quote_holds({"value": "+A -A -B", "quote": "Coil | +A | -A | -B"},
                                                 "| Coil | +A | -A | -B |", "coil_sequence"))


class Case(unittest.TestCase):
    def test_drives_targets_and_checks(self):
        m = level3.sample(random.Random(2))
        doc = level3.drive_doc(m)
        self.assertEqual(sorted(doc["targets"]), ["E01", "T01", "T02"])
        self.assertEqual({d["id"] for d in doc["drives"]}, {"nl", "ld1", "ld2"})
        self.assertEqual({c["kind"] for c in doc["checks"]}, {"depth_mm", "magnet_br_T", "turns_per_slot_side"})
        load = {d["id"]: d for d in doc["drives"]}
        self.assertAlmostEqual(load["ld2"]["vars"]["I"], level3.OVERLOAD * load["ld1"]["vars"]["I"])
        only = level3.drive_doc(m, ("E01",))
        self.assertEqual([d["id"] for d in only["drives"]], ["nl"])
        self.assertEqual(only["checks"], [])

    def test_zero_and_sense(self):
        m = {"pole_pairs": 5}
        self.assertEqual(level3_score.zero_and_sense(m, 180.0, -120.0), (36.0, 1))
        self.assertEqual(level3_score.zero_and_sense(m, -90.0, 119.0), (18.0, -1))
        with self.assertRaises(RuntimeError):
            level3_score.zero_and_sense(m, 0.0, 60.0)

    def test_enough_inputs_are_hidden(self):
        for index in range(6):
            _, machine, plan = level3_set.draw(index, "unit-test")
            exp = level3.expectations(dict(machine, emf_printed=1, torque_printed=1), plan)
            self.assertGreaterEqual(level3_set.hidden_count(exp), level3_set.MIN_HIDDEN)


class TaskFolder(unittest.TestCase):
    def test_write_task_makes_its_folders_and_keeps_the_key_out(self):
        key = _key(4)
        with tempfile.TemporaryDirectory() as tmp:
            level3_set.write_task(tmp, {"case": "x"}, key, "task text")
            folder = Path(tmp) / "tasks" / key["id"]
            self.assertEqual(sorted(p.name for p in folder.rglob("*") if p.is_file()),
                             ["paper.md", "task.json", "task.md"])
            self.assertTrue((Path(tmp) / "keys" / f"{key['id']}.json").is_file())


class Submission(unittest.TestCase):
    def test_links_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "submission"
            source.mkdir()
            (source / "model.fem").write_text("x")
            os.symlink("/etc/hostname", source / "entities.json")
            taken, problems = level3_score.take_submission(source, Path(tmp) / "taken")
            self.assertTrue(any("entities.json" in p for p in problems))
            self.assertTrue((taken / "model.fem").is_file())
            self.assertFalse((taken / "entities.json").exists())


class BenchScorer(unittest.TestCase):
    """The benchmark's own scorer, as Level 3 uses it (skipped without a daro_bench folder)."""

    def setUp(self):
        try:
            self.score = bench.scorer()
        except FileNotFoundError as error:
            self.skipTest(str(error))

    def test_moving_groups_move_together(self):
        # Moved one group at a time, a label of the first could sit for a moment on a label of
        # the second, and FEMM drops it (the Level 3 check, 2026-11-25).
        drive = {"id": "d", "measure": ["torque"], "axes": {"p": [0, 11.25]}, "position": "p"}
        model = {"header": {"LengthUnits": "millimeters"}, "boundaries": [], "circuits": []}
        for kind, extra in (("rotate", {"center": [0, 0]}), ("translate", {"axis": "x"})):
            entry = {"motion": {"kind": kind, "groups": [1, 2], **extra}}
            lua = self.score.drive_lua(drive, self.score.expand_states(drive), entry, model, "model.fem")
            steps = [line.strip() for line in lua[lua.index("function solve"):].splitlines()
                     if line.strip().startswith(("mi_selectgroup", "mi_move", "mi_clearselected"))]
            self.assertEqual(steps, ["mi_selectgroup(1)", "mi_selectgroup(2)", steps[2], "mi_clearselected()"], kind)
            self.assertTrue(steps[2].startswith("mi_move" + kind), kind)


if __name__ == "__main__":
    unittest.main()
