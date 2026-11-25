"""Level 3 host tests: sampling, the paper and its quotes, flag credit, case files (no FEMM needed)."""
from __future__ import annotations

import os
from pathlib import Path
import random
import tempfile
import unittest

from femm_gym import level3, level3_score, level3_set


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


if __name__ == "__main__":
    unittest.main()
