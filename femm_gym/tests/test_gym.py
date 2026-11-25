"""Host tests for femm_gym: generation, screening, scoring Lua and reward arithmetic (no FEMM).

The native checks are `python3 -m femm_gym selfcheck`; set FEMM_GYM_NATIVE=1 to also run one here.
"""
import hashlib
import os
from pathlib import Path
import random
import tempfile
import unittest

from femm_brain.reproduce import check_regions, overlapping_arcs

from femm_gym import reward, testset
from femm_gym.families import FAMILIES, POINTS, _f


class Generation(unittest.TestCase):
    def test_every_family_builds_closed_labelled_regions(self):
        for name, family in FAMILIES.items():
            for seed in range(10):
                p = family.sample(random.Random(seed))
                self.assertEqual(p, family.sample(random.Random(seed)), name)
                for level, variant in ((None, False), (0, False), (2, False), (None, True)):
                    model = family.build(p, level, variant)
                    self.assertEqual(check_regions(model), [], f"{name} seed {seed} level {level} variant {variant}")
                    self.assertEqual(overlapping_arcs(model), [], name)
                wrong = family.wrong(p)
                self.assertNotEqual(wrong, p, name)
                self.assertEqual(check_regions(family.build(wrong)), [], f"{name} wrong control")

    def test_text_states_the_inputs(self):
        for name, family in FAMILIES.items():
            for seed in range(5):
                p = family.sample(random.Random(seed))
                text = family.text(p)
                self.assertIn("submission/model.fem", text, name)
                for k, v in p.items():  # counts and radii appear as given; lengths may appear as halves or sums
                    if k in ("slots", "poles", "turns", "mu", "coercivity", "depth", "remanence") or k.endswith("radius"):
                        self.assertIn(_f(v), text, f"{name}: {k} = {v} is not stated")

    def test_tasks_hide_the_key(self):
        task, key, text = testset.make_task(FAMILIES["l2_coil_inductance"], 3, testset.TEST_SPLIT)
        self.assertNotIn("params", task)
        self.assertNotIn("value", task)
        self.assertEqual((task, key, text), testset.make_task(FAMILIES["l2_coil_inductance"], 3, testset.TEST_SPLIT))

    def test_test_split_never_meets_training_draws(self):
        for name, family in FAMILIES.items():
            test = {repr(family.sample(random.Random(testset.seed_for(testset.TEST_SPLIT, name, i)))) for i in range(30)}
            train = {repr(family.sample(random.Random(testset.seed_for("train", name, i)))) for i in range(300)}
            self.assertFalse(test & train, name)


FEM = """[CircuitProps]  = 2
  <BeginCircuit>
    <CircuitName> = "coil"
  <EndCircuit>
  <BeginCircuit>
    <CircuitName> = "other"
  <EndCircuit>
[NumBlockLabels] = 2
1	0	1	-1	1	0	1	10	0
2	0	2	-1	0	0	0	1	0	0	"{expr}"
"""


class Screening(unittest.TestCase):
    def test_reads_circuits_labels_and_expressions(self):
        circuits, labels, strings = reward.read_fem(FEM.format(expr="theta+180"))
        self.assertEqual(circuits, ["coil", "other"])
        self.assertEqual(labels, [{"x": 1.0, "y": 0.0, "group": 1, "circuit": "coil"},
                                  {"x": 2.0, "y": 0.0, "group": 0, "circuit": None}])
        self.assertEqual(strings, ["theta+180"])

    def test_direction_expressions_may_only_do_arithmetic(self):
        for safe in ("theta", "theta+180", "atan2(y,x)*180/pi", "1e5*theta", "-90 + 2.5e-1*r"):
            self.assertTrue(reward.safe_expression(safe), safe)
        for unsafe in ('print("GYM")', "readfrom('score.lua')", "theta;x=1", "dofile(x)", "openfile"):
            self.assertFalse(reward.safe_expression(unsafe), unsafe)

    def test_contract_problems(self):
        task = {"circuit": "coil"}
        self.assertEqual(reward.screen(FEM.format(expr="theta"), task), ([], ["coil", "other"]))
        problems = reward.screen(FEM.format(expr='print("GYM x flux 1")'), task)[0]
        self.assertTrue(any("arithmetic" in p for p in problems))
        for name in ('"x") print("y"', '"a;b"'):
            problems = reward.screen(FEM.replace('"other"', name), task)[0]
            self.assertTrue(any("not plain" in p for p in problems), name)
        plain = FEM.format(expr="theta")
        self.assertEqual(reward.screen(plain, {"circuit": "winding"})[0], ["no circuit named 'winding'"])
        self.assertEqual(reward.screen(plain, {"circuit": "other"})[0], ["no block is assigned to circuit 'other'"])
        unassigned = plain.replace("1\t0\t1\t-1\t1", "1\t0\t1\t-1\t0")
        self.assertEqual(reward.screen(unassigned, task)[0], ["no block is assigned to circuit 'coil'"])

    def test_screening_reads_the_file_as_femm_does(self):
        task = {"circuit": "coil"}
        lower = FEM.format(expr="theta").replace('<CircuitName> = "other"', '<circuitname> = "a;b"')
        self.assertTrue(any("not plain" in p for p in reward.screen(lower, task)[0]))
        second = FEM.format(expr="theta") + '[numblocklabels] = 1\n3\t0\t2\t-1\t0\t0\t0\t1\t0\t0\t"dofile(x)"\n'
        self.assertTrue(any("arithmetic" in p for p in reward.screen(second, task)[0]))
        self.assertTrue(reward.screen(FEM.format(expr="theta") + "[PREVSOLN] = \"old.ans\"\n", task)[0])
        self.assertIn("could not be read", reward.screen(FEM.replace("[NumBlockLabels] = 2", "[NumBlockLabels] = two"),
                                                         task)[0][0])

    def test_unreadable_model_is_a_contract_failure(self):
        task = {"id": "t", "family": "l2_coil_inductance", "level": 2, "circuit": "coil", "measure": "inductance"}
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "model.fem").write_text("[NumBlockLabels] = 1\nx y\n")
            record = reward.score(task, {}, folder, Path(folder) / "work")
        self.assertEqual((record["stage"], record["reward"]), ("contract", 0.0))


class ScoringLua(unittest.TestCase):
    def test_inductance_script_sets_every_circuit(self):
        task = {"measure": "inductance", "circuit": "coil", "group": None}
        lines = reward.scoring_lines(task, {"current": 2.5}, ["coil", "other"], "abc")
        self.assertIn('mi_modifycircprop("coil", 1, 2.5)', lines)
        self.assertIn('mi_modifycircprop("other", 1, 0.0)', lines)
        self.assertTrue(any("mo_getcircuitproperties" in l for l in lines))

    def test_force_is_the_reaction_on_every_current(self):
        task = {"measure": "axial_force", "circuit": "coil"}
        lines = reward.scoring_lines(task, {"current": 1.5}, ["coil"], "abc")
        force = next(i for i, l in enumerate(lines) if "mo_blockintegral(12)" in l)
        self.assertEqual(lines[force - 1], "mo_groupselectblock()")  # all blocks: mo_selectblock toggles
        self.assertFalse(any("mo_selectblock" in l for l in lines))
        values, _ = reward.parse("GYM abc coil_force -0.75\nGYM abc done\n", "abc")
        self.assertEqual(reward.measured_value(task, values), 0.75)

    def test_map_script_samples_every_point(self):
        task, key, _ = testset.make_task(FAMILIES["l1_stator"], 0, "unit")
        lines = reward.scoring_lines(task, key, [], "abc")
        self.assertEqual(sum(l.startswith("P(") for l in lines), POINTS)

    def test_only_tagged_lines_count(self):
        output = "GYM fake flux 1\nGYM abc flux 0.25\nGYM abc current 2\nGYM abc done\nflux 9\n"
        values, _ = reward.parse(output, "abc")
        self.assertEqual(values, {"flux": 0.25, "current": 2.0, "done": True})
        self.assertAlmostEqual(reward.measured_value({"measure": "inductance"}, values), 0.125)

    def test_build_lines_save_the_model(self):
        model = FAMILIES["l2_coil_inductance"].build(FAMILIES["l2_coil_inductance"].sample(random.Random(0)))
        self.assertEqual(reward.build_lines(model)[-2:], ['mi_saveas("model.fem")', "mi_close()"])


class RewardArithmetic(unittest.TestCase):
    def test_value_reward(self):
        key = {"value": 2.0, "tolerance": 0.02}
        self.assertEqual(reward.value_result(key, 2.0)["reward"], 1.0)
        self.assertTrue(reward.value_result(key, 2.039)["passed"])
        miss = reward.value_result(key, 2.06)
        self.assertFalse(miss["passed"])
        self.assertAlmostEqual(miss["reward"], 0.5 * (1 - 0.1760913), places=3)
        self.assertEqual(reward.value_result(key, 3.0)["reward"], 0.0)

    def test_map_reward(self):
        key = {"classes": "0011-22", "signatures": [[1, 0], [1, 58], [2500, 0]], "tolerance": 0.95}
        same = {0: (1, 0), 1: (1, 0), 2: (1, 58), 3: (1, 58), 4: (1, 0), 5: (2500, 0), 6: (2500, 0)}
        self.assertEqual(reward.map_result(key, same)["reward"], 1.0)
        result = reward.map_result(key, {**same, 3: (1, 0)})
        self.assertFalse(result["passed"])
        self.assertEqual(result["iou"], {"0": 0.6667, "1": 0.5, "2": 1.0})
        self.assertEqual(reward.map_result(key, {**same, 2: None})["unrecognised_points"], 1)


class Play(unittest.TestCase):
    def test_workspace_holds_the_task_and_never_the_key(self):
        from femm_gym import play
        task, key, text = testset.make_task(FAMILIES["l2_magnet_force"], 0, "unit")
        with tempfile.TemporaryDirectory() as folder:
            data = b"FEMM manual text"
            manual = Path(folder) / "manual.txt"
            manual.write_bytes(data)
            with self.assertRaises(ValueError):
                play.prepare(task, text, Path(folder) / "other", 20, 30, 900, manual)  # not the pinned text
            workspace = play.prepare(task, text, Path(folder) / "trial", 20, 30, 900, manual,
                                     hashlib.sha256(data).hexdigest())
            files = sorted(str(p.relative_to(workspace)) for p in workspace.rglob("*"))
            self.assertEqual(files, ["femm", "femm/manual.txt", "submission", "task.json", "task.md"])
            self.assertNotIn("params", (workspace / "task.json").read_text())
            prompt = (Path(folder) / "trial" / "prompt.md").read_text()
            self.assertIn("radians", prompt)
            self.assertIn("30 runs and\n  900 seconds", prompt)
        with self.assertRaises(ValueError):
            play.play(task, key, text, "/opt/femm-gym-trial", "runner.py", "model")

    def test_submission_links_are_refused_unread(self):
        from femm_gym import play
        with tempfile.TemporaryDirectory() as folder:
            workspace = Path(folder) / "workspace"
            (workspace / "submission").mkdir(parents=True)
            secret = Path(folder) / "secret.json"
            secret.write_text('{"value": 1}')
            (workspace / "submission" / "model.fem").write_text("[Format] = 4.0\n")
            target, problems = play.checked_submission(workspace, Path(folder) / "ok")
            self.assertEqual((problems, sorted(p.name for p in target.iterdir())), ([], ["model.fem"]))
            (workspace / "submission" / "answer.json").symlink_to(secret)
            target, problems = play.checked_submission(workspace, Path(folder) / "linked")
            self.assertEqual(problems, ["submission/answer.json is a link or not a regular file"])
            self.assertFalse((target / "answer.json").exists())
            (workspace / "submission" / "answer.json").unlink()
            (workspace / "submission" / "model.fem").unlink()
            (workspace / "submission").rmdir()
            (workspace / "submission").symlink_to(Path(folder))  # the whole folder points outside
            secret.rename(Path(folder) / "model.fem")
            self.assertTrue(play.checked_submission(workspace, Path(folder) / "outside")[1])


@unittest.skipUnless(os.environ.get("FEMM_GYM_NATIVE") == "1", "set FEMM_GYM_NATIVE=1 to run FEMM")
class Native(unittest.TestCase):
    def test_coil_reference_converges(self):
        task, key, _ = testset.make_task(FAMILIES["l2_coil_inductance"], 0, "native-test")
        with tempfile.TemporaryDirectory() as work:
            key = testset.complete_key(task, key, Path(work) / "ref")
        self.assertIsNotNone(key["value"])
        self.assertLess(key["reference"]["converged"], 0.004)


if __name__ == "__main__":
    unittest.main()
