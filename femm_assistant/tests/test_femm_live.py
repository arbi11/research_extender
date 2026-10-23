"""Physical checks: only run against the real FEMM 5 executable, never a mock."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from app.jobs import JobStore, SolverWorker
from app.model import CCoreSpec


@unittest.skipUnless(os.getenv("FEMM_LIVE_TEST") == "1", "Set FEMM_LIVE_TEST=1 with native FEMM 5 installed")
class FemmLiveTests(unittest.TestCase):
    def test_linear_scaling_gap_sensitivity_and_mesh_refinement(self):
        command = os.getenv("FEMM_LUA", "femm-lua")
        self.assertIsNotNone(shutil.which(command), "FEMM 5 must be installed for live checks")
        with tempfile.TemporaryDirectory() as temp:
            store = JobStore(Path(temp))
            worker = SolverWorker(store, command=command, timeout=180)
            def solve(**changes):
                job = store.create(CCoreSpec(**changes))
                worker.run(job)
                completed = store.get(job["id"])
                log = store.root / job["id"] / "solver.log"
                self.assertEqual(completed["status"], "completed", f'{completed["error"]}\n{log.read_text() if log.exists() else "no solver log"}')
                self.assertGreater((store.root / job["id"] / "model.ans").stat().st_size, 100)
                self.assertTrue((store.root / job["id"] / "download.zip").is_file())
                return {key: value["value"] for key, value in completed["result"].items()}
            base = solve()
            doubled = solve(current_a=2)
            zero = solve(current_a=0)
            wider_gap = solve(gap_mm=2)
            refined = solve(mesh_mm=1)
            self.assertLess(base["force_x"], 0, "Armature must be attracted toward the core")
            self.assertGreater(base["energy"], 0)
            self.assertAlmostEqual(doubled["flux_linkage"] / base["flux_linkage"], 2, delta=.02)
            self.assertAlmostEqual(doubled["force_x"] / base["force_x"], 4, delta=.08)
            self.assertLess(abs(zero["force_x"]), 1e-6)
            self.assertLess(abs(zero["energy"]), 1e-9)
            self.assertLess(abs(wider_gap["force_x"]), abs(base["force_x"]))
            self.assertLess(abs(refined["force_x"] / base["force_x"] - 1), .1)
            # Linear magnetostatics: stored energy = 1/2 * I * flux linkage.
            self.assertLess(abs(base["energy"] / (.5 * base["current"] * base["flux_linkage"]) - 1), .03)
            print("FEMM 5 validation measurements:", json.dumps({"base": base, "double_current": doubled, "wider_gap": wider_gap, "refined": refined}, indent=2))
