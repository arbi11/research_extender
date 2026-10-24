"""Real-solver checks of new geometry topology against Ampere's law."""
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

from app.jobs import JobStore, SolverWorker
from app.plan import GeometryPlan, wire_example


@unittest.skipUnless(os.getenv("FEMM_LIVE_TEST") == "1", "Set FEMM_LIVE_TEST=1 with native FEMM 5 installed")
class PlanLiveTests(unittest.TestCase):
    def test_round_wire_matches_ampere_law_and_current_scaling(self):
        command = os.getenv("FEMM_LUA", "femm-lua")
        self.assertIsNotNone(shutil.which(command), "FEMM 5 must be installed for live checks")
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            worker = SolverWorker(store, command=command, timeout=180)

            def solve(current=1, mesh=.5):
                data = wire_example().model_dump()
                data["mesh_mm"] = mesh
                data["circuits"][0]["current_a"] = current
                mesh_job = store.create(GeometryPlan.model_validate(data), kind="mesh")
                worker.run(mesh_job)
                meshed = store.get(mesh_job["id"])
                mesh_folder = store.root / mesh_job["id"]
                mesh_log = (mesh_folder / "solver.log").read_text() if (mesh_folder / "solver.log").exists() else "no mesh log"
                self.assertEqual(meshed["status"], "completed", f'{meshed["error"]}\n{mesh_log}')
                self.assertEqual(meshed["result"]["status"], "passed")
                self.assertGreater(meshed["result"]["element_count"], 10)
                self.assertFalse((mesh_folder / "model.ans").exists(), "Mesh validation must not solve fields")
                self.assertFalse((mesh_folder / "results.csv").exists())
                self.assertTrue((mesh_folder / "mesh.svg").is_file())
                job = store.create(GeometryPlan.model_validate(data))
                worker.run(job)
                completed = store.get(job["id"])
                folder = store.root / job["id"]
                log = (folder / "solver.log").read_text() if (folder / "solver.log").exists() else "no solver log"
                self.assertEqual(completed["status"], "completed", f'{completed["error"]}\n{log}')
                self.assertGreater((folder / "model.ans").stat().st_size, 100)
                self.assertTrue((folder / "download.zip").is_file())
                with zipfile.ZipFile(folder / "download.zip") as archive:
                    self.assertIn("model.py", archive.namelist())
                    self.assertIn("mesh.svg", archive.namelist())
                    self.assertEqual(json.loads(archive.read("mesh-report.json"))["status"], "passed")
                return {name: quantity["value"] for name, quantity in completed["result"].items()}

            base = solve()
            refined = solve(mesh=.25)
            doubled = solve(current=2)
            expected = 4 * math.pi * 1e-7 / (2 * math.pi * .002)
            base_b = math.hypot(base["bx_0"], base["by_0"])
            refined_b = math.hypot(refined["bx_0"], refined["by_0"])
            doubled_b = math.hypot(doubled["bx_0"], doubled["by_0"])
            self.assertAlmostEqual(base_b / expected, 1, delta=.08)
            self.assertAlmostEqual(refined_b / expected, 1, delta=.05)
            self.assertAlmostEqual(doubled_b / base_b, 2, delta=.04)
            self.assertGreater(base["energy"], 0)
            self.assertAlmostEqual(doubled["energy"] / base["energy"], 4, delta=.08)
            self.assertAlmostEqual(base["bx_0"], 0, delta=expected * .05)
            self.assertGreater(base["by_0"], 0)
            print("Generic geometry-plan FEMM measurements:", json.dumps({
                "analytic_b_at_2mm_t": expected, "base": base, "refined": refined, "double_current": doubled,
            }, indent=2))


if __name__ == "__main__":
    unittest.main()
