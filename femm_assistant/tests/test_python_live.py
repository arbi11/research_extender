"""Run the exported Python CAD pipeline against an installed native FEMM 5."""
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from app.plan import wire_example
from app.plan_generate import required_results, source_files


@unittest.skipUnless(os.getenv("FEMM_LIVE_TEST") == "1", "Set FEMM_LIVE_TEST=1 with native FEMM 5 installed")
class PythonExportLiveTests(unittest.TestCase):
    def test_standalone_python_mesh_then_solve_matches_ampere_law(self):
        command = shutil.which(os.getenv("FEMM_LUA", "femm-lua"))
        self.assertIsNotNone(command, "FEMM 5 must be installed for live checks")
        plan = wire_example()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name, content in source_files(plan).items():
                (folder / name).write_text(content, encoding="utf-8")
            environment = dict(os.environ, FEMM_LUA=command)

            def run(*arguments):
                result = subprocess.run(
                    [sys.executable, str(folder / "model.py"), *arguments],
                    cwd=folder, env=environment, capture_output=True, text=True, timeout=210,
                )
                self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
                return result.stdout

            mesh_output = run("--mesh-only")
            self.assertGreater((folder / "model.fem").stat().st_size, 100)
            with (folder / "mesh.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(list(csv.reader(stream)), [["status"], ["complete"]])
            for name in ("model.ans", "results.csv", "field.csv"):
                self.assertFalse((folder / name).exists(), f"Mesh-only execution unexpectedly wrote {name}")

            # Stock FEMM keeps its mesh in memory. Our deployment's opt-in
            # native patch additionally exports the actual mesh for inspection.
            exported = [(folder / name).is_file() for name in ("model.node", "model.ele")]
            if os.getenv("FEMM_MESH_EXPORT") == "1" or any(exported):
                self.assertTrue(all(exported), "Native mesh export must contain both nodes and elements")
                for name in ("model.node", "model.ele"):
                    rows = [line.split("#", 1)[0].strip() for line in (folder / name).read_text().splitlines()]
                    rows = [line for line in rows if line]
                    count = int(rows[0].split()[0])
                    self.assertGreater(count, 10)
                    self.assertEqual(len(rows)-1, count)

            solve_output = run()
            self.assertGreater((folder / "model.ans").stat().st_size, 100)
            self.assertGreater((folder / "field.csv").stat().st_size, 100)
            with (folder / "results.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            measured = {row["quantity"]: float(row["value"]) for row in rows}
            self.assertEqual(len(measured), len(rows), "Measurement quantities must be unique")
            self.assertEqual({row["quantity"]: row["unit"] for row in rows}, required_results(plan))
            self.assertTrue(all(math.isfinite(value) for value in measured.values()))
            expected_b = 1e-4  # mu0 * 1 A / (2*pi*0.002 m).
            measured_b = math.hypot(measured["bx_0"], measured["by_0"])
            self.assertAlmostEqual(measured_b / expected_b, 1, delta=.08)
            self.assertAlmostEqual(measured["bx_0"], 0, delta=expected_b*.05)
            self.assertGreater(measured["by_0"], 0)
            self.assertGreater(measured["energy"], 0)
            print("Standalone Python FEMM verification:", json.dumps({
                "mesh_output": mesh_output.strip(), "native_mesh_exported": all(exported),
                "solve_output": solve_output.strip(), "analytic_b_at_2mm_t": expected_b,
                "measurements": measured,
            }, indent=2))


if __name__ == "__main__":
    unittest.main()
