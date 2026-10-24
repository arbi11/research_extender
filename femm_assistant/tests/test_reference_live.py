"""Native reference cases derived independently from Ampere's law.

OpenStax University Physics 2, section 12.5 (thick wire, Example 12.7):
https://openstax.org/books/university-physics-volume-2/pages/12-5-amperes-law
The coaxial extension integrates uniform signed current over concentric regions.
These references check the supported planar model, not finite-wire end effects.
"""
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

from app.jobs import JobStore, SolverWorker
from app.plan import GeometryPlan, wire_example


@unittest.skipUnless(os.getenv("FEMM_LIVE_TEST") == "1", "Set FEMM_LIVE_TEST=1 inside the patched FEMM image")
class IndependentReferenceTests(unittest.TestCase):
    def execute(self, store, data):
        plan = GeometryPlan.model_validate(data)
        worker = SolverWorker(store, command=os.getenv("FEMM_LUA", "femm-lua"), timeout=180)
        # Require the same actual mesh diagnostic used by the general API.
        mesh = store.create(plan, kind="mesh")
        worker.run(mesh)
        meshed = store.get(mesh["id"])
        self.assertEqual(meshed["status"], "completed", meshed["error"])
        self.assertEqual(meshed["result"]["status"], "passed")
        job = store.create(plan)
        worker.run(job)
        solved = store.get(job["id"])
        self.assertEqual(solved["status"], "completed", solved["error"])
        return {name: value["value"] for name, value in solved["result"].items()}

    def test_thick_wire_inside_outside_refinement_and_domain_distance(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            measurements = []
            # Only mesh size changes in the first pair; only domain distance
            # changes in the second. A centered circle is a symmetry reference,
            # so agreement here is not a general open-boundary convergence claim.
            for mesh, radius in ((.4, 10), (.2, 10), (.2, 20)):
                data = wire_example().model_dump(mode="json")
                data.update(mesh_mm=mesh, probes=[{"x": .5, "y": 0}, {"x": 2, "y": 0},
                                                  {"x": 0, "y": 4}, {"x": -2, "y": 0}])
                data["domain"]["radius"] = radius
                data["regions"][0]["mesh_mm"] = mesh / 2
                result = self.execute(store, data)
                # B = mu0 I r/(2 pi a^2) inside, mu0 I/(2 pi r) outside.
                expected = [(0, 1e-4), (0, 1e-4), (-5e-5, 0), (0, -1e-4)]
                errors = []
                for i, (bx, by) in enumerate(expected):
                    magnitude = math.hypot(bx, by)
                    error = math.hypot(result[f"bx_{i}"] - bx, result[f"by_{i}"] - by) / magnitude
                    self.assertLess(error, .08 if mesh == .4 else .05, f"probe {i}: {result}")
                    errors.append(error)
                measurements.append({"mesh_mm": mesh, "domain_radius_mm": radius,
                                     "vector_relative_errors": errors, "result": result})
            for i in range(4):
                coarse, fine, far = [item["result"] for item in measurements]
                scale = math.hypot(fine[f"bx_{i}"], fine[f"by_{i}"])
                self.assertLess(math.hypot(fine[f"bx_{i}"]-coarse[f"bx_{i}"], fine[f"by_{i}"]-coarse[f"by_{i}"])/scale, .05)
                self.assertLess(math.hypot(far[f"bx_{i}"]-fine[f"bx_{i}"], far[f"by_{i}"]-fine[f"by_{i}"])/scale, .03)
            print("Thick-wire refinement and domain measurements:", json.dumps(measurements, indent=2))

    def test_coaxial_return_current_cancels_external_field(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            data = wire_example().model_dump(mode="json")
            data.update(mesh_mm=.25, probes=[{"x": .5, "y": 0}, {"x": 2, "y": 0},
                                             {"x": 3.5, "y": 0}, {"x": 6, "y": 0}])
            data["regions"][0]["mesh_mm"] = .1
            # Nested contour faces: radius-4 return shell, radius-3 air cavity,
            # radius-1 source. Labels belong to their actual annular faces.
            data["regions"].extend([
                {"id": "return_shell", "shape": {"type": "circle", "center": {"x": 0, "y": 0}, "radius": 4},
                 "material": "conductor", "label": {"x": 3.5, "y": 0}, "circuit": "return_current", "turns": -1, "mesh_mm": .15},
                {"id": "cavity", "shape": {"type": "circle", "center": {"x": 0, "y": 0}, "radius": 3},
                 "material": "air", "label": {"x": 2, "y": 0}, "mesh_mm": .15},
            ])
            data["circuits"].append({"id": "return_current", "current_a": 1})
            result = self.execute(store, data)
            # In the return shell, I_enclosed = I[1-(r^2-b^2)/(c^2-b^2)].
            enclosed = 1 - (3.5**2 - 3**2) / (4**2 - 3**2)
            expected = [1e-4, 1e-4, 2e-7 * enclosed / .0035]
            for i, by in enumerate(expected):
                self.assertLess(math.hypot(result[f"bx_{i}"], result[f"by_{i}"]-by) / by, .05)
            self.assertLess(math.hypot(result["bx_3"], result["by_3"]), 1e-6,
                            "Equal opposite total currents must cancel the exterior field")
            print("Coaxial return-current reference:", json.dumps({"expected_by_t": expected + [0], "result": result}, indent=2))


if __name__ == "__main__":
    unittest.main()
