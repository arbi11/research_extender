"""Real native FEMM axisymmetric checks; no model/API credentials required."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from app.jobs import JobStore, SolverWorker
from app.plan import GeometryPlan
from evaluation.axisymmetric import air_solenoid_plan, air_solenoid_reference


@unittest.skipUnless(os.getenv("FEMM_LIVE_TEST") == "1", "Set FEMM_LIVE_TEST=1 in the patched native FEMM image")
class AxisymmetricNativeTests(unittest.TestCase):
    def execute(self, store, plan):
        worker = SolverWorker(store, command=os.getenv("FEMM_LUA", "femm-lua"), timeout=180)
        mesh = store.create(plan, kind="mesh")
        worker.run(mesh)
        meshed = store.get(mesh["id"])
        self.assertEqual(meshed["status"], "completed", meshed["error"])
        self.assertEqual(meshed["result"]["status"], "passed")
        self.assertFalse((store.root / mesh["id"] / "model.ans").exists())
        solve = store.create(plan)
        worker.run(solve)
        solved = store.get(solve["id"])
        log = (store.root / solve["id"] / "solver.log").read_text()
        self.assertEqual(solved["status"], "completed", f'{solved["error"]}\n{log}')
        native = (store.root / solve["id"] / "model.fem").read_text()
        self.assertRegex(native, r"(?mi)^\[ProblemType\]\s*=\s*axisymmetric\s*$")
        values = {key: value["value"] for key, value in solved["result"].items()}
        self.assertGreater(values["energy"], 0)
        # This is a continuum consistency check, not an algebraic solver
        # residual. FEMM's pinned results_doc.cpp integrates B.H through
        # AxiFluxFit/AxiDensityIntegral, whereas GetFluxLinkage uses AxiInt
        # on nodal A=phi/(2*pi*r), J. These are distinct finite-element
        # interpolation/quadrature paths. Require agreement within 0.01%;
        # refinement of both winding and surrounding air is checked below.
        self.assertAlmostEqual(values["energy"], .5*values["current_coil"]*values["flux_linkage_coil"],
                               delta=values["energy"]*1e-4)
        return values

    def test_finite_air_solenoid_reference_refinement_domain_and_current_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(Path(directory))
            variants = [
                # Refine the air around axis probes as well as the winding.
                # This same-material air contour changes only discretization.
                ("base", air_solenoid_plan(domain_mm=320, nearfield_mesh_mm=2)),
                ("refined", air_solenoid_plan(domain_mm=320, winding_mesh_mm=.25, nearfield_mesh_mm=1)),
                # Hold all mesh sizes fixed: only the outer boundary moves.
                ("far", air_solenoid_plan(domain_mm=640, winding_mesh_mm=.25, nearfield_mesh_mm=1)),
                ("double_current", air_solenoid_plan(current_a=2, domain_mm=320, winding_mesh_mm=.25, nearfield_mesh_mm=1)),
            ]
            measured = {name: self.execute(store, plan) for name, plan in variants}
            for name, values in measured.items():
                current = 2 if name == "double_current" else 1
                for index, z in enumerate((0, 20, 50)):
                    expected = air_solenoid_reference(z, current)
                    self.assertLess(abs(values[f"bz_{index}"]/expected-1), .035, (name, index, values, expected))
                    self.assertLess(abs(values[f"br_{index}"])/expected, .01, (name, index, values))
            for index in range(3):
                base, fine, far = [measured[key][f"bz_{index}"] for key in ("base", "refined", "far")]
                self.assertLess(abs(base/fine-1), .02)
                self.assertLess(abs(far/fine-1), .02)
                self.assertAlmostEqual(measured["double_current"][f"bz_{index}"]/fine, 2, delta=.005)
            energy_errors = {name: abs(values["energy"]/(.5*values["current_coil"]*values["flux_linkage_coil"])-1)
                             for name, values in measured.items()}
            # The two postprocessing routes must approach one another when
            # refining the physical current region and its surrounding field.
            self.assertLess(energy_errors["refined"], energy_errors["base"])
            self.assertAlmostEqual(measured["double_current"]["energy"]/measured["refined"]["energy"], 4, delta=.01)
            self.assertAlmostEqual(measured["double_current"]["flux_linkage_coil"]/measured["refined"]["flux_linkage_coil"], 2, delta=.005)
            print("Axisymmetric Biot-Savart, mesh/domain and current references:", json.dumps({
                "expected_bz_t": [air_solenoid_reference(z) for z in (0, 20, 50)],
                "relative_energy_consistency_errors": energy_errors, "measurements": measured}, indent=2))

    def test_iron_region_can_share_axis_without_overlapping_mesh_segments(self):
        data = air_solenoid_plan(mesh_mm=2, winding_mesh_mm=.25, nearfield_mesh_mm=.5).model_dump()
        data["materials"].append({"id": "iron", "relative_permeability": 1000})
        data["regions"].append({"id": "core", "shape": {"type": "rectangle", "x_min": 0,
                                "x_max": 10, "y_min": -30, "y_max": 30}, "material": "iron",
                                "label": {"x": 5, "y": 0}, "mesh_mm": .25})
        data["probes"] = [{"x": 0, "y": 0}]
        data["force_region"] = "core"
        with tempfile.TemporaryDirectory() as directory:
            result = self.execute(JobStore(Path(directory)), GeometryPlan.model_validate(data))
            self.assertGreater(result["bz_0"], air_solenoid_reference(0))
            self.assertLess(abs(result["force_z"]), .05, "Centred symmetric core has zero axial net force")
            self.assertNotIn("force_r", result)
            print("Axis-touching iron core:", json.dumps(result, indent=2))
