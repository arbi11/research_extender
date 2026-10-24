"""Axisymmetric model contract, topology, compiler and independent references."""
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from pydantic import ValidationError

from app.plan import GeometryPlan, axis_interior, contours_intersect_off_axis, Point, Rectangle, wire_example
from app.plan_generate import lua_script, required_results, source_files
from evaluation.axisymmetric import air_solenoid_plan, air_solenoid_reference, rectangular_winding_axis_bz


def core_plan():
    data = air_solenoid_plan().model_dump()
    data["materials"].append({"id": "iron", "relative_permeability": 1000})
    data["regions"].append({"id": "core", "shape": {"type": "rectangle", "x_min": 0,
                            "x_max": 10, "y_min": -30, "y_max": 30}, "material": "iron",
                            "label": {"x": 5, "y": 0}, "circuit": None, "turns": 0, "mesh_mm": .5})
    data["force_region"] = "core"
    return GeometryPlan.model_validate(data)


class AxisymmetricContractTests(unittest.TestCase):
    def test_axisymmetric_depth_is_not_a_planar_extrusion(self):
        data = air_solenoid_plan().model_dump()
        self.assertIsNone(data["depth_mm"])
        data.pop("depth_mm")
        self.assertIsNone(GeometryPlan.model_validate(data).depth_mm)
        data["depth_mm"] = 1
        with self.assertRaisesRegex(ValidationError, "full 360-degree volume"):
            GeometryPlan.model_validate(data)
        planar = wire_example().model_dump()
        planar.pop("depth_mm")
        with self.assertRaisesRegex(ValidationError, "Planar models require"):
            GeometryPlan.model_validate(planar)

    def test_negative_radius_and_axis_tangency_are_rejected(self):
        for mutate in (lambda value: value["domain"].update(x_min=-1),
                       lambda value: value["regions"][0]["shape"].update(x_min=-1),
                       lambda value: value["probes"][0].update(x=-1)):
            data = air_solenoid_plan().model_dump()
            mutate(data)
            with self.assertRaises(ValidationError):
                GeometryPlan.model_validate(data)
        data = air_solenoid_plan().model_dump()
        data["regions"][0]["shape"] = {"type": "circle", "center": {"x": 10, "y": 0}, "radius": 10}
        with self.assertRaisesRegex(ValidationError, "straight axis edge"):
            GeometryPlan.model_validate(data)

    def test_shared_axis_is_valid_but_other_boundary_contacts_are_not(self):
        plan = core_plan()
        self.assertEqual(plan.regions[1].shape.x_min, 0)
        data = plan.model_dump()
        data["regions"][1]["shape"]["x_max"] = data["domain"]["x_max"]
        with self.assertRaises(ValidationError):
            GeometryPlan.model_validate(data)
        data = plan.model_dump()
        data["regions"][1]["shape"]["x_max"] = 20
        with self.assertRaisesRegex(ValidationError, "cross or touch"):
            GeometryPlan.model_validate(data)

    def test_axis_contacts_do_not_hide_crossings(self):
        first = Rectangle(type="rectangle", x_min=0, x_max=10, y_min=-10, y_max=10)
        contained = Rectangle(type="rectangle", x_min=0, x_max=5, y_min=-5, y_max=5)
        overlapping = Rectangle(type="rectangle", x_min=0, x_max=15, y_min=-5, y_max=5)
        self.assertFalse(contours_intersect_off_axis(first, contained))
        self.assertTrue(contours_intersect_off_axis(first, overlapping))
        self.assertTrue(axis_interior(first, Point(x=0, y=0)))
        self.assertFalse(axis_interior(first, Point(x=0, y=10)))

    def test_axis_probes_are_allowed_away_from_material_interfaces(self):
        core_plan()  # centre probe lies on the core's symmetry edge, not an interface
        data = core_plan().model_dump()
        data["probes"] = [{"x": 0, "y": 30}]
        with self.assertRaisesRegex(ValidationError, "material interfaces"):
            GeometryPlan.model_validate(data)
        data["probes"] = [{"x": 0, "y": 160}]
        with self.assertRaises(ValidationError):
            GeometryPlan.model_validate(data)

    def test_nested_axis_faces_keep_correct_label_ownership(self):
        data = core_plan().model_dump()
        data["force_region"] = None
        data["regions"].append({"id": "inner", "shape": {"type": "rectangle", "x_min": 0,
                                "x_max": 2, "y_min": -10, "y_max": 10}, "material": "air",
                                "label": {"x": 1, "y": 0}})
        GeometryPlan.model_validate(data)
        data["regions"][1]["label"] = {"x": 1, "y": 0}
        with self.assertRaisesRegex(ValidationError, "nested contour"):
            GeometryPlan.model_validate(data)


class AxisymmetricCompilerTests(unittest.TestCase):
    def test_native_problem_and_results_use_axisymmetric_semantics(self):
        plan = core_plan()
        script = lua_script(plan)
        self.assertIn('mi_probdef(0,"millimeters","axi",1e-08,0,30)', script)
        self.assertIn('write(out,"br_0,",bx,",T\\n")', script)
        self.assertIn('write(out,"bz_0,",by,",T\\n")', script)
        self.assertIn('write(out,"force_z,",fz,",N\\n")', script)
        self.assertNotIn("mo_blockintegral(18)", script)
        self.assertNotIn("force_r", script)
        self.assertEqual(required_results(plan), {"energy": "J", "force_z": "N", "current_coil": "A",
                          "flux_linkage_coil": "Wb-turn", **{f"b{component}_{i}": "T" for i in range(3) for component in ("r", "z")}})
        self.assertIn("energy = mo_blockintegral(2)", script)

    def test_axis_edges_are_split_deduplicated_and_all_get_boundary(self):
        script = lua_script(core_plan(), mesh_only=True)
        segments = re.findall(r"mi_addsegment\(([^)]+)\)", script)
        axis = [tuple(map(float, row.split(","))) for row in segments if row.split(",")[0] in ("0", "0.0") and row.split(",")[2] in ("0", "0.0")]
        intervals = sorted(tuple(sorted((row[1], row[3]))) for row in axis)
        self.assertEqual(intervals, [(-160, -30), (-30, 30), (30, 160)])
        # Every interval, including the material's symmetry edge, is A=0.
        for z in (-95, 0, 95):
            self.assertIn(f"mi_selectsegment(0.0,{float(z)})", script)
        self.assertNotIn("mi_analyze", script)

    def test_exported_python_reproduces_axisymmetric_server_compilation(self):
        plan = core_plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, value in source_files(plan).items():
                (root / name).write_text(value)
            process = subprocess.run([sys.executable, "model.py", "--write-only"], cwd=root,
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual((root / "model.lua").read_text(), lua_script(plan))
            readme = (root / "README.txt").read_text()
            self.assertIn("full 360-degree", readme.replace("complete 360-degree", "full 360-degree"))
            self.assertIn("r_mm,z_mm,br_t,bz_t", readme)


class AxisymmetricReferenceTests(unittest.TestCase):
    def test_reference_agrees_with_direct_biot_savart_quadrature(self):
        # Independently integrate circular-loop fields numerically. Simpson's
        # rule checks the analytic primitive at off-centre and edge positions.
        n = 80
        for z_mm in (0, 20, 50, -50):
            total = 0.
            for i in range(n+1):
                r = .02 + .01*i/n
                wr = 1 if i in (0, n) else (4 if i % 2 else 2)
                axial = 0.
                for j in range(n+1):
                    z = -.02 + .04*j/n
                    wz = 1 if j in (0, n) else (4 if j % 2 else 2)
                    axial += wz*r*r/(r*r + (z_mm*.001-z)**2)**1.5
                total += wr*axial*.04/(3*n)
            expected = 4*math.pi*1e-7*200/(2*.01*.04)*total*.01/(3*n)
            self.assertAlmostEqual(air_solenoid_reference(z_mm)/expected, 1, delta=1e-7)
        self.assertAlmostEqual(air_solenoid_reference(0, 2), 2*air_solenoid_reference(0))

    def test_invalid_reference_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            rectangular_winding_axis_bz(r_inner_mm=0, r_outer_mm=30, z_min_mm=-20,
                                        z_max_mm=20, turns=200, current_a=1, z_mm=0)
