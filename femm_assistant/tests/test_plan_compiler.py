import io
import json
import re
import runpy
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from pydantic import ValidationError

from app.model import CCoreSpec
from app.plan import GeometryPlan, from_ccore, wire_example
from app.plan_generate import lua_script, preview_svg, python_script, required_results, source_files, source_zip


class PlanCompilerTests(unittest.TestCase):
    def test_wire_compiles_actual_arcs_and_all_workflow_stages(self):
        plan = wire_example()
        script = lua_script(plan)
        self.assertEqual(script.count("mi_addarc("), 8)
        self.assertEqual(script.count("mi_addnode("), 8)
        self.assertEqual(script.count("mi_selectarcsegment("), 4)
        self.assertIn('mi_setarcsegmentprop(5,"outer_a_zero",0,0)', script)
        self.assertEqual(script.count("mi_addblocklabel("), len(plan.regions) + 1)
        stages = [script.index(f"-- Stage {index}:") for index in range(1, 8)]
        self.assertEqual(stages, sorted(stages))
        self.assertLess(script.index("mi_addboundprop("), script.index("mi_analyze(1)"))
        self.assertIn("mo_getpointvalues(", script)
        self.assertNotIn("mo_getb(", script)
        self.assertIn("mo_blockintegral(2)", script)
        self.assertIn('mi_saveas("model.fem")', script)
        self.assertLess(script.index("mi_createmesh()"), script.index("mi_analyze(1)"))

    def test_field_grid_clips_circle_domain_before_querying_femm(self):
        script = lua_script(wire_example())
        mask = re.search(r"inside = \{([01,]+)\}", script).group(1).split(",")
        self.assertEqual(len(mask), 41 * 31)
        self.assertEqual(mask[0], "0")
        self.assertEqual(mask[15 * 41 + 20], "1")
        self.assertIn("if inside[row*41+col+1] == 1 then\n      a, bx, by = mo_getpointvalues(x,y)", script)

    def test_result_contract_tracks_requested_measurements(self):
        plan = wire_example()
        contract = required_results(plan)
        self.assertEqual(contract["energy"], "J")
        self.assertNotIn("force_x", contract)
        for circuit in plan.circuits:
            self.assertEqual(contract[f"current_{circuit.id}"], "A")
            self.assertEqual(contract[f"flux_linkage_{circuit.id}"], "Wb-turn")
        self.assertEqual(contract["bx_0"], "T")
        self.assertEqual(contract["by_0"], "T")
        core = from_ccore(CCoreSpec())
        self.assertEqual(required_results(core)["force_x"], "N")
        self.assertIn("mo_blockintegral(18)", lua_script(core))
        self.assertIn('mi_setsegmentprop("outer_a_zero",0,1,0,0)', lua_script(core))

    def test_title_is_svg_escaped_and_never_emitted_as_lua(self):
        data = wire_example().model_dump()
        data["title"] = '</svg><script>alert("bad")</script>'
        plan = GeometryPlan.model_validate(data)
        svg = preview_svg(plan)
        self.assertIn("&lt;script&gt;", svg)
        parsed = ET.fromstring(svg)
        self.assertFalse(any(element.tag.endswith("script") for element in parsed.iter()))
        self.assertNotIn("alert", lua_script(plan))

    def test_source_bundle_is_portable_and_contains_no_fabricated_results(self):
        plan = wire_example()
        files = source_files(plan)
        self.assertEqual(set(files), {"model.lua", "model.py", "run.py", "spec.json", "geometry.svg", "README.txt"})
        compile(files["run.py"], "run.py", "exec")
        compile(files["model.py"], "model.py", "exec")
        self.assertNotIn("from app", files["run.py"])
        self.assertNotIn("from app", files["model.py"])
        self.assertEqual(json.loads(files["spec.json"]), plan.model_dump())
        with zipfile.ZipFile(io.BytesIO(source_zip(plan))) as archive:
            self.assertEqual(set(archive.namelist()), set(files))
            self.assertNotIn("model.ans", archive.namelist())
            self.assertEqual(archive.read("model.lua").decode(), lua_script(plan))

    def test_mesh_only_stops_before_analysis_or_result_extraction(self):
        plan = wire_example()
        script = lua_script(plan, mesh_only=True)
        self.assertIn("mi_createmesh()", script)
        self.assertIn('mi_saveas("model.fem")', script)
        self.assertIn('openfile("mesh.csv","w")', script)
        self.assertIn("mi_close()", script)
        self.assertNotIn("mi_analyze", script)
        self.assertNotIn("mi_loadsolution", script)
        self.assertNotIn("mo_", script)
        self.assertNotIn("results.csv", script)
        self.assertNotIn("field.csv", script)
        self.assertEqual(source_files(plan, mesh_only=True)["model.lua"], script)

    def test_exported_python_runs_independently_and_matches_server_compiler(self):
        plan = wire_example()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, content in source_files(plan).items():
                (root / name).write_text(content)
            process = subprocess.run([sys.executable, "model.py", "--write-only"], cwd=root, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual((root / "model.lua").read_text(), lua_script(plan))
            self.assertEqual((root / "geometry.svg").read_text(), preview_svg(plan))
            self.assertEqual(json.loads((root / "spec.json").read_text()), plan.model_dump())
            self.assertFalse((root / "mesh.csv").exists())
            self.assertFalse((root / "results.csv").exists())
            process = subprocess.run([sys.executable, "run.py", "--mesh-only", "--write-only"], cwd=root, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual((root / "model.lua").read_text(), lua_script(plan, mesh_only=True))

    def test_python_model_edits_regenerate_geometry_excitation_and_measurements(self):
        plan = wire_example()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "model.py"
            target.write_text(python_script(plan))
            module = runpy.run_path(str(target))
            data = module["MODEL"]
            data["regions"][0]["shape"]["radius"] = 1.5
            data["circuits"][0]["current_a"] = 2.0
            data["probes"].append({"x": 3.0, "y": 0.0})
            module["main"](["--write-only"])
            updated = GeometryPlan.model_validate(data)
            self.assertEqual(target.with_name("model.lua").read_text(), lua_script(updated))
            self.assertIn('r="1.5"', target.with_name("geometry.svg").read_text())
            self.assertIn("by_1", target.with_name("model.lua").read_text())

    def test_generated_python_treats_provider_strings_as_data(self):
        data = wire_example().model_dump()
        data["title"] = '</svg><script>bad</script>"; __import__("os").abort() #'
        plan = GeometryPlan.model_validate(data)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "model.py"
            target.write_text(python_script(plan))
            process = subprocess.run([sys.executable, str(target), "--write-only"], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertNotIn("__import__", target.with_name("model.lua").read_text())
            parsed = ET.fromstring(target.with_name("geometry.svg").read_text())
            self.assertFalse(any(element.tag.endswith("script") for element in parsed.iter()))

    def test_compilation_rejects_models_mutated_after_validation(self):
        unchecked = wire_example().model_copy(update={"depth_mm": float("nan")})
        with self.assertRaises(ValidationError):
            lua_script(unchecked)
        with self.assertRaises(ValidationError):
            preview_svg(unchecked)


if __name__ == "__main__":
    unittest.main()
