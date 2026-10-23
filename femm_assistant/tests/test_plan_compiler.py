import io
import json
import re
import unittest
import xml.etree.ElementTree as ET
import zipfile

from pydantic import ValidationError

from app.model import CCoreSpec
from app.plan import GeometryPlan, from_ccore, wire_example
from app.plan_generate import lua_script, preview_svg, required_results, source_files, source_zip


class PlanCompilerTests(unittest.TestCase):
    def test_wire_compiles_actual_arcs_and_all_workflow_stages(self):
        plan = wire_example()
        script = lua_script(plan)
        self.assertEqual(script.count("mi_addarc("), 8)
        self.assertEqual(script.count("mi_addnode("), 8)
        self.assertEqual(script.count("mi_selectarcsegment("), 4)
        self.assertIn('mi_setarcsegmentprop(5,"outer_a_zero",0,0)', script)
        self.assertEqual(script.count("mi_addblocklabel("), len(plan.regions) + 1)
        stages = [script.index(f"-- Stage {index}:") for index in range(1, 7)]
        self.assertEqual(stages, sorted(stages))
        self.assertLess(script.index("mi_addboundprop("), script.index("mi_analyze(1)"))
        self.assertIn("mo_getpointvalues(", script)
        self.assertNotIn("mo_getb(", script)
        self.assertIn("mo_blockintegral(2)", script)
        self.assertIn('mi_saveas("model.fem")', script)

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
        self.assertEqual(set(files), {"model.lua", "run.py", "spec.json", "geometry.svg", "README.txt"})
        compile(files["run.py"], "run.py", "exec")
        self.assertNotIn("from app", files["run.py"])
        self.assertEqual(json.loads(files["spec.json"]), plan.model_dump())
        with zipfile.ZipFile(io.BytesIO(source_zip(plan))) as archive:
            self.assertEqual(set(archive.namelist()), set(files))
            self.assertNotIn("model.ans", archive.namelist())
            self.assertEqual(archive.read("model.lua").decode(), lua_script(plan))

    def test_compilation_rejects_models_mutated_after_validation(self):
        unchecked = wire_example().model_copy(update={"depth_mm": float("nan")})
        with self.assertRaises(ValidationError):
            lua_script(unchecked)
        with self.assertRaises(ValidationError):
            preview_svg(unchecked)


if __name__ == "__main__":
    unittest.main()
