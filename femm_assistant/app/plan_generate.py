"""Compile validated magnetostatic geometry plans into native FEMM 5 operations.

The plan is data. Only the functions in this module decide which Lua/FEMM
operations run; neither plan titles nor user-provided executable text is emitted.
"""
import html
import inspect
import io
import json
import math
import pprint
import re
import zipfile

from .plan import GeometryPlan


class FemmCadProgram:
    """Editable Python CAD stages compiled to FEMM 5's native Lua bridge.

    The server validates the complete geometry plan before constructing this
    program. The standalone export also checks scalar types and safe names,
    but geometric topology must be checked with the native mesh stage after
    local edits. No Python FEMM binding or third-party package is needed.
    """

    def __init__(self, model):
        self.model = model
        self.lines = []
        if model.get("units") != "millimeters" or model.get("physics") != "planar_dc_magnetics":
            raise ValueError("Only millimeter, planar DC magnetic models are supported")
        self.number(model["depth_mm"])
        self.number(model["mesh_mm"])
        if model["depth_mm"] <= 0 or model["mesh_mm"] <= 0:
            raise ValueError("Depth and mesh size must be positive")
        for item in model["materials"] + model["circuits"] + model["regions"]:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", item["id"]):
                raise ValueError("Material, circuit and region ids must be ASCII identifiers")

    @staticmethod
    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("CAD coordinates and properties must be finite numbers")
        return repr(value) if isinstance(value, int) else repr(float(value))

    @staticmethod
    def quote(value):
        # Only schema identifiers and fixed compiler-owned strings reach Lua.
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_< >.=-]+", value):
            raise ValueError("Unsupported native string argument")
        return json.dumps(value, ensure_ascii=True)

    def call(self, name, *args):
        allowed = {
            "newdocument", "mi_probdef", "mi_addnode", "mi_addarc", "mi_addsegment",
            "mi_addmaterial", "mi_addcircprop", "mi_addblocklabel", "mi_selectlabel",
            "mi_setblockprop", "mi_clearselected", "mi_addboundprop",
            "mi_selectarcsegment", "mi_setarcsegmentprop", "mi_selectsegment",
            "mi_setsegmentprop", "mi_saveas", "mi_createmesh", "mi_analyze",
            "mi_loadsolution", "mi_close", "mo_close", "mo_groupselectblock",
            "mo_clearblock",
        }
        if name not in allowed:
            raise ValueError("Unsupported native FEMM operation")
        rendered = [self.quote(arg) if isinstance(arg, str) else self.number(arg) for arg in args]
        self.lines.append(name + "(" + ",".join(rendered) + ")")

    @staticmethod
    def nodes(shape):
        if shape["type"] == "circle":
            x, y, radius = shape["center"]["x"], shape["center"]["y"], shape["radius"]
            if radius <= 0:
                raise ValueError("Circle radius must be positive")
            return [(x + radius, y), (x, y + radius), (x - radius, y), (x, y - radius)]
        if shape["type"] == "rectangle":
            x1, x2, y1, y2 = (shape[key] for key in ("x_min", "x_max", "y_min", "y_max"))
            if x1 >= x2 or y1 >= y2:
                raise ValueError("Rectangle dimensions must be positive")
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        if shape["type"] == "polygon" and len(shape["points"]) >= 3:
            return [(point["x"], point["y"]) for point in shape["points"]]
        raise ValueError("Expected a rectangle, circle or polygon")

    @classmethod
    def bounds(cls, shape):
        nodes = cls.nodes(shape)
        return min(x for x, _ in nodes), max(x for x, _ in nodes), min(y for _, y in nodes), max(y for _, y in nodes)

    @classmethod
    def inside(cls, shape, x, y):
        if shape["type"] == "circle":
            return (x-shape["center"]["x"])**2 + (y-shape["center"]["y"])**2 < shape["radius"]**2 - 1e-10
        nodes = cls.nodes(shape)
        result = False
        for (ax, ay), (bx, by) in zip(nodes, nodes[1:] + nodes[:1]):
            cross = (x-ax)*(by-ay) - (y-ay)*(bx-ax)
            if abs(cross) <= 1e-10 and min(ax, bx)-1e-10 <= x <= max(ax, bx)+1e-10 and min(ay, by)-1e-10 <= y <= max(ay, by)+1e-10:
                return False
            if (ay > y) != (by > y) and x < (bx-ax)*(y-ay)/(by-ay)+ax:
                result = not result
        return result

    def define_problem(self):
        self.lines.append("-- Stage 1: define the problem.")
        self.call("newdocument", 0)
        self.call("mi_probdef", 0, "millimeters", "planar", 1e-8, self.model["depth_mm"], 30)

    def draw_geometry(self):
        self.lines.append("-- Stage 2: draw closed contours.")
        for shape in [self.model["domain"]] + [region["shape"] for region in self.model["regions"]]:
            nodes = self.nodes(shape)
            for x, y in nodes:
                self.call("mi_addnode", x, y)
            for (x1, y1), (x2, y2) in zip(nodes, nodes[1:] + nodes[:1]):
                if shape["type"] == "circle":
                    self.call("mi_addarc", x1, y1, x2, y2, 90, 5)
                else:
                    self.call("mi_addsegment", x1, y1, x2, y2)

    def assign_materials(self):
        self.lines.append("-- Stage 3: create linear isotropic materials.")
        self.call("mi_addmaterial", "air", 1, 1, 0, 0, 0)
        for material in self.model["materials"]:
            mu = material["relative_permeability"]
            if mu <= 0:
                raise ValueError("Relative permeability must be positive")
            self.call("mi_addmaterial", material["id"], mu, mu, 0, 0, 0)

    def label(self, point, material, mesh, circuit, group, turns):
        if mesh <= 0 or isinstance(turns, bool) or not isinstance(turns, int):
            raise ValueError("Mesh must be positive and turns must be an integer")
        self.call("mi_addblocklabel", point["x"], point["y"])
        self.call("mi_selectlabel", point["x"], point["y"])
        self.call("mi_setblockprop", material, 0, mesh, circuit, 0, group, turns)
        self.call("mi_clearselected")

    def set_boundaries_and_excitation(self):
        self.lines.append("-- Stage 4: assign material faces, circuits, and outer A=0 boundary.")
        for circuit in self.model["circuits"]:
            self.call("mi_addcircprop", circuit["id"], circuit["current_a"], 1)
        self.label(self.model["air_label"], "air", self.model["mesh_mm"], "<None>", 0, 0)
        for group, region in enumerate(self.model["regions"], start=1):
            self.label(region["label"], region["material"], region["mesh_mm"] or self.model["mesh_mm"], region["circuit"] or "<None>", group, region["turns"])
        shape = self.model["domain"]
        self.call("mi_addboundprop", "outer_a_zero", 0, 0, 0, 0, 0, 0, 0, 0, 0)
        if shape["type"] == "circle":
            for angle in (45, 135, 225, 315):
                radians = math.radians(angle)
                self.call("mi_selectarcsegment", shape["center"]["x"] + shape["radius"]*math.cos(radians), shape["center"]["y"] + shape["radius"]*math.sin(radians))
            self.call("mi_setarcsegmentprop", 5, "outer_a_zero", 0, 0)
        else:
            nodes = self.nodes(shape)
            for (x1, y1), (x2, y2) in zip(nodes, nodes[1:] + nodes[:1]):
                self.call("mi_selectsegment", (x1+x2)/2, (y1+y2)/2)
            self.call("mi_setsegmentprop", "outer_a_zero", 0, 1, 0, 0)
        self.call("mi_clearselected")

    def mesh(self):
        self.lines.append("-- Stage 5: save geometry and create the native finite-element mesh.")
        self.call("mi_saveas", "model.fem")
        self.call("mi_createmesh")
        # A completion marker is not a claim about element quality or topology.
        self.lines.extend(['out = openfile("mesh.csv","w")', 'write(out,"status\\ncomplete\\n")', "closefile(out)"])

    def solve(self):
        self.lines.append("-- Stage 6: solve the magnetic finite-element problem.")
        self.call("mi_analyze", 1)
        self.call("mi_loadsolution")

    def extract_results(self):
        self.lines.extend(["-- Stage 7: extract computed quantities and sampled fields.",
                           'out = openfile("results.csv","w")', 'write(out,"quantity,value,unit\\n")',
                           "mo_groupselectblock()", "energy = mo_blockintegral(2)", "mo_clearblock()",
                           'write(out,"energy,",energy,",J\\n")'])
        if self.model["force_region"]:
            group = next(index for index, region in enumerate(self.model["regions"], start=1) if region["id"] == self.model["force_region"])
            self.lines.extend([f"mo_groupselectblock({group})", "fx = mo_blockintegral(18)", "fy = mo_blockintegral(19)", "mo_clearblock()", 'write(out,"force_x,",fx,",N\\n")', 'write(out,"force_y,",fy,",N\\n")'])
        for circuit in self.model["circuits"]:
            self.lines.extend([f'amps, volts, linkage = mo_getcircuitproperties({self.quote(circuit["id"])})', f'write(out,"current_{circuit["id"]},",amps,",A\\n")', f'write(out,"flux_linkage_{circuit["id"]},",linkage,",Wb-turn\\n")'])
        for index, point in enumerate(self.model["probes"]):
            self.lines.extend([f'a, bx, by = mo_getpointvalues({self.number(point["x"])},{self.number(point["y"])})', f'write(out,"bx_{index},",bx,",T\\n")', f'write(out,"by_{index},",by,",T\\n")'])
        self.lines.extend(["closefile(out)", 'out = openfile("field.csv","w")', 'write(out,"x_mm,y_mm,bx_t,by_t\\n")'])
        xmin, xmax, ymin, ymax = self.bounds(self.model["domain"])
        dx, dy = (xmax-xmin)/40, (ymax-ymin)/30
        mask = ["1" if self.inside(self.model["domain"], xmin+col*dx, ymin+row*dy) else "0" for row in range(31) for col in range(41)]
        self.lines.extend(["inside = {" + ",".join(mask) + "}", "for row=0,30 do", f"  y = {self.number(ymin)} + row * {self.number(dy)}", "  for col=0,40 do", f"    x = {self.number(xmin)} + col * {self.number(dx)}", "    bx = 0", "    by = 0", "    if inside[row*41+col+1] == 1 then", "      a, bx, by = mo_getpointvalues(x,y)", "    end", '    write(out,x,",",y,",",bx,",",by,"\\n")', "  end", "end", "closefile(out)"])

    def lua(self, mesh_only=False):
        self.lines = ["-- research_extender Python CAD compiler. Native FEMM 5 / Lua 4."]
        self.define_problem()
        self.draw_geometry()
        self.assign_materials()
        self.set_boundaries_and_excitation()
        self.mesh()
        if not mesh_only:
            self.solve()
            self.extract_results()
            self.call("mo_close")
        self.call("mi_close")
        return "\n".join(self.lines) + "\n"

    def preview_svg(self):
        xmin, xmax, ymin, ymax = self.bounds(self.model["domain"])
        extent = max(xmax-xmin, ymax-ymin)
        pad, stroke = .035*extent, .002*extent
        title = html.escape(self.model["title"], quote=True)
        items = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{xmin-pad:g} {-ymax-pad:g} {xmax-xmin+2*pad:g} {ymax-ymin+2*pad:g}" role="img" aria-label="{title} geometry preview">']

        def contour(shape, fill, color, caption):
            properties = f'fill="{fill}" stroke="{color}" stroke-width="{stroke:g}"'
            caption = '<title>' + html.escape(caption) + '</title>'
            if shape["type"] == "circle":
                return f'<circle cx="{shape["center"]["x"]:g}" cy="{-shape["center"]["y"]:g}" r="{shape["radius"]:g}" {properties}>{caption}</circle>'
            points = " ".join(f"{x:g},{-y:g}" for x, y in self.nodes(shape))
            return f'<polygon points="{points}" {properties}>{caption}</polygon>'

        def area(region):
            shape = region["shape"]
            if shape["type"] == "circle":
                return math.pi*shape["radius"]**2
            points = self.nodes(shape)
            return abs(sum(ax*by-bx*ay for (ax, ay), (bx, by) in zip(points, points[1:]+points[:1])))/2

        items.append(contour(self.model["domain"], "#f3f7f8", "#66818c", "Outer A=0 boundary"))
        for region in sorted(self.model["regions"], key=area, reverse=True):
            fill = "#e5f1f5" if region["material"] == "air" else "#eb995f" if region["circuit"] else "#8cabb7"
            color = "#9b4c22" if region["id"] == self.model["force_region"] else "#263c46"
            items.append(contour(region["shape"], fill, color, region["id"]))
            point = region["label"]
            items.append(f'<circle cx="{point["x"]:g}" cy="{-point["y"]:g}" r="{stroke*2:g}" fill="#263c46"/>')
        for index, point in enumerate(self.model["probes"]):
            items.append(f'<circle cx="{point["x"]:g}" cy="{-point["y"]:g}" r="{stroke*3:g}" fill="#a51c42"><title>Field probe {index}</title></circle>')
        return "".join(items) + "</svg>"


def required_results(plan: GeometryPlan) -> dict[str, str]:
    result = {"energy": "J"}
    if plan.force_region:
        result.update(force_x="N", force_y="N")
    for circuit in plan.circuits:
        result[f"current_{circuit.id}"] = "A"
        result[f"flux_linkage_{circuit.id}"] = "Wb-turn"
    for index, _ in enumerate(plan.probes):
        result[f"bx_{index}"] = "T"
        result[f"by_{index}"] = "T"
    return result


def lua_script(plan: GeometryPlan, *, mesh_only: bool = False) -> str:
    """Compile validated data through the editable Python CAD pipeline."""
    plan = GeometryPlan.model_validate(plan.model_dump())
    return FemmCadProgram(plan.model_dump()).lua(mesh_only=mesh_only)


def preview_svg(plan: GeometryPlan) -> str:
    plan = GeometryPlan.model_validate(plan.model_dump())
    return FemmCadProgram(plan.model_dump()).preview_svg()


EXPORT_README = """FEMM 5 geometry-plan model

Install native FEMM 5: https://femm.info/doku/doku.php?id=download5
Linux build instructions: https://femm.info/doku/doku.php?id=bindingsetup
Put femm-lua on PATH or set FEMM_LUA to its launcher path.

Run: python3 model.py
Mesh without solving: python3 model.py --mesh-only
Regenerate Lua and geometry SVG only: python3 model.py --write-only
Compatibility entry point: python3 run.py
Direct Lua compatibility export: femm-lua -h model.lua

No AI key, repository checkout, or third-party Python package is needed to rerun.
Edit the MODEL dictionary in model.py to change the geometry, materials,
circuits, labels, mesh sizes and requested measurements. Its named Python
stages define_problem, draw_geometry, assign_materials,
set_boundaries_and_excitation, mesh, solve and extract_results regenerate a
complete model.lua. Python controls FEMM through its native Lua bridge;
this export does not use or require pyFEMM. run.py calls the same Python pipeline.
spec.json and geometry.svg are regenerated from MODEL each time. Editing
spec.json alone does not change MODEL. The server validated the original plan;
after local geometry edits, use --mesh-only before solving and inspect the mesh.
Mesh-only runs do not calculate magnetic fields or numerical simulation results.
mesh.csv only marks completed native meshing, not geometric quality approval.
Our deployment can optionally export model.node/model.ele using a documented
FEMM patch. Stock FEMM 5 keeps the mesh in memory and need not write those files;
the portable Python script also works with stock FEMM 5. Inspect or regenerate
the mesh in the FEMM GUI when those optional diagnostic files are absent.
The full solve writes model.fem, model.ans, results.csv and field.csv.

ASSUMPTIONS AND MEASUREMENTS
2D planar DC magnetostatics, millimeter coordinates and specified out-of-plane
depth. Materials are linear and isotropic. The outer contour has fixed A=0.
Each contour's labeled face excludes any nested child contours. Optional force
uses the weighted stress tensor on the selected material region. Field probes
and the 41 by 31 sampled grid report Bx and By in tesla; samples outside or on
the outer boundary are zero-filled for plotting, not field measurements there.
Circuit quantities use total flux linkage in Wb-turn. Energy is in joules.
Mesh and outer-domain convergence must be assessed for quantitative design.
Results are finite-element calculations, not AI-generated numerical predictions.

Book: https://github.com/arbi11/research_extender
"""


PYTHON_MAIN = '''
def main(argv=None):
    import argparse
    import os
    from pathlib import Path
    import shutil
    import signal
    import socket
    import subprocess

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-only", action="store_true", help="Create geometry and native mesh without solving")
    parser.add_argument("--write-only", action="store_true", help="Regenerate Lua, spec and geometry SVG without launching FEMM")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    program = FemmCadProgram(MODEL)
    (root / "model.lua").write_text(program.lua(mesh_only=args.mesh_only), encoding="utf-8")
    (root / "spec.json").write_text(json.dumps(MODEL, indent=2, allow_nan=False) + "\\n", encoding="utf-8")
    (root / "geometry.svg").write_text(program.preview_svg(), encoding="utf-8")
    if args.write_only:
        print("Regenerated model.lua, spec.json and geometry.svg from Python MODEL data.")
        return
    command = os.environ.get("FEMM_LUA", "femm-lua")
    if not shutil.which(command):
        raise SystemExit("Install native FEMM 5 and put femm-lua on PATH, or set FEMM_LUA. See README.txt.")
    # Remove prior outputs so a failed run cannot look like a successful one.
    for name in ("model.fem", "model.ans", "mesh.csv", "model.node", "model.ele", "model.edge", "results.csv", "field.csv"):
        (root / name).unlink(missing_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    environment = dict(os.environ, FEMM_MESH_EXPORT="1")
    process = subprocess.Popen([command, "-h", "-p", str(port), str(root / "model.lua")], cwd=root, env=environment, start_new_session=(os.name == "posix"))
    try:
        try:
            process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            raise SystemExit("FEMM exceeded the 180 second runtime limit.")
        if process.returncode:
            raise SystemExit("FEMM exited with an error. Inspect the solver output.")
    finally:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.kill()
        process.wait()
    expected = ("model.fem", "mesh.csv") if args.mesh_only else ("model.fem", "model.ans", "results.csv", "field.csv")
    if any(not (root / name).is_file() or (root / name).stat().st_size == 0 for name in expected):
        raise SystemExit("FEMM did not produce the expected outputs. Inspect the solver output.")
    if args.mesh_only:
        print("Native mesh created; no magnetic solve was performed. Open model.fem in FEMM to inspect the mesh.")
    else:
        print("Solved. Open model.fem in FEMM; quantities are in results.csv and fields in field.csv.")


if __name__ == "__main__":
    main()
'''


def python_script(plan: GeometryPlan) -> str:
    """Return a self-contained, editable Python compiler and native launcher."""
    plan = GeometryPlan.model_validate(plan.model_dump())
    header = '''"""Editable Python CAD model for native FEMM 5.

Edit MODEL, then run python3 model.py --mesh-only before a full solve.
The named Python stages compile deterministic operations to the native Lua
bridge. This file uses only the Python standard library, without pyFEMM.
"""
import html
import json
import math
import re

# Edit these model data. Dimensions are millimeters, currents amperes.
MODEL = '''
    return (header + pprint.pformat(plan.model_dump(), sort_dicts=False, width=88)
            + "\n\n\n" + inspect.getsource(FemmCadProgram) + "\n" + PYTHON_MAIN)


RUNNER = '''"""Compatibility entry point for the editable Python CAD pipeline."""
from model import main

if __name__ == "__main__":
    main()
'''


def source_files(plan: GeometryPlan, *, mesh_only: bool = False) -> dict[str, str]:
    plan = GeometryPlan.model_validate(plan.model_dump())
    return {"model.lua": lua_script(plan, mesh_only=mesh_only), "model.py": python_script(plan), "run.py": RUNNER,
            "spec.json": json.dumps(plan.model_dump(), indent=2, allow_nan=False) + "\n",
            "geometry.svg": preview_svg(plan), "README.txt": EXPORT_README}


def source_zip(plan: GeometryPlan, *, mesh_only: bool = False) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in source_files(plan, mesh_only=mesh_only).items():
            archive.writestr(name, content)
    return buffer.getvalue()
