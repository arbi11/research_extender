"""Editable Python CAD model for native FEMM 5.

Edit MODEL, then run python3 model.py --mesh-only before a full solve.
The named Python stages compile deterministic operations to the native Lua
bridge. This file uses only the Python standard library, without pyFEMM.
"""
import html
import json
import math
import re

# Edit these model data. Dimensions are millimeters, currents amperes.
MODEL = {'kind': 'magnetostatic_plan',
 'title': 'Planar DC magnetostatic circular conductor in circular air domain with B '
          'probe',
 'units': 'millimeters',
 'physics': 'planar_dc_magnetics',
 'depth_mm': 20.0,
 'mesh_mm': 0.5,
 'domain': {'type': 'circle', 'center': {'x': 0.0, 'y': 0.0}, 'radius': 10.0},
 'air_label': {'x': 5.0, 'y': 0.0},
 'materials': [{'id': 'mat_conductor', 'relative_permeability': 1.0}],
 'circuits': [{'id': 'circuit_dc_conductor', 'current_a': 1.0}],
 'regions': [{'id': 'r_f1dfff636e1cbc6e011587',
              'shape': {'type': 'circle',
                        'center': {'x': 0.0, 'y': 0.0},
                        'radius': 1.0},
              'material': 'mat_conductor',
              'label': {'x': 0.0, 'y': 0.0},
              'circuit': 'circuit_dc_conductor',
              'turns': 1,
              'mesh_mm': None}],
 'force_region': None,
 'probes': [{'x': 2.0, 'y': 0.0}]}


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
        if model.get("units") != "millimeters" or model.get("physics") not in ("planar_dc_magnetics", "axisymmetric_dc_magnetics"):
            raise ValueError("Only millimeter planar or axisymmetric DC magnetic models are supported")
        self.axisymmetric = model["physics"] == "axisymmetric_dc_magnetics"
        if self.axisymmetric:
            if model.get("depth_mm") is not None:
                raise ValueError("Axisymmetric depth_mm must be null: FEMM integrates the full 360-degree volume")
            shapes = [model["domain"]] + [region["shape"] for region in model["regions"]]
            if model["domain"]["type"] == "circle" or self.bounds(model["domain"])[0] != 0:
                raise ValueError("Axisymmetric domains require a straight r=0 boundary")
            for shape in shapes:
                if self.bounds(shape)[0] < 0:
                    raise ValueError("Axisymmetric geometry requires nonnegative radius (x=r, y=z)")
                if self.bounds(shape)[0] == 0:
                    nodes = self.nodes(shape)
                    if shape["type"] == "circle" or not any(a[0] == b[0] == 0 for a, b in zip(nodes, nodes[1:]+nodes[:1])):
                        raise ValueError("Contours meeting r=0 require a straight axis edge")
            if any(point["x"] < 0 for point in model["probes"]):
                raise ValueError("Axisymmetric probes require nonnegative radius")
        else:
            self.number(model["depth_mm"])
            if model["depth_mm"] <= 0:
                raise ValueError("Planar depth must be positive")
        self.number(model["mesh_mm"])
        if model["mesh_mm"] <= 0:
            raise ValueError("Mesh size must be positive")
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
        # Axisymmetric integrals cover the complete body of revolution; the
        # native depth argument is zero and never multiplies energy or force.
        self.call("mi_probdef", 0, "millimeters", "axi" if self.axisymmetric else "planar",
                  1e-8, 0 if self.axisymmetric else self.model["depth_mm"], 30)

    def contour_nodes(self, shape):
        nodes = self.nodes(shape)
        if not self.axisymmetric or shape["type"] == "circle":
            return nodes
        # A material face may meet the symmetry axis. Split the domain's axis
        # edge at every material endpoint so no overlapping edges are emitted.
        shapes = [self.model["domain"]] + [region["shape"] for region in self.model["regions"]]
        axis_z = sorted({y for item in shapes for x, y in self.nodes(item) if x == 0})
        expanded = []
        for first, second in zip(nodes, nodes[1:]+nodes[:1]):
            expanded.append(first)
            if first[0] == second[0] == 0:
                middle = [z for z in axis_z if min(first[1], second[1]) < z < max(first[1], second[1])]
                if second[1] < first[1]:
                    middle.reverse()
                expanded.extend((0, z) for z in middle)
        return expanded

    def draw_geometry(self):
        self.lines.append("-- Stage 2: draw closed contours.")
        seen_nodes, seen_segments = set(), set()
        for shape in [self.model["domain"]] + [region["shape"] for region in self.model["regions"]]:
            nodes = self.contour_nodes(shape)
            for x, y in nodes:
                if not self.axisymmetric or (x, y) not in seen_nodes:
                    self.call("mi_addnode", x, y)
                seen_nodes.add((x, y))
            for first, second in zip(nodes, nodes[1:] + nodes[:1]):
                x1, y1 = first
                x2, y2 = second
                if shape["type"] == "circle":
                    self.call("mi_addarc", x1, y1, x2, y2, 90, 5)
                else:
                    key = tuple(sorted((first, second)))
                    if not self.axisymmetric or key not in seen_segments:
                        self.call("mi_addsegment", x1, y1, x2, y2)
                    seen_segments.add(key)

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
            nodes = self.contour_nodes(shape)
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
            if self.axisymmetric:
                # Radial components cancel around a complete body of revolution.
                # Report only the physically meaningful net axial force.
                self.lines.extend([f"mo_groupselectblock({group})", "fz = mo_blockintegral(19)", "mo_clearblock()", 'write(out,"force_z,",fz,",N\\n")'])
            else:
                self.lines.extend([f"mo_groupselectblock({group})", "fx = mo_blockintegral(18)", "fy = mo_blockintegral(19)", "mo_clearblock()", 'write(out,"force_x,",fx,",N\\n")', 'write(out,"force_y,",fy,",N\\n")'])
        for circuit in self.model["circuits"]:
            self.lines.extend([f'amps, volts, linkage = mo_getcircuitproperties({self.quote(circuit["id"])})', f'write(out,"current_{circuit["id"]},",amps,",A\\n")', f'write(out,"flux_linkage_{circuit["id"]},",linkage,",Wb-turn\\n")'])
        first_component, second_component = ("br", "bz") if self.axisymmetric else ("bx", "by")
        for index, point in enumerate(self.model["probes"]):
            self.lines.extend([f'a, bx, by = mo_getpointvalues({self.number(point["x"])},{self.number(point["y"])})', f'write(out,"{first_component}_{index},",bx,",T\\n")', f'write(out,"{second_component}_{index},",by,",T\\n")'])
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
    (root / "spec.json").write_text(json.dumps(MODEL, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (root / "geometry.svg").write_text(program.preview_svg(), encoding="utf-8")
    if args.write_only:
        print("Regenerated model.lua, spec.json and geometry.svg from Python MODEL data.")
        return
    # Remove prior outputs so a failed run cannot look like a successful one.
    # A downloaded solved bundle also contains derived previews, JSON and ZIP
    # outputs. Those become stale when MODEL changes, including on mesh-only runs.
    for name in ("model.fem", "model.ans", "mesh.csv", "mesh.svg", "mesh-report.json", "model.node", "model.ele", "model.edge", "model.1.node", "model.1.ele", "model.1.edge", "results.csv", "results.json", "field.csv", "field.svg", "download.zip"):
        (root / name).unlink(missing_ok=True)
    command = os.environ.get("FEMM_LUA", "femm-lua")
    if not shutil.which(command):
        raise SystemExit("Install native FEMM 5 and put femm-lua on PATH, or set FEMM_LUA. See README.txt.")
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
