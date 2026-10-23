"""Compile validated magnetostatic geometry plans into native FEMM 5 operations.

The plan is data. Only the functions in this module decide which Lua/FEMM
operations run; neither plan titles nor user-provided executable text is emitted.
"""
import html
import io
import json
import math
import zipfile

from .generate import RUNNER
from .plan import GeometryPlan, Point, bounds, contains, shape_points


def _number(value: float) -> str:
    return repr(float(value))


def _quote(value: str) -> str:
    # Executable names are schema-validated ASCII identifiers. JSON's escaping
    # for these strings is also compatible with the bundled Lua 4 interpreter.
    return json.dumps(value, ensure_ascii=True)


def _contour_nodes(shape):
    if shape.type == "circle":
        x, y, radius = shape.center.x, shape.center.y, shape.radius
        return [(x + radius, y), (x, y + radius),
                (x - radius, y), (x, y - radius)]
    return [(point.x, point.y) for point in shape_points(shape)]


def _draw_contour(shape) -> list[str]:
    nodes = _contour_nodes(shape)
    lines = [f"mi_addnode({_number(x)},{_number(y)})" for x, y in nodes]
    for (x1, y1), (x2, y2) in zip(nodes, nodes[1:] + nodes[:1]):
        coordinates = ",".join(_number(value) for value in (x1, y1, x2, y2))
        if shape.type == "circle":
            lines.append(f"mi_addarc({coordinates},90,5)")
        else:
            lines.append(f"mi_addsegment({coordinates})")
    return lines


def _assign_outer_boundary(shape) -> list[str]:
    lines = ['mi_addboundprop("outer_a_zero",0,0,0,0,0,0,0,0,0)']
    if shape.type == "circle":
        for angle in (45, 135, 225, 315):
            radians = math.radians(angle)
            x = shape.center.x + shape.radius * math.cos(radians)
            y = shape.center.y + shape.radius * math.sin(radians)
            lines.append(f"mi_selectarcsegment({_number(x)},{_number(y)})")
        lines.append('mi_setarcsegmentprop(5,"outer_a_zero",0,0)')
    else:
        nodes = _contour_nodes(shape)
        for (x1, y1), (x2, y2) in zip(nodes, nodes[1:] + nodes[:1]):
            lines.append(f"mi_selectsegment({_number((x1+x2)/2)},{_number((y1+y2)/2)})")
        lines.append('mi_setsegmentprop("outer_a_zero",0,1,0,0)')
    return lines + ["mi_clearselected()"]


def _block_label(point, material, mesh, circuit, group, turns):
    return [f"mi_addblocklabel({_number(point.x)},{_number(point.y)})",
            f"mi_selectlabel({_number(point.x)},{_number(point.y)})",
            f"mi_setblockprop({_quote(material)},0,{_number(mesh)},{_quote(circuit)},0,{group},{turns})",
            "mi_clearselected()"]


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


def lua_script(plan: GeometryPlan) -> str:
    """Create a complete script through drawing/material/boundary/solve stages."""
    # Validate again at the compiler boundary even if a caller has used
    # Pydantic model_construct or a mutable model's attributes were changed.
    plan = GeometryPlan.model_validate(plan.model_dump())
    lines = [
        "-- research_extender geometry-plan compiler. Native FEMM 5 / Lua 4.",
        "-- Planar DC magnetostatics, millimeters, linear isotropic materials.",
        "-- Stage 1: define the problem.", "newdocument(0)",
        f'mi_probdef(0,"millimeters","planar",1e-8,{_number(plan.depth_mm)},30)',
        "-- Stage 2: draw closed contours.",
    ]
    lines.extend(_draw_contour(plan.domain))
    for region in plan.regions:
        lines.extend(_draw_contour(region.shape))
    lines.extend(["-- Stage 3: create materials and circuits.",
                  'mi_addmaterial("air",1,1,0,0,0)'])
    for material in plan.materials:
        mu = _number(material.relative_permeability)
        lines.append(f"mi_addmaterial({_quote(material.id)},{mu},{mu},0,0,0)")
    for circuit in plan.circuits:
        lines.append(f"mi_addcircprop({_quote(circuit.id)},{_number(circuit.current_a)},1)")
    lines.append("-- Stage 4: assign every material face, excitation and mesh.")
    lines.extend(_block_label(plan.air_label, "air", plan.mesh_mm, "<None>", 0, 0))
    for group, region in enumerate(plan.regions, start=1):
        lines.extend(_block_label(region.label, region.material, region.mesh_mm or plan.mesh_mm,
                                  region.circuit or "<None>", group, region.turns))
    lines.append("-- Stage 5: constrain the outer contour to A=0.")
    lines.extend(_assign_outer_boundary(plan.domain))
    lines.extend(["-- Stage 6: mesh, solve, and export measured quantities.",
                  'mi_saveas("model.fem")', "mi_analyze(1)", "mi_loadsolution()",
                  'out = openfile("results.csv","w")', 'write(out,"quantity,value,unit\\n")',
                  "mo_groupselectblock()", "energy = mo_blockintegral(2)", "mo_clearblock()",
                  'write(out,"energy,",energy,",J\\n")'])
    if plan.force_region:
        group = next(index for index, region in enumerate(plan.regions, start=1)
                     if region.id == plan.force_region)
        lines.extend([f"mo_groupselectblock({group})", "fx = mo_blockintegral(18)",
                      "fy = mo_blockintegral(19)", "mo_clearblock()",
                      'write(out,"force_x,",fx,",N\\n")', 'write(out,"force_y,",fy,",N\\n")'])
    for circuit in plan.circuits:
        lines.extend([f"amps, volts, linkage = mo_getcircuitproperties({_quote(circuit.id)})",
                      f'write(out,"current_{circuit.id},",amps,",A\\n")',
                      f'write(out,"flux_linkage_{circuit.id},",linkage,",Wb-turn\\n")'])
    for index, point in enumerate(plan.probes):
        lines.extend([f"a, bx, by = mo_getpointvalues({_number(point.x)},{_number(point.y)})",
                      f'write(out,"bx_{index},",bx,",T\\n")',
                      f'write(out,"by_{index},",by,",T\\n")'])
    lines.extend(["closefile(out)", 'out = openfile("field.csv","w")',
                  'write(out,"x_mm,y_mm,bx_t,by_t\\n")'])
    # This mask is computed from the validated contour. Its values are numeric
    # constants; the provider cannot supply a Lua predicate or other code.
    xmin, xmax, ymin, ymax = bounds(plan.domain)
    dx, dy = (xmax - xmin) / 40, (ymax - ymin) / 30
    inside = ["1" if contains(plan.domain, Point(x=xmin + col * dx, y=ymin + row * dy)) else "0"
              for row in range(31) for col in range(41)]
    lines.append("inside = {" + ",".join(inside) + "}")
    lines.extend(["for row=0,30 do", f"  y = {_number(ymin)} + row * {_number(dy)}",
                  "  for col=0,40 do", f"    x = {_number(xmin)} + col * {_number(dx)}",
                  "    bx = 0", "    by = 0", "    if inside[row*41+col+1] == 1 then",
                  "      a, bx, by = mo_getpointvalues(x,y)", "    end",
                  '    write(out,x,",",y,",",bx,",",by,"\\n")',
                  "  end", "end", "closefile(out)", "mo_close()", "mi_close()"])
    return "\n".join(lines) + "\n"


def _svg_contour(shape, fill, stroke, width, title=""):
    common = f'fill="{fill}" stroke="{stroke}" stroke-width="{width:g}"'
    caption = f"<title>{html.escape(title)}</title>" if title else ""
    if shape.type == "circle":
        return (f'<circle cx="{shape.center.x:g}" cy="{-shape.center.y:g}" r="{shape.radius:g}" '
                f'{common}>{caption}</circle>')
    points = " ".join(f"{point.x:g},{-point.y:g}" for point in shape_points(shape))
    return f'<polygon points="{points}" {common}>{caption}</polygon>'


def preview_svg(plan: GeometryPlan) -> str:
    plan = GeometryPlan.model_validate(plan.model_dump())
    xmin, xmax, ymin, ymax = bounds(plan.domain)
    extent = max(xmax-xmin, ymax-ymin)
    pad, stroke = .035 * extent, .002 * extent
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{xmin-pad:g} {-ymax-pad:g} '
           f'{xmax-xmin+2*pad:g} {ymax-ymin+2*pad:g}" role="img" '
           f'aria-label="{html.escape(plan.title, quote=True)} geometry preview">')
    items = [_svg_contour(plan.domain, "#f3f7f8", "#66818c", stroke, "Outer A=0 boundary")]
    # Parents first, children later: air holes and nested material faces remain
    # visible rather than being obscured by their enclosing solid's fill.
    def area(region):
        points = shape_points(region.shape)
        return abs(sum(a.x*b.y-b.x*a.y for a, b in zip(points, points[1:]+points[:1])))
    for region in sorted(plan.regions, key=area, reverse=True):
        fill = "#e5f1f5" if region.material == "air" else "#eb995f" if region.circuit else "#8cabb7"
        color = "#9b4c22" if region.id == plan.force_region else "#263c46"
        items.append(_svg_contour(region.shape, fill, color, stroke, region.id))
        items.append(f'<circle cx="{region.label.x:g}" cy="{-region.label.y:g}" r="{stroke*2:g}" fill="#263c46"/>')
    for index, point in enumerate(plan.probes):
        items.append(f'<circle cx="{point.x:g}" cy="{-point.y:g}" r="{stroke*3:g}" fill="#a51c42"><title>Field probe {index}</title></circle>')
    return svg + "".join(items) + "</svg>"


EXPORT_README = """FEMM 5 geometry-plan model

Install native FEMM 5: https://femm.info/doku/doku.php?id=download5
Linux build instructions: https://femm.info/doku/doku.php?id=bindingsetup
Put femm-lua on PATH or set FEMM_LUA to its launcher path.

Run: python3 run.py
Or:  femm-lua -h model.lua

No AI key, repository checkout, or third-party Python package is needed to rerun.
model.lua contains the complete editable problem/drawing/material/excitation/
boundary/mesh/solve/export sequence. spec.json records the validated geometry
plan used to generate it. Editing spec.json alone does not regenerate model.lua.
The script writes model.fem, model.ans, results.csv and field.csv.

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


def source_files(plan: GeometryPlan) -> dict[str, str]:
    plan = GeometryPlan.model_validate(plan.model_dump())
    return {"model.lua": lua_script(plan), "run.py": RUNNER,
            "spec.json": json.dumps(plan.model_dump(), indent=2, allow_nan=False) + "\n",
            "geometry.svg": preview_svg(plan), "README.txt": EXPORT_README}


def source_zip(plan: GeometryPlan) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in source_files(plan).items():
            archive.writestr(name, content)
    return buffer.getvalue()
