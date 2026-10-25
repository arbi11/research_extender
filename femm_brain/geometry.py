"""Exact, inspectable drawing recipes and a passive FEMM geometry reader.

Recipes are independent reconstructions with illustrative defaults. They are
geometry demonstrations, not complete machine designs or validated analyses.
No downloaded script is executed by this module. Lengths are millimeters.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path
import re
import shlex

MAX_ENTITIES = 50000
EPS = 1e-9


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Geometry parameters must be finite numbers.")
    return float(value)


def _positive(value, name):
    value = _number(value)
    if value <= 0 or value > 10000:
        raise ValueError(f"{name} must be positive and no greater than 10000 mm.")
    return value


def _polar(radius, angle):
    a = math.radians(angle)
    return (radius * math.cos(a), radius * math.sin(a))


def arc_geometry(first, second, angle):
    """Return the exact center/radius of the signed circular arc in degrees."""
    x1, y1 = first
    x2, y2 = second
    angle = _number(angle)
    if not EPS < abs(angle) < 360 - EPS:
        raise ValueError("An arc angle must be nonzero and smaller than 360 degrees.")
    dx, dy = x2 - x1, y2 - y1
    chord = math.hypot(dx, dy)
    if chord < EPS:
        raise ValueError("An arc must have distinct endpoints.")
    offset = chord / (2 * math.tan(math.radians(angle) / 2))
    center = ((x1 + x2) / 2 - dy / chord * offset,
              (y1 + y2) / 2 + dx / chord * offset)
    return {"center": list(center), "radius": math.dist(first, center),
            "start_angle": math.degrees(math.atan2(y1-center[1], x1-center[0])), "angle": angle}


def _svg(geometry):
    nodes = geometry["nodes"]
    points = [(p["x"], p["y"]) for p in nodes]
    for arc in geometry["arcs"]:
        first, second = points[arc["n0"]], points[arc["n1"]]
        info = arc_geometry(first, second, arc["angle"])
        start, sweep = info["start_angle"], info["angle"]
        for degree in (0, 90, 180, 270):
            delta = ((degree-start) % 360) if sweep > 0 else ((start-degree) % 360)
            if delta <= abs(sweep) + EPS:
                q = _polar(info["radius"], degree)
                points.append((info["center"][0]+q[0], info["center"][1]+q[1]))
    if not points:
        raise ValueError("A geometry preview needs at least one point.")
    xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
    ymin, ymax = min(y for _, y in points), max(y for _, y in points)
    span = max(xmax-xmin, ymax-ymin, 1e-3)
    pad = span * .06
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{xmin-pad:.12g} {-ymax-pad:.12g} {xmax-xmin+2*pad:.12g} {ymax-ymin+2*pad:.12g}" role="img" aria-label="Exact construction geometry">',
             '<title>Exact lines and circular arcs; a geometry preview does not establish numerical accuracy.</title>',
             f'<g fill="none" stroke="#3a746e" stroke-width="{span/400:.12g}" stroke-linecap="round">']
    for segment in geometry["segments"]:
        a, b = nodes[segment["n0"]], nodes[segment["n1"]]
        ident = html.escape(segment["id"], quote=True)
        op = html.escape(segment.get("operation_id", ""), quote=True)
        parts.append(f'<line id="{ident}" data-geometry-id="{ident}" data-operation-id="{op}" x1="{a["x"]:.12g}" y1="{-a["y"]:.12g}" x2="{b["x"]:.12g}" y2="{-b["y"]:.12g}"><title>{ident}</title></line>')
    for arc in geometry["arcs"]:
        a, b = nodes[arc["n0"]], nodes[arc["n1"]]
        info = arc_geometry((a["x"], a["y"]), (b["x"], b["y"]), arc["angle"])
        radius = info["radius"]
        ident = html.escape(arc["id"], quote=True)
        op = html.escape(arc.get("operation_id", ""), quote=True)
        large, sweep = int(abs(arc["angle"]) > 180), int(arc["angle"] < 0)
        parts.append(f'<path id="{ident}" data-geometry-id="{ident}" data-operation-id="{op}" d="M {a["x"]:.12g},{-a["y"]:.12g} A {radius:.12g},{radius:.12g} 0 {large} {sweep} {b["x"]:.12g},{-b["y"]:.12g}"><title>{ident}: {arc["angle"]:g} degrees</title></path>')
    parts.append('</g>')
    for node in nodes:
        ident = html.escape(node["id"], quote=True)
        op = html.escape(node.get("operation_id", ""), quote=True)
        parts.append(f'<circle id="{ident}" data-geometry-id="{ident}" data-operation-id="{op}" cx="{node["x"]:.12g}" cy="{-node["y"]:.12g}" r="{span/450:.12g}" fill="#3a746e"><title>{ident}</title></circle>')
    for label in geometry.get("labels", []):
        x, y = label["x"], label["y"]
        text = html.escape(str(label.get("name", label["id"])))
        ident = html.escape(label["id"], quote=True)
        op = html.escape(label.get("operation_id", ""), quote=True)
        parts.append(f'<circle id="{ident}" data-geometry-id="{ident}" data-operation-id="{op}" cx="{x:.12g}" cy="{-y:.12g}" r="{span/180:.12g}" fill="#c28b37"><title>{text}</title></circle>')
    return "".join(parts) + "</svg>"


class _Drawing:
    def __init__(self):
        self.nodes, self.segments, self.arcs, self.labels, self.operations = [], [], [], [], []
        self._points = {}
        self._lines = set()

    def _op(self, function, arguments, geometry_ids):
        ident = f"op:{len(self.operations)}"
        self.operations.append({"id": ident, "function": function, "arguments": arguments,
                                "geometry_ids": geometry_ids})
        return ident

    def point(self, x, y):
        x, y = _number(x), _number(y)
        key = (round(x, 11), round(y, 11))
        if key in self._points:
            return self._points[key]
        index = len(self.nodes)
        ident = f"node:{index}"
        operation = self._op("mi_addnode", [x, y], [ident])
        self.nodes.append({"id": ident, "x": x, "y": y, "group": 0, "operation_id": operation})
        self._points[key] = index
        return index

    def line(self, first, second):
        a, b = self.point(*first), self.point(*second)
        if a == b:
            raise ValueError("A segment must have distinct endpoints.")
        key = tuple(sorted((a, b)))
        if key in self._lines:
            return
        self._lines.add(key)
        ident = f"segment:{len(self.segments)}"
        op = self._op("mi_addsegment", [*first, *second], [ident])
        self.segments.append({"id": ident, "n0": a, "n1": b, "group": 0, "operation_id": op})

    def arc(self, first, second, angle):
        a, b = self.point(*first), self.point(*second)
        arc_geometry(first, second, angle)
        ident = f"arc:{len(self.arcs)}"
        op = self._op("mi_addarc", [*first, *second, angle, 5], [ident])
        self.arcs.append({"id": ident, "n0": a, "n1": b, "angle": angle,
                          "max_segment_degrees": 5., "group": 0, "operation_id": op})

    def circle(self, radius, cx=0., cy=0.):
        points = [(cx+x, cy+y) for x, y in (_polar(radius, a) for a in (0, 90, 180, 270))]
        for first, second in zip(points, points[1:]+points[:1]):
            self.arc(first, second, 90.)

    def polygon(self, points):
        for first, second in zip(points, points[1:]+points[:1]):
            self.line(first, second)

    def label(self, x, y, name):
        ident = f"label:{len(self.labels)}"
        op = self._op("mi_addblocklabel", [x, y], [ident])
        self.labels.append({"id": ident, "x": x, "y": y, "name": name,
                            "material": None, "operation_id": op})

    def result(self):
        return {"nodes": self.nodes, "segments": self.segments, "arcs": self.arcs,
                "labels": self.labels, "operations": self.operations}


def _param(name, default, unit, description, minimum=None, maximum=None, integer=False):
    result = {"name": name, "default": default, "unit": unit, "description": description,
              "integer": integer}
    if minimum is not None:
        result["minimum"] = minimum
    if maximum is not None:
        result["maximum"] = maximum
    return result


_RECIPES = [
    {"id": "circle", "title": "Circular contour", "parameters": [
        _param("radius", 10., "mm", "Radius of the contour", .01, 1000)],
     "description": "Four exact quarter-circle arcs, without polyline approximation."},
    {"id": "rectangle", "title": "Rectangular section", "parameters": [
        _param("width", 20., "mm", "Horizontal width", .01, 1000),
        _param("height", 10., "mm", "Vertical height", .01, 1000)],
     "description": "A closed, centered rectangular cross-section."},
    {"id": "coil_section", "title": "Coil cross-section", "parameters": [
        _param("width", 8., "mm", "Cross-section width", .01, 1000),
        _param("height", 15., "mm", "Cross-section height", .01, 1000),
        _param("center_x", 20., "mm", "Horizontal center", -1000, 1000)],
     "description": "A winding-region geometry with a label; current and turns must be assigned separately."},
    {"id": "stator_tooth", "title": "Curved-bore stator tooth", "parameters": [
        _param("bore_radius", 20., "mm", "Bore radius; center is the drawing origin", .1, 1000),
        _param("tip_width", 7., "mm", "Chord width of the tooth tip", .01, 1000),
        _param("body_width", 5., "mm", "Width of the parallel tooth body", .01, 1000),
        _param("tip_height", 1.5, "mm", "Tip flange radial thickness", .01, 1000),
        _param("height", 8., "mm", "Overall tooth height beyond the bore radius", .1, 1000)],
     "description": "A parameterized tooth with an exact circular bore face, straight body and explicit tip shoulders."},
    {"id": "slotted_stator", "title": "Repeated semi-closed-slot stator", "parameters": [
        _param("slot_count", 12, "count", "Number of repeated slots", 3, 96, True),
        _param("bore_radius", 20., "mm", "Stator bore radius", .1, 1000),
        _param("outer_radius", 35., "mm", "Outer stator/yoke radius", .2, 1000),
        _param("slot_depth", 8., "mm", "Slot depth beyond the bore", .01, 1000),
        _param("slot_opening", 3., "mm", "Slot-opening chord width", .01, 1000),
        _param("slot_width", 6., "mm", "Slot-root chord width", .01, 1000),
        _param("tip_depth", 1.2, "mm", "Depth of the narrow opening neck", .01, 1000)],
     "description": "One connected steel-region outline: exact outer circle and a continuous inner boundary with repeated semi-closed slots."},
    {"id": "circular_airgap", "title": "Concentric rotor, air gap and stator", "parameters": [
        _param("rotor_radius", 20., "mm", "Outer rotor radius", .1, 1000),
        _param("gap", 1., "mm", "Physical radial air-gap width", .01, 1000),
        _param("outer_radius", 35., "mm", "Outer stator radius", .2, 1000)],
     "description": "Three exact circular contours and named region seeds. Boundary-condition choices are a separate recipe."},
    {"id": "rotor_bar", "title": "Rounded rotor-bar cross-section", "parameters": [
        _param("length", 8., "mm", "Overall capsule length", .01, 1000),
        _param("width", 3., "mm", "Capsule width", .01, 1000),
        _param("center_x", 15., "mm", "Horizontal center", -1000, 1000),
        _param("angle", 0., "deg", "Rotation about the capsule center", -360, 360)],
     "description": "Two straight flanks and two exact semicircular ends; conductor properties are not inferred."},
    {"id": "flux_barrier", "title": "Curved flux barrier with rounded ends", "parameters": [
        _param("inner_radius", 12., "mm", "Inner radius of the barrier", .1, 1000),
        _param("thickness", 2., "mm", "Radial barrier width", .01, 1000),
        _param("span", 60., "deg", "Angular span between end-cap centers", 1, 150),
        _param("center_angle", 90., "deg", "Angle of the barrier center", -360, 360)],
     "description": "Two concentric circular arcs joined by tangent semicircular end caps. Suitable as a geometry primitive, not a complete SynRM/IPM rotor."},
]

_CONSTRUCTION = {
    "circle": ["Calculate points at 0, 90, 180 and 270 degrees.", "Join consecutive points with four 90-degree circular arcs."],
    "rectangle": ["Calculate the four corners as (+/- width/2, +/- height/2).", "Join the corners in order with straight segments."],
    "coil_section": ["Offset the four rectangular corners by center_x.", "Draw four straight segments and place an interior winding-region label.", "Assign current, turns and material separately; a label alone is not an excitation."],
    "stator_tooth": ["Calculate the bore-face chord endpoints: x = +/- tip_width/2; y = sqrt(bore_radius^2 - (tip_width/2)^2).", "Join the endpoints with the bore arc: sweep = 2 asin(tip_width/(2 bore_radius)).", "Add the tip-flange, body-side and outer-root segments in boundary order.", "Check that the shoulders lie outside the bore arc and that the boundary closes."],
    "slotted_stator": ["Set pitch = 360/slot_count; set slot-root radius = bore_radius + slot_depth.", "Calculate opening half-angle = asin(slot_opening/(2 bore_radius)) and root half-angle = asin(slot_width/(2 slot_root_radius)).", "For each pitch, connect opening, neck and root points, draw the circular slot-root arc, then continue along the circular bore face to the next slot.", "Draw the separate outer-circle boundary. The steel yoke remains connected around every slot.", "Reject overlapping slots and a nonpositive remaining yoke thickness."],
    "circular_airgap": ["Draw circles at rotor_radius, rotor_radius + gap, and outer_radius.", "Place separate seeds inside the rotor, radial air gap and stator annulus.", "Choose and assign physical or periodic air-gap boundary conditions separately."],
    "rotor_bar": ["Set end-cap radius = width/2 and end-center offset = (length-width)/2.", "Draw parallel straight flanks and two exact 180-degree end-cap arcs.", "Rotate all coordinates about the bar center by angle and translate to center_x."],
    "flux_barrier": ["Calculate start/end angles as center_angle +/- span/2.", "Draw arcs at inner_radius and inner_radius + thickness over the requested span.", "Join corresponding ends using tangent 180-degree caps of radius thickness/2.", "Check that the rounded end caps remain separate."],
}


def recipe_catalog():
    """Return portable metadata for the independently reconstructed recipes."""
    return json.loads(json.dumps([{**item, "source_ids": ["manual5"],
        "verification": {"status": "reconstructed", "note": "Illustrative parameterized geometry, independently reconstructed from documented FEMM primitive operations; no numerical simulation is claimed."}}
        for item in _RECIPES]))


def _parameters(recipe, supplied):
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, dict):
        raise ValueError("Parameters must be an object.")
    specs = {p["name"]: p for p in recipe["parameters"]}
    if set(supplied) - set(specs):
        raise ValueError("Unknown recipe parameter: " + ", ".join(sorted(set(supplied)-set(specs))))
    result = {}
    for name, spec in specs.items():
        value = supplied.get(name, spec["default"])
        value = _number(value)
        if spec.get("integer"):
            if not value.is_integer():
                raise ValueError(name + " must be an integer.")
            value = int(value)
        if value < spec.get("minimum", -math.inf) or value > spec.get("maximum", math.inf):
            raise ValueError(name + " is outside its supported range.")
        result[name] = value
    return result


def construct(recipe_id, parameters=None):
    """Build exact primitives from checked parameters and return a linked SVG."""
    recipes = {r["id"]: r for r in recipe_catalog()}
    if recipe_id not in recipes:
        raise ValueError("Unknown geometry recipe.")
    recipe = recipes[recipe_id]
    p = _parameters(recipe, parameters)
    drawing = _Drawing()
    if recipe_id == "circle":
        drawing.circle(p["radius"])
    elif recipe_id in ("rectangle", "coil_section"):
        w, h, cx = p["width"], p["height"], p.get("center_x", 0.)
        drawing.polygon([(cx-w/2, -h/2), (cx+w/2, -h/2), (cx+w/2, h/2), (cx-w/2, h/2)])
        if recipe_id == "coil_section":
            drawing.label(cx, 0., "winding: assign current and turns")
    elif recipe_id == "stator_tooth":
        r, tip, body = p["bore_radius"], p["tip_width"], p["body_width"]
        if not body <= tip < 2*r:
            raise ValueError("Tooth body width must not exceed tip width, and tip width must be smaller than the bore diameter.")
        if p["tip_height"] >= p["height"]:
            raise ValueError("Tip height must be smaller than tooth height.")
        y = math.sqrt(r*r-(tip/2)**2)
        if y+p["tip_height"] <= r:
            raise ValueError("Tip height must exceed the bore-face sagitta to avoid crossing shoulders.")
        first, second = (tip/2, y), (-tip/2, y)
        drawing.arc(first, second, math.degrees(2*math.asin(tip/(2*r))))
        points = [second, (-tip/2, y+p["tip_height"]), (-body/2, y+p["tip_height"]),
                  (-body/2, r+p["height"]), (body/2, r+p["height"]),
                  (body/2, y+p["tip_height"]), (tip/2, y+p["tip_height"]), first]
        # Equality of body and tip widths removes zero-length shoulders.
        for a, b in zip(points, points[1:]):
            if math.dist(a, b) > EPS:
                drawing.line(a, b)
        drawing.label(0., r+p["height"]*.6, "tooth geometry")
    elif recipe_id == "slotted_stator":
        n, rb, ro = p["slot_count"], p["bore_radius"], p["outer_radius"]
        rr, rn = rb+p["slot_depth"], rb+p["tip_depth"]
        pitch = 360/n
        if not rb < rn < rr < ro:
            raise ValueError("Require bore < neck < slot root < outer radius, leaving a positive yoke thickness.")
        if p["slot_opening"] >= 2*rb or p["slot_width"] >= 2*rr:
            raise ValueError("Slot chord widths must be smaller than their supporting circle diameters.")
        opening = math.degrees(math.asin(p["slot_opening"]/(2*rb)))
        root = math.degrees(math.asin(p["slot_width"]/(2*rr)))
        if root < opening or max(root, opening) >= pitch/2:
            raise ValueError("Slot roots must be at least as wide angularly as the opening and must not overlap adjacent slots.")
        drawing.circle(ro)
        for i in range(n):
            c = i*pitch
            a, b = _polar(rb,c-opening), _polar(rn,c-opening)
            d, e = _polar(rr,c-root), _polar(rr,c+root)
            f, g = _polar(rn,c+opening), _polar(rb,c+opening)
            drawing.line(a,b); drawing.line(b,d)
            drawing.arc(d,e,2*root)
            drawing.line(e,f); drawing.line(f,g)
            drawing.arc(g,_polar(rb,(i+1)*pitch-opening),pitch-2*opening)
        drawing.label((rr+ro)/2,0.,"stator yoke")
    elif recipe_id == "circular_airgap":
        r, gap, ro = p["rotor_radius"], p["gap"], p["outer_radius"]
        if r+gap >= ro:
            raise ValueError("The stator outer radius must exceed rotor radius plus air gap.")
        for radius in (r,r+gap,ro):
            drawing.circle(radius)
        drawing.label(0.,0.,"rotor"); drawing.label(r+gap/2,0.,"air gap")
        drawing.label((r+gap+ro)/2,0.,"stator")
    elif recipe_id == "rotor_bar":
        length, width, cx, angle = p["length"], p["width"], p["center_x"], p["angle"]
        if length <= width:
            raise ValueError("Capsule length must be greater than its width.")
        radius, half = width/2, (length-width)/2
        def transform(point):
            a = math.radians(angle)
            return (cx+point[0]*math.cos(a)-point[1]*math.sin(a),point[0]*math.sin(a)+point[1]*math.cos(a))
        a,b,c,d = map(transform,[(-half,-radius),(half,-radius),(half,radius),(-half,radius)])
        drawing.line(a,b); drawing.arc(b,c,180.); drawing.line(c,d); drawing.arc(d,a,180.)
        drawing.label(cx,0.,"rotor bar")
    elif recipe_id == "flux_barrier":
        ri, width, span, center = p["inner_radius"], p["thickness"], p["span"], p["center_angle"]
        if width >= 2*ri:
            raise ValueError("Barrier width must be smaller than twice the inner radius.")
        if 2*(ri+width/2)*math.sin(math.radians(span)/2) <= width:
            raise ValueError("Barrier angular span must keep its rounded end caps separate.")
        start, end = center-span/2, center+span/2
        a,b,c,d = _polar(ri+width,start),_polar(ri+width,end),_polar(ri,end),_polar(ri,start)
        drawing.arc(a,b,span); drawing.arc(b,c,180.)
        drawing.arc(d,c,span); drawing.arc(d,a,180.)
        q = _polar(ri+width/2,center)
        drawing.label(*q,"flux barrier")
    geometry = drawing.result()
    validation = _validate_geometry(geometry)
    return {"recipe_id": recipe_id, "title": recipe["title"], "description": recipe["description"],
            "parameters": p, "parameter_schema": recipe["parameters"], **geometry,
            "construction_steps": _CONSTRUCTION[recipe_id],
            "svg": _svg(geometry), "validation": validation, "source_ids": ["manual5"],
            "verification": {"status": "geometry_checked", "note": "Finite coordinates, unique entities, endpoint connectivity and closed boundary loops checked. Defaults are illustrative; native mesh, physical assignments and numerical convergence have not been verified."},
            "metadata": {"units": "millimeters", "coordinate_system": "cartesian", "origin": "independently_reconstructed", "dimension_source": "illustrative_parameters"}}


def _validate_geometry(geometry):
    nodes = geometry["nodes"]
    if not nodes or len(nodes) > MAX_ENTITIES:
        raise ValueError("Geometry node-count limit exceeded or no nodes are present.")
    degree = [0]*len(nodes)
    for item in geometry["segments"]+geometry["arcs"]:
        a,b = item["n0"],item["n1"]
        if not 0 <= a < len(nodes) or not 0 <= b < len(nodes) or a == b:
            raise ValueError("A primitive references an invalid endpoint.")
        degree[a]+=1; degree[b]+=1
    closed = all(value == 2 for value in degree)
    if not closed:
        raise ValueError("Reconstructed recipe boundaries must form closed non-branching loops.")
    return {"status": "passed", "checks": ["finite parameters", "parameter constraints", "unique nodes and segments", "valid primitive endpoints", "closed non-branching boundary loops"],
            "node_count": len(nodes), "segment_count": len(geometry["segments"]), "arc_count": len(geometry["arcs"]),
            "numerical_convergence_verified": False, "native_mesh_verified": False}


def _split_values(line):
    try:
        return shlex.split(line, comments=False, posix=True)
    except ValueError as exc:
        raise ValueError("Malformed quoted FEMM record.") from exc


def parse_fem(text, *, physics="magnetics"):
    """Read native FEMM 4/5 text geometry passively, preserving source attributes.

    Native entity row numbers become stable IDs. Material/circuit/boundary
    records are retained as property dictionaries; no source script is executed.
    ``physics`` selects the native drawing-call namespace because the text
    geometry sections are shared across .fem, .fee, .feh and .fec files.
    """
    namespaces = {
        "magnetics": ("magnetics", "mi"), "mi": ("magnetics", "mi"), "fem": ("magnetics", "mi"),
        "electrostatics": ("electrostatics", "ei"), "ei": ("electrostatics", "ei"), "fee": ("electrostatics", "ei"),
        "heatflow": ("heatflow", "hi"), "heat_flow": ("heatflow", "hi"), "hi": ("heatflow", "hi"), "feh": ("heatflow", "hi"),
        "currentflow": ("currentflow", "ci"), "current_flow": ("currentflow", "ci"), "ci": ("currentflow", "ci"), "fec": ("currentflow", "ci"),
    }
    if not isinstance(physics, str) or physics.lower().lstrip(".") not in namespaces:
        raise ValueError("Physics must be magnetics, electrostatics, heatflow or currentflow.")
    physics, prefix = namespaces[physics.lower().lstrip(".")]
    if not isinstance(text,str) or len(text.encode("utf-8")) > 32*1024*1024:
        raise ValueError("FEMM input must be text smaller than 32 MiB.")
    lines = text.lstrip("\ufeff").splitlines()
    sections,metadata,properties = {},{"physics":physics}, {"materials":[],"boundaries":[],"circuits":[],"point_properties":[]}
    aliases = {"numpoints":"nodes","numsegments":"segments","numarcsegments":"arcs","numblocklabels":"labels","numholes":"holes"}
    blocks = {"block":"materials","bdry":"boundaries","circuit":"circuits","point":"point_properties"}
    current_block = None
    i = 0
    while i < len(lines):
        line = lines[i].strip(); i+=1
        if not line or line.startswith(("#",";")):
            continue
        begin = re.fullmatch(r"<Begin(Block|Bdry|Circuit|Point)>",line,re.I)
        if begin:
            current_block = {"source_line":i,"properties":{}}
            properties[blocks[begin.group(1).lower()]].append(current_block)
            continue
        if re.fullmatch(r"<End(?:Block|Bdry|Circuit|Point)>",line,re.I):
            current_block = None; continue
        match = re.match(r"\[([^]]+)\]\s*=\s*(.*)",line)
        if match:
            key,value = match.group(1),match.group(2).strip()
            canonical = key.lower().replace("_","")
            if canonical in aliases:
                try: count=int(value)
                except ValueError: raise ValueError("Invalid FEMM entity count.") from None
                if not 0 <= count <= MAX_ENTITIES:
                    raise ValueError("FEMM entity count exceeds the supported limit.")
                rows=[]
                for _ in range(count):
                    while i<len(lines) and not lines[i].strip(): i+=1
                    if i>=len(lines) or lines[i].lstrip().startswith("["):
                        raise ValueError("Truncated FEMM geometry section.")
                    rows.append((i+1,_split_values(lines[i]))); i+=1
                sections[aliases[canonical]]=rows
            else:
                metadata[key]=value.strip('"')
            continue
        prop = re.match(r"<([^>]+)>\s*=\s*(.*)",line)
        if prop and current_block is not None:
            key,value = prop.groups()
            current_block["properties"][key]=value.strip().strip('"')
            if key.lower() in ("blockname","bdryname","circuitname","pointname"):
                current_block["name"]=value.strip().strip('"')
            if key.lower() == "bhpoints":
                try:
                    count = int(value)
                except ValueError:
                    raise ValueError("Invalid FEMM B-H point count.") from None
                if not 0 <= count <= MAX_ENTITIES:
                    raise ValueError("FEMM B-H point count exceeds the supported limit.")
                curve = []
                for _ in range(count):
                    if i >= len(lines):
                        raise ValueError("Truncated FEMM B-H curve.")
                    fields = _split_values(lines[i]); i += 1
                    if len(fields) != 2:
                        raise ValueError("Malformed FEMM B-H curve row.")
                    try:
                        point = [float(field) for field in fields]
                    except ValueError:
                        raise ValueError("Invalid FEMM B-H curve value.") from None
                    if not all(math.isfinite(v) for v in point):
                        raise ValueError("Non-finite FEMM B-H curve value.")
                    curve.append(point)
                current_block["bh_points"] = curve
    if "nodes" not in sections or not sections["nodes"]:
        raise ValueError("No native FEMM point section was found.")
    drawing = {"nodes":[],"segments":[],"arcs":[],"labels":[],"operations":[]}
    def number(value):
        try: result=float(value)
        except (ValueError,OverflowError): raise ValueError("Invalid numeric FEMM coordinate.") from None
        if not math.isfinite(result): raise ValueError("Non-finite FEMM coordinate.")
        return result
    def integer(value):
        if not re.fullmatch(r"[-+]?\d+",value): raise ValueError("Invalid integer FEMM index.")
        return int(value)
    def operation(function,args,ident):
        op=f'op:{len(drawing["operations"])}'
        drawing["operations"].append({"id":op,"function":function,"arguments":args,"geometry_ids":[ident],"origin":"native_file_reconstruction"})
        return op
    for index,(line,row) in enumerate(sections["nodes"]):
        if len(row)<2: raise ValueError("Malformed FEMM point row.")
        x,y=number(row[0]),number(row[1]); ident=f"node:{index}"
        drawing["nodes"].append({"id":ident,"x":x,"y":y,"source_line":line,"attributes":row[2:],"group":integer(row[3]) if len(row)>3 else 0,"operation_id":operation(prefix+"_addnode",[x,y],ident)})
    points=drawing["nodes"]
    for section,function in (("segments",prefix+"_addsegment"),("arcs",prefix+"_addarc")):
        for index,(line,row) in enumerate(sections.get(section,[])):
            minimum=4 if section=="arcs" else 2
            if len(row)<minimum: raise ValueError("Malformed FEMM primitive row.")
            n0,n1=integer(row[0]),integer(row[1])
            if not 0<=n0<len(points) or not 0<=n1<len(points) or n0==n1:
                raise ValueError("FEMM primitive references an invalid endpoint.")
            ident=("arc:" if section=="arcs" else "segment:")+str(index)
            a,b=points[n0],points[n1]
            args=[a["x"],a["y"],b["x"],b["y"]]
            record={"id":ident,"n0":n0,"n1":n1,"source_line":line,"attributes":row[2:]}
            if section=="arcs":
                record.update(angle=number(row[2]),max_segment_degrees=number(row[3]),group=integer(row[6]) if len(row)>6 else 0)
                arc_geometry((a["x"],a["y"]),(b["x"],b["y"]),record["angle"])
                args += [record["angle"],record["max_segment_degrees"]]
            else:
                record["group"]=integer(row[5]) if len(row)>5 else 0
            record["operation_id"]=operation(function,args,ident)
            drawing[section].append(record)
    for index,(line,row) in enumerate(sections.get("labels",[])):
        if len(row)<2: raise ValueError("Malformed FEMM block label.")
        x,y=number(row[0]),number(row[1]); ident=f"label:{index}"
        drawing["labels"].append({"id":ident,"x":x,"y":y,"source_line":line,"attributes":row[2:],"material_index":integer(row[2]) if len(row)>2 else None,"operation_id":operation(prefix+"_addblocklabel",[x,y],ident)})
    return {**drawing,**properties,"metadata":metadata,"holes":[{"x":number(row[0]),"y":number(row[1]),"source_line":line} for line,row in sections.get("holes",[])],
            "svg":_svg(drawing),"validation":{"status":"parsed","node_count":len(points),"segment_count":len(drawing["segments"]),"arc_count":len(drawing["arcs"]),"native_mesh_verified":False,"numerical_convergence_verified":False},
            "verification":{"status":"documented","note":"Exact geometry and attributes read from a native model file. Operation order is a reconstruction, not the original authoring history or proof of numerical accuracy."},"parameters":{},"source_ids":[]}


def _lua(geometry):
    lines=['-- Independently reconstructed geometry only; dimensions are illustrative.','newdocument(0)','mi_probdef(0,"millimeters","planar",1e-8,20,30)']
    for operation in geometry["operations"]:
        if operation["function"] not in {"mi_addnode","mi_addsegment","mi_addarc","mi_addblocklabel"}:
            raise ValueError("Unsupported construction operation.")
        args=','.join(format(float(value),'.17g') for value in operation["arguments"])
        lines.append(f'{operation["function"]}({args}) -- {operation["id"]}')
    lines.extend(['mi_saveas("geometry.fem")','-- Assign materials, excitation and boundaries before meshing or solving.'])
    return '\n'.join(lines)+'\n'


def _fem(geometry):
    lines=['[Format] = 4.0','[Frequency] = 0','[Precision] = 1e-8','[MinAngle] = 30','[Depth] = 20','[LengthUnits] = millimeters','[ProblemType] = planar','[Coordinates] = cartesian','[Comment] = "Reconstructed geometry demonstration; physical assignments are intentionally incomplete."','[PointProps] = 0','[BdryProps] = 0','[BlockProps] = 0','[CircuitProps] = 0',f'[NumPoints] = {len(geometry["nodes"])}']
    lines += [f'{p["x"]:.17g} {p["y"]:.17g} 0 {p.get("group",0)}' for p in geometry["nodes"]]
    lines.append(f'[NumSegments] = {len(geometry["segments"])}')
    lines += [f'{p["n0"]} {p["n1"]} -1 0 0 {p.get("group",0)}' for p in geometry["segments"]]
    lines.append(f'[NumArcSegments] = {len(geometry["arcs"])}')
    lines += [f'{p["n0"]} {p["n1"]} {p["angle"]:.17g} {p["max_segment_degrees"]:.17g} 0 0 {p.get("group",0)}' for p in geometry["arcs"]]
    lines.extend(['[NumHoles] = 0',f'[NumBlockLabels] = {len(geometry["labels"])}'])
    lines += [f'{p["x"]:.17g} {p["y"]:.17g} 0 0 0 0 0 0 0 0' for p in geometry["labels"]]
    return '\n'.join(lines)+'\n'


def export_recipe(recipe_id, parameters=None, format="python"):
    geometry=construct(recipe_id,parameters)
    if format=="svg": return geometry["svg"]
    if format=="lua": return _lua(geometry)
    if format=="fem": return _fem(geometry)
    if format!="python": raise ValueError("Export format must be python, lua, fem or svg.")
    # Embed the actual standard-library implementation, not an import back into
    # this repository. Downloaded scripts remain editable and independent.
    source=Path(__file__).read_text(encoding='utf-8')
    footer='''

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Regenerate exact FEMM geometry from editable parameters; no solve is performed.")
    parser.add_argument("--output",default=".")
    parser.add_argument("--parameters",help="Optional JSON parameter file")
    parser.add_argument("--write-only",action="store_true",help="Geometry export is always write-only; accepted for clarity")
    args=parser.parse_args()
    values=PARAMETERS
    if args.parameters:
        values=json.loads(Path(args.parameters).read_text())
    result=construct(RECIPE_ID,values)
    destination=Path(args.output); destination.mkdir(parents=True,exist_ok=True)
    (destination/"geometry.lua").write_text(_lua(result))
    (destination/"geometry.fem").write_text(_fem(result))
    (destination/"geometry.svg").write_text(result["svg"])
    print("Wrote geometry.lua, geometry.fem and geometry.svg. Assign physical properties before solving.")
'''
    return source+'\n\nRECIPE_ID = '+repr(recipe_id)+'\nPARAMETERS = '+repr(geometry['parameters'])+footer
