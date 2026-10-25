"""Reusable, executable construction components and data-only assemblies.

A component is the executable form of a layer-2 method: it adds geometry,
properties and labels to a typed model from explicit parameters. Its knowledge
record (parameters, units, applicability, sources and graph binding) lives in
``data/components.json``; this module only implements behaviour. An assembly is
data: a problem definition, named derived values and an ordered list of
component uses. Article builds and new designs are assemblies, so no example
needs its own builder code.

Derived values use a small arithmetic language (numbers, parameter names, + - *
/ **, and a few math functions). It is parsed and evaluated without ``eval``.
"""
import ast
import hashlib
import inspect
import json
import math
from pathlib import Path

from .reproduce import COLLECTIONS, PROPERTY_KEYS, UNITS, property_values

DATA = Path(__file__).parent / "data"
EPSILON0 = 8.8541878188e-12  # FEMM's own eo constant (libfemm elec_results_doc.h)
MU0 = 4e-7 * math.pi
FEMM_DEFAULTS = {"precision": 1e-8, "min_angle": 30}
CONSTANTS = {"pi": math.pi, "mu0": MU0, "eps0": EPSILON0}
FUNCTIONS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "atan2": math.atan2,
             "radians": math.radians, "degrees": math.degrees, "min": min, "max": max, "abs": abs, "log": math.log}
LIBRARY_PHYSICS = {"matlib.dat": "magnetics", "statlib.dat": "electrostatics", "heatlib.dat": "heatflow"}
PROBLEM_KEYS = {"magnetics": {"units", "type", "frequency", "depth", "precision", "min_angle", "ac_solver", "smart_mesh"},
                "electrostatics": {"units", "type", "depth", "precision", "min_angle"},
                "heatflow": {"units", "type", "depth", "precision", "min_angle", "prev_soln", "dt"}}
POSITIVE_PROPERTIES = {"Mu_x", "Mu_y", "ex", "ey", "Kx", "Ky", "LamFill"}
NONNEGATIVE_PROPERTIES = {"Sigma", "WireD", "NStrands", "d_lam", "Kt"}
OPERATORS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
             ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b}


def evaluate(expression, names):
    """Evaluate a whitelisted arithmetic expression over named numbers."""
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in names:
                return names[node.id]
            if node.id in CONSTANTS:
                return CONSTANTS[node.id]
            raise ValueError("Unknown name in expression: " + node.id)
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = walk(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and not node.keywords:
            return FUNCTIONS[node.func.id](*[walk(arg) for arg in node.args])
        raise ValueError("Unsupported expression element: " + type(node).__name__)
    if not isinstance(expression, str) or len(expression) > 400:
        raise ValueError("Expressions must be short strings")
    value = walk(ast.parse(expression, mode="eval"))
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expression did not produce a finite number: " + expression)
    return value


def resolve(value, names):
    """Resolve nested data: '=' starts an expression, '$' names a parameter of any type."""
    if isinstance(value, str) and value.startswith("="):
        return evaluate(value[1:], names)
    if isinstance(value, str) and value.startswith("$"):
        if value[1:] not in names:
            raise ValueError("Unknown parameter reference: " + value)
        return names[value[1:]]
    if isinstance(value, list):
        return [resolve(item, names) for item in value]
    if isinstance(value, dict):
        return {key: resolve(item, names) for key, item in value.items()}
    return value


class ModelBuilder:
    """Accumulate a typed model; repeated coordinates reuse one node."""

    def __init__(self, physics, **problem):
        self.physics = physics
        self.problem = dict(FEMM_DEFAULTS, **problem)
        self.collections = {name: [] for name in COLLECTIONS[physics]}
        self.nodes, self.segments, self.arcs, self.labels, self.holes = [], [], [], [], []
        self._index = {}

    def node(self, x, y, group=0):
        """Reuse or add a node; it takes the group of the first grouped entity that uses it."""
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (x, y)):
            raise ValueError(f"Node coordinates must be finite numbers: {x!r}, {y!r}")
        key = (round(x, 12), round(y, 12))
        if key not in self._index:
            self._index[key] = len(self.nodes)
            self.nodes.append({"x": float(x), "y": float(y), "point": None, "group": group,
                               "conductor": None})
        node = self.nodes[self._index[key]]
        if not node["group"] and group:
            node["group"] = group
        return self._index[key]

    def _contour(self, boundary=None, hidden=False, group=0, conductor=None):
        if conductor and self.physics == "magnetics":
            raise ValueError("Magnetics contours carry boundaries, not conductors")
        return {"boundary": boundary, "hidden": hidden, "group": group, "conductor": conductor}

    def _ends(self, a, b, kind, group=0):
        n0, n1 = self.node(*a, group=group), self.node(*b, group=group)
        if n0 == n1:
            raise ValueError(f"Degenerate {kind}: both ends are at {a}")
        return n0, n1

    def segment(self, a, b, *, mesh=None, **properties):
        n0, n1 = self._ends(a, b, "segment", properties.get("group", 0))
        if any({s["n0"], s["n1"]} == {n0, n1} for s in self.segments):
            raise ValueError(f"Duplicate segment between {a} and {b}")
        if mesh is not None and not mesh > 0:
            raise ValueError("Segment mesh sizes must be positive")
        self.segments.append({"n0": n0, "n1": n1, "mesh": mesh, **self._contour(**properties)})

    def arc(self, a, b, angle, max_segment, **properties):
        n0, n1 = self._ends(a, b, "arc", properties.get("group", 0))
        if not 0 < angle <= 180 or not max_segment > 0:
            raise ValueError("Arcs need 0 < angle <= 180 degrees and a positive maximum segment")
        if any(x["n0"] == n0 and x["n1"] == n1 and math.isclose(x["angle"], angle) for x in self.arcs):
            raise ValueError(f"Duplicate arc from {a} to {b}")
        self.arcs.append({"n0": n0, "n1": n1, "angle": float(angle),
                          "max_segment": float(max_segment), **self._contour(**properties)})

    def label(self, x, y, material, *, mesh=None, group=0, circuit=None, magdir=0, turns=1, external=False, magdir_function=None):
        if not (_finite(x) and _finite(y)):
            raise ValueError(f"Label coordinates must be finite numbers: {x!r}, {y!r}")
        if any(math.isclose(l["x"], x, abs_tol=1e-12) and math.isclose(l["y"], y, abs_tol=1e-12) for l in self.labels):
            raise ValueError(f"Duplicate block label at ({x}, {y})")
        if mesh is not None and not mesh > 0:
            raise ValueError("Block mesh sizes must be positive")
        if circuit is None and turns != 1:
            raise ValueError("Turns are meaningful only for a label in a circuit")
        if circuit is not None and (turns == 0 or not float(turns).is_integer()):
            raise ValueError("A winding label needs a nonzero integer number of turns")
        label = {"x": float(x), "y": float(y), "material": material, "mesh": mesh, "group": group,
                 "external": external, "default": False}
        if self.physics == "magnetics":
            # A direction function replaces the angle; the engine then stores the angle as zero.
            label.update(circuit=circuit, magdir=0.0 if magdir_function else float(magdir), magdir_function=magdir_function,
                         turns=turns)
        elif circuit or turns != 1 or magdir or magdir_function:
            raise ValueError("Circuits, turns and magnetization belong to magnetics labels")
        self.labels.append(label)

    def hole(self, x, y):
        self.holes.append({"x": float(x), "y": float(y)})

    def property(self, collection, name, curve=None, **values):
        keys = PROPERTY_KEYS[(self.physics, collection)]
        unknown = set(values) - set(keys)
        if unknown:
            raise ValueError("Unknown property keys: " + ", ".join(sorted(unknown)))
        if any(item["name"] == name for item in self.collections[collection]):
            raise ValueError(f"Duplicate {collection} property: {name}")
        record = {"name": name, "props": property_values(keys, values)}
        for key, value in record["props"].items():
            if (key in POSITIVE_PROPERTIES and not value > 0) or (key in NONNEGATIVE_PROPERTIES and value < 0):
                raise ValueError(f"{collection} {name}: {key} = {value} is not physical")
        if curve:
            record["curve"] = [[float(a), float(b)] for a, b in curve]
            _check_curve(name, record["curve"], self.physics)
        self.collections[collection].append(record)

    def result(self):
        return {"physics": self.physics, "problem": self.problem, **self.collections, "nodes": self.nodes,
                "segments": self.segments, "arcs": self.arcs, "labels": self.labels, "holes": self.holes}


def _check_curve(name, curve, physics):
    """B-H curves rise monotonically from the origin side; k(T) curves have increasing T and positive k.

    FEMM's own heat library holds two-point k(T) curves, so those need two points; B-H curves need three.
    """
    least = 3 if physics == "magnetics" else 2
    if len(curve) < least or not all(_finite(a) and _finite(b) for a, b in curve):
        raise ValueError(f"Material {name}: a curve needs at least {least} finite points")
    if physics == "magnetics":
        if any(b < 0 or h < 0 for b, h in curve) or any(curve[i][0] <= curve[i-1][0] or curve[i][1] <= curve[i-1][1]
                                                        for i in range(1, len(curve))):
            raise ValueError(f"Material {name}: B-H points must be nonnegative with B and H strictly increasing")
    elif any(k <= 0 for _, k in curve) or any(curve[i][0] <= curve[i-1][0] for i in range(1, len(curve))):
        raise ValueError(f"Material {name}: conductivity curve needs increasing temperature and positive conductivity")


def _contour_options(options):
    return {key: options[key] for key in ("boundary", "conductor", "hidden", "group") if options.get(key) not in (None, False)}


# --- Geometry primitives -----------------------------------------------------

def segment(m, start, end, mesh=None, **options):
    m.segment(start, end, mesh=mesh, **_contour_options(options))


def polyline(m, points, closed=True, edges=None, **options):
    if len(points) < (3 if closed else 2):
        raise ValueError("A closed polyline needs at least three points; an open one two")
    pairs = list(zip(points, points[1:] + (points[:1] if closed else [])))
    if edges is not None and len(edges) != len(pairs):
        raise ValueError("Give one edge property entry per polyline edge")
    for index, (a, b) in enumerate(pairs):
        edge = dict(_contour_options(options), **{k: v for k, v in (edges[index] if edges else {}).items() if v is not None})
        mesh = edge.pop("mesh", None)
        m.segment(a, b, mesh=mesh, **edge)


def arc(m, start, end, angle, max_segment=1, **options):
    m.arc(start, end, angle, max_segment, **_contour_options(options))


def circle(m, radius, center=(0, 0), max_segment=1, upper=None, lower=None, **options):
    """Two 180-degree arcs; ``upper``/``lower`` override contour properties per half."""
    cx, cy = center
    m.arc((cx + radius, cy), (cx - radius, cy), 180, max_segment, **dict(_contour_options(options), **(upper or {})))
    m.arc((cx - radius, cy), (cx + radius, cy), 180, max_segment, **dict(_contour_options(options), **(lower or {})))


def point(m, at, group=0):
    """An isolated node, which the mesher keeps as a vertex."""
    m.node(at[0], at[1], group=group)


def block_label(m, at, material, mesh=None, circuit=None, turns=1, magdir=0, group=0, external=False, magdir_function=None):
    m.label(at[0], at[1], material, mesh=mesh, circuit=circuit, turns=turns, magdir=magdir, group=group, external=external,
            magdir_function=magdir_function)


def hole(m, at):
    m.hole(*at)


# --- Properties --------------------------------------------------------------

def material(m, name, values=None, curve=None):
    m.property("materials", name, curve=curve, **(values or {}))


def library_material(m, name, library, as_name=None, folder=None):
    """Copy an official library entry exactly; ``as_name`` names it in the model, ``folder`` disambiguates."""
    if LIBRARY_PHYSICS.get(library) != m.physics:
        raise ValueError(f"{library} holds {LIBRARY_PHYSICS.get(library, 'unknown')} materials; this is a {m.physics} model")
    entries = json.loads((DATA / "library_materials.json").read_text())["materials"]
    matches = [e for e in entries if e["name"] == name and e["library"] == library and (folder is None or e["folder"] == folder)]
    if len(matches) > 1:
        raise ValueError(f"{name} is ambiguous in {library}; give its folder: {[e['folder'] for e in matches]}")
    entry = matches[0] if matches else None
    if entry is None:
        raise ValueError(f"{name} is not in the bundled {library}")
    keys = PROPERTY_KEYS[(m.physics, "materials")]
    dropped = set(entry["properties"]) - set(keys)
    if dropped:
        raise ValueError(f"Library entry {name} has properties this model cannot represent: {sorted(dropped)}")
    m.property("materials", as_name or name, curve=entry["curve"], **entry["properties"])


def circuit(m, name, current, series=True):
    real, imaginary = (current if isinstance(current, list) else [current, 0])
    m.property("circuits", name, TotalAmps_re=real, TotalAmps_im=imaginary, CircuitType=1 if series else 0)


def conductor(m, name, value=0, prescribed="value", charge=0):
    if prescribed not in ("value", "charge"):
        raise ValueError("A conductor prescribes its value or its charge")
    key = "Vc" if m.physics == "electrostatics" else "Tc"
    m.property("conductors", name, **{key: value, "qc": charge, "ConductorType": 1 if prescribed == "value" else 0})


def boundary(m, name, values=None):
    m.property("boundaries", name, **(values or {}))


def fixed_potential_boundary(m, name, value=0):
    key = {"magnetics": "A_0", "electrostatics": "Vs", "heatflow": "Tset"}[m.physics]
    m.property("boundaries", name, BdryType=0, **{key: value})


def mixed_boundary(m, name, c0, c1=0):
    """Robin boundary; open-space coefficients are given explicitly by the citing assembly."""
    format_code = {"magnetics": 2, "electrostatics": 1}.get(m.physics)
    if format_code is None:
        raise ValueError("Mixed open boundaries are implemented for magnetics and electrostatics")
    m.property("boundaries", name, BdryType=format_code, c0=c0, c1=c1)


def convection_boundary(m, name, ambient_temperature, heat_transfer_coefficient):
    if m.physics != "heatflow":
        raise ValueError("Convection boundaries apply to heat flow")
    m.property("boundaries", name, BdryType=2, Tinf=ambient_temperature, h=heat_transfer_coefficient)


# --- Reusable methods --------------------------------------------------------

def iabc_open_boundary(m, radius, layers=7, center_y=0.0, dirichlet=True):
    """FEMM's improvised asymptotic boundary builder (femm_makeabc.cpp), reproduced exactly."""
    data = json.loads((DATA / "iabc_tables.json").read_text())
    axisymmetric = m.problem["type"] == "axisymmetric"
    if m.physics == "magnetics":
        table = ("uAx" if axisymmetric else "u2D") + ("0" if dirichlet else "1")
    else:
        table = ("uAx" if axisymmetric else "u2D") + ("1" if dirichlet else "0")
    prefix = "u" if m.physics == "magnetics" else "e"
    edges = [radius * (1 + k / (10 * layers)) for k in range(layers + 1)]
    for k, r in enumerate(edges):
        outer = k == layers and dirichlet
        properties = {"boundary": ("A=0" if m.physics == "magnetics" else "V=0")} if outer else {}
        m.arc((0, center_y - r), (0, center_y + r), 180, 1, **properties)
        if not axisymmetric:
            m.arc((0, center_y + r), (0, center_y - r), 180, 1, **properties)
    if axisymmetric:
        # FEMM draws one axis line from -1.1R to +1.1R and splits it at every
        # shell endpoint; interior axis nodes would split it further.
        if any(abs(n["x"]) < 1e-12 and abs(n["y"] - center_y) < radius for n in m.nodes):
            raise ValueError("Open-boundary recipe expects no interior geometry on the symmetry axis")
        stations = sorted({center_y + s * r for r in edges for s in (-1, 1)})
        for a, b in zip(stations, stations[1:]):
            m.segment((0, a), (0, b))
    for k in range(1, layers + 1):
        r = radius * (1 + (2 * k - 1) / (20 * layers))
        theta = math.pi * k / (2 * (layers + 1))
        name = prefix + str(k)
        mu = data["tables"][table][layers - 1][k - 1]
        if m.physics == "magnetics":
            m.property("materials", name, Mu_x=mu, Mu_y=mu)
        else:
            m.property("materials", name, ex=mu, ey=mu)
        m.label(r * math.cos(theta), center_y + r * math.sin(theta), name)
    if dirichlet:
        m.property("boundaries", "A=0" if m.physics == "magnetics" else "V=0")


def axisymmetric_coil_section(m, inner_radius, outer_radius, z_min, z_max, material, circuit, turns, mesh=None):
    if m.problem["type"] != "axisymmetric" or not 0 <= inner_radius < outer_radius or z_min >= z_max:
        raise ValueError("A coil section needs an axisymmetric problem and 0 <= inner < outer radius")
    corners = [(inner_radius, z_min), (outer_radius, z_min), (outer_radius, z_max), (inner_radius, z_max)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        m.segment(a, b)
    m.label((inner_radius + outer_radius) / 2, (z_min + z_max) / 2, material, mesh=mesh, circuit=circuit, turns=turns)


def bar_magnet(m, length, width, coercivity, direction=0, center=(0, 0), material="magnet", mesh=None, relative_permeability=1):
    if m.physics != "magnetics" or m.problem["type"] != "planar":
        raise ValueError("The bar magnet component is a planar magnetics cross-section")
    cx, cy = center
    m.property("materials", material, Mu_x=relative_permeability, Mu_y=relative_permeability, H_c=coercivity)
    corners = [(cx - length / 2, cy - width / 2), (cx + length / 2, cy - width / 2),
               (cx + length / 2, cy + width / 2), (cx - length / 2, cy + width / 2)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        m.segment(a, b)
    m.label(cx, cy, material, mesh=mesh, magdir=direction)


def air_gap_element(m, rotor_radius, stator_radius, max_segment=0.5, name="AGE", antiperiodic=False):
    """Concentric circles bounding a sliding-band air gap whose interior is not meshed."""
    if m.physics != "magnetics" or not 0 < rotor_radius < stator_radius:
        raise ValueError("An air-gap element needs magnetics and 0 < rotor radius < stator radius")
    m.property("boundaries", name, BdryType=7 if antiperiodic else 6)
    circle(m, rotor_radius, max_segment=max_segment, boundary=name)
    circle(m, stator_radius, max_segment=max_segment, boundary=name)
    m.hole(0, (rotor_radius + stator_radius) / 2)


def kelvin_periodic_exterior(m, domain_radius, exterior_center, applied_field, field_direction=0, max_segment=1,
                             exterior_mesh=None, material="Ext", boundaries=("pbc1", "pbc2")):
    """Unbounded planar space with a uniform applied field, via a periodically linked Kelvin disk."""
    if m.physics != "magnetics" or m.problem["type"] != "planar":
        raise ValueError("This Kelvin exterior is the planar magnetics construction")
    if math.hypot(*exterior_center) <= 2 * domain_radius:
        raise ValueError("The exterior disk must not overlap the domain")
    upper, lower = boundaries
    for name in boundaries:
        m.property("boundaries", name, BdryType=4)
    m.property("materials", material, Mu_x=1, Mu_y=1, H_c=2 * applied_field / MU0)
    circle(m, domain_radius, max_segment=max_segment, upper={"boundary": upper}, lower={"boundary": lower})
    circle(m, domain_radius, center=exterior_center, max_segment=max_segment, upper={"boundary": upper}, lower={"boundary": lower})
    m.label(exterior_center[0], exterior_center[1], material, mesh=exterior_mesh, magdir=field_direction)


# --- Electric-machine components ---------------------------------------------

def _rotated(point, degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return (point[0] * c - point[1] * s, point[0] * s + point[1] * c)


def slotted_stator(m, slot_count, slot_bottom_radius, tip_radius, tooth_width, tip_arc_length, shoe_axial_distance,
                   first_tooth_angle=0, close_slot_openings=True, slot_divider=False, tip_max_segment=1,
                   slot_bottom_max_segment=5, group=0, slot_opening_group=None, iron_material=None, iron_mesh=None):
    """Parallel-sided teeth with shoes, repeated around the axis.

    Each tooth rises from the slot-bottom circle to a shoe whose underside lies
    at ``shoe_axial_distance`` along the tooth axis; the shoe corner sits on the
    radial line through the end of the tip arc, which spans ``tip_arc_length``
    on ``tip_radius``. Teeth point outward when the tip radius exceeds the
    slot-bottom radius (outer-rotor machines) and inward otherwise.
    """
    if slot_divider and not close_slot_openings:
        raise ValueError("A slot divider needs closed slot openings")
    pitch = 360 / slot_count
    half = tooth_width / 2
    tip_half = math.degrees(tip_arc_length / (2 * tip_radius))
    if half >= min(slot_bottom_radius, tip_radius) or tip_half >= pitch / 2:
        raise ValueError("Teeth or tips overlap: reduce the tooth width or tip arc length")
    bottom_x = math.sqrt(slot_bottom_radius ** 2 - half ** 2)
    # Parallel-sided teeth are angularly widest where they are nearest the axis.
    if math.degrees(math.atan2(half, min(bottom_x, shoe_axial_distance))) >= pitch / 2:
        raise ValueError("Teeth overlap: reduce the tooth width")
    bottom_half = math.degrees(math.asin(half / slot_bottom_radius))
    corner_y = shoe_axial_distance * math.tan(math.radians(tip_half))
    corner_radius = math.hypot(shoe_axial_distance, corner_y)
    outward = tip_radius > slot_bottom_radius
    if corner_y <= half or not (min(slot_bottom_radius, tip_radius) < corner_radius < max(slot_bottom_radius, tip_radius)):
        raise ValueError("The shoe underside must lie between the slot bottom and the tips, and the shoe must overhang the tooth")
    if (shoe_axial_distance > bottom_x) != outward:
        raise ValueError("The shoe underside must be on the tip side of the slot bottom")
    tip_point = (tip_radius * math.cos(math.radians(tip_half)), tip_radius * math.sin(math.radians(tip_half)))
    opening_group = group if slot_opening_group is None else slot_opening_group
    # Points shared by neighbouring teeth are computed once, so a slot never
    # joins two copies of a corner that differ in the last bits.
    teeth = []
    for k in range(slot_count):
        angle = first_tooth_angle + k * pitch
        teeth.append({(name, sign): _rotated((x, sign * y), angle) for sign in (-1, 1)
                      for name, x, y in (("bottom", bottom_x, half), ("side", shoe_axial_distance, half),
                                         ("corner", shoe_axial_distance, corner_y), ("tip", tip_point[0], tip_point[1]))})
    for k in range(slot_count):
        angle = first_tooth_angle + k * pitch
        tooth, following = teeth[k], teeth[(k + 1) % slot_count]
        for sign in (-1, 1):
            m.segment(tooth["bottom", sign], tooth["side", sign], group=group)
            m.segment(tooth["side", sign], tooth["corner", sign], group=group)
            m.segment(tooth["corner", sign], tooth["tip", sign], group=group)
        m.arc(tooth["tip", -1], tooth["tip", 1], 2 * tip_half, tip_max_segment, group=group)
        start, end = tooth["bottom", 1], following["bottom", -1]
        corner, next_corner = tooth["corner", 1], following["corner", -1]
        if slot_divider:
            middle = _rotated((slot_bottom_radius, 0), angle + pitch / 2)
            opening_middle = ((corner[0] + next_corner[0]) / 2, (corner[1] + next_corner[1]) / 2)
            span = pitch / 2 - bottom_half
            m.arc(start, middle, span, slot_bottom_max_segment, group=group)
            m.arc(middle, end, span, slot_bottom_max_segment, group=group)
            m.segment(corner, opening_middle, group=opening_group)
            m.segment(opening_middle, next_corner, group=opening_group)
            m.segment(middle, opening_middle, group=group)
        else:
            m.arc(start, end, pitch - 2 * bottom_half, slot_bottom_max_segment, group=group)
            if close_slot_openings:
                m.segment(corner, next_corner, group=opening_group)
    if iron_material:
        m.label(*_rotated(((bottom_x + shoe_axial_distance) / 2, 0), first_tooth_angle), iron_material, mesh=iron_mesh, group=group)


def trapezoidal_slot_stator(m, slot_count, bore_radius, opening_width, opening_height, slot_width_top, slot_width_bottom,
                            slot_height, wedge_height=0, rotor_side="inner", slot_bottom="flat", first_tooth_angle=0,
                            close_slot_openings=True, slot_divider=False, bore_max_segment=1, slot_bottom_max_segment=1,
                            group=0, slot_opening_group=None, iron_material=None, iron_mesh=None):
    """Slots of the profile papers tabulate, repeated around the axis; the teeth are what lies between.

    From the bore, each slot has a parallel-sided opening (width and height),
    an optional wedge that tapers to the top width, straight sides from the top
    width to the bottom width over the slot height, and a flat or arc bottom.
    "Top" is the end nearest the air gap and widths are chords across the slot
    centerline. Slots open toward the axis for inner rotors and away from it
    for outer rotors. Slot k is centred half a pitch after tooth k.
    """
    if rotor_side not in ("inner", "outer") or slot_bottom not in ("flat", "arc"):
        raise ValueError("The rotor side is inner or outer, and the slot bottom flat or arc")
    if opening_width > slot_width_top:
        raise ValueError("The slot opening must not be wider than the slot top")
    if slot_divider and not close_slot_openings:
        raise ValueError("A slot divider needs closed slot openings")
    sign = 1 if rotor_side == "inner" else -1
    pitch = 360 / slot_count
    if opening_width >= 2 * bore_radius:
        raise ValueError("The slot opening is wider than the bore")
    bore_x = math.sqrt(bore_radius ** 2 - (opening_width / 2) ** 2)
    top_x = bore_radius + sign * opening_height
    side_x = top_x + sign * wedge_height
    bottom_x = top_x + sign * (wedge_height + slot_height)
    if min(top_x, side_x, bottom_x) <= max(slot_width_top, slot_width_bottom) / 2:
        raise ValueError("The slot reaches the axis: check the radii and slot heights")
    widest = max(math.degrees(math.asin(opening_width / (2 * bore_radius))),
                 math.degrees(math.atan2(slot_width_top / 2, side_x)), math.degrees(math.atan2(slot_width_bottom / 2, bottom_x)))
    if widest >= pitch / 2:
        raise ValueError("Adjacent slots overlap: reduce the slot widths or the slot count")
    corner_radius = math.hypot(bottom_x, slot_width_bottom / 2)
    bottom_half = math.degrees(math.atan2(slot_width_bottom / 2, bottom_x))
    opening_half = math.degrees(math.asin(opening_width / (2 * bore_radius)))
    opening_group = group if slot_opening_group is None else slot_opening_group
    profile = {("bore", -1): (bore_x, -opening_width / 2), ("bore", 1): (bore_x, opening_width / 2),
               ("top", -1): (top_x, -opening_width / 2), ("top", 1): (top_x, opening_width / 2), ("top", 0): (top_x, 0),
               ("side", -1): (side_x, -slot_width_top / 2), ("side", 1): (side_x, slot_width_top / 2),
               ("bottom", -1): (bottom_x, -slot_width_bottom / 2), ("bottom", 1): (bottom_x, slot_width_bottom / 2),
               ("bottom", 0): (bottom_x, 0) if slot_bottom == "flat" else (corner_radius, 0)}
    # Each slot's points are computed once; neighbouring slots share the bore points.
    slots = [{key: _rotated(point, first_tooth_angle + (k + 0.5) * pitch) for key, point in profile.items()}
             for k in range(slot_count)]
    for k, slot in enumerate(slots):
        for s in (-1, 1):
            m.segment(slot["bore", s], slot["top", s], group=group)
            if wedge_height or slot_width_top != opening_width:
                m.segment(slot["top", s], slot["side", s], group=group)
            m.segment(slot["side", s], slot["bottom", s], group=group)
        # An arc bottom is centred on the axis and runs counterclockwise for either rotor side.
        low, high, middle = slot["bottom", -1], slot["bottom", 1], slot["bottom", 0]
        if slot_bottom == "flat":
            if slot_divider:
                m.segment(low, middle, group=group)
                m.segment(middle, high, group=group)
            else:
                m.segment(low, high, group=group)
        elif slot_divider:
            m.arc(low, middle, bottom_half, slot_bottom_max_segment, group=group)
            m.arc(middle, high, bottom_half, slot_bottom_max_segment, group=group)
        else:
            m.arc(low, high, 2 * bottom_half, slot_bottom_max_segment, group=group)
        if close_slot_openings:
            if slot_divider:
                m.segment(slot["top", -1], slot["top", 0], group=opening_group)
                m.segment(slot["top", 0], slot["top", 1], group=opening_group)
                m.segment(slot["top", 0], middle, group=group)
            else:
                m.segment(slot["top", -1], slot["top", 1], group=opening_group)
        m.arc(slot["bore", 1], slots[(k + 1) % slot_count]["bore", -1], pitch - 2 * opening_half, bore_max_segment, group=group)
    if iron_material:
        m.label(*_rotated(((top_x + bottom_x) / 2, 0), first_tooth_angle), iron_material, mesh=iron_mesh, group=group)


def winding_layout(slot_count, pole_pairs, phases, layout="tooth_coils", layers="single", phase_sequence="forward"):
    """Star-of-slots phase and polarity for each coil (tooth coils) or slot (distributed single layer).

    Each coil's (or slot's) EMF phasor at k*p*360/Q electrical degrees joins the
    phase whose positive or negative axis is nearest; reversing the sequence
    mirrors the phasors. Returns {index: (phase, polarity)} and refuses windings
    whose phases receive different numbers of coil sides.
    """
    m = len(phases)
    if m < 1 or m % 2 == 0 or len(set(phases)) != m:
        raise ValueError("Phases need an odd number of distinct names (signed axes coincide for even counts)")
    if phase_sequence not in ("forward", "reverse") or layers not in ("single", "double"):
        raise ValueError("The phase sequence is forward or reverse, and layers are single or double")
    sign = 1 if phase_sequence == "forward" else -1
    if layout == "tooth_coils":
        if layers == "single" and slot_count % 2:
            raise ValueError("Single-layer tooth coils need an even number of slots")
        indices = range(0, slot_count, 2) if layers == "single" else range(slot_count)
    elif layout == "slot_star":
        if layers != "single":
            raise ValueError("The distributed layout is single-layer")
        indices = range(slot_count)
    else:
        raise ValueError("Unknown winding layout: " + str(layout))
    # Signed phase axes sorted by angle are 180/m apart; each owns the half-open
    # belt [axis - 90/m, axis + 90/m), so phasors on a belt boundary are never tied.
    belt = 180 / m
    axes = sorted(((j * 360 / m + (0 if polarity > 0 else 180)) % 360, phase, polarity)
                  for j, phase in enumerate(phases) for polarity in (1, -1))
    result = {}
    for k in indices:
        phasor = round((sign * k * pole_pairs * 360 / slot_count) % 360, 9)
        index = int(((phasor + belt / 2) % 360) // belt)
        _, phase, polarity = axes[index]
        result[k] = (phase, polarity)
    counts = {p: sum(1 for phase, _ in result.values() if phase == p) for p in phases}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Unbalanced winding for {slot_count} slots and {pole_pairs} pole pairs: {counts}")
    return result


def tooth_coil_winding(m, slot_count, pole_pairs, turns, material, label_radius, phases=("A", "B", "C"), layers="single",
                       layout="tooth_coils", phase_sequence="forward", first_tooth_angle=0, mesh=None, group=0,
                       layer_offset=None):
    """Block labels carrying each slot's circuit and signed turns.

    A positive coil around tooth k carries +turns in the slot clockwise of the
    tooth and -turns in the slot counterclockwise of it. Double layers place one
    label in each half of a divided slot, ``layer_offset`` degrees from the slot
    centre (a quarter pitch by default; narrow slots need less).
    """
    if not float(turns).is_integer() or turns <= 0:
        raise ValueError("Turns per coil side must be a positive integer")
    pitch = 360 / slot_count
    assignment = winding_layout(slot_count, pole_pairs, list(phases), layout, layers, phase_sequence)
    if layout == "slot_star":
        for k, (phase, polarity) in assignment.items():
            m.label(*_rotated((label_radius, 0), first_tooth_angle + (k + 0.5) * pitch), material, mesh=mesh,
                    circuit=phase, turns=int(polarity * turns), group=group)
        return
    if layer_offset is not None and not 0 < layer_offset < pitch / 2:
        raise ValueError("A layer offset lies between the slot centre and half a pitch")
    for k, (phase, polarity) in assignment.items():
        tooth = first_tooth_angle + k * pitch
        offset = pitch / 2 if layers == "single" else pitch / 2 - (pitch / 4 if layer_offset is None else layer_offset)
        m.label(*_rotated((label_radius, 0), tooth - offset), material, mesh=mesh, circuit=phase, turns=int(polarity * turns), group=group)
        m.label(*_rotated((label_radius, 0), tooth + offset), material, mesh=mesh, circuit=phase, turns=int(-polarity * turns), group=group)


def surface_magnet_rotor(m, pole_count, magnet_material, thickness, magnet_shape="block", width=None, near_face_distance=None,
                         inner_radius=None, arc_angle=None, air_gap_side="inner", magnetization="radial",
                         first_pole_polarity="outward", first_pole_angle=0, rotor_angle=0, mesh=None, air_gap_face_mesh=None,
                         arc_max_segment=1, seat_arcs=False, group=0):
    """Surface permanent magnets with alternating polarity around the axis.

    Block magnets are rectangles of the given width between ``near_face_distance``
    and that distance plus ``thickness`` along each pole axis; arc magnets span
    ``arc_angle`` between ``inner_radius`` and ``inner_radius + thickness``. The
    air-gap side (inner for outer-rotor machines) may carry its own mesh size.
    Radial magnetization of arc magnets follows each element's angle.
    ``seat_arcs`` draws the core surface between arc magnets on the face away
    from the air gap, so the rotor core is closed.
    """
    if pole_count % 2 or pole_count < 2:
        raise ValueError("Pole count must be even")
    if first_pole_polarity not in ("outward", "inward") or air_gap_side not in ("inner", "outer"):
        raise ValueError("Polarity is outward or inward; the air-gap side is inner or outer")
    if magnetization not in ("radial", "parallel"):
        raise ValueError("Magnetization is radial or parallel")
    pitch = 360 / pole_count
    seat_corners = []
    for k in range(pole_count):
        axis = first_pole_angle + rotor_angle + k * pitch
        outward = (k % 2 == 0) == (first_pole_polarity == "outward")
        if magnet_shape == "block":
            if not width or near_face_distance is None or near_face_distance <= 0:
                raise ValueError("Block magnets need a width and a positive near-face distance")
            near, far, half = near_face_distance, near_face_distance + thickness, width / 2
            corners = [(near, -half), (far, -half), (far, half), (near, half)]
            if math.degrees(math.atan2(half, near)) >= pitch / 2:
                raise ValueError("Adjacent block magnets overlap")
            gap_face = {(near, -half), (near, half)} if air_gap_side == "inner" else {(far, -half), (far, half)}
            for a, b in zip(corners, corners[1:] + corners[:1]):
                face_mesh = air_gap_face_mesh if {a, b} == gap_face else None
                m.segment(_rotated(a, axis), _rotated(b, axis), mesh=face_mesh, group=group)
            center = ((near + far) / 2, 0)
            function = None
        elif magnet_shape == "arc":
            if not arc_angle or not inner_radius or arc_angle >= pitch:
                raise ValueError("Arc magnets need an inner radius and an arc angle below the pole pitch")
            if air_gap_face_mesh:
                raise ValueError("Arc magnets set their air-gap discretization through arc_max_segment, not a face mesh")
            outer = inner_radius + thickness
            ends = [_rotated((r, 0), axis + s * arc_angle / 2) for r in (inner_radius, outer) for s in (-1, 1)]
            m.arc(ends[0], ends[1], arc_angle, arc_max_segment, group=group)
            m.arc(ends[2], ends[3], arc_angle, arc_max_segment, group=group)
            m.segment(ends[0], ends[2], group=group)
            m.segment(ends[1], ends[3], group=group)
            seat_corners.append((ends[0], ends[1]) if air_gap_side == "outer" else (ends[2], ends[3]))
            center = (inner_radius + thickness / 2, 0)
            function = ("theta" if outward else "theta+180") if magnetization == "radial" else None
        else:
            raise ValueError("Magnet shape is block or arc")
        direction = axis if outward else axis + 180
        m.label(*_rotated(center, axis), magnet_material, mesh=mesh, magdir=direction % 360, group=group,
                magdir_function=function)
    if seat_arcs:
        if magnet_shape != "arc":
            raise ValueError("Seat arcs join arc magnets; block magnets sit on their own faces")
        # Each seat arc joins the very corner points its two magnets were drawn with.
        for k in range(pole_count):
            m.arc(seat_corners[k][1], seat_corners[(k + 1) % pole_count][0], pitch - arc_angle, arc_max_segment, group=group)


IMPLEMENTATIONS = {function.__name__: function for function in (
    segment, polyline, arc, circle, point, block_label, hole, material, library_material, circuit, conductor, boundary,
    fixed_potential_boundary, mixed_boundary, convection_boundary, iabc_open_boundary, axisymmetric_coil_section,
    bar_magnet, air_gap_element, kelvin_periodic_exterior, slotted_stator, trapezoidal_slot_stator, tooth_coil_winding,
    surface_magnet_rotor)}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_parameter(component, spec, value):
    """Enforce a declared parameter type, finiteness and bounds."""
    point = lambda v: isinstance(v, (list, tuple)) and len(v) == 2 and all(_finite(x) for x in v)
    checks = {"number": lambda v: _finite(v), "integer": lambda v: _finite(v) and float(v).is_integer(),
              "boolean": lambda v: isinstance(v, bool), "string": lambda v: isinstance(v, str) and bool(v.strip()),
              "point": point, "points": lambda v: isinstance(v, list) and len(v) >= 2 and all(point(p) for p in v),
              "list": lambda v: isinstance(v, list), "object": lambda v: isinstance(v, dict),
              "number_or_complex": lambda v: _finite(v) or point(v)}
    if not checks[spec["type"]](value):
        raise ValueError(f"{component}.{spec['name']} must be a {spec['type'].replace('_', ' ')}; got {value!r}")
    if spec["type"] in ("number", "integer"):
        low, high, above = spec.get("minimum"), spec.get("maximum"), spec.get("exclusive_minimum")
        if (low is not None and value < low) or (high is not None and value > high) or (above is not None and not value > above):
            raise ValueError(f"{component}.{spec['name']}={value} is outside its declared range")
    if "choices" in spec and value not in spec["choices"]:
        raise ValueError(f"{component}.{spec['name']} must be one of {spec['choices']}")


def catalog():
    return {item["id"]: item for item in json.loads((DATA / "components.json").read_text())["components"]}


def component_digests(identifiers):
    """Implementation digests for components and everything they use.

    A digest covers the component's function, its knowledge record, the shared
    model builder and the digests of components it uses, so evidence recorded
    under one digest goes stale when any of them changes.
    """
    records = catalog()
    shared = inspect.getsource(ModelBuilder) + inspect.getsource(_contour_options) + inspect.getsource(property_values)
    digests = {}

    def digest(identifier, trail=()):
        if identifier in digests:
            return digests[identifier]
        if identifier in trail:
            raise ValueError("Component use cycle: " + identifier)
        record = records[identifier]
        parts = [inspect.getsource(IMPLEMENTATIONS[identifier]), json.dumps(record, sort_keys=True), shared]
        parts += [digest(used, trail + (identifier,)) for used in sorted(record["uses_components"])]
        digests[identifier] = hashlib.sha256("\n".join(parts).encode()).hexdigest()
        return digests[identifier]

    for identifier in identifiers:
        digest(identifier)
    return digests


def assemble(assembly, parameters):
    """Build a typed model from an assembly; return it with the components used."""
    records = catalog()
    names = dict(parameters)
    for name, value in (assembly.get("let") or {}).items():
        if name in names or name in CONSTANTS:
            raise ValueError("Derived value shadows a parameter or constant: " + name)
        names[name] = resolve(value, names)
    problem = dict(resolve(assembly["problem"], names))
    physics = problem.pop("physics", None)
    if physics not in PROBLEM_KEYS or set(problem) - PROBLEM_KEYS[physics]:
        raise ValueError(f"Unsupported problem definition: {physics} {sorted(set(problem) - PROBLEM_KEYS.get(physics, set()))}")
    if problem.get("units") not in UNITS or problem.get("type") not in ("planar", "axisymmetric"):
        raise ValueError("Problems need known length units and a planar or axisymmetric type")
    for key, low in (("depth", 0), ("frequency", None), ("precision", 0)):
        value = problem.get(key)
        if value is not None and not (_finite(value) and (value > low if low is not None else value >= 0)):
            raise ValueError(f"Problem {key} = {value!r} is invalid")
    m = ModelBuilder(physics, **problem)
    used = []
    for index, step in enumerate(assembly["steps"]):
        identifier = step.get("use")
        record = records.get(identifier)
        if record is None or identifier not in IMPLEMENTATIONS:
            raise ValueError(f"Assembly step {index} uses an unknown component: {identifier}")
        if physics not in record["physics"] or m.problem["type"] not in record["problem_types"]:
            raise ValueError(f"{identifier} does not apply to {physics} {m.problem['type']} problems")
        arguments = resolve({k: v for k, v in step.items() if k != "use"}, names)
        declared = {p["name"]: p for p in record["parameters"]}
        unknown = set(arguments) - set(declared)
        missing = {name for name, p in declared.items() if p.get("required") and name not in arguments}
        if unknown or missing:
            raise ValueError(f"{identifier}: unknown {sorted(unknown)}, missing {sorted(missing)}")
        for name, value in arguments.items():
            check_parameter(identifier, declared[name], value)
        IMPLEMENTATIONS[identifier](m, **arguments)
        if identifier not in used:
            used.append(identifier)
    return m.result(), used
