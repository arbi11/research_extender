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

from .geometry import arc_geometry
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


def sector_air_gap(m, rotor_radius, stator_radius, start_angle, end_angle, band_inner_radius, band_outer_radius, name="AGE",
                   antiperiodic=True, max_segment=0.5, air_material="Air", mesh=None, rotor_group=0, group=0):
    """A sliding-band air-gap element across a sector of a machine, for models of some poles with (anti)periodic ends.

    The rotor surface and the stator bore are drawn by other components, with
    nodes where they meet the sector's ends. This draws the band's two arcs
    (inside the gap, between ``band_inner_radius`` and ``band_outer_radius``)
    with FEMM's sliding-band boundary, the band's ends, the radial ends of the
    thin gap regions on either side with one (anti)periodic boundary pair each
    (``<name>_rotor_ends`` and ``<name>_stator_ends``), an air label in each thin
    region and a hole in the band, which is not meshed. FEMM turns the rotor side
    through the band's inner angle (boundary property 10), so rotor positions need
    no remeshing.
    """
    if m.physics != "magnetics" or not 0 < rotor_radius < band_inner_radius < band_outer_radius < stator_radius:
        raise ValueError("A sector air gap needs rotor radius < band inner radius < band outer radius < stator radius")
    span = (end_angle - start_angle) % 360
    if not 0 < span <= 180:
        raise ValueError("The sector spans more than zero and at most 180 degrees")
    edge_type = 5 if antiperiodic else 4
    m.property("boundaries", name, BdryType=7 if antiperiodic else 6)
    m.property("boundaries", name + "_rotor_ends", BdryType=edge_type)
    m.property("boundaries", name + "_stator_ends", BdryType=edge_type)
    point = lambda r, angle: (r * math.cos(math.radians(angle)), r * math.sin(math.radians(angle)))
    for radius in (band_inner_radius, band_outer_radius):
        m.arc(point(radius, start_angle), point(radius, end_angle), span, max_segment, boundary=name, group=group)
    for angle in (start_angle, end_angle):
        m.segment(point(rotor_radius, angle), point(band_inner_radius, angle), boundary=name + "_rotor_ends", group=rotor_group)
        m.segment(point(band_inner_radius, angle), point(band_outer_radius, angle), group=group)
        m.segment(point(band_outer_radius, angle), point(stator_radius, angle), boundary=name + "_stator_ends", group=group)
    middle = start_angle + span / 2
    m.label(*point((rotor_radius + band_inner_radius) / 2, middle), air_material, mesh=mesh, group=rotor_group)
    m.label(*point((band_outer_radius + stator_radius) / 2, middle), air_material, mesh=mesh, group=group)
    m.hole(*point((band_inner_radius + band_outer_radius) / 2, middle))


def arc_boundary(m, name, at):
    """Give the boundary ``name`` to existing arcs, each picked by a point lying on it."""
    if name not in {record["name"] for record in m.collections["boundaries"]}:
        raise ValueError(f"Define the boundary {name} before assigning it")
    if not at or not all(isinstance(p, (list, tuple)) and len(p) == 2 and all(_finite(v) for v in p) for p in at):
        raise ValueError("Give one finite point per arc")
    for x, y in at:
        hits = []
        for index, arc in enumerate(m.arcs):
            a, b = m.nodes[arc["n0"]], m.nodes[arc["n1"]]
            info = arc_geometry((a["x"], a["y"]), (b["x"], b["y"]), arc["angle"])
            (cx, cy), radius = info["center"], info["radius"]
            along = (math.degrees(math.atan2(y - cy, x - cx)) - info["start_angle"]) % 360
            if abs(math.hypot(x - cx, y - cy) - radius) <= 1e-9 * max(1.0, radius) and 0 < along < arc["angle"]:
                hits.append(index)
        if len(hits) != 1:
            raise ValueError(f"The point ({x}, {y}) must lie inside exactly one arc; it lies on {len(hits)}")
        m.arcs[hits[0]]["boundary"] = name


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


def segment_boundary(m, name, at):
    """Give the boundary ``name`` to existing segments, each picked by a point lying on it."""
    if name not in {record["name"] for record in m.collections["boundaries"]}:
        raise ValueError(f"Define the boundary {name} before assigning it")
    if not at or not all(isinstance(p, (list, tuple)) and len(p) == 2 and all(_finite(v) for v in p) for p in at):
        raise ValueError("Give one finite point per segment")
    for x, y in at:
        hits = []
        for index, s in enumerate(m.segments):
            a, b = m.nodes[s["n0"]], m.nodes[s["n1"]]
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            t = ((x - a["x"]) * dx + (y - a["y"]) * dy) / (dx * dx + dy * dy)
            distance = math.hypot(a["x"] + t * dx - x, a["y"] + t * dy - y)
            if 0 < t < 1 and distance <= 1e-9 * max(1.0, math.hypot(dx, dy)):
                hits.append(index)
        if len(hits) != 1:
            raise ValueError(f"The point ({x}, {y}) must lie inside exactly one segment; it lies on {len(hits)}")
        m.segments[hits[0]]["boundary"] = name


def fixed_point_potential(m, at, name, value=0):
    """Hold the potential (A, V or T) at one node, for example to fix the gauge of a model with only periodic and natural boundaries."""
    key = {"magnetics": "A_re", "electrostatics": "Vp", "heatflow": "Tp"}[m.physics]
    m.property("point_properties", name, **{key: value})
    m.nodes[m.node(at[0], at[1])]["point"] = name


def magnet_array(m, start, inner, outer, segments, magnet_material, axis="y", mesh=None, group=0, draw_ends=True):
    """Rectangular magnet blocks laid end to end along an axis, each with its own magnetization direction.

    ``segments`` lists [length, direction] in order from ``start``; the blocks
    span ``inner`` to ``outer`` across the axis. Along y in an axisymmetric
    problem this is a tubular armature (radial blocks at 0 or 180 degrees,
    axial ones at 90 or 270); along x in a planar problem, a linear array.
    Quasi-Halbach, Halbach and radial arrays are different segment lists.
    """
    if axis not in ("x", "y") or not inner < outer:
        raise ValueError("The axis is x or y, and inner must be below outer")
    if m.problem.get("type") == "axisymmetric" and axis == "y" and inner < 0:
        raise ValueError("An axisymmetric array lies at nonnegative radii")
    if not segments or any(len(item) != 2 or not _finite(item[0]) or item[0] <= 0 or not _finite(item[1]) for item in segments):
        raise ValueError("Each segment is [positive length, finite direction]")
    point = (lambda across, along: (across, along)) if axis == "y" else (lambda across, along: (along, across))
    edges = [start]
    for length, _ in segments:
        edges.append(edges[-1] + length)
    for k, (_, direction) in enumerate(segments):
        for across in (inner, outer):
            m.segment(point(across, edges[k]), point(across, edges[k + 1]), group=group)
        m.label(*point((inner + outer) / 2, (edges[k] + edges[k + 1]) / 2), magnet_material, mesh=mesh, magdir=direction % 360,
                group=group)
    for k, position in enumerate(edges):
        if draw_ends or 0 < k < len(edges) - 1:
            m.segment(point(inner, position), point(outer, position), group=group)


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


def _segment_distance(a, b):
    """Distance from the axis to the nearest point of the segment ab."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, -(a[0] * dx + a[1] * dy) / (dx * dx + dy * dy)))
    return math.hypot(a[0] + t * dx, a[1] + t * dy)


def salient_poles(m, pole_count, tip_radius, root_radius, pole_arc, root_width=None, root_shape="arc", first_pole_angle=0,
                  coil_start=None, coil_end=None, coil_width=None, tip_max_segment=1, root_max_segment=1, group=0,
                  iron_material=None, iron_mesh=None):
    """Salient poles of a stator or rotor, repeated around the axis, with optional coil blocks against their sides.

    Each pole face is an arc of ``pole_arc`` degrees on ``tip_radius``. Straight
    sides run from the face corners to root corners ``root_width`` apart on
    ``root_radius`` (the face chord by default, so parallel sides), where an arc
    centred on the axis, or a straight chord, joins the next pole. Poles point
    toward the axis when the root radius exceeds the tip radius (stators) and away
    from it otherwise (rotors). Coil blocks, when given, lie against both sides of
    every pole between ``coil_start`` and ``coil_end`` (distances from the axis
    along the pole axis) and reach ``coil_width`` beyond the side at right angles
    to the pole axis; the sides are split at the coil corners so coil and pole
    share edges.
    """
    if root_shape not in ("arc", "flat"):
        raise ValueError("The root shape is arc or flat")
    if tip_radius == root_radius:
        raise ValueError("The tip and root radii must differ")
    pitch = 360 / pole_count
    tip_half = pole_arc / 2
    face_half_width = tip_radius * math.sin(math.radians(tip_half))
    root_half_width = face_half_width if root_width is None else root_width / 2
    if tip_half >= pitch / 2:
        raise ValueError("Pole faces overlap: reduce the pole arc")
    if root_half_width >= root_radius or math.degrees(math.asin(root_half_width / root_radius)) >= pitch / 2:
        raise ValueError("Pole roots overlap: reduce the root width")
    root_half = math.degrees(math.asin(root_half_width / root_radius))
    tip_x, root_x = tip_radius * math.cos(math.radians(tip_half)), math.sqrt(root_radius ** 2 - root_half_width ** 2)
    if (root_x > tip_x) != (root_radius > tip_radius):
        raise ValueError("The pole sides must run from the face toward the root")
    coil = [coil_start, coil_end, coil_width]
    if any(value is not None for value in coil) and any(value is None for value in coil):
        raise ValueError("Coil blocks need a start, an end and a width")
    profile = {("tip", -1): (tip_x, -face_half_width), ("tip", 1): (tip_x, face_half_width),
               ("root", -1): (root_x, -root_half_width), ("root", 1): (root_x, root_half_width)}
    sides = {s: [("tip", s), ("root", s)] for s in (-1, 1)}
    if coil_start is not None:
        if not (min(tip_x, root_x) < coil_start < coil_end < max(tip_x, root_x)) or not coil_width > 0:
            raise ValueError("Coil blocks lie along the pole sides, strictly between the face and root corners")
        slope = (root_half_width - face_half_width) / (root_x - tip_x)
        on_side = lambda x: face_half_width + slope * (x - tip_x)
        inner, outer = min(tip_radius, root_radius), max(tip_radius, root_radius)
        corners = [(x, on_side(x) + across) for x in (coil_start, coil_end) for across in (0, coil_width)]
        block = [corners[0], corners[2], corners[3], corners[1]]
        flat_root = None
        if root_shape == "flat":
            following = _rotated((root_x, -root_half_width), pitch)
            flat_root = lambda p: ((following[0] - root_x) * (p[1] - root_half_width)
                                   - (following[1] - root_half_width) * (p[0] - root_x))
        for p in block:
            angle = math.degrees(math.atan2(p[1], p[0]))
            # A flat root is a chord: stator coils stay on the axis side of it, rotor coils beyond it.
            beyond_root = flat_root and not (flat_root(p) > 0 if root_radius > tip_radius else flat_root(p) < 0)
            if not 0 < angle < pitch / 2 or math.hypot(*p) >= outer or beyond_root:
                raise ValueError("Coil blocks must stay inside their half of the gap between poles")
        if any(_segment_distance(a, b) <= inner for a, b in zip(block, block[1:] + block[:1])):
            raise ValueError("Coil blocks must not reach the pole faces' circle")
        for s in (-1, 1):
            for name, (x, y) in zip(("coil_near", "coil_far", "coil_far_out", "coil_near_out"), block):
                profile[name, s] = (x, s * y)
            # Along each side from the face: the coil end nearer the face first.
            if abs(coil_start - tip_x) > abs(coil_end - tip_x):
                profile["coil_near", s], profile["coil_far", s] = profile["coil_far", s], profile["coil_near", s]
                profile["coil_near_out", s], profile["coil_far_out", s] = profile["coil_far_out", s], profile["coil_near_out", s]
            sides[s] = [("tip", s), ("coil_near", s), ("coil_far", s), ("root", s)]
    # Every pole's points are computed once; the root between two poles joins the very points each pole was drawn with.
    poles = [{key: _rotated(point, first_pole_angle + k * pitch) for key, point in profile.items()} for k in range(pole_count)]
    for k, pole in enumerate(poles):
        m.arc(pole["tip", -1], pole["tip", 1], pole_arc, tip_max_segment, group=group)
        for s in (-1, 1):
            for a, b in zip(sides[s], sides[s][1:]):
                m.segment(pole[a], pole[b], group=group)
            if coil_start is not None:
                for a, b in (("coil_far", "coil_far_out"), ("coil_far_out", "coil_near_out"), ("coil_near_out", "coil_near")):
                    m.segment(pole[a, s], pole[b, s], group=group)
        start, end = pole["root", 1], poles[(k + 1) % pole_count]["root", -1]
        if root_shape == "arc":
            m.arc(start, end, pitch - 2 * root_half, root_max_segment, group=group)
        else:
            m.segment(start, end, group=group)
    if iron_material:
        m.label(*_rotated(((tip_x + root_x) / 2, 0), first_pole_angle), iron_material, mesh=iron_mesh, group=group)


def salient_pole_winding(m, pole_count, phases, turns, material, label_axial, label_offset, first_pole_angle=0, polarities=None,
                         mesh=None, group=0):
    """Block labels of coils wound around salient poles: pole k belongs to phase k mod m.

    A coil of positive polarity carries +turns on the clockwise side of its pole
    and -turns on the counterclockwise side. By default the polarity alternates
    between successive poles of each phase, so the opposite poles of a
    switched-reluctance phase (or neighbouring poles of a field winding with one
    circuit) are north and south; ``polarities`` gives one sign per pole instead.
    Labels sit ``label_axial`` from the axis along each pole axis and
    ``label_offset`` to either side of it.
    """
    phases = list(phases)
    if not phases or len(set(phases)) != len(phases) or pole_count % len(phases):
        raise ValueError("Phases need distinct names, and the pole count must be a multiple of the phase count")
    if not float(turns).is_integer() or turns <= 0:
        raise ValueError("Turns per coil side must be a positive integer")
    if polarities is None:
        polarities = [1 if (k // len(phases)) % 2 == 0 else -1 for k in range(pole_count)]
    if len(polarities) != pole_count or any(p not in (1, -1) for p in polarities):
        raise ValueError("Give one polarity, +1 or -1, per pole")
    pitch = 360 / pole_count
    if not 0 < math.degrees(math.atan2(label_offset, label_axial)) < pitch / 2:
        raise ValueError("Each label must lie between its pole axis and the middle of the gap to the next pole")
    for k in range(pole_count):
        axis, phase, polarity = first_pole_angle + k * pitch, phases[k % len(phases)], polarities[k]
        for side in (-1, 1):
            m.label(*_rotated((label_axial, side * label_offset), axis), material, mesh=mesh, circuit=phase,
                    turns=int(-side * polarity * turns), group=group)


def belt_winding(m, inner_radius, outer_radius, belts, material, gap_material, max_segment=1, mesh=None, gap_mesh=None, group=0):
    """A slotless winding: an annulus divided into belts, each a block in a circuit with signed turns, and gaps of another material.

    ``belts`` lists [start, end, circuit, turns] with angles in degrees,
    counterclockwise from start to end. The component draws the annulus itself:
    its arcs are split at every belt edge (and wherever an arc would exceed 180
    degrees), radial edges are drawn once where neighbours meet, and every
    sector between edges that no belt covers gets a ``gap_material`` label.
    """
    if not 0 < inner_radius < outer_radius:
        raise ValueError("A belt winding needs 0 < inner radius < outer radius")
    spans = []
    for belt in belts:
        if len(belt) != 4 or not (_finite(belt[0]) and _finite(belt[1])) or not isinstance(belt[2], str):
            raise ValueError("Each belt is [start degrees, end degrees, circuit, turns]")
        start, end = belt[0] % 360, belt[1] % 360
        width = (end - start) % 360
        if not 0 < width < 360:
            raise ValueError("A belt spans more than zero and less than 360 degrees")
        spans.append((start, width, belt[2], belt[3]))
    # Belts may not overlap: walk them in order of their starts around the circle.
    ordered = sorted(spans)
    for (s0, w0, _, _), (s1, _, _, _) in zip(ordered, ordered[1:] + ordered[:1]):
        if len(ordered) > 1 and (s1 - s0) % 360 < w0 - 1e-9:
            raise ValueError("Belts overlap")
    # One list of edge angles serves arcs, radial edges and labels, so shared nodes are computed once.
    radial = sorted({round(s % 360, 12) for s, _, _, _ in spans} | {round((s + w) % 360, 12) for s, w, _, _ in spans})
    edges = []
    for a, b in zip(radial, radial[1:] + [radial[0] + 360]):
        pieces = max(1, math.ceil((b - a) / 180 - 1e-12))  # split stretches over 180 degrees so every arc is drawable
        edges += [round((a + (b - a) * k / pieces) % 360, 12) for k in range(pieces)]
    edges.sort()
    point = lambda r, angle: (r * math.cos(math.radians(angle)), r * math.sin(math.radians(angle)))
    for a, b in zip(edges, edges[1:] + [edges[0]]):
        for r in (inner_radius, outer_radius):
            m.arc(point(r, a), point(r, b), (b - a) % 360 or 360, max_segment, group=group)
    for angle in radial:
        m.segment(point(inner_radius, angle), point(outer_radius, angle), group=group)
    # One label per region between radial edges: a belt's circuit and turns, or the gap material.
    middle = (inner_radius + outer_radius) / 2
    for a, b in zip(radial, radial[1:] + [radial[0] + 360]):
        centre = (a + b) / 2
        owner = next((belt for belt in spans if (centre - belt[0]) % 360 < belt[1]), None)
        if owner is None:
            m.label(*point(middle, centre), gap_material, mesh=gap_mesh, group=group)
        else:
            m.label(*point(middle, centre), material, mesh=mesh, circuit=owner[2], turns=owner[3], group=group)


def ring_sectors(m, inner_radius, outer_radius, sectors, copies=1, angle_step=0, gap_material=None, max_segment=1, mesh=None,
                 gap_mesh=None, group=0):
    """An annulus divided into angular sectors, each a block of its own material and magnetization.

    ``sectors`` lists [start, end, material, magnetization] with angles in
    degrees, counterclockwise from start to end. Magnetization is null, "outward"
    or "inward" (radial, through FEMM's direction function) or a number (a
    parallel direction in degrees). Contiguous sectors make a closed ring of
    magnets (a magnetic gear's or Halbach rotor's); sectors with gaps between
    them make pole pieces or teeth, and the gaps take ``gap_material``. The
    sectors may describe one period, repeated ``copies`` times by ``angle_step``
    degrees (a parallel direction turns with its copy), so a pole count can be a
    parameter. Arcs are split at every sector edge and wherever they would exceed
    180 degrees, and radial edges are drawn once where neighbours meet.
    """
    if not 0 < inner_radius < outer_radius:
        raise ValueError("A ring needs 0 < inner radius < outer radius")
    if not (_finite(copies) and float(copies).is_integer() and copies >= 1) or not _finite(angle_step):
        raise ValueError("Copies is a positive whole number and the angle step a number of degrees")
    repeated = []
    for k in range(int(copies)):
        for sector in sectors:
            if len(sector) != 4 or not (_finite(sector[0]) and _finite(sector[1])):
                raise ValueError("Each sector is [start degrees, end degrees, material, magnetization]")
            turn = k * angle_step
            direction = sector[3] + turn if _finite(sector[3]) else sector[3]
            repeated.append([sector[0] + turn, sector[1] + turn, sector[2], direction])
    spans = []
    for sector in repeated:
        if len(sector) != 4 or not (_finite(sector[0]) and _finite(sector[1])) or not isinstance(sector[2], str):
            raise ValueError("Each sector is [start degrees, end degrees, material, magnetization]")
        magnetization = sector[3]
        if magnetization is not None and magnetization not in ("outward", "inward") and not _finite(magnetization):
            raise ValueError("Magnetization is null, outward, inward or a direction in degrees")
        start, width = sector[0] % 360, (sector[1] - sector[0]) % 360
        if not 0 < width < 360:
            raise ValueError("A sector spans more than zero and less than 360 degrees")
        spans.append((start, width, sector[2], magnetization))
    ordered = sorted(spans)
    for (s0, w0, _, _), (s1, _, _, _) in zip(ordered, ordered[1:] + ordered[:1]):
        if len(ordered) > 1 and (s1 - s0) % 360 < w0 - 1e-9:
            raise ValueError("Sectors overlap")
    radial = sorted({round(s % 360, 12) for s, _, _, _ in spans} | {round((s + w) % 360, 12) for s, w, _, _ in spans})
    covered = abs(sum(w for _, w, _, _ in spans) - 360) < 1e-9
    if not covered and gap_material is None:
        raise ValueError("Sectors leave gaps: give a gap material")
    edges = []
    for a, b in zip(radial, radial[1:] + [radial[0] + 360]):
        pieces = max(1, math.ceil((b - a) / 180 - 1e-12))
        edges += [round((a + (b - a) * k / pieces) % 360, 12) for k in range(pieces)]
    edges.sort()
    point = lambda r, angle: (r * math.cos(math.radians(angle)), r * math.sin(math.radians(angle)))
    for a, b in zip(edges, edges[1:] + [edges[0]]):
        for r in (inner_radius, outer_radius):
            m.arc(point(r, a), point(r, b), (b - a) % 360 or 360, max_segment, group=group)
    for angle in radial:
        m.segment(point(inner_radius, angle), point(outer_radius, angle), group=group)
    middle = (inner_radius + outer_radius) / 2
    for a, b in zip(radial, radial[1:] + [radial[0] + 360]):
        centre = (a + b) / 2
        owner = next((sector for sector in spans if (centre - sector[0]) % 360 < sector[1]), None)
        if owner is None:
            m.label(*point(middle, centre), gap_material, mesh=gap_mesh, group=group)
        elif owner[3] in ("outward", "inward"):
            m.label(*point(middle, centre), owner[2], mesh=mesh, group=group,
                    magdir_function="theta" if owner[3] == "outward" else "theta+180")
        else:
            m.label(*point(middle, centre), owner[2], mesh=mesh, magdir=(owner[3] or 0) % 360, group=group)


def flux_barrier_rotor(m, pole_count, outer_radius, barriers, rib, first_q_angle=None, poles=None, max_segment=1, material="Air",
                       mesh=None, group=0):
    """Flux barriers of a synchronous reluctance rotor, each symmetric about its pole's q-axis.

    Each barrier is [p, length, width, end_angle, end_width]: a middle section
    perpendicular to the q-axis whose edge nearer the axis lies ``p`` from it,
    ``length`` long on its centre line and ``width`` wide, and two legs from the
    middle section's ends to end points ``end_angle`` degrees from the pole's two
    d-axes. The legs taper from ``width`` at the bends to ``end_width`` at the
    ends, which are semicircles of that diameter clearing the rotor surface by
    ``rib`` (the tangential rib). The q-axis of the first pole is at
    ``first_q_angle`` (half a pole pitch by default, with a d-axis at zero); the
    barriers are drawn for ``poles`` poles (all by default) and labelled with
    ``material``. The rotor surface, core and shaft are drawn by other components.
    """
    if pole_count < 2 or pole_count % 2 or not 0 < rib < outer_radius:
        raise ValueError("A barrier rotor needs an even pole count and a rib thinner than the rotor")
    span = 360 / pole_count
    first = span / 2 if first_q_angle is None else first_q_angle
    count = pole_count if poles is None else poles
    if not (_finite(count) and float(count).is_integer() and 1 <= count <= pole_count):
        raise ValueError("Draw between one and all poles")
    shapes = []
    previous_outer = 0
    for barrier in barriers:
        if len(barrier) != 5 or not all(_finite(v) and v > 0 for v in barrier):
            raise ValueError("Each barrier is [p, length, width, end_angle, end_width], all positive")
        p, length, width, end_angle, end_width = barrier
        if p < previous_outer or end_angle >= span / 2 or end_width > 2 * width:
            raise ValueError("Barriers must lie outside one another, end inside the pole and taper sensibly")
        previous_outer = p + width
        half = math.radians(span / 2 - end_angle)
        reach = outer_radius - rib - end_width / 2
        outline = []
        for side in (-1, 1):
            # Local frame: the q-axis along +x, this leg on the side of y = side.
            bend = (p + width / 2, side * length / 2)
            end = (reach * math.cos(half), side * reach * math.sin(half))
            dx, dy = end[0] - bend[0], end[1] - bend[1]
            size = math.hypot(dx, dy)
            if size <= 0 or end[0] <= bend[0] - width:
                raise ValueError("A barrier's end must lie beyond its bend")
            d = (dx / size, dy / size)
            n = (-d[1], d[0])
            corners = {}
            for s in (-1, 1):
                a = (bend[0] + s * n[0] * width / 2, bend[1] + s * n[1] * width / 2)
                b = (end[0] + s * n[0] * end_width / 2, end[1] + s * n[1] * end_width / 2)
                # Where this edge of the leg meets the middle section's near (x = p) or far (x = p + width) edge.
                line_x = p if a[0] < bend[0] else p + width
                if abs(b[0] - a[0]) < 1e-12:
                    raise ValueError("A barrier leg runs parallel to its middle section")
                t = (line_x - a[0]) / (b[0] - a[0])
                corners[s] = ((line_x, a[1] + t * (b[1] - a[1])), b, line_x == p)
            outline.append((side, end, n, corners))
        shapes.append((p, width, end_width, outline))
    for k in range(int(count)):
        q = first + k * span
        place = lambda point: _rotated(point, q)
        for p, width, end_width, outline in shapes:
            near, far = {}, {}
            for side, end, n, corners in outline:
                for s, (corner, tip, is_near) in corners.items():
                    (near if is_near else far)[side] = (corner, tip)
                    m.segment(place(corner), place(tip), group=group)
                # The rounded end: counterclockwise from the edge at -n to the edge at +n passes outward.
                minus = (end[0] - n[0] * end_width / 2, end[1] - n[1] * end_width / 2)
                plus = (end[0] + n[0] * end_width / 2, end[1] + n[1] * end_width / 2)
                m.arc(place(minus), place(plus), 180, max_segment, group=group)
            m.segment(place(near[-1][0]), place(near[1][0]), group=group)
            m.segment(place(far[-1][0]), place(far[1][0]), group=group)
            m.label(*place((p + width / 2, 0)), material, mesh=mesh, group=group)


def entity_pattern(m, entities, copies=1, angle_step=0, first_angle=0, snap=1e-9, max_segment=1, max_segment_bands=None,
                   min_segment_length=0, group=0):
    """Contours given as data (from a drawing a paper supplies), repeated around the axis as a polar array.

    ``entities`` holds ["segment", x0, y0, x1, y1] and ["arc", x0, y0, x1, y1,
    angle] items, arcs counterclockwise and no wider than 180 degrees. Copy k is
    turned by ``first_angle + k * angle_step`` degrees. An endpoint within
    ``snap`` of a point this component already drew is that point, so copies
    join exactly and drawing noise cannot split a corner; a contour drawn twice
    (the shared edge of two copies) is drawn once; and a contour is split where
    another contour's endpoint lies on it (a T-junction, as a magnet corner
    resting on a pocket edge), as FEMM itself would split it. ``max_segment_bands``
    lists [r_min, r_max, max_segment]: arcs whose radius lies in a band take that
    angular discretization (finer near an air gap) instead of ``max_segment``, and
    no arc is cut into pieces shorter than ``min_segment_length`` (a 0.2 mm
    fillet at one degree would otherwise force micrometre elements).
    """
    if copies < 1 or not snap > 0:
        raise ValueError("A pattern needs at least one copy and a positive snap distance")
    cells, drawn, pieces = {}, set(), []

    def point(x, y):
        key = (round(x / snap), round(y / snap))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for px, py in cells.get((key[0] + dx, key[1] + dy), []):
                    if math.hypot(px - x, py - y) <= snap:
                        return (px, py)
        cells.setdefault(key, []).append((x, y))
        return (x, y)

    for k in range(copies):
        turn = first_angle + k * angle_step
        for item in entities:
            kind = item[0]
            if kind not in ("segment", "arc") or len(item) != (5 if kind == "segment" else 6):
                raise ValueError("Pattern entities are ['segment', x0, y0, x1, y1] or ['arc', x0, y0, x1, y1, angle]")
            a = point(*_rotated((item[1], item[2]), turn))
            b = point(*_rotated((item[3], item[4]), turn))
            if a == b:
                continue  # shorter than the snap distance: not a contour
            key = (kind, frozenset((a, b))) if kind == "segment" else (kind, a, b, round(item[5], 9))
            if key not in drawn:
                drawn.add(key)
                pieces.append((kind, a, b, item[5] if kind == "arc" else None))
    # Points on a contour's interior split it. A coarse grid keeps the search local.
    points = [p for bucket in cells.values() for p in bucket]
    size = max(snap * 1000, 1e-12)
    grid = {}
    for p in points:
        grid.setdefault((math.floor(p[0] / size), math.floor(p[1] / size)), []).append(p)

    def near(xmin, ymin, xmax, ymax):
        i0, i1, j0, j1 = math.floor(xmin / size), math.floor(xmax / size), math.floor(ymin / size), math.floor(ymax / size)
        if (i1 - i0 + 1) * (j1 - j0 + 1) > len(grid):
            # A box wider than the occupied cells (a small snap distance): scan the occupied cells instead.
            for (i, j), bucket in grid.items():
                if i0 <= i <= i1 and j0 <= j <= j1:
                    yield from bucket
            return
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                yield from grid.get((i, j), [])

    for kind, a, b, angle in pieces:
        if kind == "segment":
            dx, dy = b[0] - a[0], b[1] - a[1]
            length2 = dx * dx + dy * dy
            inner = []
            for p in near(min(a[0], b[0]) - snap, min(a[1], b[1]) - snap, max(a[0], b[0]) + snap, max(a[1], b[1]) + snap):
                t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2
                if p not in (a, b) and 0 < t < 1 and math.hypot(a[0] + t * dx - p[0], a[1] + t * dy - p[1]) <= snap:
                    inner.append((t, p))
            chain = [a] + [p for _, p in sorted(inner)] + [b]
            for u, v in zip(chain, chain[1:]):
                m.segment(u, v, group=group)
        else:
            info = arc_geometry(a, b, angle)
            (cx, cy), radius, start = info["center"], info["radius"], info["start_angle"]
            inner = []
            for p in near(cx - radius - snap, cy - radius - snap, cx + radius + snap, cy + radius + snap):
                if p in (a, b) or abs(math.hypot(p[0] - cx, p[1] - cy) - radius) > snap:
                    continue
                along = (math.degrees(math.atan2(p[1] - cy, p[0] - cx)) - start) % 360
                if 0 < along < angle:
                    inner.append((along, p))
            chain = [(0.0, a)] + sorted(inner) + [(angle, b)]
            discretization = next((band[2] for band in max_segment_bands or [] if band[0] <= radius <= band[1]), max_segment)
            discretization = max(discretization, math.degrees(min_segment_length / radius))
            for (u_angle, u), (v_angle, v) in zip(chain, chain[1:]):
                m.arc(u, v, v_angle - u_angle, discretization, group=group)


IMPLEMENTATIONS = {function.__name__: function for function in (
    segment, polyline, arc, circle, point, block_label, hole, material, library_material, circuit, conductor, boundary,
    fixed_potential_boundary, mixed_boundary, convection_boundary, iabc_open_boundary, axisymmetric_coil_section,
    bar_magnet, air_gap_element, kelvin_periodic_exterior, segment_boundary, fixed_point_potential, magnet_array,
    slotted_stator, trapezoidal_slot_stator, tooth_coil_winding, surface_magnet_rotor, salient_poles, salient_pole_winding,
    belt_winding, entity_pattern, sector_air_gap, arc_boundary, ring_sectors, flux_barrier_rotor)}


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
