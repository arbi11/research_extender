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
import json
import math
from pathlib import Path

from .reproduce import COLLECTIONS, PROPERTY_KEYS, property_values

DATA = Path(__file__).parent / "data"
EPSILON0 = 8.8541878188e-12  # FEMM's own eo constant (libfemm elec_results_doc.h)
MU0 = 4e-7 * math.pi
FEMM_DEFAULTS = {"precision": 1e-8, "min_angle": 30}
CONSTANTS = {"pi": math.pi, "mu0": MU0, "eps0": EPSILON0}
FUNCTIONS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "atan2": math.atan2,
             "radians": math.radians, "degrees": math.degrees, "min": min, "max": max, "abs": abs, "log": math.log}
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

    def node(self, x, y):
        key = (round(x, 12), round(y, 12))
        if key not in self._index:
            self._index[key] = len(self.nodes)
            self.nodes.append({"x": float(x), "y": float(y), "point": None, "group": 0,
                               "conductor": None})
        return self._index[key]

    def _contour(self, boundary=None, hidden=False, group=0, conductor=None):
        if conductor and self.physics == "magnetics":
            raise ValueError("Magnetics contours carry boundaries, not conductors")
        return {"boundary": boundary, "hidden": hidden, "group": group, "conductor": conductor}

    def segment(self, a, b, *, mesh=None, **properties):
        self.segments.append({"n0": self.node(*a), "n1": self.node(*b), "mesh": mesh, **self._contour(**properties)})

    def arc(self, a, b, angle, max_segment, **properties):
        self.arcs.append({"n0": self.node(*a), "n1": self.node(*b), "angle": float(angle),
                          "max_segment": float(max_segment), **self._contour(**properties)})

    def label(self, x, y, material, *, mesh=None, group=0, circuit=None, magdir=0, turns=1, external=False):
        label = {"x": float(x), "y": float(y), "material": material, "mesh": mesh, "group": group,
                 "external": external, "default": False}
        if self.physics == "magnetics":
            label.update(circuit=circuit, magdir=float(magdir), magdir_function=None, turns=turns)
        elif circuit or turns != 1 or magdir:
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
        if curve:
            record["curve"] = [[float(a), float(b)] for a, b in curve]
        self.collections[collection].append(record)

    def result(self):
        return {"physics": self.physics, "problem": self.problem, **self.collections, "nodes": self.nodes,
                "segments": self.segments, "arcs": self.arcs, "labels": self.labels, "holes": self.holes}


def _contour_options(options):
    return {key: options[key] for key in ("boundary", "conductor", "hidden", "group") if options.get(key) not in (None, False)}


# --- Geometry primitives -----------------------------------------------------

def segment(m, start, end, mesh=None, **options):
    m.segment(start, end, mesh=mesh, **_contour_options(options))


def polyline(m, points, closed=True, edges=None, **options):
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


def block_label(m, at, material, mesh=None, circuit=None, turns=1, magdir=0, group=0, external=False):
    m.label(at[0], at[1], material, mesh=mesh, circuit=circuit, turns=turns, magdir=magdir, group=group, external=external)


def hole(m, at):
    m.hole(*at)


# --- Properties --------------------------------------------------------------

def material(m, name, values=None, curve=None):
    m.property("materials", name, curve=curve, **(values or {}))


def library_material(m, name, library):
    entries = json.loads((DATA / "library_materials.json").read_text())["materials"]
    entry = next((e for e in entries if e["name"] == name and e["library"] == library), None)
    if entry is None:
        raise ValueError(f"{name} is not recorded from {library}; add it to library_materials.json with provenance")
    keys = PROPERTY_KEYS[(m.physics, "materials")]
    m.property("materials", name, curve=entry["curve"], **{k: v for k, v in entry["properties"].items() if k in keys})


def circuit(m, name, current, series=True):
    real, imaginary = (current if isinstance(current, list) else [current, 0])
    m.property("circuits", name, TotalAmps_re=real, TotalAmps_im=imaginary, CircuitType=1 if series else 0)


def conductor(m, name, value=0, prescribed="value", charge=0):
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


IMPLEMENTATIONS = {function.__name__: function for function in (
    segment, polyline, arc, circle, block_label, hole, material, library_material, circuit, conductor, boundary,
    fixed_potential_boundary, mixed_boundary, convection_boundary, iabc_open_boundary, axisymmetric_coil_section,
    bar_magnet, air_gap_element, kelvin_periodic_exterior)}


def catalog():
    return {item["id"]: item for item in json.loads((DATA / "components.json").read_text())["components"]}


def assemble(assembly, parameters):
    """Build a typed model from an assembly; return it with the components used."""
    records = catalog()
    names = dict(parameters)
    for name, value in (assembly.get("let") or {}).items():
        if name in names or name in CONSTANTS:
            raise ValueError("Derived value shadows a parameter or constant: " + name)
        names[name] = resolve(value, names)
    problem = dict(resolve(assembly["problem"], names))
    physics = problem.pop("physics")
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
            low, high = declared[name].get("minimum"), declared[name].get("maximum")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and (
                    (low is not None and value < low) or (high is not None and value > high)):
                raise ValueError(f"{identifier}.{name}={value} is outside [{low}, {high}]")
        IMPLEMENTATIONS[identifier](m, **arguments)
        if identifier not in used:
            used.append(identifier)
    return m.result(), used
