"""Level-3 construction of official examples from their article text.

Each recipe receives only the parameters recorded for its case, every one with
the article passage it comes from, plus official library entries and FEMM's
documented defaults where the article relies on them. It never reads the
example's native model file. The result is a typed model compiled by
``reproduce.compile_model``, so article builds and rebuilds share one compiler.
"""
import json
import math
from pathlib import Path

from .reproduce import COLLECTIONS, PROPERTY_KEYS, property_values

DATA = Path(__file__).parent / "data"
EPSILON0 = 8.8541878188e-12  # FEMM's own eo constant (libfemm elec_results_doc.h)
MU0 = 4e-7 * math.pi
FEMM_DEFAULTS = {"precision": 1e-8, "min_angle": 30}


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
        return {"boundary": boundary, "hidden": hidden, "group": group, "conductor": conductor}

    def segment(self, a, b, *, mesh=None, **properties):
        self.segments.append({"n0": self.node(*a), "n1": self.node(*b), "mesh": mesh, **self._contour(**properties)})

    def arc(self, a, b, angle, max_segment, **properties):
        self.arcs.append({"n0": self.node(*a), "n1": self.node(*b), "angle": float(angle),
                          "max_segment": float(max_segment), **self._contour(**properties)})

    def circle(self, radius, center=(0, 0), max_segment=1, **properties):
        cx, cy = center
        self.arc((cx + radius, cy), (cx - radius, cy), 180, max_segment, **properties)
        self.arc((cx - radius, cy), (cx + radius, cy), 180, max_segment, **properties)

    def label(self, x, y, material, *, mesh=None, group=0, circuit=None, magdir=0, turns=1, external=False):
        label = {"x": float(x), "y": float(y), "material": material, "mesh": mesh, "group": group,
                 "external": external, "default": False}
        if self.physics == "magnetics":
            label.update(circuit=circuit, magdir=float(magdir), magdir_function=None, turns=turns)
        self.labels.append(label)

    def hole(self, x, y):
        self.holes.append({"x": float(x), "y": float(y)})

    def property(self, collection, name, curve=None, **values):
        keys = PROPERTY_KEYS[(self.physics, collection)]
        unknown = set(values) - set(keys)
        if unknown:
            raise ValueError("Unknown property keys: " + ", ".join(sorted(unknown)))
        record = {"name": name, "props": property_values(keys, values)}
        if curve:
            record["curve"] = [[float(a), float(b)] for a, b in curve]
        self.collections[collection].append(record)

    def library_material(self, name, library):
        entries = json.loads((DATA / "library_materials.json").read_text())["materials"]
        entry = next(e for e in entries if e["name"] == name and e["library"] == library)
        keys = PROPERTY_KEYS[(self.physics, "materials")]
        self.property("materials", name, curve=entry["curve"], **{k: v for k, v in entry["properties"].items() if k in keys})

    def result(self):
        return {"physics": self.physics, "problem": self.problem, **self.collections, "nodes": self.nodes,
                "segments": self.segments, "arcs": self.arcs, "labels": self.labels, "holes": self.holes}


def open_boundary(builder, radius, layers=7, center_y=0.0, dirichlet=True):
    """FEMM's improvised asymptotic boundary builder (femm_makeabc.cpp), reproduced exactly."""
    data = json.loads((DATA / "iabc_tables.json").read_text())
    axisymmetric = builder.problem["type"] == "axisymmetric"
    if builder.physics == "magnetics":
        table = ("uAx" if axisymmetric else "u2D") + ("0" if dirichlet else "1")
    else:
        table = ("uAx" if axisymmetric else "u2D") + ("1" if dirichlet else "0")
    prefix = "u" if builder.physics == "magnetics" else "e"
    edges = [radius * (1 + k / (10 * layers)) for k in range(layers + 1)]
    for k, r in enumerate(edges):
        outer = k == layers and dirichlet
        properties = {"boundary": ("A=0" if builder.physics == "magnetics" else "V=0")} if outer else {}
        builder.arc((0, center_y - r), (0, center_y + r), 180, 1, **properties)
        if not axisymmetric:
            builder.arc((0, center_y + r), (0, center_y - r), 180, 1, **properties)
    if axisymmetric:
        # FEMM draws one axis line from -1.1R to +1.1R and splits it at every
        # shell endpoint; interior axis nodes would split it further.
        if any(abs(n["x"]) < 1e-12 and abs(n["y"] - center_y) < radius for n in builder.nodes):
            raise ValueError("Open-boundary recipe expects no interior geometry on the symmetry axis")
        stations = sorted({center_y + s * r for r in edges for s in (-1, 1)})
        for a, b in zip(stations, stations[1:]):
            builder.segment((0, a), (0, b))
    for k in range(1, layers + 1):
        r = radius * (1 + (2 * k - 1) / (20 * layers))
        theta = math.pi * k / (2 * (layers + 1))
        name = prefix + str(k)
        mu = data["tables"][table][layers - 1][k - 1]
        if builder.physics == "magnetics":
            builder.property("materials", name, Mu_x=mu, Mu_y=mu)
        else:
            builder.property("materials", name, ex=mu, ey=mu)
        builder.label(r * math.cos(theta), center_y + r * math.sin(theta), name)
    if dirichlet:
        if builder.physics == "magnetics":
            builder.property("boundaries", "A=0")
        else:
            builder.property("boundaries", "V=0")


def _values(case):
    return {name: item["value"] for name, item in case["article"]["parameters"].items()}


def magnetics_tutorial_coil(case):
    p = _values(case)
    m = ModelBuilder("magnetics", units="inches", type="axisymmetric", frequency=0, depth=1, ac_solver=0)
    ri, ro, half = p["inner_diameter_in"] / 2, p["outer_diameter_in"] / 2, p["length_in"] / 2
    for a, b in [((ri, -half), (ro, -half)), ((ro, -half), (ro, half)), ((ro, half), (ri, half)), ((ri, half), (ri, -half))]:
        m.segment(a, b)
    m.library_material("Air", "matlib.dat")
    m.library_material(p["wire"], "matlib.dat")
    m.property("circuits", "Coil", TotalAmps_re=p["current_A"], CircuitType=1)
    m.label((ri + ro) / 2, 0, p["wire"], circuit="Coil", turns=p["turns"])
    m.label((ro + p["boundary_radius_in"]) / 2, 0, "Air")
    open_boundary(m, p["boundary_radius_in"], p["boundary_layers"])
    return m.result(), ["Air label sits midway between the coil and the open boundary; its position does not change the model.",
                        "Mesh sizes are left to the mesher, as the tutorial recommends for the air region."]


def electrostatics_square_capacitor(case):
    p = _values(case)
    m = ModelBuilder("electrostatics", units="centimeters", type="planar", depth=p["depth_cm"])
    m.property("materials", "air", ex=1, ey=1)
    m.property("conductors", "zero", Vc=0, ConductorType=1)
    m.property("conductors", "one", Vc=p["inner_voltage_V"], ConductorType=1)
    inner, outer = p["inner_half_width_cm"], p["outer_half_width_cm"]
    m.segment((0, inner), (inner, inner), conductor="one")
    m.segment((inner, inner), (inner, 0), conductor="one")
    m.segment((outer, 0), (outer, outer), conductor="zero")
    m.segment((outer, outer), (0, outer), conductor="zero")
    m.segment((0, inner), (0, outer))
    m.segment((inner, 0), (outer, 0))
    m.label((inner + outer) / 2, (inner + outer) / 2, "air")
    return m.result(), ["Quarter model with the two symmetry planes left as natural (zero normal flux) edges.",
                        "Default automatic mesh, as stated for the published charge."]


def electrostatics_two_spheres(case):
    p = _values(case)
    m = ModelBuilder("electrostatics", units="meters", type="axisymmetric", depth=1)
    bottom, top, extent = p["sphere_bottom_z_m"], p["sphere_top_z_m"], p["domain_radius_m"]
    m.property("materials", "Air", ex=1, ey=1)
    m.property("conductors", "+100V", Vc=p["sphere_voltage_V"], ConductorType=1)
    m.property("boundaries", "zero", Vs=0, BdryType=0)
    m.property("boundaries", "open_bc", c0=2 * EPSILON0 / extent, c1=0, BdryType=1)
    m.segment((0, 0), (0, bottom))
    m.segment((0, top), (0, extent))
    m.segment((0, 0), (extent, 0), boundary="zero")
    m.arc((0, bottom), (0, top), 180, p["max_segment_deg"], conductor="+100V")
    m.arc((extent, 0), (0, extent), 90, p["max_segment_deg"], boundary="open_bc")
    m.label(extent / 2, extent / 2, "Air", mesh=p["air_mesh_size_m"])
    return m.result(), ["The construction coordinates give a 25 m sphere radius; the article's prose says 25 m diameter. The explicit coordinates are followed.",
                        f"c0 = 2*eo/{extent:g} uses FEMM's eo = {EPSILON0} F/m."]


def tube_eddy_current(case):
    p = _values(case)
    m = ModelBuilder("magnetics", units="inches", type="planar", frequency=p["frequency_Hz"], depth=p["depth_in"], ac_solver=0)
    skin_depth_in = math.sqrt(2 / (2 * math.pi * p["frequency_Hz"] * p["relative_permeability"] * MU0 * p["conductivity_MS_per_m"] * 1e6)) / .0254
    m.property("materials", "Air", Mu_x=1, Mu_y=1)
    m.property("materials", "wire", Mu_x=1, Mu_y=1)
    m.property("materials", "steel", Mu_x=p["relative_permeability"], Mu_y=p["relative_permeability"], Sigma=p["conductivity_MS_per_m"])
    m.property("circuits", "Current", TotalAmps_re=p["wire_current_A"], CircuitType=1)
    m.property("circuits", "Zero Net Current", TotalAmps_re=0, CircuitType=0)
    m.property("boundaries", "A=0")
    m.circle(p["wire_radius_in"])
    m.circle(p["inner_radius_in"])
    m.circle(p["outer_radius_in"], boundary="A=0")
    m.label(0, 0, "wire", circuit="Current")
    m.label(0, (p["wire_radius_in"] + p["inner_radius_in"]) / 2, "Air")
    m.label(0, (p["inner_radius_in"] + p["outer_radius_in"]) / 2, "steel", circuit="Zero Net Current",
            mesh=skin_depth_in / p["elements_per_skin_depth"])
    return m.result(), [f"Pipe elements are one {p['elements_per_skin_depth']:g}th of the {skin_depth_in:.4g} in skin depth; other regions use the automatic mesh.",
                        "The wire is a nonconducting source region carrying a uniform 100 A; the article gives no wire conductivity, and the analytic pipe solution depends only on its total current.",
                        "Out-of-plane depth is one meter; current density is independent of depth."]


def torque_benchmark(case):
    p = _values(case)
    m = ModelBuilder("magnetics", units="centimeters", type="planar", frequency=0, depth=p["magnet_depth_cm"], ac_solver=0)
    half_l, half_w = p["magnet_length_cm"] / 2, p["magnet_width_cm"] / 2
    rotor, gap, domain, offset = p["rotor_radius_cm"], p["stator_radius_cm"], p["domain_radius_cm"], p["exterior_center_x_cm"]
    m.property("materials", "Air", Mu_x=1, Mu_y=1)
    m.property("materials", "magnet", Mu_x=1, Mu_y=1, H_c=p["magnet_coercivity_A_per_m"])
    m.property("materials", "Ext", Mu_x=1, Mu_y=1, H_c=2 * p["applied_field_T"] / MU0)
    m.property("boundaries", "AGE", BdryType=6)
    m.property("boundaries", "pbc1", BdryType=4)
    m.property("boundaries", "pbc2", BdryType=4)
    corners = [(-half_l, -half_w), (half_l, -half_w), (half_l, half_w), (-half_l, half_w)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        m.segment(a, b)
    m.circle(rotor, max_segment=p["gap_max_segment_deg"], boundary="AGE")
    m.circle(gap, max_segment=p["gap_max_segment_deg"], boundary="AGE")
    m.arc((domain, 0), (-domain, 0), 180, p["boundary_max_segment_deg"], boundary="pbc1")
    m.arc((-domain, 0), (domain, 0), 180, p["boundary_max_segment_deg"], boundary="pbc2")
    m.arc((offset + domain, 0), (offset - domain, 0), 180, p["boundary_max_segment_deg"], boundary="pbc1")
    m.arc((offset - domain, 0), (offset + domain, 0), 180, p["boundary_max_segment_deg"], boundary="pbc2")
    mesh = p["mesh_size_cm"]
    m.label(0, 0, "magnet", mesh=mesh, magdir=p["magnetization_direction_deg"])
    m.label(0, (half_w + rotor) / 2, "Air", mesh=mesh)
    m.hole(0, (rotor + gap) / 2)
    m.label(0, (gap + domain) / 2, "Air", mesh=mesh)
    m.label(offset, 0, "Ext", mesh=p["exterior_mesh_size_cm"], magdir=p["exterior_magnetization_deg"])
    return m.result(), ["Rotor, gap and domain radii are not stated in the article; they are chosen here and the Kelvin construction is exact for any radius.",
                        "Both halves of the domain circle are linked to the same halves of the exterior circle, drawn in the same direction.",
                        "The applied-field direction is measured in a magnet-free step rather than assumed; torque is then checked against m x B."]


def heat_flow_chimney(case):
    p = _values(case)
    m = ModelBuilder("heatflow", units="meters", type="planar", depth=p["depth_m"], prev_soln="", dt=0)
    m.library_material(p["material"], "heatlib.dat")
    m.property("boundaries", "Inner Boundary", BdryType=2, Tinf=p["inner_air_temperature_K"], h=p["inner_heat_transfer_W_per_m2K"])
    m.property("boundaries", "Outer Boundary", BdryType=2, Tinf=p["outer_air_temperature_K"], h=p["outer_heat_transfer_W_per_m2K"])
    inner, outer = p["inner_half_width_m"], p["outer_half_width_m"]
    m.segment((0, inner), (0, outer))
    m.segment((0, outer), (outer, outer), boundary="Outer Boundary")
    m.segment((outer, outer), (outer, 0), boundary="Outer Boundary")
    m.segment((outer, 0), (inner, 0))
    m.segment((inner, 0), (inner, inner), boundary="Inner Boundary")
    m.segment((inner, inner), (0, inner), boundary="Inner Boundary")
    m.label((inner + outer) / 2, (inner + outer) / 2, p["material"], mesh=p["mesh_size_m"])
    return m.result(), ["Quarter model; the two cut planes are insulated by symmetry, FEMM's default for an unassigned edge."]


RECIPES = {"magnetics_tutorial_coil": magnetics_tutorial_coil, "electrostatics_square_capacitor": electrostatics_square_capacitor,
           "electrostatics_two_spheres": electrostatics_two_spheres, "tube_eddy_current": tube_eddy_current,
           "torque_benchmark": torque_benchmark, "heat_flow_chimney": heat_flow_chimney}


def build(case):
    article = case.get("article") or {}
    if article.get("status") == "not_attempted":
        return {"status": "not_attempted", "reason": article["reason"]}
    if article.get("recipe") not in RECIPES:
        raise ValueError("No article recipe registered for case " + case["id"])
    model, assumptions = RECIPES[article["recipe"]](case)
    return {"status": "built", "model": model, "parameters": article["parameters"], "assumptions": assumptions}
