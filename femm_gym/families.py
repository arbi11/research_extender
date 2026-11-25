"""Task families for Levels 1 and 2 of the ladder.

A family samples a device, builds its reference model from femm_brain's tested
components, and writes the task the player sees. Every input the reference uses is
stated in the task text, so Levels 1 and 2 have no gaps (gaps start at Level 3).

- Level 1 asks for geometry and materials. The scorer checks which material lies
  where, point by point.
- Level 2 asks for a small device whose one result (an inductance, a flux linkage or
  a force) the scorer measures itself from the saved model.

``build(p, level, variant)`` returns a femm_brain typed model:
- ``level=None`` leaves every mesh to FEMM (what a player gets by default);
- ``level=k`` sets explicit mesh sizes, halved at each level (the reference solves);
- ``variant=True`` draws the same device differently (coarser arcs, labels elsewhere,
  a larger open boundary), for the control that must still pass.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Optional

from femm_brain import components as fc
from femm_brain.components import ModelBuilder

MU0 = 4e-7 * math.pi
POINTS = 4000  # Level 1 sample points per task
SCORE_SECONDS = 180  # the scorer's limit for one solve, stated in every task


@dataclass(frozen=True)
class Family:
    name: str
    level: int
    title: str
    measure: str                 # material_map | inductance | flux_linkage | axial_force
    unit: str
    tolerance: float             # Level 1: least intersection-over-union per material; Level 2: relative error
    sample: Callable[[random.Random], dict]
    build: Callable[..., dict]
    text: Callable[[dict], str]
    wrong: Callable[[dict], dict]
    region: Optional[Callable[[dict, random.Random, int], list]] = None
    current: Optional[Callable[[dict], float]] = None   # Level 2: the current the scorer sets on circuit "coil"

    @property
    def circuit(self):
        return "coil" if self.level == 2 else None


# --- helpers -----------------------------------------------------------------

def _r(value, step=0.1):
    """Round to the resolution the task states, so the text and the reference use one number."""
    return round(round(value / step) * step, 10)


def _f(value):
    """A number as the task text shows it."""
    value = round(float(value), 6)
    return str(int(value)) if value.is_integer() else f"{value:.6g}"


def _rot(point, degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return (point[0] * c - point[1] * s, point[0] * s + point[1] * c)


def _materials(m, table):
    for name, values in table.items():
        fc.material(m, name, values)


def _material_lines(table):
    lines = []
    for name, v in table.items():
        parts = [f"relative permeability {_f(v.get('Mu_x', 1))}", f"conductivity {_f(v.get('Sigma', 0))} MS/m"]
        if v.get("H_c"):
            parts.append(f"coercivity {_f(v['H_c'])} A/m")
        lines.append(f"- `{name}`: " + ", ".join(parts))
    return "\n".join(lines)


def _rect(m, lower, upper, mesh=None):
    (x0, y0), (x1, y1) = lower, upper
    fc.polyline(m, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], edges=[{"mesh": mesh}] * 4)


def _mesh(level, size, radius):
    """Device and air mesh sizes at a reference level; None leaves both to FEMM."""
    if level is None:
        return None, None
    return size / (4 * 2 ** level), radius / (25 * 2 ** level)


def _air_point(radius, axisymmetric):
    """A label point in the air just inside an open boundary of this radius."""
    angle = math.radians(60 if axisymmetric else 45)
    return (0.9 * radius * math.cos(angle), 0.9 * radius * math.sin(angle))


def _disk(radius):
    def region(p, rng, n):
        out = []
        for _ in range(n):
            r, t = radius(p) * math.sqrt(rng.random()), 2 * math.pi * rng.random()
            out.append((r * math.cos(t), r * math.sin(t)))
        return out
    return region


def _header(title, problem, save="`submission/model.fem`"):
    return f"# {title}\n\nBuild this in FEMM 5 and save it as {save}: {problem}.\n"


LIMIT = f"A solve that runs longer than {SCORE_SECONDS // 60} minutes scores 0.\n"

L1_FOOTER = """
## How it is scored

The scorer opens your saved model, solves it with no currents, and checks which material lies
where at a few thousand points inside {domain}. Materials are recognised by their relative
permeability and conductivity, so give each one exactly the values above; names do not matter.
""" + LIMIT

L2_FOOTER = """
## What to save

- `submission/model.fem`: your model. Name the winding's circuit `coil` (series circuit).
- `submission/answer.json` (optional, not scored): `{{"value": <your result>, "unit": "{unit}"}}`.

## How it is scored

The scorer opens your saved model, sets circuit `coil` to {current} A and every other circuit
to 0 A, solves it with the mesh you saved, and measures {quantity}. Your own number is never
used, so the result depends on the model, including how fine its mesh is.
""" + LIMIT


# --- Level 1: stator lamination ---------------------------------------------

STATOR_MATERIALS = {"Steel": {"Mu_x": 2500, "Mu_y": 2500}, "Copper": {"Sigma": 58}, "Air": {}}


def _stator_sample(rng):
    for _ in range(1000):
        p = {"slots": rng.choice([12, 18, 24, 36, 48]), "bore_radius": _r(rng.uniform(25, 60))}
        pitch = 2 * math.pi * p["bore_radius"] / p["slots"]
        p["opening_width"] = _r(max(1.0, rng.uniform(0.15, 0.3) * pitch))
        p["opening_height"] = _r(rng.uniform(0.5, 1.5))
        p["slot_width_top"] = _r(rng.uniform(max(p["opening_width"] + 0.5, 0.35 * pitch), 0.55 * pitch))
        p["slot_width_bottom"] = _r(p["slot_width_top"] * rng.uniform(1.0, 1.6))
        p["slot_height"] = _r(rng.uniform(0.25, 0.6) * p["bore_radius"])
        top = p["bore_radius"] + p["opening_height"]
        bottom = top + p["slot_height"]
        corner = math.hypot(bottom, p["slot_width_bottom"] / 2)
        p["outer_radius"] = _r(corner + max(3.0, rng.uniform(0.15, 0.35) * p["bore_radius"]))
        if (2 * math.pi * top / p["slots"] - p["slot_width_top"] < 1.5
                or 2 * math.pi * bottom / p["slots"] - p["slot_width_bottom"] < 1.5):
            continue
        try:
            _stator_build(p)
        except ValueError:
            continue
        return p
    raise RuntimeError("No valid stator in 1000 draws")


def _stator_build(p, level=None, variant=False):
    m = ModelBuilder("magnetics", units="millimeters", type="planar", frequency=0, depth=50)
    _materials(m, STATOR_MATERIALS)
    fc.fixed_potential_boundary(m, "A=0")
    arc = 5 if variant else 1
    fc.trapezoidal_slot_stator(m, slot_count=p["slots"], bore_radius=p["bore_radius"], opening_width=p["opening_width"],
                               opening_height=p["opening_height"], slot_width_top=p["slot_width_top"],
                               slot_width_bottom=p["slot_width_bottom"], slot_height=p["slot_height"],
                               bore_max_segment=arc, slot_bottom_max_segment=arc, iron_material="Steel")
    fc.circle(m, p["outer_radius"], max_segment=arc, boundary="A=0")
    depth = p["bore_radius"] + p["opening_height"] + p["slot_height"] * (0.3 if variant else 0.5)
    for k in range(p["slots"]):
        fc.block_label(m, _rot((depth, 0), (k + 0.5) * 360 / p["slots"]), "Copper")
    fc.block_label(m, (0, p["bore_radius"] / 3) if variant else (p["bore_radius"] / 2, 0), "Air")
    return m.result()


def _stator_text(p):
    pitch = 360 / p["slots"]
    top = p["bore_radius"] + p["opening_height"]
    bottom = top + p["slot_height"]
    return (_header(f"Stator lamination with {p['slots']} trapezoidal slots",
                    "a magnetics problem at 0 Hz, planar, lengths in millimetres, depth 50 mm") + f"""
## Geometry

- The lamination is a ring of `Steel` centred on the origin, from the bore circle (radius
  {_f(p['bore_radius'])} mm) to the outer circle (radius {_f(p['outer_radius'])} mm).
- It has {p['slots']} identical slots. Slot k (k = 0, 1, ..., {p['slots'] - 1}) is symmetric about the
  ray at (k + 1/2) x {_f(pitch)} degrees from the +x axis, so a tooth is centred on the +x axis.
- In each slot's own frame (x outward along its centre line from the origin, y across it):
  - the opening is the strip |y| <= {_f(p['opening_width'] / 2)} mm from the bore circle out to
    x = {_f(top)} mm. It is air, part of the bore;
  - the slot body is the trapezoid with corners (x, y) = ({_f(top)}, +/-{_f(p['slot_width_top'] / 2)}) mm and
    ({_f(bottom)}, +/-{_f(p['slot_width_bottom'] / 2)}) mm: straight sides and a flat bottom. It is `Copper`.
- Inside the bore circle is `Air`.

## Materials (linear, isotropic, not laminated)

{_material_lines(STATOR_MATERIALS)}

## Boundary

A = 0 on the outer circle. Nothing lies outside it.
""" + L1_FOOTER.format(domain="the outer circle"))


def _stator_wrong(p):
    return dict(p, slot_height=_r(p["slot_height"] * 0.85))


# --- Level 1: surface-magnet rotor --------------------------------------------

SPM_MATERIALS = {"Steel": {"Mu_x": 1000, "Mu_y": 1000}, "Magnet": {"Mu_x": 1.05, "Mu_y": 1.05, "H_c": 900000, "Sigma": 0.667},
                 "Air": {}}


def _spm_sample(rng):
    poles = rng.choice([4, 6, 8, 10, 12])
    p = {"poles": poles, "core_radius": _r(rng.uniform(20, 50)), "magnet_thickness": _r(rng.uniform(2, 6)),
         "magnet_arc": _r(rng.uniform(0.6, 0.9) * 360 / poles, 0.5), "air_gap": _r(rng.uniform(1, 3))}
    p["outer_radius"] = _r(p["core_radius"] + p["magnet_thickness"] + p["air_gap"])
    return p


def _spm_build(p, level=None, variant=False):
    m = ModelBuilder("magnetics", units="millimeters", type="planar", frequency=0, depth=50)
    _materials(m, SPM_MATERIALS)
    fc.fixed_potential_boundary(m, "A=0")
    arc = 5 if variant else 1
    fc.surface_magnet_rotor(m, pole_count=p["poles"], magnet_material="Magnet", thickness=p["magnet_thickness"],
                            magnet_shape="arc", inner_radius=p["core_radius"], arc_angle=p["magnet_arc"],
                            air_gap_side="outer", seat_arcs=True, arc_max_segment=arc)
    fc.circle(m, p["outer_radius"], max_segment=arc, boundary="A=0")
    fc.block_label(m, (0, p["core_radius"] / 3) if variant else (p["core_radius"] / 2, 0), "Steel")
    r = p["core_radius"] + (p["magnet_thickness"] + p["air_gap"] / 2 if variant else p["magnet_thickness"] / 2)
    fc.block_label(m, _rot((r, 0), 180 / p["poles"]), "Air")
    return m.result()


def _spm_text(p):
    outer = p["core_radius"] + p["magnet_thickness"]
    return (_header(f"Surface-magnet rotor with {p['poles']} arc magnets",
                    "a magnetics problem at 0 Hz, planar, lengths in millimetres, depth 50 mm") + f"""
## Geometry

- The rotor core is a solid `Steel` disc of radius {_f(p['core_radius'])} mm centred on the origin.
- {p['poles']} arc magnets of `Magnet` sit on it, each from r = {_f(p['core_radius'])} mm to r = {_f(outer)} mm and
  spanning {_f(p['magnet_arc'])} degrees. Magnet k (k = 0, 1, ..., {p['poles'] - 1}) is centred on the ray at
  k x {_f(360 / p['poles'])} degrees from the +x axis.
- Everything else inside the circle r = {_f(p['outer_radius'])} mm is `Air`: the gaps between the magnets and
  the ring above them.

## Materials (linear, isotropic, not laminated)

{_material_lines(SPM_MATERIALS)}

Magnetize the magnets radially, alternating outward and inward; the direction is not scored here.

## Boundary

A = 0 on the circle r = {_f(p['outer_radius'])} mm. Nothing lies outside it.
""" + L1_FOOTER.format(domain=f"the circle r = {_f(p['outer_radius'])} mm"))


def _spm_wrong(p):
    return dict(p, magnet_thickness=_r(p["magnet_thickness"] * 0.8))


# --- Level 1: pot core ---------------------------------------------------------

POT_MATERIALS = {"Ferrite": {"Mu_x": 2000, "Mu_y": 2000}, "Copper": {"Sigma": 58}, "Air": {}}


def _pot_sample(rng):
    p = {"post_radius": _r(rng.uniform(3, 10)), "window_width": _r(rng.uniform(3, 12)), "wall": _r(rng.uniform(2, 6)),
         "window_half_height": _r(rng.uniform(3, 12)), "yoke": _r(rng.uniform(2, 6)), "clearance": _r(rng.uniform(0.3, 1.0))}
    p["domain_radius"] = _r(2 * (p["post_radius"] + p["window_width"] + p["wall"]))
    p["domain_half_height"] = _r(2 * (p["window_half_height"] + p["yoke"]))
    return p


def _pot_dims(p):
    a = p["post_radius"]
    b = a + p["window_width"]
    w = p["window_half_height"]
    return a, b, b + p["wall"], w, w + p["yoke"], p["clearance"]


def _pot_build(p, level=None, variant=False):
    m = ModelBuilder("magnetics", units="millimeters", type="axisymmetric", frequency=0, depth=1)
    _materials(m, POT_MATERIALS)
    fc.fixed_potential_boundary(m, "A=0")
    a, b, c, w, h, g = _pot_dims(p)
    R, Z = p["domain_radius"], p["domain_half_height"]
    fc.polyline(m, [(0, -Z), (R, -Z), (R, Z), (0, Z)], closed=False, boundary="A=0")
    for z0, z1 in ((Z, h), (h, -h), (-h, -Z)):
        fc.segment(m, (0, z0), (0, z1))
    fc.polyline(m, [(0, -h), (c, -h), (c, h), (0, h)], closed=False)
    fc.polyline(m, [(a, -w), (b, -w), (b, w), (a, w)])
    fc.polyline(m, [(a + g, g - w), (b - g, g - w), (b - g, w - g), (a + g, w - g)])
    fc.block_label(m, (a / 2, h - p["yoke"] / 2) if variant else (a / 2, 0), "Ferrite")
    fc.block_label(m, (b - g / 2, 0) if variant else (a + g / 2, 0), "Air")
    fc.block_label(m, ((a + b) / 2, (w - g) / 2) if variant else ((a + b) / 2, 0), "Copper")
    fc.block_label(m, (R / 2, (h + Z) / 2) if variant else ((c + R) / 2, 0), "Air")
    return m.result()


def _pot_region(p, rng, n):
    R, Z = p["domain_radius"], p["domain_half_height"]
    return [(R * rng.random(), Z * (2 * rng.random() - 1)) for _ in range(n)]


def _pot_text(p):
    a, b, c, w, h, g = _pot_dims(p)
    R, Z = p["domain_radius"], p["domain_half_height"]
    return (_header("Pot core with its winding",
                    "a magnetics problem at 0 Hz, axisymmetric about the z axis (r >= 0), lengths in millimetres") + f"""
## Geometry (r and z in mm)

- The core is `Ferrite`: the rectangle 0 <= r <= {_f(c)}, -{_f(h)} <= z <= {_f(h)}, minus its winding window
  {_f(a)} <= r <= {_f(b)}, -{_f(w)} <= z <= {_f(w)}.
- Inside the window the winding is `Copper`: {_f(a + g)} <= r <= {_f(b - g)}, -{_f(w - g)} <= z <= {_f(w - g)}.
  The rest of the window is `Air`.
- `Air` fills the rest of the region 0 <= r <= {_f(R)}, -{_f(Z)} <= z <= {_f(Z)}.

## Materials (linear, isotropic, not laminated)

{_material_lines(POT_MATERIALS)}

## Boundary

A = 0 on the edges r = {_f(R)} and z = +/-{_f(Z)}. The axis r = 0 needs no condition.
""" + L1_FOOTER.format(domain="that region"))


def _pot_wrong(p):
    return dict(p, window_half_height=_r(p["window_half_height"] * 0.8))


# --- Level 2: air-core coil inductance -----------------------------------------

def _coil_sample(rng):
    return {"inner_radius": _r(rng.uniform(5, 40)), "radial_build": _r(rng.uniform(2, 30)),
            "length": _r(rng.uniform(5, 60)), "turns": rng.randrange(10, 1001, 10)}


def _coil_build(p, level=None, variant=False):
    m = ModelBuilder("magnetics", units="millimeters", type="axisymmetric", frequency=0, depth=1)
    _materials(m, {"Copper": {"Sigma": 58}, "Air": {}})
    fc.circuit(m, "coil", 1.0)
    r0, r1, half = p["inner_radius"], p["inner_radius"] + p["radial_build"], p["length"] / 2
    radius = (3.0 if variant else 2.5) * math.hypot(r1, half)
    device, air = _mesh(level, min(p["radial_build"], p["length"]), radius)
    _rect(m, (r0, -half), (r1, half), device)
    fc.block_label(m, ((r0 + r1) / 2, 0), "Copper", mesh=device, circuit="coil", turns=p["turns"])
    fc.iabc_open_boundary(m, radius)
    fc.block_label(m, _air_point(radius, True), "Air", mesh=air)
    return m.result()


def _coil_text(p):
    r1, half = p["inner_radius"] + p["radial_build"], p["length"] / 2
    return (_header("Inductance of an air-core coil",
                    "a magnetics problem at 0 Hz, axisymmetric about the z axis (r >= 0), lengths in millimetres") + f"""
## Device (r and z in mm)

- One winding of `Copper` with a rectangular cross-section {_f(p['inner_radius'])} <= r <= {_f(r1)},
  -{_f(half)} <= z <= {_f(half)}, and {p['turns']} turns.
- Everything around it is `Air`. Model the surroundings as open (unbounded) space, by a method of your choice.

## Materials (linear)

{_material_lines({"Copper": {"Sigma": 58}, "Air": {}})}

## Scored

The coil's self-inductance, in henries.
""" + L2_FOOTER.format(unit="H", current="1", quantity="the inductance from the circuit's flux linkage"))


def _coil_wrong(p):
    return dict(p, turns=int(round(p["turns"] * 1.1)))


# --- Level 2: gapped C-core inductance ------------------------------------------

def _core_sample(rng):
    for _ in range(1000):
        W, H = _r(rng.uniform(40, 120), 1), _r(rng.uniform(40, 120), 1)
        t = _r(rng.uniform(0.15, 0.3) * min(W, H), 0.5)
        p = {"width": W, "height": H, "limb": t, "gap": _r(rng.uniform(0.2, 3.0)), "mu": rng.choice([500, 1000, 2000, 5000]),
             "depth": _r(rng.uniform(10, 60), 1), "turns": rng.randrange(20, 401, 5),
             "coil_width": _r(rng.uniform(0.2, 0.4) * (W - 2 * t)), "coil_height": _r(rng.uniform(0.4, 0.8) * (H - 2 * t)),
             "clearance": 1.0}
        if p["coil_width"] >= 2 and p["coil_height"] >= 4:
            return p
    raise RuntimeError("No valid core in 1000 draws")


def _core_dims(p):
    W2, H2, t, g2, cl, cw = p["width"] / 2, p["height"] / 2, p["limb"], p["gap"] / 2, p["clearance"], p["coil_width"]
    inside = (-W2 + t + cl, -W2 + t + cl + cw)
    outside = (-W2 - cl - cw, -W2 - cl)
    return W2, H2, t, g2, inside, outside


def _core_build(p, level=None, variant=False):
    m = ModelBuilder("magnetics", units="millimeters", type="planar", frequency=0, depth=p["depth"])
    _materials(m, {"Steel": {"Mu_x": p["mu"], "Mu_y": p["mu"]}, "Copper": {"Sigma": 58}, "Air": {}})
    fc.circuit(m, "coil", 1.0)
    W2, H2, t, g2, inside, outside = _core_dims(p)
    radius = (1.8 if variant else 1.5) * math.hypot(W2 + p["clearance"] + p["coil_width"], H2) + 5
    device, air = _mesh(level, min(t, p["coil_width"], p["coil_height"]), radius)
    gap_mesh = None if level is None else p["gap"] / (4 * 2 ** level)
    outline = [(W2, g2), (W2, H2), (-W2, H2), (-W2, -H2), (W2, -H2), (W2, -g2), (W2 - t, -g2), (W2 - t, -H2 + t),
               (-W2 + t, -H2 + t), (-W2 + t, H2 - t), (W2 - t, H2 - t), (W2 - t, g2)]
    edges = [{"mesh": device} for _ in outline]
    edges[5] = edges[11] = {"mesh": gap_mesh}  # the two gap faces
    fc.polyline(m, outline, edges=edges)
    half = p["coil_height"] / 2
    for (x0, x1), turns in ((inside, p["turns"]), (outside, -p["turns"])):
        _rect(m, (x0, -half), (x1, half), device)
        fc.block_label(m, ((x0 + x1) / 2, 0), "Copper", mesh=device, circuit="coil", turns=turns)
    fc.block_label(m, (0, H2 - t / 2) if variant else (-W2 + t / 2, 0), "Steel", mesh=device)
    fc.iabc_open_boundary(m, radius)
    fc.block_label(m, _air_point(radius, False) if variant else (0, 0), "Air", mesh=air)
    return m.result()


def _core_text(p):
    W2, H2, t, g2, inside, outside = _core_dims(p)
    return (_header("Inductance of a gapped C-core inductor",
                    f"a magnetics problem at 0 Hz, planar, lengths in millimetres, depth {_f(p['depth'])} mm") + f"""
## Device (x and y in mm)

- The core is `Steel`: the rectangle |x| <= {_f(W2)}, |y| <= {_f(H2)}, minus its window |x| <= {_f(W2 - t)},
  |y| <= {_f(H2 - t)}, and minus an air gap that cuts the right limb: {_f(W2 - t)} <= x <= {_f(W2)},
  |y| <= {_f(g2)}.
- A coil of {p['turns']} turns is wound on the left limb. Its two sides are `Copper` rectangles with
  |y| <= {_f(p['coil_height'] / 2)}: one inside the window, {_f(inside[0])} <= x <= {_f(inside[1])}, and one
  outside the core, {_f(outside[0])} <= x <= {_f(outside[1])}. In FEMM terms the inside side carries +{p['turns']}
  turns of circuit `coil` and the outside side -{p['turns']}.
- Everything else is `Air`. Model the surroundings as open (unbounded) space, by a method of your choice.

## Materials (linear, isotropic, not laminated)

{_material_lines({"Steel": {"Mu_x": p["mu"]}, "Copper": {"Sigma": 58}, "Air": {}})}

## Scored

The coil's self-inductance, in henries.
""" + L2_FOOTER.format(unit="H", current="1", quantity="the inductance from the circuit's flux linkage"))


def _core_wrong(p):
    return dict(p, gap=_r(p["gap"] * 1.5))


# --- Level 2: ring magnet and coil (force, linkage) ------------------------------

def _magnet_sample(with_current):
    def sample(rng):
        rmi = _r(rng.uniform(2, 10))
        br = _r(rng.uniform(1.0, 1.4), 0.01)
        p = {"magnet_inner_radius": rmi, "magnet_outer_radius": _r(rmi + rng.uniform(3, 15)),
             "magnet_height": _r(rng.uniform(2, 15)), "remanence": br, "coercivity": round(br / (MU0 * 1.05))}
        p["coil_inner_radius"] = _r(rng.uniform(max(1.0, rmi - 2), p["magnet_outer_radius"]))
        p["coil_outer_radius"] = _r(p["coil_inner_radius"] + rng.uniform(2, 15))
        p.update(coil_gap=_r(rng.uniform(1, 10)), coil_height=_r(rng.uniform(3, 20)), turns=rng.randrange(50, 501, 10),
                 current=_r(rng.uniform(0.5, 5.0)) if with_current else 0.0)
        return p
    return sample


def _magnet_dims(p):
    hm = p["magnet_height"] / 2
    z0 = hm + p["coil_gap"]
    return hm, z0, z0 + p["coil_height"]


def _magnet_build(p, level=None, variant=False):
    m = ModelBuilder("magnetics", units="millimeters", type="axisymmetric", frequency=0, depth=1)
    _materials(m, {"NdFeB": {"Mu_x": 1.05, "Mu_y": 1.05, "H_c": p["coercivity"]}, "Copper": {"Sigma": 58}, "Air": {}})
    fc.circuit(m, "coil", p["current"])
    hm, z0, z1 = _magnet_dims(p)
    rmi, rmo, rci, rco = (p["magnet_inner_radius"], p["magnet_outer_radius"], p["coil_inner_radius"], p["coil_outer_radius"])
    radius = (3.0 if variant else 2.5) * math.hypot(max(rmo, rco), max(hm, z1))
    device, air = _mesh(level, min(rmo - rmi, 2 * hm, rco - rci, p["coil_height"]), radius)
    _rect(m, (rmi, -hm), (rmo, hm), device)
    fc.block_label(m, ((rmi + rmo) / 2, 0), "NdFeB", mesh=device, magdir=90)
    _rect(m, (rci, z0), (rco, z1), device)
    fc.block_label(m, ((rci + rco) / 2, (z0 + z1) / 2), "Copper", mesh=device, circuit="coil", turns=p["turns"])
    fc.iabc_open_boundary(m, radius)
    fc.block_label(m, _air_point(radius, True), "Air", mesh=air)
    return m.result()


def _magnet_text(scored):
    def text(p):
        hm, z0, z1 = _magnet_dims(p)
        materials = {"NdFeB": {"Mu_x": 1.05, "H_c": p["coercivity"]}, "Copper": {"Sigma": 58}, "Air": {}}
        if scored == "force":
            title, what = "Axial force between a ring magnet and a coil", (
                f"The axial (z) force on the magnet, in newtons, positive along +z, with {_f(p['current'])} A in the coil.")
            footer = L2_FOOTER.format(unit="N", current=_f(p["current"]),
                                      quantity="the force on the magnet as minus the Lorentz force on the model's "
                                               "currents (the coil's)")
        else:
            title, what = "Flux linkage of a coil beside a ring magnet", (
                "The coil's flux linkage, in webers, with no current in the coil (the magnet's field alone).")
            footer = L2_FOOTER.format(unit="Wb", current="0", quantity="the circuit's flux linkage")
        return (_header(title, "a magnetics problem at 0 Hz, axisymmetric about the z axis (r >= 0), "
                               "lengths in millimetres") + f"""
## Device (r and z in mm)

- A ring magnet of `NdFeB`: {_f(p['magnet_inner_radius'])} <= r <= {_f(p['magnet_outer_radius'])},
  -{_f(hm)} <= z <= {_f(hm)}, magnetized along +z (remanence {_f(p['remanence'])} T).
- A coil of `Copper` above it: {_f(p['coil_inner_radius'])} <= r <= {_f(p['coil_outer_radius'])},
  {_f(z0)} <= z <= {_f(z1)}, with {p['turns']} turns (positive).
- Everything else is `Air`. Model the surroundings as open (unbounded) space, by a method of your choice.

## Materials (linear, isotropic)

{_material_lines(materials)}

## Scored

{what}
""" + footer)
    return text


def _magnet_wrong(p):
    return dict(p, turns=int(round(p["turns"] * 1.1)))


# --- the catalogue -----------------------------------------------------------------

FAMILIES = {f.name: f for f in (
    Family("l1_stator", 1, "Stator lamination", "material_map", "", 0.95, _stator_sample, _stator_build, _stator_text,
           _stator_wrong, region=_disk(lambda p: p["outer_radius"])),
    Family("l1_spm_rotor", 1, "Surface-magnet rotor", "material_map", "", 0.95, _spm_sample, _spm_build, _spm_text,
           _spm_wrong, region=_disk(lambda p: p["outer_radius"])),
    Family("l1_pot_core", 1, "Pot core", "material_map", "", 0.95, _pot_sample, _pot_build, _pot_text, _pot_wrong,
           region=_pot_region),
    Family("l2_coil_inductance", 2, "Air-core coil inductance", "inductance", "H", 0.02, _coil_sample, _coil_build,
           _coil_text, _coil_wrong, current=lambda p: 1.0),
    Family("l2_core_inductance", 2, "Gapped C-core inductance", "inductance", "H", 0.02, _core_sample, _core_build,
           _core_text, _core_wrong, current=lambda p: 1.0),
    Family("l2_magnet_force", 2, "Ring magnet and coil: axial force", "axial_force", "N", 0.02, _magnet_sample(True),
           _magnet_build, _magnet_text("force"), _magnet_wrong, current=lambda p: p["current"]),
    Family("l2_magnet_linkage", 2, "Ring magnet and coil: flux linkage", "flux_linkage", "Wb", 0.02, _magnet_sample(False),
           _magnet_build, _magnet_text("linkage"), _magnet_wrong, current=lambda p: 0.0),
)}
