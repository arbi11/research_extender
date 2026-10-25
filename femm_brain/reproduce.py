"""Ladder reproduction of official FEMM examples on the native engine.

Every case climbs up to three independent levels that share one typed
measurement plan, so their results are directly comparable:

1. ``original`` opens the unmodified official model file as data and solves it.
2. ``rebuild`` compiles a fresh model from the brain's stored passive model
   record, then checks every saved property and attachment against it.
3. ``article`` constructs the model from dimensions and settings stated in the
   article text (see ``article_recipes``), never from the official model file.

Downloaded scripts are never executed. Original model files are opened only by
the network-disabled engine container. Published targets are frozen in the case
catalog before execution, and every comparison is reported, including failures.
"""
from __future__ import annotations

import cmath
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import random
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

from .collect import DEFAULT_CACHE
from .geometry import arc_geometry, parse_fem

DATA = Path(__file__).parent / "data"
PHYSICS = {
    "magnetics": {"extension": ".fem", "document": 0, "pre": "mi", "post": "mo"},
    "electrostatics": {"extension": ".fee", "document": 1, "pre": "ei", "post": "eo"},
    "heatflow": {"extension": ".feh", "document": 2, "pre": "hi", "post": "ho"},
}
UNITS = {"inches": .0254, "millimeters": .001, "centimeters": .01, "meters": 1., "mils": 2.54e-5, "micrometers": 1e-6}
# Native file keys, in the argument order of each property's add function.
PROPERTY_KEYS = {
    ("magnetics", "materials"): ("Mu_x", "Mu_y", "H_c", "J_re", "J_im", "Sigma", "d_lam", "Phi_h", "LamFill", "LamType",
                                 "Phi_hx", "Phi_hy", "NStrands", "WireD", "H_cAngle"),
    ("magnetics", "boundaries"): ("A_0", "A_1", "A_2", "Phi", "Mu_ssd", "Sigma_ssd", "c0", "c0i", "c1", "c1i", "BdryType",
                                  "innerangle", "outerangle"),
    ("magnetics", "circuits"): ("TotalAmps_re", "TotalAmps_im", "CircuitType"),
    ("magnetics", "point_properties"): ("A_re", "A_im", "I_re", "I_im"),
    ("electrostatics", "materials"): ("ex", "ey", "qv"),
    ("electrostatics", "boundaries"): ("Vs", "qs", "c0", "c1", "BdryType"),
    ("electrostatics", "conductors"): ("Vc", "qc", "ConductorType"),
    ("electrostatics", "point_properties"): ("Vp", "qp"),
    ("heatflow", "materials"): ("Kx", "Ky", "qv", "Kt"),
    ("heatflow", "boundaries"): ("BdryType", "Tset", "qs", "Tinf", "h", "beta", "TinfRad"),
    ("heatflow", "conductors"): ("Tc", "qc", "ConductorType"),
    ("heatflow", "point_properties"): ("Tp", "qp"),
}
# Native loader defaults for keys an older file format may omit.
DEFAULTS = {"Mu_x": 1, "Mu_y": 1, "LamFill": 1, "ex": 1, "ey": 1, "Kx": 1, "Ky": 1}


def property_values(keys, given):
    """Complete native property values with the loader's defaults.

    FEMM 5 reads a missing radiation ambient as the convection ambient
    (heat_document.cpp, backward compatibility), so TinfRad follows Tinf.
    """
    values = {k: _number(given.get(k, DEFAULTS.get(k, 0)), k) for k in keys}
    if "TinfRad" in keys and "TinfRad" not in given:
        values["TinfRad"] = values["Tinf"]
    return values
COLLECTIONS = {"magnetics": ("materials", "boundaries", "circuits", "point_properties"),
               "electrostatics": ("materials", "boundaries", "conductors", "point_properties"),
               "heatflow": ("materials", "boundaries", "conductors", "point_properties")}
POINT_FIELDS = {"magnetics": ("A", "B1", "B2", "Sig", "E", "H1", "H2", "Je", "Js", "Mu1", "Mu2", "Pe", "Ph", "ff"),
                "electrostatics": ("V", "Dx", "Dy", "Ex", "Ey", "ex", "ey", "nrg"),
                "heatflow": ("T", "Fx", "Fy", "Gx", "Gy", "kx", "ky")}
LEVELS = ("original", "rebuild", "article", "components")
RELATIVE = 1e-9


def _number(value, what="value"):
    if isinstance(value, bool):
        raise ValueError(what + " cannot be boolean")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(what + " must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(what + " must be a finite number")
    return value


def _integer(value, what="index"):
    value = _number(value, what)
    if not value.is_integer():
        raise ValueError(what + " must be integral")
    return int(value)


def _name(value):
    if not isinstance(value, str) or not value or len(value) > 120 or any(ord(c) < 32 or c in '"\\' for c in value):
        raise ValueError("Unsupported property name: " + repr(value))
    return value


def _lua(value):
    if isinstance(value, str):
        return '"' + (_name(value) if value else "") + '"'
    if isinstance(value, complex):
        if value.imag == 0:
            return format(value.real, ".17g")
        return "(" + format(value.real, ".17g") + "+I*(" + format(value.imag, ".17g") + "))"
    if isinstance(value, bool):
        return str(int(value))
    return format(_number(value), ".17g")


def _call(function, *values):
    return function + "(" + ",".join(_lua(v) for v in values) + ")"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Typed model: one representation for stored records, article builders and
# saved native files, so all three are compiled and compared the same way.
# ---------------------------------------------------------------------------

def normalize(parsed):
    """Convert a passive ``parse_fem`` record into the typed model."""
    metadata = parsed["metadata"]
    physics = metadata.get("physics")
    if physics not in PHYSICS:
        raise ValueError("Reproduction supports magnetics, electrostatics and heat flow models")
    problem = {"units": metadata.get("LengthUnits", "inches"), "type": metadata.get("ProblemType", "planar"),
               "precision": _number(metadata.get("Precision", 1e-8)), "depth": _number(metadata.get("Depth", 1)),
               "min_angle": _number(metadata.get("MinAngle", 30)), "coordinates": metadata.get("Coordinates", "cartesian")}
    if problem["units"] not in UNITS or problem["type"] not in ("planar", "axisymmetric"):
        raise ValueError("Unsupported length units or problem type")
    if physics == "magnetics":
        problem.update(frequency=_number(metadata.get("Frequency", 0)), ac_solver=_integer(metadata.get("ACSolver", 0)))
        if "DoSmartMesh" in metadata:
            problem["smart_mesh"] = _integer(metadata["DoSmartMesh"])
    if physics == "heatflow":
        problem.update(prev_soln=metadata.get("PrevSoln", ""), dt=_number(metadata.get("dT", 0)))
    if all(k in metadata for k in ("extZo", "extRo", "extRi")):
        problem["outer_space"] = {k: _number(metadata[k]) for k in ("extZo", "extRo", "extRi")}
    model = {"physics": physics, "problem": problem}
    for collection in COLLECTIONS[physics]:
        keys = PROPERTY_KEYS[(physics, collection)]
        records = []
        for item in parsed.get(collection, []):
            props = item["properties"]
            unknown = {k for k in props if k.lower() not in {key.lower() for key in keys}} - {
                "BlockName", "BdryName", "CircuitName", "PointName", "ConductorName", "BHPoints", "TKPoints"}
            if unknown:
                raise ValueError("Unsupported native property keys: " + ", ".join(sorted(unknown)))
            lookup = {k.lower(): v for k, v in props.items()}
            values = property_values(keys, {k: lookup[k.lower()] for k in keys if k.lower() in lookup})
            record = {"name": _name(item["name"]), "props": values}
            if item.get("bh_points"):
                record["curve"] = [[_number(a), _number(b)] for a, b in item["bh_points"]]
            if item.get("tk_points"):
                record["curve"] = [[_number(a), _number(b)] for a, b in item["tk_points"]]
            records.append(record)
        model[collection] = records

    def named(collection, index):
        index = _integer(index)
        if index == 0:
            return None
        if not 1 <= index <= len(model[collection]):
            raise ValueError("Native property index is out of range: " + collection)
        return model[collection][index-1]["name"]

    scalar = physics != "magnetics"
    conductors = "conductors" if scalar else None

    def attrs(entity, count):
        values = list(entity.get("attributes", []))
        return values + ["0"] * (count - len(values))

    nodes = []
    for node in parsed["nodes"]:
        a = attrs(node, 4 if scalar else 3)
        lfs = _number(a[3 if scalar else 2]) if len(node.get("attributes", [])) > (3 if scalar else 2) else -1
        if lfs > 0:
            raise ValueError("Native nodal local feature sizes are not supported yet")
        nodes.append({"x": _number(node["x"]), "y": _number(node["y"]), "point": named("point_properties", a[0]),
                      "group": _integer(a[1]), "conductor": named(conductors, a[2]) if scalar else None})
    segments = []
    for item in parsed["segments"]:
        a = attrs(item, 5 if scalar else 4)
        mesh = _number(a[0])
        segments.append({"n0": item["n0"], "n1": item["n1"], "mesh": mesh if mesh > 0 else None,
                         "boundary": named("boundaries", a[1]), "hidden": bool(_integer(a[2])), "group": _integer(a[3]),
                         "conductor": named(conductors, a[4]) if scalar else None})
    arcs = []
    for item in parsed["arcs"]:
        a = attrs(item, 6 if scalar else 5)
        arcs.append({"n0": item["n0"], "n1": item["n1"], "angle": _number(a[0]), "max_segment": _number(a[1]),
                     "boundary": named("boundaries", a[2]), "hidden": bool(_integer(a[3])), "group": _integer(a[4]),
                     "conductor": named(conductors, a[5]) if scalar else None})
    labels = []
    for item in parsed["labels"]:
        if scalar:
            a = attrs(item, 4)
            flags = _integer(a[3])
            label = {"material": named("materials", a[0]), "group": _integer(a[2])}
        else:
            a = attrs(item, 7)
            flags = _integer(a[6])
            fctn = a[7] if len(a) > 7 else None
            label = {"material": named("materials", a[0]), "circuit": named("circuits", a[2]),
                     "magdir": _number(a[3]), "magdir_function": fctn or None, "group": _integer(a[4]),
                     "turns": _integer(a[5])}
        mesh = _number(a[1])
        label.update(x=_number(item["x"]), y=_number(item["y"]), mesh=mesh if mesh > 0 else None,
                     external=bool(flags & 1), default=bool(flags & 2))
        labels.append(label)
    model.update(nodes=nodes, segments=segments, arcs=arcs, labels=labels,
                 holes=[{"x": _number(h["x"]), "y": _number(h["y"])} for h in parsed.get("holes", [])])
    return model


def compile_model(model):
    """Return construction Lua for a typed model; nothing is copied from source scripts."""
    physics = model["physics"]
    spec = PHYSICS[physics]
    pre = spec["pre"]
    problem = model["problem"]
    kind = "axi" if problem["type"] == "axisymmetric" else "planar"
    lines = ["newdocument(" + str(spec["document"]) + ")"]
    if physics == "magnetics":
        lines.append(_call("mi_probdef", problem["frequency"], problem["units"], kind, problem["precision"], problem["depth"],
                           problem["min_angle"], problem.get("ac_solver", 0)))
        if problem.get("smart_mesh") is not None:
            lines.append(_call("mi_smartmesh", problem["smart_mesh"]))
    elif physics == "electrostatics":
        lines.append(_call("ei_probdef", problem["units"], kind, problem["precision"], problem["depth"], problem["min_angle"]))
    else:
        lines.append(_call("hi_probdef", problem["units"], kind, problem["precision"], problem["depth"], problem["min_angle"],
                           problem.get("prev_soln", ""), problem.get("dt", 0)))
    for material in model["materials"]:
        p = material["props"]
        if physics == "magnetics":
            if p["H_cAngle"] != 0:
                raise ValueError("Native coercivity angles are not supported by mi_addmaterial")
            lines.append(_call("mi_addmaterial", material["name"], p["Mu_x"], p["Mu_y"], p["H_c"], complex(p["J_re"], p["J_im"]),
                               p["Sigma"], p["d_lam"], p["Phi_h"], p["LamFill"], p["LamType"], p["Phi_hx"], p["Phi_hy"],
                               p["NStrands"], p["WireD"]))
            lines += [_call("mi_addbhpoint", material["name"], b, h) for b, h in material.get("curve", [])]
        elif physics == "electrostatics":
            lines.append(_call("ei_addmaterial", material["name"], p["ex"], p["ey"], p["qv"]))
        else:
            lines.append(_call("hi_addmaterial", material["name"], p["Kx"], p["Ky"], p["qv"], p["Kt"]))
            lines += [_call("hi_addtkpoint", material["name"], t, k) for t, k in material.get("curve", [])]
    for boundary in model["boundaries"]:
        p = boundary["props"]
        if physics == "magnetics":
            lines.append(_call("mi_addboundprop", boundary["name"], p["A_0"], p["A_1"], p["A_2"], p["Phi"], p["Mu_ssd"],
                               p["Sigma_ssd"], complex(p["c0"], p["c0i"]), complex(p["c1"], p["c1i"]), p["BdryType"],
                               p["innerangle"], p["outerangle"]))
        elif physics == "electrostatics":
            lines.append(_call("ei_addboundprop", boundary["name"], p["Vs"], p["qs"], p["c0"], p["c1"], p["BdryType"]))
        else:
            lines.append(_call("hi_addboundprop", boundary["name"], p["BdryType"], p["Tset"], p["qs"], p["Tinf"], p["h"], p["beta"],
                               p["TinfRad"]))
    for circuit in model.get("circuits", []):
        p = circuit["props"]
        lines.append(_call("mi_addcircprop", circuit["name"], complex(p["TotalAmps_re"], p["TotalAmps_im"]), p["CircuitType"]))
    for conductor in model.get("conductors", []):
        p = conductor["props"]
        lines.append(_call(pre + "_addconductorprop", conductor["name"], p["Vc" if physics == "electrostatics" else "Tc"], p["qc"],
                           p["ConductorType"]))
    for point in model["point_properties"]:
        p = point["props"]
        values = ((complex(p["A_re"], p["A_im"]), complex(p["I_re"], p["I_im"])) if physics == "magnetics" else
                  (p["Vp"], p["qp"]) if physics == "electrostatics" else (p["Tp"], p["qp"]))
        lines.append(_call(pre + "_addpointprop", point["name"], *values))
    if problem.get("outer_space"):
        o = problem["outer_space"]
        lines.append(_call(pre + "_defineouterspace", o["extZo"], o["extRo"], o["extRi"]))
    nodes = model["nodes"]
    lines += [_call(pre + "_addnode", n["x"], n["y"]) for n in nodes]
    for s in model["segments"]:
        a, b = nodes[s["n0"]], nodes[s["n1"]]
        lines.append(_call(pre + "_addsegment", a["x"], a["y"], b["x"], b["y"]))
    for arc in model["arcs"]:
        a, b = nodes[arc["n0"]], nodes[arc["n1"]]
        lines.append(_call(pre + "_addarc", a["x"], a["y"], b["x"], b["y"], arc["angle"], arc["max_segment"]))
    scalar = physics != "magnetics"
    for n in nodes:
        if n["point"] or n["group"] or n.get("conductor"):
            lines.append(_call(pre + "_selectnode", n["x"], n["y"]))
            lines.append(_call(pre + "_setnodeprop", n["point"] or "<None>", n["group"], *([n["conductor"] or "<None>"] if scalar else [])))
            lines.append(pre + "_clearselected()")
    for s in model["segments"]:
        a, b = nodes[s["n0"]], nodes[s["n1"]]
        lines.append(_call(pre + "_selectsegment", (a["x"]+b["x"])/2, (a["y"]+b["y"])/2))
        lines.append(_call(pre + "_setsegmentprop", s["boundary"] or "<None>", s["mesh"] or 0, int(s["mesh"] is None),
                           int(s["hidden"]), s["group"], *([s["conductor"] or "<None>"] if scalar else [])))
        lines.append(pre + "_clearselected()")
    for arc in model["arcs"]:
        a, b = nodes[arc["n0"]], nodes[arc["n1"]]
        info = arc_geometry((a["x"], a["y"]), (b["x"], b["y"]), arc["angle"])
        theta = math.radians(info["start_angle"] + arc["angle"]/2)
        lines.append(_call(pre + "_selectarcsegment", info["center"][0] + info["radius"]*math.cos(theta),
                           info["center"][1] + info["radius"]*math.sin(theta)))
        lines.append(_call(pre + "_setarcsegmentprop", arc["max_segment"], arc["boundary"] or "<None>", int(arc["hidden"]),
                           arc["group"], *([arc["conductor"] or "<None>"] if scalar else [])))
        lines.append(pre + "_clearselected()")
    for label in model["labels"]:
        lines += [_call(pre + "_addblocklabel", label["x"], label["y"]), _call(pre + "_selectlabel", label["x"], label["y"])]
        mesh = (int(label["mesh"] is None), label["mesh"] or 0)
        if scalar:
            lines.append(_call(pre + "_setblockprop", label["material"] or "<None>", *mesh, label["group"]))
        else:
            direction = label["magdir_function"] if label.get("magdir_function") else label["magdir"]
            lines.append(_call("mi_setblockprop", label["material"] or "<None>", *mesh, label["circuit"] or "<None>",
                               direction, label["group"], label["turns"]))
        if label["external"]:
            lines.append(pre + "_attachouterspace()")
        if label["default"]:
            lines.append(pre + "_attachdefault()")
        lines.append(pre + "_clearselected()")
    for hole in model["holes"]:
        lines += [_call(pre + "_addblocklabel", hole["x"], hole["y"]), _call(pre + "_selectlabel", hole["x"], hole["y"]),
                  _call(pre + "_setblockprop", "<No Mesh>", 1, 0, *(["<None>", 0, 0, 0] if not scalar else [0])),
                  pre + "_clearselected()"]
    lines.append(_call(pre + "_saveas", "built" + spec["extension"]))
    return lines


def _face_cycles(model, arc_step=1.0):
    """Bounded faces of the contour graph, smallest first.

    Contours (segments and arcs discretized at ``arc_step`` degrees) form a planar
    graph; walking each half-edge with the face on its left yields face cycles,
    of which the counterclockwise ones bound regions.
    """
    vertex_index, coordinates, adjacency = {}, [], {}

    def vertex(x, y):
        key = (round(x, 7), round(y, 7))
        if key not in vertex_index:
            vertex_index[key] = len(coordinates)
            coordinates.append(key)
            adjacency[vertex_index[key]] = set()
        return vertex_index[key]

    def edge(a, b):
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)

    nodes = model["nodes"]
    for s in model["segments"]:
        a, b = nodes[s["n0"]], nodes[s["n1"]]
        edge(vertex(a["x"], a["y"]), vertex(b["x"], b["y"]))
    for arc in model["arcs"]:
        a, b = nodes[arc["n0"]], nodes[arc["n1"]]
        info = arc_geometry((a["x"], a["y"]), (b["x"], b["y"]), arc["angle"])
        pieces = max(1, math.ceil(arc["angle"] / arc_step))
        chain = [vertex(a["x"], a["y"])]
        for i in range(1, pieces):
            theta = math.radians(info["start_angle"] + arc["angle"] * i / pieces)
            chain.append(vertex(info["center"][0] + info["radius"] * math.cos(theta), info["center"][1] + info["radius"] * math.sin(theta)))
        chain.append(vertex(b["x"], b["y"]))
        for u, v in zip(chain, chain[1:]):
            edge(u, v)
    ordered = {v: sorted(adjacency[v], key=lambda w: math.atan2(coordinates[w][1] - coordinates[v][1],
                                                               coordinates[w][0] - coordinates[v][0])) for v in adjacency}
    visited, cycles = set(), []
    for u in adjacency:
        for v in adjacency[u]:
            if (u, v) in visited:
                continue
            cycle, a, b = [], u, v
            while (a, b) not in visited:
                visited.add((a, b))
                cycle.append(a)
                around = ordered[b]
                a, b = b, around[(around.index(a) - 1) % len(around)]
            area = sum(coordinates[cycle[i]][0] * coordinates[cycle[(i + 1) % len(cycle)]][1]
                       - coordinates[cycle[(i + 1) % len(cycle)]][0] * coordinates[cycle[i]][1] for i in range(len(cycle))) / 2
            if area > 1e-12:
                cycles.append((area, [coordinates[i] for i in cycle]))
    cycles.sort(key=lambda item: item[0])
    return [cycle for _, cycle in cycles]


def _region_of(point, faces):
    """Index of the smallest face containing the point, or None outside every closed contour."""
    def inside(polygon):
        x, y, result = point[0], point[1], False
        for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                result = not result
        return result
    return next((index for index, face in enumerate(faces) if inside(face)), None)


def region_keys(model, points, arc_step=1.0):
    """Identify the region each point lies in, from the model's contours alone.

    A point's region is the smallest counterclockwise face containing it, so two
    models with identical contours give identical keys wherever labels sit.
    """
    faces = _face_cycles(model, arc_step)
    keys = []
    for point in points:
        index = _region_of(point, faces)
        face = None if index is None else faces[index]
        keys.append(None if face is None else (len(face), min(face), round(sum(p[0] for p in face), 5)))
    return keys


def check_regions(model, arc_step=1.0):
    """Every block label and hole must lie inside a closed region of its own."""
    faces = _face_cycles(model, arc_step)
    owners, problems = {}, []
    markers = ([(l["x"], l["y"], l["material"] or "<No Mesh>") for l in model["labels"]]
               + [(h["x"], h["y"], "hole") for h in model["holes"]])
    for x, y, name in markers:
        index = _region_of((x, y), faces)
        if index is None:
            problems.append(f"{name} label at ({x:g}, {y:g}) lies outside every closed contour")
        else:
            owners.setdefault(index, []).append(f"{name} at ({x:g}, {y:g})")
    problems += ["One region holds several labels: " + "; ".join(names) for names in owners.values() if len(names) > 1]
    return problems


def _used_properties(model):
    used = {"materials": {l["material"] for l in model["labels"] if l["material"]},
            "circuits": {l.get("circuit") for l in model["labels"] if l.get("circuit")},
            "boundaries": {x["boundary"] for x in model["segments"] + model["arcs"] if x["boundary"]},
            "conductors": {x.get("conductor") for x in model["segments"] + model["arcs"] + model["nodes"] if x.get("conductor")},
            "point_properties": {n["point"] for n in model["nodes"] if n["point"]}}
    return used


def compare_regions(expected, actual):
    """Region-by-region equivalence: identical contours, used properties and the
    assignments of every region, wherever each model places its labels."""
    differences = []
    for model in (expected, actual):
        model["_used"] = _used_properties(model)
    trimmed = []
    for model in (expected, actual):
        copy = dict(model)
        for collection, names in model["_used"].items():
            copy[collection] = [item for item in model.get(collection, []) if item["name"] in names]
        copy["labels"], copy["holes"] = [], []
        copy.pop("_used")
        trimmed.append(copy)
    for model in (expected, actual):
        model.pop("_used")
    differences += compare_models(*trimmed)

    def assignments(model):
        points = [(l["x"], l["y"]) for l in model["labels"]] + [(h["x"], h["y"]) for h in model["holes"]]
        keys = region_keys(model, points)
        regions = {}
        for key, label in zip(keys, model["labels"] + [{"material": "<No Mesh>"} for _ in model["holes"]]):
            value = {k: label.get(k) for k in ("material", "circuit", "turns", "magdir_function", "group", "mesh", "external")}
            if label.get("magdir") is not None and not label.get("magdir_function"):
                value["magdir"] = round(label["magdir"] % 360, 9) % 360
            if key is None:
                differences.append({"where": "regions.unbounded_label", "expected": None, "observed": label.get("material")})
            elif key in regions:
                differences.append({"where": "regions.duplicate_label", "expected": None, "observed": [regions[key], value]})
            regions[key] = value
        return regions

    a, b = assignments(expected), assignments(actual)
    if set(a) != set(b):
        differences.append({"where": "regions.labelled", "expected": len(a), "observed": len(b),
                            "unmatched": len(set(a) ^ set(b))})
    for key in set(a) & set(b):
        for field, value in a[key].items():
            other = b[key].get(field)
            same = math.isclose(value, other, rel_tol=RELATIVE, abs_tol=1e-9) if isinstance(value, float) and isinstance(other, float) else value == other
            if not same:
                differences.append({"where": f"regions.{a[key].get('material')}.{field}", "expected": value, "observed": other})
    return differences


def compare_models(expected, actual):
    """Return differences between two typed models, independent of entity order."""
    differences = []

    def close(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            return math.isclose(a, b, rel_tol=RELATIVE, abs_tol=1e-12)
        return a == b

    def compare(where, a, b):
        if not close(a, b):
            differences.append({"where": where, "expected": a, "observed": b})

    if expected["physics"] != actual["physics"]:
        return [{"where": "physics", "expected": expected["physics"], "observed": actual["physics"]}]
    for key, value in expected["problem"].items():
        # Axisymmetric problems ignore depth, and the engine re-saves it as one.
        if key == "depth" and expected["problem"].get("type") == "axisymmetric":
            continue
        if key == "coordinates" and value == "cartesian" and key not in actual["problem"]:
            continue
        compare("problem." + key, value, actual["problem"].get(key))
    for collection in COLLECTIONS[expected["physics"]]:
        wanted = {item["name"]: item for item in expected.get(collection, [])}
        seen = {item["name"]: item for item in actual.get(collection, [])}
        if set(wanted) != set(seen):
            differences.append({"where": collection + ".names", "expected": sorted(wanted), "observed": sorted(seen)})
        for name in set(wanted) & set(seen):
            for key, value in wanted[name]["props"].items():
                compare(f"{collection}.{name}.{key}", value, seen[name]["props"].get(key))
            a, b = wanted[name].get("curve", []), seen[name].get("curve", [])
            if len(a) != len(b) or any(not close(x, y) for pa, pb in zip(a, b) for x, y in zip(pa, pb)):
                differences.append({"where": f"{collection}.{name}.curve", "expected": a, "observed": b})

    def point(model, index):
        n = model["nodes"][index]
        return round(n["x"], 9), round(n["y"], 9)

    def keyed(model, section):
        result = {}
        for item in model[section]:
            if section == "nodes":
                key = (round(item["x"], 9), round(item["y"], 9))
            elif section == "segments":
                key = tuple(sorted((point(model, item["n0"]), point(model, item["n1"]))))
            elif section == "arcs":
                ends, angle = (point(model, item["n0"]), point(model, item["n1"])), item["angle"]
                key = ends + (round(angle, 7),)
            else:
                key = (round(item["x"], 9), round(item["y"], 9))
            if key in result:
                differences.append({"where": section + ".duplicate", "expected": None, "observed": list(key)})
            result[key] = {k: v for k, v in item.items() if k not in ("x", "y", "n0", "n1")}
        return result

    for section in ("nodes", "segments", "arcs", "labels", "holes"):
        a, b = keyed(expected, section), keyed(actual, section)
        if set(a) != set(b):
            differences.append({"where": section + ".entities", "expected": len(a), "observed": len(b),
                                "missing": [list(k) for k in sorted(set(a) - set(b))][:20],
                                "unexpected": [list(k) for k in sorted(set(b) - set(a))][:20]})
        for key in set(a) & set(b):
            for field, value in a[key].items():
                compare(f"{section}.{list(key)}.{field}", value, b[key].get(field))
    return differences


# ---------------------------------------------------------------------------
# Measurement plan: a small typed vocabulary rendered to postprocessor calls.
# ---------------------------------------------------------------------------

def _selection_lines(case, physics, selection, labels):
    post = PHYSICS[physics]["post"]
    if selection.get("all"):
        return [post + "_groupselectblock()"]
    if "group" in selection:
        return [_call(post + "_groupselectblock", selection["group"])]
    if "conductor" in selection:
        return [_call(post + "_selectconductor", selection["conductor"])]
    if "labels" in selection:
        points = selection["labels"]
    else:
        key, value = next((k, selection[k]) for k in ("material", "circuit") if k in selection)
        points = [(label["x"], label["y"]) for label in labels if label.get(key) == value]
    if not points:
        raise ValueError(f"{case['id']}: block selection {selection} matches no label")
    return [_call(post + "_selectblock", x, y) for x, y in points]


def measurement_lines(case, physics, labels, step):
    post = PHYSICS[physics]["post"]
    lines = []
    for m in case["measurements"]:
        ident, op = m["id"], m["op"]
        results = "v1,v2,v3"
        if op == "circuit":
            lines.append(results + "=" + _call("mo_getcircuitproperties", m["name"]))
        elif op == "conductor":
            lines.append(results + "=" + _call(post + "_getconductorproperties", m["name"]))
        elif op == "block_integral":
            lines += _selection_lines(case, physics, m["select"], labels)
            lines += [results + "=" + _call(post + "_blockintegral", m["type"]), post + "_clearblock()"]
        elif op == "line_integral":
            lines += [_call(post + "_addcontour", x, y) for x, y in m["points"]]
            lines += [results + "=" + _call(post + "_lineintegral", m["type"]), post + "_clearcontour()"]
        elif op == "gap_integral":
            lines.append(results + "=" + _call("mo_gapintegral", m["boundary"], m["type"]))
        elif op == "point_values":
            fields = POINT_FIELDS[physics]
            unknown = set(m["fields"]) - set(fields)
            if unknown:
                raise ValueError("Unknown point fields: " + ", ".join(sorted(unknown)))
            names = ",".join("p" + str(i) for i in range(len(fields)))
            for index, (x, y) in enumerate(m["points"]):
                lines.append(names + "=" + _call(post + "_getpointvalues", x, y))
                lines += [f'emit("{step}","{ident}:{field}",{index},p{fields.index(field)})' for field in m["fields"]]
            continue
        else:
            raise ValueError("Unknown measurement operation: " + op)
        lines += [f'emit("{step}","{ident}",{i},v{i+1})' for i in range(3)]
    lines += [f'emit("{step}","mesh_nodes",0,{post}_numnodes())', f'emit("{step}","mesh_elements",0,{post}_numelements())']
    return lines


def solve_lines(case, physics, labels):
    pre, post = PHYSICS[physics]["pre"], PHYSICS[physics]["post"]
    lines = ['out=openfile("raw.csv","w")', 'write(out,"step,id,index,re,im\\n")',
             'function emit(step,id,index,v) if v then write(out,step,",",id,",",index,",",re(v),",",im(v),"\\n") end end']
    steps = case.get("steps") or [{"label": "base", "modify": []}]
    for step in steps:
        label = step["label"]
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+", label):
            raise ValueError("Step labels must be simple identifiers")
        for change in step.get("modify", []):
            if change["function"] == "rotate_group":
                # Rotate a group (for example a rotor) about the origin, as machine scripts do between positions.
                lines += [_call(pre + "_selectgroup", change["group"]), _call(pre + "_moverotate", 0, 0, change["angle"]),
                          pre + "_clearselected()"]
                continue
            if change["function"] not in ("modifyboundprop", "modifymaterial", "modifycircprop", "modifyconductorprop"):
                raise ValueError("Unsupported step modification")
            lines.append(_call(pre + "_" + change["function"], change["name"], change["property"], change["value"]))
        lines += [pre + "_createmesh()", pre + "_analyze(1)", pre + "_loadsolution()"]
        lines += measurement_lines(case, physics, labels, label)
        lines.append(post + "_close()")
    lines += ["closefile(out)", 'out=openfile("solve-completed.txt","w")', 'write(out,"completed\\n")', "closefile(out)"]
    return lines


def read_raw(path):
    values = {}
    with open(path) as stream:
        for row in csv.DictReader(stream):
            value = complex(_number(row["re"]), _number(row["im"]))
            key = (row["step"], row["id"], int(row["index"]))
            if key in values:
                raise ValueError("Duplicate native measurement: " + str(key))
            values[key] = value
    return values


def compute_metrics(case, raw):
    """Apply the catalog's declared arithmetic to raw native values, per step."""
    metrics = {}
    steps = [s["label"] for s in case.get("steps") or [{"label": "base"}]]
    for step in steps:
        # Cases that declare steps name their metrics per step, even a single one.
        suffix = "@" + step if case.get("steps") else ""
        for metric in case["metrics"]:
            if metric.get("steps") and step not in metric["steps"]:
                continue
            value = raw[(step, metric["value"][0], metric["value"][1])]
            if metric.get("divide"):
                value = value / raw[(step, metric["divide"][0], metric["divide"][1])]
            value *= metric.get("scale", 1)
            part = metric.get("part", "re")
            metrics[metric["name"] + suffix] = value.real if part == "re" else value.imag if part == "im" else abs(value)
        metrics["mesh_nodes" + suffix] = raw[(step, "mesh_nodes", 0)].real
        metrics["mesh_elements" + suffix] = raw[(step, "mesh_elements", 0)].real
    return metrics


def assess(target, metrics):
    observed = metrics.get(target["metric"])
    if observed is None or not math.isfinite(observed):
        return {"status": "failed", "metric": target["metric"], "error": "Metric missing or nonfinite", "target": target}
    error = abs(observed - target["value"])
    tolerance = max(abs(target["value"]) * target.get("relative_tolerance", 0), target.get("absolute_tolerance", 0))
    return {"status": "passed" if error <= tolerance else "failed", "metric": target["metric"], "observed": observed,
            "expected": target["value"], "absolute_error": error,
            "relative_error": error / abs(target["value"]) if target["value"] else None,
            "absolute_tolerance": tolerance, "target": target}


# ---------------------------------------------------------------------------
# Catalog and execution
# ---------------------------------------------------------------------------

BINDING_FIELDS = ("id", "source_model_sha256", "steps", "measurements", "metrics", "targets", "assessors", "article", "components",
                  "components_agreement_tolerance")


def case_binding_digest(case):
    """Digest of everything that determines what a case builds, measures and accepts."""
    binding = {key: case.get(key) for key in BINDING_FIELDS}
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_cases():
    return json.loads((DATA / "reproductions.json").read_text())["cases"]


def load_designs():
    """New designs assembled from components; they run at the article (assembly) level only."""
    return json.loads((DATA / "designs.json").read_text())["designs"]


def load_papers():
    """Paper replication cases, one file each under data/papers/; they run at the article level."""
    folder = DATA / "papers"
    return [json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))] if folder.is_dir() else []


def catalog_file(case):
    """The catalog file a case is defined in, and so bound to."""
    if case.get("example_id"):
        return DATA / "reproductions.json"
    if case.get("source") and case.get("set"):
        return DATA / "papers" / (case["id"] + ".json")
    return DATA / "designs.json"


def catalog_digest(catalog):
    """Digest of a whole catalog.

    Paper cases gain findings and a retrospective after their runs, so the
    papers catalog is bound by each case's definition (what it builds, measures
    and accepts) rather than by its file bytes.
    """
    if catalog == "papers":
        bindings = sorted((case["id"], case_binding_digest(case)) for case in load_papers())
        return hashlib.sha256(json.dumps(bindings).encode()).hexdigest()
    return _sha(DATA / (catalog + ".json"))


def _source_record(case):
    sha = case.get("source_model_sha256")
    if not sha:
        return None, None
    record = json.loads((DATA / "models" / (sha + ".json")).read_text())
    if record.get("original_sha256") != sha:
        raise ValueError("Stored model provenance differs from the catalog binding")
    return record, normalize(record)


def scale_mesh(model, factor):
    """Scale every explicit mesh constraint; regions left to the mesher stay automatic."""
    if not 0.05 <= factor <= 20:
        raise ValueError("Mesh scale must be between 0.05 and 20")
    scaled = deepcopy(model)
    for item in scaled["labels"] + scaled["segments"]:
        if item["mesh"]:
            item["mesh"] *= factor
    for arc in scaled["arcs"]:
        arc["max_segment"] = max(arc["max_segment"] * factor, 0.1)
    return scaled, {"explicit_labels": sum(bool(l["mesh"]) for l in model["labels"]),
                    "automatic_labels": sum(not l["mesh"] for l in model["labels"]),
                    "explicit_segments": sum(bool(x["mesh"]) for x in model["segments"]), "arcs": len(model["arcs"])}


def permute_entities(model, seed):
    """The same model with its nodes, contours and labels in another order.

    The mesher's output depends on input order, so a result that moves under
    this permutation is mesh noise rather than a property of the problem.
    """
    rng = random.Random(seed)
    permuted = deepcopy(model)
    order = list(range(len(model["nodes"])))
    rng.shuffle(order)
    position = {old: new for new, old in enumerate(order)}
    permuted["nodes"] = [deepcopy(model["nodes"][i]) for i in order]
    for section in ("segments", "arcs"):
        for item in permuted[section]:
            item["n0"], item["n1"] = position[item["n0"]], position[item["n1"]]
    for section in ("segments", "arcs", "labels", "holes"):
        rng.shuffle(permuted[section])
    return permuted


def jitter_coordinates(model, seed, size):
    """The same model with every node moved by at most ``size`` model units in x and y.

    Symmetric machine geometry puts many points on common circles, where the
    mesher's decisions turn on the last bits of the coordinates. A result that
    moves under a sub-nanometre perturbation is mesh noise rather than physics.
    """
    if not 0 < size <= 1e-9:
        raise ValueError("Coordinate jitter must be positive and at most 1e-9 model units")
    rng = random.Random(seed)
    jittered = deepcopy(model)
    for node in jittered["nodes"]:
        node["x"] += rng.uniform(-size, size)
        node["y"] += rng.uniform(-size, size)
    return jittered


def plan_level(case, level, control=None):
    """Freeze construction, measurement Lua and targets for one level of one case.

    A control is a diagnostic variant (scaled explicit mesh, overridden article
    or component parameters, the same entities in another order, or coordinates
    moved by a sub-nanometre jitter). It is reported beside the level and never
    changes the level's acceptance.
    """
    if level not in LEVELS:
        raise ValueError("Unknown reproduction level: " + str(level))
    if control:
        if control.get("level") != level or level == "original":
            raise ValueError("Controls vary the rebuild, article or components level they name")
        if control.get("parameters"):
            if level not in ("article", "components"):
                raise ValueError("Only article and component controls may override parameters")
            case = deepcopy(case)
            for name, value in control["parameters"].items():
                if name not in case[level]["parameters"]:
                    raise ValueError(f"Unknown {level} parameter in control: " + name)
                case[level]["parameters"][name]["value"] = value
    record, model = _source_record(case)
    if level in ("original", "rebuild") and model is None:
        return {"case_id": case["id"], "level": level, "status": "not_applicable",
                "reason": "No official native model file is published for this example."}
    if level == "article":
        from .article_recipes import build as build_article
        built = build_article(case)
        if built.get("status") == "not_attempted":
            return {"case_id": case["id"], "level": level, "status": "not_attempted", "reason": built["reason"]}
        model = built["model"]
    reference = None
    if level == "components":
        spec = case.get("components")
        if not spec:
            return {"case_id": case["id"], "level": level, "status": "not_attempted",
                    "reason": "No reconstruction from machine components is registered for this case."}
        from .components import assemble
        reference = model
        model, used = assemble(spec["assembly"], {name: item["value"] for name, item in spec["parameters"].items()})
        built = {"parameters": spec["parameters"], "components": used, "assumptions": spec.get("assumptions", [])}
    mesh_scaling = None
    if control and control.get("mesh_scale", 1) != 1:
        model, mesh_scaling = scale_mesh(model, control["mesh_scale"])
    if control and control.get("entity_order_seed") is not None:
        model = permute_entities(model, control["entity_order_seed"])
    if control and control.get("coordinate_jitter"):
        model = jitter_coordinates(model, control["coordinate_jitter"]["seed"], control["coordinate_jitter"]["size"])
    if control and control.get("problem"):
        unknown = set(control["problem"]) - {"smart_mesh", "precision", "min_angle"}
        if unknown:
            raise ValueError("Controls may only override mesher and solver settings: " + ", ".join(sorted(unknown)))
        model = deepcopy(model)
        model["problem"].update(control["problem"])
    physics = model["physics"]
    extension = PHYSICS[physics]["extension"]
    if level == "original":
        construction = [_call("open", "original" + extension), _call(PHYSICS[physics]["pre"] + "_saveas", "built" + extension)]
    else:
        construction = compile_model(model)
    # The engine re-saves its document before every analysis, so the built
    # snapshot is kept separately from the file that later steps modify.
    construction.append(_call(PHYSICS[physics]["pre"] + "_saveas", "model" + extension))
    lua = "\n".join(construction + solve_lines(case, physics, model["labels"])) + "\n"
    targets = [t for t in case["targets"] if level in t.get("levels", LEVELS)]
    plan = {"schema_version": 1, "case_id": case["id"], "example_id": case.get("example_id"), "level": level,
            "physics": physics, "source_ids": case.get("source_ids", []),
            "source_url": case.get("source_url") or (case.get("source") or {}).get("url"),
            "source_model_sha256": case.get("source_model_sha256"),
            "expected_model": model if level != "original" else None,
            "article_parameters": built.get("parameters") if level == "article" else None,
            "article_assumptions": built.get("assumptions") if level == "article" else None,
            "component_parameters": built.get("parameters") if level == "components" else None,
            "reference_model": reference,
            "targets": targets, "assessors": case.get("assessors", []), "lua": lua,
            "control": control, "mesh_scaling": mesh_scaling, "case_binding_sha256": case_binding_digest(case),
            "module_sha256": _sha(__file__),
            "catalog_sha256": (_sha(catalog_file(case)) if catalog_file(case).is_file()
                               else hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()),
            "requires_ai": False}
    if level in ("article", "components"):
        plan["article_recipe_sha256"] = _sha(Path(__file__).with_name("article_recipes.py"))
        plan["components_sha256"] = _sha(Path(__file__).with_name("components.py"))
        plan["component_catalog_sha256"] = _sha(DATA / "components.json")
        plan["components_used"] = built.get("components", [])
        from .components import component_digests
        plan["component_digests"] = component_digests(plan["components_used"])
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return plan


def _run_engine(folder, *, engine, image, timeout):
    if engine == "native":
        from .native_check import _run
        executable = shutil.which(os.getenv("FEMM_LUA", "femm-lua"))
        if executable is None:
            raise RuntimeError("Native FEMM 5 is required; set FEMM_LUA")
        _run(executable, folder / "model.lua", folder, timeout)
        return {"engine": "native", "launcher": executable}
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("Docker is required for native solves")
    probe = subprocess.run([docker, "image", "inspect", "--format", "{{.Id}}", image], capture_output=True, text=True, timeout=15)
    if probe.returncode or not probe.stdout.strip().startswith("sha256:"):
        raise RuntimeError("The FEMM Docker image is unavailable: " + image)
    image_id = probe.stdout.strip()
    name = "femm-brain-reproduce-" + uuid.uuid4().hex[:16]
    args = [docker, "run", "--rm", "--init", "--name", name, "--network", "none", "--read-only", "--memory", "2g",
            "--cpus", "2", "--pids-limit", "256", "--tmpfs", "/tmp:rw,size=512m", "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "HOME=/tmp", "-e", "QT_QPA_PLATFORM=offscreen", "-e", "FEMM_DIR=/opt/femm/bin", "-e", "FEMM_MESH_EXPORT=1",
            "-v", f"{folder}:/work", "-w", "/work", image_id, "/opt/femm/bin/femm-lua", "-h", "-p", "48015", "model.lua"]
    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        subprocess.run([docker, "rm", "-f", name], capture_output=True, timeout=30)
        raise RuntimeError("Native solve exceeded its time limit") from exc
    (folder / "native.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError("Native FEMM exited unsuccessfully; inspect native.log")
    return {"engine": "docker", "image": image, "image_id": image_id, "network": "none", "read_only_container": True}


def run_level(case, level, output, *, engine="docker", image=None, timeout=900, control=None):
    """Execute one frozen level in an empty directory and retain all evidence."""
    plan = plan_level(case, level, control)
    folder = Path(output).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise ValueError("Reproduction output must be an empty directory; evidence is never overwritten")
    if plan.get("status") in ("not_applicable", "not_attempted"):
        report = dict(plan, completed_at=datetime.now(timezone.utc).isoformat())
        (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    physics = plan["physics"]
    extension = PHYSICS[physics]["extension"]
    (folder / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (folder / "model.lua").write_text(plan["lua"])
    if level == "original":
        source = DEFAULT_CACHE / "objects" / plan["source_model_sha256"]
        if not source.is_file() or _sha(source) != plan["source_model_sha256"]:
            raise ValueError("The official model file is missing from the verified cache; run restore-cache")
        shutil.copyfile(source, folder / ("original" + extension))
    report = {"schema_version": 1, "case_id": case["id"], "example_id": case.get("example_id"), "level": level,
              "physics": physics, "plan_sha256": plan["plan_sha256"], "started_at": datetime.now(timezone.utc).isoformat(),
              "checks": {name: {"status": "not_run"} for name in ("build", "solve", "targets")
                         } | ({"reference": {"status": "not_run"}} if level == "components" else {}),
              "requires_ai": False, "status": "failed", "metrics": {}, "control": control,
              "mesh_scaling": plan.get("mesh_scaling"), "components_used": plan.get("components_used"),
              "component_digests": plan.get("component_digests"), "case_binding_sha256": plan["case_binding_sha256"],
              "coverage": case.get("coverage")}
    started = time.monotonic()
    try:
        report["execution"] = _run_engine(folder, engine=engine, image=image or os.getenv("FEMM_BRAIN_IMAGE", ""), timeout=timeout)
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        report["execution_error"] = str(exc)
    stage = "build"
    try:
        saved = folder / ("built" + extension)
        if not saved.is_file():
            raise ValueError("The engine did not save the constructed model")
        actual = normalize(parse_fem(saved.read_text(errors="replace"), physics=physics))
        if level == "original":
            report["checks"]["build"] = {"status": "passed", "note": "Official file opened as data and re-saved by the engine.",
                                         "original_sha256": _sha(folder / ("original" + extension))}
        else:
            differences = compare_models(plan["expected_model"], actual)
            against = {"rebuild": "stored passive model record", "article": "article recipe model",
                       "components": "component assembly"}[level]
            report["checks"]["build"] = {"status": "passed" if not differences else "failed", "compared_against": against,
                                         "differences": differences[:50], "difference_count": len(differences)}
            if level in ("article", "components"):
                # Assembled models must give every label a closed region of its own.
                problems = check_regions(plan["expected_model"])
                report["checks"]["build"]["regions"] = {"status": "passed" if not problems else "failed", "problems": problems[:20]}
                if problems:
                    report["checks"]["build"]["status"] = "failed"
            if level == "components" and control:
                report["checks"]["reference"] = {"status": "not_applicable",
                                                 "reason": "A control varies the component model on purpose."}
            elif level == "components":
                # The component model must be the official model region by region:
                # identical contours, used properties and assignments.
                reference = compare_regions(deepcopy(plan["reference_model"]), deepcopy(plan["expected_model"]))
                report["checks"]["reference"] = {"status": "passed" if not reference else "failed",
                                                 "compared_against": "official model, region by region",
                                                 "differences": reference[:50], "difference_count": len(reference)}
                if reference:
                    report["checks"]["build"]["status"] = "failed"
        stage = "solve"
        if not (folder / "solve-completed.txt").is_file():
            raise ValueError("Native solve or measurement did not complete")
        raw = read_raw(folder / "raw.csv")
        report["metrics"] = compute_metrics(case, raw)
        report["raw"] = {f"{step}|{ident}|{index}": [value.real, value.imag] for (step, ident, index), value in sorted(raw.items())}
        report["checks"]["solve"] = {"status": "passed", "raw_values": len(raw)}
        stage = "targets"
        from .reference_models import run_assessor
        results = [assess(t, report["metrics"]) for t in plan["targets"]]
        for assessor in plan["assessors"]:
            if level in assessor.get("levels", LEVELS):
                results.append(run_assessor(assessor, case, raw, report["metrics"],
                                            plan["expected_model"] or _source_record(case)[1]))
        acceptance = [r for r in results if r.get("target", {}).get("used_for_acceptance", True)]
        # A level with only comparators has nothing to pass; it is reported as such.
        target_status = ("comparators_only" if not acceptance else
                         "passed" if all(r["status"] == "passed" for r in acceptance) else "failed")
        report["checks"]["targets"] = {"status": target_status, "acceptance_count": len(acceptance), "results": results}
        build_ok = report["checks"]["build"]["status"] == "passed"
        report["status"] = target_status if build_ok and target_status != "failed" else "failed"
    except (ValueError, OSError, KeyError, IndexError, csv.Error) as exc:
        report["checks"][stage] = {"status": "failed", "error": str(exc)}
    if report.get("execution_error"):
        report["status"] = "failed"
    report["duration_seconds"] = round(time.monotonic() - started, 3)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["artifacts"] = {p.name: _sha(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name != "report.json"}
    (folder / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def acceptance_measurements(case):
    """Measurement ids that feed an acceptance target or an assessor; the rest feed only comparators."""
    definitions = {metric["name"]: metric for metric in case["metrics"]}

    def sources(metric_name):
        metric = definitions.get(metric_name.split("@")[0])
        if metric is None:
            return set()
        return {metric["value"][0].split(":")[0]} | ({metric["divide"][0].split(":")[0]} if metric.get("divide") else set())

    used = set()
    for target in case["targets"]:
        if target.get("used_for_acceptance", True):
            used |= sources(target["metric"])
    for assessor in case.get("assessors", []):
        for key, value in assessor.items():
            names = value.values() if isinstance(value, dict) else value if isinstance(value, list) else [value]
            if key.endswith(("metric", "metrics")):
                for name in names:
                    used |= sources(name) if isinstance(name, str) else set()
            elif key == "measurement":
                used.add(value)
    return used


def compare_levels(case, reports):
    """Rebuild must reproduce the original's native numbers; article results are compared for information."""
    original = reports.get("original")
    if not original or original.get("status") in ("not_applicable", "not_attempted") or not original.get("metrics"):
        return {}
    result = {}
    tolerance = case.get("rebuild_relative_tolerance", 1e-6)
    rebuild = reports.get("rebuild")
    if rebuild and rebuild.get("raw"):
        # Every raw native value is compared, each relative to the largest
        # magnitude of its own quantity so near-zero components stay meaningful.
        scales = {}
        for key, (re_part, im_part) in original["raw"].items():
            ident = key.split("|")[1]
            scales[ident] = max(scales.get(ident, 0), abs(complex(re_part, im_part)))
        worst, missing = 0.0, sorted(set(original["raw"]) ^ set(rebuild["raw"]))
        for key in set(original["raw"]) & set(rebuild["raw"]):
            a, b = complex(*original["raw"][key]), complex(*rebuild["raw"][key])
            worst = max(worst, abs(a - b) / max(scales[key.split("|")[1]], 1e-300))
        result["rebuild"] = {"status": "passed" if worst <= tolerance and not missing else "failed",
                             "compared_raw_values": len(original["raw"]), "maximum_scaled_difference": worst,
                             "relative_tolerance": tolerance, "unmatched_values": missing[:20]}
    components = reports.get("components")
    if components and components.get("raw"):
        # Component builds mesh differently from the original (entity order,
        # segment direction and label positions all reach the mesher), so their
        # agreement bound is declared per case. It binds the measurements behind
        # acceptance targets and assessors; comparator-only quantities are
        # reported with their differences, since comparators never decide acceptance.
        bound = case.get("components_agreement_tolerance")
        bounded = acceptance_measurements(case)
        scales = {}
        for key, (re_part, im_part) in original["raw"].items():
            ident = key.split("|")[1]
            scales[ident] = max(scales.get(ident, 0), abs(complex(re_part, im_part)))
        worst, worst_key, comparators = 0.0, None, {}
        for key in set(original["raw"]) & set(components["raw"]):
            ident = key.split("|")[1]
            if ident.startswith("mesh_"):
                continue
            a, b = complex(*original["raw"][key]), complex(*components["raw"][key])
            difference = abs(a - b) / max(scales[ident], 1e-300)
            if ident.split(":")[0] not in bounded:
                comparators[ident] = max(comparators.get(ident, 0.0), difference)
            elif difference > worst:
                worst, worst_key = difference, key
        missing = sorted((set(original["raw"]) ^ set(components["raw"])))
        result["components"] = {"status": "passed" if bound is not None and worst_key is not None and worst <= bound and not missing
                                else "failed",
                                "maximum_scaled_difference": worst, "at": worst_key, "bound": bound,
                                "bounded_measurements": sorted(bounded), "comparator_differences": comparators,
                                "unmatched_values": missing[:20]}
    article = reports.get("article")
    if article and article.get("metrics"):
        rows = {}
        for name, value in original["metrics"].items():
            if name in article["metrics"]:
                observed = article["metrics"][name]
                rows[name] = {"original": value, "article": observed,
                              "relative_difference": abs(observed - value) / max(abs(value), abs(observed), 1e-300)}
        result["article"] = {"status": "informational", "metrics": rows}
    return result


def run_controls(case, reports, output, **options):
    """Run each catalog control and report metric changes against its base level."""
    results = {}
    for control in case.get("controls", []):
        base = reports.get(control["level"])
        report = run_level(case, control["level"], output / ("control-" + control["id"]), control=control, **options)
        changes = {}
        if base and base.get("metrics") and report.get("metrics"):
            for name, value in base["metrics"].items():
                if name in report["metrics"]:
                    observed = report["metrics"][name]
                    changes[name] = {"base": value, "control": observed,
                                     "relative_change": abs(observed - value) / max(abs(value), abs(observed), 1e-300)}
        results[control["id"]] = {"level": control["level"], "purpose": control.get("purpose"),
                                  "mesh_scale": control.get("mesh_scale", 1), "parameters": control.get("parameters"),
                                  "status": report["status"], "metric_changes": changes,
                                  "targets": [{k: r.get(k) for k in ("metric", "assessor", "status", "observed", "expected", "relative_error", "rms_error_ratio")}
                                              for r in report.get("checks", {}).get("targets", {}).get("results", [])]}
    return results


def reproduce(case_id, output, *, levels=LEVELS, engine="docker", image=None, timeout=900, controls=False, cases=None):
    cases = {case["id"]: case for case in (cases if cases is not None else load_cases())}
    if case_id not in cases:
        raise ValueError("Unknown reproduction case: " + str(case_id))
    case = cases[case_id]
    output = Path(output)
    reports = {level: run_level(case, level, output / level, engine=engine, image=image, timeout=timeout) for level in levels}
    control_results = run_controls(case, reports, output, engine=engine, image=image, timeout=timeout) if controls else {}
    summary = {"schema_version": 1, "case_id": case_id, "example_id": case.get("example_id"), "title": case["title"],
               "levels": {level: {"status": r["status"], **({"reason": r["reason"]} if r.get("reason") else {})} for level, r in reports.items()},
               "level_comparison": compare_levels(case, reports), "controls": control_results, "llm_calls": 0,
               "completed_at": datetime.now(timezone.utc).isoformat()}
    article = reports.get("article", {})
    original = _source_record(case)[1]
    if original is not None and article.get("status") not in (None, "not_applicable", "not_attempted"):
        # How far the article-only construction lands from the official file it
        # never read: evidence of what the article leaves unstated.
        plan = json.loads((output / "article" / "plan.json").read_text())
        differences = compare_models(original, plan["expected_model"])
        summary["article_model_vs_original"] = {"difference_count": len(differences), "differences": differences[:40]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
