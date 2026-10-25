"""Strict reconstruction of curated magnetic source models from passive data.

The caller pins both the original native asset and its normalized JSON digest.
Geometry operations are generated from typed entities, never copied from source
operation strings. Every saved material, B-H curve, circuit and entity attachment
is compared independently with the frozen expected assignments after execution.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .geometry import arc_geometry

MATERIAL_KEYS = ("Mu_x", "Mu_y", "H_c", "J_re", "Sigma", "d_lam", "Phi_h",
                 "LamFill", "LamType", "Phi_hx", "Phi_hy", "NStrands", "WireD")
BOUNDARY_KEYS = ("A_0", "A_1", "A_2", "Phi", "Mu_ssd", "Sigma_ssd", "c0", "c1", "BdryType")
UNITS = {"inches": .0254, "millimeters": .001, "centimeters": .01, "meters": 1.}


def _number(value):
    if isinstance(value, bool):
        raise ValueError("Source numeric fields cannot be boolean")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Source numeric fields must contain finite numbers") from exc
    if not math.isfinite(value):
        raise ValueError("Source numeric fields must contain finite numbers")
    return value


def _integer(value):
    value = _number(value)
    if not value.is_integer():
        raise ValueError("Source index or group must be integral")
    return int(value)


def _name(value):
    if not isinstance(value, str) or not value or len(value) > 120 or any(ord(c) < 32 for c in value):
        raise ValueError("Invalid source property name")
    return value


def _call(name, *values):
    return name + "(" + ",".join(json.dumps(v) if isinstance(v, str) else format(v, ".17g") for v in values) + ")"


def _index(value, collection, allow_none=True):
    index = _integer(value)
    if not (0 if allow_none else 1) <= index <= len(collection):
        raise ValueError("Source property index is out of range")
    return index


def _attrs(entity, count):
    values = entity.get("attributes", [])
    if len(values) < count or any(_number(v) != 0 for v in values[count:]):
        raise ValueError("Unsupported or incomplete source entity attributes")
    return [_number(v) for v in values[:count]]


def load_source(path, original_sha256, data_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != data_sha256:
        raise ValueError("Curated passive source data changed; source recipe review is required")
    source = json.loads(raw)
    if source.get("original_sha256") != original_sha256:
        raise ValueError("Original source asset provenance does not match the frozen recipe")
    return source


def compile_source(source, *, mesh_scale=1., circuit_currents=None):
    """Return geometry, fresh Lua lines and expected physical assignments.

    Supports curated DC Cartesian magnetic models, including nonlinear B-H
    materials. Point properties, external-region flags and complex excitations
    are deliberately rejected until explicitly supported by another recipe.
    """
    mesh_scale = _number(mesh_scale)
    if mesh_scale <= 0:
        raise ValueError("Source mesh scale must be positive")
    metadata = source["metadata"]
    if (metadata.get("physics") != "magnetics" or _number(metadata["Frequency"]) != 0
            or metadata.get("Coordinates") != "cartesian" or metadata.get("LengthUnits") not in UNITS
            or metadata.get("ProblemType") not in ("planar", "axisymmetric")
            or source.get("holes") or source.get("point_properties")):
        raise ValueError("Source recipe supports explicit DC Cartesian magnetic models without holes or point properties")
    depth, precision, min_angle = (_number(metadata[k]) for k in ("Depth", "Precision", "MinAngle"))
    if depth <= 0 or not 0 < precision < 1 or not 0 < min_angle < 90:
        raise ValueError("Invalid source problem definition")
    expected = {"metadata": {k: metadata[k] for k in ("Frequency", "Precision", "MinAngle", "Depth", "LengthUnits", "ProblemType", "Coordinates")},
                "materials": [], "boundaries": [], "circuits": [], "labels": [], "nodes": [], "segments": [], "arcs": []}
    lines = ["newdocument(0)", _call("mi_probdef", 0, metadata["LengthUnits"],
             "axi" if metadata["ProblemType"] == "axisymmetric" else "planar", precision, depth, min_angle)]
    for material in source["materials"]:
        name = _name(material["name"]); props = material["properties"]
        allowed = set(MATERIAL_KEYS) | {"BlockName", "BHPoints", "H_cAngle", "J_im"}
        if set(props) - allowed or any(_number(props.get(k, 0)) != 0 for k in ("H_cAngle", "J_im")):
            raise ValueError("Unsupported source material property")
        values = {k: _number(props[k]) for k in MATERIAL_KEYS}
        if values["Mu_x"] <= 0 or values["Mu_y"] <= 0:
            raise ValueError("Source permeability must be positive")
        bh = [[_number(b), _number(h)] for b, h in material.get("bh_points", [])]
        if _integer(props["BHPoints"]) != len(bh) or any(b < 0 or h < 0 for b, h in bh):
            raise ValueError("Invalid source B-H curve")
        if any(bh[i][0] <= bh[i-1][0] or bh[i][1] < bh[i-1][1] for i in range(1, len(bh))):
            raise ValueError("Source B-H curve must be ordered and monotone")
        lines.append(_call("mi_addmaterial", name, *(values[k] for k in MATERIAL_KEYS)))
        lines.extend(_call("mi_addbhpoint", name, b, h) for b, h in bh)
        expected["materials"].append({"name": name, "properties": dict(values, H_cAngle=0, J_im=0, BHPoints=len(bh)), "bh_points": bh})
    for boundary in source["boundaries"]:
        name = _name(boundary["name"]); props = boundary["properties"]
        if set(props) - set(BOUNDARY_KEYS) - {"BdryName", "c0i", "c1i", "InnerAngle", "OuterAngle"}:
            raise ValueError("Unsupported source boundary property")
        if any(_number(props.get(k, 0)) != 0 for k in ("c0i", "c1i", "InnerAngle", "OuterAngle")):
            raise ValueError("Complex or sliding-band source boundary is unsupported")
        values = {k: _number(props[k]) for k in BOUNDARY_KEYS}
        if values["BdryType"] not in (0, 2):
            raise ValueError("Only fixed-potential or mixed source boundary properties are supported")
        lines.append(_call("mi_addboundprop", name, *(values[k] for k in BOUNDARY_KEYS)))
        expected["boundaries"].append({"name": name, "properties": dict(values, c0i=0, c1i=0)})
    currents = circuit_currents or {}
    if set(currents) - {c["name"] for c in source["circuits"]}:
        raise ValueError("Unknown source circuit current override")
    for circuit in source["circuits"]:
        name = _name(circuit["name"]); props = circuit["properties"]
        if set(props) - {"CircuitName", "TotalAmps_re", "TotalAmps_im", "CircuitType"} or _number(props.get("TotalAmps_im", 0)) != 0:
            raise ValueError("Unsupported complex source circuit")
        values = {"TotalAmps_re": _number(currents.get(name, props["TotalAmps_re"])), "CircuitType": _integer(props["CircuitType"])}
        if values["CircuitType"] not in (0, 1):
            raise ValueError("Invalid source circuit type")
        lines.append(_call("mi_addcircprop", name, values["TotalAmps_re"], values["CircuitType"]))
        expected["circuits"].append({"name": name, "properties": dict(values, TotalAmps_im=0)})
    geometry = {k: [] for k in ("nodes", "segments", "arcs", "labels", "operations")}
    def operation(function, arguments, entity):
        ident = "op:" + str(len(geometry["operations"]))
        geometry["operations"].append({"id": ident, "function": function, "arguments": arguments, "geometry_ids": [entity["id"]]})
        entity["operation_id"] = ident
        lines.append(_call(function, *arguments))
    for i, node in enumerate(source["nodes"]):
        x, y = _number(node["x"]), _number(node["y"])
        prop, group = _attrs(node, 2)
        if prop != 0:
            raise ValueError("Source point property attachment is unsupported")
        group = _integer(group)
        entity = {"id": "node:" + str(i), "x": x, "y": y, "group": group}
        operation("mi_addnode", [x, y], entity)
        geometry["nodes"].append(entity); expected["nodes"].append({"x": x, "y": y, "group": group})
        if group:
            lines += [_call("mi_selectnode", x, y), _call("mi_setnodeprop", "<None>", group), "mi_clearselected()"]
    for section in ("segments", "arcs"):
        for i, item in enumerate(source[section]):
            n0, n1 = _integer(item["n0"]), _integer(item["n1"])
            if n0 == n1 or min(n0, n1) < 0 or max(n0, n1) >= len(geometry["nodes"]):
                raise ValueError("Invalid source geometry connectivity")
            a, b = (geometry["nodes"][n] for n in (n0, n1))
            entity = {"id": ("arc:" if section == "arcs" else "segment:") + str(i), "n0": n0, "n1": n1}
            if section == "segments":
                mesh, bi, hidden, group = _attrs(item, 4)
                if mesh == 0 or mesh < -1:
                    raise ValueError("Unsupported source segment mesh value")
                mesh = mesh * mesh_scale if mesh > 0 else -1
                midpoint = ((a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2)
                operation("mi_addsegment", [a["x"], a["y"], b["x"], b["y"]], entity)
            else:
                angle, mesh, bi, hidden, group = _attrs(item, 5)
                if mesh <= 0 or mesh * mesh_scale > 180:
                    raise ValueError("Invalid source arc mesh angle")
                mesh *= mesh_scale
                info = arc_geometry((a["x"], a["y"]), (b["x"], b["y"]), angle)
                theta = math.radians(info["start_angle"] + angle / 2)
                midpoint = (info["center"][0] + info["radius"] * math.cos(theta), info["center"][1] + info["radius"] * math.sin(theta))
                entity.update(angle=angle, max_segment_degrees=mesh)
                operation("mi_addarc", [a["x"], a["y"], b["x"], b["y"], angle, mesh], entity)
            bi = _index(bi, expected["boundaries"]); hidden = _integer(hidden); group = _integer(group)
            if hidden not in (0, 1):
                raise ValueError("Invalid source hidden-segment flag")
            entity["group"] = group; geometry[section].append(entity)
            name = expected["boundaries"][bi-1]["name"] if bi else "<None>"
            lines.append(_call("mi_selectarcsegment" if section == "arcs" else "mi_selectsegment", *midpoint))
            lines.append(_call("mi_setarcsegmentprop", mesh, name, hidden, group) if section == "arcs" else
                         _call("mi_setsegmentprop", name, 0 if mesh < 0 else mesh, int(mesh < 0), hidden, group))
            lines.append("mi_clearselected()")
            expected[section].append({"n0": n0, "n1": n1, "boundary_index": bi, "mesh": mesh, "hidden": hidden, "group": group, **({"angle": angle} if section == "arcs" else {})})
    for i, label in enumerate(source["labels"]):
        x, y = _number(label["x"]), _number(label["y"])
        material, mesh, circuit, direction, group, turns, flag = _attrs(label, 7)
        material = _index(material, expected["materials"], False); circuit = _index(circuit, expected["circuits"])
        group, turns = _integer(group), _integer(turns)
        if mesh < 0 or flag != 0:
            raise ValueError("Unsupported source block mesh or external/default flag")
        mesh *= mesh_scale
        material_name = expected["materials"][material-1]["name"]
        circuit_name = expected["circuits"][circuit-1]["name"] if circuit else "<None>"
        entity = {"id": "label:" + str(i), "x": x, "y": y, "name": material_name}
        operation("mi_addblocklabel", [x, y], entity); geometry["labels"].append(entity)
        lines += [_call("mi_selectlabel", x, y), _call("mi_setblockprop", material_name, int(mesh == 0), mesh, circuit_name, direction, group, turns), "mi_clearselected()"]
        expected["labels"].append({"x": x, "y": y, "material_name": material_name, "mesh": mesh, "circuit_name": circuit_name if circuit else None, "direction": direction, "group": group, "turns": turns})
    geometry["metadata"] = {k: metadata[k] for k in ("LengthUnits", "ProblemType")}
    return geometry, lines, expected, UNITS[metadata["LengthUnits"]]


def verify_source_physics(expected, actual):
    """Compare saved native assignments without trusting source or generated Lua."""
    def close(a, b):
        return math.isclose(_number(a), _number(b), rel_tol=1e-10, abs_tol=1e-12)
    for key, value in expected["metadata"].items():
        observed = actual["metadata"].get(key)
        if (observed != value if key in ("LengthUnits", "ProblemType", "Coordinates") else not close(observed, value)):
            raise ValueError("Saved native source problem definition differs: " + key)
    for section in ("materials", "boundaries", "circuits"):
        native = {item["name"]: item for item in actual[section]}
        if len(native) != len(actual[section]) or set(native) != {item["name"] for item in expected[section]}:
            raise ValueError("Saved native source property names differ: " + section)
        for frozen in expected[section]:
            observed = native[frozen["name"]]
            if any(key not in observed["properties"] or not close(observed["properties"][key], value) for key, value in frozen["properties"].items()):
                raise ValueError("Saved native source physical property differs: " + frozen["name"])
            if section == "materials":
                a, b = frozen["bh_points"], observed.get("bh_points", [])
                if len(a) != len(b) or any(not close(x, y) for pair_a, pair_b in zip(a, b) for x, y in zip(pair_a, pair_b)):
                    raise ValueError("Saved native source B-H curve differs: " + frozen["name"])
    def point_key(x, y):
        return round(_number(x), 9), round(_number(y), 9)
    expected_nodes = {point_key(item["x"], item["y"]): item for item in expected["nodes"]}
    if len(actual["nodes"]) != len(expected_nodes):
        raise ValueError("Saved native source node count differs")
    for item in actual["nodes"]:
        frozen = expected_nodes.get(point_key(item["x"], item["y"]))
        attrs = _attrs(item, 2)
        if frozen is None or attrs != [0, frozen["group"]]:
            raise ValueError("Saved native source node attachment differs")
    def contour_key(nodes, item, arc=False):
        ends = tuple(point_key(nodes[item[n]]["x"], nodes[item[n]]["y"]) for n in ("n0", "n1"))
        return ends + (round(item["angle"], 9),) if arc else tuple(sorted(ends))
    for section in ("segments", "arcs"):
        arc = section == "arcs"
        wanted = {contour_key(expected["nodes"], item, arc): item for item in expected[section]}
        seen = set()
        for item in actual[section]:
            key = contour_key(actual["nodes"], item, arc); frozen = wanted.get(key)
            attrs = _attrs(item, 5 if arc else 4)
            mesh, bi, hidden, group = attrs[1:] if arc else attrs
            bi = _index(bi, actual["boundaries"])
            native_boundary = actual["boundaries"][bi-1]["name"] if bi else None
            expected_boundary = expected["boundaries"][frozen["boundary_index"]-1]["name"] if frozen and frozen["boundary_index"] else None
            if frozen is None or key in seen or native_boundary != expected_boundary or not all(close(a, b) for a, b in ((mesh, frozen["mesh"]), (hidden, frozen["hidden"]), (group, frozen["group"]))):
                raise ValueError("Saved native source contour boundary/mesh attachment differs: " + section)
            seen.add(key)
        if seen != set(wanted):
            raise ValueError("Saved native source contour attachments are incomplete: " + section)
    labels = {point_key(item["x"], item["y"]): item for item in actual["labels"]}
    if len(labels) != len(actual["labels"]) or len(labels) != len(expected["labels"]):
        raise ValueError("Saved native source label count differs")
    for frozen in expected["labels"]:
        observed = labels.get(point_key(frozen["x"], frozen["y"]))
        if observed is None:
            raise ValueError("Saved native source label is missing")
        material, mesh, circuit, direction, group, turns, flag = _attrs(observed, 7)
        mi = _index(material, actual["materials"], False); ci = _index(circuit, actual["circuits"])
        if (actual["materials"][mi-1]["name"] != frozen["material_name"]
                or (actual["circuits"][ci-1]["name"] if ci else None) != frozen["circuit_name"]
                or flag != 0 or any(not close(a, b) for a, b in ((mesh, frozen["mesh"]), (direction, frozen["direction"]), (group, frozen["group"]), (turns, frozen["turns"])))):
            raise ValueError("Saved native source material/winding/block assignment differs")
    return {"status": "passed", "source_assignments_verified": True,
            "materials": len(expected["materials"]), "circuit_count": len(expected["circuits"]),
            "nonlinear_bh_points": sum(len(m["bh_points"]) for m in expected["materials"]),
            "boundary_attachments": {"status": "passed", "assigned_segments": sum(bool(s["boundary_index"]) for s in expected["segments"]),
                                     "assigned_arcs": sum(bool(a["boundary_index"]) for a in expected["arcs"])}}
