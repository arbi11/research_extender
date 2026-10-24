"""Independent comparison of saved native FEMM models with test specifications.

Public interfaces:
    verify_plan(spec: dict, path: str | Path) -> dict
    verify_asymmetric(spec: dict, path: str | Path) -> dict  # alias
    verify_capacitor(path: str | Path) -> dict

This module reads files only. It neither invokes FEMM nor consumes model-generated
explanations. ``passed`` is tri-state: False for a mismatch, None for incomplete
evidence, True only when every check passed. Probe locations are not stored in a
FEMM model: the caller must independently verify actual post-processing calls.
``geometry_and_assignments_passed`` excludes mesh and post-processing requests.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
from typing import Any

from femm_agent import femfile


TOL = 1e-7


def _near(a: Any, b: Any) -> bool:
    try:
        return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=TOL)
    except (TypeError, ValueError, OverflowError):
        return False


def _pt(p: Any) -> tuple[float, float]:
    if isinstance(p, dict):
        p = p["x"], p["y"]
    x, y = float(p[0]), float(p[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("geometry coordinates must be finite")
    return round(x, 7), round(y, 7)


def _between(point, a, b) -> bool:
    cross = (point[0] - a[0]) * (b[1] - a[1]) - (point[1] - a[1]) * (b[0] - a[0])
    scale = max(1.0, math.dist(a, b))
    return abs(cross) <= TOL * scale and all(min(a[k], b[k]) - TOL <= point[k] <= max(a[k], b[k]) + TOL for k in (0, 1))


def _simplify(points):
    pts = [_pt(p) for p in points]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            if _between(pts[i], pts[i - 1], pts[(i + 1) % len(pts)]):
                pts.pop(i)
                changed = True
                break
    return pts


def _edge_key(points):
    pts = _simplify(points)
    return tuple(sorted(tuple(sorted((pts[i], pts[(i + 1) % len(pts)]))) for i in range(len(pts))))


def _polygon(shape):
    if shape["type"] == "polygon":
        return [_pt(p) for p in shape["points"]]
    if shape["type"] == "rectangle":
        x0, y0, x1, y1 = (float(shape[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    raise ValueError(f"saved polygon verifier does not support shape {shape.get('type')!r}")


def _on_polygon_segment(a, b, polygon):
    return any(_between(a, polygon[i], polygon[(i + 1) % len(polygon)]) and
               _between(b, polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon)))


def _geometry(model):
    points = [_pt(p) for p in model.geometry.get("NumPoints", [])]
    adjacency = defaultdict(list)
    edges = []
    for row in model.geometry.get("NumSegments", []):
        i, j = int(row[0]), int(row[1])
        if not (0 <= i < len(points) and 0 <= j < len(points)):
            raise ValueError("native segment references a nonexistent node")
        a, b = points[i], points[j]
        if a == b:
            raise ValueError("native segment has identical endpoints")
        adjacency[a].append(b)
        adjacency[b].append(a)
        edges.append((a, b, row))
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise ValueError("native straight-segment graph is not a collection of disjoint closed polygons")
    remaining = set(adjacency)
    loops = []
    while remaining:
        start = min(remaining)
        previous, current = None, start
        loop = []
        while True:
            if current in loop:
                if current != start:
                    raise ValueError("native segment graph revisits an interior vertex")
                break
            loop.append(current)
            choices = adjacency[current]
            nxt = choices[0] if choices[0] != previous else choices[1]
            previous, current = current, nxt
            if len(loop) > len(adjacency):
                raise ValueError("native polygon traversal did not close")
        if len(loop) < 3:
            raise ValueError("native closed component has fewer than three vertices")
        loops.append(loop)
        remaining.difference_update(loop)
    free = sorted(set(points) - set(adjacency))
    return loops, edges, free


def _check(checks, name, passed, expected=None, actual=None, reason=None, *, scope="geometry_and_assignments"):
    checks[name] = {"passed": passed, "status": "unknown" if passed is None else ("passed" if passed else "failed"),
                    "expected": expected, "actual": actual, "scope": scope}
    if reason:
        checks[name]["reason"] = reason


def _rollup(checks, scope=None):
    selected = [c["passed"] for c in checks.values() if scope is None or c["scope"] == scope]
    if any(value is False for value in selected):
        return False
    if not selected or any(value is None for value in selected):
        return None
    return True


def _final(checks, model=None, path=None):
    actual = {}
    if model is not None:
        try:
            actual = model.summary()
        except Exception as exc:
            actual = {"summary_error": f"{type(exc).__name__}: {exc}"}
        actual.update({"raw_header": model.header, "raw_geometry": model.geometry,
                       "raw_properties": model.props, "resolved_labels": model.block_labels()})
    if path is not None:
        p = Path(path)
        actual["file_path"] = str(p.resolve())
        if p.is_file():
            data = p.read_bytes()
            actual["file_sha256"] = hashlib.sha256(data).hexdigest()
            actual["file_bytes"] = len(data)
    return {"passed": _rollup(checks), "geometry_and_assignments_passed": _rollup(checks, "geometry_and_assignments"),
            "mesh_constraint_passed": _rollup(checks, "mesh"), "checks": checks, "actual_summary": actual,
            "verification_source": "saved native file parsed independently; no model titles or prose used"}


def _headers(checks, model, physics, depth):
    _check(checks, "physics", model.physics == physics, physics, model.physics)
    for key, expected in (("LengthUnits", "millimeters"), ("ProblemType", "planar")):
        value = model.header.get(key)
        _check(checks, key.lower(), None if value is None else str(value).lower() == expected, expected, value,
               "Required native header is absent" if value is None else None)
    value = model.header.get("Depth")
    _check(checks, "depth", None if value is None else _near(value, depth), depth, value)
    if physics == "magnetics":
        value = model.header.get("Frequency")
        _check(checks, "frequency", None if value is None else _near(value, 0), 0, value)


def _no_extra_geometry(checks, model, expected_polygons):
    _check(checks, "no_arcs", len(model.geometry.get("NumArcSegments", [])) == 0, 0,
           len(model.geometry.get("NumArcSegments", [])))
    _check(checks, "no_unmeshed_holes", len(model.geometry.get("NumHoles", [])) == 0, 0,
           len(model.geometry.get("NumHoles", [])))
    try:
        loops, edges, free = _geometry(model)
        actual_keys = Counter(_edge_key(loop) for loop in loops)
        expected_keys = Counter(_edge_key(poly) for poly in expected_polygons)
        _check(checks, "closed_polygon_edges", actual_keys == expected_keys,
               [list(k) for k in expected_keys.elements()], [list(k) for k in actual_keys.elements()],
               "Closed undirected edge cycles compared after removing only collinear intermediate nodes; start and direction ignored.")
        _check(checks, "no_extra_free_nodes", not free, [], free)
        return edges
    except (ValueError, TypeError, IndexError) as exc:
        _check(checks, "closed_polygon_edges", False, "disjoint closed edge cycles", str(exc))
        return []


def _label_at(model, coordinate):
    wanted = _pt(coordinate)
    result = []
    for index, label in enumerate(model.block_labels()):
        if _pt((label["x"], label["y"])) == wanted:
            result.append((index, label, model.geometry["NumBlockLabels"][index]))
    return result


def _indexed(props, row, index):
    if len(row) <= index:
        return None
    pointer = int(row[index])
    return props[pointer - 1] if 0 < pointer <= len(props) else None


def _zero_props(prop, keys):
    if any(key not in prop for key in keys):
        return None
    return all(_near(prop[key], 0) for key in keys)


def _verify_plan(spec, model, path=None):
    checks = {}
    _headers(checks, model, "magnetics", spec["depth_mm"])
    if spec.get("units") != "millimeters" or spec.get("physics") != "planar_dc_magnetics":
        _check(checks, "verifier_scope", None, "millimeter planar DC specification", {"units": spec.get("units"), "physics": spec.get("physics")})
    domain = _polygon(spec["domain"])
    regions = spec["regions"]
    edges = _no_extra_geometry(checks, model, [domain] + [_polygon(region["shape"]) for region in regions])
    _check(checks, "label_count", len(model.block_labels()) == len(regions) + 1, len(regions) + 1, len(model.block_labels()))
    materials = {m["id"]: m["relative_permeability"] for m in spec["materials"]}
    expected_labels = [("background_air", spec["air_label"], 1.0, None)] + [
        (region["id"], region["label"], materials[region["material"]], region) for region in regions]
    mesh_values = []
    assigned_circuits = []
    for identity, point, mu, region in expected_labels:
        found = _label_at(model, point)
        _check(checks, f"label_{identity}", len(found) == 1, _pt(point), [label for _i, label, _r in found])
        if len(found) != 1:
            continue
        _index, label, raw = found[0]
        material = _indexed(model.materials(), raw, 2)
        if material is None:
            _check(checks, f"material_{identity}", False, {"mu_x": mu, "mu_y": mu}, None)
            continue
        vals = {key: material.get(key) for key in ("Mu_x", "Mu_y", "BHPoints", "H_c", "J_re", "J_im")}
        _check(checks, f"permeability_{identity}",
               None if any(key not in material for key in ("Mu_x", "Mu_y")) else all(_near(material[key], mu) for key in ("Mu_x", "Mu_y")),
               {"Mu_x": mu, "Mu_y": mu}, vals)
        nonlinear = material.get("BHPoints")
        _check(checks, f"linear_material_{identity}", None if nonlinear is None else _near(nonlinear, 0) and not material.get("BHPoints_data"), 0, nonlinear)
        _check(checks, f"no_material_excitation_{identity}", _zero_props(material, ("H_c", "J_re", "J_im")),
               {"H_c": 0, "J_re": 0, "J_im": 0}, vals)
        circuit = _indexed(model.circuits(), raw, 4)
        expected_circuit = region.get("circuit") if region else None
        if expected_circuit:
            assigned_circuits.append(circuit.get("CircuitName") if circuit else None)
            _check(checks, f"circuit_assignment_{identity}", circuit is not None, "assigned to the shared circuit", circuit)
            _check(checks, f"signed_turns_{identity}", None if len(raw) <= 7 else _near(raw[7], region["turns"]), region["turns"], raw[7] if len(raw) > 7 else None)
        else:
            _check(checks, f"no_circuit_{identity}", circuit is None and len(raw) > 4 and int(raw[4]) == 0, None, circuit)
        target = float(region.get("mesh_mm") or spec["mesh_mm"]) if region else float(spec["mesh_mm"])
        mesh_values.append({"label": identity, "native_mesh_size": float(raw[3]) if len(raw) > 3 else None, "requested_max_mm": target})
    expected_circuits = spec["circuits"]
    _check(checks, "circuit_count", len(model.circuits()) == len(expected_circuits), len(expected_circuits), model.circuits())
    if len(expected_circuits) == 1 and len(model.circuits()) == 1:
        circuit = model.circuits()[0]
        current = expected_circuits[0]["current_a"]
        vals = {key: circuit.get(key) for key in ("CircuitName", "TotalAmps_re", "TotalAmps_im", "CircuitType")}
        _check(checks, "circuit_current", None if any(key not in circuit for key in ("TotalAmps_re", "TotalAmps_im")) else
               _near(circuit["TotalAmps_re"], current) and _near(circuit["TotalAmps_im"], 0), {"real_A": current, "imag_A": 0}, vals)
        _check(checks, "series_circuit", None if "CircuitType" not in circuit else int(circuit["CircuitType"]) == 1, 1, vals)
        _check(checks, "shared_circuit", bool(assigned_circuits) and all(value == circuit.get("CircuitName") for value in assigned_circuits),
               "all requested winding labels refer to the same native circuit", assigned_circuits)
    else:
        _check(checks, "circuit_current", None, "one expected and actual shared circuit", model.circuits())

    boundary_rows = []
    internal_rows = []
    for a, b, raw in edges:
        if _on_polygon_segment(a, b, domain):
            prop = _indexed(model.boundaries(), raw, 3)
            valid = None if prop is not None and any(k not in prop for k in ("BdryType", "A_0", "A_1", "A_2")) else (
                prop is not None and int(prop["BdryType"]) == 0 and all(_near(prop[k], 0) for k in ("A_0", "A_1", "A_2")))
            boundary_rows.append({"edge": [a, b], "property": prop, "passed": valid})
        else:
            internal_rows.append({"edge": [a, b], "native_boundary_index": raw[3] if len(raw) > 3 else None})
    boundary_pass = False if not boundary_rows or any(r["passed"] is False for r in boundary_rows) else (
        None if any(r["passed"] is None for r in boundary_rows) else True)
    _check(checks, "outer_A_zero", boundary_pass, "prescribed A=0 on every enclosing edge", boundary_rows)
    _check(checks, "no_internal_boundary_constraints", bool(edges) and all(row["native_boundary_index"] == 0 for row in internal_rows),
           "no prescribed internal segment boundaries", internal_rows)
    node_props = [row[2] if len(row) > 2 else None for row in model.geometry.get("NumPoints", [])]
    _check(checks, "no_extra_point_constraints", all(value == 0 for value in node_props), "native point property index0", node_props)

    if len(mesh_values) != len(expected_labels):
        mesh_ok = None
    elif any(v["native_mesh_size"] is not None and v["native_mesh_size"] > v["requested_max_mm"] + TOL for v in mesh_values):
        mesh_ok = False
    elif any(v["native_mesh_size"] is None or v["native_mesh_size"] <= 0 for v in mesh_values):
        mesh_ok = None
    else:
        mesh_ok = True
    _check(checks, "native_region_mesh_limits", mesh_ok,
           "positive explicit native maximum size <=requested size at every material/air label", mesh_values,
           "Verifies per-region native controls covering the model, not a global UI setting. Zero/negative/absent means automatic and cannot prove the requested limit.", scope="mesh")
    _check(checks, "requested_probes", None, spec.get("probes", []), None,
           "FEMM model files do not persist post-processing probe requests. Verify actual native point-values call arguments/results separately.", scope="runtime_postprocessing")
    return _final(checks, model, path)


def verify_plan(spec: dict, path: str | Path) -> dict:
    """Verify the original/modified asymmetric typed plan against the native file."""
    try:
        model = femfile.parse_file(path)
        return _verify_plan(spec, model, path)
    except Exception as exc:
        checks = {}
        _check(checks, "file_or_specification_read", None, "readable native model and supported typed plan", f"{type(exc).__name__}: {exc}")
        return _final(checks, path=path)


verify_asymmetric = verify_plan


def verify_capacitor(path: str | Path) -> dict:
    """Check a10x1mm,100mm-deep,eps_r4 capacitor: bottom0V/top10V, insulated sides."""
    checks = {}
    try:
        model = femfile.parse_file(path)
        _headers(checks, model, "electrostatics", 100)
        rectangle = [(0, 0), (10, 0), (10, 1), (0, 1)]
        edges = _no_extra_geometry(checks, model, [rectangle])
        labels = model.block_labels()
        _check(checks, "label_count", len(labels) == 1, 1, labels)
        if len(labels) == 1:
            raw = model.geometry["NumBlockLabels"][0]
            material = _indexed(model.materials(), raw, 2)
            _check(checks, "dielectric_label_inside", 0 < labels[0]["x"] < 10 and 0 < labels[0]["y"] < 1,
                   "label strictly inside the dielectric rectangle", labels[0])
            wanted = {"ex": 4, "ey": 4, "qv": 0}
            passed = False if material is None else (None if any(k not in material for k in wanted) else all(_near(material[k], v) for k, v in wanted.items()))
            _check(checks, "dielectric", passed, wanted, material)
        else:
            _check(checks, "dielectric", None, {"ex": 4, "ey": 4, "qv": 0}, None)
        boundaries = model.boundaries()
        conductors = model.props.get("ConductorProps", [])
        plate_facts = {"bottom": [], "top": [], "sides": []}
        for a, b, raw in edges:
            bottom = _near(a[1], 0) and _near(b[1], 0)
            top = _near(a[1], 1) and _near(b[1], 1)
            boundary = _indexed(boundaries, raw, 3)
            conductor = _indexed(conductors, raw, 6)
            fact = {"edge": [a, b], "boundary": boundary, "conductor": conductor, "raw": raw}
            if bottom or top:
                voltage = 0 if bottom else 10
                evidence = []
                if conductor is not None:
                    evidence.append(None if any(k not in conductor for k in ("ConductorType", "Vc")) else
                                    int(conductor["ConductorType"]) == 1 and _near(conductor["Vc"], voltage))
                if boundary is not None:
                    evidence.append(None if any(k not in boundary for k in ("BdryType", "Vs")) else
                                    int(boundary["BdryType"]) == 0 and _near(boundary["Vs"], voltage))
                fact["passed"] = False if not evidence or any(v is False for v in evidence) else (None if any(v is None for v in evidence) else True)
                plate_facts["bottom" if bottom else "top"].append(fact)
            else:
                # Natural electrostatic boundary is zero normal displacement.
                # Explicit surface-charge boundary of exactly zero is equivalent.
                if conductor is not None:
                    fact["passed"] = False
                elif boundary is None:
                    fact["passed"] = len(raw) > 6 and int(raw[3]) == 0 and int(raw[6]) == 0
                elif any(k not in boundary for k in ("BdryType", "qs")):
                    fact["passed"] = None
                else:
                    fact["passed"] = int(boundary["BdryType"]) == 2 and _near(boundary["qs"], 0)
                plate_facts["sides"].append(fact)
        for name, facts in plate_facts.items():
            value = False if not facts or any(f["passed"] is False for f in facts) else (None if any(f["passed"] is None for f in facts) else True)
            _check(checks, f"electrode_{name}", value,
                   "bottom0V" if name == "bottom" else ("top10V" if name == "top" else "zero normal electric displacement"), facts)
        return _final(checks, model, path)
    except Exception as exc:
        _check(checks, "file_or_specification_read", None, "readable native capacitor model", f"{type(exc).__name__}: {exc}")
        return _final(checks, path=path)
