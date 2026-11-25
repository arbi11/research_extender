"""Check native field queries and independently read a static planar .ans mesh."""
import math
from pathlib import Path
import re


def read_observations(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) != 3 or fields[0] in values:
            raise ValueError("Malformed or duplicate native observation")
        pair = [float(v) for v in fields[1:]]
        if not all(math.isfinite(v) for v in pair):
            raise ValueError("Nonfinite native observation")
        values[fields[0]] = pair
    return values


def read_static_solution(path, analytical):
    """Derive B and energy from solver nodal potentials, independently of mo_* calls."""
    text = Path(path).read_text()
    marker = re.search(r"(?mi)^\[Solution\]\s*$", text)
    if not marker:
        raise ValueError("Missing native .ans solution section")
    lines = text[marker.end():].strip().splitlines()
    if not lines:
        raise ValueError("Empty native solution section")
    count = int(lines[0])
    if count < 3:
        raise ValueError("Native solution has no usable mesh nodes")
    if len(lines) < count + 2:
        raise ValueError("Truncated native solution nodes")
    nodes = [[float(v) for v in row.split()[:3]] for row in lines[1:1+count]]
    if len(nodes) != count or any(len(n) != 3 or not all(math.isfinite(v) for v in n) for n in nodes):
        raise ValueError("Malformed native solution nodes")
    elements = int(lines[1+count])
    triangles = [[int(v) for v in row.split()[:3]] for row in lines[count+2:count+2+elements]]
    if elements < 1 or len(triangles) != elements:
        raise ValueError("Native solution has no usable mesh elements")
    B, offset = analytical["Bx"], analytical["A0"]
    potential_error = max(abs(a-(offset+B*y)) for x, y, a in nodes)
    field_error = 0.0
    area = energy = 0.0
    for triangle in triangles:
        if len(triangle) != 3 or len(set(triangle)) != 3 or any(n < 0 or n >= count for n in triangle):
            raise ValueError("Invalid native solution triangle")
        (x0,y0,a0),(x1,y1,a1),(x2,y2,a2) = [nodes[n] for n in triangle]
        twice = (x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)
        if twice == 0:
            raise ValueError("Degenerate native solution triangle")
        dAdx = ((a1-a0)*(y2-y0)-(a2-a0)*(y1-y0))/twice
        dAdy = ((x1-x0)*(a2-a0)-(x2-x0)*(a1-a0))/twice
        bx, by = dAdy, -dAdx
        field_error = max(field_error, abs(bx-B), abs(by))
        triangle_area = abs(twice)/2
        area += triangle_area
        energy += (bx*bx+by*by)/(2*analytical["mu0"]*analytical["mu_r"])*triangle_area*analytical["depth"]
    return {"nodes": count, "elements": elements, "max_potential_error": potential_error,
            "max_field_error": field_error, "area_m2": area, "energy_J": energy}


def evaluate_solution(folder, case):
    folder = Path(folder)
    errors = []
    values = read_observations(folder / case["observations_file"])
    if set(values) != set(case["observation_keys"]):
        errors.append("Missing or unexpected native field observations")
    for key, expected in case["expected_values"].items():
        if key not in values:
            errors.append("Missing native field value: " + key)
            continue
        real, imag = values[key]
        if abs(real-expected["value"]) > expected["absolute_tolerance"] or abs(imag) > expected["imaginary_tolerance"]:
            errors.append(f"{key}: expected {expected['value']}, observed {real}+I*{imag}")
    headers = {}
    text = (folder / case["file"]).read_text()
    for key, expected in case["expected_headers"].items():
        match = re.search(r"(?m)^\[" + re.escape(key) + r"\]\s*=\s*(.*?)\s*$", text)
        if not match:
            errors.append("Missing native model header: " + key)
            continue
        raw = match.group(1).strip('"')
        actual = float(raw) if isinstance(expected, (int, float)) else raw
        headers[key] = actual
        if actual != expected:
            errors.append("Native model header mismatch: " + key)
    analytical = case["analytical"]
    mesh = read_static_solution(folder / case["solution_file"], analytical)
    mesh_counts = [("mesh.solved_nodes", mesh["nodes"]), ("mesh.solved_elements", mesh["elements"])]
    if case.get("require_mesh_return", True):
        mesh_counts.append(("mesh.created_nodes", mesh["nodes"]))
    for key, target in mesh_counts:
        if key not in values or values[key] != [target, 0.0]:
            errors.append("Native mesh count mismatch: " + key)
    if mesh["max_potential_error"] > analytical["potential_tolerance"]:
        errors.append("Solver nodal potentials differ from analytical A")
    if mesh["max_field_error"] > analytical["field_tolerance"]:
        errors.append("Solver element fields differ from analytical B")
    if abs(mesh["area_m2"]-analytical["width"]*analytical["height"]) > 1e-10:
        errors.append("Solver mesh area differs from the frozen rectangle")
    if abs(mesh["energy_J"]-analytical["energy"]) > analytical["energy_tolerance"]:
        errors.append("Energy derived from solver nodes differs from analytical energy")
    return {"headers": headers, "values": values, "solution_mesh": mesh}, errors
