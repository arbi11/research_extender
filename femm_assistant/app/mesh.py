"""Inspect an actual native FEMM/tangle triangular mesh before solving.

The format is Triangle-compatible ASCII, as documented and written by the
FEMM 5 mesher: https://github.com/dcm3c/tangle (writeNodeFile/writeEleFile).
Its library API is in-memory; callers must explicitly arrange mesh exports.
No mesh is inferred from the geometry preview or from a Lua return value.
"""
from collections import Counter
from pathlib import Path
import math

from .plan import GeometryPlan, Point, area, contains


MAX_NODES = 250_000
MAX_ELEMENTS = 500_000
MAX_EDGES = 750_000
MAX_FILE_BYTES = 64 * 1024 * 1024


class MeshValidationError(ValueError):
    """The mesher's exported artifacts cannot be accepted for this model."""


def _rows(path):
    if not path.is_file() or path.is_symlink():
        raise MeshValidationError(f"Missing regular mesh artifact: {path.name}.")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise MeshValidationError(f"Mesh artifact {path.name} exceeds the file-size limit.")
    with path.open(encoding="ascii") as stream:
        for number, line in enumerate(stream, 1):
            if len(line) > 4096:
                raise MeshValidationError(f"Overlong line in {path.name}:{number}.")
            fields = line.split("#", 1)[0].split()
            if fields:
                yield fields


def _integer(value):
    try:
        # Native files use decimal integer fields; reject floats and exponents.
        if len(value) > 12 or not value.lstrip("+-").isascii() or not value.lstrip("+-").isdigit():
            raise ValueError
        return int(value)
    except (ValueError, TypeError):
        raise MeshValidationError("Mesh contains an invalid integer field.") from None


def _finite(value):
    try:
        result = float(value)
    except (ValueError, OverflowError):
        raise MeshValidationError("Mesh contains an invalid numeric field.") from None
    if not math.isfinite(result):
        raise MeshValidationError("Mesh contains a non-finite value.")
    return result


def _header(rows, columns, name):
    row = next(rows, None)
    if row is None or len(row) != columns:
        raise MeshValidationError(f"Invalid {name} header.")
    return tuple(_integer(value) for value in row)


def _next(rows, name):
    row = next(rows, None)
    if row is None:
        raise MeshValidationError(f"Truncated {name} mesh artifact.")
    return row


def _finish(rows, ids, count, name):
    if next(rows, None) is not None:
        raise MeshValidationError(f"Unexpected extra rows in {name} mesh artifact.")
    if len(ids) != count or min(ids, default=-1) not in (0, 1) or max(ids, default=-1) != min(ids, default=-1) + count - 1:
        raise MeshValidationError(f"{name} indices must be unique, consecutive and start at 0 or 1.")


def _nodes(path):
    rows = iter(_rows(path))
    count, dimension, attributes, markers = _header(rows, 4, "node")
    if not 3 <= count <= MAX_NODES or dimension != 2 or not 0 <= attributes <= 8 or markers not in (0, 1):
        raise MeshValidationError("Unsupported node header or node-count limit exceeded.")
    result = {}
    coordinates = set()
    for _ in range(count):
        row = _next(rows, "node")
        size = 3 + attributes + markers
        # Tangle can append an LFS value without changing the Triangle header.
        if len(row) not in (size, size + 1):
            raise MeshValidationError("Invalid node row width.")
        index = _integer(row[0])
        x, y = _finite(row[1]), _finite(row[2])
        if abs(x) > 1000.000001 or abs(y) > 1000.000001:
            raise MeshValidationError("Mesh node is outside the coordinate limits.")
        for value in row[3:3 + attributes]:
            _finite(value)
        if markers:
            _integer(row[3 + attributes])
        if len(row) > size:
            _finite(row[-1])
        if index in result or (x, y) in coordinates:
            raise MeshValidationError("Mesh contains duplicate node indices or coordinates.")
        result[index] = (x, y)
        coordinates.add((x, y))
    _finish(rows, result, count, "node")
    return result


def _elements(path, nodes):
    rows = iter(_rows(path))
    count, vertices, attributes = _header(rows, 3, "element")
    if not 1 <= count <= MAX_ELEMENTS or vertices != 3 or attributes not in (0, 1):
        raise MeshValidationError("Unsupported element header or element-count limit exceeded.")
    result, ids, distinct = [], set(), set()
    for _ in range(count):
        row = _next(rows, "element")
        if len(row) != 4 + attributes:
            raise MeshValidationError("Invalid element row width.")
        index = _integer(row[0])
        triangle = tuple(_integer(value) for value in row[1:4])
        key = tuple(sorted(triangle))
        if index in ids or key in distinct:
            raise MeshValidationError("Mesh contains duplicate element indices or triangles.")
        if len(set(triangle)) != 3 or any(vertex not in nodes for vertex in triangle):
            raise MeshValidationError("Mesh element references an invalid node.")
        attribute = _finite(row[4]) if attributes else None
        if attribute is not None and (not attribute.is_integer() or not 1 <= attribute <= 33):
            raise MeshValidationError("Invalid FEMM block-label attribute.")
        ids.add(index)
        distinct.add(key)
        result.append((triangle, int(attribute) if attribute is not None else None))
    _finish(rows, ids, count, "element")
    return result, bool(attributes)


def _edge_artifact(path, nodes, expected):
    rows = iter(_rows(path))
    count, markers = _header(rows, 2, "edge")
    if not 1 <= count <= MAX_EDGES or markers not in (0, 1):
        raise MeshValidationError("Unsupported edge header or edge-count limit exceeded.")
    actual, ids = set(), set()
    for _ in range(count):
        row = _next(rows, "edge")
        if len(row) != 3 + markers:
            raise MeshValidationError("Invalid edge row width.")
        index, first, second = (_integer(value) for value in row[:3])
        key = (min(first, second), max(first, second))
        if index in ids or key in actual or first == second or first not in nodes or second not in nodes:
            raise MeshValidationError("Mesh contains an invalid or duplicate edge.")
        if markers:
            _integer(row[3])
        ids.add(index)
        actual.add(key)
    _finish(rows, ids, count, "edge")
    if actual != set(expected):
        raise MeshValidationError("Edge artifact does not match the triangular mesh.")


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _in_triangle(point, vertices, tolerance):
    return all(_cross(a, b, point) >= -tolerance for a, b in zip(vertices, vertices[1:] + vertices[:1]))


def _svg(nodes, elements):
    xmin, xmax = min(x for x, _ in nodes.values()), max(x for x, _ in nodes.values())
    ymin, ymax = min(y for _, y in nodes.values()), max(y for _, y in nodes.values())
    span = max(xmax - xmin, ymax - ymin)
    pad = span * .025
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{xmin-pad:g} {-ymax-pad:g} {xmax-xmin+2*pad:g} {ymax-ymin+2*pad:g}" role="img" aria-label="Actual finite-element mesh">',
             '<title>Actual native FEMM mesh; element geometry does not establish numerical convergence.</title>',
             f'<g stroke="#355765" stroke-width="{span/1800:g}" stroke-linejoin="round">']
    # Every triangle is represented, with no resampling or geometry substitution.
    palette = ("#edf5f5", "#b5d1cb", "#edc197", "#c4c7df", "#dbd6b5")
    for triangle, attribute in elements:
        points = " ".join(f"{nodes[index][0]:.10g},{-nodes[index][1]:.10g}" for index in triangle)
        fill = palette[(attribute or 0) % len(palette)]
        lines.append(f'<polygon points="{points}" fill="{fill}"/>')
    return "".join(lines) + "</g></svg>"


def validate_mesh(folder: Path, plan: GeometryPlan | None = None) -> dict:
    """Validate mesh files, write mesh.svg and return a JSON-safe inspection.

    Accepts ``model.node``/``model.ele`` or tangle CLI's ``model.1.*``.
    ``.edge`` is optional: the authoritative edge set is derived from triangles.
    FEMM's element attribute is the 1-based block-label index, not a material ID.
    Our compiler inserts background air first, then regions in plan order.
    """
    folder = Path(folder)
    if plan:
        plan = GeometryPlan.model_validate(plan.model_dump())
    bases = [name for name in ("model", "model.1") if (folder / f"{name}.node").is_file() and (folder / f"{name}.ele").is_file()]
    if len(bases) != 1:
        raise MeshValidationError("Expected one complete model.node/.ele or model.1.node/.ele mesh export.")
    base = bases[0]
    files = [f"{base}.node", f"{base}.ele"]
    try:
        nodes = _nodes(folder / files[0])
        elements, has_attributes = _elements(folder / files[1], nodes)
    except (UnicodeError, OSError) as exc:
        raise MeshValidationError("Could not read native mesh artifacts.") from exc
    edges, directions, used = Counter(), Counter(), set()
    min_angle, max_aspect = 180., 0.
    min_area, max_area, total_area = math.inf, 0., 0.
    assignments, covered = Counter(), set()
    labels = ([plan.air_label] + [region.label for region in plan.regions]) if plan else []
    for triangle, attribute in elements:
        points = [nodes[index] for index in triangle]
        signed = _cross(*points)
        lengths_sq = [math.dist(a, b) ** 2 for a, b in zip(points, points[1:] + points[:1])]
        twice_area = abs(signed)
        if twice_area <= max(lengths_sq) * 1e-14:
            raise MeshValidationError("Mesh contains a zero-area or numerically degenerate triangle.")
        if signed < 0:
            triangle = tuple(reversed(triangle))
            points.reverse()
        triangle_area = twice_area / 2
        min_area, max_area = min(min_area, triangle_area), max(max_area, triangle_area)
        total_area += triangle_area
        max_aspect = max(max_aspect, max(lengths_sq) / twice_area)
        for index in range(3):
            a, b, c = points[index], points[(index + 1) % 3], points[(index + 2) % 3]
            dot = (b[0] - a[0]) * (c[0] - a[0]) + (b[1] - a[1]) * (c[1] - a[1])
            min_angle = min(min_angle, math.degrees(math.atan2(twice_area, dot)))
        for first, second in zip(triangle, triangle[1:] + triangle[:1]):
            key = (min(first, second), max(first, second))
            edges[key] += 1
            directions[key] += 1 if first < second else -1
        used.update(triangle)
        if attribute is not None:
            if plan and attribute > len(labels):
                raise MeshValidationError("Mesh references a block label absent from the plan.")
            assignments[attribute] += 1
            if plan and attribute not in covered:
                label = labels[attribute - 1]
                if _in_triangle((label.x, label.y), points, max(lengths_sq) * 1e-12):
                    covered.add(attribute)
    if len(edges) > MAX_EDGES:
        raise MeshValidationError("Mesh edge-count limit exceeded.")
    if any(count > 2 or (count == 2 and directions[edge] != 0) for edge, count in edges.items()):
        raise MeshValidationError("Mesh has non-manifold or overlapping adjacent triangles.")
    edge_path = folder / f"{base}.edge"
    if edge_path.exists():
        try:
            _edge_artifact(edge_path, nodes, edges)
        except (UnicodeError, OSError) as exc:
            raise MeshValidationError("Could not read native edge artifact.") from exc
        files.append(edge_path.name)
    checks = [dict(name="native_artifacts", status="passed", detail="Finite 2D nodes, indexed three-node elements, unique triangles and consistent edges."),
              dict(name="element_geometry", status="passed", detail="Every element has positive area; each internal edge has two incident triangles on opposite sides.")]
    warnings = []
    if len(used) < len(nodes):
        warnings.append(f"{len(nodes)-len(used)} exported nodes are unused by the triangular mesh.")
    if plan:
        if any(not contains(plan.domain, Point(x=x, y=y), include_boundary=True) for x, y in nodes.values()):
            raise MeshValidationError("Mesh contains a node outside the problem domain.")
        boundary = [edge for edge, count in edges.items() if count == 1]
        adjacency = {}
        for first, second in boundary:
            adjacency.setdefault(first, set()).add(second)
            adjacency.setdefault(second, set()).add(first)
        if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
            raise MeshValidationError("Mesh outer boundary is not a closed contour.")
        reached, pending = set(), [next(iter(adjacency))]
        while pending:
            node = pending.pop()
            if node not in reached:
                reached.add(node)
                pending.extend(adjacency[node] - reached)
        if len(reached) != len(adjacency) or len(used) - len(edges) + len(elements) != 1:
            raise MeshValidationError("Mesh has a hole or disconnected component in the filled domain.")
        expected_area = area(plan.domain)
        # Arc meshes use chords. The compiler caps arc segments at 5 degrees,
        # giving <0.13% area loss for a circular outer boundary.
        tolerance = expected_area * (.002 if plan.domain.type == "circle" else 1e-7)
        if abs(total_area - expected_area) > max(tolerance, 1e-10):
            raise MeshValidationError("Mesh area does not cover the problem domain.")
        if not has_attributes or set(assignments) != set(range(1, len(labels) + 1)) or len(covered) != len(labels):
            raise MeshValidationError("Mesh block-label assignments or label-seed coverage are incomplete.")
        checks.extend([dict(name="domain_coverage", status="passed", detail="Nodes lie inside the domain; one closed outer boundary and the expected filled area are present."),
                       dict(name="block_labels", status="passed", detail="Every compiler block-label index is present and covers its intended label seed.")])
    else:
        warnings.append("No geometry plan was supplied; domain coverage and block-label seed checks were skipped.")
    if min_angle < 10 or max_aspect > 10:
        warnings.append("Slender elements are present. Inspect the geometry and refine the mesh before relying on results.")
    warnings.append("Mesh structure and element shape were inspected. Mesh-refinement and outer-boundary convergence have not been verified.")
    (folder / "mesh.svg").write_text(_svg(nodes, elements), encoding="utf-8")
    return dict(status="passed", node_count=len(nodes), element_count=len(elements), edge_count=len(edges),
                quality=dict(min_angle_deg=min_angle, max_aspect_ratio=max_aspect, min_area_mm2=min_area,
                             max_area_mm2=max_area, total_area_mm2=total_area,
                             aspect_definition="longest edge divided by its altitude (equilateral: 2/sqrt(3))"),
                checks=checks, warnings=warnings, convergence_verified=False, files=files,
                region_element_counts={str(key): value for key, value in sorted(assignments.items())})
