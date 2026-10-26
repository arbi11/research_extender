"""Geometry from vector drawings inside PDF reports.

Many reports embed a dimensioned drawing as vector art rather than a scan. A
page converted with ``pdftocairo -svg`` (a converter: nothing in the PDF is run)
keeps every stroke as a path; this module reads those strokes as data. It gives
the straight pieces of each stroked path with its colour (drawings often colour
the rotor, stator and dimensions apart), fits an axis-aligned ellipse to points
that should lie on a circle (a figure pasted into a document with unequal x and
y scales draws circles as ellipses, and their axes give the two scales), joins
pieces into polylines and simplifies polylines whose arcs the drawing
approximated by many short chords.

Coordinates are PDF points with y upward (the page height minus SVG y), after
each path's own ``transform``; cubic curves are split into straight pieces.
"""
import math
import re

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _matrix(transform):
    """The 2x3 affine matrix of an SVG transform attribute (matrix, translate and scale forms)."""
    a, b, c, d, e, f = 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
    for kind, args in re.findall(r"(matrix|translate|scale)\s*\(([^)]*)\)", transform or ""):
        values = [float(v) for v in re.findall(_NUMBER, args)]
        if kind == "matrix" and len(values) == 6:
            m = values
        elif kind == "translate":
            m = [1, 0, 0, 1, values[0], values[1] if len(values) > 1 else 0]
        elif kind == "scale":
            m = [values[0], 0, 0, values[1] if len(values) > 1 else values[0], 0, 0]
        else:
            raise ValueError("Unsupported transform: " + kind)
        # Compose: the existing matrix applied after the new one (SVG lists outer transforms first).
        a, b, c, d, e, f = (a * m[0] + c * m[1], b * m[0] + d * m[1], a * m[2] + c * m[3], b * m[2] + d * m[3],
                            a * m[4] + c * m[5] + e, b * m[4] + d * m[5] + f)
    return a, b, c, d, e, f


def _subpaths(data, curve_pieces=8):
    """Absolute polylines of an SVG path's data (M, L, H, V, C, Z and their relative forms)."""
    tokens = re.findall(r"[MLHVCZmlhvcz]|" + _NUMBER, data)
    out, current, x, y, start, command, i = [], [], 0.0, 0.0, (0.0, 0.0), None, 0
    while i < len(tokens):
        if re.fullmatch(r"[A-Za-z]", tokens[i]):
            command = tokens[i]
            i += 1
            if command in "Zz":
                if current:
                    current.append(start)
                    out.append(current)
                current, (x, y) = [], start
                continue
        if command is None:
            raise ValueError("Path data must start with a command")
        relative = command.islower()
        upper = command.upper()
        count = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6}.get(upper)
        if count is None:
            raise ValueError("Unsupported path command: " + command)
        values = [float(v) for v in tokens[i:i + count]]
        if len(values) != count:
            raise ValueError("Truncated path data")
        i += count
        if upper == "M":
            if current and len(current) > 1:
                out.append(current)
            x, y = (x + values[0], y + values[1]) if relative else values
            start, current = (x, y), [(x, y)]
            command = "l" if relative else "L"  # further pairs after a move are lines
        elif upper in "LHV":
            if upper == "L":
                x, y = (x + values[0], y + values[1]) if relative else values
            elif upper == "H":
                x = x + values[0] if relative else values[0]
            else:
                y = y + values[0] if relative else values[0]
            current.append((x, y))
        else:
            p0 = (x, y)
            c1, c2, p3 = [((x + values[k], y + values[k + 1]) if relative else (values[k], values[k + 1])) for k in (0, 2, 4)]
            for k in range(1, curve_pieces + 1):
                t = k / curve_pieces
                u = 1 - t
                current.append(tuple(u ** 3 * p0[j] + 3 * u * u * t * c1[j] + 3 * u * t * t * c2[j] + t ** 3 * p3[j] for j in (0, 1)))
            x, y = p3
    if current and len(current) > 1:
        out.append(current)
    return out


def segments(svg_text, min_length=0.0, page_height=None):
    """Straight pieces of every stroked path outside <defs>: [(colour, (x0, y0), (x1, y1)), ...] in PDF points.

    Glyphs of text live in <defs> and are skipped. Pieces shorter than
    ``min_length`` are dropped (some plotters end every stroke with a short tail).
    ``page_height`` defaults to the SVG's viewBox height and turns SVG y (down)
    into PDF y (up).
    """
    body = re.sub(r"<defs>.*?</defs>", "", svg_text, flags=re.S)
    if page_height is None:
        box = re.search(r'viewBox="([^"]*)"', svg_text)
        if not box:
            raise ValueError("The SVG has no viewBox; give the page height")
        page_height = float(box.group(1).split()[3])
    out = []
    for element in re.findall(r"<path\b[^>]*>", body):
        stroke = re.search(r'\bstroke="([^"]*)"', element)
        if not stroke or stroke.group(1) == "none":
            continue
        data = re.search(r'\bd="([^"]*)"', element)
        if not data:
            continue
        a, b, c, d, e, f = _matrix((re.search(r'\btransform="([^"]*)"', element) or [None, ""])[1])
        colour = stroke.group(1).replace(" ", "")
        for line in _subpaths(data.group(1)):
            mapped = [(a * px + c * py + e, page_height - (b * px + d * py + f)) for px, py in line]
            for p, q in zip(mapped, mapped[1:]):
                if math.dist(p, q) >= max(min_length, 1e-12):
                    out.append((colour, p, q))
    return out


def fit_axes_ellipse(points):
    """Least-squares axis-aligned ellipse through points: (x0, y0, semi_x, semi_y, largest relative residual).

    Points of one drawn circle give its centre, and semi_x / semi_y is the
    ratio of the figure's x and y scales.
    """
    if len(points) < 4:
        raise ValueError("An ellipse needs at least four points")
    # Solve A u^2 + C v^2 + D u + E v = 1 by normal equations, in coordinates centred on the points' mean
    # and scaled to unit size so the equations stay well conditioned far from the page origin.
    mx, my = sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points)
    size = max(max(abs(x - mx), abs(y - my)) for x, y in points) or 1.0
    original = points
    points = [((x - mx) / size, (y - my) / size) for x, y in points]
    rows = [(x * x, y * y, x, y) for x, y in points]
    n = 4
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    atb = [sum(r[i] for r in rows) for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(ata[r][col]))
        if abs(ata[pivot][col]) < 1e-300:
            raise ValueError("The points do not determine an ellipse")
        ata[col], ata[pivot], atb[col], atb[pivot] = ata[pivot], ata[col], atb[pivot], atb[col]
        for r in range(n):
            if r != col:
                factor = ata[r][col] / ata[col][col]
                ata[r] = [u - factor * v for u, v in zip(ata[r], ata[col])]
                atb[r] -= factor * atb[col]
    A, C, D, E = (atb[k] / ata[k][k] for k in range(n))
    if A <= 0 or C <= 0:
        raise ValueError("The points do not lie on an ellipse")
    x0, y0 = -D / (2 * A), -E / (2 * C)
    scale = 1 + A * x0 * x0 + C * y0 * y0
    a, b = math.sqrt(scale / A) * size, math.sqrt(scale / C) * size
    x0, y0 = mx + x0 * size, my + y0 * size
    residual = max(abs(math.hypot((x - x0) / a, (y - y0) / b) - 1) for x, y in original)
    return x0, y0, a, b, residual


def chains(pieces, tolerance):
    """Join pieces [(p, q), ...] whose ends lie within ``tolerance`` into polylines (lists of points).

    A closed polyline repeats its first point at its end only when the drawing
    closes it within the tolerance.
    """
    if not tolerance > 0:
        raise ValueError("The join tolerance must be positive")
    remaining = [list(piece) for piece in pieces]
    out = []
    while remaining:
        line = remaining.pop(0)
        grown = True
        while grown:
            grown = False
            for index, (p, q) in enumerate(remaining):
                if math.dist(line[-1], p) <= tolerance:
                    line.append(q)
                elif math.dist(line[-1], q) <= tolerance:
                    line.append(p)
                elif math.dist(line[0], q) <= tolerance:
                    line.insert(0, p)
                elif math.dist(line[0], p) <= tolerance:
                    line.insert(0, q)
                else:
                    continue
                remaining.pop(index)
                grown = True
                break
        out.append(line)
    return out


def simplify(points, tolerance):
    """Douglas-Peucker: drop points that lie within ``tolerance`` of the chord through their neighbours' kept ends."""
    points = list(points)
    if len(points) < 3:
        return points
    (ax, ay), (bx, by) = points[0], points[-1]
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    distances = [abs(dx * (py - ay) - dy * (px - ax)) / length if length else math.hypot(px - ax, py - ay)
                 for px, py in points[1:-1]]
    index = max(range(len(distances)), key=distances.__getitem__) + 1
    if distances[index - 1] <= tolerance:
        return [points[0], points[-1]]
    return simplify(points[:index + 1], tolerance)[:-1] + simplify(points[index:], tolerance)
