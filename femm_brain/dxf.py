"""Geometry from DXF drawings that papers supply as data.

Supplementary vector files are read, never executed. ASCII DXF entities LINE,
ARC, CIRCLE and LWPOLYLINE (straight and bulged vertices) are turned into
contour primitives, and INSERTs place the BLOCKs they name (with insertion
point, scale and rotation). Primitives are segments [x0, y0, x1, y1] and
counterclockwise arcs [x0, y0, x1, y1, angle] no wider than 180 degrees, the
form the entity_pattern component draws.
"""
import math


def _pairs(text):
    lines = text.splitlines()
    return [(lines[i].strip(), lines[i + 1].strip()) for i in range(0, len(lines) - 1, 2)]


def _entities(pairs):
    """Split group-code pairs into entities: [(type, [(code, value), ...]), ...]."""
    out, current = [], None
    for code, value in pairs:
        if code == "0":
            if current:
                out.append(current)
            current = (value, [])
        elif current is not None:
            current[1].append((code, value))
    if current:
        out.append(current)
    return out


def read(text):
    """Entities of the ENTITIES section with INSERTs expanded: dicts with a type and the fields used here."""
    pairs = _pairs(text)
    sections, name, body = {}, None, []
    for code, value in pairs:
        if code == "0" and value == "SECTION":
            name, body = None, []
        elif code == "2" and name is None:
            name = value
        elif code == "0" and value == "ENDSEC":
            sections[name] = body
            name = None
        elif name is not None:
            body.append((code, value))
    blocks, current, entities = {}, None, []
    for kind, fields in _entities(sections.get("BLOCKS", [])):
        if kind == "BLOCK":
            current = dict(fields).get("2")
            blocks[current] = []
        elif kind == "ENDBLK":
            current = None
        elif current is not None:
            blocks[current].append(_parse(kind, fields))
    for kind, fields in _entities(sections.get("ENTITIES", [])):
        entities.append(_parse(kind, fields))
    return _expand(entities, blocks, 0)


def _parse(kind, fields):
    item = {"type": kind}
    if kind == "LWPOLYLINE":
        vertices = []
        for code, value in fields:
            if code == "10":
                vertices.append([float(value), 0.0, 0.0])
            elif code == "20":
                vertices[-1][1] = float(value)
            elif code == "42":
                vertices[-1][2] = float(value)
            elif code == "70":
                item["closed"] = bool(int(value) & 1)
        item["vertices"] = vertices
        return item
    for code, value in fields:
        if code in ("10", "20", "11", "21", "40", "41", "42", "50", "51"):
            item[code] = float(value)
        elif code in ("2", "8"):
            item[code] = value
    return item


def _expand(entities, blocks, depth):
    if depth > 8:
        raise ValueError("DXF blocks nest too deeply")
    out = []
    for item in entities:
        if item["type"] != "INSERT":
            out.append(item)
            continue
        if item.get("2") not in blocks:
            raise ValueError("DXF INSERT names an unknown block: " + str(item.get("2")))
        sx, sy = item.get("41", 1.0), item.get("42", 1.0)
        if not math.isclose(sx, sy) or sx <= 0:
            raise ValueError("Only uniform positive block scales are supported")
        angle, ox, oy = item.get("50", 0.0), item.get("10", 0.0), item.get("20", 0.0)
        for inner in _expand(blocks[item["2"]], blocks, depth + 1):
            out.append(_place(inner, sx, angle, ox, oy))
    return out


def _place(item, scale, angle, ox, oy):
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    move = lambda x, y: (ox + scale * (x * c - y * s), oy + scale * (x * s + y * c))
    placed = dict(item)
    if item["type"] == "LWPOLYLINE":
        placed["vertices"] = [[*move(x, y), bulge] for x, y, bulge in item["vertices"]]
        return placed
    for a, b in (("10", "20"), ("11", "21")):
        if a in item:
            placed[a], placed[b] = move(item[a], item[b])
    if "40" in item:
        placed["40"] = item["40"] * scale
    for key in ("50", "51"):
        if key in item and item["type"] in ("ARC",):
            placed[key] = item[key] + angle
    return placed


def _arc_pieces(cx, cy, r, start, end):
    """A counterclockwise arc from start to end degrees as primitives no wider than 180 degrees."""
    sweep = (end - start) % 360 or 360
    pieces = max(1, math.ceil(sweep / 180 - 1e-12))
    out = []
    for k in range(pieces):
        a0, a1 = start + sweep * k / pieces, start + sweep * (k + 1) / pieces
        out.append(["arc", cx + r * math.cos(math.radians(a0)), cy + r * math.sin(math.radians(a0)),
                    cx + r * math.cos(math.radians(a1)), cy + r * math.sin(math.radians(a1)), sweep / pieces])
    return out


def primitives(entities, scale=1.0):
    """Segments and counterclockwise arcs (as lists) from DXF entities, scaled to model units."""
    out = []
    for item in entities:
        kind = item["type"]
        if kind == "LINE":
            out.append(["segment", item["10"], item["20"], item["11"], item["21"]])
        elif kind == "ARC":
            out += _arc_pieces(item["10"], item["20"], item["40"], item["50"], item["51"])
        elif kind == "CIRCLE":
            out += _arc_pieces(item["10"], item["20"], item["40"], 0, 360)
        elif kind == "LWPOLYLINE":
            vertices = item["vertices"]
            ends = list(zip(vertices, vertices[1:] + (vertices[:1] if item.get("closed") else [])))
            for (x0, y0, bulge), (x1, y1, _) in ends:
                if abs(bulge) < 1e-12:
                    out.append(["segment", x0, y0, x1, y1])
                    continue
                # A bulge is tan(included angle / 4), positive for counterclockwise arcs.
                included = 4 * math.degrees(math.atan(abs(bulge)))
                chord = math.hypot(x1 - x0, y1 - y0)
                radius = chord / (2 * math.sin(math.radians(included / 2)))
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                offset = math.sqrt(max(radius ** 2 - (chord / 2) ** 2, 0.0))
                nx, ny = -(y1 - y0) / chord, (x1 - x0) / chord
                sign = 1 if (bulge > 0) == (included < 180) else -1
                cx, cy = mx + sign * offset * nx, my + sign * offset * ny
                a0, a1 = math.degrees(math.atan2(y0 - cy, x0 - cx)), math.degrees(math.atan2(y1 - cy, x1 - cx))
                # FEMM arcs run counterclockwise, so a clockwise (negative) bulge is drawn from its end to its start.
                out += _arc_pieces(cx, cy, radius, a0, a1) if bulge > 0 else _arc_pieces(cx, cy, radius, a1, a0)
        # Other entity types (text, dimensions, hatches) carry no contours.
    return [[p[0]] + [v * scale for v in p[1:5]] + p[5:] for p in out]
