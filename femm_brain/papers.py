"""Paper replications: case checks, model drawings and the paper index.

A paper case (``data/papers/<id>.json``) is an article-level assembly whose
parameters carry their provenance from a paper. ``check`` enforces the rules of
the replicate-paper skill (``skills/replicate-paper/``) that a JSON schema
cannot: derived values must follow from their expressions, every assumption
needs a control that varies it, quotes must be verbatim in the cached source
text, frozen cases need a measurement plan, and the model must give every label
a closed region of its own.
"""
import html
import json
import math

from .collect import DEFAULT_CACHE
from .components import catalog as component_catalog, evaluate
from .geometry import arc_geometry
from .reproduce import DATA, check_regions, load_papers, plan_level

PAPER_CACHE = DEFAULT_CACHE / "papers"
PROVENANCE = {"quoted": ("quote", "location"), "derived": ("expression", "inputs"),
              "figure": ("figure", "scale_reference", "uncertainty"), "assumed": ("reason", "sensitivity_control"),
              "numerical": ("reason",)}
TARGET_KINDS = ("analytic", "2d_fea", "3d_fea", "measured", "digitized")
STATUSES = ("candidate", "extracting", "frozen", "replicated", "partial", "failed")
FROZEN = STATUSES[2:]
GAP_DECISIONS = ("ask", "figure", "assumed", "out_of_scope", "resolved")


def load_index():
    return json.loads((DATA / "papers.json").read_text())


def source_text(case, cache=PAPER_CACHE):
    """The cached text of the case's source PDF (extracted text plus OCR of its pages, where tables
    are images), or None when it is not on this machine."""
    sha = (case.get("source") or {}).get("pdf_sha256")
    if not sha:
        return None
    parts = [(cache / sha / name) for name in ("text.txt", "ocr.txt")]
    texts = [path.read_text(errors="replace") for path in parts if path.is_file()]
    return "\n".join(texts) if texts else None


def _squash(text):
    return " ".join(text.replace(" ", " ").replace("−", "-").split()).lower()


def _found(quote, text):
    parts = [_squash(part) for part in quote.replace("...", "…").split("…") if part.strip()]
    return all(part in text for part in parts)


def _close(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def check(case, cache=PAPER_CACHE, build=True):
    """Return {"problems": [...], "notes": [...]}; a case with problems may not be frozen or run."""
    problems, notes = [], []
    for key in ("schema_version", "id", "set", "status", "title", "source", "machine"):
        if key not in case:
            problems.append(f"missing {key}")
    if problems:
        return {"problems": problems, "notes": notes}
    if case["status"] not in STATUSES:
        problems.append(f"unknown status {case['status']}")
    frozen = case["status"] in FROZEN
    if case["set"] == "held_out" and frozen and not case.get("skill_version"):
        problems.append("a held-out case records the skill_version it ran with")
    text = source_text(case, cache)
    text = _squash(text) if text else None
    if text is None and frozen:
        notes.append("source text not cached here; quotes were not checked")

    parameters = (case.get("article") or {}).get("parameters", {})
    values = {name: item.get("value") for name, item in parameters.items()}
    controls = {control["id"]: control for control in case.get("controls", [])}
    for name, item in parameters.items():
        kind = item.get("provenance")
        if kind not in PROVENANCE:
            problems.append(f"parameter {name}: provenance must be one of {', '.join(PROVENANCE)}")
            continue
        missing = [field for field in PROVENANCE[kind] + ("value", "unit") if field not in item]
        if missing:
            problems.append(f"parameter {name}: {kind} needs {', '.join(missing)}")
            continue
        if kind == "derived":
            unknown = [n for n in item["inputs"] if n not in values]
            if unknown:
                problems.append(f"parameter {name}: unknown inputs {unknown}")
                continue
            try:
                computed = evaluate(item["expression"], {n: values[n] for n in item["inputs"]})
            except (ValueError, SyntaxError, KeyError, ZeroDivisionError) as exc:
                problems.append(f"parameter {name}: expression fails ({exc})")
                continue
            if not _close(computed, item["value"]):
                problems.append(f"parameter {name}: value {item['value']} differs from its expression ({computed})")
        elif kind == "assumed":
            control = controls.get(item["sensitivity_control"])
            if control is None or name not in (control.get("parameters") or {}):
                problems.append(f"parameter {name}: assumption needs a control that varies it ({item['sensitivity_control']})")
        elif kind == "numerical" and not any(control.get("mesh_scale", 1) != 1 for control in controls.values()):
            problems.append(f"parameter {name}: a numerical choice needs a mesh control in the case")
        elif kind == "figure" and not (item.get("discrete") or item["uncertainty"] > 0):
            problems.append(f"parameter {name}: a figure reading needs a positive uncertainty, unless it is discrete (a direction or a count)")
        elif kind == "quoted" and text is not None and not _found(item["quote"], text):
            problems.append(f"parameter {name}: quote not found verbatim in the source text")
    for control in controls.values():
        if control.get("level") != "article":
            problems.append(f"control {control['id']}: paper controls vary the article level")
        unknown = set(control.get("parameters") or {}) - set(parameters)
        if unknown:
            problems.append(f"control {control['id']}: unknown parameters {sorted(unknown)}")

    for gap in case.get("gaps", []):
        if gap.get("decision") not in GAP_DECISIONS:
            problems.append(f"gap {gap.get('item')}: decision must be one of {', '.join(GAP_DECISIONS)}")
        elif gap["decision"] == "ask" and frozen:
            problems.append(f"gap {gap.get('item')}: resolve the question before freezing")

    acceptance = 0
    # Assessors that compare against a value from the paper follow the same rules as targets.
    compared = [(f"target {t.get('metric')}", t) for t in case.get("targets", [])]
    compared += [(f"assessor {a.get('metric') or a.get('measurement')}:{a.get('statistic') or a.get('order')}",
                  dict(a, kind=a.get("value_kind"), metric=a.get("metric") or f"{a.get('measurement')}:{a.get('field')}"))
                 for a in case.get("assessors", []) if "value" in a]
    for where, target in compared:
        missing = [field for field in ("metric", "value", "kind", "quote", "location", "used_for_acceptance") if field not in target]
        if missing:
            problems.append(f"{where}: needs {', '.join(missing)}")
            continue
        if target["kind"] not in TARGET_KINDS:
            problems.append(f"{where}: kind must be one of {', '.join(TARGET_KINDS)}")
        if not (target.get("relative_tolerance") or target.get("absolute_tolerance")):
            problems.append(f"{where}: needs a tolerance")
        if target["used_for_acceptance"]:
            acceptance += 1
            if target["kind"] == "digitized":
                # The uncertainty may be relative, absolute (in the target's unit) or both, like the tolerance;
                # the tolerance band at the target value must be at least twice the uncertainty band.
                size = abs(target["value"]) if isinstance(target["value"], (int, float)) else 0
                spread = max(size * (target.get("digitization_uncertainty") or 0), target.get("absolute_digitization_uncertainty") or 0)
                band = max(size * (target.get("relative_tolerance") or 0), target.get("absolute_tolerance") or 0)
                if not spread or band < 2 * spread:
                    problems.append(f"{where}: a digitized acceptance target needs a stated uncertainty and a tolerance of at least twice it")
            if target["kind"] in ("measured", "3d_fea") and not target.get("allowance"):
                problems.append(f"{where}: a {target['kind']} acceptance target needs an argued allowance")
        if text is not None and not _found(target["quote"], text):
            problems.append(f"{where}: quote not found verbatim in the source text")
    if frozen:
        if not (acceptance or any(a.get("kind") == "phase_linkage" for a in case.get("assessors", []))):
            problems.append("a frozen case needs at least one acceptance target or assessor")
        for key in ("measurements", "metrics", "coverage"):
            if not case.get(key):
                problems.append(f"a frozen case needs {key}")

    if build and (case.get("article") or {}).get("assembly"):
        try:
            plan = plan_level(case, "article")
            problems += [f"region: {problem}" for problem in check_regions(plan["expected_model"])]
        except (ValueError, KeyError, TypeError) as exc:
            problems.append(f"assembly does not build: {exc}")
    return {"problems": problems, "notes": notes}


def check_index():
    """The index and the case files must agree."""
    index, problems = load_index(), []
    cases = {case["id"]: case for case in load_papers()}
    seen = set()
    for entry in index["papers"]:
        if entry["id"] in seen:
            problems.append(f"{entry['id']}: listed twice")
        seen.add(entry["id"])
        case = cases.get(entry["id"])
        if entry["status"] != "candidate" and case is None:
            problems.append(f"{entry['id']}: status {entry['status']} but no case file")
        if case is not None and (case["status"] != entry["status"] or case["set"] != entry["set"]):
            problems.append(f"{entry['id']}: index and case disagree on status or set")
    problems += [f"{identifier}: case file not in the index" for identifier in set(cases) - seen]
    return problems


def components_listing():
    return [{"id": identifier, "layer": record["layer"], "title": record["title"], "physics": record["physics"],
             "problem_types": record["problem_types"]} for identifier, record in sorted(component_catalog().items())]


def render_svg(model, title=""):
    """A drawing of a typed model: contours by group, labels with their material or winding sign, magnet directions."""
    nodes = model["nodes"]
    points = [(n["x"], n["y"]) for n in nodes] + [(l["x"], l["y"]) for l in model["labels"]]
    if not points:
        raise ValueError("Nothing to draw")
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
    for arc in model["arcs"]:
        info = arc_geometry((nodes[arc["n0"]]["x"], nodes[arc["n0"]]["y"]), (nodes[arc["n1"]]["x"], nodes[arc["n1"]]["y"]), arc["angle"])
        cx, cy, r = info["center"][0], info["center"][1], info["radius"]
        for step in range(0, 33):
            angle = math.radians(info["start_angle"] + arc["angle"] * step / 32)
            x, y = cx + r * math.cos(angle), cy + r * math.sin(angle)
            xmin, xmax, ymin, ymax = min(xmin, x), max(xmax, x), min(ymin, y), max(ymax, y)
    span = max(xmax - xmin, ymax - ymin, 1e-9)
    pad, stroke, font = span * 0.04, span / 900, span / 70
    colours = ["#47605c", "#2f6fae", "#b5452f", "#7a5aa6", "#3d8a4f", "#a17c2a"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{xmin - pad:.9g} {-ymax - pad:.9g} {xmax - xmin + 2 * pad:.9g} '
             f'{ymax - ymin + 2 * pad:.9g}" width="1400" height="1400"><title>{html.escape(title)}</title>',
             f'<rect x="{xmin - pad:.9g}" y="{-ymax - pad:.9g}" width="{xmax - xmin + 2 * pad:.9g}" height="{ymax - ymin + 2 * pad:.9g}" fill="#ffffff"/>']
    for s in model["segments"]:
        a, b = nodes[s["n0"]], nodes[s["n1"]]
        width = stroke * (3 if s.get("boundary") else 1)
        parts.append(f'<line x1="{a["x"]:.9g}" y1="{-a["y"]:.9g}" x2="{b["x"]:.9g}" y2="{-b["y"]:.9g}" '
                     f'stroke="{colours[s.get("group", 0) % len(colours)]}" stroke-width="{width:.6g}"/>')
    for arc in model["arcs"]:
        a, b = nodes[arc["n0"]], nodes[arc["n1"]]
        r = arc_geometry((a["x"], a["y"]), (b["x"], b["y"]), arc["angle"])["radius"]
        width = stroke * (3 if arc.get("boundary") else 1)
        parts.append(f'<path d="M {a["x"]:.9g},{-a["y"]:.9g} A {r:.9g},{r:.9g} 0 0 0 {b["x"]:.9g},{-b["y"]:.9g}" fill="none" '
                     f'stroke="{colours[arc.get("group", 0) % len(colours)]}" stroke-width="{width:.6g}"/>')
    coercive = {m["name"] for m in model.get("materials", []) if m["props"].get("H_c")}
    for label in model["labels"]:
        x, y = label["x"], label["y"]
        if label.get("circuit"):
            text, colour = f'{label["circuit"]}{"+" if label["turns"] > 0 else "-"}', "#2e7d32" if label["turns"] > 0 else "#8e3b8e"
        else:
            text, colour = (label.get("material") or "hole")[:14], "#333333"
        if label.get("material") in coercive:
            function = label.get("magdir_function") or ""
            theta = math.degrees(math.atan2(y, x))
            direction = math.radians(theta + (180 if "180" in function else 0) if function else label["magdir"])
            length = font * 2.4
            parts.append(f'<line x1="{x:.9g}" y1="{-y:.9g}" x2="{x + length * math.cos(direction):.9g}" '
                         f'y2="{-(y + length * math.sin(direction)):.9g}" stroke="#c62828" stroke-width="{stroke * 2:.6g}"/>')
            parts.append(f'<circle cx="{x + length * math.cos(direction):.9g}" cy="{-(y + length * math.sin(direction)):.9g}" '
                         f'r="{stroke * 3:.6g}" fill="#c62828"/>')
            continue
        parts.append(f'<text x="{x:.9g}" y="{-y:.9g}" font-size="{font:.6g}" text-anchor="middle" fill="{colour}" '
                     f'font-family="sans-serif">{html.escape(text)}</text>')
    return "".join(parts) + "</svg>"


def render_case(case):
    return render_svg(plan_level(case, "article")["expected_model"], case["id"])
