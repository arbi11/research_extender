"""The reward: re-solve the submitted model with the scorer's own Lua and compare it with the key.

The scorer never uses a number the player reports. Before running anything it screens the
file: circuit names must be plain, and the only Lua a .fem file can carry (magnetization
direction expressions) may use only arithmetic and position variables. So a submission cannot
print fake results or read the scorer's script, and each run also tags its output lines with
a fresh random token that only the scorer knows.

Reward: 1 for a pass. A miss earns partial credit, at most 0.5, so passing always pays more:
Level 2 gets 0.5 x (1 - log10(error / tolerance)), floored at 0; Level 1 gets 0.5 x the mean
intersection-over-union of the materials. Every failure stage before a result scores 0.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import re
import secrets
import shutil

from femm_brain.reproduce import compile_model

from . import engine
from .families import FAMILIES, POINTS, SCORE_SECONDS

MAX_BYTES = 20_000_000
NAME = re.compile(r"[A-Za-z0-9_.+\- ]{1,40}")
TOKEN = re.compile(r'"[^"]*"|\S+')
NUMBER = re.compile(r"\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?")
WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
EXPRESSION_CHARS = re.compile(r"[\sA-Za-z0-9_.+\-*/^(),]*")
EXPRESSION_WORDS = {"theta", "r", "R", "x", "y", "z", "pi", "PI", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
                    "sqrt", "abs", "exp", "log", "deg", "rad"}


# --- reading and screening a submission -------------------------------------------

def read_fem(text):
    """Circuit names, block labels and quoted label strings (direction expressions) of a .fem file.

    FEMM reads tags whatever their case, so this does too, and it reads every block-label
    section. A block label row is: x, y, material, mesh size, circuit (1-based, 0 for none),
    direction, group, turns, external, ... Each label comes back as {"x", "y", "circuit", "group"}.
    Raises ValueError on a file it cannot read.
    """
    circuits = re.findall(r'^\s*<CircuitName>\s*=\s*"(.*)"\s*$', text, re.M | re.I)
    labels, strings = [], []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("[numblocklabels]"):
            count = int(line.split("=", 1)[1])
            for row in lines[i + 1:i + 1 + count]:
                tokens = TOKEN.findall(row)
                strings += [t[1:-1] for t in tokens if t.startswith('"')]
                if len(tokens) < 7:
                    continue
                index = int(float(tokens[4]))
                labels.append({"x": float(tokens[0]), "y": float(tokens[1]), "group": int(float(tokens[6])),
                               "circuit": circuits[index - 1] if 0 < index <= len(circuits) else None})
    return circuits, labels, strings


def safe_expression(expression):
    if not EXPRESSION_CHARS.fullmatch(expression):
        return False
    return not set(WORD.findall(NUMBER.sub(" ", expression))) - EXPRESSION_WORDS


def screen(text, task):
    """Problems that stop a submission before any solve, and its circuit names."""
    try:
        circuits, labels, strings = read_fem(text)
    except (ValueError, IndexError) as error:
        return [f"model.fem could not be read ({error})"], []
    problems = [f"circuit name {name!r} is not plain" for name in circuits if not NAME.fullmatch(name)]
    problems += [f"label expression {s!r} uses more than arithmetic" for s in strings if not safe_expression(s)]
    if re.search(r'\[PrevSoln\]\s*=\s*"[^"]+"', text, re.I):
        problems.append("the model refers to a previous solution file")
    if task.get("circuit") and task["circuit"] not in circuits:
        problems.append(f"no circuit named {task['circuit']!r}")
    elif task.get("circuit") and not any(l["circuit"] == task["circuit"] for l in labels):
        problems.append(f"no block is assigned to circuit {task['circuit']!r}")
    return problems, circuits


# --- the scorer's Lua ---------------------------------------------------------------

def points_for(key):
    family = FAMILIES[key["family"]]
    return family.region(key["params"], random.Random(key["points_seed"]), POINTS)


def scoring_lines(task, key, circuits, nonce):
    """Lua that opens model.fem, sets every circuit, solves, and prints tagged measurements.

    The axial force on the magnet is minus the Lorentz force summed over every block (Newton's
    third law: the coil is the only current, and a magnet adds no J x B term). On the same
    meshes it converges far faster than the stress tensor taken around the magnet. All blocks
    are selected at once because mo_selectblock toggles: clicking one region twice, say through
    two labels in one coil, leaves it unselected.
    """
    tag = f"GYM {nonce}"
    lines = ['opendocument("model.fem")']
    for name in circuits:
        amps = key["current"] if name == task.get("circuit") else 0
        lines.append(f'mi_modifycircprop("{name}", 1, {float(amps)!r})')
    lines += ['mi_saveas("scored.fem")', "mi_analyze(1)", "mi_loadsolution()",
              f'print("{tag} elements " .. mo_numelements())']
    if task["measure"] in ("inductance", "flux_linkage"):
        lines += [f'I, V, PHI = mo_getcircuitproperties("{task["circuit"]}")',
                  f'print("{tag} current " .. format("%.12g", I))', f'print("{tag} flux " .. format("%.12g", PHI))']
    elif task["measure"] == "axial_force":
        lines += ["mo_groupselectblock()", f'print("{tag} coil_force " .. format("%.12g", mo_blockintegral(12)))',
                  "mo_clearblock()"]
    elif task["measure"] == "material_map":
        lines += ["function P(i, x, y)",
                  "  local a, b1, b2, sig, e, h1, h2, je, js, mu1 = mo_getpointvalues(x, y)",
                  f'  if mu1 then print("{tag} P " .. i .. " " .. format("%.6g", mu1) .. " " .. format("%.6g", sig))',
                  f'  else print("{tag} P " .. i .. " none") end', "end"]
        lines += [f"P({i}, {x!r}, {y!r})" for i, (x, y) in enumerate(points_for(key))]
    return lines + ["mo_close()", "mi_close()", f'print("{tag} done")']


def parse(output, nonce):
    tag = f"GYM {nonce} "
    values, points = {}, {}
    for line in output.splitlines():
        if not line.startswith(tag):
            continue
        parts = line[len(tag):].split()
        if parts[0] == "P" and len(parts) >= 3:
            points[int(parts[1])] = None if parts[2] == "none" else (float(parts[2]), float(parts[3]))
        elif len(parts) == 1:
            values[parts[0]] = True
        else:
            values[parts[0]] = float(parts[1])
    return values, points


# --- comparing with the key ---------------------------------------------------------

def measured_value(task, values):
    if task["measure"] == "inductance":
        return values["flux"] / values["current"] if values.get("current") else None
    if task["measure"] == "axial_force":
        return -values["coil_force"] if "coil_force" in values else None
    return values.get("flux")


def classify(sample, signatures, rel=1e-3):
    """Index of the reference material with this (permeability, conductivity), or -1."""
    if sample is None:
        return -1
    mu, sigma = sample
    for k, (m, s) in enumerate(signatures):
        if abs(mu - m) <= rel * abs(m) and abs(sigma - s) <= rel * abs(s) + 1e-12:
            return k
    return -1


def map_result(key, points):
    """Intersection-over-union of each reference material; points outside the reference mesh are ignored."""
    used = [i for i, c in enumerate(key["classes"]) if c != "-"]
    reference = [key["classes"][i] for i in used]
    found = [classify(points.get(i), key["signatures"]) for i in used]
    ious = {}
    for k in range(len(key["signatures"])):
        inside_ref = [c == str(k) for c in reference]
        inside_sub = [c == k for c in found]
        union = sum(a or b for a, b in zip(inside_ref, inside_sub))
        ious[k] = sum(a and b for a, b in zip(inside_ref, inside_sub)) / union if union else 1.0
    least, mean = min(ious.values()), sum(ious.values()) / len(ious)
    passed = least >= key["tolerance"]
    return {"passed": passed, "reward": 1.0 if passed else round(0.5 * mean, 4), "least_iou": round(least, 4),
            "mean_iou": round(mean, 4), "iou": {str(k): round(v, 4) for k, v in ious.items()},
            "unrecognised_points": found.count(-1)}


def value_result(key, value):
    reference = key["value"]
    ratio = abs(value - reference) / (key["tolerance"] * abs(reference))
    passed = ratio <= 1
    reward = 1.0 if passed else round(0.5 * max(0.0, 1 - math.log10(ratio)), 4)
    return {"passed": passed, "reward": reward, "value": value, "reference": reference,
            "relative_error": round((value - reference) / reference, 6), "error_ratio": round(ratio, 4)}


def evaluate(task, key, values, points):
    if task["measure"] == "material_map":
        return map_result(key, points)
    value = measured_value(task, values)
    if value is None or not math.isfinite(value):
        return None
    return value_result(key, value)


# --- running ------------------------------------------------------------------------

def _failed(record, stage, detail):
    return dict(record, stage=stage, detail=detail[-2000:], passed=False, reward=0.0)


def score(task, key, submission, work, timeout=SCORE_SECONDS):
    """Score one submission folder; the player's mistakes come back as a record, never an exception."""
    record = {"task": task["id"], "family": task["family"], "level": task["level"]}
    fem = Path(submission) / "model.fem"
    if not fem.is_file():
        return _failed(record, "no submission", "submission/model.fem is missing")
    if fem.stat().st_size > MAX_BYTES:
        return _failed(record, "contract", "model.fem is larger than 20 MB")
    text = fem.read_text(errors="replace")
    problems, circuits = screen(text, task)
    if problems:
        return _failed(record, "contract", "; ".join(problems))
    work = Path(work)
    work.mkdir(parents=True)
    shutil.copyfile(fem, work / "model.fem")
    nonce = secrets.token_hex(8)
    (work / "score.lua").write_text("\n".join(scoring_lines(task, key, circuits, nonce)) + "\n")
    code, output, seconds = engine.run_lua(work, "score.lua", timeout=timeout)
    (work / "score.out").write_text(output)
    values, points = parse(output, nonce)
    record.update(seconds=round(seconds, 2), elements=values.get("elements"))
    if code == 124:
        return _failed(record, "scoring timeout", output)
    if not values.get("done"):
        return _failed(record, "solve failed", output)
    result = evaluate(task, key, values, points)
    if result is None:
        return _failed(record, "no measurement", output)
    record.update(result, stage="scored")
    answer = Path(submission) / "answer.json"
    if answer.is_file():
        try:
            record["reported"] = float(json.loads(answer.read_text())["value"])
        except (ValueError, KeyError, TypeError):
            record["reported"] = "unreadable"
    return record


def build_lines(model, save="model.fem"):
    """Construction Lua for a typed model, saved under `save` and closed."""
    lines = compile_model(model)
    if lines[-1] != 'mi_saveas("built.fem")':
        raise ValueError("Unexpected end of the compiled model")
    return lines[:-1] + [f'mi_saveas("{save}")', "mi_close()"]


def run_model(model, task, key, work, timeout=300):
    """Build a typed model and measure it in one engine run (for references); return (values, points, seconds)."""
    work = Path(work)
    work.mkdir(parents=True)
    nonce = secrets.token_hex(8)
    circuits = [c["name"] for c in model.get("circuits", [])]
    script = build_lines(model) + scoring_lines(task, key, circuits, nonce)
    (work / "run.lua").write_text("\n".join(script) + "\n")
    code, output, seconds = engine.run_lua(work, "run.lua", timeout=timeout)
    (work / "run.out").write_text(output)
    values, points = parse(output, nonce)
    if code != 0 or not values.get("done"):
        raise RuntimeError(f"Reference run failed in {work} (exit {code}): {output[-800:]}")
    return values, points, seconds


def write_model(model, folder, timeout=300):
    """Save a typed model as folder/model.fem, as a player would (for the controls)."""
    folder = Path(folder)
    folder.mkdir(parents=True)
    (folder / "build.lua").write_text("\n".join(build_lines(model)) + "\n")
    code, output, _ = engine.run_lua(folder, "build.lua", timeout=timeout)
    (folder / "build.out").write_text(output)
    if code != 0 or not (folder / "model.fem").is_file():
        raise RuntimeError(f"Building {folder} failed (exit {code}): {output[-800:]}")
    return folder
