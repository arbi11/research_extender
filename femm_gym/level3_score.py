"""Level 3 keys, reward and controls, on the benchmark's own scorer (see LEVEL3.md)."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil

from . import bench, level3, reward

REFERENCE_INPUTS = {"inputs": [{"name": "reference", "value": 1, "unit": "", "provenance": "assumed",
                                "reason": "the gym's own reference model"}]}


def write_case(folder, task_id, machine, targets=tuple(level3.TARGETS), values=None):
    """A benchmark case folder (task.json, drive.json, spec.json) for the given targets."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files = {"task.json": level3.task_json(task_id, machine), "drive.json": level3.drive_doc(machine, targets),
             "spec.json": level3.spec_doc(task_id, machine, values, targets)}
    for name, doc in files.items():
        (folder / name).write_text(json.dumps(doc, indent=1) + "\n")
    return folder


def write_trial(folder, fem, entities, inputs=None):
    """A trial folder in the benchmark's layout: workspace/submission with the model and its files."""
    submission = Path(folder) / "workspace" / "submission"
    submission.mkdir(parents=True)
    shutil.copyfile(fem, submission / "model.fem")
    (submission / "entities.json").write_text(json.dumps(entities, indent=1) + "\n")
    (submission / "inputs.json").write_text(json.dumps(inputs or REFERENCE_INPUTS, indent=1) + "\n")
    return Path(folder)


def zero_and_sense(machine, phase_a, b_minus_a):
    """The motion that puts phase A's no-load flux-linkage peak at p = 0, and the positive direction."""
    pp = machine["pole_pairs"]
    if abs(abs(b_minus_a) - 120) > 10:
        raise RuntimeError(f"phase B's flux linkage is {b_minus_a:.1f} electrical degrees from A's, not 120")
    return round((-phase_a / pp) % (360 / pp), 6), (1 if b_minus_a < 0 else -1)


def _computed(record, label):
    values = {}
    for target in record["targets"]:
        if target.get("computed") is None:
            drives = {d: (i.get("status"), i.get("errors")) for d, i in record["drives"].items()}
            raise RuntimeError(f"{label}: {target['id']} has no result ({target.get('reason')}); drives {drives}; "
                               f"contract {record.get('contract', {}).get('errors')}")
        values[target["id"]] = target["computed"]
    return values


def solve(machine, work, level=1, variant=False, change=None, task_id="l3", bench_path=None):
    """E01, T01 and T02 of the machine as `level3.build` draws it, measured by the benchmark's scorer.

    The no-load sweep runs first; its phase-A flux-linkage phase and the B-to-A phase give the
    zero angle and sense, as a player must find them, and the load sweeps use those. The case
    (drives and checks) is always the given machine's; `change` alters only the drawn model.
    """
    work = Path(work)
    if work.exists():
        shutil.rmtree(work)
    drawn = dict(machine, **(change or {}))
    fem = reward.write_model(level3.build(machine, level, variant, change), work / "model") / "model.fem"
    record = bench.score(write_case(work / "case-nl", task_id, machine, ("E01",)),
                         write_trial(work / "trial-nl", fem, level3.entities(drawn, 0.0, 1, variant)),
                         work / "score-nl", bench_path)
    values = _computed(record, "no-load sweep")
    diagnostics = record["diagnostics"]
    zero, sense = zero_and_sense(machine, diagnostics["nl_phase_A_deg"], diagnostics["nl_B_minus_A_deg"])
    seconds = sum(d.get("seconds", 0) for d in record["drives"].values())
    record = bench.score(write_case(work / "case-ld", task_id, machine, ("T01", "T02")),
                         write_trial(work / "trial-ld", fem, level3.entities(drawn, zero, sense, variant)),
                         work / "score-ld", bench_path)
    values.update(_computed(record, "load sweeps"))
    seconds += sum(d.get("seconds", 0) for d in record["drives"].values())
    return {"level": level, "variant": variant, "change": change or {}, "values": values, "zero": zero, "sense": sense,
            "phase_A_deg": diagnostics["nl_phase_A_deg"], "B_minus_A_deg": diagnostics["nl_B_minus_A_deg"],
            "solver_seconds": round(seconds, 1)}


def agree(a, b, share):
    """Every target of `b` within `share` of its tolerance of `a`."""
    return all(abs(b[t] - a[t]) <= share * level3.REL_TOL * abs(a[t]) for t in a)


def reference(machine, work, task_id="l3", bench_path=None, levels=(0, 1, 2), max_level=3):
    """Solve the reference at increasing mesh levels until the last two agree within a fifth of the tolerance."""
    runs = [solve(machine, Path(work) / f"level{lv}", lv, task_id=task_id, bench_path=bench_path) for lv in levels]
    while not agree(runs[-1]["values"], runs[-2]["values"], 0.2) and runs[-1]["level"] < max_level:
        lv = runs[-1]["level"] + 1
        runs.append(solve(machine, Path(work) / f"level{lv}", lv, task_id=task_id, bench_path=bench_path))
    return {"converged": agree(runs[-1]["values"], runs[-2]["values"], 0.2), "runs": runs, "key": runs[-1]["values"]}


# --- keys --------------------------------------------------------------------------------------

INSENSITIVE_SHARE = 0.4   # an absent input's range ends must keep every target within this share of its tolerance


def range_ends(machine, name):
    """The two ends of the plausible range of an input the paper may leave out."""
    if name == "tooth_tip_depth":
        return list(level3.RANGES["tooth_tip_depth"])
    if name == "shaft_radius":
        low, high = level3.RANGES["shaft_radius"]
        return [low * machine["rotor_core_radius"], round(high * machine["rotor_core_radius"] * 2) / 2]
    if name == "rotor_steel":
        laminated = machine["rotor_steel"] in level3.LAMINATIONS
        return list(level3.RANGES["rotor_lamination" if laminated else "rotor_steel"])
    raise ValueError(name)


def insensitive(machine, name, base, work, bench_path=None):
    """Solve the reference (level 0) at both ends of an absent input's range; True if no target moves much."""
    ends = []
    for k, value in enumerate(range_ends(machine, name)):
        try:
            run = solve(machine, Path(work) / f"{name}-{k}", 0, change={name: value}, bench_path=bench_path)
        except ValueError as error:   # the drawing does not exist at this end of the range
            ends.append({"value": value, "error": str(error)})
            continue
        ends.append({"value": value, "values": run["values"],
                     "share": max(abs(run["values"][t] - base[t]) / (level3.REL_TOL * abs(base[t])) for t in base)})
    ok = all("share" in e and e["share"] <= INSENSITIVE_SHARE for e in ends)
    return ok, ends


def make_key(task_id, machine, plan, work, bench_path=None):
    """Solve the reference, check the absent inputs, then fix the paper and the case. Returns the key."""
    work = Path(work)
    ref = reference(machine, work / "reference", task_id, bench_path)
    if not ref["converged"]:
        raise RuntimeError(f"{task_id}: the reference did not converge: {[r['values'] for r in ref['runs']]}")
    base = next(r["values"] for r in ref["runs"] if r["level"] == 0)
    checks = {}
    for name in ("tooth_tip_depth", "shaft_radius", "rotor_steel"):
        if plan[name] == "absent":
            ok, ends = insensitive(machine, name, base, work / "insensitive", bench_path)
            checks[name] = {"insensitive": ok, "ends": ends}
            if not ok:
                plan = dict(plan, **{name: "stated"})   # it matters, so the paper states it
    finest = ref["runs"][-1]
    published = level3.publish(machine, ref["key"], plan)
    text, quotes = level3.write_paper(published, plan)
    return {"id": task_id, "family": level3.FAMILY, "level": level3.LEVEL, "machine": published, "plan": plan,
            "values": ref["key"], "expectations": level3.expectations(published, plan),
            "quotes": {k: list(v) for k, v in quotes.items()}, "paper": text,
            "reference": {"zero": finest["zero"], "sense": finest["sense"], "level": finest["level"],
                          "runs": [{k: r[k] for k in ("level", "values", "zero", "sense", "phase_A_deg",
                                                      "B_minus_A_deg", "solver_seconds")} for r in ref["runs"]]},
            "absent_checks": checks}


def reference_inputs(key):
    """The inputs.json a faithful player writes: quotes for what the paper states, the rest derived or assumed."""
    items = []
    for name, exp in key["expectations"].items():
        item = {"name": name, "value": exp["truth"], "unit": exp["unit"]}
        if exp["expect"] == "quoted":
            quote, location = key["quotes"][name]
            item.update(provenance="quoted", quote=quote, location=location)
        elif exp["expect"] == "assumed":
            item.update(provenance="assumed", reason="the paper does not give it; a typical value")
        else:
            item.update(provenance="derived", expression="from the paper's stated values")
        items.append(item)
    return {"inputs": items}


# --- the reward --------------------------------------------------------------------------------

CREDIT = {"derived": {"derived": 1.0, "assumed": 0.5, "figure": 0.5},
          "assumed": {"assumed": 1.0, "derived": 0.5, "figure": 0.5},
          "library": {"derived": 1.0, "assumed": 1.0, "figure": 0.5}}
SIGNED_COIL = re.compile(r"(?<![a-z0-9])[+-][abc](?![a-z0-9])")
NUMBER_TEXT = re.compile(r"(?<![a-z])[-+]?\d*\.?\d+(?:e[-+]?\d+)?")
MAX_FILE_BYTES = {".fem": 64 * 1024 * 1024}
MAX_OTHER_BYTES = 2 * 1024 * 1024
MAX_FILES = 20


def normalize(text):
    """Text as quotes are compared: lower case, plain dashes, no table or emphasis marks, single spaces."""
    text = str(text).lower()
    for a, b in (("−", "-"), ("–", "-"), ("—", "-"), ("×", "x"), (" ", " ")):
        text = text.replace(a, b)
    return " ".join(re.sub(r"[|*_`]", " ", text).split())


def _same_number(a, b):
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


def value_matches(value, truth, name=""):
    if name == "coil_sequence":
        return SIGNED_COIL.findall(normalize(value)) == SIGNED_COIL.findall(normalize(truth))
    if isinstance(truth, str):
        return isinstance(value, str) and normalize(value) == normalize(truth)
    try:
        return _same_number(float(value), float(truth))
    except (TypeError, ValueError):
        return False


def quote_holds(item, paper, name=""):
    """The quote is copied from the paper and contains the recorded value."""
    quote = item.get("quote")
    if not isinstance(quote, str) or not quote.strip() or normalize(quote) not in normalize(paper):
        return False
    q, value = normalize(quote), item.get("value")
    if name == "coil_sequence":
        tokens, wanted = SIGNED_COIL.findall(q), SIGNED_COIL.findall(normalize(value))
        return bool(wanted) and any(tokens[i:i + len(wanted)] == wanted for i in range(len(tokens)))
    if isinstance(value, str):
        return normalize(value) in q
    try:
        return any(_same_number(float(n), float(value)) for n in NUMBER_TEXT.findall(q))
    except (TypeError, ValueError):
        return False


def flag_credit(inputs_doc, expectations, paper):
    """Credit for recording which inputs the paper states and which it hides (LEVEL3.md)."""
    items = {}
    for item in (inputs_doc or {}).get("inputs", []) if isinstance(inputs_doc, dict) else []:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            items.setdefault(item["name"], item)
    hidden_credit, hidden, misses, rows = 0.0, 0, 0, {}
    for name, expect in expectations.items():
        item = items.get(name)
        kind = item.get("provenance") if item else None
        if expect["expect"] == "quoted":
            ok = (kind == "quoted" and quote_holds(item, paper, name)
                  and value_matches(item.get("value"), expect["truth"], name))
            misses += 0 if ok else 1
            rows[name] = {"expect": "quoted", "recorded": kind, "credit": 1.0 if ok else 0.0}
        else:
            credit = CREDIT[expect["expect"]].get(kind, 0.0)
            hidden += 1
            hidden_credit += credit
            rows[name] = {"expect": expect["expect"], "recorded": kind, "credit": credit}
    total = hidden + misses
    return (hidden_credit / total if total else 1.0), rows


def target_credit(record):
    acceptance = [t for t in record.get("targets", []) if t["role"] == "acceptance"]
    if acceptance and all(t.get("status") == "pass" for t in acceptance):
        return 1.0
    credits = []
    for t in acceptance:
        if t.get("status") == "pass":
            credits.append(1.0)
        elif t.get("ratio") is not None and t["ratio"] > 0:
            credits.append(max(0.0, 1.0 - math.log10(t["ratio"])))
        else:
            credits.append(0.0)
    return 0.5 * sum(credits) / len(credits) if credits else 0.0


def conformance_credit(record):
    checks = record.get("conformance", [])
    return sum(1 for c in checks if c.get("status") == "ok") / len(checks) if checks else 1.0


def take_submission(source, target):
    """Copy the submission's regular files for the scorer; links and oversized files are problems."""
    source, target = Path(source), Path(target)
    target.mkdir(parents=True)
    problems = []
    if not source.is_dir() or source.is_symlink():
        return target, ["submission/ is missing or a link"]
    entries = sorted(source.iterdir())
    if len(entries) > MAX_FILES:
        problems.append(f"submission/ holds more than {MAX_FILES} entries")
    for path in entries[:MAX_FILES]:
        if path.is_symlink() or not path.is_file():
            if path.is_symlink() or path.exists():
                problems.append(f"submission/{path.name} is a link or not a regular file")
            continue
        limit = MAX_FILE_BYTES.get(path.suffix.lower(), MAX_OTHER_BYTES)
        if path.stat().st_size > limit:
            problems.append(f"submission/{path.name} is larger than {limit // 1_000_000} MB")
            continue
        shutil.copyfile(path, target / path.name)
    return target, problems


def score(key, submission, work, bench_path=None, evidence=None):
    """The reward record for a Level 3 submission folder (model, entities.json, inputs.json)."""
    work = Path(work)
    if work.exists():
        shutil.rmtree(work)
    trial = work / "trial"
    taken, problems = take_submission(submission, trial / "workspace" / "submission")
    record = {"task": key["id"], "family": key["family"], "level": key["level"]}
    for fem in sorted(taken.glob("*.fem")):
        found, _ = reward.screen(fem.read_text(errors="replace"), {})
        problems += [f"{fem.name}: {p}" for p in found]
    if problems:
        return dict(record, stage="contract", detail="; ".join(problems), passed=False, reward=0.0)
    if evidence and Path(evidence).is_dir():
        shutil.copytree(evidence, trial / "evidence", symlinks=True)
    case = work / "case"
    case.mkdir(parents=True)
    for name in ("task.json", "spec.json", "drive.json"):
        (case / name).write_text(json.dumps(key["case"][name], indent=1) + "\n")
    result = bench.score(case, trial, work / "score", bench_path)
    try:
        inputs_doc = json.loads((taken / "inputs.json").read_text())
    except (OSError, ValueError):
        inputs_doc = None
    flags, rows = flag_credit(inputs_doc, key["expectations"], key["paper"])
    targets, conformance = target_credit(result), conformance_credit(result)
    value = round(0.7 * targets + 0.15 * conformance + 0.15 * flags, 6)
    passed = targets == 1.0 and conformance == 1.0 and flags >= 0.999
    return dict(record, stage=result["stage"], passed=passed, reward=1.0 if passed else min(value, 0.999),
                targets=[{k: t.get(k) for k in ("id", "status", "computed", "value", "rel_error", "ratio", "reason")}
                         for t in result["targets"]],
                conformance=[{k: c.get(k) for k in ("kind", "value", "found", "status")}
                             for c in result.get("conformance", [])],
                credit={"targets": targets, "conformance": conformance, "flags": round(flags, 6)},
                flags=rows, contract_errors=result.get("contract", {}).get("errors", []),
                diagnostics=result.get("diagnostics"), drives={d: {k: i.get(k) for k in ("status", "states", "seconds")}
                                                                for d, i in result.get("drives", {}).items()})
