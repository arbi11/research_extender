"""Tasks, answer keys, the controls that check them, and the frozen test set.

A task is what the player sees (task.md, task.json). Its key is hidden: the generating
parameters and the reference result. Level 2 references are solved at mesh levels 0, 1, 2
(and 3 if needed); a key is kept only when the last two levels agree within a fifth of the
tolerance. Seeds come from (split, family, index), so the test split never meets a training
draw.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess

from . import engine, reward
from .families import FAMILIES, POINTS

TEST_SPLIT = "test-v1"
ROOT = Path(__file__).resolve().parent.parent


def seed_for(split, family, index, salt=""):
    digest = hashlib.sha256(f"{split}/{family}/{index}{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def make_task(family, index, split):
    """The public task and the parts of its key that need no solve."""
    seed = seed_for(split, family.name, index)
    params = family.sample(random.Random(seed))
    task = {"id": f"{family.name}-{index:03d}", "split": split, "family": family.name, "level": family.level,
            "title": family.title, "measure": family.measure, "unit": family.unit, "circuit": family.circuit,
            "submit": ["submission/model.fem"] + (["submission/answer.json"] if family.level == 2 else [])}
    key = {"id": task["id"], "family": family.name, "split": split, "seed": seed, "params": params,
           "tolerance": family.tolerance}
    if family.level == 1:
        key["points_seed"] = seed_for(split, family.name, index, "/points")
    else:
        key["current"] = family.current(params)
    return task, key, family.text(params)


def complete_key(task, key, work):
    """Solve the reference and fill in the key's result; return the key, or None when it does not converge."""
    family = FAMILIES[task["family"]]
    work = Path(work)
    if family.level == 1:
        values, points, seconds = reward.run_model(family.build(key["params"]), task, key, work / "reference")
        samples = [points.get(i) for i in range(POINTS)]
        signatures = sorted({s for s in samples if s is not None})
        if len(signatures) > 10:
            raise RuntimeError(f"{task['id']}: {len(signatures)} material signatures; classes are single digits")
        classes = "".join("-" if s is None else str(signatures.index(s)) for s in samples)
        return dict(key, signatures=[list(s) for s in signatures], classes=classes,
                    reference={"elements": values.get("elements"), "seconds": round(seconds, 2)})
    levels = []
    for level in range(4):
        values, _, seconds = reward.run_model(family.build(key["params"], level), task, key, work / f"level{level}")
        levels.append({"level": level, "value": reward.measured_value(task, values), "elements": values.get("elements"),
                       "seconds": round(seconds, 2)})
        if level >= 2:
            a, b = levels[-2]["value"], levels[-1]["value"]
            if abs(b - a) <= family.tolerance / 5 * abs(b):
                return dict(key, value=b, reference={"levels": levels, "converged": round(abs(b - a) / abs(b), 6)})
    return dict(key, value=None, reference={"levels": levels, "converged": None})


def controls(task, key, work):
    """Score two stand-in submissions: the same device drawn differently (must pass) and a wrong one (must fail)."""
    family = FAMILIES[task["family"]]
    work = Path(work)
    results = {}
    cases = {"same_auto": (key["params"], None, True), "wrong": (family.wrong(key["params"]), None if family.level == 1 else 1, False)}
    if family.level == 2:
        cases["same_level0"] = (key["params"], 0, True)
    for name, (params, level, variant) in cases.items():
        folder = reward.write_model(family.build(params, level, variant), work / name / "submission")
        results[name] = reward.score(task, key, folder, work / name / "score")
    return results


# --- the frozen test set -------------------------------------------------------------

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()


GENERATOR_FILES = ("femm_gym/__init__.py", "femm_gym/engine.py", "femm_gym/families.py", "femm_gym/reward.py",
                   "femm_gym/testset.py", "femm_brain/components.py", "femm_brain/reproduce.py", "femm_brain/geometry.py",
                   "femm_brain/data/iabc_tables.json")


def generator_digests():
    """SHA-256 of every file that decides a task or its key, so a frozen set names its exact code."""
    return {name: _sha(ROOT / name) for name in GENERATOR_FILES}


def freeze(out, per_family=30, families=None, work=None, log=print):
    """Generate the test split, solve every key, and write it with digests; never overwrites."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; a frozen set is never overwritten")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    work = Path(work or Path.home() / ".cache" / "femm-gym" / "work" / f"{TEST_SPLIT}-{stamp}")
    (out / "tasks").mkdir(parents=True)
    (out / "keys").mkdir()
    manifest = {"split": TEST_SPLIT, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "generator_files": generator_digests(), "git_head": _git("rev-parse", "HEAD"),
                "git_uncommitted": bool(_git("status", "--porcelain", "--", *GENERATOR_FILES)),
                "image": engine.PINNED_IMAGE, "per_family": per_family, "families": {}, "tasks": [], "dropped": []}
    for name in families or sorted(FAMILIES):
        family = FAMILIES[name]
        manifest["families"][name] = {"level": family.level, "measure": family.measure, "unit": family.unit,
                                      "tolerance": family.tolerance}
        for index in range(per_family):
            task, key, text = make_task(family, index, TEST_SPLIT)
            key = complete_key(task, key, work / task["id"])
            if family.level == 2 and key["value"] is None:
                manifest["dropped"].append({"id": task["id"], "reason": "reference did not converge",
                                            "levels": key["reference"]["levels"]})
                log(f"{task['id']}: dropped, reference did not converge")
                continue
            files = {"task.json": out / "tasks" / f"{task['id']}.json", "task.md": out / "tasks" / f"{task['id']}.md",
                     "key.json": out / "keys" / f"{task['id']}.json"}
            files["task.json"].write_text(json.dumps(task, indent=1) + "\n")
            files["task.md"].write_text(text)
            files["key.json"].write_text(json.dumps(key, indent=1) + "\n")
            manifest["tasks"].append({"id": task["id"], **{k: _sha(v) for k, v in files.items()}})
            log(f"{task['id']}: " + (f"{key['value']:.6g} {family.unit}" if family.level == 2
                                     else f"{len(key['signatures'])} materials"))
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    digests = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "FROZEN.sha256")
    (out / "FROZEN.sha256").write_text("".join(f"{_sha(p)}  {p.relative_to(out)}\n" for p in digests))
    return manifest


def verify(out):
    """Files of a frozen set that no longer match its digests, and generator files that changed since."""
    out = Path(out)
    bad = []
    for line in (out / "FROZEN.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = out / name
        if not path.is_file() or _sha(path) != digest:
            bad.append(name)
    frozen = json.loads((out / "MANIFEST.json").read_text())["generator_files"]
    current = generator_digests()
    return bad, sorted(name for name in frozen if current.get(name) != frozen[name])


def load(out, task_id):
    out = Path(out)
    return (json.loads((out / "tasks" / f"{task_id}.json").read_text()),
            json.loads((out / "keys" / f"{task_id}.json").read_text()))
