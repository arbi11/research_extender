"""Level 3 tasks, the frozen Level 3 set and its self-check (see LEVEL3.md).

A task folder holds what the player gets: task.md (the case section), task.json (the benchmark's
case format) and paper/paper.md. Its key is hidden: the machine, how the paper presents each
input, the expected provenance of every input, the reference values and the benchmark case files
(spec.json and drive.json). Seeds come from (split, family, index), as in Levels 1 and 2.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import threading
import traceback

from . import bench, engine, level3, level3_score, reward
from .testset import ROOT, _git, _sha, seed_for

L3_SPLIT = "l3-test-v1"
CHECK_SPLIT = "l3-check"
MIN_HIDDEN = 4
GENERATOR_FILES = ("femm_gym/__init__.py", "femm_gym/engine.py", "femm_gym/families.py", "femm_gym/reward.py",
                   "femm_gym/testset.py", "femm_gym/bench.py", "femm_gym/level3.py", "femm_gym/level3_score.py",
                   "femm_gym/level3_set.py", "femm_brain/components.py", "femm_brain/reproduce.py",
                   "femm_brain/geometry.py", "femm_brain/data/library_materials.json")


def task_id(index):
    return f"{level3.FAMILY}-{index:03d}"


def draw(index, split):
    """The machine and the paper's presentation plan for one task; at least MIN_HIDDEN inputs are hidden."""
    seed = seed_for(split, level3.FAMILY, index)
    rng = random.Random(seed)
    machine = level3.sample(rng)
    while True:
        plan = level3.presentation(machine, rng)
        if hidden_count(level3.expectations(dict(machine, emf_printed=0, torque_printed=0), plan)) >= MIN_HIDDEN:
            return seed, dict(machine, id=task_id(index)), plan


def hidden_count(expectations):
    return sum(1 for e in expectations.values() if e["expect"] != "quoted")


def make(index, split, work, bench_path=None):
    """Solve one task's key; returns (task.json, key, task.md)."""
    seed, machine, plan = draw(index, split)
    tid = machine["id"]
    key = level3_score.make_key(tid, machine, plan, Path(work) / tid, bench_path)
    key.update(split=split, seed=seed)
    drive = level3.drive_doc(key["machine"])
    drive["case"] = tid
    key["case"] = {"task.json": level3.task_json(tid, key["machine"]),
                   "spec.json": level3.spec_doc(tid, key["machine"], key["values"]), "drive.json": drive}
    return key["case"]["task.json"], key, level3.task_md(key["machine"])


def write_task(out, task, key, text):
    """The player's task folder and the hidden key; returns their digests."""
    folder = Path(out) / "tasks" / key["id"]
    (folder / "paper").mkdir(parents=True)
    (Path(out) / "keys").mkdir(exist_ok=True)
    files = {"task.json": folder / "task.json", "task.md": folder / "task.md", "paper.md": folder / "paper" / "paper.md",
             "key.json": Path(out) / "keys" / f"{key['id']}.json"}
    files["task.json"].write_text(json.dumps(task, indent=1) + "\n")
    files["task.md"].write_text(text)
    files["paper.md"].write_text(key["paper"])
    files["key.json"].write_text(json.dumps(key, indent=1) + "\n")
    return {name: _sha(path) for name, path in files.items()}


def generator_digests():
    return {name: _sha(ROOT / name) for name in GENERATOR_FILES}


def freeze(out, count=24, jobs=3, work=None, bench_path=None, log=print):
    """Generate the Level 3 test split, solve every key, and write it with digests; never overwrites."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; a frozen set is never overwritten")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    work = Path(work or Path.home() / ".cache" / "femm-gym" / "work" / f"{L3_SPLIT}-{stamp}")
    (out / "tasks").mkdir(parents=True)
    (out / "keys").mkdir()
    bench.scorer(bench_path)   # load once, before the threads
    manifest = {"split": L3_SPLIT, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "generator_files": generator_digests(), "scorer_files": bench.digests(bench_path),
                "scorer_folder": str(bench.folder(bench_path)), "git_head": _git("rev-parse", "HEAD"),
                "git_uncommitted": bool(_git("status", "--porcelain", "--", *GENERATOR_FILES)),
                "image": engine.PINNED_IMAGE, "count": count, "family": level3.FAMILY, "tasks": [], "dropped": []}
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(make, index, L3_SPLIT, work, bench_path): index for index in range(count)}
        for future in as_completed(futures):
            tid = task_id(futures[future])
            try:
                task, key, text = future.result()
            except Exception as error:  # noqa: BLE001 - a task that cannot be keyed is dropped, with its reason
                with lock:
                    manifest["dropped"].append({"id": tid, "reason": f"{type(error).__name__}: {error}",
                                                "trace": traceback.format_exc()[-1500:]})
                log(f"{tid}: dropped ({error})")
                continue
            with lock:
                manifest["tasks"].append({"id": tid, **write_task(out, task, key, text)})
            log(f"{tid}: {key['machine']['slots']}/{key['machine']['poles']}, "
                f"E01 {key['values']['E01']:.4g} V, T01 {key['values']['T01']:.4g}, T02 {key['values']['T02']:.4g} N m, "
                f"{hidden_count(key['expectations'])} hidden")
    manifest["tasks"].sort(key=lambda t: t["id"])
    manifest["dropped"].sort(key=lambda t: t["id"])
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    digests = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "FROZEN.sha256")
    (out / "FROZEN.sha256").write_text("".join(f"{_sha(p)}  {p.relative_to(out)}\n" for p in digests))
    return manifest


def verify(out, bench_path=None):
    """Frozen files that changed, generator files that changed, and benchmark scorer files that changed."""
    out = Path(out)
    bad = []
    for line in (out / "FROZEN.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = out / name
        if not path.is_file() or _sha(path) != digest:
            bad.append(name)
    manifest = json.loads((out / "MANIFEST.json").read_text())
    current = generator_digests()
    scorer = bench.digests(bench_path)
    return (bad, sorted(n for n in manifest["generator_files"] if current.get(n) != manifest["generator_files"][n]),
            sorted(n for n in manifest["scorer_files"] if scorer.get(n) != manifest["scorer_files"][n]))


def load(out, tid):
    """(task folder, key) of a frozen Level 3 task."""
    out = Path(out)
    return out / "tasks" / tid, json.loads((out / "keys" / f"{tid}.json").read_text())


# --- self-check --------------------------------------------------------------------------------------

def _submission(folder, key, level, variant=False, change=None, inputs=None):
    """A submission a player could make: the drawn model, its entities and inputs.json."""
    folder = Path(folder)
    model = reward.write_model(level3.build(key["machine"], level, variant, change), folder)
    ref = key["reference"]
    (model / "entities.json").write_text(json.dumps(level3.entities(dict(key["machine"], **(change or {})),
                                                                    ref["zero"], ref["sense"], variant), indent=1) + "\n")
    (model / "inputs.json").write_text(json.dumps(inputs or level3_score.reference_inputs(key), indent=1) + "\n")
    for name in ("build.lua", "build.out"):
        (model / name).unlink(missing_ok=True)
    return model


def wrong_magnet(machine):
    magnet = machine["magnet"]
    if magnet.get("grade"):
        grades = level3.MAGNET_GRADES
        i = grades.index(magnet["grade"])
        return {"magnet": {"grade": grades[i - 2] if i >= 2 else grades[i + 2]}}
    return {"magnet": dict(magnet, br=round(magnet["br"] * 1.06, 3))}


def flag_controls(key):
    """Flag credit for honest, all-assumed and invented-quote inputs (no solve needed)."""
    honest = level3_score.reference_inputs(key)
    assumed = {"inputs": [dict(i, provenance="assumed", reason="not checked") for i in honest["inputs"]]}
    invented = {"inputs": [dict(i, provenance="quoted", quote=f"{i['name']} is {i['value']}", location="Table 1")
                           for i in honest["inputs"]]}
    return {name: level3_score.flag_credit(doc, key["expectations"], key["paper"])[0]
            for name, doc in (("honest", honest), ("all_assumed", assumed), ("invented_quotes", invented))}


REPRODUCE = 1e-3   # the reference drawing, scored as a submission, must give its own level-1 values to 0.1%


def _controls(key, work, bench_path, log):
    """The four solved controls for one key."""
    controls = {}
    for name, kwargs, should_pass in (
            ("reference", {"level": 1}, True),
            ("variant", {"level": 1, "variant": True}, True),
            ("wrong_magnet", {"level": 1, "change": wrong_magnet(key["machine"])}, False),
            ("wrong_turns", {"level": 1, "change": {"turns_per_coil": round(key["machine"]["turns_per_coil"] * 1.15)}},
             False)):
        folder = Path(work) / key["id"] / "controls" / name
        submission = _submission(folder / "submission", key, **kwargs)
        record = level3_score.score(key, submission, folder / "score", bench_path)
        controls[name] = {"expected": "pass" if should_pass else "fail", "passed": record["passed"],
                          "reward": record["reward"], "credit": record.get("credit"),
                          "targets": record.get("targets"), "detail": record.get("detail")}
        log(f"{key['id']} {name}: {'ok' if record['passed'] == should_pass else 'WRONG'} "
            f"(reward {record['reward']}, credit {record.get('credit')})")
    return controls


def _reproduces(key, controls):
    """How far the reference control's values are from the key's own level-1 run (relative)."""
    own = next((r["values"] for r in key["reference"]["runs"] if r["level"] == 1), None)
    got = {t["id"]: t["computed"] for t in controls["reference"].get("targets") or []}
    if own is None or set(got) != set(own) or any(v is None for v in got.values()):
        return None
    return max(abs(got[t] - own[t]) / abs(own[t]) for t in own)


def selfcheck(count=1, split=CHECK_SPLIT, work=None, bench_path=None, log=print, jobs=1):
    """Controls on fresh tasks: the reference and a variant drawing must pass, wrong machines must fail.

    The tasks and keys are written under `work` in a frozen set's layout (tasks/, keys/), so
    `l3-play --testset WORK --task ID` can play exactly the tasks whose controls were checked.
    """
    work = Path(work or Path.home() / ".cache" / "femm-gym" / "work" /
                f"{split}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}")
    bench.scorer(bench_path)   # load once, before the threads

    def one(index):
        task, key, text = make(index, split, work, bench_path)
        write_task(work, task, key, text)   # kept, so a check can play the tasks the controls verified
        absent = {name: check["insensitive"] for name, check in key["absent_checks"].items()}
        log(f"{key['id']}: keyed at level {key['reference']['level']}, values {key['values']}, "
            f"absent inputs insensitive: {absent}")
        flags = flag_controls(key)
        controls = _controls(key, work, bench_path, log)
        drift = _reproduces(key, controls)
        ok = (all(c["passed"] == (c["expected"] == "pass") for c in controls.values())
              and drift is not None and drift <= REPRODUCE
              and flags["honest"] == 1.0 and flags["all_assumed"] < 0.5 and flags["invented_quotes"] < 0.5)
        log(f"{key['id']} flags: {flags}; reference reproduces its level-1 run to {drift}")
        return {"id": key["id"], "ok": ok, "controls": controls, "flags": flags, "reproduction": drift,
                "absent_checks": key["absent_checks"], "reference": key["reference"]}

    results = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(one, index): index for index in range(count)}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:  # noqa: BLE001 - reported as a failed check, with its reason
                tid = task_id(futures[future])
                log(f"{tid}: could not be checked ({type(error).__name__}: {error})")
                results.append({"id": tid, "ok": False, "error": f"{type(error).__name__}: {error}",
                                "trace": traceback.format_exc()[-1500:]})
    results.sort(key=lambda r: r["id"])
    (work / "selfcheck.json").write_text(json.dumps(results, indent=1) + "\n")
    return results
