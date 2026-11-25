"""The `daro_bench` scorer, used read-only from its own folder.

Level 3 tasks are benchmark cases, so they are scored by the benchmark's `score.py` itself. It
is imported from the benchmark folder (`--bench` or `$DARO_BENCH`) without writing bytecode
there, and its digests go into every Level 3 freeze. One thing changes: its FEMM runs use the
gym's low-priority container (one CPU at a low weight), so a long freeze cannot starve other
FEMM work. The scoring logic is the benchmark's.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .engine import PINNED_IMAGE

ENV = "DARO_BENCH"
DEFAULT = Path.home() / "code" / "research_extender_daro" / "daro_bench"
# The benchmark files that decide a Level 3 score.
SCORER_FILES = ("score.py", "stats.py", "engine.py", "sandbox_bin/femm_contract.py")
_LOADED = {}


def folder(path=None):
    """The benchmark folder: `path`, else $DARO_BENCH, else the usual checkout."""
    root = Path(path or os.environ.get(ENV) or DEFAULT).expanduser().resolve()
    if not (root / "score.py").is_file() or not (root / "sandbox_bin" / "femm_contract.py").is_file():
        raise FileNotFoundError(f"{root} is not a daro_bench folder (no score.py); give --bench or set ${ENV}")
    return root


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digests(path=None):
    root = folder(path)
    return {name: _sha(root / name) for name in SCORER_FILES}


def run_lua(folder, script, timeout=600, image=PINNED_IMAGE):
    """The benchmark's engine call with the gym's priority: no network, read-only root, 1 CPU at a low weight, 2 GB."""
    folder = Path(folder).resolve()
    name = "femm-gym-l3-" + uuid.uuid4().hex[:12]
    args = ["docker", "run", "--rm", "--init", "--name", name, "--network", "none", "--read-only",
            "--memory", "2g", "--cpus", "1", "--cpu-shares", "128", "--pids-limit", "256",
            "--tmpfs", "/tmp:rw,size=512m", "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "HOME=/tmp", "-e", "QT_QPA_PLATFORM=offscreen", "-e", "FEMM_DIR=/opt/femm/bin",
            "-v", f"{folder}:/work", "-w", "/work", image,
            "/opt/femm/bin/femm-lua", "-h", "-p", "48015", Path(script).name]
    started = time.time()
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        code, output = done.returncode, done.stdout + done.stderr
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)
        code, output = 124, f"FEMM exceeded the {timeout} s time limit and was stopped.\n"
    return code, output, time.time() - started


def scorer(path=None):
    """The benchmark's score module, loaded once per folder with the gym's engine call."""
    root = folder(path)
    if root not in _LOADED:
        before = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec = importlib.util.spec_from_file_location("daro_bench_score", root / "score.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # it puts its folder and sandbox_bin on sys.path itself
        finally:
            sys.dont_write_bytecode = before
        if module.PINNED_IMAGE != PINNED_IMAGE:
            raise RuntimeError(f"the benchmark pins image {module.PINNED_IMAGE}, the gym {PINNED_IMAGE}")
        module.run_lua = run_lua
        _LOADED[root] = module
    return _LOADED[root]


def score(case_dir, trial_dir, out_dir, path=None, keep_work=False):
    """The benchmark's score record for a trial folder holding workspace/submission."""
    return scorer(path).score(case_dir, trial_dir, out_dir, PINNED_IMAGE, keep_work)


def check_submission(submission, task, path=None):
    """The benchmark's contract check (the one femm-check runs); returns (report, models, entities)."""
    scorer(path)
    return sys.modules["femm_contract"].check_submission(submission, task)
