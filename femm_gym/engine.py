"""Run a FEMM Lua script in the pinned engine container, at low priority.

The container has no network, a read-only root, one CPU at a low scheduling weight and
768 MB of memory, and sees only the one folder holding the script. The low weight means it
cannot starve other FEMM work on the machine, such as a benchmark trial running alongside.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time
import uuid

# research-extender-femm:dev-hierarchy-resumed-20261025, the image the benchmarks use.
PINNED_IMAGE = "sha256:85f07dce379bff58dc539f8ed49ffba53a94eba332fbf81d4c9fd6c3df94725d"


def run_lua(folder, script, timeout=300, image=PINNED_IMAGE):
    """Run `script` (a file name inside `folder`) with femm-lua; return (returncode, output, seconds)."""
    folder = Path(folder).resolve()
    name = "femm-gym-" + uuid.uuid4().hex[:12]
    args = ["docker", "run", "--rm", "--init", "--name", name, "--network", "none", "--read-only",
            "--memory", "768m", "--cpus", "1", "--cpu-shares", "128", "--pids-limit", "128",
            "--tmpfs", "/tmp:rw,size=256m", "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "HOME=/tmp", "-e", "QT_QPA_PLATFORM=offscreen", "-e", "FEMM_DIR=/opt/femm/bin",
            "-v", f"{folder}:/work", "-w", "/work", image,
            "/opt/femm/bin/femm-lua", "-h", "-p", "48015", Path(script).name]
    started = time.time()
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        code, output = done.returncode, done.stdout + done.stderr
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)
        code, output = 124, f"FEMM exceeded the {timeout} s limit and was stopped.\n"
    return code, output, time.time() - started
