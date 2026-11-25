"""Run Python with pyFEMM in the pinned FEMM engine container and collect the results.

  python3 -I femm_brain_daro_v2/verify/run_femm.py inventory
  python3 -I femm_brain_daro_v2/verify/run_femm.py batch NAME PY_FILE...
  python3 -I femm_brain_daro_v2/verify/run_femm.py entries DRAFT.json [--only NAME...]
  python3 -I femm_brain_daro_v2/verify/run_femm.py tests [--only NAME...]

`batch` puts py/harness.py and py/fixtures.py in front of the given files, starts FEMM with
openfemm(1), runs the result and prints its RESULT lines. `entries` runs the tests of a draft
list of entries without touching the catalogue. `tests` builds batches from the tests in
catalogue.json, runs them and writes each outcome back into the catalogue (see AUTHORING.md).

Scripts run with the image's own Python 3 (/opt/venv/bin/python3), which reaches FEMM through
femm-qt's pyFEMM over a TCP port inside the container. Every run uses the image the benchmarks
use, with no network, a read-only root, one CPU at low priority and 768 MB of memory, so it
cannot starve other FEMM work on the machine.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
PY = HERE / "py"
CATALOGUE = ROOT / "catalogue.json"
IMAGE = "sha256:85f07dce379bff58dc539f8ed49ffba53a94eba332fbf81d4c9fd6c3df94725d"
IMAGE_TAG = "research-extender-femm:dev-hierarchy-resumed-20261025"
PYTHON = "/opt/venv/bin/python3"


def run_python(script: Path, timeout: int = 600) -> tuple[int, str, float]:
    """Run one Python file in the engine; return (code, output, seconds)."""
    name = "v2-femm-" + uuid.uuid4().hex[:10]
    args = ["docker", "run", "--rm", "--init", "--name", name, "--network", "none", "--read-only",
            "--memory", "768m", "--cpus", "1", "--cpu-shares", "128", "--pids-limit", "128",
            "--tmpfs", "/tmp:rw,size=256m", "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "HOME=/tmp", "-e", "QT_QPA_PLATFORM=offscreen", "-e", "FEMM_DIR=/opt/femm/bin",
            "-v", f"{script.parent}:/work", "-w", "/work", IMAGE, PYTHON, "-I", script.name]
    started = time.time()
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return done.returncode, done.stdout + done.stderr, time.time() - started
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)
        return 124, f"timed out after {timeout} s\n", time.time() - started


def parse_results(output: str) -> dict[str, tuple[str, str]]:
    found = {}
    for line in output.splitlines():
        if line.startswith("RESULT\t"):
            parts = line.split("\t", 3) + [""]
            found[parts[1]] = (parts[2], parts[3])
    return found


def run_batch(name: str, body: str) -> tuple[dict, int, str, float]:
    """Write harness + fixtures + body to results/NAME.py, run it, keep the output beside it."""
    RESULTS.mkdir(exist_ok=True)
    script = RESULTS / f"{name}.py"
    script.write_text((PY / "harness.py").read_text() + "\n" + (PY / "fixtures.py").read_text()
                      + "\n\nopenfemm(1)\n" + body + '\nprint("BATCH_DONE", flush=True)\nclosefemm()\n')
    code, output, seconds = run_python(script)
    (RESULTS / f"{name}.out").write_text(output)
    return parse_results(output), code, output, seconds


def summarize(name: str, results: dict, code: int, output: str, seconds: float) -> None:
    passed = sum(1 for status, _ in results.values() if status == "pass")
    finished = "BATCH_DONE" in output
    print(f"{name}: {passed}/{len(results)} passed, exit {code}, {seconds:.1f} s"
          + ("" if finished else "  ** batch did not finish **"))
    for test, (status, detail) in results.items():
        mark = "ok  " if status == "pass" else "FAIL"
        print(f"  {mark} {test}: {detail[:300]}")
    if not finished:
        print("  last output:", output.strip().splitlines()[-5:] if output.strip() else "(none)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    batch = sub.add_parser("batch")
    batch.add_argument("name")
    batch.add_argument("files", nargs="+")
    entries = sub.add_parser("entries")
    entries.add_argument("file")
    entries.add_argument("--only", nargs="*", default=None)
    tests = sub.add_parser("tests")
    tests.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    if args.command == "inventory":
        code, output, seconds = run_python(HERE / "inventory.py")
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "inventory.txt").write_text(output)
        print(f"inventory: exit {code}, {seconds:.1f} s, "
              f"{sum(1 for l in output.splitlines() if l.startswith('FN '))} functions")
        sys.exit(code)
    if args.command == "batch":
        body = "\n".join(Path(f).read_text() for f in args.files)
        summarize(args.name, *run_batch(args.name, body))
        return
    import catalogue_tests  # kept beside this file
    if args.command == "entries":
        catalogue_tests.run_entries_file(run_batch, summarize, Path(args.file).resolve(), args.only)
        return
    if not args.only:  # a full run replaces the batches of the last one
        for old in RESULTS.glob("*"):
            if re.fullmatch(r"(%s)\d*\.(py|out)" % "|".join(catalogue_tests.BATCHES), old.name):
                old.unlink()
    catalogue_tests.run_catalogue_tests(run_batch, summarize, CATALOGUE, IMAGE, IMAGE_TAG, args.only)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
