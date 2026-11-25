"""Play one task with an agent in the benchmark's sandbox, then score what it saved.

The agent sees only its workspace: task.md, task.json, the FEMM manual and an empty
submission/ folder. The key is kept outside it, where the sandbox cannot see it. The sandbox
runner (daro_bench/sandbox_run.py, used unchanged) runs Claude Code headless with no network
beyond the model API, and gives it FEMM through `femm-lua` with a run and time budget.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from . import reward
from .families import SCORE_SECONDS

# The FEMM 5.0 manual as text, named by the SHA-256 of its source in femm_brain's source manifest;
# the text the player gets is pinned by its own digest.
MANUAL = (Path.home() / ".cache" / "femm-brain" / "artifact_texts" /
          "e1273290fd7856a0f19779af87ce3c184e3eabc0771e4b814ece26210c61d237.txt")
MANUAL_TEXT_SHA256 = "f00bbd56bfd72f98c50e20233d4644c7960d4da111144bd7a67fc0badf2dc484"

PROMPT = """# Task: build a FEMM model

You are working alone and non-interactively: nobody will answer questions or approve
anything. You have about {minutes} minutes; the session is stopped at that limit, so save a
valid submission early and improve it as you go.

`task.md` describes a device. Build it in FEMM 5 and save it where `task.md` says. An
independent scorer then re-solves your saved model and measures the result itself, as the
end of `task.md` explains.

## What you have

- `task.md`, and `task.json` with the same task in machine form.
- `femm/manual.txt`: the FEMM 5.0 manual. Chapter 3 lists the Lua commands.
- `femm-lua script.lua`: runs a FEMM Lua script with FEMM 5 (headless). Run it from the
  folder holding the script, which must be this task folder or a folder inside it; that
  folder is FEMM's working directory, and files FEMM writes (`mi_saveas`, `openfile`) appear
  there. Scripts call FEMM commands directly (no
  `openfemm()`). FEMM's Lua is Lua 4.0 with FEMM's own changes: `sin`, `cos` and `atan2` work
  in radians and `PI` is defined, there is no `math.` table, and you write files with
  `openfile`, `write` and `closefile`. This task allows {max_solves} runs and
  {max_solver_seconds} seconds of FEMM time in total; the remaining budget is printed after
  every run. The scorer allows {score_minutes} minutes for its own solve of your model.
- Python 3 with numpy and the usual Unix tools. There is no internet access.
- Ignore `femm-check` on the PATH: it checks another benchmark's submissions, not this one.

## Rules

1. `task.md` states every input. Where FEMM needs a choice it does not state (mesh sizes,
   how to model open space), make it yourself.
2. Only the saved model is scored. A number you compute yourself is recorded, never used.
3. Save the files in `submission/` as regular files; links are refused.
"""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hidden_from_sandbox(path):
    """The sandbox replaces the home directory and /tmp with empty folders; a key must live under one."""
    path = Path(path).resolve()
    return any(path.is_relative_to(root) for root in (Path.home().resolve(), Path("/tmp")))


def prepare(task, text, out, minutes, max_solves, max_solver_seconds, manual=MANUAL, manual_sha256=MANUAL_TEXT_SHA256):
    """Write the agent's workspace and prompt under `out`; return the workspace."""
    if _sha(manual) != manual_sha256:
        raise ValueError(f"{manual} is not the pinned manual text (SHA-256 {manual_sha256})")
    workspace = Path(out) / "workspace"
    (workspace / "femm").mkdir(parents=True)
    (workspace / "submission").mkdir()
    (workspace / "task.md").write_text(text)
    (workspace / "task.json").write_text(json.dumps(task, indent=1) + "\n")
    shutil.copyfile(manual, workspace / "femm" / "manual.txt")
    (Path(out) / "prompt.md").write_text(PROMPT.format(minutes=_number(minutes), max_solves=max_solves,
                                                       max_solver_seconds=_number(max_solver_seconds),
                                                       score_minutes=SCORE_SECONDS // 60))
    return workspace


def _number(value):
    return str(int(value)) if float(value).is_integer() else str(value)


def checked_submission(workspace, target):
    """Copy the submission's regular files into `target` for the scorer; return (target, problems).

    The scorer runs on the host, where a link the agent made could point at any of the user's
    files (the sandbox hides them, but a link names them all the same). So only regular files
    inside the workspace are copied, nothing larger than the scorer accepts, and anything else
    is a contract problem.
    """
    workspace, target = Path(workspace), Path(target)
    target.mkdir(parents=True)
    problems = []
    for name in ("model.fem", "answer.json"):
        path = workspace / "submission" / name
        if not (path.is_symlink() or path.exists()):
            continue
        try:
            inside = path.resolve().is_relative_to(workspace.resolve())
        except (OSError, RuntimeError):
            inside = False
        if path.is_symlink() or not inside or not path.is_file():
            problems.append(f"submission/{name} is a link or not a regular file")
        elif path.stat().st_size > reward.MAX_BYTES:
            problems.append(f"submission/{name} is larger than {reward.MAX_BYTES // 1_000_000} MB")
        else:
            shutil.copyfile(path, target / name)
    return target, problems


def play(task, key, text, out, runner, model, effort="high", minutes=20, max_solves=30, max_solver_seconds=900,
         manual=MANUAL):
    """Run one trial and score it; returns the score record (also written to out/score.json)."""
    out = Path(out)
    if not _hidden_from_sandbox(out):
        raise ValueError(f"{out} must be under the home directory or /tmp, which the sandbox hides")
    out.mkdir(parents=True, exist_ok=False)
    (out / "key.json").write_text(json.dumps(key, indent=1) + "\n")
    workspace = prepare(task, text, out, minutes, max_solves, max_solver_seconds, manual)
    command = [sys.executable, str(Path(runner).resolve()), "--workspace", str(workspace),
               "--prompt-file", str(out / "prompt.md"), "--out", str(out / "trial"), "--model", model,
               "--effort", effort, "--minutes", str(minutes), "--max-solves", str(max_solves),
               "--max-solver-seconds", str(max_solver_seconds)]
    with open(out / "runner.log", "w") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    submitted, problems = checked_submission(workspace, out / "submitted")
    if problems:
        record = {"task": task["id"], "family": task["family"], "level": task["level"], "stage": "contract",
                  "detail": "; ".join(problems), "passed": False, "reward": 0.0}
    else:
        record = reward.score(task, key, submitted, out / "score")
    record.update(runner_exit=code, task_md_sha256=_sha(workspace / "task.md"), prompt_sha256=_sha(out / "prompt.md"),
                  manual_sha256=_sha(workspace / "femm" / "manual.txt"))
    trial = out / "trial" / "trial.json"
    if trial.is_file():
        summary = json.loads(trial.read_text())
        record["trial"] = {k: summary.get(k) for k in ("model", "effort", "claude_version", "wall_seconds", "timed_out",
                                                       "solver_runs", "solver_seconds", "solver_refused")}
        record["trial"]["cost_usd"] = (summary.get("result") or {}).get("total_cost_usd")
    (out / "score.json").write_text(json.dumps(record, indent=1) + "\n")
    return record
