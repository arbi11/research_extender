"""Command line for femm_gym.

  python3 -m femm_gym list
  python3 -m femm_gym sample FAMILY [--index N] [--split NAME] [--key]
  python3 -m femm_gym selfcheck [--families F ...] [--per-family N] [--work DIR]
  python3 -m femm_gym freeze --out DIR [--per-family N] [--families F ...] [--work DIR]
  python3 -m femm_gym verify --testset DIR
  python3 -m femm_gym score --testset DIR --task ID --submission DIR [--work DIR]
  python3 -m femm_gym play --runner PATH --out DIR --model M (--testset DIR --task ID | --family F --index N --split S)

selfcheck, freeze, score and play run FEMM in the pinned engine container (see engine.py);
play also runs an agent in the benchmark's sandbox (see play.py).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from . import reward, testset
from .families import FAMILIES


def _work(path, label):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return Path(path) if path else Path.home() / ".cache" / "femm-gym" / "work" / f"{label}-{stamp}"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m femm_gym", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sample = sub.add_parser("sample")
    sample.add_argument("family", choices=sorted(FAMILIES))
    sample.add_argument("--index", type=int, default=0)
    sample.add_argument("--split", default="sample")
    sample.add_argument("--key", action="store_true", help="also print the hidden parameters")
    check = sub.add_parser("selfcheck")
    check.add_argument("--families", nargs="*", default=None)
    check.add_argument("--per-family", type=int, default=2)
    check.add_argument("--work")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--out", required=True)
    freeze.add_argument("--per-family", type=int, default=30)
    freeze.add_argument("--families", nargs="*", default=None)
    freeze.add_argument("--work")
    verify = sub.add_parser("verify")
    verify.add_argument("--testset", required=True)
    score = sub.add_parser("score")
    score.add_argument("--testset", required=True)
    score.add_argument("--task", required=True)
    score.add_argument("--submission", required=True, help="folder holding model.fem (and optional answer.json)")
    score.add_argument("--work")
    play = sub.add_parser("play", help="run an agent on one task in the sandbox, then score it")
    play.add_argument("--runner", required=True, help="path to daro_bench/sandbox_run.py")
    play.add_argument("--out", required=True, help="new folder for the trial (under ~ or /tmp)")
    play.add_argument("--model", required=True)
    play.add_argument("--effort", default="high")
    play.add_argument("--minutes", type=float, default=20)
    play.add_argument("--max-solves", type=int, default=30)
    play.add_argument("--max-solver-seconds", type=float, default=900)
    play.add_argument("--testset", help="a frozen set; with --task")
    play.add_argument("--task")
    play.add_argument("--family", choices=sorted(FAMILIES), help="or a fresh task; with --index and --split")
    play.add_argument("--index", type=int, default=0)
    play.add_argument("--split", default="check")
    args = parser.parse_args(argv)

    if args.command == "list":
        for f in FAMILIES.values():
            tolerance = f"min IoU {f.tolerance}" if f.level == 1 else f"{f.tolerance:.0%} relative"
            print(f"{f.name:20s} level {f.level}  {f.title} ({f.measure}; {tolerance})")
    elif args.command == "sample":
        task, key, text = testset.make_task(FAMILIES[args.family], args.index, args.split)
        print(text)
        if args.key:
            print(json.dumps(key["params"], indent=1))
    elif args.command == "selfcheck":
        work = _work(args.work, "selfcheck")
        failures = 0
        for name in args.families or sorted(FAMILIES):
            for index in range(args.per_family):
                task, key, _ = testset.make_task(FAMILIES[name], index, "selfcheck")
                key = testset.complete_key(task, key, work / task["id"])
                if FAMILIES[name].level == 2 and key["value"] is None:
                    print(f"{task['id']}: reference did not converge {key['reference']['levels']}")
                    failures += 1
                    continue
                results = testset.controls(task, key, work / task["id"] / "controls")
                for control, r in results.items():
                    expected = not control.startswith("wrong")
                    ok = r.get("passed") == expected and r["stage"] == "scored"
                    failures += not ok
                    detail = (f"error ratio {r.get('error_ratio')}" if FAMILIES[name].level == 2 else
                              f"least IoU {r.get('least_iou')}") if r["stage"] == "scored" else f"{r['stage']}: {r.get('detail', '')[:200]}"
                    print(f"{task['id']:24s} {control:12s} {'ok  ' if ok else 'BAD '} passed={r.get('passed')} {detail}"
                          f" elements={r.get('elements')}")
                if FAMILIES[name].level == 2:
                    levels = ", ".join(f"L{l['level']} {l['value']:.6g} ({l['elements']} el, {l['seconds']} s)"
                                       for l in key["reference"]["levels"])
                    print(f"{'':24s} reference  {levels}")
        print(f"work: {work}")
        print("all controls behaved as expected" if not failures else f"{failures} control(s) misbehaved")
        return 1 if failures else 0
    elif args.command == "freeze":
        manifest = testset.freeze(args.out, args.per_family, args.families, work=args.work)
        print(f"froze {len(manifest['tasks'])} tasks, dropped {len(manifest['dropped'])}, into {args.out}")
    elif args.command == "verify":
        bad, drifted = testset.verify(args.testset)
        print("all files match FROZEN.sha256" if not bad else "changed: " + ", ".join(bad))
        if drifted:
            print("generator code changed since the freeze (tasks are still valid; new draws would differ): "
                  + ", ".join(drifted))
        return 1 if bad else 0
    elif args.command == "score":
        task, key = testset.load(args.testset, args.task)
        print(json.dumps(reward.score(task, key, args.submission, _work(args.work, "score-" + args.task)), indent=1))
    elif args.command == "play":
        from . import play as player
        if args.testset:
            task, key = testset.load(args.testset, args.task)
            text = (Path(args.testset) / "tasks" / f"{args.task}.md").read_text()
        elif args.family:
            if args.split == testset.TEST_SPLIT:
                parser.error("fresh tasks may not come from the test split; use --testset")
            task, key, text = testset.make_task(FAMILIES[args.family], args.index, args.split)
            key = testset.complete_key(task, key, _work(None, "reference-" + task["id"]))
            if FAMILIES[args.family].level == 2 and key["value"] is None:
                print(f"{task['id']}: reference did not converge; pick another index")
                return 1
        else:
            parser.error("give --testset and --task, or --family")
        record = player.play(task, key, text, args.out, args.runner, args.model, args.effort, args.minutes,
                             args.max_solves, args.max_solver_seconds)
        print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
