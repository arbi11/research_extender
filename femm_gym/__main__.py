"""Command line for femm_gym.

  python3 -m femm_gym list
  python3 -m femm_gym sample FAMILY [--index N] [--split NAME] [--key]
  python3 -m femm_gym selfcheck [--families F ...] [--per-family N] [--work DIR]
  python3 -m femm_gym freeze --out DIR [--per-family N] [--families F ...] [--work DIR]
  python3 -m femm_gym verify --testset DIR
  python3 -m femm_gym score --testset DIR --task ID --submission DIR [--work DIR]
  python3 -m femm_gym play --runner PATH --out DIR --model M (--testset DIR --task ID | --family F --index N --split S)

Level 3 (generated papers; scored by the daro_bench scorer, found with --bench or $DARO_BENCH):
  python3 -m femm_gym l3-sample [--index N] [--split NAME] [--key]
  python3 -m femm_gym l3-selfcheck [--count N] [--work DIR]
  python3 -m femm_gym l3-freeze --out DIR [--count N] [--jobs J] [--work DIR]
  python3 -m femm_gym l3-verify --testset DIR
  python3 -m femm_gym l3-score --testset DIR --task ID --submission DIR [--work DIR]
  python3 -m femm_gym l3-play --runner PATH --out DIR --model M (--testset DIR --task ID | --index N --split S)

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
    l3_sample = sub.add_parser("l3-sample", help="print a Level 3 paper and task without solving")
    l3_sample.add_argument("--index", type=int, default=0)
    l3_sample.add_argument("--split", default="sample")
    l3_sample.add_argument("--key", action="store_true", help="also print the machine and the expected provenance")
    l3_check = sub.add_parser("l3-selfcheck")
    l3_check.add_argument("--count", type=int, default=1)
    l3_check.add_argument("--jobs", type=int, default=1)
    l3_freeze = sub.add_parser("l3-freeze")
    l3_freeze.add_argument("--out", required=True)
    l3_freeze.add_argument("--count", type=int, default=24)
    l3_freeze.add_argument("--jobs", type=int, default=3)
    l3_verify = sub.add_parser("l3-verify")
    l3_verify.add_argument("--testset", required=True)
    l3_score = sub.add_parser("l3-score")
    l3_score.add_argument("--testset", required=True)
    l3_score.add_argument("--task", required=True)
    l3_score.add_argument("--submission", required=True, help="folder holding model.fem, entities.json, inputs.json")
    l3_play = sub.add_parser("l3-play", help="run an agent on one Level 3 task in the sandbox, then score it")
    l3_play.add_argument("--runner", required=True, help="path to daro_bench/sandbox_run.py")
    l3_play.add_argument("--out", required=True, help="new folder for the trial (under ~ or /tmp)")
    l3_play.add_argument("--model", required=True)
    l3_play.add_argument("--effort", default="high")
    l3_play.add_argument("--minutes", type=float, default=75, help="default: the benchmark's budget")
    l3_play.add_argument("--max-solves", type=int, default=60)
    l3_play.add_argument("--max-solver-seconds", type=float, default=3600)
    l3_play.add_argument("--testset", help="a frozen Level 3 set, or an l3-selfcheck work folder; with --task")
    l3_play.add_argument("--task")
    l3_play.add_argument("--index", type=int, default=None, help="or a fresh task; with --split")
    l3_play.add_argument("--split", default="l3-check")
    for p in (l3_check, l3_freeze, l3_verify, l3_score, l3_play):
        p.add_argument("--bench", help="the daro_bench folder (default $DARO_BENCH or ~/code/research_extender_daro/daro_bench)")
    for p in (l3_check, l3_freeze, l3_score):
        p.add_argument("--work")
    args = parser.parse_args(argv)
    if args.command.startswith("l3-"):
        return _level3(args, parser)

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


def _level3(args, parser):
    from . import level3, level3_score, level3_set
    if args.command == "l3-sample":
        seed, machine, plan = level3_set.draw(args.index, args.split)
        published = level3.publish(machine, None, plan)
        text, _ = level3.write_paper(published, plan)
        print(text)
        print("---- task.md (case section) ----\n")
        print(level3.task_md(machine))
        if args.key:
            print(json.dumps({"seed": seed, "machine": machine, "plan": plan,
                              "expectations": level3.expectations(published, plan)}, indent=1))
    elif args.command == "l3-selfcheck":
        results = level3_set.selfcheck(args.count, work=_work(args.work, "l3-selfcheck"), bench_path=args.bench,
                                       jobs=args.jobs)
        bad = [r["id"] for r in results if not r["ok"]]
        print("all Level 3 controls behaved as expected" if not bad else "controls misbehaved for " + ", ".join(bad))
        return 1 if bad else 0
    elif args.command == "l3-freeze":
        manifest = level3_set.freeze(args.out, args.count, args.jobs, work=args.work, bench_path=args.bench)
        print(f"froze {len(manifest['tasks'])} Level 3 tasks, dropped {len(manifest['dropped'])}, into {args.out}")
    elif args.command == "l3-verify":
        bad, drifted, scorer = level3_set.verify(args.testset, args.bench)
        print("all files match FROZEN.sha256" if not bad else "changed: " + ", ".join(bad))
        if drifted:
            print("generator code changed since the freeze (tasks are still valid; new draws would differ): "
                  + ", ".join(drifted))
        if scorer:
            print("the benchmark scorer changed since the freeze, so scores may differ: " + ", ".join(scorer))
        return 1 if bad or scorer else 0
    elif args.command == "l3-score":
        _, key = level3_set.load(args.testset, args.task)
        print(json.dumps(level3_score.score(key, args.submission, _work(args.work, "l3-score-" + args.task),
                                            args.bench), indent=1))
    elif args.command == "l3-play":
        from . import play_l3
        if args.testset and args.task:
            task_dir, key = level3_set.load(args.testset, args.task)
        elif args.index is not None:
            if args.split == level3_set.L3_SPLIT:
                parser.error("fresh tasks may not come from the test split; use --testset")
            task, key, text = level3_set.make(args.index, args.split, _work(None, f"l3-reference-{args.index}"), args.bench)
            task_dir = _work(None, f"l3-task-{key['id']}")
            level3_set.write_task(task_dir, task, key, text)
            task_dir = task_dir / "tasks" / key["id"]
        else:
            parser.error("give --testset and --task, or --index and --split")
        record = play_l3.play(task_dir, key, args.out, args.runner, args.model, args.effort, args.minutes,
                              args.max_solves, args.max_solver_seconds, bench_path=args.bench)
        print(json.dumps({k: v for k, v in record.items() if k != "flags"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
