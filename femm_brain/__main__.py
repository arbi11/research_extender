"""Build, query and render the source-backed engineering knowledge graph."""
import argparse
import json
import os
from pathlib import Path
import sys

from .core import DATA, load_graph
from .store import GraphStore


def database_default():
    return Path(os.getenv("FEMM_BRAIN_DATA_DIR", str(Path.home() / ".local/share/femm-brain"))) / "brain.sqlite3"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "validate", "stats", "serve", "search", "export"):
        command = sub.add_parser(name)
        command.add_argument("--db", type=Path, default=database_default())
        if name == "serve":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8010)
        elif name == "search":
            command.add_argument("query", nargs="+")
            command.add_argument("--limit", type=int, default=50)
        elif name == "export":
            command.add_argument("--output", type=Path, required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--cache", type=Path)
    ingest.add_argument("--manifest", type=Path, default=DATA / "source_manifest.json")
    ingest.add_argument("--pages-only", action="store_true")
    ingest.add_argument("--workers", type=int, default=4)
    ingest.add_argument("--max-pages", type=int, default=100)
    ingest.add_argument("--max-attachments", type=int, default=200)
    restore = sub.add_parser("restore-cache", help="Re-fetch the recorded source snapshot, accepting identical bytes only")
    restore.add_argument("--cache", type=Path)
    restore.add_argument("--manifest", type=Path, default=DATA / "source_manifest.json")
    restore.add_argument("--report", type=Path)
    restore.add_argument("--workers", type=int, default=4)
    restore.add_argument("--for-reproductions", action="store_true",
                         help="Restore only the official model files that reproduction cases open")
    reproduce = sub.add_parser("reproduce", help="Reproduce official examples from the original model, stored data and article text")
    reproduce.add_argument("case_ids", nargs="*")
    reproduce.add_argument("--all", action="store_true")
    reproduce.add_argument("--levels", default="original,rebuild,article")
    reproduce.add_argument("--output", type=Path, required=True)
    reproduce.add_argument("--engine", choices=("docker", "native"), default="docker")
    reproduce.add_argument("--image")
    reproduce.add_argument("--timeout", type=int, default=900)
    reproduce.add_argument("--controls", action="store_true", help="Also run each case's diagnostic mesh and parameter controls")
    report = sub.add_parser("reproduction-report", help="Record reproduction evidence and render its audit page")
    report.add_argument("--results", type=Path, required=True)
    report.add_argument("--output", type=Path)
    report.add_argument("--html", type=Path)
    sub.add_parser("studies", help="List executable, source-backed studies")
    plan = sub.add_parser("plan", help="Select a known study locally, without an LLM")
    plan.add_argument("request", nargs="*")
    plan.add_argument("--case")
    plan.add_argument("--parameters", default="{}")
    plan.add_argument("--output", type=Path)
    for name in ("solve", "verify-solves"):
        cmd = sub.add_parser(name, help="Run registered native studies with frozen references")
        if name == "solve":
            cmd.add_argument("case_id")
            cmd.add_argument("--parameters", default="{}")
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--engine", choices=("docker", "native"), default="docker")
        cmd.add_argument("--image")
        cmd.add_argument("--command", dest="native_command")
        cmd.add_argument("--timeout", type=int, default=180)
    # Browsing remains standard-library-only; execution is an explicit option.
    serve_cmd = sub.choices["serve"]
    serve_cmd.add_argument("--enable-solver", action="store_true")
    serve_cmd.add_argument("--image")
    serve_cmd.add_argument("--jobs", type=Path, default=database_default().parent / "jobs")
    args = parser.parse_args(argv)
    if args.command == "ingest":
        from . import collect
        cache = args.cache or collect.DEFAULT_CACHE
        result = collect.crawl(cache=cache, max_pages=args.max_pages, max_attachments=args.max_attachments,
                               workers=args.workers, download=not args.pages_only)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        if not args.pages_only:
            from .catalog import build_function_catalog, build_model_previews, build_recipe_graph
            directory = args.manifest.parent
            build_function_catalog(result, cache, directory / "functions.json")
            build_model_previews(result, cache, directory)
            build_recipe_graph(directory / "recipes.json")
        print(json.dumps(result["crawl"], indent=2))
        return 0
    if args.command == "restore-cache":
        from . import collect
        only = None
        if args.for_reproductions:
            from .reproduce import load_cases
            only = sorted({c["source_model_sha256"] for c in load_cases() if c.get("source_model_sha256")})
        result = collect.restore(json.loads(args.manifest.read_text()), cache=args.cache or collect.DEFAULT_CACHE,
                                 workers=args.workers, only=only)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: result[key] for key in ("cache", "expected", "pages", "artifacts")}, indent=2))
        return 0 if set(result["artifacts"]) <= {"restored"} else 1
    if args.command == "reproduce":
        from .reproduce import LEVELS, load_cases, reproduce as run_reproduction
        identifiers = [c["id"] for c in load_cases()] if args.all else args.case_ids
        levels = tuple(args.levels.split(","))
        if not identifiers or set(levels) - set(LEVELS):
            parser.error("Name case IDs or --all, and levels from: " + ",".join(LEVELS))
        summaries = [run_reproduction(i, args.output / i, levels=levels, engine=args.engine, image=args.image,
                                      timeout=args.timeout, controls=args.controls) for i in identifiers]
        print(json.dumps([{"case_id": s["case_id"], "levels": s["levels"],
                           "level_comparison": {k: v["status"] for k, v in s["level_comparison"].items()}} for s in summaries], indent=2))
        return 0 if all(v["status"] in ("passed", "comparators_only", "not_applicable", "not_attempted") for s in summaries for v in s["levels"].values()) else 1
    if args.command == "reproduction-report":
        from .reproduction_report import RECORD, write
        record = write(args.results, args.output or RECORD, args.html)
        print(json.dumps(record["totals"], indent=2))
        return 0
    graph = load_graph()
    if args.command == "validate":
        print(json.dumps(graph["validation"], indent=2))
        return 0 if graph["validation"]["status"] == "passed" else 1
    if graph["validation"]["status"] != "passed":
        print(json.dumps(graph["validation"], indent=2), file=sys.stderr)
        return 1
    if args.command in {"studies", "plan", "solve", "verify-solves"}:
        from .simulation import simulation_catalog, plan_simulation
        if args.command == "studies":
            print(json.dumps({"cases": simulation_catalog(graph), "llm_calls": 0}, indent=2))
            return 0
        if args.command == "plan":
            from .planner import plan_request
            if args.case and args.request:
                parser.error("Choose --case with typed parameters or a natural-language request; do not combine them")
            parameters = json.loads(args.parameters)
            if args.case:
                result = {"selection": {"status": "ready", "selected_case_id": args.case,
                                         "parameters": parameters, "llm_calls": 0},
                          "plan": plan_simulation(args.case, parameters, graph)}
            else:
                selection = plan_request(" ".join(args.request), graph)
                if parameters:
                    parser.error("--parameters requires --case; natural-language values are parsed explicitly")
                result = {"selection": selection}
                if selection["status"] == "ready":
                    result["plan"] = plan_simulation(selection["selected_case_id"], selection["parameters"], graph)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with args.output.open("x") as stream:
                        stream.write(json.dumps(result, indent=2) + "\n")
                except FileExistsError:
                    parser.error("Refusing to overwrite an existing frozen plan; choose a new --output path")
                print(args.output)
            else:
                print(json.dumps(result, indent=2))
            return 0 if result["selection"]["status"] == "ready" else 2
        from .execution import run_case
        if args.timeout < 1 or args.timeout > 1800:
            parser.error("--timeout must be between 1 and 1800 seconds")
        identifiers = [args.case_id] if args.command == "solve" else [c["id"] for c in simulation_catalog(graph)]
        reports = []
        for identifier in identifiers:
            destination = args.output if args.command == "solve" else args.output / identifier
            reports.append(run_case(identifier, destination,
                                    json.loads(args.parameters) if args.command == "solve" else {},
                                    engine=args.engine, image=args.image, command=args.native_command,
                                    timeout=args.timeout))
        value = reports[0] if args.command == "solve" else {"status": "passed" if all(r["status"] == "passed" for r in reports) else "failed", "cases": reports, "llm_calls": 0}
        if args.command == "verify-solves":
            (args.output / "summary.json").write_text(json.dumps(value, indent=2) + "\n")
        print(json.dumps(value, indent=2))
        return 0 if all(r["status"] in {"passed", "solved_unassessed"} for r in reports) else 1
    if args.command == "stats":
        print(json.dumps(graph["stats"], indent=2))
        return 0
    store = GraphStore(args.db)
    store.replace(graph)
    if args.command == "build":
        print(json.dumps({"database": str(args.db), **graph["stats"]}, indent=2))
    elif args.command == "serve":
        from .server import serve
        from .execution import SimulationJobs
        jobs = SimulationJobs(args.jobs, args.image) if args.enable_solver else None
        serve(graph, store, args.host, args.port, jobs)
    elif args.command == "search":
        print(json.dumps(store.search(" ".join(args.query), args.limit), indent=2))
    elif args.command == "export":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
