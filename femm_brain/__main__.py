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
    graph = load_graph()
    if args.command == "validate":
        print(json.dumps(graph["validation"], indent=2))
        return 0 if graph["validation"]["status"] == "passed" else 1
    if graph["validation"]["status"] != "passed":
        print(json.dumps(graph["validation"], indent=2), file=sys.stderr)
        return 1
    if args.command == "stats":
        print(json.dumps(graph["stats"], indent=2))
        return 0
    store = GraphStore(args.db)
    store.replace(graph)
    if args.command == "build":
        print(json.dumps({"database": str(args.db), **graph["stats"]}, indent=2))
    elif args.command == "serve":
        from .server import serve
        serve(graph, store, args.host, args.port)
    elif args.command == "search":
        print(json.dumps(store.search(" ".join(args.query), args.limit), indent=2))
    elif args.command == "export":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
