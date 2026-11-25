"""Build and query the local, source-backed FEMM API catalogue."""
import argparse
import json
from pathlib import Path
import sqlite3
import subprocess

from .catalogue import DEFAULT_DB, build, get, load_batch, search, stats, validate_batch
from .source import DEFAULT_CACHE, fetch_sources, load_sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch-sources", "validate", "build", "search", "get", "stats", "export", "check-geometry"):
        p = commands.add_parser(name)
        p.add_argument("--db", type=Path, default=DEFAULT_DB)
        if name in ("fetch-sources", "validate", "build", "check-geometry"):
            p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
        if name == "build":
            p.add_argument("--evidence", type=Path)
        if name == "check-geometry":
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--image")
            p.add_argument("--timeout", type=int, default=120)
        if name in ("search", "get"):
            p.add_argument("--version")
            p.add_argument("--binding")
        if name == "search":
            p.add_argument("query")
            p.add_argument("--physics")
            p.add_argument("--limit", type=int, default=5)
        if name == "get":
            p.add_argument("name")
        if name == "stats":
            p.add_argument("--all-pending", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch-sources":
            result = {"sources": fetch_sources(load_sources(), args.cache)}
        elif args.command == "validate":
            result = validate_batch(load_batch(), load_sources(), args.cache)
            result.pop("inventory", None)
        elif args.command == "build":
            result = build(args.db, cache=args.cache, evidence_path=args.evidence)
        elif args.command == "check-geometry":
            from .geometry_check import run
            result = run(args.output, batch=load_batch(), cache=args.cache, image=args.image, timeout=args.timeout)
        elif args.command == "get":
            result = get(args.name, args.db, version=args.version, binding=args.binding)
        elif args.command == "search":
            records = search(args.query, args.db, version=args.version, binding=args.binding, physics=args.physics, limit=args.limit)
            keys = ("name", "version", "binding", "physics", "category", "purpose", "signature", "review", "execution", "references")
            result = {"query": args.query, "count": len(records), "results": [{k: r[k] for k in keys} for r in records]}
        elif args.command == "export":
            info = stats(args.db)
            from .catalogue import connect_readonly
            from contextlib import closing
            with closing(connect_readonly(args.db)) as db:
                records = [json.loads(r[0]) for r in db.execute("SELECT data FROM commands ORDER BY name,version,binding")]
            result = {"metadata": info, "commands": records}
        else:
            result = stats(args.db, include_pending=args.all_pending)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 2 if result.get("status") in ("failed", "not_found", "not_reviewed") else 0
    except (OSError, ValueError, sqlite3.Error, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
