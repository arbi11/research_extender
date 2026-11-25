"""Look up FEMM 5 magnetics commands for Python (pyFEMM) in the verified catalogue.

  python3 -I lookup.py get NAME [NAME ...] [--test]   full entries (and the Python test that checked them)
  python3 -I lookup.py search WORD [WORD ...]         best matches, one line each
  python3 -I lookup.py topic [TOPIC]                  the topics, or the commands in one topic
  python3 -I lookup.py facts                          the engine facts, in full
  python3 -I lookup.py cheatsheet                     rewrite CHEATSHEET.md from the catalogue
  python3 -I lookup.py check                          check the catalogue's schema and consistency
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATALOGUE = ROOT / "catalogue.json"
CHEATSHEET = ROOT / "CHEATSHEET.md"

TOPICS = ("general document draw select edit assign materials boundaries circuits points "
          "problem mesh solve file view results contour gap mesh-data").split()
SOURCES = ("manual", "binding", "console")
NOTE_TAGS = ("[manual]", "[binding]", "[tested]")
ENTRY_KEYS = {"name", "topic", "source", "call", "does", "args", "returns", "notes", "test"}
FACT_KEYS = {"name", "says", "test"}
OPTIONAL_KEYS = {"untested_because", "verified"}
STATUSES = ("pass", "fail", "not run")


def load(path: Path = CATALOGUE) -> dict:
    return json.loads(path.read_text())


def items(data: dict) -> list[dict]:
    return data["facts"] + data["entries"]


def find(data: dict, name: str) -> dict | None:
    wanted = name.strip().lower()
    for item in items(data):
        if item["name"].lower() in (wanted, "fact:" + wanted):
            return item
    return None


def status_line(item: dict) -> str:
    verified = item.get("verified")
    if not verified:
        return "not verified yet"
    if verified["status"] == "not run":
        return "untested: " + item.get("untested_because", verified.get("observed", ""))
    return f"{verified['status']} in FEMM on {verified['date']}"


def describe(item: dict, with_test: bool = False) -> str:
    wrap = lambda text, indent: textwrap.fill(text, 100, initial_indent=indent,
                                              subsequent_indent=indent + "  ")
    if item["name"].startswith("fact:"):
        lines = [f"{item['name']}  ({status_line(item)})", wrap(item["says"], "  ")]
    else:
        lines = [item["call"], f"  {item['topic']} | {item['source']} | {status_line(item)}",
                 wrap(item["does"], "  ")]
        if item["args"]:
            lines.append("  args:")
            lines += [wrap(arg, "    ") for arg in item["args"]]
        lines.append(wrap("returns: " + (item["returns"] or "nothing"), "  "))
        if item["notes"]:
            lines.append("  notes:")
            lines += [wrap("- " + note, "    ") for note in item["notes"]]
    observed = (item.get("verified") or {}).get("observed")
    if observed and item.get("test"):
        lines.append(wrap("observed: " + observed, "  "))
    if with_test and item.get("test"):
        lines.append("  test (Python with pyFEMM, run in FEMM):")
        lines += ["    " + line for line in item["test"].splitlines()]
    return "\n".join(lines)


def search(data: dict, words: list[str], limit: int = 12) -> list[tuple[int, dict]]:
    """Rank items by how many of the words they contain, then by how often the words occur,
    counting the name 4 times, the call and description twice, and the rest once."""
    words = [w.lower() for w in words if w.strip()]
    ranked = []
    for item in items(data):
        fields = [(4, item["name"]),
                  (2, item.get("call", "") + " " + item.get("does", "")),
                  (1, " ".join([item.get("says", "")] + item.get("args", []) + item.get("notes", [])
                               + [item.get("returns") or ""]))]
        counts = [sum(weight * text.lower().count(word) for weight, text in fields) for word in words]
        matched = sum(1 for c in counts if c)
        if matched:
            ranked.append((matched, sum(counts), item))
    ranked.sort(key=lambda r: (-r[0], -r[1], r[2]["name"]))
    return [(score, item) for _, score, item in ranked[:limit]]


def one_line(item: dict) -> str:
    text = item.get("does") or item.get("says", "")
    return f"{item['name']:30} {textwrap.shorten(text, 110, placeholder=' ...')}"


TAGS_AT_END = re.compile(r"(\s*\[(?:manual|binding|tested)\])+\s*$")
MENTIONS_MANUAL = re.compile(r"manual|FEMM 4\.2", re.I)
SILENT = re.compile(r"silent|no error|without an error|raises nothing|ignored|ignores|instead of", re.I)
PROVENANCE_ONLY = re.compile(r"not in the|empty (lua )?function", re.I)


def pitfalls(data: dict) -> tuple[list, list]:
    """Notes worth reading before writing a model, found by their wording:
    where the engine or binding departs from the manual (or FEMM 4.2), and failures that raise
    no error. Left out: notes taken from the manual alone, notes that only say where a command
    is defined or that it is an empty stub, and headless no-op notes."""
    differs, silent = [], []
    for entry in sorted(data["entries"], key=lambda e: e["name"]):
        for note in entry["notes"]:
            body = TAGS_AT_END.sub("", note)
            tags = note[len(body):].split()
            if tags == ["[manual]"] or PROVENANCE_ONLY.search(body):
                continue
            if MENTIONS_MANUAL.search(body):
                differs.append((entry["name"], note))
            elif SILENT.search(body) and "headless" not in body.lower():
                silent.append((entry["name"], note))
    return differs, silent


def cheatsheet(data: dict) -> str:
    about = data["about"]
    counts = about.get("verification", {}).get("counts", {})
    out = ["# FEMM 5 cheat sheet for Python (pyFEMM, magnetics)", "",
           "Generated from `catalogue.json` by `python3 -I lookup.py cheatsheet`; do not edit by hand.",
           f"Engine: {about['engine']['python']} in `{about['engine']['tag']}`. "
           f"Checked in FEMM on {about.get('verification', {}).get('date', '?')}: "
           + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())) + ".",
           "What a command does, all its notes and the test that checked it: "
           "`python3 -I lookup.py get NAME --test`; find commands with `lookup.py search WORDS`.",
           "", "## Engine facts", ""]
    out += [f"- **{fact['name'][5:]}**: {fact['says']}" for fact in data["facts"]]
    differs, silent = pitfalls(data)
    out += ["", "## Where this engine differs from the manual or FEMM 4.2", ""]
    out += [f"- `{name}`: {note}" for name, note in differs]
    out += ["", "## Failures that raise no error", ""]
    out += [f"- `{name}`: {note}" for name, note in silent]
    out += ["", "## Commands by topic", ""]
    for topic in TOPICS:
        members = sorted((e for e in data["entries"] if e["topic"] == topic), key=lambda e: e["name"])
        if members:
            calls = [f"`{e['call']}`" + ("" if e.get("test") else " (untested)") for e in members]
            out.append(f"- **{topic}**: " + "; ".join(calls))
    return "\n".join(out) + "\n"


def validate(data: dict) -> list[str]:
    """Schema and consistency problems, as readable strings (empty when all is well)."""
    problems = []
    for key in ("about", "facts", "entries"):
        if key not in data:
            problems.append(f"catalogue has no '{key}'")
    if problems:
        return problems
    seen = set()
    for item in items(data):
        name = item.get("name", "?")
        if name in seen:
            problems.append(f"{name}: duplicate name")
        seen.add(name)
        is_fact = name.startswith("fact:")
        required = FACT_KEYS if is_fact else ENTRY_KEYS
        missing = required - item.keys()
        unknown = item.keys() - required - OPTIONAL_KEYS
        if missing:
            problems.append(f"{name}: missing {sorted(missing)}")
            continue
        if unknown:
            problems.append(f"{name}: unknown keys {sorted(unknown)}")
        if item["test"] is None and not item.get("untested_because"):
            problems.append(f"{name}: no test and no untested_because")
        if item["test"] is not None and (not isinstance(item["test"], str) or not item["test"].strip()):
            problems.append(f"{name}: test must be Python text or null")
        elif item["test"] is not None:
            try:  # the harness runs the test as the body of a function
                compile("def _test():\n" + textwrap.indent(item["test"], "    "), name, "exec")
            except SyntaxError as error:
                problems.append(f"{name}: test does not compile: {error.msg} (line {error.lineno})")
        verified = item.get("verified")
        if verified is not None and verified.get("status") not in STATUSES:
            problems.append(f"{name}: verified status {verified.get('status')!r}")
        if is_fact:
            if not item["says"].strip():
                problems.append(f"{name}: empty 'says'")
            continue
        if item["topic"] not in TOPICS:
            problems.append(f"{name}: topic {item['topic']!r} is not one of {TOPICS}")
        if item["source"] not in SOURCES:
            problems.append(f"{name}: source {item['source']!r}")
        if item["source"] == "console":  # reached through callfemm('name(...)') or callfemm_noeval
            if not re.match(r"callfemm(_noeval)?\(['\"]" + re.escape(name) + r"\(", item["call"]):
                problems.append(f"{name}: a console call must be callfemm('{name}(...)'), not {item['call']!r}")
        elif not item["call"].startswith(name + "("):
            problems.append(f"{name}: call {item['call']!r} does not start with '{name}('")
        if not item["does"].strip():
            problems.append(f"{name}: empty 'does'")
        if not isinstance(item["args"], list) or not all(isinstance(a, str) for a in item["args"]):
            problems.append(f"{name}: args must be a list of strings")
        if item["returns"] is not None and not isinstance(item["returns"], str):
            problems.append(f"{name}: returns must be text or null")
        for note in item["notes"]:
            if not note.rstrip().endswith(NOTE_TAGS):
                problems.append(f"{name}: note does not end with a source tag: {note[:60]!r}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="\n".join(__doc__.splitlines()[2:]))
    sub = parser.add_subparsers(dest="command", required=True)
    get = sub.add_parser("get")
    get.add_argument("names", nargs="+")
    get.add_argument("--test", action="store_true", help="also print the Python test")
    sub.add_parser("search").add_argument("words", nargs="+")
    sub.add_parser("topic").add_argument("topic", nargs="?")
    sub.add_parser("facts")
    sub.add_parser("cheatsheet")
    sub.add_parser("check")
    args = parser.parse_args()
    data = load()

    if args.command == "get":
        names = [i["name"] for i in items(data)]
        for name in args.names:
            item = find(data, name)
            if item:
                print(describe(item, args.test), end="\n\n")
            else:
                close = difflib.get_close_matches(name, names, n=5, cutoff=0.6)
                print(f"{name}: not in the catalogue" + (f"; did you mean {', '.join(close)}?" if close else ""),
                      end="\n\n")
    elif args.command == "search":
        found = search(data, args.words)
        print("\n".join(one_line(item) for _, item in found) or "no matches")
    elif args.command == "topic":
        if not args.topic:
            for topic in TOPICS:
                n = sum(1 for e in data["entries"] if e["topic"] == topic)
                print(f"{topic:12} {n} commands")
        else:
            members = sorted((e for e in data["entries"] if e["topic"] == args.topic), key=lambda e: e["name"])
            print("\n".join(one_line(e) for e in members) or f"no topic {args.topic!r}; topics: {' '.join(TOPICS)}")
    elif args.command == "facts":
        print("\n\n".join(describe(f) for f in data["facts"]))
    elif args.command == "cheatsheet":
        CHEATSHEET.write_text(cheatsheet(data))
        print(f"wrote {CHEATSHEET.name}: {len(data['entries'])} commands, {len(data['facts'])} facts")
    elif args.command == "check":
        problems = validate(data)
        print("\n".join(problems) if problems else
              f"ok: {len(data['entries'])} commands and {len(data['facts'])} facts")
        sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
