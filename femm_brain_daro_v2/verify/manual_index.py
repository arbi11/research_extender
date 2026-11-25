"""Index the commands documented in the FEMM 5 manual's magnetics scripting sections.

Usage:  python3 -I manual_index.py MANUAL_TEXT [--out manual_index.json]

MANUAL_TEXT is the text extraction of the FEMM 5 manual (kept outside the repository).
For every bullet that starts with a command name, the index records the name, the call
exactly as printed, the manual section and the line number. Only the sections the
catalogue covers are read: 3.3 to 3.5 (general) and 3.6 to 3.7 (magnetics).

It also records, under "mentioned", every engine function (from results/inventory.txt)
that the manual names as a call anywhere in its text, such as create(doctype) inside the
newdocument bullet, with the first line it appears on. Method calls such as os.chdir(path)
do not count. Lines are counted as Python's str.splitlines() counts them: page breaks in the
text end a line too, so the numbers run ahead of grep -n.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SECTION = re.compile(r"^(3\.\d+(?:\.\d+)?)\s+(\S.*?)\s*$")
BULLET = re.compile(r"^\s*•\s*([A-Za-z_][A-Za-z0-9_]*)(.*)$")
COVERED = ("3.3", "3.4", "3.5", "3.6", "3.7")


def covered(number: str) -> bool:
    return any(number == c or number.startswith(c + ".") for c in COVERED)


def index(lines: list[str]) -> list[dict]:
    entries, section, title = [], None, None
    for number, line in enumerate(lines, start=1):
        heading = SECTION.match(line)
        if heading:
            section, title = heading.group(1), heading.group(2)
            continue
        if section is None or not covered(section):
            continue
        bullet = BULLET.match(line)
        if not bullet:
            continue
        name, rest = bullet.group(1), bullet.group(2)
        call = name
        if rest.lstrip().startswith("("):
            text, follow = rest.lstrip(), number
            while ")" not in text and follow < len(lines) and follow - number < 4:
                text += " " + lines[follow].strip()
                follow += 1
            call = name + text[: text.index(")") + 1] if ")" in text else name + text
            call = re.sub(r"\s+", " ", call.replace("- ", "-"))
        entries.append({"name": name, "call": call, "section": section,
                        "section_title": title, "line": number})
    return entries


def mentioned(lines: list[str], functions: set[str]) -> dict[str, int]:
    """First line on which the manual writes each engine function as a call: name( ..."""
    found = {}
    for number, line in enumerate(lines, start=1):
        for name in re.findall(r"(?<![\w.])([A-Za-z_][A-Za-z0-9_]*)\s*\(", line):
            if name in functions and name not in found:
                found[name] = number
    return dict(sorted(found.items()))


def main() -> None:
    results = Path(__file__).with_name("results")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manual")
    parser.add_argument("--out", default=str(results / "manual_index.json"))
    args = parser.parse_args()
    raw = Path(args.manual).read_bytes()
    lines = raw.decode("utf-8").splitlines()
    entries = index(lines)
    functions = {line.split()[1] for line in (results / "inventory.txt").read_text().splitlines()
                 if line.startswith("FN ")}
    out = {"manual_text_sha256": hashlib.sha256(raw).hexdigest(), "entries": entries,
           "mentioned": mentioned(lines, functions)}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(entries)} documented commands, {len(out['mentioned'])} engine functions named -> {args.out}")


if __name__ == "__main__":
    main()
