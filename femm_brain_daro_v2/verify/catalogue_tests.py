"""Turn catalogue items into Python test batches, run them in FEMM and record the outcomes.

Each item's "test" is the body of a Python function: it raises to fail and returns an
observation (anything but None) to pass. Items without a test give "untested_because".
Used by run_femm.py ("entries" and "tests").
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path


BATCHES = ("facts", "mi", "mo", "general", "restart")
# Tests per FEMM process. Headless, mo_close frees nothing (about 6 MB per loaded solution), so a
# long batch would grow towards the container's 768 MB; a batch with more tests is split.
CHUNK = 25


def batch_of(item: dict) -> str:
    """Tests share one FEMM session per batch; a test that closes or reopens FEMM itself goes to
    the restart batch, so it cannot disturb others. (The harness also relaunches FEMM after any
    test that leaves it closed or dead.)"""
    name = item["name"]
    test = item.get("test") or ""
    if "openfemm(" in test or "closefemm(" in test:
        return "restart"
    if name.startswith("fact:"):
        return "facts"
    if name.startswith("mi_"):
        return "mi"
    if name.startswith("mo_"):
        return "mo"
    return "general"


def render(items: list[dict]) -> str:
    """Python body for one batch: run(name, source) per test. The harness compiles each source
    on its own, so a test with a syntax error fails alone instead of stopping the batch."""
    return "\n".join(f'run({item["name"]!r}, {item["test"]!r})' for item in items if item.get("test"))


def run_items(run_batch, summarize, items: list[dict], prefix: str, only=None) -> dict:
    """Run the tests of `items` grouped into batches; return {name: (status, detail)}."""
    chosen = [i for i in items if i.get("test") and (not only or i["name"] in only)]
    results = {}
    for batch in BATCHES:
        members = [i for i in chosen if batch_of(i) == batch]
        for start in range(0, len(members), CHUNK):
            part = members[start:start + CHUNK]
            label = prefix + batch + (str(start // CHUNK + 1) if len(members) > CHUNK else "")
            found, code, output, seconds = run_batch(label, render(part))
            summarize(label, found, code, output, seconds)
            for name in [i["name"] for i in part if i["name"] not in found]:
                found[name] = ("fail", "no result: the batch stopped before this test")
            results.update(found)
    return results


def run_entries_file(run_batch, summarize, path: Path, only=None) -> None:
    """Try the tests in a draft entries file without touching the catalogue."""
    items = json.loads(path.read_text())
    run_items(run_batch, summarize, items, prefix=path.stem + "-", only=only)


def run_catalogue_tests(run_batch, summarize, catalogue: Path, image: str, tag: str, only=None) -> None:
    data = json.loads(catalogue.read_text())
    items = data["facts"] + data["entries"]
    results = run_items(run_batch, summarize, items, prefix="", only=only)
    today = datetime.date.today().isoformat()
    for item in items:
        if only and item["name"] not in only:
            continue
        if item.get("test"):
            status, detail = results.get(item["name"], ("fail", "not run"))
            item["verified"] = {"status": status, "observed": detail, "engine": tag, "date": today}
        else:
            item["verified"] = {"status": "not run", "observed": item.get("untested_because", ""),
                                "engine": tag, "date": today}
    data["about"]["engine"]["image"] = image
    counts = {}
    for item in items:
        counts[item["verified"]["status"]] = counts.get(item["verified"]["status"], 0) + 1
    data["about"]["verification"] = {"date": today, "counts": counts}
    catalogue.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print("catalogue updated:", counts)
