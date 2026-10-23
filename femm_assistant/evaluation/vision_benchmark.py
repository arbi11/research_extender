"""Bounded live A/B comparison of optional, advisory geometry observations.

These public development fixtures compare the current image-plus-typed-plan
review, not image-only reasoning, numerical accuracy, or production approval.
No provider call occurs without --live. Calls run sequentially with no retries.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Literal

import httpx

from app import plan_generate, vision
from app.plan import GeometryPlan, wire_example
from .semantic import CASES, save_report


DEFAULT_MODELS = ("qwen/qwen3-vl-30b-a3b-instruct", "google/gemini-3.1-flash-lite")
MAX_CASES = 12
MAX_MODELS = 2
USAGE_METRICS = ("cost", "prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens")


@dataclass(frozen=True)
class Case:
    id: str
    prompt: str
    candidate: GeometryPlan
    expected: Literal["matches", "issues"]
    rationale: str

    def fixture(self) -> dict:
        return {"id": self.id, "prompt": self.prompt, "candidate": self.candidate.model_dump(mode="json"),
                "expected": self.expected, "rationale": self.rationale}


def _mirror_shape(shape: dict) -> None:
    if shape["type"] == "rectangle":
        shape["x_min"], shape["x_max"] = -shape["x_max"], -shape["x_min"]
    elif shape["type"] == "circle":
        shape["center"]["x"] *= -1
    else:
        for point in shape["points"]:
            point["x"] *= -1


def build_cases() -> list[Case]:
    """Derive six valid candidates with explicit, public comparison labels."""
    wire = wire_example().model_dump(mode="json")
    wire["probes"] = [{"x": 3.0, "y": 0.0}]
    wire_prompt = ("Create a 2D planar DC model in millimeters: a circular conductor centered at (0,0), "
        "radius 1 mm, relative permeability 1, current +1 A with one positive turn, out-of-plane depth 20 mm. "
        "Use an enclosing circular air domain of radius 10 mm with A=0, air label (5,0), global mesh 0.5 mm "
        "and a field probe at (3,0).")
    wrong_radius = copy.deepcopy(wire)
    wrong_radius["regions"][0]["shape"]["radius"] = 1.6
    historical = json.loads(CASES.read_text())
    asymmetric = next(item for item in historical if item["id"] == "asymmetric_core_faithful")
    core = copy.deepcopy(asymmetric["candidate"])
    missing_return = copy.deepcopy(core)
    missing_return["regions"] = [region for region in missing_return["regions"] if region["id"] != "coil2"]
    reversed_return = copy.deepcopy(core)
    next(region for region in reversed_return["regions"] if region["id"] == "coil2")["turns"] = 50
    mirrored = copy.deepcopy(core)
    _mirror_shape(mirrored["domain"])
    mirrored["air_label"]["x"] *= -1
    for region in mirrored["regions"]:
        _mirror_shape(region["shape"])
        region["label"]["x"] *= -1
    for probe in mirrored["probes"]:
        probe["x"] *= -1
    entries = [
        ("correct_wire", wire_prompt, wire, "matches", "The radius-1 conductor, excitation, domain and probe match the request."),
        ("wrong_wire_radius", wire_prompt, wrong_radius, "issues", "The conductor radius is 1.6 mm instead of the requested 1 mm; the probe remains outside at x=3 mm."),
        ("correct_asymmetric_core", asymmetric["prompt"], core, "matches", "All three requested regions and signed circuit assignments are present."),
        ("missing_return_winding", asymmetric["prompt"], missing_return, "issues", "The requested negative-turn return winding is absent; the positive winding still uses the circuit."),
        ("reversed_return_turns", asymmetric["prompt"], reversed_return, "issues", "The return winding has +50 turns rather than -50. The typed plan deliberately accompanies the image."),
        ("mirrored_asymmetric_core", asymmetric["prompt"], mirrored, "issues", "The entire geometry, domain, labels and probe are reflected across x=0; the request specifies the original coordinates."),
    ]
    return [Case(case_id, prompt, GeometryPlan.model_validate(candidate), expected, rationale)
            for case_id, prompt, candidate, expected, rationale in entries]


def _total_or_none(values: list[int | float]) -> int | float | None:
    return sum(values) if values else None


def summarize(cases: list[Case], rows: list[dict], models: list[str]) -> dict:
    """Keep errors, unmeasured cases, and absent usage distinct from success/zero."""
    by_id = {case.id: case for case in cases}
    seen = set()
    per_model = {}
    for model in models:
        tally = Counter()
        statuses = Counter()
        metrics = {name: [] for name in USAGE_METRICS}
        latencies = []
        for row in rows:
            if row["requested_model"] not in models or row["id"] not in by_id:
                raise ValueError("Unknown benchmark model or fixture.")
            if row["requested_model"] != model:
                continue
            identity = (model, row["id"])
            if identity in seen:
                raise ValueError("Duplicate model/fixture evidence.")
            seen.add(identity)
            tally["attempted"] += 1
            elapsed = row.get("latency_seconds")
            if type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0:
                latencies.append(elapsed)
            usage = row.get("usage") or {}
            for name in USAGE_METRICS:
                value = usage.get(name)
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    metrics[name].append(value)
            if row.get("error"):
                tally["errors"] += 1
                continue
            advisory = row.get("advisory")
            if not isinstance(advisory, dict) or advisory.get("advisory") is not True or advisory.get("status") not in {"matches", "issues", "uncertain"}:
                raise ValueError("Missing valid advisory evidence; do not count empty rows as correct.")
            actual = advisory["status"]
            expected = by_id[row["id"]].expected
            tally["evaluated"] += 1
            statuses[actual] += 1
            tally["matches"] += int(actual == expected)
            tally["false_flags"] += int(expected == "matches" and actual == "issues")
            tally["missed_issues"] += int(expected == "issues" and actual == "matches")
            tally["uncertain"] += int(actual == "uncertain")
        per_model[model] = {
            **{name: tally[name] for name in ("attempted", "evaluated", "matches", "false_flags", "missed_issues", "uncertain", "errors")},
            "unmeasured": len(cases) - tally["attempted"],
            "status_counts": {name: statuses[name] for name in ("matches", "issues", "uncertain")},
            "cost": _total_or_none(metrics["cost"]),
            "cost_reported_rows": len(metrics["cost"]),
            "cost_unknown_rows": tally["attempted"] - len(metrics["cost"]),
            "tokens": {name: _total_or_none(metrics[name]) for name in USAGE_METRICS if name != "cost"},
            "tokens_reported_rows": {name: len(metrics[name]) for name in USAGE_METRICS if name != "cost"},
            "latency_seconds": {"total": _total_or_none(latencies),
                                "mean": sum(latencies) / len(latencies) if latencies else None,
                                "max": max(latencies) if latencies else None},
        }
    return per_model


async def run_benchmark(cases: list[Case], models: list[str], output: Path, *, key: str,
                        base_url: str = "https://openrouter.ai/api/v1") -> dict:
    """Create fresh evidence and checkpoint after each strictly sequential call."""
    if not key:
        raise ValueError("OPENROUTER_API_KEY is required for a live benchmark.")
    if not 1 <= len(cases) <= MAX_CASES or len({case.id for case in cases}) != len(cases):
        raise ValueError("Use 1 to 12 unique fixtures.")
    if not 1 <= len(models) <= MAX_MODELS or len(set(models)) != len(models):
        raise ValueError("Use one or two distinct image-input models.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write("{}\n")
    fixtures = [case.fixture() for case in cases]
    report = {
        "source": "live_public_geometry_advisory_ab", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "advisory_only": True, "production_gate": False, "sequential": True,
        "requested_models": models, "planned_calls": len(cases) * len(models),
        "fixture_sha256": hashlib.sha256(json.dumps(fixtures, sort_keys=True).encode()).hexdigest(),
        "fixtures": fixtures, "rows": [],
        "limitations": [
            "Public development fixtures, not an independent holdout or CAD-domain certification.",
            "The current review receives rendered geometry AND the exact typed plan; this is not an image-only benchmark.",
            "No native mesh is supplied. Geometry observations do not establish mesh validity, convergence or numerical accuracy.",
            "Labels describe advisory matches/issues only and never authorize or block production model execution.",
            "Errors and unmeasured cases are separate from correct answers; absent provider usage stays unknown, not zero.",
            "Reversed turns are a typed-assignment discrepancy and may not be distinguishable from the image alone.",
            "One call per model/fixture without retries; model variability and independent repetitions are not measured.",
        ],
    }
    report["summary"] = summarize(cases, [], models)
    save_report(output, report)
    preview_dir = output.with_name(output.stem + "-cases")
    preview_dir.mkdir(exist_ok=False)
    try:
        for case, fixture in zip(cases, report["fixtures"]):
            preview_path = preview_dir / (case.id + ".png")
            preview_path.write_bytes(vision._render_png(plan_generate.preview_svg(case.candidate)))
            fixture["geometry_png"] = str(preview_path.relative_to(output.parent))
    except vision.VisionError as exc:
        report["setup_error"] = {"status_code": exc.status_code, "message": exc.message}
        save_report(output, report)
        return report
    save_report(output, report)
    async with httpx.AsyncClient(follow_redirects=False) as client:
        for model in models:
            for case in cases:
                row = {"id": case.id, "requested_model": model, "expected": case.expected}
                started = time.perf_counter()
                try:
                    result = await vision.review(client, key, base_url, model, case.prompt, case.candidate)
                    row.update(advisory=result.model_dump(mode="json"), usage=dict(result.usage))
                except vision.VisionError as exc:
                    row["error"] = {"status_code": exc.status_code, "message": exc.message}
                row["latency_seconds"] = round(time.perf_counter() - started, 6)
                report["rows"].append(row)
                report["summary"] = summarize(cases, report["rows"], models)
                save_report(output, report)
                print(model + " / " + case.id + ": " + ("error" if row.get("error") else row["advisory"]["status"]), flush=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True, help="Explicitly enable paid provider calls")
    parser.add_argument("--output", type=Path, required=True, help="A new report path; existing reports are never replaced")
    parser.add_argument("--max-cases", type=int, default=6, help="Fixture limit, between 1 and 12 (default: 6)")
    parser.add_argument("--model", action="append", default=[], help="Image-input model; repeat up to twice for A/B")
    args = parser.parse_args(argv)
    if not 1 <= args.max_cases <= MAX_CASES:
        parser.error("--max-cases must be between 1 and 12")
    models = args.model or list(DEFAULT_MODELS)
    if not 1 <= len(models) <= MAX_MODELS or len(set(models)) != len(models) or any(not value.strip() for value in models):
        parser.error("Use one or two distinct --model values")
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        parser.error("Set OPENROUTER_API_KEY in the environment; never pass a key as a CLI argument")
    report = asyncio.run(run_benchmark(build_cases()[:args.max_cases], models, args.output, key=key,
                                      base_url=os.getenv("AI_BASE_URL", "https://openrouter.ai/api/v1")))
    print(json.dumps(report["summary"], indent=2))
    return int(bool(report.get("setup_error")) or any(item["errors"] for item in report["summary"].values()))


if __name__ == "__main__":
    sys.exit(main())
