"""Evaluate the production semantic gates; never change runtime thresholds.

Live runs send only the public cases to the configured Decisions provider.
No provider request is made without --live. Reports contain evidence, never keys.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app import decisions
from app.main import Settings
from app.plan import GeometryPlan


CASES = Path(__file__).with_name("semantic_cases.json")


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    category: str
    stage: Literal["preflight", "fidelity"]
    prompt: str = Field(min_length=5, max_length=3000)
    current: GeometryPlan | None = None
    candidate: GeometryPlan | None = None
    explanation: str = ""
    expected: dict[str, str | bool]
    expected_allowed: bool
    rationale: str

    @model_validator(mode="after")
    def consistent_labels(self):
        if self.stage == "preflight":
            if self.candidate is not None or self.expected.get("route") not in {"create", "modify", "clarify"}:
                raise ValueError("Preflight cases need an intent label and no candidate.")
            if not set(self.expected) <= {"route", "support", "sufficiency"}:
                raise ValueError("Unexpected preflight labels.")
            for name in ("support", "sufficiency"):
                if name in self.expected and type(self.expected[name]) is not bool:
                    raise ValueError("Binary labels must be booleans.")
            operation = "modify" if self.current is not None else "create"
            allowed = (self.expected["route"] == operation
                       and self.expected.get("support") is True
                       and self.expected.get("sufficiency") is True)
        else:
            if self.candidate is None or set(self.expected) != {"faithfulness"} or type(self.expected["faithfulness"]) is not bool:
                raise ValueError("Fidelity cases need a valid candidate and a binary faithfulness label.")
            allowed = self.expected["faithfulness"]
        if allowed != self.expected_allowed:
            raise ValueError("The expected gate result contradicts the question labels.")
        return self


def load_cases(path=CASES):
    cases = [Case.model_validate(item) for item in json.loads(Path(path).read_text())]
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("Cases must be nonempty and have unique IDs.")
    return cases


def save_report(path: Path, report: dict):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def gate(case: Case, traces: list[decisions.DecisionTrace], threshold: float):
    passed = []
    for trace in traces:
        if trace.question == "route":
            operation = "modify" if case.current is not None else "create"
            passed.append(trace.answer == operation and trace.probability >= threshold
                          and trace.confidence is not None and trace.confidence >= threshold)
        else:
            passed.append(trace.probabilities["yes"] >= threshold)
    return all(passed)


def summarize(cases: list[Case], rows: list[dict], threshold: float):
    by_id = {case.id: case for case in cases}
    counts = Counter()
    scores: dict[str, list[float]] = {}
    bins: dict[str, list[list[tuple[float, int]]]] = {}
    sweep = {str(value): Counter() for value in (.6, .7, .8, .9, .95, .99)}
    seen = set()
    for row in rows:
        case = by_id[row["id"]]
        if case.id in seen:
            raise ValueError("Duplicate evidence ID.")
        seen.add(case.id)
        if row.get("error"):
            counts["provider_errors"] += 1
            continue
        traces = [decisions.DecisionTrace.model_validate(item) for item in row["trace"]]
        required = {"route", "support", "sufficiency"} if case.stage == "preflight" else {"faithfulness"}
        if len(traces) != len(required) or {item.question for item in traces} != required or any(item.stage != case.stage for item in traces):
            raise ValueError("Incomplete or mismatched decision traces.")
        actual = gate(case, traces, threshold)
        counts["evaluated"] += 1
        counts["correct" if actual == case.expected_allowed else "incorrect"] += 1
        if actual and not case.expected_allowed:
            counts["false_accepts"] += 1
        if not actual and case.expected_allowed:
            counts["false_rejects"] += 1
        for value, tally in sweep.items():
            accepted = gate(case, traces, float(value))
            tally["false_accepts"] += int(accepted and not case.expected_allowed)
            tally["false_rejects"] += int(not accepted and case.expected_allowed)
        for trace in traces:
            if trace.question not in case.expected:
                continue
            label = case.expected[trace.question]
            if trace.question == "route":
                score = sum((p - int(name == label)) ** 2 for name, p in trace.probabilities.items())
                probability, outcome = trace.probability, int(trace.answer == label)
            else:
                probability, outcome = trace.probabilities["yes"], int(label)
                score = (probability - outcome) ** 2
            scores.setdefault(trace.question, []).append(score)
            bins.setdefault(trace.question, [[] for _ in range(10)])[min(int(probability * 10), 9)].append((probability, outcome))
    metrics = {}
    for name, values in scores.items():
        reliability = [{"range": [i / 10, (i + 1) / 10], "count": len(bucket),
                        "mean_probability": sum(p for p, _ in bucket) / len(bucket),
                        "observed_frequency": sum(y for _, y in bucket) / len(bucket)}
                       for i, bucket in enumerate(bins[name]) if bucket]
        metrics[name] = {"count": len(values), "brier_score": sum(values) / len(values),
                         "reliability": reliability}
    return {**{name: counts[name] for name in ("evaluated", "correct", "incorrect", "false_accepts", "false_rejects", "provider_errors")},
            "unmeasured": len(cases) - len(seen), "questions": metrics,
            "threshold_sweep": {value: dict(tally) for value, tally in sweep.items()}}


async def collect(cases, settings, checkpoint):
    rows = []
    async with httpx.AsyncClient() as client:
        for case in cases:
            started = time.monotonic()
            try:
                arguments = (client, settings.ai_key, settings.decision_base_url, settings.decision_model,
                             settings.decision_threshold, case.prompt, case.current)
                outcome = (await decisions.preflight(*arguments) if case.stage == "preflight"
                           else await decisions.fidelity(*arguments, case.candidate, case.explanation))
                row = {"id": case.id, "allowed": outcome.allowed,
                       "trace": [trace.model_dump() for trace in outcome.trace]}
            except decisions.DecisionError as exc:
                row = {"id": case.id, "error": {"status_code": exc.status_code, "message": exc.message}}
            row["seconds"] = round(time.monotonic() - started, 3)
            rows.append(row)
            checkpoint(rows)
            print(case.id + ": " + ("provider error" if row.get("error") else "measured"), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true", help="Validate labels/plans without contacting a provider")
    mode.add_argument("--live", action="store_true", help="Make paid Decisions requests using existing environment settings")
    mode.add_argument("--replay", type=Path, help="Rescore a saved report without provider calls")
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--max-cases", type=int, default=6, help="Live call cap (1-50), sequential; default 6")
    parser.add_argument("--case", action="append", default=[], help="Select case IDs; repeat as needed")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = load_cases(args.cases)
    dataset_hash = hashlib.sha256(args.cases.read_bytes()).hexdigest()
    if args.case:
        if set(args.case) - {case.id for case in cases}:
            parser.error("Unknown case ID")
        cases = [case for case in cases if case.id in args.case]
    if args.validate:
        print(json.dumps({"valid_cases": len(cases), "categories": dict(Counter(case.category for case in cases)),
                          "dataset_sha256": dataset_hash}, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required for live runs and replay")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Check output permissions before spending any provider credits. Preserve
    # prior evidence rather than silently overwriting another run.
    with args.output.open("x") as stream:
        stream.write("{}\n")
    if args.live:
        if not 1 <= args.max_cases <= 50:
            parser.error("--max-cases must be between 1 and 50")
        settings = Settings.from_env()
        if not settings.ai_key:
            parser.error("Set OPENROUTER_API_KEY in the environment; never pass it on the command line")
        decisions.validate_configuration(settings.decision_model, settings.decision_base_url, settings.decision_threshold)
        selected = cases[:args.max_cases]
        report = {"schema_version": 1, "source": "live_provider", "created_utc": datetime.now(timezone.utc).isoformat(),
                  "dataset_sha256": dataset_hash, "criteria_sha256": hashlib.sha256(Path(decisions.__file__).read_bytes()).hexdigest(),
                  "requested_model": settings.decision_model, "provider_base_url": settings.decision_base_url,
                  "threshold": settings.decision_threshold, "selected_ids": [case.id for case in selected], "rows": []}
        def checkpoint(rows):
            report["rows"] = rows
            report["summary"] = summarize(cases, rows, report["threshold"])
            save_report(args.output, report)
        asyncio.run(collect(selected, settings, checkpoint))
    else:
        report = json.loads(args.replay.read_text())
        if report["dataset_sha256"] != dataset_hash:
            parser.error("Report and dataset hashes differ; use the original dataset")
        if args.case:
            report["rows"] = [row for row in report["rows"] if row["id"] in args.case]
    report["summary"] = summarize(cases, report["rows"], report["threshold"])
    report["limitations"] = ["Public development seed cases; not an independently labeled holdout or CAD-domain certification.",
                             "Missing/error cases are unmeasured, never counted as correct rejections.",
                             "Binary Brier scores use P(yes); route uses unnormalized multiclass Brier and winner-probability reliability.",
                             "Threshold sweeps reuse the same evidence; they do not change application settings.",
                             "Semantic confidence does not measure field accuracy or numerical convergence."]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_report(args.output, report)
    print(json.dumps(report["summary"], indent=2))
    return int(bool(report["summary"]["provider_errors"] or report["summary"]["incorrect"]))


if __name__ == "__main__":
    sys.exit(main())
