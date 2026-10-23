"""Repeatable live natural-language CAD -> preview -> Docker FEMM evaluation.

Run inside the development FEMM container. Each case has an isolated application
and job directory, retains normal application limits, and uses real providers.
No live calls occur without the explicit --live command-line flag.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import tempfile
import time
import zipfile

from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.main import Settings, create_app
from app.plan import GeometryPlan


DEFAULT_CASES = Path(__file__).with_name("prompt_cases.json")


class PromptCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    category: str = Field(pattern=r"^(create|modify|clarify)$")
    tags: list[str] = Field(min_length=1, max_length=10)
    prompts: list[str] = Field(min_length=1, max_length=2)
    expected: GeometryPlan | None
    current: GeometryPlan | None = None

    @model_validator(mode="after")
    def complete_case(self):
        if any(not 5 <= len(prompt) <= 3000 for prompt in self.prompts):
            raise ValueError("Every prompt must have 5 to 3000 characters.")
        if (self.expected is None) != (self.category == "clarify"):
            raise ValueError("Clarification cases alone must omit a golden model.")
        if (self.current is not None) != (self.category == "modify"):
            raise ValueError("Each modification needs its own explicit current model.")
        return self


def load_cases(path: Path = DEFAULT_CASES) -> list[PromptCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not 1 <= len(data) <= 80:
        raise ValueError("The batch dataset must contain 1 to 80 cases.")
    cases = [PromptCase.model_validate(value) for value in data]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Batch case ids must be unique.")
    return cases


def _number(value):
    return round(float(value), 9)


def _point(point):
    return [_number(point.x), _number(point.y)]


def _sort(values):
    return sorted(values, key=lambda value: json.dumps(value, sort_keys=True))


def _shape(shape):
    if shape.type == "circle":
        return {"type": "circle", "center": _point(shape.center), "radius": _number(shape.radius)}
    if shape.type == "rectangle":
        points = [[_number(shape.x_min), _number(shape.y_min)], [_number(shape.x_max), _number(shape.y_min)],
                  [_number(shape.x_max), _number(shape.y_max)], [_number(shape.x_min), _number(shape.y_max)]]
    else:
        points = [_point(point) for point in shape.points]
    # The same polygon may start at any vertex and run in either direction.
    cycles = [sequence[index:] + sequence[:index] for sequence in (points, list(reversed(points))) for index in range(len(points))]
    return {"type": "polygon", "points": min(cycles)}


def physical_signature(data: dict | GeometryPlan) -> dict:
    """Independent physical comparison, with names and polygon order removed.

    Circuit membership is retained even for equal-current circuits. A force
    target is resolved to its region's shape rather than compared by its id.
    """
    plan = GeometryPlan.model_validate(data.model_dump() if isinstance(data, GeometryPlan) else data)
    materials = {"air": 1.0, **{item.id: _number(item.relative_permeability) for item in plan.materials}}
    circuits = {item.id: {"current_a": _number(item.current_a),
                         "regions": _sort([_shape(region.shape) for region in plan.regions if region.circuit == item.id])}
                for item in plan.circuits}
    regions = [{"shape": _shape(region.shape), "label": _point(region.label),
                "permeability": materials[region.material], "circuit": circuits.get(region.circuit),
                "turns": region.turns, "mesh_mm": _number(region.mesh_mm or plan.mesh_mm)} for region in plan.regions]
    force_target = next((_shape(region.shape) for region in plan.regions if region.id == plan.force_region), None)
    return {"physics": plan.physics, "units": plan.units, "depth_mm": _number(plan.depth_mm),
            "mesh_mm": _number(plan.mesh_mm), "domain": _shape(plan.domain), "air_label": _point(plan.air_label),
            "circuits": _sort(list(circuits.values())), "regions": _sort(regions),
            "probes": _sort([_point(point) for point in plan.probes]), "force_target": force_target}


def _percentile(values, percentile):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * percentile
    lower = math.floor(position)
    return round(values[lower] + (values[math.ceil(position)] - values[lower]) * (position - lower), 4)


def summarize(rows: list[dict], planned: int | None = None) -> dict:
    counts = Counter(row["status"] for row in rows)
    latencies = [row["latency_seconds"] for row in rows if isinstance(row.get("latency_seconds"), (int, float))]
    failures = Counter(failure["stage"] for row in rows for failure in row.get("failures", []))
    result = {"completed": len(rows), "passed": counts["passed"], "failed": counts["failed"], "errors": counts["error"],
              "pass_rate": round(counts["passed"] / len(rows), 4) if rows else None,
              "p50_latency_seconds": round(statistics.median(latencies), 4) if latencies else None,
              "p95_latency_seconds": _percentile(latencies, .95),
              "repair_attempts": sum(max(0, row.get("attempts", 0) - 1) for row in rows),
              "stage_failures": dict(sorted(failures.items()))}
    if planned is not None:
        result["planned"] = planned
        result["remaining"] = max(0, planned - len(rows))
    return result


def save_report(path: Path, report: dict):
    """Replace a checkpoint atomically; no failure rows are discarded."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _redact(value, settings):
    if isinstance(value, dict):
        return {key: _redact(item, settings) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, settings) for item in value]
    if isinstance(value, str):
        for secret in (settings.ai_key, settings.token):
            if secret:
                value = value.replace(secret, "[redacted]")
        value = re.sub(r"(?i)(Bearer\s+)[^\s\"']+", r"\1[redacted]", value)
        return value
    return value


class StageFailure(Exception):
    pass


def _failure(row, stage, message):
    row["failures"].append({"stage": stage, "error": str(message)[:1500]})


def _request(client, method, endpoint, stage, row, expected_status, **kwargs):
    started = time.monotonic()
    response = client.request(method, endpoint, **kwargs)
    row["stages"][stage] = {"http_status": response.status_code, "latency_seconds": round(time.monotonic() - started, 4)}
    if response.status_code != expected_status:
        try:
            detail = response.json().get("detail", "Unexpected response")
        except (ValueError, AttributeError):
            detail = response.text[:1000]
        raise StageFailure(f"HTTP {response.status_code}: {detail}")
    return response


def _wait_job(client, identifier, headers, stage, row, timeout):
    started = time.monotonic()
    deadline = started + timeout + 30
    while time.monotonic() < deadline:
        response = client.get("/api/jobs/" + identifier, headers=headers)
        if response.status_code != 200:
            row["stages"][stage]["poll_http_status"] = response.status_code
            raise StageFailure("Could not read the temporary native job")
        job = response.json()
        if job["status"] in {"completed", "failed"}:
            row["jobs"][stage] = job
            row["stages"][stage]["native_latency_seconds"] = round(time.monotonic() - started, 4)
            if job["status"] == "failed":
                raise StageFailure(job.get("error") or "Native job failed")
            return job
        time.sleep(.1)
    raise StageFailure("The temporary native job exceeded the evaluation deadline")


def _save_logs(store, row, directory, prefix, settings):
    for stage, job in row.get("jobs", {}).items():
        identifier = job.get("id", "")
        if not re.fullmatch(r"[0-9a-f]{32}", identifier):
            continue
        source = store.root / identifier / "solver.log"
        if source.is_file() and (row["failures"] or job.get("status") == "failed"):
            with source.open("r", encoding="utf-8", errors="replace") as stream:
                content = stream.read(128 * 1024)
                if stream.read(1):
                    content += "\n[Evaluation log copy truncated at 128 KiB.]\n"
            target = directory / (prefix + "-" + stage + "-solver.log")
            target.write_text(_redact(content, settings), encoding="utf-8")
            row["artifacts"].append(target.name)


def run_case(case: PromptCase, round_number: int, settings: Settings, artifact_dir: Path) -> dict:
    """Exercise production routes with a separate store and normal limits."""
    started = time.monotonic()
    variant = (round_number - 1) % len(case.prompts)
    row = {"id": case.id, "category": case.category, "tags": case.tags, "round": round_number,
           "prompt_variant": variant + 1, "prompt": case.prompts[variant], "status": "failed",
           "attempts": 0, "stages": {}, "failures": [], "jobs": {}, "artifacts": []}
    prefix = f"round-{round_number:02d}-{case.id}"
    stage = "setup"
    try:
        with tempfile.TemporaryDirectory(prefix="femm-batch-") as directory:
            local_settings = replace(settings, data_dir=Path(directory))
            app = create_app(local_settings)
            headers = {"Authorization": "Bearer " + settings.token}
            try:
                with TestClient(app) as client:
                    stage = "planner"
                    body = {"prompt": row["prompt"]}
                    if case.current is not None:
                        body["current"] = case.current.model_dump(mode="json")
                    proposal = _request(client, "POST", "/api/plan/interpret", stage, row, 200, json=body, headers=headers).json()
                    row["proposal"] = proposal
                    row["attempts"] = proposal.get("attempts", 0)
                    row["function_trace"] = proposal.get("function_trace", proposal.get("decision_trace", []))
                    row["actual_spec"] = proposal.get("spec")
                    if any(trace.get("stage") not in {"routing", "function_error"} for trace in row["function_trace"]):
                        _failure(row, "function_trace", "Unexpected geometry-approval decision stage was used")
                    if case.expected is None:
                        stage = "clarification"
                        if proposal.get("spec") is not None or not proposal.get("questions"):
                            raise StageFailure("Expected a clarification with no applied model and at least one question")
                        with closing(app.state.store.connect()) as database:
                            count = database.execute("SELECT count(*) FROM jobs").fetchone()[0]
                        if count:
                            raise StageFailure("A clarification unexpectedly created a native job")
                        row["no_job_created"] = True
                    else:
                        if proposal.get("spec") is None or proposal.get("questions"):
                            raise StageFailure("The complete supported request did not produce an executable model without questions")
                        stage = "physical_signature"
                        actual = GeometryPlan.model_validate(proposal["spec"])
                        actual_signature = physical_signature(actual)
                        expected_signature = physical_signature(case.expected)
                        row["physical_match"] = actual_signature == expected_signature
                        if not row["physical_match"]:
                            different = [key for key in expected_signature if actual_signature.get(key) != expected_signature[key]]
                            row["different_physical_fields"] = different
                            _failure(row, stage, "Generated model differs from the explicit request: " + ", ".join(different))
                        stage = "preview"
                        response = _request(client, "POST", "/api/plan/preview", stage, row, 200, json=proposal["spec"], headers=headers)
                        if "<svg" not in response.text:
                            raise StageFailure("The generated model has no SVG preview")
                        preview = artifact_dir / (prefix + "-geometry.svg")
                        preview.write_text(_redact(response.text, settings), encoding="utf-8")
                        row["artifacts"].append(preview.name)
                        stage = "mesh"
                        submitted = _request(client, "POST", "/api/plan/mesh", stage, row, 202, json=proposal["spec"], headers=headers).json()
                        row["jobs"][stage] = submitted
                        mesh = _wait_job(client, submitted["id"], headers, stage, row, settings.timeout)
                        row["mesh"] = mesh["result"]
                        if mesh["result"].get("status") != "passed":
                            raise StageFailure("The native mesh inspection did not pass")
                        stage = "solve"
                        submitted = _request(client, "POST", "/api/plan/jobs", stage, row, 202,
                                             json={"spec": proposal["spec"], "mesh_job_id": mesh["id"]}, headers=headers).json()
                        row["jobs"][stage] = submitted
                        solved = _wait_job(client, submitted["id"], headers, stage, row, settings.timeout)
                        row["quantities"] = solved["result"]
                        stage = "export"
                        response = _request(client, "GET", "/api/jobs/" + solved["id"] + "/download.zip", stage, row, 200, headers=headers)
                        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                            names = archive.namelist()
                            required = {"spec.json", "model.py", "model.lua", "model.fem", "model.ans", "mesh.svg", "mesh-report.json", "results.json"}
                            if not required <= set(names):
                                raise StageFailure("The solved export is missing required native artifacts")
                            if json.loads(archive.read("mesh-report.json")).get("status") != "passed":
                                raise StageFailure("The exported mesh report did not pass")
                        target = artifact_dir / (prefix + ".zip")
                        target.write_bytes(response.content)
                        row["artifacts"].append(target.name)
                        row["archive_members"] = names
                    row["status"] = "failed" if row["failures"] else "passed"
            except StageFailure as error:
                _failure(row, stage, str(error))
                row["status"] = "failed"
            except Exception as error:
                # Exceptions can contain provider URLs or credentials. Keep only
                # the exception type; the API's safe errors are captured above.
                _failure(row, stage, "Unexpected " + type(error).__name__ + "; private exception details were not recorded")
                row["status"] = "error"
            finally:
                _save_logs(app.state.store, row, artifact_dir, prefix, settings)
    except Exception as error:
        _failure(row, stage, "Evaluation setup/cleanup failed with " + type(error).__name__)
        row["status"] = "error"
    row["latency_seconds"] = round(time.monotonic() - started, 4)
    return _redact(row, settings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True, help="Explicitly authorize real provider and native solver calls")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for name, value, maximum in (("rounds", args.rounds, 5), ("max-cases", args.max_cases, 80), ("workers", args.workers, 3)):
        if not 1 <= value <= maximum:
            parser.error(f"--{name} must be between 1 and {maximum}")
    cases = load_cases()[:args.max_cases]
    settings = Settings.from_env()
    if not settings.ai_key:
        parser.error("Set OPENROUTER_API_KEY in the environment; never pass it on the command line")
    if not settings.start_worker:
        parser.error("Live batch evaluation needs the normal native solver worker enabled")
    if not shutil.which(settings.femm_lua):
        parser.error("Run the batch inside the development FEMM container with the configured native solver installed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.output.with_name(args.output.stem + "-artifacts")
    if args.output.exists() or artifact_dir.exists():
        parser.error("Use a fresh output JSON path and artifact directory; earlier evidence is never overwritten")
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write("{}\n")
    artifact_dir.mkdir()
    report = {"schema_version": 1, "source": "live_openrouter_and_native_femm", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "dataset_sha256": hashlib.sha256(DEFAULT_CASES.read_bytes()).hexdigest(),
              "drafting_model": settings.ai_model, "decision_model": settings.decision_model,
              "rounds_requested": args.rounds, "workers": args.workers, "selected_ids": [case.id for case in cases],
              "artifact_directory": artifact_dir.name, "rows": [], "round_summaries": [],
              "limitations": ["Public development cases, not an independently labeled holdout.",
                              "Passing meshing and solving does not establish numerical accuracy or convergence.",
                              "All failed and error attempts are retained; no whole-case retries replace evidence."]}
    planned = len(cases) * args.rounds
    report["summary"] = summarize([], planned)
    save_report(args.output, report)
    for round_number in range(1, args.rounds + 1):
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="femm-batch") as executor:
            pending = {executor.submit(run_case, case, round_number, settings, artifact_dir): case for case in cases}
            for future in as_completed(pending):
                try:
                    row = future.result()
                except Exception as error:
                    # Retain even an unexpected evaluator failure as its own
                    # failed attempt instead of silently dropping a paid case.
                    case = pending[future]
                    row = {"id": case.id, "category": case.category, "round": round_number, "status": "error",
                           "attempts": 0, "latency_seconds": 0, "stages": {}, "artifacts": [],
                           "failures": [{"stage": "evaluator", "error": "Unexpected " + type(error).__name__ + "; private details were not recorded"}]}
                report["rows"].append(row)
                report["summary"] = summarize(report["rows"], planned)
                save_report(args.output, report)
                print(f"round {round_number} {row['id']}: {row['status']} ({row['latency_seconds']:.2f}s)", flush=True)
        report["round_summaries"].append({"round": round_number, **summarize([row for row in report["rows"] if row["round"] == round_number])})
        save_report(args.output, report)
    report["status"] = "complete"
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save_report(args.output, report)
    print(json.dumps(report["summary"], indent=2), flush=True)
    return int(any(row["status"] != "passed" for row in report["rows"]))


if __name__ == "__main__":
    sys.exit(main())
