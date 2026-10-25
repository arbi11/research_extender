"""Collect reproduction runs into a versioned evidence record and an audit page.

The record keeps each level's status, every acceptance and comparator result,
mesh sizes, engine identity and artifact digests. It never upgrades a failed or
unassessed level; the graph marks only examples with a passing level as
numerically checked, citing this record.
"""
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

from .reproduce import LEVELS, load_cases

DATA = Path(__file__).parent / "data"
RECORD = DATA / "verification" / "official_reproductions.json"


def _target_row(result):
    target = result.get("target", {})
    if "assessor" in result:
        row = {"assessor": result["assessor"], "status": result["status"], "acceptance": True}
        for key in ("rms_error_ratio", "maximum_rms_error_ratio", "published_rms_error_ratio", "applied_field_magnitude_T",
                    "measured_applied_field_T", "absolute_tolerance_N_m", "error"):
            if key in result:
                row[key] = result[key]
        if result.get("rows"):
            row["rows"] = [{k: r[k] for k in ("step", "rotor_angle_deg", "expected_N_m", "observed_N_m", "passed")} for r in result["rows"]]
        if result.get("analytic_constants"):
            row["analytic_constants"] = [{k: c.get(k) for k in ("name", "compare", "relative_difference", "passed", "used_for_acceptance", "note")}
                                         for c in result["analytic_constants"]]
        return row
    return {"metric": result.get("metric"), "status": result["status"], "acceptance": target.get("used_for_acceptance", True),
            "kind": target.get("kind"), "observed": result.get("observed"), "expected": result.get("expected"),
            "relative_error": result.get("relative_error"), "absolute_tolerance": result.get("absolute_tolerance"),
            "quote": target.get("quote")}


def collect(results):
    results = Path(results)
    cases = {c["id"]: c for c in load_cases()}
    entries = []
    for identifier, case in cases.items():
        summary_path = results / identifier / "summary.json"
        if not summary_path.is_file():
            entries.append({"case_id": identifier, "example_id": case["example_id"], "title": case["title"], "status": "not_run"})
            continue
        summary = json.loads(summary_path.read_text())
        levels = {}
        for level in LEVELS:
            path = results / identifier / level / "report.json"
            if not path.is_file():
                continue
            report = json.loads(path.read_text())
            entry = {"status": report["status"]}
            if report.get("reason"):
                entry["reason"] = report["reason"]
            if report.get("checks"):
                entry["build"] = report["checks"]["build"].get("status")
                entry["build_differences"] = report["checks"]["build"].get("difference_count", 0)
                entry["solve"] = report["checks"]["solve"].get("status")
                entry["targets"] = [_target_row(r) for r in report["checks"]["targets"].get("results", [])]
                entry["mesh_nodes"] = {k: v for k, v in report["metrics"].items() if k.startswith("mesh_nodes")}
                entry["plan_sha256"] = report.get("plan_sha256")
                entry["report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                entry["engine_image_id"] = report.get("execution", {}).get("image_id")
                entry["errors"] = {k: v["error"] for k, v in report["checks"].items() if v.get("error")}
                if report.get("execution_error"):
                    entry["errors"]["execution"] = report["execution_error"]
            levels[level] = entry
        passed = [level for level, entry in levels.items() if entry["status"] == "passed"]
        attempted = [level for level, entry in levels.items() if entry["status"] in ("passed", "failed")]
        comparators = [level for level, entry in levels.items() if entry["status"] == "comparators_only"]
        status = ("passed" if attempted and len(passed) == len(attempted) else "failed" if attempted else
                  "comparators_only" if comparators else "not_attempted")
        entries.append({"case_id": identifier, "example_id": case["example_id"], "title": case["title"], "source_url": case["source_url"],
                        "status": status,
                        "levels_passed": passed, "levels": levels,
                        "rebuild_matches_original": summary.get("level_comparison", {}).get("rebuild"),
                        "article_model_vs_original": {"difference_count": summary["article_model_vs_original"]["difference_count"]}
                        if summary.get("article_model_vs_original") else None})
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "catalog_sha256": hashlib.sha256((DATA / "reproductions.json").read_bytes()).hexdigest(),
            "scope": "Official FEMM examples reproduced from the unmodified model file (original), from the brain's stored model record "
                     "(rebuild) and from article-stated parameters (article). Comparators are reported but never used for acceptance.",
            "llm_calls": 0, "cases": entries,
            "totals": {"cases": len(entries), "passed": sum(e["status"] == "passed" for e in entries),
                       "failed": sum(e["status"] == "failed" for e in entries),
                       "comparators_only": sum(e["status"] == "comparators_only" for e in entries),
                       "level_passes": {level: sum(level in e.get("levels_passed", []) for e in entries) for level in LEVELS}}}


def _format(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return html.escape(str(value)) if value is not None else "—"


def render_html(record):
    """A dependency-free audit page; every number comes from the record."""
    rows = []
    for case in record["cases"]:
        cells = []
        for level in LEVELS:
            entry = case.get("levels", {}).get(level)
            status = entry["status"] if entry else "not_run"
            title = html.escape(entry.get("reason", "")) if entry else ""
            cells.append(f'<td class="s-{status}" title="{title}">{status.replace("_", " ")}</td>')
        detail = []
        for level in LEVELS:
            entry = case.get("levels", {}).get(level)
            if not entry or not entry.get("targets"):
                continue
            items = []
            for t in entry["targets"]:
                if "assessor" in t:
                    extra = f'RMS ratio {_format(t["rms_error_ratio"])}' if "rms_error_ratio" in t else f'|B| {_format(t.get("applied_field_magnitude_T"))} T'
                    items.append(f'<li class="s-{t["status"]}">{html.escape(t["assessor"])}: {t["status"]} ({extra})</li>')
                else:
                    role = "" if t["acceptance"] else " <em>comparator</em>"
                    items.append(f'<li class="s-{t["status"]}">{html.escape(t["metric"])}: {_format(t["observed"])} vs {_format(t["expected"])}'
                                 f' ({_format(t["relative_error"] * 100 if t["relative_error"] is not None else None)}%){role}</li>')
            detail.append(f'<div><b>{level}</b> · mesh {html.escape(json.dumps(entry.get("mesh_nodes", {})))}<ul>{"".join(items)}</ul></div>')
        rows.append(f'<tr><td><a href="{html.escape(case.get("source_url", "#"))}">{html.escape(case["title"])}</a><br><code>{case["case_id"]}</code></td>'
                    + "".join(cells) + f'<td><details><summary>{len(detail)} level(s)</summary>{"".join(detail)}</details></td></tr>')
    totals = record["totals"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Official Example Reproductions</title><style>
:root{{--bg:#fbfaf7;--fg:#1f2a28;--muted:#5d6b68;--line:#d9dcd6;--pass:#1d6b4f;--fail:#a1321b;--na:#6c6c6c}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151917;--fg:#e8ece9;--muted:#9aa7a3;--line:#2f3633;--pass:#5fc296;--fail:#f08a6c;--na:#9a9a9a}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:24px 16px}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid var(--line);padding:8px;vertical-align:top;text-align:left}}
.s-passed{{color:var(--pass)}}.s-failed{{color:var(--fail)}}.s-not_applicable,.s-not_attempted,.s-not_run{{color:var(--na)}}
code{{font-size:12px;color:var(--muted)}}ul{{margin:4px 0 8px 18px;padding:0}}p{{color:var(--muted)}}
</style></head><body><h1>Official FEMM example reproductions</h1>
<p>{html.escape(record["scope"])} Generated {html.escape(record["generated_at"])}; catalog {record["catalog_sha256"][:12]}; LLM calls: {record["llm_calls"]}.</p>
<p><b>{totals["passed"]}</b> of {totals["cases"]} cases pass every attempted level; {totals["failed"]} fail; {totals.get("comparators_only", 0)} have comparators only. Level passes — original {totals["level_passes"]["original"]},
rebuild {totals["level_passes"]["rebuild"]}, article {totals["level_passes"]["article"]}.</p>
<div style="overflow-x:auto"><table><thead><tr><th>Case</th><th>Original</th><th>Rebuild</th><th>Article</th><th>Results</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div></body></html>
"""


def audit(record, registry_path=DATA / "verification" / "reviewed_reproduction_failures.json"):
    """Accept only reviewed failures; flag new failures and reviewed ones that now pass."""
    registry = json.loads(Path(registry_path).read_text())["failures"]
    reviewed = {(f["case_id"], level, metric) for f in registry for level in f["levels"] for metric in f["metrics"]}
    observed = set()
    for case in record["cases"]:
        for level, entry in case.get("levels", {}).items():
            if entry["status"] != "failed":
                continue
            failed = [t["metric"] or t.get("assessor") for t in entry.get("targets", []) if t["acceptance"] and t["status"] != "passed"]
            if entry.get("build") == "failed" or entry.get("errors") or not failed:
                failed.append("build-or-execution")
            observed |= {(case["case_id"], level, metric) for metric in failed}
    unreviewed = sorted(observed - reviewed)
    resolved = sorted(reviewed - observed)
    return {"status": "passed" if not unreviewed and not resolved else "failed",
            "accepted_reviewed_failures": sorted(observed & reviewed), "unreviewed_failures": unreviewed,
            "reviewed_failures_now_passing": resolved,
            "note": "A passing audit means every failure is a reviewed, recorded finding, not that all published targets passed."}


def write(results, output=RECORD, html_output=None):
    record = collect(results)
    record["audit"] = audit(record)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=1) + "\n")
    if html_output:
        Path(html_output).write_text(render_html(record))
    return record
