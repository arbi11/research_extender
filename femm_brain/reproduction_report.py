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

from .reproduce import LEVELS, case_binding_digest, catalog_digest, load_cases, load_designs, load_papers

DATA = Path(__file__).parent / "data"
RECORD = DATA / "verification" / "official_reproductions.json"
DESIGN_RECORD = DATA / "verification" / "component_designs.json"
PAPER_RECORD = DATA / "verification" / "paper_replications.json"
RECORDS = {"reproductions": RECORD, "designs": DESIGN_RECORD, "papers": PAPER_RECORD}
REGISTRIES = {"reproductions": "reviewed_reproduction_failures.json", "designs": "reviewed_design_failures.json",
              "papers": "reviewed_paper_failures.json"}
TITLES = {"reproductions": "Official FEMM example reproductions", "designs": "Component designs",
          "papers": "Paper replications"}
SCOPES = {"reproductions": "Official FEMM examples reproduced from the unmodified model file (original), from the brain's stored model record "
                           "(rebuild), from article-stated parameters (article) and from reusable machine components (components). "
                           "Comparators are reported but never used for acceptance.",
          "designs": "New designs assembled only from reusable components and checked against closed-form results at the assembly level.",
          "papers": "Machines rebuilt from papers and reports with reusable components, from parameters whose provenance is recorded, "
                    "against targets frozen before solving. Comparators are reported but never used for acceptance."}


def catalog_cases(catalog):
    """Cases a record must cover: every reproduction and design, and every paper case past extraction."""
    if catalog == "papers":
        return [case for case in load_papers() if case.get("status") not in ("candidate", "extracting")]
    return load_cases() if catalog == "reproductions" else load_designs()


def _target_row(result):
    target = result.get("target", {})
    if "assessor" in result:
        row = {"assessor": result["assessor"], "status": result["status"], "acceptance": target.get("used_for_acceptance", True)}
        for key in ("rms_error_ratio", "maximum_rms_error_ratio", "published_rms_error_ratio", "applied_field_magnitude_T",
                    "measured_applied_field_T", "absolute_tolerance_N_m", "amplitude_imbalance", "winding_imbalance",
                    "predicted_sequence_check", "measured_sequence_check", "expected_sequence", "rotor_positions",
                    "metric", "statistic", "observed", "expected", "relative_error", "absolute_tolerance", "samples", "error"):
            if key in result:
                row[key] = result[key]
        if result.get("rows"):
            keep = ("step", "rotor_angle_deg", "expected_N_m", "observed_N_m", "point", "measured_T", "direction_deg",
                    "relative_vector_error", "phase", "predicted_phase_deg", "measured_phase_deg", "phase_error_deg",
                    "fundamental_Wb", "passed")
            row["rows"] = [{k: r[k] for k in keep if k in r} for r in result["rows"]]
        for key in ("predicted_applied_field_T", "predicted_field_direction_deg", "measured_field_direction_deg"):
            if key in result:
                row[key] = result[key]
        if result.get("analytic_constants"):
            row["analytic_constants"] = [{k: c.get(k) for k in ("name", "compare", "relative_difference", "passed", "used_for_acceptance", "note")}
                                         for c in result["analytic_constants"]]
        return row
    return {"metric": result.get("metric"), "status": result["status"], "acceptance": target.get("used_for_acceptance", True),
            "kind": target.get("kind"), "observed": result.get("observed"), "expected": result.get("expected"),
            "relative_error": result.get("relative_error"), "absolute_tolerance": result.get("absolute_tolerance"),
            "quote": target.get("quote")}


def collect(results, catalog="reproductions"):
    results = Path(results)
    cases = {c["id"]: c for c in catalog_cases(catalog)}
    entries = []
    for identifier, case in cases.items():
        summary_path = results / identifier / "summary.json"
        if not summary_path.is_file():
            entries.append({"case_id": identifier, "example_id": case.get("example_id"), "title": case["title"], "status": "not_run"})
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
                if "reference" in report["checks"]:
                    entry["reference"] = report["checks"]["reference"].get("status")
                    entry["reference_differences"] = report["checks"]["reference"].get("difference_count")
                entry["build_differences"] = report["checks"]["build"].get("difference_count", 0)
                entry["solve"] = report["checks"]["solve"].get("status")
                entry["targets"] = [_target_row(r) for r in report["checks"]["targets"].get("results", [])]
                entry["mesh_nodes"] = {k: v for k, v in report["metrics"].items() if k.startswith("mesh_nodes")}
                entry["plan_sha256"] = report.get("plan_sha256")
                if report.get("components_used"):
                    entry["components_used"] = report["components_used"]
                    entry["component_digests"] = report.get("component_digests")
                entry["case_binding_sha256"] = report.get("case_binding_sha256")
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
        entries.append({"case_id": identifier, "example_id": case.get("example_id"), "title": case["title"],
                        "source_url": case.get("source_url") or (case.get("source") or {}).get("url"),
                        "status": status, "coverage": case.get("coverage"), "case_binding_sha256": case_binding_digest(case),
                        "levels_passed": passed, "levels": levels,
                        "rebuild_matches_original": summary.get("level_comparison", {}).get("rebuild"),
                        "components_match_original": summary.get("level_comparison", {}).get("components"),
                        "article_model_vs_original": {"difference_count": summary["article_model_vs_original"]["difference_count"]}
                        if summary.get("article_model_vs_original") else None})
    return {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "catalog_sha256": catalog_digest(catalog),
            "catalog": catalog, "scope": SCOPES[catalog],
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
                    extra = (f'RMS ratio {_format(t["rms_error_ratio"])}' if "rms_error_ratio" in t else
                             f'largest phase error {_format(max(r["phase_error_deg"] for r in t["rows"]))} deg, imbalance '
                             f'{_format(t["amplitude_imbalance"])}' if "amplitude_imbalance" in t else
                             f'{html.escape(t["metric"])} {_format(t["observed"])} vs {_format(t["expected"])}'
                             + ('' if t["acceptance"] else ', comparator') if "statistic" in t else
                             f'|B| {_format(t.get("applied_field_magnitude_T"))} T')
                    items.append(f'<li class="s-{t["status"]}">{html.escape(t["assessor"])}: {t["status"]} ({extra})</li>')
                else:
                    role = "" if t["acceptance"] else " <em>comparator</em>"
                    items.append(f'<li class="s-{t["status"]}">{html.escape(t["metric"])}: {_format(t["observed"])} vs {_format(t["expected"])}'
                                 f' ({_format(t["relative_error"] * 100 if t["relative_error"] is not None else None)}%){role}</li>')
            detail.append(f'<div><b>{level}</b> · mesh {html.escape(json.dumps(entry.get("mesh_nodes", {})))}<ul>{"".join(items)}</ul></div>')
        rows.append(f'<tr><td><a href="{html.escape(case.get("source_url") or "#")}">{html.escape(case["title"])}</a><br><code>{case["case_id"]}</code></td>'
                    + "".join(cells) + f'<td><details><summary>{len(detail)} level(s)</summary>{"".join(detail)}</details></td></tr>')
    totals = record["totals"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(TITLES.get(record.get("catalog"), "Reproductions"))}</title><style>
:root{{--bg:#fbfaf7;--fg:#1f2a28;--muted:#5d6b68;--line:#d9dcd6;--pass:#1d6b4f;--fail:#a1321b;--na:#6c6c6c}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151917;--fg:#e8ece9;--muted:#9aa7a3;--line:#2f3633;--pass:#5fc296;--fail:#f08a6c;--na:#9a9a9a}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:24px 16px}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid var(--line);padding:8px;vertical-align:top;text-align:left}}
.s-passed{{color:var(--pass)}}.s-failed{{color:var(--fail)}}.s-not_applicable,.s-not_attempted,.s-not_run{{color:var(--na)}}
code{{font-size:12px;color:var(--muted)}}ul{{margin:4px 0 8px 18px;padding:0}}p{{color:var(--muted)}}
</style></head><body><h1>{html.escape(TITLES.get(record.get("catalog"), "Reproductions"))}</h1>
<p>{html.escape(record["scope"])} Generated {html.escape(record["generated_at"])}; catalog {record["catalog_sha256"][:12]}; LLM calls: {record["llm_calls"]}.</p>
<p><b>{totals["passed"]}</b> of {totals["cases"]} cases pass every attempted level; {totals["failed"]} fail; {totals.get("comparators_only", 0)} have comparators only. Level passes — original {totals["level_passes"]["original"]},
rebuild {totals["level_passes"]["rebuild"]}, article {totals["level_passes"]["article"]}, components {totals["level_passes"]["components"]}.</p>
<div style="overflow-x:auto"><table><thead><tr><th>Case</th><th>Original</th><th>Rebuild</th><th>Article</th><th>Components</th><th>Results</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div></body></html>
"""


def expected_levels(case, catalog):
    """The state each level must reach for a complete record."""
    if catalog in ("designs", "papers"):
        return {"article": "executed"}
    native = "executed" if case.get("source_model_sha256") else "not_applicable"
    article = "executed" if "assembly" in (case.get("article") or {}) else "not_attempted"
    components = "executed" if case.get("components") else "not_attempted"
    return {"original": native, "rebuild": native, "article": article, "components": components}


def audit(record, registry_path, catalog="reproductions"):
    """Complete, bound and bounded: the record must cover every expected case and level,
    agree with the current catalog, keep rebuilds identical to originals, and fail only
    where a reviewed failure stays within its frozen regression bound."""
    cases = catalog_cases(catalog)
    problems = []
    current_catalog = catalog_digest(catalog)
    if record.get("catalog_sha256") != current_catalog:
        problems.append("The record was produced from a different catalog version.")
    registry = json.loads(Path(registry_path).read_text())["failures"] if Path(registry_path).exists() else []
    reviewed = {(f["case_id"], level, m["metric"]): (f, m) for f in registry for level in f["levels"] for m in f["metrics"]}
    entries = {c["case_id"]: c for c in record.get("cases", [])}
    observed_failures, accepted = set(), []
    for case in cases:
        identifier = case["id"]
        entry = entries.get(identifier)
        if entry is None or entry.get("status") == "not_run":
            problems.append(f"{identifier}: missing from the record.")
            continue
        if entry.get("case_binding_sha256") != case_binding_digest(case):
            problems.append(f"{identifier}: recorded under a different case definition.")
        targets = {t["metric"]: t for t in case.get("targets", []) if t.get("used_for_acceptance", True)}
        for level, expectation in expected_levels(case, catalog).items():
            result = entry.get("levels", {}).get(level)
            if result is None:
                problems.append(f"{identifier}/{level}: missing level.")
                continue
            if expectation != "executed":
                if result["status"] != expectation:
                    problems.append(f"{identifier}/{level}: expected {expectation}, found {result['status']}.")
                continue
            if result["status"] not in ("passed", "failed", "comparators_only") or result.get("build") != "passed" \
                    or result.get("solve") != "passed" or result.get("errors"):
                problems.append(f"{identifier}/{level}: did not build, solve and assess completely ({result['status']}).")
                continue
            for target in result.get("targets", []):
                if not target["acceptance"] or target["status"] == "passed":
                    continue
                name = target.get("metric") or target.get("assessor")
                key = (identifier, level, name)
                observed_failures.add(key)
                if key not in reviewed:
                    problems.append(f"{identifier}/{level}/{name}: unreviewed failure.")
                    continue
                failure, metric = reviewed[key]
                frozen = targets.get(name)
                if failure.get("case_binding_sha256") != case_binding_digest(case):
                    problems.append(f"{identifier}: reviewed failure is bound to a different case definition.")
                elif frozen is None or frozen["value"] != metric["target"] or frozen.get("relative_tolerance") != metric["relative_tolerance"]:
                    problems.append(f"{identifier}/{name}: the frozen target differs from the reviewed one.")
                elif target.get("observed") is None or abs(target["observed"] - metric["observed"]) > \
                        metric["regression_relative_tolerance"] * abs(metric["observed"]):
                    problems.append(f"{identifier}/{level}/{name}: {target.get('observed')} left the reviewed value "
                                    f"{metric['observed']} by more than {metric['regression_relative_tolerance']:g} (relative).")
                else:
                    accepted.append(list(key))
        levels = entry.get("levels", {})
        if all(levels.get(level, {}).get("status") in ("passed", "failed", "comparators_only") for level in ("original", "rebuild")):
            if (entry.get("rebuild_matches_original") or {}).get("status") != "passed":
                problems.append(f"{identifier}: rebuild does not reproduce the original's native values.")
        if levels.get("components", {}).get("status") in ("passed", "failed", "comparators_only"):
            if (entry.get("components_match_original") or {}).get("status") != "passed":
                problems.append(f"{identifier}: the component build does not agree with the original within its declared bound.")
            if levels["components"].get("reference") != "passed":
                problems.append(f"{identifier}: the component build is not the official model region by region.")
    resolved = sorted(set(reviewed) - observed_failures)
    problems += [f"{'/'.join(key)}: reviewed failure no longer fails; review and remove it." for key in resolved]
    return {"status": "passed" if not problems else "failed", "problems": problems, "accepted_reviewed_failures": accepted,
            "note": "A passing audit means the record is complete and current and every failure is a reviewed finding within its "
                    "regression bound, not that all published targets passed."}


def write(results, output=None, html_output=None, catalog="reproductions"):
    record = collect(results, catalog)
    record["audit"] = audit(record, DATA / "verification" / REGISTRIES[catalog], catalog)
    output = output or RECORDS[catalog]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=1) + "\n")
    if html_output:
        Path(html_output).write_text(render_html(record))
    return record
