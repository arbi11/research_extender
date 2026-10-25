"""Portable knowledge graph, deterministic validation and source consolidation."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re

DATA = Path(__file__).parent / "data"
RELATIONS = {"decomposes_into", "implemented_by", "depends_on", "alternative_to", "reuses", "precedes"}
STATUSES = {"documented", "reconstructed", "geometry_checked", "partially_checked", "numerically_checked"}


def _common():
    stages = [("geometry", "Geometry"), ("materials", "Materials"), ("excitation", "Excitation"),
              ("boundaries", "Boundary conditions"), ("mesh_solve", "Mesh and solve"), ("solution", "Solution extraction")]
    nodes = [{"id": "workflow:magnetics", "layer": 1, "kind": "workflow", "title": "FEMM engineering workflow",
              "description": "A shared workflow whose component and analysis alternatives depend on the requested physical study.",
              "inputs": ["Study objective", "Physical dimensions and operating conditions"], "outputs": ["A reproducible field study"],
              "parameters": [], "constraints": ["A documented recipe does not itself establish numerical convergence."],
              "capabilities": [], "source_ids": ["manual5"], "tags": ["workflow"],
              "verification": {"status": "documented", "note": "Synthesis of the official preprocessor and postprocessor command groups."}}]
    edges = []
    for order, (slug, title) in enumerate(stages):
        identifier = "stage:" + slug
        nodes.append({"id": identifier, "layer": 1, "kind": "stage", "title": title, "order": order,
                      "description": "Expand an example to inspect its applicable " + title.lower() + " procedures.",
                      "inputs": [], "outputs": [], "parameters": [], "constraints": [], "capabilities": [],
                      "source_ids": ["manual5"], "tags": [slug],
                      "verification": {"status": "documented", "note": "Official scripting-stage organization."}})
        edges.append({"id": "common:" + slug, "source": "workflow:magnetics", "target": identifier,
                      "relation": "decomposes_into", "order": order, "when": "Select methods appropriate to the study.",
                      "parameter_bindings": {}, "source_ids": ["manual5"]})
        if order:
            edges.append({"id": "sequence:" + slug, "source": "stage:" + stages[order-1][0], "target": identifier,
                          "relation": "precedes", "when": "An execution dependency, distinct from component containment.",
                          "parameter_bindings": {}, "source_ids": ["manual5"]})
    return nodes, edges


def validate_graph(graph):
    """Return explicit issues; never manufacture evidence for missing records."""
    issues = []
    def issue(code, message, identifier=None):
        issues.append({"code": code, "message": message, "id": identifier})
    if graph.get("schema_version") != 1:
        issue("schema_version", "Expected schema version 1.")
    nodes = graph.get("nodes", [])
    ids = [n.get("id") for n in nodes]
    for identifier, count in Counter(ids).items():
        if count != 1:
            issue("duplicate_node", "Node IDs must be unique.", identifier)
    index = {n.get("id"): n for n in nodes}
    sources = {s.get("id") for s in graph.get("sources", [])}
    for node in nodes:
        identifier = node.get("id")
        if node.get("layer") not in (1, 2, 3) or isinstance(node.get("layer"), bool):
            issue("layer", "Knowledge nodes must belong to exactly layer 1, 2 or 3.", identifier)
        if not node.get("title") or not node.get("description"):
            issue("node_description", "Each node needs a title and explanation.", identifier)
        if node.get("verification", {}).get("status") not in STATUSES:
            issue("verification", "An explicit evidence status is required.", identifier)
        if not node.get("source_ids"):
            issue("source_missing", "Every knowledge node needs a source reference.", identifier)
        for source in node.get("source_ids", []):
            if source not in sources:
                issue("source_unknown", "Unknown source: " + str(source), identifier)
        if node.get("verification", {}).get("status") in ("numerically_checked", "partially_checked") and not node.get("verification", {}).get("report"):
            issue("numerical_evidence", "Numerically checked nodes require an actual report reference.", identifier)
    edge_ids, containment = set(), defaultdict(list)
    for edge in graph.get("edges", []):
        identifier = edge.get("id")
        if identifier in edge_ids:
            issue("duplicate_edge", "Edge IDs must be unique.", identifier)
        edge_ids.add(identifier)
        if edge.get("source") not in index or edge.get("target") not in index:
            issue("dangling_edge", "Both endpoints must exist.", identifier)
        if edge.get("relation") not in RELATIONS:
            issue("relation", "Unsupported relationship.", identifier)
        if edge.get("relation") == "alternative_to" and not edge.get("when", "").strip():
            issue("alternative_condition", "Alternatives require applicability conditions.", identifier)
        for source in edge.get("source_ids", []):
            if source not in sources:
                issue("edge_source_unknown", "Unknown edge source: " + source, identifier)
        if edge.get("relation") in {"decomposes_into", "implemented_by"}:
            containment[edge.get("source")].append(edge.get("target"))
    visiting, visited = set(), set()
    def visit(node):
        if node in visiting:
            issue("composition_cycle", "Composition must not contain cycles.", node)
            return
        if node in visited:
            return
        visiting.add(node)
        for child in containment[node]:
            visit(child)
        visiting.remove(node)
        visited.add(node)
    for identifier in list(index):
        visit(identifier)
    for example in graph.get("examples", []):
        identifier = example.get("id")
        if example.get("workflow_id") not in index:
            issue("example_workflow", "Example workflow must exist.", identifier)
        for node in example.get("node_ids", []):
            if node not in index:
                issue("example_node", "Unknown example node: " + node, identifier)
        for source in example.get("source_ids", []):
            if source not in sources:
                issue("example_source", "Unknown example source: " + source, identifier)
    return issues


def _current_case_bindings(directory):
    from .reproduce import case_binding_digest
    bindings = {}
    for name, key in (("reproductions.json", "cases"), ("designs.json", "designs")):
        path = directory / name
        if path.exists():
            bindings.update({c["id"]: case_binding_digest(c) for c in json.loads(path.read_text())[key]})
    return bindings


def _attach_components(graph, directory):
    """Bind executable components to their method nodes and record native evidence.

    A component counts as numerically checked only when an article-level build
    that used it, directly or inside another component, passed natively.
    """
    path = directory / "components.json"
    if not path.exists():
        return
    catalog = json.loads(path.read_text())["components"]
    index = {node["id"]: node for node in graph["nodes"]}
    by_id = {c["id"]: c for c in catalog}
    node_of = {}
    for component in catalog:
        executable = {k: component[k] for k in ("id", "parameters", "physics", "problem_types", "constraints",
                                                "femm_functions", "uses_components")}
        if component.get("node_id") in index:
            index[component["node_id"]]["executable_component"] = executable
            node_of[component["id"]] = component["node_id"]
            continue
        identifier = "component:" + component["id"]
        node = {"id": identifier, "layer": component["layer"],
                "kind": "executable_component" if component["layer"] == 2 else "executable_operation",
                "title": component["title"], "description": component["description"], "parameters": component["parameters"],
                "inputs": ["The typed model under construction and the parameters listed here."],
                "outputs": ["Geometry, properties or labels added to the typed model."],
                "constraints": component["constraints"], "capabilities": [component["stage"]],
                "source_ids": component["source_ids"], "tags": ["component", component["id"]], "executable_component": executable,
                "verification": {"status": "reconstructed",
                                 "note": "Executable component; no passing native build has used it yet."}}
        graph["nodes"].append(node)
        index[identifier] = node
        node_of[component["id"]] = identifier
    prefixes = {"magnetics": "mi", "electrostatics": "ei", "heatflow": "hi"}
    for component in catalog:
        source = node_of[component["id"]]
        targets = [node_of[u] for u in component["uses_components"]]
        targets += [f"primitive:{prefixes[p]}_{f}" for p in component["physics"] for f in component["femm_functions"]
                    if f"primitive:{prefixes[p]}_{f}" in index]
        if not any(e["target"] == source for e in graph["edges"]) and "stage:" + component["stage"] in index and not any(
                component["id"] in other["uses_components"] for other in catalog):
            targets.insert(0, None)
        for target in targets:
            edge_source, edge_target = (("stage:" + component["stage"], source) if target is None else (source, target))
            graph["edges"].append({"id": f"component:{edge_source}->{edge_target}", "source": edge_source, "target": edge_target,
                                   "relation": "decomposes_into" if target is None else "implemented_by",
                                   "when": "Executable construction used by data-only assemblies.", "parameter_bindings": {},
                                   "source_ids": component["source_ids"]})
    examples = {e["id"]: e for e in graph["examples"]}
    checked, reports = {}, {}
    cases = []
    for name in ("official_reproductions.json", "component_designs.json"):
        record_path = directory / "verification" / name
        if record_path.exists():
            cases += [dict(c, _report="verification/" + name) for c in json.loads(record_path.read_text()).get("cases", [])]
    from .components import component_digests
    current_digests = component_digests(list(node_of))
    current_bindings = _current_case_bindings(directory)
    stale = {}
    for case in cases:
        article = case.get("levels", {}).get("article", {})
        used = article.get("components_used") or []
        example = examples.get(case.get("example_id"))
        for component in used:
            if example and component in node_of:
                graph["edges"].append({"id": f"assembly:{example['id']}:{component}", "source": example["workflow_id"],
                                       "target": node_of[component], "relation": "reuses",
                                       "when": "The example's article build assembles this component.",
                                       "parameter_bindings": {}, "source_ids": example.get("source_ids", [])})
        if article.get("status") == "passed":
            recorded = article.get("component_digests") or {}
            case_fresh = current_bindings.get(case["case_id"]) == case.get("case_binding_sha256")
            for component, digest in recorded.items():
                if component not in by_id:
                    continue
                # Evidence counts only for the implementation and case definition it ran under.
                if digest != current_digests.get(component) or not case_fresh:
                    stale.setdefault(component, []).append(case["case_id"])
                elif case["case_id"] not in checked.setdefault(component, []):
                    checked[component].append(case["case_id"])
                    reports.setdefault(component, set()).add(case["_report"])
    for component, stale_cases in stale.items():
        if component not in checked:
            node = index[node_of[component]]
            node["verification"] = dict(node["verification"], stale_evidence_case_ids=sorted(stale_cases),
                                        note=node["verification"].get("note", "") + " Passing evidence exists for an earlier "
                                             "implementation or case definition and no longer applies.")
    for component, cases in checked.items():
        index[node_of[component]]["verification"] = {
            "status": "numerically_checked", "report": sorted(reports[component])[0], "reports": sorted(reports[component]),
            "case_ids": sorted(cases),
            "note": "Used by article-level or new-design assemblies that passed natively; applicability limits still apply."}


def load_graph(data_dir=None):
    directory = Path(data_dir) if data_dir else DATA
    manifest_file = directory / "source_manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {"sources": [], "artifacts": [], "crawl": {}}
    # Manually reviewed vendor sources are separate from the FEMM-only crawler.
    # Refreshing official FEMM pages must not erase an external method transfer.
    external_file = directory / "external_sources.json"
    external = json.loads(external_file.read_text()) if external_file.exists() else {}
    source_conflicts = []
    for collection in ("sources", "artifacts"):
        values = {item["id"]: item for item in manifest.get(collection, [])}
        for item in external.get(collection, []):
            if item["id"] in values and values[item["id"]] != item:
                source_conflicts.append({"id": item["id"], "code": "external_source_conflict", "message": "Crawler and external source identities conflict."})
            else:
                values[item["id"]] = item
        manifest[collection] = list(values.values())
    nodes, edges = _common()
    examples, example_bindings = [], []
    for name in ("functions.json", "machines.json", "methods.json", "recipes.json", "transfer_cases.json", "benchmark_methods.json"):
        path = directory / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        nodes.extend(data.get("nodes", []))
        edges.extend(data.get("edges", []))
        examples.extend(data.get("examples", []))
        example_bindings.extend(data.get("example_bindings", []))
    # New checked execution methods extend an existing study without duplicating
    # its workflow or silently replacing its original source interpretation.
    example_index = {example["id"]: example for example in examples}
    for binding in example_bindings:
        example = example_index.get(binding["example_id"])
        if example is None:
            source_conflicts.append({"id": binding["example_id"], "code": "benchmark_example_unknown",
                                     "message": "Benchmark method binding references an unknown study."})
            continue
        for field in ("source_ids", "node_ids", "assumptions", "limitations"):
            example[field] = list(dict.fromkeys(example.get(field, []) + binding.get(field, [])))
    # Identical shared concepts are consolidated with an evidence union. Layer
    # and title conflicts remain visible rather than being silently overwritten.
    merged, conflicts = {}, source_conflicts
    for node in nodes:
        node = deepcopy(node)
        identifier = node["id"]
        if identifier not in merged:
            merged[identifier] = node
            continue
        old = merged[identifier]
        structural = ("layer", "title", "kind", "recipe_id", "parameters", "source_locator", "construction_steps", "arguments", "geometry_ids")
        if any(old.get(field) != node.get(field) for field in structural):
            conflicts.append({"id": identifier, "code": "shared_node_conflict", "message": "Conflicting shared node definitions."})
        if old.get("verification", {}).get("status") != node.get("verification", {}).get("status"):
            conflicts.append({"id": identifier, "code": "shared_evidence_conflict", "message": "Shared nodes cannot silently inherit different verification levels."})
        contexts = old.setdefault("source_contexts", [{field: deepcopy(old.get(field, [] if field != "description" else "")) for field in ("source_ids", "description", "constraints", "capabilities", "inputs", "outputs")}])
        context = {field: deepcopy(node.get(field, [] if field != "description" else "")) for field in ("source_ids", "description", "constraints", "capabilities", "inputs", "outputs")}
        if context not in contexts:
            contexts.append(context)
        for field in ("source_ids", "tags", "constraints", "capabilities", "inputs", "outputs"):
            old[field] = list(dict.fromkeys(old.get(field, []) + node.get(field, [])))
        if old.get("description") != node.get("description"):
            old.setdefault("additional_descriptions", []).append(node.get("description"))
    merged_edges = {}
    for edge in edges:
        old = merged_edges.get(edge["id"])
        if old and any(old.get(k) != edge.get(k) for k in ("source", "target", "relation", "when", "order", "parameter_bindings")):
            conflicts.append({"id": edge["id"], "code": "edge_conflict", "message": "Conflicting edge identity."})
        elif old:
            old["source_ids"] = list(dict.fromkeys(old.get("source_ids", []) + edge.get("source_ids", [])))
        else:
            merged_edges[edge["id"]] = deepcopy(edge)
    model_manifest = directory / "model_manifest.json"
    previews = json.loads(model_manifest.read_text()).get("models", []) if model_manifest.exists() else []
    preview_index = {m["artifact_id"]: m for m in previews}
    artifacts = []
    for artifact in manifest.get("artifacts", []):
        value = deepcopy(artifact)
        preview = preview_index.get(artifact["id"])
        value.update(preview_available=bool(preview), geometry_available=bool(preview))
        if preview:
            value["geometry_model_file"] = preview["model_file"]
            value["geometry_summary"] = preview.get("summary", {})
        artifacts.append(value)
    graph = {"schema_version": 1, "nodes": sorted(merged.values(), key=lambda n: n["id"]),
             "edges": sorted(merged_edges.values(), key=lambda e: e["id"]),
             "examples": sorted(examples, key=lambda e: e["id"]), "sources": manifest.get("sources", []),
             "artifacts": artifacts, "crawl": manifest.get("crawl", {}), "consolidation_conflicts": conflicts}
    # Native reproduction evidence upgrades only the reproduced example's own
    # workflow; shared methods keep their status for other applications.
    reproductions = directory / "verification" / "official_reproductions.json"
    if reproductions.exists():
        record = json.loads(reproductions.read_text())
        node_index = {node["id"]: node for node in graph["nodes"]}
        current = _current_case_bindings(directory)
        for example in graph["examples"]:
            cases = [c for c in record.get("cases", []) if c.get("example_id") == example["id"]]
            if not cases:
                continue
            for case in cases:
                case["fresh"] = bool(case.get("case_binding_sha256")) and current.get(case["case_id"]) == case["case_binding_sha256"]
            example["reproductions"] = [dict({k: c.get(k) for k in ("case_id", "title", "status", "levels_passed", "fresh")},
                                             scope=(c.get("coverage") or {}).get("scope")) for c in cases]
            passed = [c for c in cases if c.get("status") == "passed" and c["fresh"]]
            core = [c for c in passed if (c.get("coverage") or {}).get("scope") == "core"]
            workflow = node_index.get(example.get("workflow_id"))
            if passed and workflow:
                chosen = core or passed
                workflow["verification"] = {
                    "status": "numerically_checked" if core else "partially_checked",
                    "report": "verification/official_reproductions.json",
                    "case_ids": [c["case_id"] for c in chosen],
                    "levels_passed": sorted({level for c in chosen for level in c["levels_passed"]}),
                    "reproduced": [c["coverage"]["reproduced"] for c in chosen if c.get("coverage")],
                    "not_reproduced": [c["coverage"]["not_reproduced"] for c in chosen if (c.get("coverage") or {}).get("not_reproduced")],
                    "note": ("This official example's core published results were solved natively and matched. "
                             if core else "Only part of this official example was reproduced; see what was and was not. ") +
                            "Shared methods it uses are not thereby certified for other applications."}
    _attach_components(graph, directory)
    baseline_path = directory / "curation_manifest.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    baseline["source_hashes"] = {**external.get("curated_source_hashes", {}), **baseline.get("source_hashes", {})}
    observed_sources = {s["id"]: s.get("sha256") for s in graph["sources"]}
    source_records = {s["id"]: s for s in graph["sources"]}
    text_hashes = baseline.get("text_hashes", {})

    def source_changed(identifier, digest):
        # Page markup carries a per-request timestamp, so a refreshed crawl is
        # compared on extracted text whenever the baseline recorded it.
        record = source_records.get(identifier, {})
        if identifier in text_hashes and record.get("text_sha256"):
            return record["text_sha256"] != text_hashes[identifier]
        return observed_sources.get(identifier) != digest

    changed_sources = {identifier for identifier, digest in baseline.get("source_hashes", {}).items()
                       if source_changed(identifier, digest)}
    for node in graph["nodes"]:
        node.setdefault("version", 1)
        node["source_versions"] = {identifier: baseline.get("source_hashes", {}).get(identifier, observed_sources.get(identifier)) for identifier in node.get("source_ids", [])}
        changed = sorted(set(node.get("source_ids", [])) & changed_sources)
        if changed:
            node["verification"] = {**node["verification"], "source_current": False,
                                    "note": node["verification"].get("note", "") + " Source content changed after this curation; review the procedure before reuse.",
                                    "changed_source_ids": changed}
    graph["source_changes"] = sorted(changed_sources)
    errors = validate_graph(graph) + conflicts
    layers = Counter(str(n.get("layer")) for n in graph["nodes"])
    graph["stats"] = {"nodes": len(graph["nodes"]), "edges": len(graph["edges"]), "examples": len(examples),
                      "sources": len(graph["sources"]), "fetched_sources": sum(s.get("status") == "fetched" for s in graph["sources"]),
                      "artifacts": len(artifacts), "geometry_previews": len(previews), "layers": dict(layers),
                      "families": dict(Counter(e.get("family", "unspecified") for e in examples)),
                      "verification": dict(Counter(n.get("verification", {}).get("status", "unknown") for n in graph["nodes"])),
                      "changed_sources": len(changed_sources),
                      "validation_errors": len(errors)}
    graph["validation"] = {"status": "passed" if not errors else "failed", "issues": errors}
    return graph


def example_subgraph(graph, example_id):
    example = next((e for e in graph["examples"] if e["id"] == example_id), None)
    if example is None:
        return None
    explicit = set(example.get("node_ids", [])) | {example["workflow_id"]}
    selected = set(explicit)
    source_ids = set(example.get("source_ids", []))
    index = {node["id"]: node for node in graph["nodes"]}
    attached_recipes = set()
    outgoing = defaultdict(list)
    for edge in graph["edges"]:
        if edge["relation"] in {"decomposes_into", "implemented_by", "reuses"}:
            outgoing[edge["source"]].append(edge)

    def applicable(edge):
        source, target = index.get(edge["source"], {}), index.get(edge["target"], {})
        source_scope = bool(source_ids.intersection(edge.get("source_ids", [])))
        if source.get("kind") == "stage":
            # Shared stages organize the whole library. Their other children
            # are alternatives, not part of every example that uses the stage.
            return edge["target"] in explicit or (
                edge["source"].startswith("stage:" + example_id + ":") and source_scope)
        if edge["target"] in attached_recipes:
            return True
        if edge["relation"] == "implemented_by" and target.get("layer") == 3:
            return True
        return source_scope

    pending = list(selected)
    while pending:
        parent = pending.pop()
        recipe_id = index.get(parent, {}).get("recipe_id")
        recipe = "recipe:" + recipe_id if recipe_id else None
        if recipe in index:
            attached_recipes.add(recipe)
            if recipe not in selected:
                selected.add(recipe)
                pending.append(recipe)
        for edge in outgoing[parent]:
            if not applicable(edge):
                continue
            child = edge["target"]
            if child not in selected:
                selected.add(child)
                pending.append(child)
    def retained_edge(edge):
        if edge["source"] not in selected or edge["target"] not in selected:
            return False
        if edge["relation"] in {"decomposes_into", "implemented_by", "reuses"}:
            return applicable(edge) or edge["target"] in attached_recipes
        return bool(source_ids.intersection(edge.get("source_ids", []))) or (
            edge["relation"] == "precedes" and edge["source"].startswith("stage:") and edge["target"].startswith("stage:"))
    return {**deepcopy(example), "nodes": [n for n in graph["nodes"] if n["id"] in selected],
            "edges": [e for e in graph["edges"] if retained_edge(e)],
            "artifacts": [a for a in graph["artifacts"] if source_ids.intersection(a.get("source_ids", []))]}
