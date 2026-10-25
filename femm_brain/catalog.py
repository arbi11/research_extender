"""Build functional API facts from the official manual and passive examples."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess


GLOBALS = {"newdocument", "create", "open", "opendocument", "openfemm", "closefemm", "quit", "smartmesh", "showconsole", "hideconsole", "clearconsole", "messagebox"}
PREFIXES = {"mi": "magnetics preprocessor", "mo": "magnetics postprocessor", "ei": "electrostatics preprocessor", "eo": "electrostatics postprocessor", "hi": "heat-flow preprocessor", "ho": "heat-flow postprocessor", "ci": "current-flow preprocessor", "co": "current-flow postprocessor"}
FUNCTION = re.compile(r"\b((?:mi|mo|ei|eo|hi|ho|ci|co)_[a-zA-Z0-9_]+|" + "|".join(sorted(GLOBALS)) + r")\s*\(")


def _signature(text, offset):
    start = text.find("(", offset)
    depth = 0
    for end in range(start, min(len(text), start + 800)):
        if text[end] == "(": depth += 1
        elif text[end] == ")":
            depth -= 1
            if depth == 0:
                return re.sub(r"\s+", " ", text[offset:end+1]).strip()
    return text[offset:start] + "(...)"


def _purpose(name):
    suffix = name.split("_", 1)[-1]
    known = {"addnode": "Create a point used by geometry entities.", "addsegment": "Connect existing geometry points with a straight segment.",
             "addarc": "Connect geometry points with a circular arc of a specified sweep.", "drawline": "Create endpoints and the connecting straight segment.",
             "drawarc": "Create endpoints and a circular arc.", "copyrotate": "Replicate selected geometry about an origin by an angular increment.",
             "setblockprop": "Assign selected-region material, mesh and other physics-specific properties.",
             "addblocklabel": "Create a seed identifying a material region.", "analyze": "Run the field solver using the configured model.",
             "createmesh": "Generate the finite-element discretization.", "loadsolution": "Open the field solution for postprocessing.",
             "getpointvalues": "Read field quantities at a specified position.", "blockintegral": "Evaluate a selected-region integral using its documented type.",
             "getcircuitproperties": "Read electrical circuit quantities from the solution.", "gapintegral": "Evaluate a supported air-gap integral.",
             "addboundprop": "Define a named boundary-property record.", "addmaterial": "Define a material-property record.",
             "probdef": "Configure the physics, units and numerical problem settings.", "selectgroup": "Select entities assigned to a group.",
             "clearselected": "Clear the current preprocessor selection.", "saveas": "Write the current model to a named file.",
             "getnode": "Read an indexed solution-mesh node.", "getelement": "Read an indexed solution-mesh element.",
             "getb": "Read magnetic flux-density components at a position."}
    return known.get(suffix, "A documented " + PREFIXES.get(name.split("_")[0], "FEMM session") + " operation; inspect the cited command contract before binding arguments.")


def build_function_catalog(manifest, cache_dir, destination):
    cache = Path(cache_dir)
    manual = next(s for s in manifest["sources"] if s["id"] == "manual5")
    text_path = cache / manual.get("text_cache_key", "missing")
    if text_path.is_file():
        text = text_path.read_text(errors="replace")
    else:
        pdf = cache / manual["cache_key"]
        result = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True, timeout=60)
        text = result.stdout
    pages = text.split("\f")
    catalog = {}
    for page_index, page in enumerate(pages, 1):
        for match in FUNCTION.finditer(page):
            name = match.group(1)
            # Prefer the bullet definition over incidental prose references.
            line_start = page.rfind("\n", 0, match.start()) + 1
            is_definition = "•" in page[line_start:match.start()] or "\u2022" in page[line_start:match.start()]
            if name in catalog and not is_definition:
                continue
            signature = _signature(page, match.start())
            if len(signature) > 500:
                continue
            catalog[name] = {"signature": signature, "page": page_index, "source_ids": ["manual5"], "definition": is_definition}
    # Some no-argument manual entries omit parentheses.
    for page_index, page in enumerate(pages, 1):
        for name in re.findall(r"•\s+((?:mi|mo|ei|eo|hi|ho|ci|co)_[a-zA-Z0-9_]+)\b", page):
            catalog.setdefault(name, {"signature": name + "()", "page": page_index, "source_ids": ["manual5"], "definition": True})
    # Binding/session operations documented in companion binding pages can also
    # be directly evidenced by the official example scripts. Read text only.
    script_evidence = {}
    for artifact in manifest.get("artifacts", []):
        if artifact.get("status") != "fetched" or artifact.get("kind") not in {"script", "lua_script", "python_script", "octave_script", "scilab_script"}:
            continue
        path = cache / artifact.get("cache_key", "missing")
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        script = path.read_text(errors="replace")
        for match in FUNCTION.finditer(script):
            script_evidence.setdefault(match.group(1), set()).update(artifact.get("source_ids", []))
    for name, source_ids in script_evidence.items():
        catalog.setdefault(name, {"signature": name + "(...)", "page": None, "source_ids": sorted(source_ids), "definition": False})
    nodes = []
    for name, item in sorted(catalog.items()):
        category = PREFIXES.get(name.split("_")[0], "FEMM session")
        args = item["signature"].split("(", 1)[1].rsplit(")", 1)[0] if "(" in item["signature"] else ""
        nodes.append({"id": "primitive:" + name, "layer": 3, "kind": "femm_function", "title": name,
                      "description": _purpose(name), "signature": item["signature"],
                      "parameters": [{"name": arg.strip(), "unit": "command-specific", "description": "See the versioned source contract."} for arg in args.split(",") if arg.strip()],
                      "inputs": ["Valid selection and problem state as required by this command."], "outputs": ["The documented command result or model-state change."],
                      "constraints": ["FEMM version and binding compatibility must be checked.", "Function documentation is not a numerical validation result."],
                      "capabilities": [category], "source_ids": item["source_ids"], "tags": [category, name],
                      "source_locator": {"page": item["page"], "function": name},
                      "verification": {"status": "documented", "note": "Function signature extracted from an official source; examples have not been implicitly executed."}})
    result = {"schema_version": 1, "nodes": nodes, "edges": [], "examples": [],
              "manual_sha256": manual.get("sha256"), "function_count": len(nodes)}
    Path(destination).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def build_model_previews(manifest, cache_dir, data_dir):
    from .geometry import parse_fem
    data = Path(data_dir)
    folder = data / "models"
    folder.mkdir(parents=True, exist_ok=True)
    models, failures = [], []
    for artifact in manifest.get("artifacts", []):
        name = artifact.get("filename", "")
        if artifact.get("status") != "fetched" or not name.lower().endswith((".fem", ".fee", ".feh", ".fec")):
            continue
        path = Path(cache_dir) / artifact.get("cache_key", "missing")
        if not path.is_file():
            failures.append({"artifact_id": artifact["id"], "reason": "Original cached file unavailable."})
            continue
        try:
            physics = {".fem": "magnetics", ".fee": "electrostatics", ".feh": "heatflow", ".fec": "currentflow"}[Path(name).suffix.lower()]
            parsed = parse_fem(path.read_text(encoding="utf-8-sig", errors="replace"), physics=physics)
            parsed.update(artifact_id=artifact["id"], source_ids=artifact.get("source_ids", []),
                          original_sha256=artifact["sha256"], original_url=artifact.get("url"),
                          verification={"status": "geometry_checked", "note": "Native geometry parsed and rendered; original construction history and numerical accuracy are not established."})
            filename = "models/" + artifact["sha256"] + ".json"
            (data / filename).write_text(json.dumps(parsed, ensure_ascii=False, separators=(",", ":")) + "\n")
            summary = {key: len(parsed.get(key, [])) for key in ("nodes", "segments", "arcs", "labels")}
            models.append({"artifact_id": artifact["id"], "model_file": filename, "summary": summary})
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            failures.append({"artifact_id": artifact["id"], "reason": str(exc)[:500]})
    result = {"schema_version": 1, "models": models, "failures": failures}
    (data / "model_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_recipe_graph(destination):
    from .geometry import construct, recipe_catalog
    nodes, edges = [], []
    for recipe in recipe_catalog():
        built = construct(recipe["id"])
        identifier = "recipe:" + recipe["id"]
        nodes.append({"id": identifier, "layer": 2, "kind": "construction_recipe", "title": recipe["title"],
                      "description": recipe["description"], "recipe_id": recipe["id"],
                      "parameters": recipe["parameters"], "inputs": ["Explicit dimensions in millimeters and angles in degrees."],
                      "outputs": ["Exact nodes, straight segments, circular arcs and an inspectable operation sequence."],
                      "constraints": ["Illustrative defaults are independently reconstructed, not dimensions attributed to an official example.",
                                      "The geometry must receive appropriate physical properties and analysis settings before solving."],
                      "capabilities": ["geometry"], "source_ids": ["manual5"],
                      "construction_steps": built.get("construction_steps", []), "tags": ["geometry", recipe["id"]],
                      "verification": {"status": "geometry_checked", "note": "Analytical closure, arc geometry and standalone export tests; no numerical accuracy claim."}})
        edges.append({"id": "recipe-stage:" + recipe["id"], "source": "stage:geometry", "target": identifier,
                      "relation": "decomposes_into", "when": "Choose this parameterized construction when its shape and constraints fit the design.",
                      "parameter_bindings": {}, "source_ids": ["manual5"]})
        previous = None
        for order, op in enumerate(built["operations"]):
            operation_id = identifier + ":" + op["id"]
            nodes.append({"id": operation_id, "layer": 3, "kind": "construction_operation", "title": op["function"] + " · " + str(order+1),
                          "description": "Default-instance construction call: " + op["function"] + "(" + ", ".join(str(x) for x in op["arguments"]) + ")",
                          "parameters": [], "arguments": op["arguments"], "geometry_ids": op["geometry_ids"],
                          "inputs": ["The preceding construction state and the recipe parameter bindings."],
                          "outputs": op["geometry_ids"], "constraints": ["Values shown instantiate the illustrative defaults; edited parameters regenerate the operation list."],
                          "capabilities": ["geometry"], "source_ids": ["manual5"], "tags": [recipe["id"], op["function"]],
                          "verification": {"status": "geometry_checked", "note": "Generated by the deterministic drawing recipe and checked by geometry tests."}})
            edges.extend([
                {"id": "recipe-op:" + operation_id, "source": identifier, "target": operation_id,
                 "relation": "implemented_by", "order": order, "when": "This ordered operation belongs to the illustrative default construction.", "parameter_bindings": {}, "source_ids": ["manual5"]},
                {"id": "recipe-api:" + operation_id, "source": operation_id, "target": "primitive:" + op["function"],
                 "relation": "reuses", "when": "The versioned FEMM command implements the generated primitive.", "parameter_bindings": {"arguments": op["arguments"]}, "source_ids": ["manual5"]}])
            if previous:
                edges.append({"id": "recipe-order:" + operation_id, "source": previous, "target": operation_id,
                              "relation": "precedes", "when": "Maintain the recipe's explicit construction order.", "parameter_bindings": {}, "source_ids": ["manual5"]})
            previous = operation_id
    result = {"schema_version": 1, "nodes": nodes, "edges": edges, "examples": []}
    Path(destination).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result
