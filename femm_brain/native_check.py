"""Verify reconstructed geometry through native FEMM, without field solves.

This runner executes only compiler-generated recipe operations. It never runs
downloaded source scripts. All recipes receive native geometry save/readback
checks; simple closed-domain recipes additionally receive native mesh checks.
Passing these checks does not establish a machine design or numerical accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

from .geometry import construct, export_recipe, parse_fem, recipe_catalog

MESH_RECIPES = {"circle", "rectangle", "circular_airgap"}
COUNTS = ("nodes", "segments", "arcs", "labels")


def native_script(recipe_id):
    """Return an audited construction script and its intended geometry."""
    geometry = construct(recipe_id)
    script = export_recipe(recipe_id, format="lua")
    if recipe_id in MESH_RECIPES:
        lines = ['mi_addmaterial("native_check_air",1,1,0,0,0)']
        labels = geometry["labels"]
        if not labels:
            labels = [{"x": 0., "y": 0.}]
            lines.append("mi_addblocklabel(0,0)")
        for label in labels:
            x, y = label["x"], label["y"]
            lines.extend([f"mi_selectlabel({x:.17g},{y:.17g})",
                          'mi_setblockprop("native_check_air",0,1,"<None>",0,0,0)',
                          "mi_clearselected()"])
        lines.extend(['mi_saveas("meshed.fem")', "mi_createmesh()",
                      'out = openfile("mesh-completed.txt","w")',
                      'write(out,"Native mesh creation completed. No field solve was requested.\\n")',
                      "closefile(out)"])
        script += "\n".join(lines) + "\n"
    script += ('mi_close()\nout = openfile("geometry-completed.txt","w")\n'
               'write(out,"Native geometry construction completed.\\n")\nclosefile(out)\n')
    return geometry, script


def mesh_summary(folder):
    """Read the actual diagnostic node/element exports from patched FEMM 5."""
    def rows(name):
        path = folder / name
        if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Missing or oversized native mesh artifact: " + name)
        return [line.split("#", 1)[0].split() for line in path.read_text().splitlines()
                if line.split("#", 1)[0].strip()]
    node_rows, element_rows = rows("meshed.node"), rows("meshed.ele")
    header = [int(v) for v in node_rows[0]]
    element_header = [int(v) for v in element_rows[0]]
    if len(header) != 4 or header[1] != 2 or len(node_rows)-1 != header[0]:
        raise ValueError("Native node header/count mismatch")
    if len(element_header) != 3 or element_header[1:] != [3, 1] or len(element_rows)-1 != element_header[0]:
        raise ValueError("Native element header/count mismatch")
    if header[0] < 3 or element_header[0] < 1:
        raise ValueError("Native mesh contains no finite elements")
    nodes = {}
    for row in node_rows[1:]:
        index, x, y = int(row[0]), float(row[1]), float(row[2])
        if index in nodes or not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Invalid native mesh node")
        nodes[index] = (x, y)
    areas = []
    for row in element_rows[1:]:
        vertices = [int(v) for v in row[1:4]]
        if len(set(vertices)) != 3 or any(v not in nodes for v in vertices):
            raise ValueError("Invalid native mesh connectivity")
        a, b, c = (nodes[v] for v in vertices)
        area = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])) / 2
        attribute = float(row[4])
        if area <= 0 or not math.isfinite(area) or not attribute.is_integer() or attribute < 1:
            raise ValueError("Invalid native mesh element or material label")
        areas.append(area)
    return {"node_count": header[0], "element_count": element_header[0],
            "minimum_element_area_mm2": min(areas), "total_element_area_mm2": sum(areas),
            "status": "passed", "field_solve_performed": False}


def geometry_signature(geometry):
    """Compare native primitives by coordinates, independent of node ordering."""
    points = [(round(p["x"], 9), round(p["y"], 9)) for p in geometry["nodes"]]
    segments = [tuple(sorted((points[item["n0"]], points[item["n1"]])))
                for item in geometry["segments"]]
    arcs = []
    for item in geometry["arcs"]:
        first, second, angle = points[item["n0"]], points[item["n1"]], item["angle"]
        if angle < 0:
            first, second, angle = second, first, -angle
        arcs.append((first, second, round(angle, 9)))
    labels = [(round(p["x"], 9), round(p["y"], 9)) for p in geometry["labels"]]
    return {"points": sorted(points), "segments": sorted(segments),
            "arcs": sorted(arcs), "labels": sorted(labels)}


def _version(command):
    engine = Path(command).resolve().with_name("femm_headless")
    if not engine.is_file():
        return {"launcher": str(command), "engine_version": None}
    result = subprocess.run([str(engine), "--version"], capture_output=True, text=True, timeout=10)
    text = (result.stdout + "\n" + result.stderr).strip()
    return {"launcher": str(command), "engine_version": text[:2000], "version_exit_code": result.returncode}


def _run(command, script, folder, timeout):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", FEMM_MESH_EXPORT="1")
    with (folder / "native.log").open("w") as log:
        process = subprocess.Popen([command, "-h", "-p", str(port), str(script)], cwd=folder,
                                   env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=(os.name == "posix"))
        try:
            process.wait(timeout=timeout)
            if process.returncode:
                raise RuntimeError("Native FEMM exited unsuccessfully; inspect native.log")
        finally:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.kill()
            process.wait(timeout=10)


def run_checks(output, command=None, timeout=180):
    """Run real native checks and retain independent per-recipe evidence."""
    executable = shutil.which(command or os.getenv("FEMM_LUA", "femm-lua"))
    if executable is None:
        raise ValueError("Native FEMM 5 launcher is required; set FEMM_LUA or put femm-lua on PATH")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "engine": _version(executable), "scope": "Native recipe construction and selected meshing; no field solves or engineering accuracy certification.",
              "recipes": [], "status": "running", "numerical_convergence_verified": False}
    report["geometry_module_sha256"] = hashlib.sha256(Path(__file__).with_name("geometry.py").read_bytes()).hexdigest()
    report["verification_protocol_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for archive in (Path("/tmp/femm5-source.zip"), Path("/opt/femm/upstream-source.zip")):
        if archive.is_file():
            report["source_archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
            break
    for recipe in recipe_catalog():
        recipe_id = recipe["id"]
        folder = output / recipe_id
        folder.mkdir(exist_ok=True)
        # A rerun cannot inherit output from an earlier successful execution.
        for name in ("geometry.fem", "meshed.fem", "meshed.node", "meshed.ele", "geometry-completed.txt", "mesh-completed.txt"):
            (folder / name).unlink(missing_ok=True)
        intended, script = native_script(recipe_id)
        script_path = folder / "construct.lua"
        script_path.write_text(script)
        started = time.monotonic()
        result = {"recipe_id": recipe_id, "parameters": intended["parameters"], "status": "failed",
                  "source_ids": ["manual5"], "dimension_source": "illustrative_parameters",
                  "expected_counts": {key: len(intended[key]) for key in COUNTS}, "mesh_requested": recipe_id in MESH_RECIPES,
                  "field_solve_performed": False}
        try:
            _run(executable, script_path, folder, timeout)
            if not (folder / "geometry-completed.txt").is_file():
                raise ValueError("Native geometry completion marker is missing")
            actual = parse_fem((folder / "geometry.fem").read_text())
            result["observed_counts"] = {key: len(actual[key]) for key in COUNTS}
            if result["observed_counts"] != result["expected_counts"]:
                raise ValueError("Native saved geometry differs from the intended primitive counts")
            if geometry_signature(intended) != geometry_signature(actual):
                raise ValueError("Native saved coordinates, segments, arcs or labels differ from the intended geometry")
            if recipe_id in MESH_RECIPES:
                if not (folder / "mesh-completed.txt").is_file():
                    raise ValueError("Native meshing completion marker is missing")
                result["mesh"] = mesh_summary(folder)
            if any(folder.glob("*.ans")):
                raise ValueError("Unexpected field solution: the verification protocol must not solve fields")
            result["status"] = "passed"
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
            result["error"] = str(exc)
        result["duration_seconds"] = round(time.monotonic()-started, 4)
        result["artifacts"] = [{"filename": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size_bytes": p.stat().st_size}
                               for p in sorted(folder.iterdir()) if p.is_file()]
        report["recipes"].append(result)
        report["status"] = "passed" if all(r["status"] == "passed" for r in report["recipes"]) else "failed"
        (output / "native_geometry.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args(argv)
    report = run_checks(args.output, args.command, args.timeout)
    print(json.dumps({"status": report["status"], "recipes": len(report["recipes"]),
                      "meshed": sum("mesh" in r for r in report["recipes"]), "report": str(args.output / "native_geometry.json")}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
