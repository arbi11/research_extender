"""Execute frozen toy geometry scenarios in native FEMM and check saved models."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid

from .source import DEFAULT_CACHE, fingerprint, load_sources, source_file, verified_text

FIXTURES = Path(__file__).with_name("tests") / "native_geometry"
SECTIONS = {"points": "NumPoints", "segments": "NumSegments", "arcs": "NumArcSegments", "labels": "NumBlockLabels"}


def batch_digest(batch):
    return fingerprint(json.dumps(batch, sort_keys=True).encode())


def read_model(path):
    """Read the numeric geometry tables in a saved magnetic FEMM file."""
    lines = Path(path).read_text().splitlines()
    result = {}
    for name, heading in SECTIONS.items():
        match = next(((i, re.fullmatch(r"\[" + heading + r"\]\s*=\s*(\d+)", line.strip()))
                      for i, line in enumerate(lines) if line.strip().startswith("[" + heading + "]")), None)
        if not match or not match[1]:
            raise ValueError("Missing geometry table " + heading)
        index, marker = match
        count = int(marker.group(1))
        rows = [[float(v) for v in line.split()] for line in lines[index+1:index+1+count]]
        if len(rows) != count or any(not row or any(not math.isfinite(v) for v in row) for row in rows):
            raise ValueError("Invalid geometry table " + heading)
        result[name] = rows
    return result


def evaluate(folder, protocol):
    """Compare observed counts, coordinates and properties with frozen expectations."""
    results = []
    tolerance = protocol["absolute_tolerance"]
    for case in protocol["cases"]:
        errors = []
        observed = None
        try:
            observed = read_model(Path(folder) / case["file"])
            for section, count in case["expected_counts"].items():
                if len(observed[section]) != count:
                    errors.append(f"{section}: expected {count}, observed {len(observed[section])}")
            if "expected_points" in case:
                unmatched = [row[:2] for row in observed["points"]]
                for expected in case["expected_points"]:
                    index = next((i for i, point in enumerate(unmatched)
                                  if all(abs(a-b) <= tolerance for a, b in zip(point, expected))), None)
                    if index is None:
                        errors.append("Missing point " + str(expected))
                    else:
                        unmatched.pop(index)
                if unmatched:
                    errors.append("Unexpected points " + str(unmatched))
            for section, rows in case["expected_rows"].items():
                for index, fields in rows.items():
                    for column, expected in fields.items():
                        try:
                            actual = observed[section][int(index)][int(column)]
                        except IndexError:
                            errors.append(f"{section}[{index}][{column}]: missing")
                            continue
                        if abs(actual-expected) > tolerance:
                            errors.append(f"{section}[{index}][{column}]: expected {expected}, observed {actual}")
            for expected in case.get("expected_segments", []):
                matches = []
                for row in observed["segments"]:
                    endpoints = [observed["points"][int(n)][:2] for n in row[:2]]
                    if any(all(all(abs(a-b) <= tolerance for a, b in zip(point, target))
                               for point, target in zip(endpoints, order))
                           for order in (expected["endpoints"], expected["endpoints"][::-1])):
                        matches.append(row)
                if len(matches) != 1:
                    errors.append("Expected one segment at " + str(expected["endpoints"]))
                else:
                    for column, value in expected["fields"].items():
                        if abs(matches[0][int(column)]-value) > tolerance:
                            errors.append("Segment property mismatch at " + str(expected["endpoints"]) + ", column " + column)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
        results.append({"id": case["id"], "status": "failed" if errors else "passed", "errors": errors,
                        "tested_commands": case["tested_commands"], "observed": observed})
    return results


def run(output, *, batch, image=None, cache=DEFAULT_CACHE, timeout=120):
    if not isinstance(timeout, int) or timeout < 1:
        raise ValueError("timeout must be a positive integer")
    protocol = json.loads((FIXTURES / "protocol.json").read_text())
    source = next(s for s in load_sources() if s["id"] == protocol["binding_source_id"])
    verified_text(source, cache)
    probe = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image or protocol["reference_image"]],
                           capture_output=True, text=True, check=True, timeout=15)
    image_id = probe.stdout.strip()
    if image_id != protocol["reference_image"]:
        raise ValueError("This frozen protocol requires image " + protocol["reference_image"])
    folder = Path(output).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise ValueError("Native output must be an empty directory; earlier evidence is never overwritten")
    for filename in ("geometry.lua", "protocol.json"):
        shutil.copyfile(FIXTURES / filename, folder / filename)
    # Execute the exact reviewed binding, rather than the older copy bundled in the image.
    shutil.copyfile(source_file(source, cache), folder / "binding.lua")
    name = "femm-brain2-geometry-" + uuid.uuid4().hex[:12]
    arguments = ["docker", "run", "--rm", "--init", "--name", name, "--network", "none", "--read-only",
                 "--memory", "2g", "--cpus", "2", "--pids-limit", "256", "--tmpfs", "/tmp:rw,size=512m",
                 "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-e", "QT_QPA_PLATFORM=offscreen",
                 "-e", "FEMM_DIR=/opt/femm/bin", "-v", f"{folder}:/work", "-w", "/work", image_id,
                 "/opt/femm/bin/lua4", "-e", "dofile('/work/binding.lua'); _femm_port=48015; openfemm(1)",
                 "-f", "geometry.lua"]
    started = datetime.now(timezone.utc).isoformat()
    try:
        process = subprocess.run(arguments, capture_output=True, text=True, timeout=timeout)
        log, returncode = process.stdout + process.stderr, process.returncode
    except subprocess.TimeoutExpired as exc:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
        chunks = (exc.stdout or b"", exc.stderr or b"")
        log = "\n".join(c.decode(errors="replace") if isinstance(c, bytes) else c for c in chunks)
        log += "\nNative geometry run exceeded its time limit.\n"
        returncode = -1
    (folder / "native.log").write_text(log)
    checks = evaluate(folder, protocol)
    report = {"schema_version": 1, "scope": protocol["scope"], "protocol_id": protocol["id"],
              "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
              "status": "passed" if returncode == 0 and all(c["status"] == "passed" for c in checks) else "failed",
              "batch_sha256": batch_digest(batch), "binding_source_sha256": source["sha256"],
              "engine": {"kind": "native_femm_docker", "image_id": image_id, "network": "none", "read_only": True,
                         "returncode": returncode, "argv": arguments},
              "checks": checks,
              "artifacts": {p.name: fingerprint(p.read_bytes()) for p in sorted(folder.iterdir())
                            if p.is_file() and p.name != "binding.lua"}}
    (folder / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return {"status": report["status"], "report": str(folder / "report.json"),
            "passed": sum(c["status"] == "passed" for c in checks), "total": len(checks), "image_id": image_id,
            "failures": [{"id": c["id"], "errors": c["errors"]} for c in checks if c["status"] != "passed"]}


def verify_evidence(path, batch):
    """Recheck artifacts and observations before attaching execution claims to lookup."""
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    protocol_bytes = (FIXTURES / "protocol.json").read_bytes()
    script_bytes = (FIXTURES / "geometry.lua").read_bytes()
    protocol = json.loads(protocol_bytes)
    engine = report["engine"]
    if report.get("batch_sha256") != batch_digest(batch):
        raise ValueError("Native evidence belongs to another reviewed command batch")
    source = next(s for s in load_sources() if s["id"] == protocol["binding_source_id"])
    if report.get("binding_source_sha256") != source["sha256"]:
        raise ValueError("Native evidence used another Lua binding")
    if engine.get("kind") != "native_femm_docker" or engine.get("image_id") != protocol["reference_image"] or engine.get("network") != "none" or engine.get("read_only") is not True:
        raise ValueError("Native evidence does not match the frozen container configuration")
    artifacts = report["artifacts"]
    required = {"protocol.json", "geometry.lua", "native.log"} | {c["file"] for c in protocol["cases"]}
    # Missing native models are permitted only as explicitly failed scenario observations.
    if not {"protocol.json", "geometry.lua", "native.log"} <= artifacts.keys():
        raise ValueError("Native evidence is missing its protocol, script or log")
    if any((path.parent / c["file"]).is_file() and c["file"] not in artifacts for c in protocol["cases"]):
        raise ValueError("Native evidence contains an unhashed model")
    for filename, digest in artifacts.items():
        if Path(filename).name != filename or filename not in required:
            raise ValueError("Unexpected native evidence artifact: " + filename)
        if fingerprint((path.parent / filename).read_bytes()) != digest:
            raise ValueError("Native artifact hash mismatch: " + filename)
    for filename, content in (("protocol.json", protocol_bytes), ("geometry.lua", script_bytes)):
        if artifacts[filename] != fingerprint(content):
            raise ValueError("Native evidence uses a different frozen " + filename)
    checks = evaluate(path.parent, protocol)
    if checks != report["checks"]:
        raise ValueError("Native observations do not agree with the recorded checks")
    expected_status = "passed" if engine["returncode"] == 0 and all(c["status"] == "passed" for c in checks) else "failed"
    if expected_status != report["status"]:
        raise ValueError("Native report status does not agree with its checks")
    per_command = {}
    for check in checks:
        for command in check["tested_commands"]:
            per_command.setdefault(command, []).append({"id": check["id"], "status": check["status"]})
    for command, cases in per_command.items():
        per_command[command] = {"status": "passed" if engine["returncode"] == 0 and all(c["status"] == "passed" for c in cases) else "check_failed",
                                "scope": protocol["scope"], "cases": cases, "image_id": engine["image_id"],
                                "binding_source_sha256": source["sha256"], "report": str(path),
                                "report_sha256": fingerprint(path.read_bytes())}
    return per_command
