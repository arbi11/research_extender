"""Local execution of registered studies; no model service or downloaded script."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import uuid

DEFAULT_IMAGE = "femm-brain-native:local"


def run_case(case_id, output, parameters=None, *, engine="docker", image=None,
             command=None, timeout=180):
    from .simulation import plan_simulation, run_simulation
    # Validation happens before allocating a container or writing any result.
    plan = plan_simulation(case_id, parameters)
    if engine == "native":
        return run_simulation(case_id, output, parameters, command, timeout)
    if engine != "docker":
        raise ValueError("Choose docker or native execution")
    output = Path(output).resolve()
    receipt_path = output.with_name(output.name + "-execution.json")
    if receipt_path.exists():
        raise ValueError("Use a new result directory; a previous execution receipt already exists")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new, empty result directory; previous evidence is never overwritten")
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("Docker is required for solving. Browsing and planning work without it.")
    image = image or os.getenv("FEMM_BRAIN_IMAGE", DEFAULT_IMAGE)
    probe = subprocess.run([docker, "image", "inspect", "--format", "{{.Id}}", image],
                           capture_output=True, text=True, timeout=15)
    if probe.returncode or not probe.stdout.strip().startswith("sha256:"):
        raise RuntimeError("The FEMM Docker image is unavailable. Build it or pass --image; see the README.")
    image_id = probe.stdout.strip()
    output.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parent.parent
    name = "femm-brain-solve-" + uuid.uuid4().hex[:16]
    args = [docker, "run", "--rm", "--init", "--name", name,
            "--network", "none", "--read-only", "--memory", "2g", "--cpus", "2",
            "--pids-limit", "256", "--tmpfs", "/tmp:rw,size=512m",
            "--user", str(os.getuid()) + ":" + str(os.getgid()),
            "-e", "HOME=/tmp", "-e", "QT_QPA_PLATFORM=offscreen",
            "-e", "FEMM_DIR=/opt/femm/bin", "-e", "FEMM_MESH_EXPORT=1",
            "-v", str(repository / "femm_brain") + ":/repo/femm_brain:ro", "-v", str(output) + ":/reports",
            "-w", "/repo", image_id, "python3", "-m", "femm_brain", "solve", case_id,
            "--engine", "native", "--command", "/opt/femm/bin/femm-lua",
            "--parameters", json.dumps(parameters or {}, allow_nan=False),
            "--output", "/reports", "--timeout", str(timeout)]
    receipt = {"captured_utc": datetime.now(timezone.utc).isoformat(), "engine": "docker",
               "image_id": image_id, "network": "none", "read_only_container": True,
               "memory": "2g", "cpus": 2, "status": "running", "case_id": case_id,
               "plan_sha256": plan["plan_sha256"],
               "llm_calls": 0, "provider_credentials_passed": False}
    # Reserve this attempt's receipt exclusively before starting a container.
    # Only this attempt may update its own running receipt to a terminal state.
    with receipt_path.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")

    def save_receipt():
        receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = receipt_path.with_name(receipt_path.name + "." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(receipt_path)

    def captured_text(value):
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""

    def cleanup():
        try:
            cleaned = subprocess.run([docker, "rm", "-f", name], capture_output=True, text=True, timeout=15)
            return {"attempted": True, "exit_code": cleaned.returncode,
                    "stdout": cleaned.stdout, "stderr": cleaned.stderr,
                    "status": "completed" if cleaned.returncode == 0 else "failed"}
        except (OSError, subprocess.SubprocessError) as exc:
            return {"attempted": True, "status": "failed", "error": str(exc),
                    "stdout": captured_text(getattr(exc, "stdout", None)),
                    "stderr": captured_text(getattr(exc, "stderr", None))}

    try:
        completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout + 60)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        receipt.update(status="timed_out" if isinstance(exc, subprocess.TimeoutExpired) else "interrupted",
                       error="Native container execution timed out" if isinstance(exc, subprocess.TimeoutExpired) else "Execution interrupted",
                       stdout=captured_text(getattr(exc, "stdout", None)),
                       stderr=captured_text(getattr(exc, "stderr", None)), cleanup=cleanup())
        save_receipt()
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RuntimeError("Native study exceeded its execution time limit; partial output and cleanup evidence were retained") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        receipt.update(status="failed", error=str(exc), cleanup=cleanup())
        save_receipt()
        raise
    receipt.update(exit_code=completed.returncode,
                   status="completed" if completed.returncode == 0 else "failed")
    report_path = output / "report.json"
    if not report_path.is_file():
        receipt["error"] = (completed.stderr or completed.stdout)[-3000:]
        receipt.update(status="failed", stdout=completed.stdout, stderr=completed.stderr)
        save_receipt()
        raise RuntimeError("The native container produced no report: " + receipt["error"])
    receipt["native_report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    save_receipt()
    report = json.loads(report_path.read_text())
    if completed.returncode and report.get("status") in {"passed", "solved_unassessed"}:
        report = {**report, "status": "failed", "native_report_original_status": report["status"],
                  "execution_error": "The native container exited unsuccessfully after producing its report"}
    return {**report, "execution": receipt}


class SimulationJobs:
    """One native job at a time, with explicit opt-in for the local web server."""
    def __init__(self, root, image=None, timeout=180, max_pending=4):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.image, self.timeout, self.max_pending = image, timeout, max_pending
        self.jobs = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="femm-brain")

    def _persist_job_locked(self, identifier, *, create=False):
        path = self.root / (identifier + "-job.json")
        body = json.dumps(self.jobs[identifier], indent=2, allow_nan=False) + "\n"
        if create:
            with path.open("x") as stream:
                stream.write(body)
        else:
            temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
            temporary.write_text(body)
            temporary.replace(path)

    def submit(self, case_id, parameters=None, expected_plan_sha256=None):
        from .simulation import plan_simulation
        plan = plan_simulation(case_id, parameters)
        if expected_plan_sha256 is not None and expected_plan_sha256 != plan["plan_sha256"]:
            raise ValueError("The displayed frozen plan has changed; regenerate it before solving")
        with self.lock:
            if sum(j["status"] in {"queued", "running"} for j in self.jobs.values()) >= self.max_pending:
                raise ValueError("The local solve queue is full; wait for a study to complete")
            identifier = uuid.uuid4().hex
            self.jobs[identifier] = {"id": identifier, "case_id": case_id, "status": "queued",
                                     "plan_sha256": plan["plan_sha256"], "llm_calls": 0,
                                     "expected_displayed_plan_sha256": expected_plan_sha256,
                                     "queued_utc": datetime.now(timezone.utc).isoformat()}
            self._persist_job_locked(identifier, create=True)
        self.pool.submit(self._run, identifier, case_id, deepcopy(parameters or {}))
        return self.get(identifier)

    def _run(self, identifier, case_id, parameters):
        with self.lock:
            self.jobs[identifier]["status"] = "running"
            self.jobs[identifier]["started_utc"] = datetime.now(timezone.utc).isoformat()
            self._persist_job_locked(identifier)
        try:
            report = run_case(case_id, self.root / identifier, parameters,
                              image=self.image, timeout=self.timeout)
            with self.lock:
                expected = self.jobs[identifier]["plan_sha256"]
            actual = report.get("plan_sha256")
            if actual != expected:
                error = "The executed plan differs from the submitted frozen plan; retained native evidence was not accepted"
                report = {**report, "status": "failed", "native_report_original_status": report.get("status"),
                          "checks": {**report.get("checks", {}), "submitted_plan_binding": {
                              "status": "failed", "expected_plan_sha256": expected, "executed_plan_sha256": actual}},
                          "execution_error": error}
                update = {"status": "failed", "report": report, "error": error}
            else:
                update = {"status": "completed", "report": report}
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
            update = {"status": "failed", "error": str(exc)}
        with self.lock:
            self.jobs[identifier].update(update)
            self.jobs[identifier]["completed_utc"] = datetime.now(timezone.utc).isoformat()
            self._persist_job_locked(identifier)

    def get(self, identifier):
        with self.lock:
            return deepcopy(self.jobs.get(identifier))

    def artifact(self, identifier, filename):
        if self.get(identifier) is None or Path(filename).name != filename:
            return None
        if filename == "execution.json":
            path = self.root / (identifier + "-execution.json")
        elif filename == "job.json":
            path = self.root / (identifier + "-job.json")
        else:
            path = self.root / identifier / filename
        return path if path.is_file() else None

    def close(self):
        self.pool.shutdown(wait=True)
