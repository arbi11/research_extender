"""Persistent jobs and a single, resource-bounded local solver worker."""
from contextlib import closing
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import sqlite3
import subprocess
import threading
import time
import uuid
import zipfile
from .model import CCoreSpec
from .generate import source_files
from .plan import GeometryPlan
from . import plan_generate
from .mesh import validate_mesh


class BusyError(Exception):
    pass


class JobStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "jobs.sqlite3"
        with closing(self.connect()) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, spec TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, result TEXT, error TEXT)")
            if "kind" not in {row[1] for row in db.execute("PRAGMA table_info(jobs)")}:
                db.execute("ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'solve'")
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            for column in ("revision_id", "evidence", "target_contract_id"):
                if column not in columns:
                    db.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT")

    def connect(self):
        db = sqlite3.connect(self.db, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def create(self, spec: CCoreSpec | GeometryPlan, max_pending=5, *, kind="solve", revision_id=None, target_contract_id=None):
        if kind not in {"solve", "mesh"} or (kind == "mesh" and not isinstance(spec, GeometryPlan)):
            raise ValueError("Mesh checks require a general geometry plan.")
        now = time.time()
        job_id = uuid.uuid4().hex
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if count >= max_pending:
                raise BusyError("The simulation queue is full. Please try again after a job finishes.")
            db.execute("INSERT INTO jobs (id,spec,status,created,updated,result,error,kind,revision_id,target_contract_id) VALUES (?,?,?,?,?,?,?,?,?,?)", (job_id, spec.model_dump_json(), "queued", now, now, None, None, kind, revision_id, target_contract_id))
        return self.get(job_id)

    def get(self, job_id):
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            return None
        with closing(self.connect()) as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        value = dict(row)
        value["spec"] = json.loads(value["spec"])
        value["result"] = json.loads(value["result"]) if value["result"] else None
        value["evidence"] = json.loads(value["evidence"]) if value["evidence"] else {}
        return value

    def annotate(self, job_id, **evidence):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT evidence FROM jobs WHERE id=?", (job_id,)).fetchone()
            previous = json.loads(row[0]) if row and row[0] else {}
            previous.update(evidence)
            db.execute("UPDATE jobs SET evidence=? WHERE id=?", (json.dumps(previous, allow_nan=False), job_id))

    def update(self, job_id, status, result=None, error=None):
        with closing(self.connect()) as db, db:
            db.execute("UPDATE jobs SET status=?, updated=?, result=?, error=? WHERE id=?",
                       (status, time.time(), json.dumps(result, allow_nan=False) if result else None, error, job_id))

    def next(self):
        with closing(self.connect()) as db:
            row = db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
        return self.get(row[0]) if row else None

    def recover(self):
        with closing(self.connect()) as db, db:
            db.execute("UPDATE jobs SET status='failed', error='Server restarted during simulation. Submit the model again.', updated=? WHERE status='running'", (time.time(),))

    def expire(self, hours):
        cutoff = time.time() - hours * 3600
        with closing(self.connect()) as db, db:
            rows = db.execute("SELECT id FROM jobs WHERE status IN ('completed','failed') AND updated < ?", (cutoff,)).fetchall()
            for row in rows:
                shutil.rmtree(self.root / row[0], ignore_errors=True)
                db.execute("DELETE FROM jobs WHERE id=?", (row[0],))


def read_results(folder: Path, required=None):
    if required is None:
        required = {"force_x": "N", "force_y": "N", "energy": "J", "flux_linkage": "Wb-turn", "current": "A"}
    result = {}
    if not (folder / "model.ans").is_file() or (folder / "model.ans").stat().st_size == 0:
        raise RuntimeError("FEMM did not produce a solution file.")
    with (folder / "results.csv").open() as stream:
        for row in csv.DictReader(stream):
            name = row["quantity"]
            value = float(row["value"])
            if name not in required or name in result or row["unit"] != required[name] or not math.isfinite(value):
                raise RuntimeError("FEMM returned invalid quantities or units.")
            result[name] = {"value": value, "unit": row["unit"]}
    if set(result) != set(required):
        raise RuntimeError("FEMM did not return all requested quantities.")
    return result


def field_plot(folder: Path, spec=None):
    with (folder / "field.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 31 * 41:
        raise RuntimeError("FEMM returned an incomplete field grid.")
    values = [(float(r["x_mm"]), float(r["y_mm"]), math.hypot(float(r["bx_t"]), float(r["by_t"]))) for r in rows]
    if not all(math.isfinite(x) and math.isfinite(y) and math.isfinite(b) for x, y, b in values):
        raise RuntimeError("FEMM returned a non-finite field value.")
    peak = max(b for _, _, b in values)
    cells = []
    for i, (_, _, b) in enumerate(values):
        hue = 220 - 210 * b / peak if peak else 220
        cells.append(f'<rect x="{(i%41)*10}" y="{(30-i//41)*10}" width="10" height="10" fill="hsl({hue:.2f},78%,54%)"/>')
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 410 344" role="img" aria-label="Sampled magnetic flux density magnitude">'
    axes = "r →, z ↑" if getattr(spec, "physics", None) == "axisymmetric_dc_magnetics" else "x →, y ↑"
    svg += "".join(cells) + f'<text x="8" y="332" fill="#172b36" font-family="sans-serif" font-size="12">|B|: blue = 0 T; red = {peak:.4g} T. {axes}</text></svg>'
    (folder / "field.svg").write_text(svg)


class SolverWorker:
    def __init__(self, store: JobStore, command="femm-lua", timeout=180, retention_hours=24, *, revision_store=None, replication_store=None):
        self.store, self.command, self.timeout, self.retention = store, command, timeout, retention_hours
        self.stopping = threading.Event()
        self.thread = None
        self.process = None
        self.revision_store = revision_store
        self.replication_store = replication_store

    def record_check(self, job, folder, stage, status, details):
        if not self.revision_store or not job.get("revision_id"):
            return
        # Capture actual files that exist at this stage, not inferred output paths.
        artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in folder.iterdir() if p.is_file() and p.name in
                     {"spec.json", "model.lua", "model.fem", "model.ans", "mesh-report.json", "results.csv", "solver.log", "target-contract.json", "target-report.json"}}
        manifest = {"job_id": job["id"], "revision_id": job["revision_id"],
                    "stage": stage, "status": status, "artifacts": artifacts}
        encoded = json.dumps(manifest, sort_keys=True, allow_nan=False)
        (folder / (stage + "-execution.json")).write_text(encoded)
        check = self.revision_store.record_check(job["revision_id"], stage=stage, status=status,
                    job_id=job["id"], artifact_hash=hashlib.sha256(encoded.encode()).hexdigest(),
                    details={**details, "manifest": manifest})
        previous = self.store.get(job["id"])["evidence"].get("check_ids", [])
        self.store.annotate(job["id"], check_ids=[*previous, check["id"]],
                            **({"failure_id": check["id"]} if status == "failed" else {}))

    def fail(self, job, folder, stage, error):
        try:
            self.record_check(job, folder, stage, "failed", {"error": error})
        except Exception:
            # An unavailable/tampered journal must not leave a job running or
            # turn an unrecorded computation into a successful job.
            error += " Model check evidence could not be recorded."
        self.store.update(job["id"], "failed", error=error)

    def start(self):
        self.store.recover()
        self.thread = threading.Thread(target=self.loop, name="femm-worker", daemon=True)
        self.thread.start()

    def stop(self):
        self.stopping.set()
        self.kill_process()
        if self.thread:
            self.thread.join(timeout=5)

    def kill_process(self):
        process = self.process
        if process:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def loop(self):
        while not self.stopping.is_set():
            try:
                self.store.expire(self.retention)
                job = self.store.next()
                if job:
                    self.run(job)
            except Exception:
                # A damaged job must not terminate the worker loop.
                import logging
                logging.exception("FEMM worker iteration failed")
            self.stopping.wait(1)

    def run(self, job):
        job_id = job["id"]
        self.store.update(job_id, "running")
        folder = self.store.root / job_id
        folder.mkdir(exist_ok=True)
        stage = "mesh" if job.get("kind") == "mesh" else "solver"
        try:
            is_plan = job["spec"].get("kind") == "magnetostatic_plan"
            mesh_only = job.get("kind", "solve") == "mesh"
            spec = GeometryPlan.model_validate(job["spec"]) if is_plan else CCoreSpec.model_validate(job["spec"])
            target_record = None
            if job.get("target_contract_id"):
                if not self.replication_store:
                    raise RuntimeError("The frozen target store is unavailable.")
                from .targets import TargetContract, model_fingerprint
                target_record = self.replication_store.get_contract(job["target_contract_id"])
                if (not target_record or target_record["revision_id"] != job.get("revision_id") or
                        TargetContract.model_validate(target_record["contract"]).model_sha256 != model_fingerprint(spec)):
                    raise RuntimeError("The frozen target contract does not match this job revision.")
                (folder / "target-contract.json").write_text(json.dumps(target_record, indent=2, allow_nan=False))
            files = plan_generate.source_files(spec, mesh_only=mesh_only) if is_plan else source_files(spec)
            expected = plan_generate.required_results(spec) if is_plan else None
            for name, content in files.items():
                (folder / name).write_text(content)
            if self.revision_store and job.get("revision_id"):
                revision = self.revision_store.get(job["revision_id"])
                if revision is None or revision["plan"] != job["spec"]:
                    raise RuntimeError("Job arguments differ from their immutable model revision.")
                (folder / "revision.json").write_text(json.dumps(revision, indent=2, allow_nan=False))
            if not shutil.which(self.command):
                raise RuntimeError("FEMM 5 launcher not found. Install FEMM 5 and configure FEMM_LUA.")
            # One worker owns this port; it is never exposed by the container.
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "LD_LIBRARY_PATH", "FEMM_DIR") if key in os.environ}
            env["QT_QPA_PLATFORM"] = "offscreen"
            if is_plan:
                env["FEMM_MESH_EXPORT"] = "1"
            started = time.monotonic()
            with (folder / "solver.log").open("w") as log:
                try:
                    self.process = subprocess.Popen([self.command, "-h", "-p", str(port), str(folder / "model.lua")], cwd=folder, env=env,
                                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    self.process.wait(timeout=self.timeout)
                    if self.process.returncode:
                        raise RuntimeError("FEMM exited with an error. Download the job log to inspect the failure.")
                finally:
                    self.kill_process()
                    if self.process:
                        self.process.wait(timeout=5)
                    self.process = None
            # A full solve also performs native meshing. Attribute deterministic
            # mesh validation failures to that stage before reading field output.
            if is_plan:
                stage = "mesh"
            mesh_report = validate_mesh(folder, spec) if is_plan else None
            if mesh_report is not None:
                (folder / "mesh-report.json").write_text(json.dumps(mesh_report, indent=2, allow_nan=False))
                self.record_check(job, folder, "mesh", "passed", {"mesh_report": mesh_report})
            if mesh_only:
                if (folder / "model.ans").exists() or (folder / "results.csv").exists():
                    raise RuntimeError("A mesh-only job unexpectedly produced a field solution.")
                result = mesh_report
            else:
                stage = "solver"
                result = read_results(folder, expected)
                field_plot(folder, spec)
                self.record_check(job, folder, "solver", "passed", {"quantity_names": list(result)})
                if target_record:
                    from .targets import TargetContract, evaluate_target_file, contract_fingerprint
                    stage = "target"
                    contract = TargetContract.model_validate(target_record["contract"])
                    target_report = evaluate_target_file(contract, folder, plan=spec,
                                            expected_contract_sha256=target_record["contract_sha256"])
                    (folder / "target-report.json").write_text(json.dumps(target_report, indent=2, allow_nan=False))
                    self.store.annotate(job_id, target_report=target_report)
                    self.record_check(job, folder, "target", "passed" if target_report["status"] == "passed" else "failed",
                                      {"target_report": target_report, "target_contract_id": job["target_contract_id"]})
            result["runtime"] = {"value": round(time.monotonic()-started, 3), "unit": "s"}
            (folder / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
            allowed = ["model.lua", "model.py", "run.py", "spec.json", "geometry.svg", "README.txt", "model.fem", "model.ans", "results.csv", "field.csv", "field.svg", "results.json", "solver.log", "mesh.csv", "mesh.svg", "mesh-report.json", "model.node", "model.ele", "model.edge", "model.1.node", "model.1.ele", "model.1.edge"]
            allowed.extend(["revision.json", "mesh-execution.json", "solver-execution.json"])
            allowed.extend(["target-contract.json", "target-report.json", "target-execution.json"])
            with zipfile.ZipFile(folder / "download.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for name in allowed:
                    if (folder / name).is_file():
                        archive.write(folder / name, name)
            self.store.update(job_id, "completed", result=result)
        except subprocess.TimeoutExpired:
            error = f"Simulation exceeded the {self.timeout} second limit. Try a coarser mesh."
            self.fail(job, folder, stage, error)
        except Exception as exc:
            self.fail(job, folder, stage, str(exc))
