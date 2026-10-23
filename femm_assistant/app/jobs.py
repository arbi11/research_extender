"""Persistent jobs and a single, resource-bounded local solver worker."""
from contextlib import closing
import csv
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


class BusyError(Exception):
    pass


class JobStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "jobs.sqlite3"
        with closing(self.connect()) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, spec TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, result TEXT, error TEXT)")

    def connect(self):
        db = sqlite3.connect(self.db, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def create(self, spec: CCoreSpec | GeometryPlan, max_pending=5):
        now = time.time()
        job_id = uuid.uuid4().hex
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if count >= max_pending:
                raise BusyError("The simulation queue is full. Please try again after a job finishes.")
            db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?)", (job_id, spec.model_dump_json(), "queued", now, now, None, None))
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
        return value

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


def field_plot(folder: Path):
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
    svg += "".join(cells) + f'<text x="8" y="332" fill="#172b36" font-family="sans-serif" font-size="12">|B|: blue = 0 T; red = {peak:.4g} T. x →, y ↑</text></svg>'
    (folder / "field.svg").write_text(svg)


class SolverWorker:
    def __init__(self, store: JobStore, command="femm-lua", timeout=180, retention_hours=24):
        self.store, self.command, self.timeout, self.retention = store, command, timeout, retention_hours
        self.stopping = threading.Event()
        self.thread = None
        self.process = None

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
        try:
            is_plan = job["spec"].get("kind") == "magnetostatic_plan"
            spec = GeometryPlan.model_validate(job["spec"]) if is_plan else CCoreSpec.model_validate(job["spec"])
            files = plan_generate.source_files(spec) if is_plan else source_files(spec)
            expected = plan_generate.required_results(spec) if is_plan else None
            for name, content in files.items():
                (folder / name).write_text(content)
            if not shutil.which(self.command):
                raise RuntimeError("FEMM 5 launcher not found. Install FEMM 5 and configure FEMM_LUA.")
            # One worker owns this port; it is never exposed by the container.
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "LD_LIBRARY_PATH", "FEMM_DIR") if key in os.environ}
            env["QT_QPA_PLATFORM"] = "offscreen"
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
            result = read_results(folder, expected)
            field_plot(folder)
            result["runtime"] = {"value": round(time.monotonic()-started, 3), "unit": "s"}
            (folder / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
            allowed = ["model.lua", "run.py", "spec.json", "geometry.svg", "README.txt", "model.fem", "model.ans", "results.csv", "field.csv", "field.svg", "results.json", "solver.log"]
            with zipfile.ZipFile(folder / "download.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for name in allowed:
                    archive.write(folder / name, name)
            self.store.update(job_id, "completed", result=result)
        except subprocess.TimeoutExpired:
            self.store.update(job_id, "failed", error=f"Simulation exceeded the {self.timeout} second limit. Try a coarser mesh.")
        except Exception as exc:
            self.store.update(job_id, "failed", error=str(exc))
