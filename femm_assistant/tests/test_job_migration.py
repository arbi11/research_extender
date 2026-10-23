"""Existing pilot installations retain their jobs when mesh jobs are introduced."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from app.jobs import JobStore
from app.model import CCoreSpec
from app.plan import wire_example


class JobMigrationTests(unittest.TestCase):
    def test_existing_seven_column_database_preserves_jobs_and_adds_mesh_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_id = "a" * 32
            with sqlite3.connect(root / "jobs.sqlite3") as db:
                db.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, spec TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, result TEXT, error TEXT)")
                db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?)", (
                    original_id, CCoreSpec().model_dump_json(), "completed", 1, 1,
                    json.dumps({"energy": {"value": 1, "unit": "J"}}), None))
            store = JobStore(root)
            old = store.get(original_id)
            self.assertEqual(old["kind"], "solve")
            self.assertEqual(old["result"]["energy"]["value"], 1)
            mesh = store.create(wire_example(), kind="mesh")
            self.assertEqual(mesh["kind"], "mesh")
            self.assertEqual(JobStore(root).get(mesh["id"])["spec"], mesh["spec"])
            self.assertEqual(store.next()["id"], mesh["id"])


if __name__ == "__main__":
    unittest.main()
