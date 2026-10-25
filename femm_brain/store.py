"""A disposable SQLite query index over the portable canonical JSON graph."""
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3


class GraphStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS graph (id INTEGER PRIMARY KEY CHECK(id=1), data TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, entity TEXT, title TEXT, layer INTEGER, kind TEXT, text TEXT, data TEXT)")
            try:
                db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(id UNINDEXED, title, body)")
            except sqlite3.OperationalError:
                pass

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def replace(self, graph):
        from .core import validate_graph
        issues = validate_graph(graph) + graph.get("consolidation_conflicts", [])
        if issues:
            raise ValueError("Cannot index an invalid graph: " + json.dumps(issues[:5]))
        with closing(self.connect()) as db, db:
            db.execute("DELETE FROM documents")
            fts = db.execute("SELECT 1 FROM sqlite_master WHERE name='search_index'").fetchone() is not None
            if fts:
                db.execute("DELETE FROM search_index")
            for collection in ("nodes", "examples", "sources"):
                for item in graph.get(collection, []):
                    identifier = collection + ":" + item["id"]
                    body = json.dumps(item, ensure_ascii=False, allow_nan=False)
                    title = item.get("title", item["id"])
                    db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?)", (identifier, collection, title, item.get("layer"), item.get("kind", collection), body, body))
                    if fts:
                        db.execute("INSERT INTO search_index VALUES (?,?,?)", (identifier, title, body))
            db.execute("INSERT OR REPLACE INTO graph VALUES (1,?)", (json.dumps(graph, ensure_ascii=False, allow_nan=False),))

    def graph(self):
        with closing(self.connect()) as db:
            row = db.execute("SELECT data FROM graph WHERE id=1").fetchone()
        if not row:
            raise ValueError("No built graph. Run python -m femm_brain build.")
        return json.loads(row[0])

    def search(self, query, limit=50):
        limit = min(max(int(limit), 1), 200)
        terms = re.findall(r"[\w]+", str(query), flags=re.UNICODE)[:16]
        if not terms:
            return []
        with closing(self.connect()) as db:
            fts = db.execute("SELECT 1 FROM sqlite_master WHERE name='search_index'").fetchone() is not None
            if fts:
                rows = db.execute("SELECT d.*, bm25(search_index) AS score FROM search_index JOIN documents d USING(id) WHERE search_index MATCH ? ORDER BY score LIMIT ?",
                                  (" OR ".join('"' + t + '"' for t in terms), limit)).fetchall()
            else:
                rows = db.execute("SELECT *, 0 AS score FROM documents WHERE " + " OR ".join("text LIKE ?" for _ in terms) + " LIMIT ?",
                                  tuple("%" + t + "%" for t in terms) + (limit,)).fetchall()
        return [{"id": json.loads(r["data"])["id"], "entity": r["entity"], "title": r["title"],
                 "layer": r["layer"], "kind": r["kind"], "description": json.loads(r["data"]).get("description", json.loads(r["data"]).get("summary", "")),
                 "score": r["score"]} for r in rows]
