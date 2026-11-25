"""Validated imports and read-only SQLite/FTS5 lookup with optional native evidence."""
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile

from .source import DATA, DEFAULT_CACHE, fingerprint, implementation_definitions, load_sources, lua_definitions, manual_definitions, verified_text

APPLICATION_ID = 1178758466
DEFAULT_DB = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "femm-brain2/catalogue.sqlite3"
STOPWORDS = {"a", "an", "the", "how", "do", "does", "i", "to", "in", "of", "for", "with", "using"}


def load_batch(path=None):
    return json.loads(Path(path or DATA / "commands.json").read_text())


def validate_batch(batch, sources, cache=DEFAULT_CACHE):
    """Check contracts and frozen source anchors. Semantic review remains explicit."""
    errors, definitions, texts, seen = [], {}, {}, set()
    if batch.get("schema_version") != 1:
        errors.append("Unsupported batch schema version")
    source_index = {s["id"]: s for s in sources}
    for source in sources:
        text = verified_text(source, cache)
        texts[source["id"]] = text
        if source["role"] == "api_manual":
            definitions[source["id"]] = manual_definitions(text)
        elif source["role"] == "lua_binding":
            definitions[source["id"]] = lua_definitions(text)
        elif source["role"] == "implementation":
            definitions[source["id"]] = implementation_definitions(text, source)
    for record in batch.get("commands", []):
        name = record.get("name", "<missing>")
        identity = (name, record.get("version"), record.get("binding"))
        if identity in seen:
            errors.append(name + ": duplicate version/binding entry")
        seen.add(identity)
        required = {"name", "version", "binding", "physics", "category", "purpose", "signature", "parameters",
                    "returns", "preconditions", "effects", "examples", "references", "review", "execution", "notes"}
        if required - set(record):
            errors.append(name + ": missing fields " + str(sorted(required - set(record))))
            continue
        if record["review"].get("status") != "source_checked":
            errors.append(name + ": review status must be source_checked")
        if record["execution"].get("status") != "not_tested":
            errors.append(name + ": this documentation-only import cannot assert execution success")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            errors.append(name + ": invalid command name")
        if record["version"] != batch.get("target", {}).get("version") or record["binding"] != batch.get("target", {}).get("binding"):
            errors.append(name + ": version/binding disagrees with this import batch")
        signature = re.fullmatch(re.escape(name) + r"\(([^()]*)\)", record["signature"])
        names = [p.get("name") for p in record["parameters"]]
        if not signature or [n.strip() for n in signature.group(1).split(",") if n.strip()] != names:
            errors.append(name + ": signature and parameter order disagree")
        if len(set(names)) != len(names):
            errors.append(name + ": duplicate parameter")
        optional_seen = False
        for parameter in record["parameters"]:
            if not {"name", "type", "required", "units", "description", "default", "default_status", "allowed_values"} <= set(parameter):
                errors.append(name + ": incomplete parameter metadata")
                continue
            if not isinstance(parameter["required"], bool):
                errors.append(name + ": required must be boolean")
            if optional_seen and parameter["required"]:
                errors.append(name + ": required parameter follows optional parameter")
            optional_seen = optional_seen or parameter["required"] is False
            if parameter["default_status"] == "not_documented" and parameter["default"] is not None:
                errors.append(name + ": undocumented default must remain null")
        roles = set()
        for reference in record["references"]:
            source = source_index.get(reference.get("source_id"))
            definition = definitions.get(reference.get("source_id"), {}).get(name)
            if source is None or definition is None:
                errors.append(name + ": cited command definition is missing")
                continue
            roles.add(source["role"])
            if reference.get("definition_sha256") != definition.digest or reference.get("pdf_pages", []) != list(definition.pages):
                errors.append(name + ": definition fingerprint or page locator differs from source")
            if reference.get("anchor") != name:
                errors.append(name + ": source anchor must identify the command")
            if list(definition.line_range) != reference.get("line_range", []):
                errors.append(name + ": implementation line locator differs from source")
            if source["role"] == "lua_binding":
                exported = re.match(r"function\s+\w+\(([^)]*)\)", definition.text)
                arguments = [a.strip() for a in exported.group(1).split(",") if a.strip()] if exported else []
                if len(arguments) != len(names):
                    errors.append(name + ": parameter count disagrees with the Lua wrapper")
        if not {"api_manual", "lua_binding"} <= roles:
            errors.append(name + ": both manual and Lua binding must be checked")
        if not record["examples"] or any(e.get("origin") != "agent_authored" or e.get("execution_status") != "not_tested"
                                        or name + "(" not in e.get("code", "") for e in record["examples"]):
            errors.append(name + ": illustrative examples must be labelled and contain the command")
    manual = next((s for s in sources if s["role"] == "api_manual"), None)
    inventory = definitions.get(manual["id"], {}) if manual else {}
    return {"status": "passed" if not errors else "failed", "errors": errors,
            "source_count": len(sources), "reviewed_commands": len(seen),
            "detected_manual_definitions": len(inventory),
            "inventory": [{"name": d.name, "pdf_pages": list(d.pages), "section": d.section,
                           "other_definition_pages": [list(p) for p in d.additional_pages],
                           "definition_sha256": d.digest, "source_id": manual["id"],
                           "additional_definition_pages": list(d.additional_pages)} for d in inventory.values()]}


def connect_readonly(path=DEFAULT_DB):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("Catalogue is not built. Run: python3 -m femm_brain2 build")
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    if db.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
        db.close()
        raise ValueError("This file is not a femm_brain2 catalogue")
    return db


def build(path=DEFAULT_DB, *, batch_path=None, sources_path=None, cache=DEFAULT_CACHE, evidence_path=None,
          knowledge_path=None, paper_evidence=None):
    batch, sources = load_batch(batch_path), load_sources(sources_path)
    validation = validate_batch(batch, sources, cache)
    if validation["errors"]:
        raise ValueError("Import rejected: " + "; ".join(validation["errors"]))
    knowledge = None
    if knowledge_path is not None:
        from .knowledge import KNOWLEDGE, load_knowledge, validate_knowledge
        knowledge = load_knowledge(KNOWLEDGE if knowledge_path is True else knowledge_path)
        validate_knowledge(knowledge, source_check=True)
    elif paper_evidence is not None:
        raise ValueError('Paper evidence requires --knowledge')
    executions = {}
    evidence_paths = [] if evidence_path is None else (list(evidence_path) if isinstance(evidence_path, (list, tuple)) else [evidence_path])
    if evidence_path is not None:
        from .geometry_check import verify_evidence
        for evidence in evidence_paths:
            verified = verify_evidence(evidence, batch)
            if executions.keys() & verified.keys():
                raise ValueError("Overlapping native evidence; select one report per command")
            executions.update(verified)
    target = Path(path).resolve()
    if target.exists():
        with closing(connect_readonly(target)):
            pass
    target.parent.mkdir(parents=True, exist_ok=True)
    # Build completely before replacing this project's disposable search database.
    with tempfile.TemporaryDirectory(prefix=".femm-brain2-", dir=target.parent) as temporary:
        staging = Path(temporary) / "catalogue.sqlite3"
        with closing(sqlite3.connect(staging)) as db, db:
            db.execute('PRAGMA foreign_keys=ON')
            db.executescript(Path(__file__).with_name("schema.sql").read_text())
            metadata = {"schema_version": 1, "target": batch["target"], "milestone": batch["milestone"],
                        "built_at": datetime.now(timezone.utc).isoformat(),
                        "batch_sha256": fingerprint(json.dumps(batch, sort_keys=True).encode()),
                        "source_verification": "hashes, definition fingerprints, page locators and Lua exports checked",
                        "inventory_scope": "Detected FEMM command definitions in the pinned manual; inventory entries beyond the reviewed batch are not semantic catalogue entries."}
            if evidence_path is not None:
                metadata["native_evidence"] = {"reports": [{"report": str(Path(p).resolve()),
                    "report_sha256": fingerprint(Path(p).read_bytes())} for p in evidence_paths],
                    "scope": "Native API checks and, where supplied, analytical field-solution checks. See individual reports for coverage."}
            db.executemany("INSERT INTO metadata VALUES (?,?)", [(k, json.dumps(v)) for k, v in metadata.items()])
            db.executemany("INSERT INTO sources VALUES (?,?)", [(s["id"], json.dumps(s)) for s in sources])
            db.executemany("INSERT INTO inventory VALUES (?,?)", [(d["name"], json.dumps(d)) for d in validation["inventory"]])
            for record in batch["commands"]:
                if record["name"] in executions:
                    record = {**record, "execution": executions[record["name"]]}
                cursor = db.execute("INSERT INTO commands(name,version,binding,physics,category,data) VALUES (?,?,?,?,?,?)",
                                    (*[record[k] for k in ("name", "version", "binding", "physics", "category")], json.dumps(record)))
                details = json.dumps({k: record[k] for k in ("parameters", "returns", "preconditions", "effects", "notes")})
                for selectors in ("property_selectors", "result_selectors"):
                    if selectors in record:
                        details += " " + json.dumps(record[selectors])
                db.execute("INSERT INTO search_index(rowid,name,purpose,details) VALUES (?,?,?,?)",
                           (cursor.lastrowid, record["name"], record["purpose"], details))
            if knowledge is not None:
                from .knowledge import install
                install(db, knowledge)
                if paper_evidence is not None:
                    from .knowledge_run import audit
                    report = audit(paper_evidence, knowledge)
                    db.execute('INSERT INTO metadata VALUES (?,?)',('knowledge_native_evidence',json.dumps(report)))
            if db.execute('PRAGMA foreign_key_check').fetchone() is not None:
                raise ValueError('SQLite relationship integrity check failed')
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("SQLite integrity check failed")
        staging.replace(target)
    return {"status": "built", "database": str(target), "commands": len(batch["commands"]),
            "detected_manual_definitions": validation["detected_manual_definitions"]}


def _source_links(record, db):
    for reference in record["references"]:
        row = db.execute("SELECT data FROM sources WHERE id=?", (reference["source_id"],)).fetchone()
        source = json.loads(row[0])
        reference.update(url=source["url"], source_title=source["title"], source_sha256=source["sha256"])
    return record


def search(query, path=DEFAULT_DB, *, version=None, binding=None, physics=None, limit=5):
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    terms = [t for t in re.findall(r"\w+", query.lower()) if t not in STOPWORDS][:24]
    if not terms:
        return []
    match = " AND ".join('"' + term + '"' for term in terms)
    sql = "SELECT c.data FROM search_index JOIN commands c ON c.id=search_index.rowid WHERE search_index MATCH ?"
    arguments = [match]
    for field, value in (("version", version), ("binding", binding), ("physics", physics)):
        if value is not None:
            sql += " AND c." + field + "=?"
            arguments.append(value)
    sql += " ORDER BY bm25(search_index,10,5,1), c.name LIMIT ?"
    with closing(connect_readonly(path)) as db:
        return [_source_links(json.loads(row[0]), db) for row in db.execute(sql, (*arguments, limit)).fetchall()]


def get(name, path=DEFAULT_DB, *, version=None, binding=None):
    sql, arguments = "SELECT data FROM commands WHERE name=?", [name]
    for field, value in (("version", version), ("binding", binding)):
        if value is not None:
            sql += " AND " + field + "=?"
            arguments.append(value)
    with closing(connect_readonly(path)) as db:
        matches = [_source_links(json.loads(r[0]), db) for r in db.execute(sql, arguments).fetchall()]
        pending = db.execute("SELECT data FROM inventory WHERE name=?", (name,)).fetchone()
        target = json.loads(db.execute("SELECT value FROM metadata WHERE key='target'").fetchone()[0])
        if (version and version != target["version"]) or (binding and binding != target["binding"]):
            pending = None
    if len(matches) > 1:
        raise ValueError("Name is ambiguous; specify --version and --binding")
    if matches:
        return matches[0]
    return {"status": "not_reviewed" if pending else "not_found", "name": name,
            "inventory_reference": json.loads(pending[0]) if pending else None}


def stats(path=DEFAULT_DB, *, include_pending=False):
    with closing(connect_readonly(path)) as db:
        metadata = {r[0]: json.loads(r[1]) for r in db.execute("SELECT key,value FROM metadata")}
        reviewed = db.execute("SELECT count(*) FROM commands").fetchone()[0]
        inventory = db.execute("SELECT count(*) FROM inventory").fetchone()[0]
        pending = [r[0] for r in db.execute("SELECT name FROM inventory WHERE name NOT IN (SELECT name FROM commands) ORDER BY name")]
        sources = [json.loads(r[0]) for r in db.execute("SELECT data FROM sources ORDER BY id")]
        records = [json.loads(r[0]) for r in db.execute("SELECT data FROM commands")]
    return {**metadata, "reviewed_commands": reviewed, "detected_manual_definitions": inventory,
            "pending_commands": len(pending), "pending_sample": pending[:10], "sources": sources,
            **({"pending_names": pending} if include_pending else {}),
            "needs_clarification": [r["name"] for r in records if r["review"]["semantic_status"] == "needs_clarification"],
            "execution_tested_commands": sum(r["execution"]["status"] != "not_tested" for r in records),
            "execution_passed_commands": sum(r["execution"]["status"] == "passed" for r in records),
            "execution_failed_commands": [r["name"] for r in records if r["execution"]["status"] == "check_failed"]}
