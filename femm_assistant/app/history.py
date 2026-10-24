"""Immutable model revisions and bounded recovery from recorded native failures.

Paths address stable feature ids (``/regions/wire/shape/radius``), never a
region's incidental list index. This journal records evidence, not a claim of
causation: the last writer is a useful candidate, not necessarily the root cause.
Only the application should record checks after observing their real outcomes.
"""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .plan import GeometryPlan


ID_COLLECTIONS = {"regions", "materials", "circuits"}
STAGES = ("mesh", "solver", "target")
PATH_PATTERN = r"^/[a-zA-Z0-9_]+(?:/[a-zA-Z0-9_]+)*$"


class HistoryConflict(ValueError):
    """The caller is editing a stale branch or has conflicting evidence."""


class HistoryIntegrityError(ValueError):
    """A persisted checkpoint no longer matches its content hash."""


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source: Literal["paper", "user", "calculation", "assumption", "artifact"]
    reference: str = Field(min_length=1, max_length=2000)
    detail: str = Field(default="", max_length=4000)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class FeatureEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    op: Literal["replace", "add", "remove"]
    path: str = Field(pattern=PATH_PATTERN, max_length=250)
    value: Any = None

    @model_validator(mode="after")
    def operation_value(self):
        if self.op != "remove" and "value" not in self.model_fields_set:
            raise ValueError("Add and replace edits require an explicit value.")
        if self.op == "remove" and self.value is not None:
            raise ValueError("Remove edits do not accept a value.")
        if "id" in self.path.split("/")[1:]:
            raise ValueError("Feature ids are immutable; add or remove a named feature explicitly.")
        return self


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _tree(plan: GeometryPlan) -> dict:
    result = plan.model_dump(mode="json")
    for name in ID_COLLECTIONS:
        result[name] = {item["id"]: item for item in result[name]}
    return result


def _plan(tree: dict) -> GeometryPlan:
    value = deepcopy(tree)
    for name in ID_COLLECTIONS:
        value[name] = list(value[name].values())
    return GeometryPlan.model_validate(value)


def _path(value: str) -> list[str]:
    if len(value) > 250 or not re.fullmatch(PATH_PATTERN, value):
        raise ValueError("Use an absolute feature path with names or canonical list indices.")
    return value[1:].split("/")


def _within(path: str, scope: str) -> bool:
    return path == scope or path.startswith(scope + "/")


def _related(first: str, second: str) -> bool:
    return _within(first, second) or _within(second, first)


def _diff(before: Any, after: Any, path="") -> list[dict]:
    if type(before) is not type(after):
        return [{"path": path, "before": before, "after": after}]
    if isinstance(before, dict):
        changes = []
        for key in sorted(set(before) | set(after)):
            at = path + "/" + key
            if key not in before:
                changes.append({"path": at, "before": None, "after": after[key], "op": "add"})
            elif key not in after:
                changes.append({"path": at, "before": before[key], "after": None, "op": "remove"})
            else:
                changes.extend(_diff(before[key], after[key], at))
        return changes
    if isinstance(before, list) and len(before) == len(after):
        return [change for i, (old, new) in enumerate(zip(before, after))
                for change in _diff(old, new, path + "/" + str(i))]
    return [] if before == after else [{"path": path, "before": before, "after": after}]


def plan_diff(before: GeometryPlan, after: GeometryPlan) -> list[dict]:
    """Return an id-addressed diff; feature reordering is not an edit."""
    return _diff(_tree(before), _tree(after))


def patch_plan(plan: GeometryPlan, edits: list[FeatureEdit | dict], allowed_paths: list[str]) -> tuple[GeometryPlan, list[dict]]:
    """Apply only explicit scopes and validate the complete resulting model.

    Add/remove is restricted to whole named features. Existing scalar, shape,
    point or list fields can be replaced, with unchanged fields retained exactly.
    A caller may use a narrow path such as one vertex coordinate as its scope.
    """
    if not edits or len(edits) > 64 or not allowed_paths or len(allowed_paths) > 64:
        raise ValueError("A patch needs 1–64 edits and 1–64 explicit allowed paths.")
    for scope in allowed_paths:
        _path(scope)
        if scope.split("/")[-1] == "id":
            raise ValueError("Feature ids cannot be an edit scope.")
    typed = [FeatureEdit.model_validate(edit) for edit in edits]
    paths = [edit.path for edit in typed]
    if any(_related(first, second) for i, first in enumerate(paths) for second in paths[i + 1:]):
        raise ValueError("Patch paths may not repeat, overlap or depend on edit order.")
    tree = _tree(plan)
    for edit in typed:
        if not any(_within(edit.path, scope) for scope in allowed_paths):
            raise ValueError(f"Edit '{edit.path}' is outside the explicitly allowed paths.")
        parts = _path(edit.path)
        if parts[0] in ID_COLLECTIONS and len(parts) == 1:
            raise ValueError("Edit a named feature; whole feature collections cannot be replaced.")
        if edit.op != "replace" and not (parts[0] in ID_COLLECTIONS and len(parts) == 2):
            raise ValueError("Add/remove operations require a whole named region, material or circuit.")
        parent: Any = tree
        for part in parts[:-1]:
            if isinstance(parent, list):
                if not part.isdecimal() or str(int(part)) != part or int(part) >= len(parent):
                    raise ValueError(f"Unknown list index in '{edit.path}'.")
                parent = parent[int(part)]
            elif isinstance(parent, dict) and part in parent:
                parent = parent[part]
            else:
                raise ValueError(f"Unknown feature path '{edit.path}'.")
        key: str | int = parts[-1]
        if isinstance(parent, list):
            if not key.isdecimal() or str(int(key)) != key or int(key) >= len(parent):
                raise ValueError(f"Unknown list index in '{edit.path}'.")
            key = int(key)
        elif not isinstance(parent, dict):
            raise ValueError(f"Unknown feature path '{edit.path}'.")
        exists = (key in parent) if isinstance(parent, dict) else True
        if edit.op == "add":
            if exists:
                raise ValueError(f"Feature '{edit.path}' already exists.")
            if not isinstance(edit.value, dict) or edit.value.get("id") != parts[-1]:
                raise ValueError("A new feature's id must match its path.")
            parent[key] = deepcopy(edit.value)
        elif not exists:
            raise ValueError(f"Unknown feature path '{edit.path}'.")
        elif edit.op == "remove":
            del parent[key]
        else:
            if parts[0] in ID_COLLECTIONS and len(parts) == 2:
                if not isinstance(edit.value, dict) or edit.value.get("id") != parts[-1]:
                    raise ValueError("Replacement feature ids must match their existing paths.")
            parent[key] = deepcopy(edit.value)
    candidate = _plan(tree)
    changes = plan_diff(plan, candidate)
    if not changes:
        raise ValueError("The patch does not change the model.")
    if any(not any(_within(item["path"], scope) for scope in allowed_paths) for item in changes):
        raise ValueError("Validated changes exceed the explicitly allowed paths.")
    return candidate, changes


def apply_feature_edits(plan: GeometryPlan, edits: list[FeatureEdit | dict],
                        allowed_paths: list[str] | None = None) -> tuple[GeometryPlan, list[dict]]:
    """Apply typed planner output without persistence, preserving all other fields.

    If a caller has no independently supplied scope, the concrete edit paths are
    the scopes. This bounds execution but does not establish that a language
    model interpreted the human's permission correctly.
    """
    typed = [FeatureEdit.model_validate(edit) for edit in edits]
    return patch_plan(plan, typed, allowed_paths if allowed_paths is not None else [edit.path for edit in typed])


def affected_stages(changes: list[dict]) -> list[str]:
    """Conservative invalidation; never reuse a mesh across physical changes."""
    paths = [change["path"] for change in changes]
    if all(_within(path, "/title") for path in paths):
        return []
    if all(path.startswith("/probes") or path == "/force_region" or path == "/title" for path in paths):
        return ["solver", "target"]
    return list(STAGES)


def feature_dependencies(plan: GeometryPlan) -> dict[str, list[str]]:
    """Explicit model dependencies for failure localization, not causal proof."""
    dependencies = {"/air_label": ["/domain"] + [f"/regions/{r.id}/shape" for r in plan.regions]}
    for region in plan.regions:
        base = "/regions/" + region.id
        dependencies[base] = ["/domain"]
        if region.material != "air":
            dependencies[base].append("/materials/" + region.material)
        if region.circuit:
            dependencies[base].append("/circuits/" + region.circuit)
        dependencies[base + "/label"] = [base + "/shape"]
    return dependencies


class ModelRevisionStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "model_revisions.sqlite3"
        with closing(self.connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS revisions (
                    id TEXT PRIMARY KEY, parent_id TEXT REFERENCES revisions(id), record_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS branches (
                    name TEXT PRIMARY KEY, head_id TEXT NOT NULL REFERENCES revisions(id));
                CREATE TABLE IF NOT EXISTS checks (
                    id TEXT PRIMARY KEY, revision_id TEXT NOT NULL REFERENCES revisions(id), record_json TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS checks_by_revision ON checks(revision_id);
                CREATE TRIGGER IF NOT EXISTS immutable_revisions_update BEFORE UPDATE ON revisions
                    BEGIN SELECT RAISE(ABORT, 'Model revisions are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_revisions_delete BEFORE DELETE ON revisions
                    BEGIN SELECT RAISE(ABORT, 'Model revisions are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_checks_update BEFORE UPDATE ON checks
                    BEGIN SELECT RAISE(ABORT, 'Recorded checks are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_checks_delete BEFORE DELETE ON checks
                    BEGIN SELECT RAISE(ABORT, 'Recorded checks are immutable'); END;
            """)

    def connect(self):
        db = sqlite3.connect(self.db, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @staticmethod
    def _branch(name: str):
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name):
            raise ValueError("Use a short branch name with lower-case letters, digits, underscore or hyphen.")

    @staticmethod
    def _evidence(values):
        if len(values) > 128:
            raise ValueError("At most 128 evidence references are allowed.")
        return [EvidenceReference.model_validate(value).model_dump(mode="json") for value in values]

    def _get(self, db, revision_id: str) -> dict | None:
        row = db.execute("SELECT record_json FROM revisions WHERE id=?", (revision_id,)).fetchone()
        if row is None:
            return None
        record = json.loads(row[0])
        if _hash(record) != revision_id or _hash(record["plan"]) != record["plan_hash"]:
            raise HistoryIntegrityError("Stored model revision does not match its immutable hash.")
        return {"id": revision_id, **record}

    def get(self, revision_id: str) -> dict | None:
        with closing(self.connect()) as db:
            return self._get(db, revision_id)

    def head(self, branch="main") -> dict | None:
        self._branch(branch)
        with closing(self.connect()) as db:
            row = db.execute("SELECT head_id FROM branches WHERE name=?", (branch,)).fetchone()
            return self._get(db, row[0]) if row else None

    def _insert(self, db, plan, parent_id, action, evidence, changes, recovery=None):
        value = plan.model_dump(mode="json")
        record = {"parent_id": parent_id, "plan": value, "plan_hash": _hash(value),
                  "action": action, "evidence": self._evidence(evidence), "changes": changes,
                  "dependencies": feature_dependencies(plan), "invalidated_stages": affected_stages(changes),
                  "created": time.time(), "recovery": recovery}
        revision_id = _hash(record)
        db.execute("INSERT INTO revisions VALUES (?,?,?)", (revision_id, parent_id, _json(record)))
        return {"id": revision_id, **record}

    def create(self, plan: GeometryPlan | dict, *, evidence=(), branch="main") -> dict:
        self._branch(branch)
        plan = GeometryPlan.model_validate(plan)
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM branches WHERE name=?", (branch,)).fetchone():
                raise HistoryConflict("This branch already has a model. Edit its current head.")
            result = self._insert(db, plan, None, {"type": "create"}, evidence,
                                  [{"path": "/", "before": None, "after": "initial model"}])
            db.execute("INSERT INTO branches VALUES (?,?)", (branch, result["id"]))
            return result

    def apply_patch(self, parent_id: str, edits: list[FeatureEdit | dict], *, allowed_paths: list[str],
                    evidence=(), branch="main", reason="") -> dict:
        self._branch(branch)
        if len(reason) > 4000:
            raise ValueError("Patch reason is too long.")
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            parent = self._get(db, parent_id)
            if parent is None:
                raise ValueError("Unknown parent revision.")
            head = db.execute("SELECT head_id FROM branches WHERE name=?", (branch,)).fetchone()
            if head is None or head[0] != parent_id:
                raise HistoryConflict("The parent is not this branch's current head. Reload before editing.")
            plan, changes = patch_plan(GeometryPlan.model_validate(parent["plan"]), edits, allowed_paths)
            action = {"type": "patch", "edits": [FeatureEdit.model_validate(e).model_dump(mode="json") for e in edits],
                      "allowed_paths": allowed_paths, "reason": reason}
            result = self._insert(db, plan, parent_id, action, evidence, changes)
            db.execute("UPDATE branches SET head_id=? WHERE name=?", (result["id"], branch))
            return result

    def record_check(self, revision_id: str, *, stage: Literal["mesh", "solver", "target"],
                     status: Literal["passed", "failed"], job_id: str,
                     artifact_hash: str | None = None, details: dict | None = None,
                     implicated_paths: list[str] | None = None) -> dict:
        """Append a server-observed check; it does not certify causation or amend history."""
        if stage not in STAGES or status not in {"passed", "failed"}:
            raise ValueError("A check requires a mesh, solver or target stage and passed/failed outcome.")
        if not job_id or len(job_id) > 200:
            raise ValueError("Checks require an observed job or evaluation id.")
        if artifact_hash is not None and not re.fullmatch(r"[0-9a-f]{64}", artifact_hash):
            raise ValueError("Artifact hashes must be SHA-256 hex digests.")
        paths = implicated_paths or []
        if len(paths) > 128:
            raise ValueError("Too many implicated paths.")
        for path in paths:
            _path(path)
        record = {"revision_id": revision_id, "stage": stage, "status": status, "job_id": job_id,
                  "artifact_hash": artifact_hash, "details": details or {}, "implicated_paths": paths,
                  "created": time.time(), "nonce": uuid.uuid4().hex}
        encoded = _json(record)
        if len(encoded) > 64000:
            raise ValueError("Check evidence exceeds the journal size limit.")
        check_id = _hash(record)
        with closing(self.connect()) as db, db:
            if self._get(db, revision_id) is None:
                raise ValueError("Unknown revision.")
            db.execute("INSERT INTO checks VALUES (?,?,?)", (check_id, revision_id, encoded))
        return {"id": check_id, **record}

    def get_check(self, check_id: str) -> dict | None:
        with closing(self.connect()) as db:
            row = db.execute("SELECT record_json FROM checks WHERE id=?", (check_id,)).fetchone()
        if row is None:
            return None
        record = json.loads(row[0])
        if _hash(record) != check_id:
            raise HistoryIntegrityError("Recorded check does not match its immutable hash.")
        return {"id": check_id, **record}

    def checks(self, revision_id: str) -> list[dict]:
        with closing(self.connect()) as db:
            rows = db.execute("SELECT id FROM checks WHERE revision_id=? ORDER BY rowid", (revision_id,)).fetchall()
        return [self.get_check(row[0]) for row in rows]

    def lineage(self, revision_id: str) -> list[dict]:
        """Newest first; every link is checked before serving recovery evidence."""
        result = []
        with closing(self.connect()) as db:
            while revision_id:
                revision = self._get(db, revision_id)
                if revision is None:
                    raise HistoryIntegrityError("Model history has a missing parent.")
                result.append(revision)
                revision_id = revision["parent_id"]
        return result

    def recovery_context(self, failure_id: str) -> dict:
        failure = self.get_check(failure_id)
        if failure is None or failure["status"] != "failed":
            raise ValueError("Recovery requires an actual recorded failed check.")
        lineage = self.lineage(failure["revision_id"])
        paths = set(failure["implicated_paths"])
        # Include recorded upstream inputs of an implicated feature. Report the
        # relation explicitly; a related operation is not automatically culpable.
        dependencies = lineage[0]["dependencies"]
        while paths:
            expanded = paths | {dependency for feature, inputs in dependencies.items()
                                if any(_related(feature, path) for path in paths) for dependency in inputs}
            if expanded == paths:
                break
            paths = expanded
        candidates = []
        for revision in lineage:
            related = [change for change in revision["changes"]
                       if not paths or change["path"] == "/" or any(_related(change["path"], path) for path in paths)]
            if related:
                candidates.append({"revision_id": revision["id"], "parent_id": revision["parent_id"],
                                   "action": revision["action"], "changes": related,
                                   "evidence": revision["evidence"]})
        return {"failure": failure, "candidates": candidates,
                "dependencies": dependencies, "related_paths": sorted(paths),
                "checkpoints": [{"id": item["id"], "plan_hash": item["plan_hash"]} for item in lineage],
                "causation": "Candidates identify producing/modifying operations; the last writer is not a proven root cause.",
                "required_rerun": list(STAGES[STAGES.index(failure["stage"]):])}

    def branch_recovery(self, failure_id: str, checkpoint_id: str, *, branch: str, diagnosis: str,
                        evidence=()) -> dict:
        """Fork a retained checkpoint; applying a repair and new checks remain explicit."""
        self._branch(branch)
        if not diagnosis.strip() or len(diagnosis) > 8000:
            raise ValueError("Recovery requires a bounded diagnosis linked to the observed failure.")
        context = self.recovery_context(failure_id)
        if checkpoint_id not in {item["id"] for item in context["checkpoints"]}:
            raise ValueError("The recovery checkpoint must be an ancestor of the failed revision.")
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM branches WHERE name=?", (branch,)).fetchone():
                raise HistoryConflict("Recovery branch already exists; use a new branch name.")
            checkpoint = self._get(db, checkpoint_id)
            failed_revision = self._get(db, context["failure"]["revision_id"])
            restored_changes = plan_diff(GeometryPlan.model_validate(failed_revision["plan"]),
                                         GeometryPlan.model_validate(checkpoint["plan"]))
            required = set(context["required_rerun"]) | set(affected_stages(restored_changes))
            recovery = {"failure_id": failure_id, "failed_revision_id": context["failure"]["revision_id"],
                        "checkpoint_id": checkpoint_id, "diagnosis": diagnosis,
                        "required_rerun": [stage for stage in STAGES if stage in required]}
            result = self._insert(db, GeometryPlan.model_validate(checkpoint["plan"]), checkpoint_id,
                                  {"type": "recovery_checkpoint"}, evidence, [], recovery)
            # A restored checkpoint is not a new successful simulation. Require
            # fresh checks on this branch even if its content existed previously.
            db.execute("INSERT INTO branches VALUES (?,?)", (branch, result["id"]))
            return result
