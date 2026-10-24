"""Bind a paper operating case to the exact model produced from its evidence.

This records provenance, not semantic approval. Native mesh, solver and frozen
target checks still determine execution outcomes. A new paper/case or physical
edit requires a fresh binding; an unrelated draft cannot inherit paper targets.
"""
from .history import EvidenceReference
from .targets import model_fingerprint


PREFIX = "paper-planning:"


class UnavailablePlanningError(ValueError):
    pass


def put_planning(store, spec, evidence_id, case_id, instructions, *, preserve_paper_excitation=False, excitation_check=None):
    # Validate the selected source case even when called outside the HTTP route.
    store.planning_context(evidence_id, case_id, instructions)
    directory = store.root / "planning"
    directory.mkdir(exist_ok=True)
    if directory.is_symlink():
        raise ValueError("Paper planning storage must be a regular directory.")
    record = {"evidence_id": evidence_id, "case_id": case_id,
              "model_sha256": model_fingerprint(spec), "instructions": instructions,
              "preserve_paper_excitation": preserve_paper_excitation,
              "excitation_check": excitation_check,
              "status": "source_bound_unverified",
              "comparison_note": "Compare against published conditions; user overrides are variations, not verified reproductions."}
    return {"planning_id": store._put("planning", record), **record}


def get_planning(store, planning_id):
    record = store._get("planning", planning_id)
    if record is None:
        raise UnavailablePlanningError("The paper model planning record is unavailable. Generate the selected case again.")
    if store.get_evidence(record["evidence_id"]) is None:
        raise ValueError("The paper planning evidence is unavailable.")
    return {"planning_id": planning_id, **record}


def binding_evidence(store, spec, planning_id, evidence_id, case_id, *, preserve_paper_excitation=False):
    record = get_planning(store, planning_id)
    if (record["evidence_id"] != evidence_id or record["case_id"] != case_id
            or record.get("preserve_paper_excitation", False) != preserve_paper_excitation
            or record["model_sha256"] != model_fingerprint(spec)):
        raise ValueError("This model does not match the selected paper planning record. Generate the selected case again.")
    return EvidenceReference(source="artifact", reference=PREFIX + planning_id,
                             sha256=planning_id, detail="Exact source-bound model; mesh, solver and target checks remain required.")


def revision_binding(store, revision):
    for reference in reversed(revision.get("evidence", [])):
        if reference.get("source") == "artifact" and reference.get("reference", "").startswith(PREFIX):
            identity = reference["reference"][len(PREFIX):]
            try:
                record = get_planning(store, identity)
            except UnavailablePlanningError:
                continue
            if reference.get("sha256") != identity or record["model_sha256"] != model_fingerprint(revision["plan"]):
                continue
            return {**{key: record[key] for key in ("planning_id", "evidence_id", "case_id")},
                    "preserve_paper_excitation": record.get("preserve_paper_excitation", False)}
    return {}


def validate_reserved_references(store, spec, evidence):
    """Reject reserved references before recording any immutable model state."""
    for value in evidence:
        reference = EvidenceReference.model_validate(value)
        if reference.source == "artifact" and reference.reference.startswith("assembly-planning:"):
            from .assembly_api import AssemblyStore
            if store is None:
                raise ValueError("Assembly storage is unavailable.")
            identity = reference.reference[len("assembly-planning:"):]
            if reference.sha256 != identity:
                raise ValueError("The supplied assembly reference has an invalid content hash.")
            AssemblyStore(store).evidence(identity, spec)
        if reference.source == "artifact" and reference.reference.startswith(PREFIX):
            if store is None:
                raise ValueError("Paper planning storage is unavailable.")
            identity = reference.reference[len(PREFIX):]
            record = get_planning(store, identity)
            if reference.sha256 != identity or record["model_sha256"] != model_fingerprint(spec):
                raise ValueError("The supplied paper planning reference does not match this model.")


def require_paper_binding(store, history, revision, evidence_id, case_id):
    binding = revision_binding(store, revision)
    if binding.get("evidence_id") != evidence_id or binding.get("case_id") != case_id:
        raise ValueError("The current model was not generated for this paper operating case. Generate its model before freezing linked targets.")
    return binding
