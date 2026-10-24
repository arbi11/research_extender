"""Frozen, evidence-linked numerical checks against native solver results.

Targets are data, never expressions or executable paper code. A contract binds
the targets and tolerances to one exact model revision before it is solved. The
caller must persist that contract with the queued job; a repair gets a new model
binding but retains the original target definitions.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SourceCitation(FrozenModel):
    """Page numbers are physical PDF pages, beginning at one."""

    source_id: str = Field(min_length=1, max_length=100)
    page: int | None = Field(default=None, ge=1, le=5000, strict=True)
    table: str | None = Field(default=None, min_length=1, max_length=100)
    figure: str | None = Field(default=None, min_length=1, max_length=100)
    section: str | None = Field(default=None, min_length=1, max_length=160)
    quote: str | None = Field(default=None, min_length=1, max_length=600)

    @model_validator(mode="after")
    def has_locator(self):
        if not any((self.page, self.table, self.figure, self.section)):
            raise ValueError("A citation requires a page, table, figure or section locator.")
        return self


# Explicit conversion factors avoid executable expressions and ambiguous units.
# A dimension match is required; e.g. magnetic flux and flux linkage differ.
_UNITS = {
    "1": ("dimensionless", 1.0), "turn": ("turns", 1.0), "turns": ("turns", 1.0),
    "m": ("length", 1.0), "cm": ("length", .01), "mm": ("length", .001),
    "in": ("length", .0254), "inch": ("length", .0254), "inches": ("length", .0254),
    "um": ("length", 1e-6), "µm": ("length", 1e-6), "μm": ("length", 1e-6),
    "A": ("current", 1.0), "mA": ("current", .001),
    "T": ("flux_density", 1.0), "mT": ("flux_density", .001),
    "uT": ("flux_density", 1e-6), "µT": ("flux_density", 1e-6), "μT": ("flux_density", 1e-6),
    "J": ("energy", 1.0), "mJ": ("energy", .001),
    "uJ": ("energy", 1e-6), "µJ": ("energy", 1e-6), "μJ": ("energy", 1e-6),
    "N": ("force", 1.0), "mN": ("force", .001),
    "H": ("inductance", 1.0), "mH": ("inductance", .001),
    "uH": ("inductance", 1e-6), "µH": ("inductance", 1e-6), "μH": ("inductance", 1e-6),
    "Wb-turn": ("flux_linkage", 1.0), "Wb turns": ("flux_linkage", 1.0),
    "mWb-turn": ("flux_linkage", .001), "uWb-turn": ("flux_linkage", 1e-6),
    "µWb-turn": ("flux_linkage", 1e-6), "μWb-turn": ("flux_linkage", 1e-6), "Wb": ("magnetic_flux", 1.0),
    "Hz": ("frequency", 1.0), "kHz": ("frequency", 1000.0),
    "S/m": ("conductivity", 1.0), "MS/m": ("conductivity", 1e6),
    "A/m2": ("current_density", 1.0), "A/m²": ("current_density", 1.0),
    "A/mm2": ("current_density", 1e6), "A/mm²": ("current_density", 1e6),
    "m2": ("area", 1.0), "m²": ("area", 1.0), "cm2": ("area", 1e-4),
    "cm²": ("area", 1e-4), "mm2": ("area", 1e-6), "mm²": ("area", 1e-6),
    "A/m": ("field_strength", 1.0), "kA/m": ("field_strength", 1000.0),
    "deg": ("angle", 1.0), "rad": ("angle", 180 / math.pi),
}


def convert_units(value: float, source_unit: str, destination_unit: str) -> float:
    """Convert recognized, dimensionally compatible units without guessing."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("A quantity must be a finite real number.")
    try:
        source_dimension, source_factor = _UNITS[source_unit]
        target_dimension, target_factor = _UNITS[destination_unit]
    except (KeyError, TypeError):
        raise ValueError("An explicit supported unit is required.") from None
    if source_dimension != target_dimension:
        raise ValueError("The quantity units describe different physical dimensions.")
    converted = value * source_factor / target_factor
    if not math.isfinite(converted):
        raise ValueError("The converted quantity is not finite.")
    return converted


def unit_dimension(unit: str) -> str:
    try:
        return _UNITS[unit][0]
    except (KeyError, TypeError):
        raise ValueError("An explicit supported unit is required.") from None


class Tolerance(FrozenModel):
    """Pass if |actual-target| <= absolute + relative*|target|.

    The absolute tolerance uses the target's original unit. A zero target needs
    a positive absolute tolerance; relative tolerance alone cannot relax it.
    """

    absolute: float = Field(default=0, ge=0, strict=True)
    relative: float = Field(default=0, ge=0, le=1, strict=True)

    @model_validator(mode="after")
    def nonzero(self):
        if self.absolute == 0 and self.relative == 0:
            raise ValueError("Set a positive absolute or relative tolerance before solving.")
        return self


class Observable(FrozenModel):
    quantity: Literal["energy", "force_x", "force_y", "force_z", "force_magnitude", "current",
                      "flux_linkage", "inductance", "bx", "by", "br", "bz", "b_magnitude"]
    circuit_id: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]{0,39}$")
    probe_index: int | None = Field(default=None, ge=0, le=15, strict=True)

    @model_validator(mode="after")
    def selectors(self):
        is_circuit = self.quantity in {"current", "flux_linkage", "inductance"}
        is_probe = self.quantity in {"bx", "by", "br", "bz", "b_magnitude"}
        if not is_circuit and self.circuit_id is not None:
            raise ValueError("Only circuit observables accept a circuit id.")
        if is_probe != (self.probe_index is not None):
            raise ValueError("Field observables require exactly one probe index.")
        return self


_OBSERVABLE_UNITS = {
    "energy": "J", "force_x": "N", "force_y": "N", "force_z": "N", "force_magnitude": "N",
    "current": "A", "flux_linkage": "Wb-turn", "inductance": "H",
    "bx": "T", "by": "T", "br": "T", "bz": "T", "b_magnitude": "T",
}


class NumericalTarget(FrozenModel):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    case_id: str = Field(default="baseline", pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,79}$")
    description: str = Field(min_length=1, max_length=600)
    observable: Observable
    value: float = Field(strict=True)
    unit: str = Field(min_length=1, max_length=24)
    tolerance: Tolerance
    reference_kind: Literal["reported_numerical", "analytical", "experimental"]
    evidence_kind: Literal["explicit", "derived", "assumption"] = "explicit"
    citations: tuple[SourceCitation, ...] = Field(default=(), max_length=12)
    derivation: str | None = Field(default=None, min_length=1, max_length=1000)
    conditions: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def evidence_and_units(self):
        convert_units(self.value, self.unit, _OBSERVABLE_UNITS[self.observable.quantity])
        if self.evidence_kind != "assumption" and not self.citations:
            raise ValueError("Published or derived targets require source citations.")
        if self.evidence_kind == "derived" and not self.derivation:
            raise ValueError("A derived target requires a stated derivation.")
        if self.evidence_kind == "assumption":
            raise ValueError("An assumed number cannot be a reproduction target.")
        if self.value == 0 and self.tolerance.absolute == 0:
            raise ValueError("A zero-valued target requires an absolute tolerance.")
        return self


def _canonical(value) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def model_fingerprint(plan: BaseModel | Mapping) -> str:
    return hashlib.sha256(_canonical(plan).encode()).hexdigest()


def target_fingerprint(targets: tuple[NumericalTarget, ...]) -> str:
    return hashlib.sha256(_canonical([target.model_dump(mode="json") for target in targets]).encode()).hexdigest()


class TargetContract(FrozenModel):
    version: Literal[1] = 1
    case_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,79}$")
    model_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    targets_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    targets: tuple[NumericalTarget, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def unchanged_targets(self):
        if len({target.id for target in self.targets}) != len(self.targets):
            raise ValueError("Target ids must be unique within a case.")
        if self.targets_sha256 != target_fingerprint(self.targets):
            raise ValueError("The frozen target definitions or tolerances changed.")
        if any(target.case_id != self.case_id for target in self.targets):
            raise ValueError("Targets from different operating cases require separate model contracts.")
        return self


def freeze_targets(targets, plan: BaseModel | Mapping, *, case_id: str) -> TargetContract:
    parsed = tuple(item if isinstance(item, NumericalTarget) else NumericalTarget.model_validate(item)
                   for item in targets)
    _validate_plan_observables(parsed, plan)
    return TargetContract(case_id=case_id, model_sha256=model_fingerprint(plan),
                          targets_sha256=target_fingerprint(parsed), targets=parsed)


def rebind_targets(contract: TargetContract, plan: BaseModel | Mapping) -> TargetContract:
    """Bind a retry revision to the same target definitions and tolerances."""
    return freeze_targets(contract.targets, plan, case_id=contract.case_id)


def contract_fingerprint(contract: TargetContract) -> str:
    return hashlib.sha256(_canonical(contract).encode()).hexdigest()


def _read_quantity(results: Mapping, name: str, unit: str) -> float:
    item = results.get(name)
    if not isinstance(item, Mapping) or set(item) != {"value", "unit"}:
        raise ValueError("Native result is missing or has an invalid quantity envelope: " + name)
    return convert_units(item["value"], item["unit"], unit)


def _validate_plan_observables(targets, plan):
    spec = plan.model_dump(mode="json") if isinstance(plan, BaseModel) else dict(plan)
    axisymmetric = spec.get("physics") == "axisymmetric_dc_magnetics"
    wrong_coordinates = {"force_x", "force_y", "bx", "by"} if axisymmetric else {"force_z", "br", "bz"}
    for target in targets:
        if target.observable.quantity in wrong_coordinates:
            raise ValueError("The target uses components from a different coordinate system.")
        if target.observable.quantity in {"force_x", "force_y", "force_z", "force_magnitude"}:
            if spec.get("kind") == "magnetostatic_plan" and not spec.get("force_region"):
                raise ValueError("A force target requires a force_region in the model.")
        if target.observable.probe_index is not None:
            if target.observable.probe_index >= len(spec.get("probes", [])):
                raise ValueError("The target refers to a probe absent from the model.")
        if target.observable.quantity in {"inductance", "current", "flux_linkage"}:
            circuit = target.observable.circuit_id
            if spec.get("kind") == "magnetostatic_plan":
                if circuit is None or circuit not in {item["id"] for item in spec.get("circuits", [])}:
                    raise ValueError("A general-plan circuit target requires an existing circuit id.")


def _observe(observable: Observable, results: Mapping, *, axisymmetric=False) -> float:
    kind = observable.quantity
    if kind in {"energy", "force_x", "force_y", "force_z"}:
        return _read_quantity(results, kind, _OBSERVABLE_UNITS[kind])
    if kind == "force_magnitude":
        if axisymmetric:
            return abs(_read_quantity(results, "force_z", "N"))
        return math.hypot(_read_quantity(results, "force_x", "N"),
                          _read_quantity(results, "force_y", "N"))
    if kind in {"bx", "by", "br", "bz", "b_magnitude"}:
        index = observable.probe_index
        if kind == "b_magnitude":
            first, second = ("br", "bz") if axisymmetric else ("bx", "by")
            return math.hypot(_read_quantity(results, f"{first}_{index}", "T"),
                              _read_quantity(results, f"{second}_{index}", "T"))
        return _read_quantity(results, f"{kind}_{index}", "T")
    suffix = "_" + observable.circuit_id if observable.circuit_id is not None else ""
    if kind == "inductance":
        current = _read_quantity(results, "current" + suffix, "A")
        if current == 0:
            raise ValueError("Flux-linkage/current inductance is undefined at zero current.")
        return _read_quantity(results, "flux_linkage" + suffix, "Wb-turn") / current
    return _read_quantity(results, kind + suffix, _OBSERVABLE_UNITS[kind])


def evaluate_targets(contract: TargetContract, results: Mapping, *, plan: BaseModel | Mapping,
                     expected_contract_sha256: str | None = None) -> dict:
    """Evaluate server-owned native results, preserving all frozen target data.

    `results` is the output of jobs.read_results, never a provider completion.
    Jobs should retain `contract_fingerprint(contract)` at queue time and pass it
    as expected_contract_sha256. Model edits require a fresh solve and binding.
    """
    # Revalidate even model_copy(update=...) instances, which Pydantic can create
    # without validating frozen/hash invariants.
    contract = TargetContract.model_validate(contract.model_dump(mode="json"))
    if model_fingerprint(plan) != contract.model_sha256:
        raise ValueError("The numerical targets belong to a different model revision.")
    _validate_plan_observables(contract.targets, plan)
    spec = plan.model_dump(mode="json") if isinstance(plan, BaseModel) else dict(plan)
    axisymmetric = spec.get("physics") == "axisymmetric_dc_magnetics"
    fingerprint = contract_fingerprint(contract)
    if expected_contract_sha256 is not None and fingerprint != expected_contract_sha256:
        raise ValueError("The target contract differs from the one frozen before solving.")
    checks = []
    for target in contract.targets:
        entry = {"target_id": target.id, "reference_kind": target.reference_kind,
                 "target": target.value, "unit": target.unit,
                 "tolerance": target.tolerance.model_dump(), "conditions": target.conditions,
                 "citations": [citation.model_dump(mode="json") for citation in target.citations]}
        try:
            actual_si = _observe(target.observable, results, axisymmetric=axisymmetric)
            actual = convert_units(actual_si, _OBSERVABLE_UNITS[target.observable.quantity], target.unit)
            error = abs(actual - target.value)
            limit = target.tolerance.absolute + target.tolerance.relative * abs(target.value)
            entry.update(status="passed" if error <= limit + 1e-14 * max(limit, abs(target.value), 1e-30) else "failed",
                         actual=actual, absolute_error=error,
                         relative_error=error / abs(target.value) if target.value != 0 else None,
                         permitted_error=limit)
        except (ValueError, TypeError, OverflowError) as error:
            entry.update(status="unavailable", actual=None, error=str(error))
        checks.append(entry)
    status = "passed" if all(item["status"] == "passed" for item in checks) else (
        "failed" if any(item["status"] == "failed" for item in checks) else "incomplete")
    return {"status": status, "case_id": contract.case_id, "model_sha256": contract.model_sha256,
            "targets_sha256": contract.targets_sha256, "contract_sha256": fingerprint, "checks": checks}


def evaluate_target_file(contract: TargetContract, folder: Path, *, plan,
                         expected_contract_sha256: str | None = None) -> dict:
    """Read the actual native CSV and nonempty .ans through the worker parser."""
    from .jobs import read_results
    from .plan import GeometryPlan
    from .plan_generate import required_results

    expected = required_results(plan) if isinstance(plan, GeometryPlan) else None
    return evaluate_targets(contract, read_results(folder, expected), plan=plan,
                            expected_contract_sha256=expected_contract_sha256)
