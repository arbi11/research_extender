"""Opt-in arithmetic constraints from verified source inputs, never targets.

For one uniform winding cross-section, J*area = abs(N)*I. This mechanical
relation can catch an incorrect conversion from a cited current density to a
circuit current. It does not establish semantic fidelity or infer winding signs.
Ambiguous cases report that the contract is inapplicable instead of guessing.
"""
from __future__ import annotations

import math

from pydantic import ValidationError

from .plan import GeometryPlan, area, contains


RELATIVE_TOLERANCE = 1e-9
DENSITY_TO_A_PER_MM2 = {"A/mm2": 1.0, "A/m2": 1e-6}


def _skip(reason: str, *, status="skipped") -> dict:
    return {"contract": "uniform_single_winding_current_density", "applicable": False,
            "status": status, "reason": reason, "message": reason}


def validate_excitation(plan: GeometryPlan, evidence_context: dict | None) -> dict:
    """Validate a source-preserving uniform-winding excitation when unambiguous.

    The context must come from the verified paper-evidence pipeline. This helper
    additionally requires a cited explicit/derived density claim. Negative J or
    negative circuit current leaves the directional convention unresolved and is
    deliberately not converted to an absolute-value assertion.

    A mismatch raises the same ValidationError handled by the planner's bounded
    failed-call repair. Passing or inapplicable contracts return a report without
    changing the model, paper evidence, targets or tolerances.
    """
    if not isinstance(evidence_context, dict) or evidence_context.get("preserve_paper_excitation") is not True:
        return _skip("Paper excitation preservation was not explicitly enabled.", status="not_requested")
    parameters = evidence_context.get("parameters", [])
    if not isinstance(parameters, (list, tuple)):
        return _skip("No bounded verified parameter inventory is available.")
    densities = [parameter for parameter in parameters if isinstance(parameter, dict)
                 and parameter.get("unit") in DENSITY_TO_A_PER_MM2
                 and type(parameter.get("value")) in (int, float)
                 and math.isfinite(parameter["value"])]
    if len(densities) != 1:
        return _skip("Exactly one finite numeric current-density parameter is required; ambiguous or absent source densities are not combined.")
    density = densities[0]
    citations = density.get("citations")
    if (density.get("evidence_kind") not in {"explicit", "derived"} or not isinstance(citations, (list, tuple))
            or not citations or any(not isinstance(citation, dict) or not citation.get("source_id") for citation in citations)):
        return _skip("The current density must be an independently verified, cited explicit or derived paper parameter.")
    if density["value"] < 0:
        return _skip("A negative current density requires an explicit winding/current sign convention; no absolute-value convention is inferred.")
    if not isinstance(plan, GeometryPlan):
        plan = GeometryPlan.model_validate(plan)
    windings = [region for region in plan.regions if region.circuit is not None and region.turns != 0]
    if len(windings) != 1 or len(plan.circuits) != 1 or windings[0].circuit != plan.circuits[0].id:
        return _skip("The arithmetic contract requires exactly one wound region assigned to its sole circuit; multiple winding cross-sections are not combined.")
    winding, circuit = windings[0], plan.circuits[0]
    if circuit.current_a < 0 and density["value"] != 0:
        return _skip("A negative circuit current requires an explicit winding/current sign convention; no absolute-value convention is inferred.")
    if any(other.id != winding.id and contains(winding.shape, other.label) for other in plan.regions):
        return _skip("Nested contours occupy part of the winding cross-section; the uniform filled area is not inferred from its outer contour.")
    winding_area = area(winding.shape)
    current_density = float(density["value"]) * DENSITY_TO_A_PER_MM2[density["unit"]]
    expected = current_density * winding_area / abs(winding.turns)
    if not math.isfinite(expected):
        return _skip("The cited density and winding area do not produce a finite bounded circuit current.")
    source_ids = list(dict.fromkeys(citation["source_id"] for citation in citations))
    report = {"contract": "uniform_single_winding_current_density", "applicable": True,
              "status": "passed", "path": f"/circuits/{circuit.id}/current_a",
              "parameter_id": density.get("id"), "source_ids": source_ids,
              "source_value": density["value"], "source_unit": density["unit"],
              "current_density_a_per_mm2": current_density, "winding_id": winding.id,
              "winding_area_mm2": winding_area, "absolute_turns": abs(winding.turns),
              "expected_current_a": expected, "actual_current_a": circuit.current_a,
              "relative_tolerance": RELATIVE_TOLERANCE,
              "derivation": "I = J * winding_area_mm2 / abs(turns)",
              "direction_policy": "The constraint preserves a nonnegative circuit-current magnitude; it does not infer winding direction.",
              "message": f"Source excitation arithmetic passed: {current_density:.12g} A/mm2 * {winding_area:.12g} mm2 / {abs(winding.turns)} turns = {expected:.12g} A."}
    if math.isclose(circuit.current_a, expected, rel_tol=RELATIVE_TOLERANCE, abs_tol=0):
        return report
    source = ", ".join(str(value) for value in source_ids)
    reason = (f"Paper excitation arithmetic at {report['path']}: source current density "
              f"'{density.get('id', 'unnamed')}' ({source}) is {density['value']!r} {density['unit']} "
              f"= {current_density:.12g} A/mm2. Winding '{winding.id}' has area {winding_area:.12g} mm2 "
              f"and abs(turns)={abs(winding.turns)}, so I=J*area/abs(turns) requires "
              f"{expected:.12g} A; the supplied circuit current is {circuit.current_a:.12g} A. "
              "Preserve the cited excitation inputs; do not change a target to repair this mismatch.")
    raise ValidationError.from_exception_data("PaperExcitationContract", [{
        "type": "value_error", "loc": ("circuits", circuit.id, "current_a"), "input": None,
        "ctx": {"error": ValueError(reason)}}])
