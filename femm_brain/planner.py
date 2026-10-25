"""Conservative, local selection of executable source-backed studies.

This is a small catalog grammar, not a general natural-language CAD model.
Requests outside that grammar stay reviewable instead of silently losing data.
"""
from __future__ import annotations

import math
import re


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_NUMBER_RE = re.compile(_NUMBER)
_UNITS = {
    "mm": ("length", 0.001), "millimeter": ("length", 0.001), "millimeters": ("length", 0.001),
    "millimetre": ("length", 0.001), "millimetres": ("length", 0.001),
    "cm": ("length", 0.01), "centimeter": ("length", 0.01), "centimeters": ("length", 0.01),
    "m": ("length", 1.0), "meter": ("length", 1.0), "meters": ("length", 1.0),
    "in": ("length", 0.0254), "inch": ("length", 0.0254), "inches": ("length", 0.0254),
    '"': ("length", 0.0254), "a": ("current", 1.0), "amp": ("current", 1.0),
    "amps": ("current", 1.0), "ampere": ("current", 1.0), "amperes": ("current", 1.0),
    "t": ("field", 1.0), "tesla": ("field", 1.0), "mt": ("field", 0.001),
    "ut": ("field", 0.000001), "µt": ("field", 0.000001), "μt": ("field", 0.000001),
    "turn": ("count", 1.0), "turns": ("count", 1.0), "count": ("count", 1.0),
    "1": ("scale", 1.0),
    "ratio": ("scale", 1.0),
}
_UNIT_PATTERN = "(?:" + "|".join(re.escape(x) for x in sorted(_UNITS, key=len, reverse=True)) + r")(?![\w])"
_ALIASES = {
    "inner_radius": ["inner radius", "inside radius", "ri"],
    "outer_radius": ["outer radius", "outside radius", "ro"],
    "length": ["axial length", "coil length", "length", "z"],
    "turns": ["number of turns", "turn count", "turns", "turn", "n"],
    "current": ["circuit current", "coil current", "current"],
    "radius": ["domain radius", "interior radius", "radius"],
    "mesh_scale": ["mesh scale", "mesh multiplier"],
    "boundary_scale": ["boundary scale", "boundary multiplier"],
    "applied_field_scale": ["applied field scale", "field scale"],
    "target_field": ["target field", "applied field", "field strength", "flux density", "field"],
    "wire_radius": ["wire radius", "conductor radius"],
    "center_distance": ["center distance", "centre distance", "wire spacing", "center spacing", "centre spacing"],
    "domain_radius": ["domain radius", "outer domain radius", "exterior radius"],
    "current_scale": ["current scale", "current multiplier", "excitation scale", "excitation multiplier"],
}
_CASE_ALIASES = {
    "aircore-short": ["aircore short", "short aircore", "aircore square", "square aircore coil", "square cross section"],
    "aircore-long": ["aircore long", "long aircore", "aircore tall", "tall thin coil", "tall thin"],
    "aircore-flat": ["aircore flat", "flat aircore", "aircore pancake", "short fat coil", "short fat"],
    "uniform-planar": ["uniform planar", "planar uniform field", "uniform 2d", "2d uniform field", "uniformfield planar"],
    "uniform-axisymmetric": ["uniform axisymmetric", "axisymmetric uniform field", "uniform axial field", "uniformfield axisymmetric"],
    "parallel-wires": ["parallel wires", "parallel wire force", "comsol parallel wires", "comsol parallel wire force"],
    "gapped-ei-core": ["gapped EI core", "EI core", "EI inductor", "EI actuator", "gapped iron core"],
    "radial-bearing": ["radial bearing", "magnetic bearing", "eight pole magnetic bearing", "8-pole magnetic bearing", "nonlinear bearing"],
    "ansys-vm270": ["VM270", "TEAM23", "TEAM 23", "ANSYS VM270", "SmCo magnet and coil"],
}
_FAMILIES = {
    "aircore": ["air core", "aircore", "air cored", "solenoid"],
    "uniform": ["uniform field", "uniform magnetic field", "uniformfield"],
}
_FILLER = set("a an the please can could you i want would like to create build generate model solve run calculate compute reproduce replicate benchmark example study using use with of and for at in is are as this that from source official femm documented baseline published geometry magnetic magnetics dc magnetostatic magnetostatics axisymmetric axi planar two dimensional 2d air core cored coil solenoid inductance uniform field applied axial turn turns amperes amps ampere amp tesla millimeters millimeter millimetres millimetre centimeters centimeter meters meter inches inch count mesh boundary scale multiplier number inner outer inside outside radius axial length circuit current target strength flux density tall thin short fat square cross section pancake n ri ro z".split())
_FILLER.update("comsol parallel wire wires repulsive force conductor center centre distance spacing domain exterior opposite signed equal currents meter per length".split())
_FILLER.update("gapped ei iron core inductor actuator radial bearing eight pole nonlinear steel magnetic energy force current excitation multiplier ansys team samarium cobalt smco permanent magnet coil".split())


def _phrase(value):
    value = str(value).casefold().replace("_", " ")
    # Hyphenated words/case IDs are aliases; numeric signs and exponents matter.
    return re.sub(r"(?<=[a-z])-(?=[a-z])", " ", value)


def _find_phrase(text, phrase):
    words = _phrase(phrase).split()
    if not words:
        return []
    pattern = r"(?<![\w])" + r"\s+".join(re.escape(word) for word in words) + r"(?![\w])"
    return list(re.finditer(pattern, text))


def _catalog_cases(catalog, graph):
    if catalog is None:
        from .simulation import simulation_catalog
        catalog = simulation_catalog(graph=graph)
    if isinstance(catalog, dict):
        catalog = catalog.get("cases", catalog.get("simulations", []))
    if not isinstance(catalog, list):
        raise ValueError("The simulation catalog must contain a cases list.")
    return [dict(case) for case in catalog]


def _case_id(case):
    return case.get("id", case.get("case_id"))


def _schemas(case):
    values = case.get("parameters", case.get("parameter_schema", []))
    if isinstance(values, dict):
        values = [{"name": name, **value} for name, value in values.items()]
    return {value["name"]: value for value in values}


def _evidence(case, graph):
    nodes = list(case.get("node_ids", case.get("knowledge_node_ids", [])))
    if graph is not None:
        example_id = case.get("example_id", case.get("source_example_id"))
        if not example_id:
            identifier = _case_id(case) or ""
            example_id = "aircore" if identifier.startswith("aircore-") else {
                "uniform-planar": "uniformfield-planar", "uniform-axisymmetric": "uniformfield-axisymmetric"
            }.get(identifier)
        example = next((item for item in graph.get("examples", []) if item["id"] == example_id), None)
        if example:
            # Shared stage nodes are navigation containers, not execution closure.
            nodes.extend(example.get("node_ids", []))
    return list(dict.fromkeys(nodes))


def _blank(status, message, **values):
    return {"status": status, "selected_case_id": None, "parameters": {}, "parsed_values": [],
            "candidates": [], "node_ids": [], "missing_inputs": [], "messages": [message],
            "llm_calls": 0, **values}


def _convert(value, supplied_unit, spec):
    native = str(spec.get("unit", "1")).casefold()
    native = {"turn": "count", "turns": "count", "dimensionless": "1"}.get(native, native)
    supplied = supplied_unit.casefold() if supplied_unit else None
    if native not in _UNITS:
        raise ValueError("This parameter's catalog unit is not supported by the local parser.")
    kind, native_factor = _UNITS[native]
    if supplied is None:
        if kind not in {"count", "scale"}:
            raise ValueError("A unit is required for " + spec["name"] + ".")
        supplied = native
    if supplied not in _UNITS or _UNITS[supplied][0] != kind:
        raise ValueError("The supplied unit does not match " + spec["name"] + ".")
    converted = value * _UNITS[supplied][1] / native_factor
    if not math.isfinite(converted):
        raise ValueError("The value must be finite.")
    if spec.get("integer") and not math.isclose(converted, round(converted), abs_tol=1e-9):
        raise ValueError(spec["name"] + " must be an integer.")
    if converted < spec.get("minimum", -math.inf) or converted > spec.get("maximum", math.inf):
        raise ValueError(spec["name"] + " is outside the executable catalog range.")
    return int(round(converted)) if spec.get("integer") else converted


def plan_request(text, graph=None, *, catalog=None):
    """Select a known study and parse explicit overrides without an API call.

    ``parameters`` uses the selected simulation's native units. ``parsed_values``
    retains the input and conversion evidence. Source defaults remain owned by
    the explicit selected catalog case and are not guessed from free-form text.
    """
    if not isinstance(text, str) or not text.strip():
        return _blank("needs_input", "Name a catalog study or describe an air-core coil or uniform-field study.")
    if len(text) > 16000:
        return _blank("unsupported", "The local catalog parser accepts requests up to 16000 characters.")
    normalized = _phrase(text)
    cases = _catalog_cases(catalog, graph)
    indexed = {_case_id(case): case for case in cases}
    exact, spans, family_hits = set(), [], set()
    for identifier, case in indexed.items():
        aliases = [identifier, *_CASE_ALIASES.get(identifier, []), *case.get("aliases", [])]
        aliases += [alias.replace("aircore", "air core") for alias in aliases]
        for alias in aliases:
            for match in _find_phrase(normalized, alias):
                exact.add(identifier)
                spans.append(match.span())
    for family, aliases in _FAMILIES.items():
        for alias in aliases:
            matches = _find_phrase(normalized, alias)
            if matches:
                family_hits.add(family)
                spans.extend(match.span() for match in matches)
    selected = set(exact)
    if not selected:
        selected = {identifier for identifier in indexed if any(identifier.startswith(family + "-") for family in family_hits)}
    if not selected:
        return _blank("unsupported", "No executable catalog study matches this request. Select one of the documented executable catalog profiles.")
    if len(family_hits) > 1 or (len(exact) > 1):
        return _blank("ambiguous", "Select one study per request.", candidates=[{"id": identifier, "title": indexed[identifier].get("title", identifier)} for identifier in sorted(selected)])
    # Catalog support, rather than FEMM's global capabilities, defines this route.
    unsupported = re.search(r"\b(ac|harmonic|frequency|hz|ipm|motor|torque|eddy|transient|ellip\w*|oval|polygon|c\s+shaped)\b", normalized)
    if unsupported:
        return _blank("unsupported", "The requested " + unsupported.group(0) + " behavior is outside these executable catalog profiles.")
    if re.search(r"\b(nonlinear|saturat\w*)\b", normalized) and selected != {"radial-bearing"}:
        return _blank("unsupported", "Nonlinear steel is supported by the radial-bearing profile; the selected profile retains its source material law.")
    if re.search(r"\b(ferromagnetic|steel|iron)\b", normalized) and not selected <= {"gapped-ei-core", "radial-bearing"}:
        return _blank("unsupported", "Iron or steel is outside the selected executable profile's material definition.")
    if re.search(r"\b(permanent\s+magnet\w*|magnet|samarium|smco)\b", normalized) and selected != {"ansys-vm270"}:
        return _blank("unsupported", "Permanent magnets require an executable permanent-magnet profile.")
    if all(identifier.startswith("aircore-") for identifier in selected) and re.search(r"\bplanar\b", normalized):
        return _blank("unsupported", "The catalog air-core coil profiles are axisymmetric. A planar coil requires a different executable study.")
    if selected == {"parallel-wires"} and re.search(r"\b(axisymmetric|axi|axial)\b", normalized):
        return _blank("unsupported", "The parallel-wire transfer is a planar model of straight wires with explicit out-of-plane depth.")
    if selected == {"parallel-wires"} and re.search(r"\b(same\s+direction|equal\s+currents|attract\w*)\b", normalized):
        return _blank("unsupported", "This transfer profile fixes opposite current directions and repulsive force. Same-direction currents require a separate profile.")
    if not selected <= {"parallel-wires", "gapped-ei-core", "radial-bearing", "ansys-vm270"} and re.search(r"\bforce\b", normalized):
        return _blank("unsupported", "Force is not evaluated by the selected executable catalog profile.")
    if re.search(r"\binductance\b", normalized) and not selected <= {"aircore-short", "aircore-long", "aircore-flat", "gapped-ei-core"}:
        return _blank("unsupported", "Inductance is not evaluated by the selected executable catalog profile.")
    if re.search(r"\benergy\b", normalized) and selected != {"gapped-ei-core"}:
        return _blank("unsupported", "An energy comparison is not evaluated by the selected executable catalog profile.")
    if selected <= {"gapped-ei-core", "radial-bearing"} and re.search(r"\b(axisymmetric|axi|axial)\b", normalized):
        return _blank("unsupported", "The selected source geometry is planar; an axisymmetric study needs a different executable profile.")
    if selected == {"ansys-vm270"} and re.search(r"\bplanar\b", normalized):
        return _blank("unsupported", "The VM270 magnet and coil are axisymmetric.")
    if all(identifier.startswith("uniform-") for identifier in selected):
        if re.search(r"\b(axisymmetric|axi|axial)\b", normalized):
            selected &= {"uniform-axisymmetric"}
        if re.search(r"\b(planar|2d|two dimensional)\b", normalized):
            selected &= {"uniform-planar"}
        if not selected:
            return _blank("ambiguous", "The coordinate system conflicts with the selected uniform-field profile.")
        spans.extend(match.span() for match in re.finditer(r"\b2d\b", normalized))
    schemas = {}
    for identifier in sorted(selected):
        schemas.update(_schemas(indexed[identifier]))
    if all(identifier.startswith("uniform-") for identifier in selected):
        schemas["target_field"] = {"name": "target_field", "unit": "T"}
    matches = []
    for name, spec in schemas.items():
        aliases = [name, *_ALIASES.get(name, [])]
        for alias in set(aliases):
            phrase = r"\s+".join(re.escape(word) for word in _phrase(alias).split())
            prefix = r"(?<![\w])" + phrase + r"(?![\w])\s*(?:=|:|of|is)?\s*(" + _NUMBER + r")\s*(" + _UNIT_PATTERN + r")?"
            for match in re.finditer(prefix, normalized):
                value_end = match.end(2) if match.group(2) is not None else match.end(1)
                if value_end < len(normalized) and (normalized[value_end].isalnum() or normalized[value_end] == "_"):
                    continue
                matches.append((match.start(), match.end(), name, match.group(1), match.group(2), spec))
            if name in {"turns", "current"}:
                suffix_units = r"turns?" if name == "turns" else r"(?:a|amps?|amperes?)"
                suffix = r"(?<![\w.])(" + _NUMBER + r")\s*(" + suffix_units + r")(?![\w])"
                for match in re.finditer(suffix, normalized):
                    matches.append((match.start(), match.end(), name, match.group(1), match.group(2), spec))
    consumed = []
    parameters, parsed, errors = {}, [], []
    for start, end, name, number, unit, spec in sorted(matches, key=lambda item: (-(item[1]-item[0]), item[0])):
        if any(start < b and end > a for a, b in consumed):
            continue
        consumed.append((start, end))
        try:
            value = _convert(float(number), unit, spec)
            if name in parameters and not math.isclose(parameters[name], value, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Conflicting values were supplied for " + name + ".")
            parameters[name] = value
            parsed.append({"name": name, "value": value, "unit": spec.get("unit", "1"),
                           "input_value": float(number), "input_unit": unit,
                           "origin": "user", "input_text": text[start:end]})
        except ValueError as exc:
            errors.append(str(exc))
    all_spans = spans + consumed
    remaining = list(normalized)
    for start, end in all_spans:
        remaining[start:end] = " " * (end-start)
    remainder = "".join(remaining)
    numbers = [match.group(0) for match in _NUMBER_RE.finditer(remainder)]
    if numbers:
        errors.append("Unrecognized numeric specification(s): " + ", ".join(numbers) + ". Use named catalog parameters and explicit units.")
    words = set(re.findall(r"[a-z_]+", remainder))
    unknown = sorted(words - _FILLER - set(_UNITS))
    if unknown:
        errors.append("Unrecognized request detail(s): " + ", ".join(unknown) + ". The local parser only accepts the documented catalog grammar.")
    candidates = [{"id": identifier, "title": indexed[identifier].get("title", identifier),
                   "example_id": indexed[identifier].get("example_id", indexed[identifier].get("source_example_id")),
                   "node_ids": _evidence(indexed[identifier], graph)} for identifier in sorted(selected)]
    if errors:
        return _blank("needs_input", errors[0], parameters=parameters, parsed_values=parsed,
                      candidates=candidates, messages=errors, missing_inputs=["Resolve unrecognized or conflicting request details."])
    if len(selected) > 1 and all(identifier.startswith("aircore-") for identifier in selected):
        dimensions = {"inner_radius", "outer_radius", "length"}
        if dimensions <= parameters.keys():
            matching = [identifier for identifier in selected if all(math.isclose(parameters[name], _schemas(indexed[identifier])[name]["default"], rel_tol=1e-10, abs_tol=1e-12) for name in dimensions)]
            if len(matching) == 1:
                selected = {matching[0]}
    if len(selected) != 1:
        return _blank("ambiguous", "Choose a named source profile. Generic study names do not select a hidden baseline.",
                      parameters=parameters, parsed_values=parsed, candidates=candidates,
                      missing_inputs=["Select one candidate case ID."])
    identifier = next(iter(selected))
    if identifier == "gapped-ei-core":
        current = parameters.get("current", _schemas(indexed[identifier])["current"]["default"])
        if abs(current) < 1e-6:
            return _blank("needs_input", "Use a nonzero EI current to define linkage/current inductance.", candidates=candidates, parsed_values=parsed)
    if identifier.startswith("aircore-"):
        schema = _schemas(indexed[identifier])
        inner = parameters.get("inner_radius", schema["inner_radius"]["default"])
        outer = parameters.get("outer_radius", schema["outer_radius"]["default"])
        if outer <= inner:
            return _blank("needs_input", "outer_radius must be greater than inner_radius.", candidates=candidates, parsed_values=parsed)
    if identifier == "parallel-wires":
        schema = _schemas(indexed[identifier])
        values = {name: parameters.get(name, spec["default"]) for name, spec in schema.items()}
        if 2 * values["wire_radius"] >= values["center_distance"]:
            return _blank("needs_input", "The circular wire sections must not intersect: require 2*wire_radius < center_distance.", candidates=candidates, parsed_values=parsed)
        if values["domain_radius"] < 2 * (values["center_distance"] / 2 + values["wire_radius"]):
            return _blank("needs_input", "The exterior radius is too close to the physical wire sections for this transfer profile.", candidates=candidates, parsed_values=parsed)
    if "target_field" in parameters:
        if "applied_field_scale" in parameters:
            return _blank("needs_input", "Specify either target field or applied_field_scale, so there is one field definition.", candidates=candidates, parsed_values=parsed)
        reference = 0.5 if identifier == "uniform-planar" else 2 / 3
        scale = parameters.pop("target_field") / reference
        try:
            parameters["applied_field_scale"] = _convert(scale, None, _schemas(indexed[identifier])["applied_field_scale"])
        except ValueError as exc:
            return _blank("needs_input", str(exc), candidates=candidates, parsed_values=parsed)
        parsed.append({"name": "applied_field_scale", "value": parameters["applied_field_scale"],
                       "unit": "1", "origin": "derived", "expression": "target_field / source_analytic_field",
                       "reference_field_T": reference})
    return _blank("ready", "Selected the documented catalog profile. Only explicit supported overrides will be applied.",
                  selected_case_id=identifier, parameters=parameters, parsed_values=sorted(parsed, key=lambda item: item["name"]),
                  candidates=[candidate for candidate in candidates if candidate["id"] == identifier],
                  node_ids=_evidence(indexed[identifier], graph))
