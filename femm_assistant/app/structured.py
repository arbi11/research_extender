"""Provider-enforced JSON contracts, independent of any particular model.

The wire schema uses the common strict-output subset. Pydantic remains the
authority for bounds, units and cross-field checks after the response arrives.
No token budget or model fallback is introduced here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any

from pydantic import BaseModel


SCHEMA_VERSION = 1


def _nullable(schema: dict) -> bool:
    return (schema.get("type") == "null" or "null" in schema.get("type", [])
            or any(_nullable(item) for item in schema.get("anyOf", [])))


def strict_json_schema(response_model: type[BaseModel]) -> dict:
    """Compile a closed, required-field schema without provider-specific bounds.

    Defaulted fields are required on the wire and may use null to request their
    declared local default. ``restore_defaults`` handles only those placeholders;
    it never fills in a missing required response field.
    """
    source = response_model.model_json_schema(mode="validation")

    def visit(node):
        if not isinstance(node, dict):
            return node
        kept = {}
        for key in ("type", "$ref", "description", "enum"):
            if key in node:
                kept[key] = copy.deepcopy(node[key])
        if "const" in node:
            kept["enum"] = [node["const"]]
        for key in ("$defs",):
            if key in node:
                kept[key] = {name: visit(value) for name, value in node[key].items()}
        union = node.get("anyOf", node.get("oneOf"))
        if union is not None:
            kept["anyOf"] = [visit(value) for value in union]
        if node.get("type") == "object" or "properties" in node:
            if node.get("additionalProperties") not in (None, False):
                raise ValueError("Strict outputs need named object fields; represent open mappings as typed key/value arrays.")
            kept["type"] = "object"
            required = set(node.get("required", []))
            properties = {}
            for name, value in node.get("properties", {}).items():
                converted = visit(value)
                if name not in required and not _nullable(converted):
                    converted = {"anyOf": [converted, {"type": "null"}]}
                properties[name] = converted
            kept.update(properties=properties, required=list(properties), additionalProperties=False)
        if node.get("type") == "array":
            if "prefixItems" in node:
                # Tuple length is still checked by the local Pydantic model.
                alternatives = [visit(item) for item in node["prefixItems"]]
                kept["items"] = alternatives[0] if len(alternatives) == 1 else {"anyOf": alternatives}
            else:
                kept["items"] = visit(node.get("items", {}))
        if not any(key in kept for key in ("type", "$ref", "anyOf", "enum")):
            raise ValueError("Strict outputs cannot enforce an untyped Any field; provide a typed response contract.")
        return kept

    result = visit(source)
    if result.get("type") != "object" or "anyOf" in result:
        raise ValueError("A strict response contract must have an object at its root.")
    return result


def response_format(response_model: type[BaseModel], *, name: str | None = None) -> dict:
    contract_name = name or response_model.__name__
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", contract_name):
        raise ValueError("A structured response name needs 1 to 64 letters, digits, underscores or hyphens.")
    return {"type": "json_schema", "json_schema": {"name": contract_name,
        "strict": True, "schema": strict_json_schema(response_model)}}


def schema_hash(response_model: type[BaseModel]) -> str:
    encoded = json.dumps(strict_json_schema(response_model), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def contract_metadata(response_model: type[BaseModel]) -> dict:
    return {"schema_version": SCHEMA_VERSION, "schema_name": response_model.__name__,
            "schema_sha256": schema_hash(response_model)}


def request_payload(model: str, messages: list[dict], *, response_model: type[BaseModel],
                    name: str | None = None, temperature: float | None = None,
                    provider: dict | None = None, **parameters) -> dict:
    """Build an OpenRouter request that cannot silently drop the JSON schema.

    An explicit caller-supplied token budget is supported, but there is no
    implicit max_tokens/max_completion_tokens limit.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Choose an explicit model for structured output.")
    routing = copy.deepcopy(provider or {})
    if routing.get("require_parameters") is False:
        raise ValueError("Structured output requires a provider that enforces the supplied parameters.")
    routing["require_parameters"] = True
    if "response_format" in parameters or "messages" in parameters or "model" in parameters:
        raise ValueError("The response contract, messages and model must use their named arguments.")
    result = {key: value for key, value in parameters.items() if value is not None}
    result.update(model=model, messages=copy.deepcopy(messages),
                  response_format=response_format(response_model, name=name), provider=routing)
    if temperature is not None:
        result["temperature"] = temperature
    return result


def restore_defaults(value: Any, response_model: type[BaseModel]) -> Any:
    """Resolve explicit null default placeholders without repairing bad output.

    Missing keys stay missing; required nulls stay null. The returned value must
    still pass the original Pydantic model and all semantic validators.
    """
    schema = response_model.model_json_schema(mode="validation")
    definitions = schema.get("$defs", {})

    def resolved(node):
        if "$ref" in node and node["$ref"].startswith("#/$defs/"):
            return definitions.get(node["$ref"].split("/")[-1], node)
        return node

    def matches_shape(item, node):
        node = resolved(node)
        kind = node.get("type")
        if "const" in node:
            return item == node["const"]
        if kind == "object" and isinstance(item, dict):
            return all(key not in item or item[key] == prop["const"]
                       for key, prop in node.get("properties", {}).items() if "const" in prop)
        return ((kind == "null" and item is None) or (kind == "array" and isinstance(item, list))
                or (kind == "string" and isinstance(item, str))
                or (kind in ("number", "integer") and isinstance(item, (int, float)) and not isinstance(item, bool))
                or (kind == "boolean" and isinstance(item, bool)))

    def walk(item, node):
        node = resolved(node)
        union = node.get("anyOf", node.get("oneOf", []))
        if union:
            choices = [alternative for alternative in union if matches_shape(item, alternative)]
            if len(choices) == 1:
                return walk(item, choices[0])
        if isinstance(item, dict) and (node.get("type") == "object" or "properties" in node):
            properties, required = node.get("properties", {}), set(node.get("required", []))
            result = {}
            for key, entry in item.items():
                prop = properties.get(key)
                if prop is None:
                    result[key] = copy.deepcopy(entry)
                elif entry is None and key not in required and not _nullable(prop):
                    continue
                else:
                    result[key] = walk(entry, prop)
            return result
        if isinstance(item, list) and node.get("type") == "array":
            return [walk(entry, node.get("prefixItems", [])[index]
                         if index < len(node.get("prefixItems", [])) else node.get("items", {}))
                    for index, entry in enumerate(item)]
        return copy.deepcopy(item)

    return walk(value, schema)


class WireValidationError(ValueError):
    def __init__(self, issues):
        self.issues = tuple(issues)
        super().__init__("The provider response did not satisfy the enforced JSON contract.")


def validate_response(candidate: Any, response_model: type[BaseModel]) -> Any:
    """Check the entire emitted strict subset before restoring default placeholders.

    This validator deliberately implements only the subset emitted above, rather
    than pretending to be a general JSON Schema implementation. Pydantic and
    scientific validators still check the original, richer local contract.
    """
    schema = strict_json_schema(response_model)
    definitions = schema.get("$defs", {})

    def problem(path, code, message):
        return {"path": path, "code": code, "message": message}

    def check(item, node, path):
        if "enum" in node and not any(
                item == choice and isinstance(item, bool) == isinstance(choice, bool)
                for choice in node["enum"]):
            return [problem(path, "wire_enum", "Value is outside the declared enumeration.")]
        if "$ref" in node:
            prefix = "#/$defs/"
            reference = node["$ref"]
            if not reference.startswith(prefix) or reference[len(prefix):] not in definitions:
                raise ValueError("The structured contract contains an unsupported schema reference.")
            return check(item, definitions[reference[len(prefix):]], path)
        if "anyOf" in node:
            alternatives = [check(item, alternative, path) for alternative in node["anyOf"]]
            if any(not errors for errors in alternatives):
                return []
            # Keep details for the structurally closest alternative rather than
            # losing missing fields behind a generic union error.
            closest = min(alternatives, key=len, default=[])
            return [problem(path, "wire_any_of", "Value matches none of the declared alternatives."), *closest]
        kind = node.get("type")
        try:
            numeric = isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item)
        except OverflowError:
            numeric = False
        valid_type = {"null": item is None, "boolean": isinstance(item, bool),
            "string": isinstance(item, str), "number": numeric,
            "integer": numeric and float(item).is_integer(),
            "object": isinstance(item, dict), "array": isinstance(item, list)}
        if kind is not None and not valid_type.get(kind, False):
            return [problem(path, "wire_type", f"Expected JSON {kind}.")]
        issues = []
        if kind == "object":
            properties = node.get("properties", {})
            for key in node.get("required", []):
                if key not in item:
                    issues.append(problem(f"{path}.{key}" if path else key, "wire_required", "Required response field is missing."))
            for key, value in item.items():
                child_path = f"{path}.{key}" if path else str(key)
                if key not in properties:
                    issues.append(problem(child_path, "wire_additional_property", "Undeclared response field is forbidden."))
                else:
                    issues.extend(check(value, properties[key], child_path))
        if kind == "array":
            for index, entry in enumerate(item):
                issues.extend(check(entry, node["items"], f"{path}.{index}" if path else str(index)))
        return issues

    issues = check(candidate, schema, "")
    if issues:
        raise WireValidationError(issues)
    return restore_defaults(candidate, response_model)
