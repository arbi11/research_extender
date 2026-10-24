"""Provider contracts stay strict even when a provider returns invalid output."""
import copy
import unittest
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app import structured


class Child(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=2, pattern=r"^[A-Z]")
    current: float = Field(gt=0)
    note: str | None = None
    enabled: bool = True


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    child: Child
    names: list[str] = Field(default_factory=list)


def good_candidate():
    return {"child": {"label": "Coil", "current": 1.0, "note": None, "enabled": None}, "names": None}


class StructuredTests(unittest.TestCase):
    def test_nested_schema_is_closed_required_and_nullable_defaults(self):
        original = Contract.model_json_schema()
        result = structured.strict_json_schema(Contract)
        self.assertEqual(result["required"], ["child", "names"])
        self.assertFalse(result["additionalProperties"])
        child = result["$defs"]["Child"]
        self.assertFalse(child["additionalProperties"])
        self.assertEqual(set(child["required"]), set(child["properties"]))
        self.assertEqual(child["properties"]["enabled"]["anyOf"][1], {"type": "null"})
        self.assertNotIn("pattern", child["properties"]["label"])
        self.assertNotIn("exclusiveMinimum", child["properties"]["current"])
        self.assertNotIn("default", child["properties"]["note"])
        self.assertEqual(Contract.model_json_schema(), original)

    def test_request_uses_enforced_schema_and_no_implicit_budget_or_temperature(self):
        messages = [{"role": "user", "content": "Fill the contract"}]
        result = structured.request_payload("example/model", messages, response_model=Contract)
        self.assertEqual(result["response_format"]["type"], "json_schema")
        self.assertTrue(result["response_format"]["json_schema"]["strict"])
        self.assertEqual(result["provider"], {"require_parameters": True})
        self.assertNotIn("max_tokens", result)
        self.assertNotIn("max_completion_tokens", result)
        self.assertNotIn("temperature", result)
        result["messages"][0]["content"] = "Changed"
        self.assertEqual(messages[0]["content"], "Fill the contract")

    def test_explicit_budget_and_provider_routing_are_preserved(self):
        routing = {"order": ["provider"], "allow_fallbacks": False}
        result = structured.request_payload("example/model", [], response_model=Contract,
            temperature=0, max_tokens=12000, provider=routing)
        self.assertEqual(result["max_tokens"], 12000)
        self.assertEqual(result["temperature"], 0)
        self.assertEqual(result["provider"]["order"], ["provider"])
        self.assertNotIn("require_parameters", routing)
        with self.assertRaisesRegex(ValueError, "enforces"):
            structured.request_payload("example/model", [], response_model=Contract,
                                       provider={"require_parameters": False})

    def test_validated_null_placeholders_use_defaults_but_preserve_nullable_values(self):
        candidate = good_candidate()
        original = copy.deepcopy(candidate)
        restored = structured.validate_response(candidate, Contract)
        self.assertEqual(restored, {"child": {"label": "Coil", "current": 1.0, "note": None}})
        parsed = Contract.model_validate(restored)
        self.assertTrue(parsed.child.enabled)
        self.assertEqual(parsed.names, [])
        self.assertIsNone(parsed.child.note)
        self.assertEqual(candidate, original)

    def test_required_and_defaulted_wire_keys_cannot_be_silently_filled(self):
        for path in ("child", "names", "note", "enabled", "current"):
            candidate = good_candidate()
            del (candidate if path in ("child", "names") else candidate["child"])[path]
            with self.subTest(path=path), self.assertRaises(structured.WireValidationError) as caught:
                structured.validate_response(candidate, Contract)
            self.assertTrue(any(issue["code"] == "wire_required" for issue in caught.exception.issues))

    def test_extra_fields_and_wrong_types_are_retained_as_diagnostics(self):
        candidate = good_candidate()
        candidate["extra"] = "bad"
        candidate["child"]["extra"] = "also bad"
        candidate["child"]["current"] = True
        with self.assertRaises(structured.WireValidationError) as caught:
            structured.validate_response(candidate, Contract)
        paths = {issue["path"] for issue in caught.exception.issues}
        self.assertEqual(paths, {"extra", "child.extra", "child.current"})

    def test_original_bounds_remain_local_validation_requirements(self):
        candidate = good_candidate()
        candidate["child"].update(label="x", current=-1)
        normalized = structured.validate_response(candidate, Contract)
        with self.assertRaises(ValidationError):
            Contract.model_validate(normalized)

    def test_required_null_is_not_repaired(self):
        candidate = good_candidate()
        candidate["child"]["current"] = None
        with self.assertRaises(structured.WireValidationError):
            structured.validate_response(candidate, Contract)
        self.assertIsNone(structured.restore_defaults(candidate, Contract)["child"]["current"])

    def test_arrays_and_union_discriminators_are_validated(self):
        class Circle(BaseModel):
            kind: Literal["circle"]
            radius: float
        class Polygon(BaseModel):
            kind: Literal["polygon"]
            vertices: list[list[float]]
        class Shape(BaseModel):
            shape: Annotated[Circle | Polygon, Field(discriminator="kind")]
        schema = structured.strict_json_schema(Shape)
        self.assertIn("anyOf", schema["properties"]["shape"])
        valid = {"shape": {"kind": "polygon", "vertices": [[0, 0], [1, 0], [0, 1]]}}
        self.assertEqual(structured.validate_response(valid, Shape), valid)
        for bad in ({"shape": {"kind": "circle", "radius": "large"}},
                    {"shape": {"kind": "triangle", "vertices": []}}):
            with self.assertRaises(structured.WireValidationError):
                structured.validate_response(bad, Shape)

    def test_nonfinite_and_boolean_numeric_values_are_rejected(self):
        for current in (True, float("inf"), float("nan"), 10 ** 1000):
            candidate = good_candidate()
            candidate["child"]["current"] = current
            with self.subTest(current=type(current)), self.assertRaises(structured.WireValidationError):
                structured.validate_response(candidate, Contract)

    def test_untyped_fields_and_open_maps_require_an_explicit_contract(self):
        class Untyped(BaseModel):
            value: Any = Field(description="Any value")
        class OpenMap(BaseModel):
            values: dict[str, float]
        for contract in (Untyped, OpenMap):
            with self.subTest(contract=contract), self.assertRaises(ValueError):
                structured.strict_json_schema(contract)

    def test_schema_metadata_and_name_are_stable(self):
        self.assertEqual(structured.schema_hash(Contract), structured.schema_hash(Contract))
        self.assertEqual(len(structured.schema_hash(Contract)), 64)
        self.assertEqual(structured.contract_metadata(Contract)["schema_version"], 1)
        with self.assertRaises(ValueError):
            structured.response_format(Contract, name="not a contract")

    def test_enum_sibling_of_nullable_union_is_not_ignored(self):
        class Choice(BaseModel):
            unit: str | None = Field(json_schema_extra={"enum": ["cm", None]})
        self.assertEqual(structured.validate_response({"unit": "cm"}, Choice), {"unit": "cm"})
        with self.assertRaises(structured.WireValidationError):
            structured.validate_response({"unit": "invented"}, Choice)


if __name__ == "__main__":
    unittest.main()
