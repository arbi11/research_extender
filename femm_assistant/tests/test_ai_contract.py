"""Legacy preset requests obey the same provider contract without paid calls."""
import copy
import json
import unittest
from unittest.mock import patch

import httpx

from app.ai import interpret
from app.model import CCoreSpec


class PresetContractTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, candidate, *, finish_reason="stop", requests=None):
        def handler(request):
            if requests is not None:
                requests.append(request)
            return httpx.Response(200, json={"choices": [{"finish_reason": finish_reason,
                "message": {"content": json.dumps(candidate)}}]})
        client_type = httpx.AsyncClient
        with patch("app.ai.httpx.AsyncClient", side_effect=lambda **kwargs:
                   client_type(transport=httpx.MockTransport(handler), **kwargs)):
            return await interpret("Use 2 A and preserve the other parameters.", CCoreSpec(),
                                   "mock-key", "https://provider.test/v1", "test/model")

    async def test_complete_preset_contract_preserves_fields_without_model_caps(self):
        current = CCoreSpec()
        candidate = {"spec": current.model_copy(update={"current_a": 2}).model_dump(),
                     "explanation": "Only current changes.", "questions": []}
        requests = []
        result = await self.call(candidate, requests=requests)
        self.assertEqual(result.spec.current_a, 2)
        self.assertEqual(result.spec.model_dump(exclude={"current_a"}), current.model_dump(exclude={"current_a"}))
        payload = json.loads(requests[0].content)
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertTrue(payload["provider"]["require_parameters"])
        for parameter in ("max_tokens", "max_completion_tokens", "temperature", "reasoning", "reasoning_effort"):
            self.assertNotIn(parameter, payload)
        self.assertIsNone(requests[0].extensions["timeout"]["read"])

    async def test_missing_wire_fields_extra_fields_invalid_physics_and_truncation_fail(self):
        valid = {"spec": CCoreSpec().model_dump(), "explanation": "Unchanged.", "questions": []}
        missing = copy.deepcopy(valid)
        del missing["spec"]["current_a"]
        invalid = copy.deepcopy(valid)
        invalid["spec"]["limb_mm"] = 40
        for candidate, finish in ((missing, "stop"), ({**valid, "executable_code": "not allowed"}, "stop"),
                                  (invalid, "stop"), (valid, "length")):
            with self.subTest(finish=finish, candidate=candidate), self.assertRaises(ValueError):
                await self.call(candidate, finish_reason=finish)
