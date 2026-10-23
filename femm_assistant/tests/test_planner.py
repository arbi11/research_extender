"""Provider integration tests use HTTP mocks; no paid API key is required."""
import json
import unittest
from unittest.mock import patch

import httpx

from app.plan import wire_example
from app.planner import MAX_PROVIDER_BYTES, PlannerError, interpret_plan


def completion(content):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


class PlannerTests(unittest.IsolatedAsyncioTestCase):
    async def call_with(self, handler, prompt="Create the described round conductor model.", current=None):
        def routed_handler(request):
            if request.url.path.endswith("/decisions"):
                payload = json.loads(request.content)
                answers = {name: {"type": "noul", "noul": 1.0} for name in payload["questions"]}
                if "route" in answers:
                    operation = payload["state"]["operation"]
                    answers["route"] = {"type": "choice", "choice": operation, "confidence": 1.0,
                        "probabilities": {name: 1.0 if name == operation else 0.0 for name in ("create", "modify", "clarify")}}
                return httpx.Response(200, json={"answers": answers})
            return handler(request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(routed_handler))
        with patch("app.planner.httpx.AsyncClient", return_value=client):
            return await interpret_plan(prompt, current, "private-test-key", "https://provider.test/api/v1", "test/json-model")

    async def test_creates_new_geometry_with_typed_plan_and_no_execution(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=completion(json.dumps({"spec": wire_example().model_dump(),
                                "explanation": "Built the requested conductor in a circular air domain.", "questions": []})))

        result = await self.call_with(handler)
        self.assertEqual(result.spec.regions[0].shape.type, "circle")
        self.assertEqual(result.attempts, 1)
        payload = json.loads(requests[0].content)
        user = json.loads(payload["messages"][1]["content"])
        self.assertEqual(user["operation"], "create")
        self.assertIsNone(user["current_plan"])
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertNotIn("attempts", json.loads(payload["messages"][0]["content"].split("Response JSON schema:\n")[1])["properties"])

    async def test_modify_receives_current_plan_without_mutating_it(self):
        original = wire_example()
        before = original.model_dump_json()
        received = []

        def handler(request):
            payload = json.loads(request.content)
            received.append(json.loads(payload["messages"][1]["content"]))
            plan = original.model_dump()
            plan["circuits"][0]["current_a"] = 2
            return httpx.Response(200, json=completion(json.dumps({"spec": plan,
                                 "explanation": "Changed the current to 2 A.", "questions": []})))

        result = await self.call_with(handler, "Set the current to 2 A and preserve the geometry.", original)
        self.assertEqual(received[0]["operation"], "modify")
        self.assertEqual(received[0]["current_plan"], original.model_dump())
        self.assertEqual(result.spec.circuits[0].current_a, 2)
        self.assertEqual(before, original.model_dump_json())

    async def test_invalid_geometry_gets_one_validation_guided_repair(self):
        calls = []
        valid = wire_example().model_dump()
        invalid = wire_example().model_dump()
        invalid["regions"][0]["material"] = "undefined_material"

        def handler(request):
            calls.append(json.loads(request.content))
            plan = invalid if len(calls) == 1 else valid
            return httpx.Response(200, json=completion(json.dumps({"spec": plan,
                                 "explanation": "Defined the requested material assignments.", "questions": []})))

        result = await self.call_with(handler)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(calls[1]["messages"]), 4)
        self.assertIn("validation", calls[1]["messages"][-1]["content"])

    async def test_unsupported_or_ambiguous_request_returns_questions(self):
        def handler(request):
            return httpx.Response(200, json=completion(json.dumps({"spec": None,
                "explanation": "This harness currently supports planar DC magnetics.",
                "questions": ["Can you use a planar DC cross-section instead of a 3D AC model?"]})))

        result = await self.call_with(handler, "Build a full 3D AC motor with nonlinear steel.")
        self.assertIsNone(result.spec)
        self.assertTrue(result.questions)
        self.assertEqual(result.attempts, 1)

    async def test_invalid_model_output_stops_after_two_calls(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(200, json=completion('{"spec":null,"explanation":"Nothing to ask.","questions":[]}'))

        with self.assertRaises(PlannerError) as caught:
            await self.call_with(handler)
        self.assertEqual(len(calls), 2)
        self.assertEqual(caught.exception.status_code, 502)
        self.assertIn("one repair", caught.exception.message)

    async def test_provider_errors_are_actionable_without_echoing_response_secrets(self):
        for status, phrase in [(401, "API key"), (402, "credit"), (429, "rate limiting"), (503, "unavailable")]:
            with self.subTest(status=status):
                def handler(request):
                    return httpx.Response(status, json={"error": "private-test-key upstream-secret"})
                with self.assertRaises(PlannerError) as caught:
                    await self.call_with(handler)
                self.assertIn(phrase, caught.exception.message)
                self.assertNotIn("private-test-key", caught.exception.message)
                self.assertNotIn("upstream-secret", caught.exception.message)

    async def test_oversized_provider_body_is_rejected_before_model_parsing(self):
        def handler(request):
            return httpx.Response(200, content=b"x" * (MAX_PROVIDER_BYTES + 1))
        with self.assertRaises(PlannerError) as caught:
            await self.call_with(handler)
        self.assertIn("too much data", caught.exception.message)

    async def test_network_timeout_has_safe_message(self):
        def handler(request):
            raise httpx.ReadTimeout("private-test-key", request=request)
        with self.assertRaises(PlannerError) as caught:
            await self.call_with(handler)
        self.assertEqual(caught.exception.status_code, 504)
        self.assertNotIn("private-test-key", caught.exception.message)


if __name__ == "__main__":
    unittest.main()
