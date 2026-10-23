"""Decision evidence and orchestration tests; all provider traffic is mocked."""
import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from app import decisions
from app.plan import wire_example
from app.planner import PlannerError, interpret_plan


def choice(operation="create", probability=0.98, confidence=0.97):
    probabilities = {name: (1 - probability) / 2 for name in ("create", "modify", "clarify")}
    probabilities[operation] = probability
    return {"type": "choice", "choice": operation, "confidence": confidence, "probabilities": probabilities}


def evidence(operation="create", support=0.99, sufficiency=0.99):
    return {"answers": {"route": choice(operation), "support": {"type": "noul", "noul": support},
                        "sufficiency": {"type": "noul", "noul": sufficiency}},
            "model": "typesafe/jev-1.14-20261014", "id": "fixture-not-a-real-provider-call"}


def completion(spec, explanation="Created the requested conductor."):
    return {"choices": [{"message": {"content": json.dumps({"spec": spec,
                                "explanation": explanation, "questions": []})}}]}


class DecisionTests(unittest.IsolatedAsyncioTestCase):
    async def check_preflight(self, response, *, current=None, threshold=0.90):
        requests = []

        def handler(request):
            requests.append(request)
            return response(request) if callable(response) else httpx.Response(200, json=response)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await decisions.preflight(client, "fixture-private-key", decisions.DEFAULT_BASE_URL,
                decisions.DEFAULT_MODEL, threshold, "Create a round 1 A conductor, depth 20 mm and radius 1 mm.", current)
        return result, requests

    async def test_official_http_contract_and_typed_evidence(self):
        result, requests = await self.check_preflight(evidence())
        self.assertTrue(result.allowed)
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(str(request.url), "https://openrouter.ai/api/alpha/decisions")
        self.assertEqual(request.headers["authorization"], "Bearer fixture-private-key")
        body = json.loads(request.content)
        self.assertEqual(set(body), {"model", "state", "questions"})
        self.assertEqual(body["model"], "typesafe/jev-1.14")
        self.assertEqual(body["questions"]["route"]["type"], "choice")
        self.assertEqual(body["questions"]["support"]["type"], "noul")
        self.assertNotIn("messages", body)
        self.assertEqual(result.trace[0].confidence, 0.97)
        self.assertIsNone(result.trace[1].confidence)
        self.assertAlmostEqual(result.trace[1].probability, 0.99)
        self.assertNotIn("fixture-private-key", result.model_dump_json())
        self.assertNotIn(body["state"]["request"], result.model_dump_json())
        self.assertLessEqual(request.extensions["timeout"]["read"], decisions.CALL_TIMEOUT_SECONDS)

    async def test_support_and_sufficiency_require_positive_evidence(self):
        for support, sufficient in [(0.01, 0.99), (0.99, 0.01), (0.89, 0.99), (0.99, 0.5)]:
            with self.subTest(support=support, sufficient=sufficient):
                result, _ = await self.check_preflight(evidence(support=support, sufficiency=sufficient))
                self.assertFalse(result.allowed)
                self.assertTrue(result.questions)

    async def test_configured_threshold_changes_policy_without_changing_evidence(self):
        response = evidence(support=0.87)
        strict, _ = await self.check_preflight(response)
        lower, _ = await self.check_preflight(response, threshold=0.85)
        self.assertFalse(strict.allowed)
        self.assertTrue(lower.allowed)
        self.assertEqual(strict.trace[1].probability, lower.trace[1].probability)
        self.assertEqual(lower.trace[1].threshold, 0.85)

    async def test_route_cannot_switch_creation_and_modification(self):
        result, requests = await self.check_preflight(evidence("create"), current=wire_example())
        self.assertFalse(result.allowed)
        self.assertFalse(result.trace[0].passed)
        state = json.loads(requests[0].content)["state"]
        self.assertEqual(state["operation"], "modify")
        self.assertEqual(state["current_plan"], wire_example().model_dump(mode="json"))

    async def test_route_requires_winner_probability_and_real_confidence(self):
        for probability, confidence in [(0.60, 0.99), (0.99, 0.40)]:
            packet = evidence()
            packet["answers"]["route"] = choice(probability=probability, confidence=confidence)
            result, _ = await self.check_preflight(packet)
            self.assertFalse(result.allowed)

    async def test_malformed_or_missing_evidence_never_gets_default_probability(self):
        fixtures = []
        missing = evidence()
        del missing["answers"]["sufficiency"]
        fixtures.append(missing)
        for bad in [None, "0.99", True, -0.01, 1.01, float("nan"), float("inf")]:
            packet = evidence()
            packet["answers"]["support"]["noul"] = bad
            fixtures.append(packet)
        missing_confidence = evidence()
        del missing_confidence["answers"]["route"]["confidence"]
        fixtures.append(missing_confidence)
        contradictory = evidence()
        contradictory["answers"]["route"]["probabilities"] = {"create": 0.1, "modify": 0.8, "clarify": 0.1}
        fixtures.append(contradictory)
        invalid_sum = evidence()
        invalid_sum["answers"]["route"]["probabilities"] = {"create": 0.98, "modify": 0.8, "clarify": 0.1}
        fixtures.append(invalid_sum)
        wrong_keys = evidence()
        wrong_keys["answers"]["route"]["probabilities"] = {"create": 0.98, "modify": 0.02}
        fixtures.append(wrong_keys)
        for packet in fixtures:
            with self.subTest(packet=repr(packet)):
                with self.assertRaises(decisions.DecisionError) as caught:
                    await self.check_preflight(packet)
                self.assertEqual(caught.exception.status_code, 502)

    async def test_provider_errors_are_safe_and_actionable(self):
        for status, phrase in [(401, "API key"), (402, "credit"), (429, "rate limiting"),
                               (404, "Decisions API"), (503, "unavailable")]:
            with self.subTest(status=status):
                with self.assertRaises(decisions.DecisionError) as caught:
                    await self.check_preflight(lambda request: httpx.Response(status,
                        json={"error": "fixture-private-key provider-secret"}))
                self.assertIn(phrase, caught.exception.message)
                self.assertNotIn("fixture-private-key", caught.exception.message)
                self.assertNotIn("provider-secret", caught.exception.message)

    async def test_body_and_network_failures_do_not_bypass_decisions(self):
        def timeout(request):
            raise httpx.ReadTimeout("fixture-private-key", request=request)

        fixtures = [lambda request: httpx.Response(200, content=b"x" * (decisions.MAX_RESPONSE_BYTES + 1)),
                    lambda request: httpx.Response(200, content=b"not JSON"), timeout]
        for fixture in fixtures:
            with self.assertRaises(decisions.DecisionError) as caught:
                await self.check_preflight(fixture)
            self.assertNotIn("fixture-private-key", caught.exception.message)

    async def test_fidelity_sees_original_and_candidate_but_returns_only_trace(self):
        original = wire_example()
        modified = original.model_copy(deep=True)
        modified.circuits[0].current_a = 2
        received = []

        def handler(request):
            received.append(json.loads(request.content))
            return httpx.Response(200, json={"answers": {"faithfulness": {"type": "noul", "noul": 0.05}}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await decisions.fidelity(client, "fixture-private-key", decisions.DEFAULT_BASE_URL,
                decisions.DEFAULT_MODEL, 0.90, "Keep the current at 1 A.", original, modified, "Changed the current.")
        self.assertFalse(result.allowed)
        self.assertEqual(received[0]["state"]["current_plan"]["circuits"][0]["current_a"], 1)
        self.assertEqual(received[0]["state"]["candidate_plan"]["circuits"][0]["current_a"], 2)
        self.assertEqual(result.trace[0].stage, "fidelity")
        self.assertTrue(result.questions)

    async def test_invalid_candidate_never_reaches_fidelity_provider(self):
        bad = wire_example().model_copy(deep=True)
        bad.regions[0].material = "undefined_material"
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"answers": {"faithfulness": {"type": "noul", "noul": 1}}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(ValidationError):
                await decisions.fidelity(client, "fixture-private-key", decisions.DEFAULT_BASE_URL,
                    decisions.DEFAULT_MODEL, 0.90, "Build this model.", None, bad, "A bad candidate.")
        self.assertFalse(requests)


class OrchestrationTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, handler, current=None, **kwargs):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch("app.planner.httpx.AsyncClient", return_value=client):
            return await interpret_plan("Create a round 1 A conductor with radius 1 mm and depth 20 mm.", current,
                "fixture-private-key", "https://provider.test/api/v1", "fixture-generator",
                decision_base_url="https://provider.test/api/alpha", **kwargs)

    async def test_preflight_clarification_makes_zero_generations(self):
        calls = []

        def handler(request):
            calls.append(request.url.path)
            return httpx.Response(200, json=evidence(support=0.10))

        result = await self.call(handler)
        self.assertIsNone(result.spec)
        self.assertTrue(result.questions)
        self.assertEqual(result.attempts, 0)
        self.assertEqual(calls, ["/api/alpha/decisions"])

    async def test_maximum_four_calls_including_validation_repair(self):
        calls = []
        invalid = wire_example().model_dump()
        invalid["regions"][0]["material"] = "undefined_material"
        generations = 0

        def handler(request):
            nonlocal generations
            calls.append(request.url.path)
            packet = json.loads(request.content)
            if request.url.path.endswith("/decisions"):
                if "route" in packet["questions"]:
                    return httpx.Response(200, json=evidence())
                return httpx.Response(200, json={"answers": {"faithfulness": {"type": "noul", "noul": 0.99}}})
            generations += 1
            return httpx.Response(200, json=completion(invalid if generations == 1 else wire_example().model_dump()))

        result = await self.call(handler)
        self.assertIsNotNone(result.spec)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(calls, ["/api/alpha/decisions", "/api/v1/chat/completions",
                                 "/api/v1/chat/completions", "/api/alpha/decisions"])
        self.assertEqual(len(result.decision_trace), 4)
        self.assertTrue(all(item.passed for item in result.decision_trace))

    async def test_valid_but_unfaithful_geometry_is_not_returned_as_approvable(self):
        calls = []

        def handler(request):
            calls.append(request.url.path)
            packet = json.loads(request.content)
            if request.url.path.endswith("/decisions"):
                if "route" in packet["questions"]:
                    return httpx.Response(200, json=evidence())
                return httpx.Response(200, json={"answers": {"faithfulness": {"type": "noul", "noul": 0.45}}})
            return httpx.Response(200, json=completion(wire_example().model_dump()))

        result = await self.call(handler)
        self.assertIsNone(result.spec)
        self.assertTrue(result.questions)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(len(calls), 3)
        self.assertFalse(result.decision_trace[-1].passed)

    async def test_invalid_plans_never_reach_semantic_fidelity(self):
        calls = []
        invalid = wire_example().model_dump()
        invalid["regions"][0]["material"] = "undefined_material"

        def handler(request):
            calls.append(request.url.path)
            if request.url.path.endswith("/decisions"):
                return httpx.Response(200, json=evidence())
            return httpx.Response(200, json=completion(invalid))

        with self.assertRaises(PlannerError):
            await self.call(handler)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls.count("/api/alpha/decisions"), 1)

    async def test_unavailable_decision_provider_has_no_generator_fallback(self):
        calls = []

        def handler(request):
            calls.append(request.url.path)
            return httpx.Response(503, text="fixture-private-key upstream-secret")

        with self.assertRaises(PlannerError) as caught:
            await self.call(handler)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("fixture-private-key", caught.exception.message)

    async def test_total_timeout_covers_decisions_as_well_as_generation(self):
        calls = []

        async def handler(request):
            calls.append(request.url.path)
            await asyncio.sleep(0.05)
            return httpx.Response(200, json=evidence())

        with patch("app.planner.TOTAL_TIMEOUT_SECONDS", 0.01):
            with self.assertRaises(PlannerError) as caught:
                await self.call(handler)
        self.assertEqual(caught.exception.status_code, 504)
        self.assertEqual(calls, ["/api/alpha/decisions"])

    async def test_invalid_threshold_fails_before_any_provider_call(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(200, json=evidence())

        for threshold in [0.5, 1.01, float("nan"), True]:
            with self.subTest(threshold=threshold):
                with self.assertRaises(PlannerError) as caught:
                    await self.call(handler, decision_threshold=threshold)
                self.assertEqual(caught.exception.status_code, 503)
        self.assertFalse(calls)


if __name__ == "__main__":
    unittest.main()
