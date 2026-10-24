"""Fixed-function routing and failed-call diagnostics; provider traffic is mocked."""
import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

from app import decisions
from app.plan import wire_example
from app.model import CCoreSpec
from app.planner import PlannerError, interpret_plan


FUNCTIONS = ("create_plan", "modify_plan", "clarify")


def choice(function="create_plan", probability=0.98, confidence=0.97):
    probabilities = {name: (1 - probability) / 2 for name in FUNCTIONS}
    probabilities[function] = probability
    return {"type": "choice", "choice": function, "confidence": confidence, "probabilities": probabilities}


def evidence(function="create_plan", probability=0.98, confidence=0.97):
    return {"answers": {"route": choice(function, probability, confidence)},
            "model": "typesafe/jev-1.14-20261014", "id": "fixture-not-a-real-provider-call"}


def diagnostic(probability=.8):
    return {"answers": {"arguments_match": {"type": "noul", "noul": probability}}}


def completion(spec):
    return {"choices": [{"message": {"content": json.dumps({"spec": spec,
        "explanation": "Created the requested conductor.", "questions": []})}}]}


class DecisionTests(unittest.IsolatedAsyncioTestCase):
    async def check_route(self, response, *, current=None, prompt="Create a round 1 A conductor, depth 20 mm and radius 1 mm."):
        requests = []

        def handler(request):
            requests.append(request)
            return response(request) if callable(response) else httpx.Response(200, json=response)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await decisions.route(client, "fixture-private-key", decisions.DEFAULT_BASE_URL,
                decisions.DEFAULT_MODEL, prompt, current)
        return result, requests

    async def check_diagnostic(self, response, *, function="create_plan", arguments=None, errors=None):
        requests = []

        def handler(request):
            requests.append(request)
            return response(request) if callable(response) else httpx.Response(200, json=response)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await decisions.diagnose_call(client, "fixture-private-key", decisions.DEFAULT_BASE_URL,
                decisions.DEFAULT_MODEL, function, arguments if arguments is not None else {"candidate": {}},
                errors if errors is not None else [{"path": "spec", "message": "Field required"}],
                "Create the requested conductor.")
        return result, requests

    async def test_http_contract_sends_only_fixed_function_routing_question(self):
        result, requests = await self.check_route(evidence())
        self.assertEqual(result.function_name, "create_plan")
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(str(request.url), "https://openrouter.ai/api/alpha/decisions")
        self.assertEqual(request.headers["authorization"], "Bearer fixture-private-key")
        body = json.loads(request.content)
        self.assertEqual(set(body), {"model", "state", "questions"})
        self.assertEqual(body["model"], "typesafe/jev-1.14")
        self.assertEqual(set(body["questions"]), {"route"})
        self.assertEqual(body["questions"]["route"]["type"], "choice")
        self.assertEqual(set(body["questions"]["route"]["criteria"]), set(FUNCTIONS))
        self.assertEqual(set(body["state"]["functions"]), set(FUNCTIONS))
        self.assertNotIn("messages", body)
        self.assertEqual(result.trace[0].stage, "routing")
        self.assertEqual(result.trace[0].confidence, 0.97)
        self.assertTrue(result.trace[0].passed)
        self.assertIsNone(result.trace[0].threshold)
        self.assertNotIn("fixture-private-key", result.model_dump_json())
        self.assertNotIn(body["state"]["request"], result.model_dump_json())
        self.assertLessEqual(request.extensions["timeout"]["read"], decisions.CALL_TIMEOUT_SECONDS)

    async def test_route_probability_and_confidence_never_gate_valid_choice(self):
        for probability, confidence in [(.34, 0), (.60, .40), (.99, .01)]:
            with self.subTest(probability=probability, confidence=confidence):
                result, _ = await self.check_route(evidence(probability=probability, confidence=confidence))
                self.assertEqual(result.function_name, "create_plan")
                self.assertFalse(result.questions)
                self.assertTrue(result.trace[0].passed)
                self.assertIsNone(result.trace[0].threshold)

    async def test_create_can_be_selected_with_current_model_and_modify_is_returned_as_a_choice(self):
        for function, current in [("create_plan", wire_example()), ("modify_plan", None)]:
            with self.subTest(function=function):
                result, requests = await self.check_route(evidence(function), current=current)
                self.assertEqual(result.function_name, function)
                state = json.loads(requests[0].content)["state"]
                self.assertEqual(state["current_plan"], current.model_dump(mode="json") if current else None)
                self.assertFalse(result.questions)

    async def test_clarify_is_a_function_choice_not_a_confidence_rejection(self):
        result, _ = await self.check_route(evidence("clarify", probability=.34, confidence=0))
        self.assertEqual(result.function_name, "clarify")
        self.assertTrue(result.questions)
        self.assertTrue(result.trace[0].passed)

    async def test_malformed_or_inconsistent_choices_have_no_default_evidence(self):
        fixtures = []
        for bad in [None, "0.99", True, -.01, 1.01, float("nan"), float("inf")]:
            packet = evidence()
            packet["answers"]["route"]["confidence"] = bad
            fixtures.append(packet)
        for field in ["confidence", "probabilities", "choice"]:
            packet = evidence()
            del packet["answers"]["route"][field]
            fixtures.append(packet)
        for distribution in [
            {"create_plan": .1, "modify_plan": .8, "clarify": .1},
            {"create_plan": .98, "modify_plan": .8, "clarify": .1},
            {"create_plan": .98, "modify_plan": .02},
            {"create_plan": .98, "modify_plan": .01, "clarify": float("nan")},
            {"create_plan": .98, "modify_plan": .01, "clarify": True},
        ]:
            packet = evidence()
            packet["answers"]["route"]["probabilities"] = distribution
            fixtures.append(packet)
        unknown_function = evidence()
        unknown_function["answers"]["route"]["choice"] = "execute_shell"
        fixtures.extend([unknown_function, {"answers": {}}, {"answers": {"route": choice(), "support": {"type": "noul", "noul": 1}}}])
        for packet in fixtures:
            with self.subTest(packet=repr(packet)):
                with self.assertRaises(decisions.DecisionError) as caught:
                    await self.check_route(packet)
                self.assertEqual(caught.exception.status_code, 502)

    async def test_failed_call_diagnostic_sees_actual_function_arguments_contract_and_errors(self):
        arguments = {"candidate": {"spec": {"regions": [{"material": "unknown"}]}},
                     "schema": {"type": "object"}}
        errors = [{"path": "spec.regions.0.material", "message": "Unknown material"}]
        result, requests = await self.check_diagnostic(diagnostic(.2), arguments=arguments, errors=errors)
        self.assertFalse(result.arguments_match)
        self.assertEqual(len(requests), 1)
        body = json.loads(requests[0].content)
        self.assertEqual(set(body["questions"]), {"arguments_match"})
        self.assertEqual(body["questions"]["arguments_match"]["type"], "noul")
        self.assertEqual(body["state"]["function"]["name"], "create_plan")
        self.assertEqual(body["state"]["function"]["contract"], decisions.FUNCTION_CONTRACTS["create_plan"])
        self.assertEqual(body["state"]["arguments"], arguments)
        self.assertEqual(body["state"]["error_details"], errors)
        self.assertEqual(body["state"]["function_result"], "failed")
        self.assertEqual(result.trace[0].stage, "function_error")
        self.assertEqual(result.trace[0].question, "arguments_match")
        self.assertTrue(result.trace[0].passed)
        self.assertIsNone(result.trace[0].confidence)
        self.assertIsNone(result.trace[0].threshold)
        self.assertFalse(hasattr(result, "allowed"))
        self.assertNotIn("unknown", result.model_dump_json())

    async def test_diagnostic_answers_are_informational_even_when_yes(self):
        for probability in [0, .49, .5, .88, 1]:
            with self.subTest(probability=probability):
                result, _ = await self.check_diagnostic(diagnostic(probability))
                self.assertEqual(result.arguments_match, probability >= .5)
                self.assertTrue(result.trace[0].passed)
                self.assertIsNone(result.trace[0].threshold)
                self.assertFalse(hasattr(result, "allowed"))
                self.assertAlmostEqual(result.trace[0].probability, probability)

    async def test_diagnostic_catalog_includes_model_mesh_solver_and_target_checks(self):
        for function in ["create_plan", "modify_plan", "mesh_plan", "solve_plan", "solve_ccore", "check_targets"]:
            with self.subTest(function=function):
                _, requests = await self.check_diagnostic(diagnostic(), function=function)
                self.assertEqual(json.loads(requests[0].content)["state"]["function"]["name"], function)
        for function in ["clarify", "execute_shell", "../../other"]:
            with self.subTest(function=function):
                with self.assertRaises(decisions.DecisionError) as caught:
                    await self.check_diagnostic(diagnostic(), function=function)
                self.assertEqual(caught.exception.status_code, 422)

    async def test_legacy_ccore_diagnostic_uses_its_own_direct_solve_contract(self):
        arguments = {"spec": CCoreSpec().model_dump(mode="json")}
        result, requests = await self.check_diagnostic(diagnostic(.8), function="solve_ccore", arguments=arguments)
        state = json.loads(requests[0].content)["state"]
        self.assertTrue(result.arguments_match)
        self.assertEqual(state["arguments"], arguments)
        self.assertEqual(state["function"]["name"], "solve_ccore")
        contract = state["function"]["contract"]
        self.assertNotEqual(contract, decisions.FUNCTION_CONTRACTS["solve_plan"])
        self.assertIn("No separate mesh job or mesh_job_id is required", contract)
        self.assertNotIn("solve_ccore", decisions.ROUTING_QUESTIONS["route"]["criteria"])

    async def test_diagnostic_requires_an_actual_error_before_provider_call(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=diagnostic())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            for errors in [[], None, {}, ["a failure"], [{}]]:
                with self.subTest(errors=errors):
                    with self.assertRaises(decisions.DecisionError) as caught:
                        await decisions.diagnose_call(client, "fixture-key", decisions.DEFAULT_BASE_URL,
                            decisions.DEFAULT_MODEL, "create_plan", {}, errors, "Create a model.")
                    self.assertEqual(caught.exception.status_code, 422)
        self.assertFalse(requests)

    async def test_malformed_diagnostic_is_not_replaced_with_an_assumed_answer(self):
        for bad in [None, "0.99", True, -.01, 1.01, float("nan"), float("inf")]:
            with self.subTest(bad=bad):
                with self.assertRaises(decisions.DecisionError) as caught:
                    await self.check_diagnostic(diagnostic(bad))
                self.assertEqual(caught.exception.status_code, 502)

    async def test_provider_errors_are_safe_and_actionable(self):
        for status, phrase in [(401, "API key"), (402, "credit"), (429, "rate limiting"),
                               (404, "Decisions API"), (503, "unavailable"), (413, "too large")]:
            with self.subTest(status=status):
                with self.assertRaises(decisions.DecisionError) as caught:
                    await self.check_route(lambda request: httpx.Response(status,
                        json={"error": "fixture-private-key provider-secret"}))
                self.assertIn(phrase, caught.exception.message)
                self.assertNotIn("fixture-private-key", caught.exception.message)
                self.assertNotIn("provider-secret", caught.exception.message)

    async def test_body_and_network_failures_are_bounded_and_redacted(self):
        def timeout(request):
            raise httpx.ReadTimeout("fixture-private-key", request=request)

        def network(request):
            raise httpx.ConnectError("fixture-private-key", request=request)

        fixtures = [lambda request: httpx.Response(200, content=b"x" * (decisions.MAX_RESPONSE_BYTES + 1)),
                    lambda request: httpx.Response(200, content=b"not JSON"), timeout, network]
        for fixture in fixtures:
            with self.subTest(fixture=fixture):
                with self.assertRaises(decisions.DecisionError) as caught:
                    await self.check_route(fixture)
                self.assertNotIn("fixture-private-key", caught.exception.message)

    async def test_one_connect_timeout_retries_identical_route_without_duplicate_trace(self):
        requests = []

        def handler(request):
            requests.append(request)
            if len(requests) == 1:
                raise httpx.ConnectTimeout("fixture-private-key", request=request)
            return httpx.Response(200, json=evidence(probability=.34, confidence=0))

        with patch("app.decisions.CONNECT_RETRY_DELAY_SECONDS", 0):
            result, _ = await self.check_route(handler)
        self.assertEqual(result.function_name, "create_plan")
        self.assertEqual(len(result.trace), 1)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].content, requests[1].content)
        self.assertEqual(requests[0].headers, requests[1].headers)
        self.assertEqual(requests[0].extensions["timeout"]["connect"], decisions.CONNECT_TIMEOUT_SECONDS)
        self.assertEqual(requests[0].extensions["timeout"]["read"], decisions.CALL_TIMEOUT_SECONDS)
        self.assertLess(2 * decisions.CONNECT_TIMEOUT_SECONDS + decisions.CONNECT_RETRY_DELAY_SECONDS,
                        decisions.CALL_TIMEOUT_SECONDS)
        self.assertIsNone(result.trace[0].threshold)

    async def test_connection_error_retry_applies_to_failed_call_diagnostic_only(self):
        requests = []

        def handler(request):
            requests.append(request)
            if len(requests) == 1:
                raise httpx.ConnectError("fixture-private-key", request=request)
            return httpx.Response(200, json=diagnostic(.1))

        with patch("app.decisions.CONNECT_RETRY_DELAY_SECONDS", 0):
            result, _ = await self.check_diagnostic(handler)
        self.assertFalse(result.arguments_match)
        self.assertEqual(len(result.trace), 1)
        self.assertEqual(len(requests), 2)
        self.assertEqual(set(json.loads(requests[1].content)["questions"]), {"arguments_match"})

    async def test_repeated_connection_failures_stop_after_two_attempts_with_safe_error(self):
        for exception, status in [(httpx.ConnectTimeout, 504), (httpx.ConnectError, 502)]:
            requests = []

            def handler(request):
                requests.append(request)
                raise exception("fixture-private-key", request=request)

            with self.subTest(exception=exception.__name__), patch("app.decisions.CONNECT_RETRY_DELAY_SECONDS", 0):
                with self.assertRaises(decisions.DecisionError) as caught:
                    await self.check_route(handler)
                self.assertEqual(caught.exception.status_code, status)
                self.assertEqual(len(requests), 2)
                self.assertNotIn("fixture-private-key", caught.exception.message)
                if exception is httpx.ConnectTimeout:
                    self.assertIn("establish a connection", caught.exception.message)
                    self.assertIn("after two attempts", caught.exception.message)

    async def test_ambiguous_delivery_and_provider_responses_are_never_retried(self):
        for exception in [httpx.ReadTimeout, httpx.WriteTimeout, httpx.ReadError,
                          httpx.WriteError, httpx.RemoteProtocolError, httpx.PoolTimeout]:
            requests = []

            def handler(request):
                requests.append(request)
                raise exception("fixture-private-key", request=request)

            with self.subTest(exception=exception.__name__):
                with self.assertRaises(decisions.DecisionError):
                    await self.check_route(handler)
                self.assertEqual(len(requests), 1)
        for response in [httpx.Response(429), httpx.Response(503), httpx.Response(200, content=b"not JSON"),
                         httpx.Response(200, json={"answers": {"route": {**choice(), "choice": "unknown"}}})]:
            requests = []

            def handler(request):
                requests.append(request)
                return response

            with self.subTest(response=response.status_code):
                with self.assertRaises(decisions.DecisionError):
                    await self.check_route(handler)
                self.assertEqual(len(requests), 1)

    async def test_one_wall_deadline_covers_reconnect_and_slow_response(self):
        requests = []

        async def handler(request):
            requests.append(request)
            if len(requests) == 1:
                raise httpx.ConnectTimeout("fixture-private-key", request=request)
            await asyncio.sleep(.1)
            return httpx.Response(200, json=evidence())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with patch("app.decisions.CALL_TIMEOUT_SECONDS", .02), patch("app.decisions.CONNECT_RETRY_DELAY_SECONDS", 0):
                with self.assertRaises(decisions.DecisionError) as caught:
                    await decisions.route(client, "fixture-key", decisions.DEFAULT_BASE_URL,
                        decisions.DEFAULT_MODEL, "Create the requested wire.", None)
        self.assertEqual(caught.exception.status_code, 504)
        self.assertEqual(len(requests), 2)

    async def test_oversized_and_nonserializable_diagnostic_state_stops_before_provider(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=diagnostic())

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            for arguments in [{"candidate": "x" * decisions.MAX_STATE_BYTES}, {"candidate": object()}, {"candidate": float("nan")}]:
                with self.subTest(arguments_type=type(arguments["candidate"]).__name__):
                    with self.assertRaises(decisions.DecisionError) as caught:
                        await decisions.diagnose_call(client, "fixture-key", decisions.DEFAULT_BASE_URL,
                            decisions.DEFAULT_MODEL, "create_plan", arguments,
                            [{"message": "Actual validation failed"}], "Create a model.")
                    self.assertEqual(caught.exception.status_code, 422)
        self.assertFalse(requests)

    def test_legacy_threshold_is_ignored_and_semantic_gate_api_is_removed(self):
        for threshold in [None, .5, 1.01, float("nan"), True]:
            decisions.validate_configuration(decisions.DEFAULT_MODEL, decisions.DEFAULT_BASE_URL, threshold)
        for name in ["preflight", "fidelity", "PREFLIGHT_QUESTIONS", "FIDELITY_QUESTIONS"]:
            self.assertFalse(hasattr(decisions, name))
        with self.assertRaises(decisions.DecisionError):
            decisions.validate_configuration("", decisions.DEFAULT_BASE_URL)


class OrchestrationTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, handler, current=None, **kwargs):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch("app.planner.httpx.AsyncClient", return_value=client):
            return await interpret_plan("Create a round 1 A conductor with radius 1 mm and depth 20 mm.", current,
                "fixture-private-key", "https://provider.test/api/v1", "fixture-generator",
                decision_base_url="https://provider.test/api/alpha", **kwargs)

    async def test_valid_plan_uses_route_then_function_without_semantic_approval(self):
        calls = []

        def handler(request):
            calls.append(request.url.path)
            if request.url.path.endswith("/decisions"):
                return httpx.Response(200, json=evidence(probability=.34, confidence=0))
            return httpx.Response(200, json=completion(wire_example().model_dump()))

        result = await self.call(handler, decision_threshold=1.01)
        self.assertIsNotNone(result.spec)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(len(result.decision_trace), 1)
        self.assertEqual(calls, ["/api/alpha/decisions", "/api/v1/chat/completions"])

    async def test_maximum_four_calls_route_error_diagnostic_and_two_drafts(self):
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
                self.assertIn("undefined_material", json.dumps(packet["state"]["arguments"]))
                self.assertEqual(packet["state"]["function_result"], "failed")
                return httpx.Response(200, json=diagnostic(.99))
            generations += 1
            return httpx.Response(200, json=completion(invalid if generations == 1 else wire_example().model_dump()))

        result = await self.call(handler)
        self.assertIsNotNone(result.spec)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(calls, ["/api/alpha/decisions", "/api/v1/chat/completions",
                                 "/api/alpha/decisions", "/api/v1/chat/completions"])
        self.assertEqual([item.question for item in result.decision_trace], ["route", "arguments_match"])

    async def test_diagnostic_yes_cannot_bypass_repeated_validation_failure(self):
        calls = []
        invalid = wire_example().model_dump()
        invalid["regions"][0]["material"] = "undefined_material"

        def handler(request):
            calls.append(request.url.path)
            packet = json.loads(request.content)
            if request.url.path.endswith("/decisions"):
                return httpx.Response(200, json=evidence() if "route" in packet["questions"] else diagnostic(1))
            return httpx.Response(200, json=completion(invalid))

        with self.assertRaises(PlannerError):
            await self.call(handler)
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls.count("/api/alpha/decisions"), 2)

    async def test_failed_optional_diagnostic_still_allows_deterministic_repair(self):
        invalid = wire_example().model_dump()
        invalid["regions"][0]["material"] = "undefined_material"
        generations = 0

        def handler(request):
            nonlocal generations
            packet = json.loads(request.content)
            if request.url.path.endswith("/decisions"):
                if "route" in packet["questions"]:
                    return httpx.Response(200, json=evidence())
                return httpx.Response(503, text="fixture-private-key provider-secret")
            generations += 1
            return httpx.Response(200, json=completion(invalid if generations == 1 else wire_example().model_dump()))

        result = await self.call(handler)
        self.assertIsNotNone(result.spec)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(result.decision_trace), 1)
        self.assertIn("unavailable", result.diagnostic_error)
        self.assertNotIn("provider-secret", result.diagnostic_error)

    async def test_clarify_routes_without_generator_call(self):
        calls = []

        def handler(request):
            calls.append(request.url.path)
            return httpx.Response(200, json=evidence("clarify"))

        result = await self.call(handler)
        self.assertIsNone(result.spec)
        self.assertTrue(result.questions)
        self.assertEqual(result.attempts, 0)
        self.assertEqual(calls, ["/api/alpha/decisions"])

    async def test_unavailable_router_has_no_generator_fallback(self):
        calls = []

        def handler(request):
            calls.append(request.url.path)
            return httpx.Response(503, text="fixture-private-key upstream-secret")

        with self.assertRaises(PlannerError) as caught:
            await self.call(handler)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("fixture-private-key", caught.exception.message)

    async def test_total_timeout_can_cancel_connection_retry_without_starting_draft(self):
        calls = []

        def handler(request):
            calls.append(request.url.path)
            raise httpx.ConnectTimeout("fixture-private-key", request=request)

        with patch("app.planner.TOTAL_TIMEOUT_SECONDS", .01), patch("app.decisions.CONNECT_RETRY_DELAY_SECONDS", .1):
            with self.assertRaises(PlannerError) as caught:
                await self.call(handler)
        self.assertEqual(caught.exception.status_code, 504)
        self.assertEqual(calls, ["/api/alpha/decisions"])

    async def test_total_timeout_covers_routing(self):
        calls = []

        async def handler(request):
            calls.append(request.url.path)
            await asyncio.sleep(.05)
            return httpx.Response(200, json=evidence())

        with patch("app.planner.TOTAL_TIMEOUT_SECONDS", .01):
            with self.assertRaises(PlannerError) as caught:
                await self.call(handler)
        self.assertEqual(caught.exception.status_code, 504)
        self.assertEqual(calls, ["/api/alpha/decisions"])


if __name__ == "__main__":
    unittest.main()
