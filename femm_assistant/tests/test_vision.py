"""Optional visual review uses local images and HTTP mocks, without paid calls."""
import asyncio
import base64
import json
import struct
import unittest
from unittest.mock import patch

import httpx

from app.plan import wire_example
from app.vision import (MAX_PROVIDER_BYTES, MAX_SVG_BYTES, MAX_SVG_NODES,
                        VisualReview, VisionError, _checked_svg, _render_png, review)


def completion(answer=None, **extra):
    answer = answer or {"advisory": True, "status": "matches", "summary": "The visible conductor matches the request.", "findings": []}
    return {"choices": [{"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}], **extra}


class VisionTests(unittest.IsolatedAsyncioTestCase):
    async def call_with(self, handler, **kwargs):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await review(client, "private-test-key", "https://provider.test/api/v1", "test/vision-model",
                                "Create the described round conductor model.", wire_example(), **kwargs)

    async def test_sends_bounded_png_and_typed_plan_in_one_advisory_call(self):
        received = []
        original = wire_example()
        before = original.model_dump_json()

        def handler(request):
            received.append(request)
            return httpx.Response(200, json=completion(usage={"prompt_tokens": 2000, "completion_tokens": 40,
                "total_tokens": 2040, "cost": .0003, "provider_secret": "never-return", "completion_tokens_details": {"reasoning_tokens": 0}}))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await review(client, "private-test-key", "https://provider.test/api/v1/", "test/vision-model",
                                  "Create the described round conductor model.", original)
        self.assertIsInstance(result, VisualReview)
        self.assertTrue(result.advisory)
        self.assertEqual(result.requested_model, "test/vision-model")
        self.assertEqual(result.usage, {"prompt_tokens": 2000, "completion_tokens": 40, "total_tokens": 2040,
                                       "cost": .0003, "reasoning_tokens": 0})
        self.assertEqual(before, original.model_dump_json())
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].url.path, "/api/v1/chat/completions")
        payload = json.loads(received[0].content)
        self.assertEqual(payload["max_tokens"], 1200)
        self.assertEqual(payload["temperature"], 0)
        self.assertNotIn("reasoning", payload)
        self.assertIn("never approve or reject", payload["messages"][0]["content"])
        content = payload["messages"][1]["content"]
        self.assertEqual(len(content), 2)
        context = json.loads(content[0]["text"])
        self.assertEqual(context["typed_plan"], original.model_dump(mode="json"))
        self.assertEqual(context["image_order"], ["geometry_preview"])
        png = base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1], validate=True)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (1024, 768))

    async def test_actual_mesh_image_is_optional_and_marked_in_image_order(self):
        received = []

        def handler(request):
            received.append(json.loads(request.content))
            return httpx.Response(200, json=completion({"advisory": True, "status": "issues", "summary": "Inspect this region.",
                "findings": [{"location": "upper corner", "issue": "A slender triangle is visible.", "suggestion": "Inspect the element metrics and refine locally."}]}))

        result = await self.call_with(handler, mesh_svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><polygon points="0,0 10,0 1,1" fill="#fff"/></svg>')
        self.assertEqual(result.status, "issues")
        content = received[0]["messages"][1]["content"]
        self.assertEqual(len(content), 3)
        self.assertEqual(json.loads(content[0]["text"])["image_order"], ["geometry_preview", "actual_native_mesh"])
        for item in content[1:]:
            self.assertTrue(base64.b64decode(item["image_url"]["url"].split(",", 1)[1]).startswith(b"\x89PNG"))

    async def test_invalid_advisory_contract_and_excessive_output_are_rejected(self):
        valid = {"advisory": True, "status": "matches", "summary": "Visible comparison only.", "findings": []}
        for replacement in ({"advisory": False}, {"status": "approved"}, {"summary": "x" * 1601}, {"approved": True},
                            {"findings": [{"location": "region", "issue": "issue", "suggestion": "suggestion"}] * 9}):
            with self.subTest(replacement=list(replacement)):
                def handler(request):
                    return httpx.Response(200, json=completion({**valid, **replacement}))
                with self.assertRaises(VisionError) as caught:
                    await self.call_with(handler)
                self.assertEqual(caught.exception.status_code, 502)
                self.assertIn("invalid or incomplete", caught.exception.message)

    async def test_provider_body_and_key_are_never_returned_in_errors(self):
        for status, phrase in ((401, "access"), (402, "credit"), (429, "rate limiting"), (400, "image format"), (503, "unavailable")):
            with self.subTest(status=status):
                def handler(request):
                    return httpx.Response(status, text="private-test-key upstream-secret")
                with self.assertRaises(VisionError) as caught:
                    await self.call_with(handler)
                self.assertIn(phrase, caught.exception.message)
                self.assertNotIn("private-test-key", str(caught.exception))
                self.assertNotIn("upstream-secret", str(caught.exception))

    async def test_malformed_and_truncated_responses_have_safe_errors(self):
        packets = [b"private-test-key upstream-secret", json.dumps({"choices": []}).encode(),
                   json.dumps({"choices": [{"message": {"content": "private-test-key"}, "finish_reason": "length"}]}).encode(),
                   b"x" * (MAX_PROVIDER_BYTES + 1)]
        for packet in packets:
            with self.subTest(size=len(packet)):
                def handler(request):
                    return httpx.Response(200, content=packet)
                with self.assertRaises(VisionError) as caught:
                    await self.call_with(handler)
                self.assertEqual(caught.exception.status_code, 502)
                self.assertNotIn("private-test-key", str(caught.exception))
                self.assertNotIn("upstream-secret", str(caught.exception))

    async def test_timeout_and_network_errors_are_sanitized(self):
        for error, status in ((httpx.ReadTimeout, 504), (httpx.ConnectError, 502)):
            with self.subTest(error=error):
                def handler(request):
                    raise error("private-test-key upstream-secret", request=request)
                with self.assertRaises(VisionError) as caught:
                    await self.call_with(handler)
                self.assertEqual(caught.exception.status_code, status)
                self.assertNotIn("private-test-key", str(caught.exception))

    async def test_total_deadline_is_enforced(self):
        async def handler(request):
            await asyncio.sleep(1)
            return httpx.Response(200, json=completion())
        with patch("app.vision.TOTAL_TIMEOUT_SECONDS", .01):
            with self.assertRaises(VisionError) as caught:
                await self.call_with(handler)
        self.assertEqual(caught.exception.status_code, 504)

    async def test_credential_in_successful_content_is_rejected(self):
        def handler(request):
            return httpx.Response(200, json=completion({"advisory": True, "status": "uncertain",
                "summary": "private-test-key", "findings": []}))
        with self.assertRaises(VisionError) as caught:
            await self.call_with(handler)
        self.assertNotIn("private-test-key", str(caught.exception))

    async def test_invalid_mesh_is_rejected_before_provider_request(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(200, json=completion())

        with self.assertRaises(VisionError):
            await self.call_with(handler, mesh_svg='<svg><image href="file:///etc/passwd"/></svg>')
        self.assertFalse(calls)

    async def test_no_arbitrary_usage_metadata_or_nonfinite_counters(self):
        def handler(request):
            return httpx.Response(200, json=completion(usage={"prompt_tokens": True, "completion_tokens": -1,
                "total_tokens": 2.5, "cost": "private-test-key", "secret": "upstream-secret",
                "completion_tokens_details": {"reasoning_tokens": True}}))
        result = await self.call_with(handler)
        self.assertEqual(result.usage, {})


class RenderingTests(unittest.TestCase):
    def test_rejects_oversized_or_complex_svg_before_cairo(self):
        for source in ("x" * (MAX_SVG_BYTES + 1), "<svg>" + "<g/>" * MAX_SVG_NODES + "</svg>"):
            with self.subTest(size=len(source)), self.assertRaises(VisionError):
                _checked_svg(source)

    def test_rejects_external_resources_and_active_svg(self):
        for source in ('<!DOCTYPE svg [<!ENTITY e "secret">]><svg>&e;</svg>', '<svg><script>secret</script></svg>',
                       '<svg><image href="https://provider.test/secret"/></svg>', '<svg><g style="fill:url(file:///etc/passwd)"/></svg>',
                       '<svg><path fill="url(https://provider.test/secret)" d="M0,0"/></svg>', '<html/>', '<svg>'):
            with self.subTest(source=source), self.assertRaises(VisionError):
                _checked_svg(source)

    def test_png_size_limit_and_render_failure_are_safe(self):
        with patch("cairosvg.svg2png", side_effect=ValueError("private-test-key upstream-secret")):
            with self.assertRaises(VisionError) as caught:
                _render_png("<svg/>")
            self.assertNotIn("private-test-key", str(caught.exception))
        with patch("cairosvg.svg2png", return_value=b"not a png"):
            with self.assertRaises(VisionError):
                _render_png("<svg/>")
        with patch("app.vision.MAX_PNG_BYTES", 8), patch("cairosvg.svg2png", return_value=b"\x89PNG\r\n\x1a\nx"):
            with self.assertRaises(VisionError):
                _render_png("<svg/>")


if __name__ == "__main__":
    unittest.main()
