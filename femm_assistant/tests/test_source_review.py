"""Mocked provider comparisons using actual local PNG and PDF provenance."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import Depends, FastAPI, Header, HTTPException
import httpx
from PIL import Image
from pydantic import ValidationError

from app import paper_assets, source_review, structured
from app.paper_assets import CropBox, PaperAsset, PaperAssetStore
from app.plan import wire_example
from app.replication import ReplicationStore, _encode
from app.source_review import SourceReviewError, SourceReviewRequest, review
from tests.test_paper_assets import drawing_pdf


def png(width=400, height=300):
    output = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(output, format="PNG")
    return output.getvalue()


def source_asset(data=None, *, case_id="baseline"):
    data = data or png()
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
    paper = "a" * 64
    asset = PaperAsset(paper_id=paper, source_id="paper_" + paper[:24], page=1,
        bbox=CropBox(x0=0, y0=0, x1=400, y1=300), page_width_pt=400, page_height_pt=300,
        page_rotation=0, purpose="figure", case_id=case_id, label="Design figure",
        png_sha256=hashlib.sha256(data).hexdigest(), width_px=width, height_px=height).model_dump(mode="json")
    return {"asset_id": hashlib.sha256(_encode(asset)).hexdigest(), **asset}, data


def answer(asset, *, findings=None, status="consistent"):
    return {"advisory": True, "status": status, "view_alignment": "same_section",
            "summary": "The displayed section is consistent with the selected source figure.",
            "findings": findings or []}


def finding(asset, **change):
    return {"feature_id": wire_example().regions[0].id, "source_asset_ids": [asset["asset_id"]],
            "check": "count", "observation": "The selected source shows two objects; the model has one.",
            "uncertainty": "medium", "uncertainty_reason": "The source may be a full revolution view.",
            "suggested_inspection": "Compare the source section with the radial-axial model section.", **change}


def completion(value, **extra):
    return {"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}], **extra}


class SourceReviewTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, handler, *, asset=None, image=None, **kwargs):
        default, default_image = source_asset()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await review(client, "private-review-key", "https://provider.test/v1", "test/vision",
                "Compare the source and generated model.", wire_example(), [asset or default],
                [image or default_image], **kwargs)

    async def test_real_local_images_strict_contract_feature_ids_and_no_token_or_read_cap(self):
        asset, image = source_asset()
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=completion(answer(asset), usage={"prompt_tokens": 100,
                "completion_tokens": 50, "cost": .0001, "private_metadata": "discard"}))
        original = wire_example().model_dump_json()
        result = await self.call(handler, asset=asset, image=image)
        self.assertTrue(result.advisory)
        self.assertEqual(result.asset_ids, [asset["asset_id"]])
        self.assertEqual(result.usage, {"prompt_tokens": 100, "completion_tokens": 50, "cost": .0001})
        self.assertEqual(wire_example().model_dump_json(), original)
        self.assertEqual(len(requests), 1)
        payload = json.loads(requests[0].content)
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertTrue(payload["provider"]["require_parameters"])
        self.assertIsNone(requests[0].extensions["timeout"]["read"])
        self.assertIsNone(requests[0].extensions["timeout"]["write"])
        content = payload["messages"][1]["content"]
        context = json.loads(content[0]["text"])
        self.assertEqual(context["source_assets"][0]["png_sha256"], hashlib.sha256(image).hexdigest())
        self.assertEqual(context["model_view"], "planar_x_y_section")
        self.assertIn(wire_example().regions[0].id, context["feature_ids"])
        self.assertEqual([a["role"] for a in context["image_order"]], ["source", "geometry"])
        for part in content[1:]:
            self.assertTrue(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]).startswith(b"\x89PNG"))

    async def test_optional_jev_uses_one_batched_decisions_call_with_actual_images(self):
        asset, image = source_asset(png(100, 70))
        requests = []
        observations = [finding(asset), finding(asset, check="orientation", feature_id=None)]
        def handler(request):
            requests.append(request)
            if request.url.path.endswith("/chat/completions"):
                return httpx.Response(200, json=completion(answer(asset, findings=observations, status="needs_inspection")))
            return httpx.Response(200, json={"answers": {
                "finding_0": {"type": "choice", "choice": "uncertain", "probabilities": {"supported": .1, "contradicted": .1, "uncertain": .8}, "confidence": .7},
                "finding_1": {"type": "choice", "choice": "contradicted", "probabilities": {"supported": .1, "contradicted": .9, "uncertain": 0}, "confidence": .9}},
                "usage": {"input_tokens": 300, "cost": .001}})
        result = await self.call(handler, asset=asset, image=image, decision_model="typesafe/jev-1.14",
                                 decision_base_url="https://provider.test/alpha")
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[1].url.path, "/alpha/decisions")
        payload = json.loads(requests[1].content)
        self.assertEqual(set(payload["questions"]), {"finding_0", "finding_1"})
        self.assertIsInstance(payload["state"][0], str)
        self.assertEqual(payload["state"][1]["type"], "image_url")
        self.assertNotIn("max_tokens", payload)
        self.assertTrue(result.multimodal_decisions_experimental)
        self.assertEqual([x.answer for x in result.focused_checks], ["uncertain", "contradicted"])
        self.assertEqual(result.status, "needs_inspection")  # No score changes the advisory.
        self.assertEqual(result.focused_check_usage, {"input_tokens": 300, "cost": .001})
        self.assertTrue(result.images[0].padded_for_multimodal_limits)
        self.assertEqual((result.images[0].width_px, result.images[0].height_px), (280, 280))

    async def test_case_hash_source_references_and_feature_ids_are_not_trusted(self):
        asset, image = source_asset()
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(200, json=completion(answer(asset)))
        for modified, data, kwargs in [({**asset, "page": 2}, image, {}),
                (asset, image + b"tamper", {}), (asset, image, {"case_id": "other_case"})]:
            with self.subTest(kwargs=kwargs), self.assertRaises(SourceReviewError):
                await self.call(handler, asset=modified, image=data, **kwargs)
        self.assertFalse(calls)
        for observation in [finding(asset, source_asset_ids=["b" * 64]), finding(asset, feature_id="invented"),
                            finding(asset, check="mesh")]:
            def invalid(request):
                return httpx.Response(200, json=completion(answer(asset, findings=[observation])))
            with self.subTest(observation=observation), self.assertRaises(SourceReviewError) as caught:
                await self.call(invalid, asset=asset, image=image)
            self.assertEqual(caught.exception.status_code, 502)

    async def test_semantic_feature_map_and_actual_mesh_are_included_and_validated(self):
        asset, image = source_asset()
        region = wire_example().regions[0].id
        mapping = {"winding.cross_section@000": {"region_id": region, "fact_ids": ["geometry_fact"]}}
        received = []
        def handler(request):
            received.append(json.loads(request.content))
            return httpx.Response(200, json=completion(answer(asset, findings=[finding(asset,
                feature_id="winding.cross_section@000", check="mesh")], status="needs_inspection")))
        result = await self.call(handler, feature_map=mapping,
            mesh_svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><polygon points="0,0 10,0 1,1"/></svg>')
        self.assertEqual(result.findings[0].feature_id, "winding.cross_section@000")
        context = json.loads(received[0]["messages"][1]["content"][0]["text"])
        self.assertEqual(context["image_order"][-1]["role"], "actual_native_mesh")
        self.assertEqual(context["feature_map"], mapping)
        with self.assertRaises(SourceReviewError):
            await self.call(handler, feature_map={"feature": {"region_id": "other_model_region"}})

    async def test_unknown_assembly_can_be_compared_visually_without_native_support(self):
        from app.assembly import compile_assembly, preview_assembly
        from tests.test_assembly import custom_assembly
        asset, image = source_asset()
        raw = custom_assembly().model_dump(mode="json")
        raw["required_physics"] = ["permanent_magnets"]
        compilation = compile_assembly(raw)
        identity = next(iter(compilation.feature_map))
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=completion(answer(asset, findings=[finding(asset,
                feature_id=identity)], status="needs_inspection")))
        assembly = {"assembly": raw, "compilation": compilation.model_dump(mode="json")}
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await review(client, "private-review-key", "https://provider.test/v1", "test/vision", "Compare source.",
                None, [asset], [image], geometry_svg=preview_assembly(compilation),
                geometry_data=assembly, feature_map=compilation.feature_map)
        self.assertEqual(result.model_sha256, hashlib.sha256(_encode(assembly)).hexdigest())
        context = json.loads(requests[0]["messages"][1]["content"][0]["text"])
        self.assertEqual(context["model_view"], "assembly_2d_preview")
        self.assertEqual(context["typed_plan"], assembly)

    async def test_provider_errors_malformed_output_and_sensitive_content_are_safe(self):
        asset, image = source_asset()
        for status in [400, 401, 402, 429, 503]:
            def handler(request):
                return httpx.Response(status, text="private-review-key provider-private-data")
            with self.subTest(status=status), self.assertRaises(SourceReviewError) as caught:
                await self.call(handler)
            self.assertNotIn("private-review-key", str(caught.exception))
            self.assertNotIn("provider-private-data", str(caught.exception))
        for body in [b"private-review-key", b"not json", json.dumps(completion({**answer(asset), "advisory": False})).encode()]:
            def handler(request):
                return httpx.Response(200, content=body)
            with self.assertRaises(SourceReviewError):
                await self.call(handler)

    async def test_optional_decision_failure_retains_the_original_vision_findings(self):
        asset, image = source_asset()
        def handler(request):
            if request.url.path.endswith("/chat/completions"):
                return httpx.Response(200, json=completion(answer(asset,
                    findings=[finding(asset)], status="needs_inspection")))
            return httpx.Response(400, text="private-review-key provider-private-data")
        result = await self.call(handler, asset=asset, image=image, decision_model="typesafe/jev-1.14")
        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.focused_check_status, "failed")
        self.assertTrue(result.advisory)
        self.assertEqual(result.focused_checks, [])
        self.assertNotIn("private-review-key", result.focused_check_error)
        self.assertNotIn("provider-private-data", result.focused_check_error)

    async def test_wire_contract_missing_fields_and_bool_placeholders_are_rejected(self):
        asset, image = source_asset()
        for change in [lambda item: item.pop("view_alignment"),
                       lambda item: item.update(advisory=1),
                       lambda item: item.update(findings=[{**finding(asset), "uncertainty_reason": None}])]:
            value = answer(asset)
            change(value)
            def handler(request):
                return httpx.Response(200, json=completion(value))
            with self.assertRaises(SourceReviewError):
                await self.call(handler, asset=asset, image=image)

    def test_exactly_one_typed_model_or_server_assembly_is_required(self):
        for body in [{"asset_ids": ["a" * 64]},
                     {"asset_ids": ["a" * 64], "spec": wire_example(), "assembly_id": "b" * 64},
                     {"asset_ids": ["a" * 64], "assembly_id": "b" * 64, "geometry_svg": "<svg/>"},
                     {"asset_ids": ["a" * 64], "assembly_id": "b" * 64, "feature_map": {}}]:
            with self.subTest(body=list(body)), self.assertRaises(ValidationError):
                SourceReviewRequest.model_validate(body)


class SourceReviewAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if not shutil.which("pdftoppm"):
            self.skipTest("Poppler required for PDF/source review integration")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.papers = ReplicationStore(self.root)
        self.assets = PaperAssetStore(self.papers)
        paper = self.papers.put_paper(drawing_pdf())
        self.asset = self.assets.create(paper["paper_id"], {"page": 1, "purpose": "figure"})
        self.app = FastAPI()
        def authorize(x_assistant_token: str = Header(default="")):
            if x_assistant_token != "local-test-token":
                raise HTTPException(401, "Access token required.")
        self.secured = [Depends(authorize)]
        self.settings = SimpleNamespace(ai_key="private-review-key", ai_base_url="https://provider.test/v1",
            vision_model="test/vision", decision_model="typesafe/jev-1.14", decision_base_url="https://provider.test/alpha")
        self.budget = SimpleNamespace(use=lambda *_: None)
        self.jobs = SimpleNamespace(root=self.root, get=lambda _: None)

    async def test_authenticated_full_pdf_to_comparison_record_is_immutable(self):
        source_review.register_routes(self.app, self.secured, self.settings, self.budget, self.assets, self.jobs)
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=completion(answer(self.asset)))
        provider_factory = httpx.AsyncClient
        def provider():
            return provider_factory(transport=httpx.MockTransport(handler))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://local") as client:
            body = {"asset_ids": [self.asset["asset_id"]], "spec": wire_example().model_dump(mode="json")}
            self.assertEqual((await client.post("/api/source-review", json=body)).status_code, 401)
            client.headers["X-Assistant-Token"] = "local-test-token"
            with patch("app.source_review.httpx.AsyncClient", side_effect=provider):
                response = await client.post("/api/source-review", json=body)
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["asset_ids"], [self.asset["asset_id"]])
            stored = await client.get("/api/source-reviews/" + result["review_id"])
            self.assertEqual(stored.status_code, 200)
            self.assertEqual(stored.json()["source_assets"][0], self.asset)
            self.assertEqual(stored.json()["spec"], body["spec"])
            for metadata in result["images"]:
                image = await client.get("/api/source-reviews/" + result["review_id"] + "/images/" + metadata["transmitted_png_sha256"])
                self.assertEqual(image.status_code, 200)
                self.assertEqual(hashlib.sha256(image.content).hexdigest(), metadata["transmitted_png_sha256"])
            self.assertEqual((await client.get("/api/source-reviews/" + result["review_id"] + "/images/" + "b" * 64)).status_code, 404)
            path = self.assets.root.parent / "source-reviews" / (result["review_id"] + ".json")
            path.write_bytes(path.read_bytes() + b"tamper")
            self.assertEqual((await client.get("/api/source-reviews/" + result["review_id"])).status_code, 409)
            self.assertEqual(len(requests), 1)

    async def test_missing_sources_stale_mesh_and_server_assembly_resolution(self):
        compiled = {"assembly": {"physics": "unsupported"}, "compilation": {"native_supported": False},
            "feature_map": {"coil.section@000": {"region_id": "r_abc"}},
            "preview_svg": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><rect width="10" height="8"/></svg>'}
        resolved = []
        def resolve(identity):
            resolved.append(identity)
            return compiled
        source_review.register_routes(self.app, self.secured, self.settings, self.budget, self.assets, self.jobs,
                                      assembly_resolver=resolve)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://local",
                                    headers={"X-Assistant-Token": "local-test-token"}) as client:
            body = {"asset_ids": ["b" * 64], "spec": wire_example().model_dump(mode="json")}
            self.assertEqual((await client.post("/api/source-review", json=body)).status_code, 404)
            body["asset_ids"] = [self.asset["asset_id"]]
            body["mesh_job_id"] = "a" * 32
            self.assertEqual((await client.post("/api/source-review", json=body)).status_code, 409)
            original_factory = httpx.AsyncClient
            def provider():
                return original_factory(transport=httpx.MockTransport(lambda request: httpx.Response(200,
                    json=completion(answer(self.asset, status="insufficient_visual_evidence")))))
            with patch("app.source_review.httpx.AsyncClient", side_effect=provider):
                response = await client.post("/api/source-review", json={"asset_ids": [self.asset["asset_id"]],
                    "assembly_id": "c" * 64})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(resolved, ["c" * 64])


if __name__ == "__main__":
    unittest.main()
