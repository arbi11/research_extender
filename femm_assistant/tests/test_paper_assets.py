"""Actual local PDF rendering, crop coordinates and immutable provenance."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI, HTTPException
import httpx
from PIL import Image
from pydantic import ValidationError

from app import paper_assets
from app.paper_assets import AssetError, AssetRequest, CropBox, PaperAssetStore
from app.replication import ReplicationIntegrityError, ReplicationStore
from tests.test_papers import text_pdf


def drawing_pdf(*, rotated=False):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, NameObject, RectangleObject
    writer = PdfWriter()
    page = writer.add_blank_page(width=400, height=300)
    stream = DecodedStreamObject()
    stream.set_data(b"0 0 1 rg 50 50 100 100 re f\n1 0 0 rg 200 120 100 120 re f\n")
    page[NameObject("/Contents")] = writer._add_object(stream)
    if rotated:
        page.cropbox = RectangleObject([20, 10, 380, 290])
        page.rotate(90)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.papers = ReplicationStore(self.root)
        self.assets = PaperAssetStore(self.papers)
        self.source = self.papers.put_paper(drawing_pdf(), "two-object-design.pdf")

    def tearDown(self):
        self.temporary.cleanup()

    def test_inventory_is_honestly_pages_with_text_hints_not_figure_locations(self):
        source = self.papers.put_paper(text_pdf("Figure 2. Coils. Table 1. Dimensions."))
        inventory = self.assets.inventory(source["paper_id"])
        self.assertEqual(len(inventory["candidates"]), 1)
        page = inventory["candidates"][0]
        self.assertEqual(page["kind"], "page")
        self.assertFalse(page["automatic_figure_bounds"])
        self.assertEqual(page["width_pt"], 600)
        self.assertIn("Figure 2", page["caption_hints"][0])
        self.assertIn("select and crop", inventory["selection_policy"])
        self.assertEqual(inventory["assets"], [])

    @unittest.skipUnless(shutil.which("pdftoppm"), "Poppler required for actual PDF image tests")
    def test_real_render_and_top_left_crop_include_original_page_and_bytes(self):
        identity = self.source["paper_id"]
        request = {"page": 1, "bbox": {"x0": 50, "y0": 150, "x1": 150, "y1": 250},
                   "purpose": "figure", "case_id": "blue_design", "label": "Blue object"}
        asset = self.assets.create(identity, request)
        again = self.assets.create(identity, request)
        self.assertEqual(asset, again)
        self.assertEqual(asset["paper_id"], hashlib.sha256(drawing_pdf()).hexdigest())
        self.assertEqual(asset["bbox"], {"x0": 50., "y0": 150., "x1": 150., "y1": 250.})
        self.assertEqual(asset["source_id"], "paper_" + identity[:24])
        png = self.assets.image(asset)
        self.assertEqual(hashlib.sha256(png).hexdigest(), asset["png_sha256"])
        with Image.open(io.BytesIO(png)) as image:
            self.assertEqual(image.size, (392, 392))
            self.assertEqual(image.getpixel((image.width // 2, image.height // 2)), (0, 0, 255))
        self.assertEqual(self.assets.get(asset["asset_id"]), asset)
        self.assertEqual(self.assets.inventory(identity)["assets"], [asset])
        # Case and crop intent are immutable provenance even when pixels match.
        variation = self.assets.create(identity, {**request, "case_id": "other_case"})
        self.assertNotEqual(asset["asset_id"], variation["asset_id"])
        self.assertEqual(asset["png_sha256"], variation["png_sha256"])

    @unittest.skipUnless(shutil.which("pdftoppm"), "Poppler required for actual PDF image tests")
    def test_cropbox_and_rotation_dimensions_are_display_coordinates(self):
        source = self.papers.put_paper(drawing_pdf(rotated=True))
        page = self.assets.inventory(source["paper_id"])["candidates"][0]
        self.assertEqual((page["width_pt"], page["height_pt"], page["rotation"]), (280, 360, 90))
        asset = self.assets.create(source["paper_id"], {"page": 1})
        self.assertEqual(asset["bbox"], {"x0": 0., "y0": 0., "x1": 280., "y1": 360.})
        self.assertEqual(asset["height_px"], paper_assets.RENDER_LONG_EDGE)
        self.assertLess(asset["width_px"], asset["height_px"])

    def test_invalid_coordinates_pages_and_remote_inputs_stop_before_rendering(self):
        for body in ({"page": True}, {"page": 0}, {"page": 1, "url": "https://example.test/a.png"},
                     {"page": 1, "bbox": {"x0": 1, "y0": 2, "x1": 0, "y1": 3}},
                     {"page": 1, "bbox": {"x0": 0, "y0": 0, "x1": float("nan"), "y1": 3}}):
            with self.subTest(body=body), self.assertRaises(ValidationError):
                AssetRequest.model_validate(body)
        with patch("app.paper_assets._render_page") as renderer:
            for body in ({"page": 2}, {"page": 1, "bbox": {"x0": 0, "y0": 0, "x1": 401, "y1": 10}}):
                with self.assertRaises(AssetError):
                    self.assets.create(self.source["paper_id"], body)
            self.assertFalse(renderer.called)
        self.assertIsNone(self.assets.get("../../outside"))

    @unittest.skipUnless(shutil.which("pdftoppm"), "Poppler required for actual PDF image tests")
    def test_pdf_record_and_image_tampering_are_detected(self):
        asset = self.assets.create(self.source["paper_id"], {"page": 1})
        record_path = self.assets.root / "records" / (asset["asset_id"] + ".json")
        original = record_path.read_bytes()
        record_path.write_bytes(original.replace(b'"page":1', b'"page":2'))
        with self.assertRaises(ReplicationIntegrityError):
            self.assets.get(asset["asset_id"])
        record_path.write_bytes(original)
        image = self.assets.root / "images" / (asset["png_sha256"] + ".png")
        image.write_bytes(image.read_bytes() + b"tamper")
        with self.assertRaises(ReplicationIntegrityError):
            self.assets.get(asset["asset_id"])
        image.unlink()
        image.symlink_to(self.papers.root / "papers" / (self.source["paper_id"] + ".pdf"))
        with self.assertRaises(ReplicationIntegrityError):
            self.assets.image(asset)

    def test_missing_renderer_has_actionable_safe_error(self):
        with patch("app.paper_assets.shutil.which", return_value=None):
            with self.assertRaisesRegex(AssetError, "Poppler") as caught:
                self.assets.create(self.source["paper_id"], {"page": 1})
        self.assertEqual(caught.exception.status_code, 503)


class AssetAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_authentication_inventory_actual_image_and_crop_validation(self):
        if not shutil.which("pdftoppm"):
            self.skipTest("Poppler required for actual PDF image tests")
        from fastapi import Depends, Header
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        store = ReplicationStore(Path(temporary.name))
        source = store.put_paper(drawing_pdf())
        app = FastAPI()
        def authorize(x_assistant_token: str = Header(default="")):
            if x_assistant_token != "local-test-token":
                raise HTTPException(401, "Access token required.")
        paper_assets.register_routes(app, [Depends(authorize)], store)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
            url = "/api/papers/" + source["paper_id"] + "/assets"
            self.assertEqual((await client.get(url)).status_code, 401)
            client.headers["X-Assistant-Token"] = "local-test-token"
            inventory = await client.get(url)
            self.assertEqual(inventory.status_code, 200)
            created = await client.post(url, json={"page": 1, "purpose": "table"})
            self.assertEqual(created.status_code, 201)
            asset = created.json()
            path = "/api/paper-assets/" + asset["asset_id"]
            self.assertEqual((await client.get(path)).json(), asset)
            image = await client.get(path + "/image")
            self.assertEqual(image.headers["content-type"], "image/png")
            self.assertEqual(hashlib.sha256(image.content).hexdigest(), asset["png_sha256"])
            self.assertIn(asset["png_sha256"], image.headers["etag"])
            self.assertEqual((await client.post(url, json={"page": 2})).status_code, 422)
            self.assertEqual((await client.get("/api/paper-assets/" + "a" * 64)).status_code, 404)


if __name__ == "__main__":
    unittest.main()
