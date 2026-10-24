"""Immutable, local PDF page and explicitly selected figure/table crops.

Candidates are pages, with textual caption hints. A hint is never presented as
an automatically located figure/table bounding box. Coordinates are PDF points
from the top-left of the displayed, rotated CropBox; originals remain intact.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Literal

from fastapi import HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import papers
from .replication import ReplicationIntegrityError, _encode, _write_once


ID_PATTERN = r"^[a-f0-9]{64}$"
CASE_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]{0,79}$"
MAX_PNG_BYTES = 4 * 1024 * 1024
MAX_ASSET_RECORD_BYTES = 32 * 1024
MAX_ASSETS = 1600
RENDER_LONG_EDGE = 1568
RENDER_TIMEOUT_SECONDS = 45
COORDINATE_SYSTEM = "PDF points from top-left of displayed rotated CropBox"


class AssetError(ValueError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class CropBox(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(gt=0)
    y1: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("A crop must have positive width and height.")
        return self


class AssetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    page: int = Field(ge=1, le=papers.MAX_PAGES, strict=True)
    bbox: CropBox | None = None
    purpose: Literal["page", "figure", "table"] = "page"
    case_id: str = Field(default="baseline", pattern=CASE_PATTERN)
    label: str = Field(default="", max_length=180)


class PaperAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    version: Literal[1] = 1
    paper_id: str = Field(pattern=ID_PATTERN)
    source_id: str = Field(pattern=r"^paper_[a-f0-9]{24}$")
    page: int = Field(ge=1, le=papers.MAX_PAGES, strict=True)
    bbox: CropBox
    page_width_pt: float = Field(gt=0, le=20000)
    page_height_pt: float = Field(gt=0, le=20000)
    page_rotation: Literal[0, 90, 180, 270]
    coordinate_system: Literal[COORDINATE_SYSTEM] = COORDINATE_SYSTEM
    purpose: Literal["page", "figure", "table"]
    case_id: str = Field(pattern=CASE_PATTERN)
    label: str = Field(max_length=180)
    png_sha256: str = Field(pattern=ID_PATTERN)
    width_px: int = Field(ge=1, le=RENDER_LONG_EDGE, strict=True)
    height_px: int = Field(ge=1, le=RENDER_LONG_EDGE, strict=True)
    render_engine: Literal["poppler-pdftoppm"] = "poppler-pdftoppm"
    render_long_edge: Literal[RENDER_LONG_EDGE] = RENDER_LONG_EDGE

    @model_validator(mode="after")
    def bounded_source(self):
        if self.source_id != "paper_" + self.paper_id[:24]:
            raise ValueError("Asset source id must match its original PDF hash.")
        if self.bbox.x1 > self.page_width_pt or self.bbox.y1 > self.page_height_pt:
            raise ValueError("The asset crop exceeds its source page.")
        return self


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _png_size(data: bytes) -> tuple[int, int]:
    if not isinstance(data, bytes) or len(data) > MAX_PNG_BYTES or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssetError(422, "The local PDF renderer returned an invalid or oversized PNG.")
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise ValueError
            width, height = image.size
            if not 1 <= width <= RENDER_LONG_EDGE or not 1 <= height <= RENDER_LONG_EDGE:
                raise ValueError
            image.verify()
    except Exception:
        raise AssetError(422, "The local PDF renderer returned an invalid PNG.") from None
    return width, height


def _page_info(pdf: bytes) -> list[dict]:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf), strict=True)
        result = []
        for page in reader.pages:
            user_unit = float(page.user_unit)
            width, height = float(page.cropbox.width) * user_unit, float(page.cropbox.height) * user_unit
            rotation = int(page.rotation or 0) % 360
            if rotation not in {0, 90, 180, 270}:
                raise ValueError
            if rotation in {90, 270}:
                width, height = height, width
            if not all(math.isfinite(x) and 0 < x <= 20000 for x in (width, height)):
                raise ValueError
            # Embedded image count is descriptive only: vector drawings and
            # tables need not be embedded raster images, so no figure is inferred.
            resources = page.get("/Resources", {})
            resources = resources.get_object() if hasattr(resources, "get_object") else resources
            objects = resources.get("/XObject", {})
            objects = objects.get_object() if hasattr(objects, "get_object") else objects
            images = sum(1 for obj in objects.values() if obj.get_object().get("/Subtype") == "/Image")
            result.append({"width_pt": width, "height_pt": height, "rotation": rotation,
                           "embedded_image_count": images})
        return result
    except Exception:
        raise AssetError(422, "The stored PDF page dimensions could not be read safely.") from None


def _render_page(pdf_path: Path, page: int) -> bytes:
    executable = shutil.which("pdftoppm")
    if executable is None:
        raise AssetError(503, "PDF page rendering needs Poppler pdftoppm installed on the server.")
    try:
        with tempfile.TemporaryDirectory(prefix="femm-paper-render-") as temporary:
            prefix = Path(temporary) / "page"
            # No network URLs, shell, PDF scripts or inherited provider secrets.
            environment = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "LC_ALL", "FONTCONFIG_PATH"}}
            subprocess.run([executable, "-f", str(page), "-l", str(page), "-singlefile", "-cropbox",
                            "-scale-to", str(RENDER_LONG_EDGE), "-png", str(pdf_path), str(prefix)],
                           check=True, timeout=RENDER_TIMEOUT_SECONDS, stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=environment)
            output = prefix.with_suffix(".png")
            if not output.is_file() or output.stat().st_size > MAX_PNG_BYTES:
                raise AssetError(422, "The rendered PDF page exceeds the image limit.")
            data = output.read_bytes()
            _png_size(data)
            return data
    except AssetError:
        raise
    except subprocess.TimeoutExpired:
        raise AssetError(422, "The local PDF page renderer exceeded its processing limit.") from None
    except (OSError, subprocess.SubprocessError):
        raise AssetError(422, "The selected PDF page could not be rendered.") from None


def _crop_png(data: bytes, bbox: CropBox, width_pt: float, height_pt: float) -> bytes:
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        box = (math.floor(bbox.x0 * width / width_pt), math.floor(bbox.y0 * height / height_pt),
               math.ceil(bbox.x1 * width / width_pt), math.ceil(bbox.y1 * height / height_pt))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise AssetError(422, "The selected crop is smaller than a rendered pixel.")
        output = io.BytesIO()
        image.crop(box).convert("RGB").save(output, format="PNG", optimize=True)
    result = output.getvalue()
    _png_size(result)
    return result


class PaperAssetStore:
    def __init__(self, paper_store):
        self.paper_store = paper_store
        self.root = paper_store.root / "assets"
        if self.root.is_symlink():
            raise ReplicationIntegrityError("Paper asset storage must be a regular directory.")
        for name in ("records", "images"):
            directory = self.root / name
            directory.mkdir(parents=True, exist_ok=True)
            if directory.is_symlink():
                raise ReplicationIntegrityError("Paper asset directories must be regular directories.")

    def _source(self, paper_id):
        document = self.paper_store.get_document(paper_id)
        if document is None:
            raise AssetError(404, "Uploaded paper not found.")
        path = self.paper_store.root / "papers" / (paper_id + ".pdf")
        data = path.read_bytes()
        if _sha(data) != paper_id:
            raise ReplicationIntegrityError("The asset source PDF no longer matches its hash.")
        return document, path, data

    def inventory(self, paper_id: str) -> dict:
        document, _, data = self._source(paper_id)
        info = _page_info(data)
        pages = []
        for page, dimensions in zip(document.pages, info):
            hints = re.findall(r"\b(?:Fig(?:ure)?\.?|Table)\s+[A-Za-z0-9]+[^\n]{0,160}", page.text, re.I)[:16]
            candidate = {"paper_id": paper_id, "page": page.number, "kind": "page", **dimensions}
            pages.append({"candidate_id": _sha(_encode(candidate)), **candidate,
                          "caption_hints": hints, "automatic_figure_bounds": False})
        return {"paper_id": paper_id, "source_id": document.source_id, "coordinate_system": COORDINATE_SYSTEM,
                "candidates": pages, "assets": self.list_assets(paper_id),
                "selection_policy": "Candidates are complete PDF pages. Caption hints are text only; select and crop a figure/table explicitly. Crops retain PDF hash, page and exact bounds."}

    def create(self, paper_id: str, request: AssetRequest | dict) -> dict:
        request = AssetRequest.model_validate(request)
        document, path, data = self._source(paper_id)
        if request.page > document.page_count:
            raise AssetError(422, "The selected page is outside this PDF.")
        dimensions = _page_info(data)[request.page - 1]
        width, height = dimensions["width_pt"], dimensions["height_pt"]
        bbox = request.bbox or CropBox(x0=0, y0=0, x1=width, y1=height)
        if bbox.x1 > width or bbox.y1 > height:
            raise AssetError(422, "The selected crop extends beyond the displayed PDF page.")
        if sum(1 for _ in (self.root / "records").glob("*.json")) >= MAX_ASSETS:
            raise AssetError(409, "The stored paper asset limit has been reached.")
        png = _crop_png(_render_page(path, request.page), bbox, width, height)
        image_width, image_height = _png_size(png)
        record = PaperAsset(paper_id=paper_id, source_id=document.source_id, page=request.page,
            bbox=bbox, page_width_pt=width, page_height_pt=height, page_rotation=dimensions["rotation"],
            purpose=request.purpose, case_id=request.case_id, label=request.label,
            png_sha256=_sha(png), width_px=image_width, height_px=image_height).model_dump(mode="json")
        encoded = _encode(record)
        identity = _sha(encoded)
        _write_once(self.root / "images" / (record["png_sha256"] + ".png"), png)
        _write_once(self.root / "records" / (identity + ".json"), encoded)
        return {"asset_id": identity, **record}

    def get(self, asset_id: str) -> dict | None:
        if not isinstance(asset_id, str) or not re.fullmatch(ID_PATTERN, asset_id):
            return None
        path = self.root / "records" / (asset_id + ".json")
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_ASSET_RECORD_BYTES:
            raise ReplicationIntegrityError("The paper asset record has an invalid file type or size.")
        data = path.read_bytes()
        if _sha(data) != asset_id:
            raise ReplicationIntegrityError("The paper asset no longer matches its content hash.")
        try:
            record = PaperAsset.model_validate_json(data).model_dump(mode="json")
        except ValueError:
            raise ReplicationIntegrityError("The stored paper asset is invalid.") from None
        self._source(record["paper_id"])
        self.image({"asset_id": asset_id, **record})
        return {"asset_id": asset_id, **record}

    def image(self, record: dict) -> bytes:
        record = PaperAsset.model_validate({k: v for k, v in record.items() if k != "asset_id"})
        path = self.root / "images" / (record.png_sha256 + ".png")
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_PNG_BYTES:
            raise ReplicationIntegrityError("The paper asset image is missing or invalid.")
        data = path.read_bytes()
        if _sha(data) != record.png_sha256 or _png_size(data) != (record.width_px, record.height_px):
            raise ReplicationIntegrityError("The paper asset image no longer matches its provenance.")
        return data

    def list_assets(self, paper_id: str) -> list[dict]:
        result = []
        for index, path in enumerate(sorted((self.root / "records").glob("*.json"))):
            if index >= MAX_ASSETS:
                raise AssetError(409, "The stored paper asset inventory exceeds its limit.")
            record = self.get(path.stem)
            if record and record["paper_id"] == paper_id:
                result.append(record)
        return result


def register_routes(app, secured, paper_store):
    store = PaperAssetStore(paper_store)
    app.state.paper_assets = store

    async def call(function, *args):
        try:
            return await asyncio.to_thread(function, *args)
        except AssetError as error:
            raise HTTPException(error.status_code, error.message) from None
        except ReplicationIntegrityError as error:
            raise HTTPException(409, str(error)) from None

    @app.get("/api/papers/{paper_id}/assets", dependencies=secured)
    async def inventory(paper_id: str):
        return await call(store.inventory, paper_id)

    @app.post("/api/papers/{paper_id}/assets", dependencies=secured, status_code=201)
    async def create(paper_id: str, request: AssetRequest):
        return await call(store.create, paper_id, request)

    @app.get("/api/paper-assets/{asset_id}", dependencies=secured)
    async def get(asset_id: str):
        record = await call(store.get, asset_id)
        if record is None:
            raise HTTPException(404, "Paper asset not found.")
        return record

    @app.get("/api/paper-assets/{asset_id}/image", dependencies=secured)
    async def image(asset_id: str):
        record = await call(store.get, asset_id)
        if record is None:
            raise HTTPException(404, "Paper asset not found.")
        data = await call(store.image, record)
        return Response(data, media_type="image/png", headers={"ETag": '"' + record["png_sha256"] + '"',
                                                             "Cache-Control": "private, immutable"})
    return store
