"""Fetch pinned documentation and locate API definitions; never execute sources."""
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import unicodedata
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DATA = Path(__file__).with_name("data")
DEFAULT_CACHE = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "femm-brain2/sources"
ALLOWED_HOSTS = {"femm.info", "www.femm.info", "raw.githubusercontent.com"}


def fingerprint(value):
    return hashlib.sha256(value).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).replace("\u00ad", "").split())


@dataclass(frozen=True)
class Definition:
    name: str
    pages: tuple
    section: str
    text: str
    additional_pages: tuple = ()
    line_range: tuple = ()

    @property
    def digest(self):
        return fingerprint(normalized(self.text).encode())


def implementation_definitions(text, source):
    """Locate reviewed C++ excerpts in a hash-pinned file, using one-based lines."""
    lines = text.splitlines()
    result = {}
    for name, (first, last) in source.get("anchors", {}).items():
        if not 1 <= first <= last <= len(lines):
            raise ValueError("Invalid implementation locator: " + name)
        result[name] = Definition(name, (), "C++ implementation", "\n".join(lines[first-1:last]),
                                  line_range=(first, last))
    return result


def manual_definitions(text):
    """Find bullet definitions across PDF page boundaries, retaining their source."""
    chunks, spans, offset = [], [], 0
    for number, page in enumerate(text.split("\f"), 1):
        if not page.strip():
            continue
        # Page numbers are layout, not part of the command definition.
        page = re.sub(r"(?m)^\s*\d+\s*$", "", page).strip() + "\n"
        chunks.append(page)
        spans.append((offset, offset + len(page), number))
        offset += len(page)
    document = "".join(chunks)
    headings = list(re.finditer(r"(?m)^\s*(3\.\d+(?:\.\d+)?)\s+([^\n]+)", document))
    candidates = list(re.finditer(r"(?m)^[ \t]*•[ \t]+([a-z][a-z0-9_]*)\b", document))
    definitions = {}
    for index, match in enumerate(candidates):
        name = match.group(1)
        prior = [h for h in headings if h.start() <= match.start()]
        heading = prior[-1] if prior else None
        section = normalized(heading.group(0)) if heading else ""
        if not (re.match(r"^(mi|mo|ei|eo|hi|ho|ci|co)_", name) or
                (heading and heading.group(1) == "3.5" and document[match.end():].lstrip().startswith("("))):
            continue
        end = candidates[index + 1].start() if index + 1 < len(candidates) else len(document)
        next_heading = next((h for h in headings if match.start() < h.start() < end), None)
        if next_heading:
            end = next_heading.start()
        body = document[match.start():end].strip()
        pages = tuple(n for start, stop, n in spans if start < end and stop > match.start())
        if name in definitions:
            first = definitions[name]
            definitions[name] = replace(first, additional_pages=first.additional_pages + (pages,))
            continue
        definitions[name] = Definition(name, pages, section, body)
    return definitions


def lua_definitions(text):
    """Named binding functions, including helpers; callers select public APIs."""
    matches = list(re.finditer(r"(?m)^function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text))
    result = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result[match.group(1)] = Definition(match.group(1), (), "Lua binding function", text[match.start():end].strip())
    return result


def load_sources(path=None):
    return json.loads(Path(path or DATA / "sources.json").read_text())["sources"]


def source_file(source, cache=DEFAULT_CACHE):
    return Path(cache) / source["sha256"] / source["filename"]


def fetch_sources(sources, cache=DEFAULT_CACHE):
    fetched = []
    for source in sources:
        target = source_file(source, cache)
        if target.is_file() and fingerprint(target.read_bytes()) == source["sha256"]:
            fetched.append({"id": source["id"], "status": "cached", "path": str(target)})
            continue
        address = urlsplit(source["url"])
        if address.scheme != "https" or address.hostname not in ALLOWED_HOSTS:
            raise ValueError("Source URL is outside the approved official-documentation hosts")
        with urlopen(Request(source["url"], headers={"User-Agent": "femm-brain2-catalogue/0.1"}), timeout=30) as response:
            redirected = urlsplit(response.url)
            if redirected.scheme != "https" or redirected.hostname not in ALLOWED_HOSTS:
                raise ValueError("Source redirected outside documentation hosts")
            content = response.read(25 * 1024 * 1024 + 1)
        if len(content) > 25 * 1024 * 1024 or fingerprint(content) != source["sha256"]:
            raise ValueError("Source changed; review a new snapshot rather than replacing the pin: " + source["id"])
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".pending")
        temporary.write_bytes(content)
        temporary.replace(target)
        fetched.append({"id": source["id"], "status": "downloaded", "path": str(target)})
    return fetched


def verified_text(source, cache=DEFAULT_CACHE):
    path = source_file(source, cache)
    if not path.is_file():
        raise ValueError("Missing source; run fetch-sources: " + str(path))
    if fingerprint(path.read_bytes()) != source["sha256"]:
        raise ValueError("Cached source hash mismatch: " + source["id"])
    if source["format"] == "pdf":
        if not path.read_bytes().startswith(b"%PDF-"):
            raise ValueError("Expected a PDF: " + source["id"])
        try:
            return subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True,
                                  check=True, text=True, timeout=60).stdout
        except FileNotFoundError as exc:
            raise ValueError("Source verification requires pdftotext (Poppler)") from exc
    return path.read_text()
