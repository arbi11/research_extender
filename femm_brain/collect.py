"""Bounded, repeatable discovery of official FEMM documentation and examples.

Downloads are retained in a local cache, not redistributed with the knowledge
graph. This module never executes downloaded scripts or model files.
"""
from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import http.client
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import parse_qs, quote, unquote, urlencode, urljoin, urlsplit
import urllib.error
import urllib.request
import zipfile


AGENT = "FEMM-Brain-Research/0.1 (+source-attribution; bounded research crawler)"
SEEDS = ("examples", "contrib", "documentation")
MAX_BYTES = 25 * 1024 * 1024
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 256
DEFAULT_CACHE = Path(os.getenv("FEMM_BRAIN_CACHE", str(Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "femm-brain")))
KINDS = {".fem": "magnetics_model", ".fee": "electrostatics_model",
         ".feh": "heat_model", ".fec": "current_model", ".lua": "lua_script",
         ".m": "octave_script", ".py": "python_script", ".sce": "scilab_script", ".dxf": "cad_drawing",
         ".zip": "archive", ".pdf": "document", ".nb": "notebook",
         ".mcd": "worksheet", ".txt": "text", ".csv": "data"}
SKIP_IDS = {"start", "sidebar", "syntax", "playground:playground", "wiki:syntax",
            "download", "download5", "faq", "changes", "relatedlinks", "recentchanges"}


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def official(url):
    value = urlsplit(url)
    return value.scheme in {"http", "https"} and value.hostname in {"femm.info", "www.femm.info"} and not value.username


def page_id(url):
    """Return a stable DokuWiki page id, excluding actions and media links."""
    if not official(url):
        return None
    value = urlsplit(url)
    query = parse_qs(value.query, keep_blank_values=True)
    if value.path.startswith("/wiki/"):
        name = unquote(value.path[len("/wiki/"):]).strip("/")
    elif value.path in {"/doku/doku.php", "/doku.php"}:
        if "do" in query or "media" in query:
            return None
        name = query.get("id", [""])[0]
    else:
        return None
    name = name.lower().strip()
    if not name or not re.fullmatch(r"[a-z0-9_:-]+", name):
        return None
    return name


def page_url(name):
    return "https://femm.info/doku/doku.php?" + urlencode({"id": name})


def filename(url):
    value = urlsplit(url)
    media = parse_qs(value.query).get("media", [""])[0]
    return unquote(media.split(":")[-1] if media else value.path.rsplit("/", 1)[-1])


def file_suffix(name):
    # One official contribution uses a .sce.upload suffix to serve source text.
    return Path(name.removesuffix(".upload")).suffix.lower()


def asset_url(url):
    """Accept only known research attachment formats on the official host."""
    if not official(url):
        return None
    value = urlsplit(url)
    suffix = file_suffix(filename(url))
    if suffix not in KINDS:
        return None
    if value.path.startswith(("/doku/lib/exe/fetch.php", "/doku/lib/exe/detail.php", "/lib/exe/fetch.php", "/lib/exe/detail.php")):
        query = parse_qs(value.query)
        if "media" not in query:
            return None
        return "https://femm.info/doku/lib/exe/fetch.php?" + urlencode({"media": query["media"][0]})
    return "https://femm.info" + value.path + ("?" + value.query if value.query else "")


class PageParser(HTMLParser):
    def __init__(self, url):
        super().__init__(convert_charrefs=True)
        self.url, self.links, self.parts = url, [], []
        self.hidden = 0
        self.in_title = False
        self.title_parts = []
        self.anchor = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag == "title":
            self.in_title = True
        if tag == "a" and attrs.get("href"):
            self.anchor = {"url": urljoin(self.url, attrs["href"]), "title": ""}
        if tag in {"p", "h1", "h2", "h3", "h4", "li", "tr", "br", "pre"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        if tag == "title":
            self.in_title = False
        if tag == "a" and self.anchor:
            self.anchor["title"] = re.sub(r"\s+", " ", self.anchor["title"]).strip()
            self.links.append(self.anchor)
            self.anchor = None

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
        if not self.hidden:
            self.parts.append(data)
        if self.anchor:
            self.anchor["title"] += data

    @property
    def text(self):
        return "\n".join(re.sub(r"\s+", " ", line).strip() for line in "".join(self.parts).splitlines() if line.strip())

    @property
    def title(self):
        return "".join(self.title_parts).split("[", 1)[0].strip() or page_id(self.url) or self.url


def fetch(url, *, timeout=15, max_bytes=MAX_BYTES):
    if not official(url):
        raise ValueError("Only official FEMM HTTP(S) URLs are collected")
    request = urllib.request.Request(url, headers={"User-Agent": AGENT,
        "Accept": "text/html,application/pdf,application/octet-stream;q=0.9,*/*;q=0.8"})
    class OfficialRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, message, headers, newurl):
            if not official(newurl):
                raise ValueError("Redirect outside the official FEMM host is not collected")
            return super().redirect_request(request, fp, code, message, headers, newurl)
    # build_opener retains the inherited proxy and system CA configuration.
    opener = urllib.request.build_opener(OfficialRedirect())
    with opener.open(request, timeout=timeout) as response:
        if not official(response.url):
            raise ValueError("Attachment redirected outside the official FEMM host")
        if int(response.headers.get("Content-Length", "0")) > max_bytes:
            raise ValueError("Download exceeds size limit")
        chunks, size = [], 0
        while chunk := response.read(min(65536, max_bytes + 1 - size)):
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes:
                raise ValueError("Download exceeds size limit")
        return b"".join(chunks), response.headers.get_content_type(), response.url


def _cache(data, cache):
    digest = hashlib.sha256(data).hexdigest()
    path = cache / "objects" / digest
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    return digest, str(path.relative_to(cache))


_TASKRUNNER = re.compile(rb"(taskrunner\.php\?id=[^\"&]*)&amp;\d+")


def content_digest(data):
    """Digest page markup without DokuWiki's per-request taskrunner timestamp."""
    return hashlib.sha256(_TASKRUNNER.sub(rb"\1", data)).hexdigest()


_MODIFIED = re.compile(r"Last modified: (\d{4})/(\d{2})/(\d{2}) (\d{2}):(\d{2})")


def last_modified(text):
    """DokuWiki's page footer revision time, as an ISO minute, when present."""
    match = _MODIFIED.search(text)
    return "{}-{}-{}T{}:{}".format(*match.groups()) if match else None


def text_digest(text):
    """Digest of the extracted page text: the content curation depends on."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_page(name, cache, timeout=15):
    url = page_url(name)
    record = {"id": name, "url": url, "retrieved_at": now(), "status": "failed", "links": []}
    try:
        data, media, final_url = fetch(url, timeout=timeout)
        if "html" not in media:
            raise ValueError("Source page did not return HTML")
        digest, key = _cache(data, cache)
        parser = PageParser(final_url)
        parser.feed(data.decode("utf-8", errors="replace"))
        missing = any(phrase in parser.text.lower() for phrase in (
            "this topic does not exist yet", "a topic that doesn't exist yet"))
        text_path = cache / "texts" / (name.replace(":", "_") + ".txt")
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(parser.text, encoding="utf-8")
        record.update(title=parser.title, sha256=digest, content_sha256=content_digest(data),
                      text_sha256=text_digest(parser.text), last_modified=last_modified(parser.text), content_type=media,
                      cache_key=key, text_cache_key=str(text_path.relative_to(cache)),
                      status="missing" if missing else "fetched", links=parser.links)
    except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as error:
        record["error"] = type(error).__name__ + ": " + str(error)[:300]
    return record


def discover(sources):
    """Return deterministic source and attachment URLs from fetched pages."""
    pages, attachments = set(), {}
    for source in sources:
        for link in source.get("links", []):
            if (name := page_id(link["url"])) and name not in SKIP_IDS:
                pages.add(name)
            if url := asset_url(link["url"]):
                attachments.setdefault(url, set()).add(source["id"])
    return sorted(pages), attachments


def _safe_members(archive):
    """Yield research files without extracting arbitrary paths or symlinks."""
    members = archive.infolist()
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise ValueError("Archive contains too many members")
    if sum(member.file_size for member in members) > MAX_ARCHIVE_BYTES:
        raise ValueError("Archive uncompressed size exceeds limit")
    for member in members:
        path = PurePosixPath(member.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]):
            raise ValueError("Unsafe archive member path")
        mode = member.external_attr >> 16
        if mode & 0o170000 == 0o120000:
            raise ValueError("Archive contains a symlink")
        if member.flag_bits & 1:
            raise ValueError("Encrypted archives are not supported")
        if member.is_dir() or file_suffix(str(path)) not in KINDS or file_suffix(str(path)) == ".zip":
            continue
        yield member


def _artifact(data, *, url, sources, name, cache, media="application/octet-stream", parent=None):
    digest, key = _cache(data, cache)
    record = {"id": "asset:" + digest[:24], "source_ids": sorted(sources), "url": url,
              "sha256": digest, "size_bytes": len(data), "media_type": media,
              "kind": KINDS.get(file_suffix(name), "attachment"), "filename": name,
              "cache_key": key, "status": "fetched", "redistribution_rights": "not_established"}
    if parent:
        record["archive_id"], record["archive_member"] = parent, name
    suffix = file_suffix(name)
    if suffix in {".lua", ".m", ".py", ".sce", ".fem", ".fee", ".feh", ".fec", ".dxf", ".txt", ".csv"}:
        text = data.decode("utf-8-sig", errors="replace")
        text_path = cache / "artifact_texts" / (digest + ".txt")
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(text, encoding="utf-8")
        record["text_cache_key"] = str(text_path.relative_to(cache))
        if suffix in {".lua", ".m", ".py", ".sce"}:
            calls = sorted(set(re.findall(r"\b(?:[mhec][io]_[A-Za-z0-9_]+|newdocument|smartmesh)\s*(?=\()", text)))
            record["metadata"] = {"functions": calls,
                                  "version_mentions": sorted(set(re.findall(r"FEMM\s*[45](?:\.[0-9]+)?", text, re.I)))}
        elif suffix in {".fem", ".fee", ".feh", ".fec"}:
            keys = {key: value.strip() for key, value in re.findall(r"^\[([^\]]+)\]\s*=\s*(.*)$", text, re.M)
                    if key in {"Format", "Frequency", "Depth", "LengthUnits", "ProblemType", "NumPoints", "NumSegments", "NumArcSegments", "NumBlockLabels"}}
            record["metadata"] = keys
    elif suffix == ".pdf":
        pdf_path = cache / "objects" / digest
        text_path = cache / "artifact_texts" / (digest + ".txt")
        text_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(["pdftotext", "-layout", str(pdf_path), str(text_path)], capture_output=True, timeout=15)
            if result.returncode == 0:
                record["text_cache_key"] = str(text_path.relative_to(cache))
        except (OSError, subprocess.TimeoutExpired):
            pass
    return record


def collect_attachment(url, source_ids, cache, timeout=15):
    try:
        data, media, _ = fetch(url, timeout=timeout)
        if media == "text/html" and Path(filename(url)).suffix.lower() in KINDS:
            raise ValueError("Attachment returned an HTML page rather than its advertised file")
        if Path(filename(url)).suffix.lower() == ".pdf" and not data.startswith(b"%PDF"):
            raise ValueError("Attachment is not a PDF document")
        main = _artifact(data, url=url, sources=source_ids, name=filename(url), cache=cache, media=media)
        records = [main]
        if Path(filename(url)).suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    selected_members = list(_safe_members(archive))
                    main["metadata"] = {"archive_member_count": len(archive.infolist()),
                        "research_member_count": len(selected_members),
                        "uncompressed_size_bytes": sum(member.file_size for member in archive.infolist()),
                        "extraction_policy": "Known research formats only; no code execution or recursive archive extraction."}
                    for member in selected_members:
                        records.append(_artifact(archive.read(member), url=url, sources=source_ids,
                            name=member.filename, cache=cache, parent=main["id"]))
            except (ValueError, zipfile.BadZipFile, RuntimeError, OSError) as error:
                main["archive_error"] = str(error)[:300]
        return records
    except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as error:
        return [{"id": "failed:" + hashlib.sha256(url.encode()).hexdigest()[:24],
                 "url": url, "source_ids": sorted(source_ids), "filename": filename(url),
                 "status": "failed", "error": type(error).__name__ + ": " + str(error)[:300]}]


def crawl(*, cache=DEFAULT_CACHE, seeds=SEEDS, max_pages=100, max_attachments=160,
          depth=2, workers=4, timeout=15, download=True):
    """Collect a bounded official corpus with visible failures and coverage."""
    if not 1 <= max_pages <= 250 or not 0 <= max_attachments <= 400 or not 0 <= depth <= 4:
        raise ValueError("Use 1–250 pages, 0–400 attachments and depth 0–4")
    if not 1 <= timeout <= 60 or not 1 <= workers <= 4:
        raise ValueError("Use a timeout of 1–60 seconds and 1–4 workers")
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    started, sources, seen = now(), [], set()
    pending = sorted(set(seeds))
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max(1, workers), 4)) as pool:
        for level in range(depth + 1):
            current = [name for name in pending if name not in seen][:max_pages - len(seen)]
            if not current:
                break
            seen.update(current)
            sources.extend(pool.map(lambda name: collect_page(name, cache, timeout), current))
            pending, _ = discover(sources)
        _, attachments = discover(sources)
        manual_url = "https://femm.info/doku/lib/exe/fetch.php?media=femm-qt%3Adocs%3Amanual.pdf"
        attachments.setdefault(manual_url, set()).add("manual5")
        # The function catalog has a stable source identity distinct from the
        # documentation index that happens to link to the downloadable manual.
        sources.append({"id": "manual5", "url": manual_url, "title": "FEMM 5 User's Manual",
                        "retrieved_at": now(), "status": "indexed", "content_type": "application/pdf", "links": []})
        selected = sorted(attachments)[:max_attachments]
        # Always include the primary manual even when an attachment limit applies.
        if manual_url not in selected and max_attachments:
            selected = selected[:-1] + [manual_url]
        artifacts = []
        if download:
            for records in pool.map(lambda url: collect_attachment(url, attachments[url], cache, timeout), selected):
                artifacts.extend(records)
        # Merge identical bytes reached by multiple examples while preserving
        # source evidence and all observed download locations.
        merged = {}
        for item in artifacts:
            identity = item["id"]
            if identity in merged:
                previous = merged[identity]
                previous["source_ids"] = sorted(set(previous["source_ids"]) | set(item["source_ids"]))
                previous["alternate_urls"] = sorted(set(previous.get("alternate_urls", [])) | {item["url"]})
                previous.setdefault("locations", [{key: previous[key] for key in ("url", "filename", "archive_id", "archive_member") if key in previous}])
                location = {key: item[key] for key in ("url", "filename", "archive_id", "archive_member") if key in item}
                if location not in previous["locations"]:
                    previous["locations"].append(location)
            else:
                merged[identity] = item
        artifacts = sorted(merged.values(), key=lambda item: item["id"])
    for artifact in artifacts:
        if "manual5" in artifact.get("source_ids", []) and artifact.get("status") == "fetched":
            sources[-1].update({key: artifact[key] for key in ("sha256", "cache_key", "size_bytes")})
            sources[-1]["status"] = "fetched"
            if artifact.get("text_cache_key"):
                sources[-1]["text_cache_key"] = artifact["text_cache_key"]
    return {"schema_version": 1, "sources": sorted(sources, key=lambda source: source["id"]),
            "artifacts": artifacts, "crawl": {"started_at": started, "finished_at": now(),
            "seeds": list(seeds), "max_pages": max_pages, "max_attachments": max_attachments,
            "max_depth": depth, "discovered_attachment_count": len(attachments),
            "selected_attachment_count": len(selected), "download_enabled": download,
            "source_fetched_count": sum(s["status"] == "fetched" for s in sources),
            "source_failed_count": sum(s["status"] == "failed" for s in sources),
            "source_missing_count": sum(s["status"] == "missing" for s in sources),
            "artifact_fetched_count": sum(a["status"] == "fetched" for a in artifacts),
            "artifact_failed_count": sum(a["status"] == "failed" for a in artifacts),
            "note": "Discovery is bounded; indexed/fetched does not establish construction or numerical verification."}}


def restore(manifest, *, cache=DEFAULT_CACHE, workers=4, timeout=30, only=None):
    """Re-fetch a recorded snapshot into the cache, accepting only identical bytes.

    The manifest is never rewritten. Pages or attachments whose current bytes
    differ from the recorded digest are reported as changed and their new bytes
    are not linked to the recorded identity. ``only`` limits the restore to the
    named attachment digests (pages are then skipped).
    """
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    artifacts = [a for a in manifest["artifacts"] if a.get("status") == "fetched"]
    if only is not None:
        wanted = set(only)
        parents = {a["archive_id"] for a in artifacts if a["sha256"] in wanted and a.get("archive_id")}
        artifacts = [a for a in artifacts if a["sha256"] in wanted or a["id"] in parents]
        unknown = wanted - {a["sha256"] for a in artifacts}
        if unknown:
            raise ValueError("Requested digests are not in the manifest: " + ", ".join(sorted(unknown)))
    members = {}
    for artifact in artifacts:
        if artifact.get("archive_id"):
            members.setdefault(artifact["archive_id"], []).append(artifact)
    top = [a for a in artifacts if not a.get("archive_id")]
    pages = [] if only is not None else [s for s in manifest["sources"] if s.get("status") == "fetched" and s.get("sha256")
             and not any(s["id"] in a.get("source_ids", []) and a.get("sha256") == s["sha256"] for a in top)]

    def restore_page(source):
        result = {"id": source["id"], "url": source["url"], "expected_sha256": source["sha256"]}
        try:
            data, media, final_url = fetch(source["url"], timeout=timeout)
            digest = hashlib.sha256(data).hexdigest()
            result.update(observed_sha256=digest, observed_content_sha256=content_digest(data), retrieved_at=now())
            if source.get("content_sha256"):
                same = result["observed_content_sha256"] == source["content_sha256"]
            elif digest == source["sha256"]:
                same = True
            elif _TASKRUNNER.search(data):
                # Snapshots recorded before content_sha256 hashed the timestamped
                # markup, so their identity cannot be reproduced byte for byte.
                same = None
            else:
                same = False
            if same is False:
                result["status"] = "changed"
                return result
            _cache(data, cache)
            parser = PageParser(final_url)
            parser.feed(data.decode("utf-8", errors="replace"))
            text_path = cache / source["text_cache_key"]
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(parser.text, encoding="utf-8")
            result.update(observed_text_sha256=text_digest(parser.text), last_modified=last_modified(parser.text))
            result["status"] = "restored" if same else "current_text_unverifiable_snapshot"
        except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as error:
            result.update(status="failed", error=type(error).__name__ + ": " + str(error)[:300])
        return result

    def restore_artifact(artifact):
        results = []
        try:
            cached = cache / "objects" / artifact["sha256"]
            if cached.is_file() and hashlib.sha256(cached.read_bytes()).hexdigest() == artifact["sha256"]:
                data, media = cached.read_bytes(), artifact.get("media_type", "application/octet-stream")
            else:
                data, media, _ = fetch(artifact["url"], timeout=timeout)
            digest = hashlib.sha256(data).hexdigest()
            if digest != artifact["sha256"]:
                return [{"id": artifact["id"], "url": artifact["url"], "status": "changed",
                         "expected_sha256": artifact["sha256"], "observed_sha256": digest}]
            _artifact(data, url=artifact["url"], sources=artifact["source_ids"], name=artifact["filename"],
                      cache=cache, media=media)
            results.append({"id": artifact["id"], "url": artifact["url"], "status": "restored"})
            if members.get(artifact["id"]):
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    available = {member.filename: member for member in _safe_members(archive)}
                    for record in members[artifact["id"]]:
                        if only is not None and record["sha256"] not in set(only):
                            continue
                        member = available.get(record["archive_member"])
                        if member is None:
                            results.append({"id": record["id"], "status": "missing_member", "archive_member": record["archive_member"]})
                            continue
                        content = archive.read(member)
                        observed = hashlib.sha256(content).hexdigest()
                        if observed != record["sha256"]:
                            results.append({"id": record["id"], "status": "changed", "expected_sha256": record["sha256"], "observed_sha256": observed})
                            continue
                        _artifact(content, url=artifact["url"], sources=record["source_ids"], name=record["archive_member"],
                                  cache=cache, parent=artifact["id"])
                        results.append({"id": record["id"], "status": "restored", "archive_member": record["archive_member"]})
        except (OSError, ValueError, zipfile.BadZipFile, urllib.error.URLError, http.client.HTTPException) as error:
            results.append({"id": artifact["id"], "url": artifact["url"], "status": "failed",
                            "error": type(error).__name__ + ": " + str(error)[:300]})
        return results

    started = now()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max(1, workers), 4)) as pool:
        page_results = list(pool.map(restore_page, pages))
        artifact_results = [item for records in pool.map(restore_artifact, top) for item in records]
    counts = lambda items: {status: sum(item["status"] == status for item in items)
                            for status in sorted({item["status"] for item in items})}
    return {"schema_version": 1, "started_at": started, "finished_at": now(), "cache": str(cache),
            "expected": {"pages": len(pages), "artifacts": len(artifacts)},
            "pages": counts(page_results), "artifacts": counts(artifact_results),
            "page_results": page_results, "artifact_results": artifact_results,
            "note": "Only bytes identical to the recorded digest are restored; the source manifest is unchanged."}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data/source_manifest.json")
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--max-attachments", type=int, default=160)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--pages-only", action="store_true")
    args = parser.parse_args()
    result = crawl(cache=args.cache, max_pages=args.max_pages, max_attachments=args.max_attachments,
                   depth=args.depth, download=not args.pages_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result["crawl"], indent=2))
