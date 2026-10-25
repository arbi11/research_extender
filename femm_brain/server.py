"""Local read-only graph browser and bounded geometry-construction API."""
from __future__ import annotations

from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .core import DATA, example_subgraph
from .geometry import construct, export_recipe, recipe_catalog

STATIC = Path(__file__).parent / "static"


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, graph=None, store=None, **kwargs):
        self.graph = graph
        self.store = store
        super().__init__(*args, **kwargs)

    def _send(self, content, status=200, media="application/json; charset=utf-8", filename=None):
        if not isinstance(content, bytes):
            content = (json.dumps(content, ensure_ascii=False, allow_nan=False) if media.startswith("application/json") else str(content)).encode()
        self.send_response(status)
        self.send_header("Content-Type", media)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        if filename:
            self.send_header("Content-Disposition", 'attachment; filename="' + filename + '"')
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if len(self.path) > 32000:
            self._send({"error": "Request URL is too long."}, 414)
            return
        parts = urlsplit(self.path)
        path, query = unquote(parts.path), parse_qs(parts.query)
        first = lambda key, default="": query.get(key, [default])[0]
        try:
            if path == "/api/graph":
                self._send(self.graph)
            elif path == "/api/stats":
                self._send(self.graph["stats"])
            elif path == "/api/verification":
                report = DATA / "verification/native_geometry.json"
                self._send({"graph": self.graph["validation"], "native": json.loads(report.read_text()) if report.is_file() else None})
            elif path == "/api/recipes":
                self._send({"recipes": recipe_catalog()})
            elif path == "/api/sources":
                self._send({"sources": self.graph["sources"], "artifacts": self.graph["artifacts"], "crawl": self.graph.get("crawl", {}), "coverage": self.graph["stats"]})
            elif path == "/api/search":
                self._send({"results": self.store.search(first("q"), limit=first("limit", "50"))})
            elif path.startswith("/api/node/"):
                identifier = path[len("/api/node/"):]
                node = next((n for n in self.graph["nodes"] if n["id"] == identifier), None)
                if node is None:
                    self._send({"error": "Unknown node."}, 404)
                    return
                self._send({"node": node, "incoming": [e for e in self.graph["edges"] if e["target"] == identifier],
                            "outgoing": [e for e in self.graph["edges"] if e["source"] == identifier],
                            "examples": [e for e in self.graph["examples"] if identifier in e.get("node_ids", [])],
                            "sources": [s for s in self.graph["sources"] if s["id"] in node.get("source_ids", [])]})
            elif path.startswith("/api/examples/"):
                result = example_subgraph(self.graph, path[len("/api/examples/"):])
                self._send(result or {"error": "Unknown example."}, 200 if result else 404)
            elif path in {"/api/geometry", "/api/download"}:
                artifact_id = first("artifact")
                if artifact_id:
                    artifact = next((a for a in self.graph["artifacts"] if a["id"] == artifact_id and a.get("preview_available")), None)
                    if not artifact:
                        self._send({"error": "No bundled geometry for that source artifact."}, 404)
                        return
                    model_path = (DATA / artifact["geometry_model_file"]).resolve()
                    if not model_path.is_relative_to((DATA / "models").resolve()):
                        raise ValueError("Invalid geometry artifact path.")
                    result = json.loads(model_path.read_text())
                    self._send(result)
                    return
                recipe = first("recipe", "stator_tooth")
                parameters = json.loads(first("parameters", "{}"))
                if not isinstance(parameters, dict):
                    raise ValueError("Parameters must be a JSON object.")
                if path == "/api/geometry":
                    self._send(construct(recipe, parameters))
                else:
                    format_ = first("format", "python")
                    if format_ not in {"python", "lua", "fem", "svg"}:
                        raise ValueError("Choose python, lua, fem or svg export.")
                    extension = {"python": "py", "lua": "lua", "fem": "fem", "svg": "svg"}[format_]
                    self._send(export_recipe(recipe, parameters, format_), media="image/svg+xml" if format_ == "svg" else "text/plain; charset=utf-8", filename=recipe + "." + extension)
            elif path == "/healthz":
                self._send({"status": "ok", "graph_validation": self.graph["validation"]["status"]})
            elif path.startswith("/api/"):
                self._send({"error": "Unknown API route."}, 404)
            else:
                relative = "index.html" if path == "/" else path.lstrip("/")
                destination = (STATIC / relative).resolve()
                if not destination.is_relative_to(STATIC.resolve()) or not destination.is_file():
                    self._send({"error": "Not found."}, 404)
                    return
                self._send(destination.read_bytes(), media=mimetypes.guess_type(str(destination))[0] or "application/octet-stream")
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            self._send({"error": str(exc)[:1000]}, 400)
        except (OSError, json.JSONDecodeError):
            self._send({"error": "A bundled resource could not be read."}, 500)

    def log_message(self, format_, *args):
        pass


def serve(graph, store, host="127.0.0.1", port=8010):
    server = ThreadingHTTPServer((host, port), partial(Handler, graph=graph, store=store))
    print(f"FEMM Second Brain: http://{host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
