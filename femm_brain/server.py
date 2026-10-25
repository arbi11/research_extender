"""Local read-only graph browser and bounded geometry-construction API."""
from __future__ import annotations

from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import ipaddress
from urllib.parse import parse_qs, unquote, urlsplit

from .core import DATA, example_subgraph
from .geometry import construct, export_recipe, recipe_catalog

STATIC = Path(__file__).parent / "static"


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, graph=None, store=None, solver=None, **kwargs):
        self.graph = graph
        self.store = store
        self.solver = solver
        super().__init__(*args, **kwargs)

    def _send(self, content, status=200, media="application/json; charset=utf-8", filename=None):
        if not isinstance(content, bytes):
            content = (json.dumps(content, ensure_ascii=False, allow_nan=False) if media.startswith("application/json") else str(content)).encode()
        self.send_response(status)
        self.send_header("Content-Type", media)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
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
            elif path == "/api/simulations":
                from .simulation import simulation_catalog
                self._send({"cases": simulation_catalog(self.graph), "solver_enabled": self.solver is not None,
                            "llm_calls": 0})
            elif path == "/api/simulation-request":
                from .planner import plan_request
                self._send(plan_request(first("q"), self.graph))
            elif path in {"/api/simulation-plan", "/api/simulation-download"}:
                from .simulation import plan_simulation
                plan = plan_simulation(first("case"), json.loads(first("parameters", "{}")), self.graph)
                if first("plan_sha256") and first("plan_sha256") != plan["plan_sha256"]:
                    raise ValueError("The displayed frozen plan has changed; regenerate before downloading")
                if path.endswith("-plan"):
                    self._send(plan)
                else:
                    fmt = first("format", "lua")
                    if fmt not in {"lua", "json", "svg"}:
                        raise ValueError("Choose lua, json or svg export")
                    content = plan["lua"] if fmt == "lua" else plan["preview_svg"] if fmt == "svg" else plan
                    self._send(content, media="application/json" if fmt == "json" else "image/svg+xml" if fmt == "svg" else "text/plain; charset=utf-8",
                               filename=plan["case_id"] + "." + fmt)
            elif path.startswith("/api/simulation-jobs/"):
                identifier = path.removeprefix("/api/simulation-jobs/")
                job = self.solver.get(identifier) if self.solver else None
                self._send(job or {"error": "Unknown local study job"}, 200 if job else 404)
            elif path.startswith("/api/simulation-artifacts/"):
                pieces = path.removeprefix("/api/simulation-artifacts/").split("/")
                artifact = self.solver.artifact(*pieces) if self.solver and len(pieces) == 2 else None
                if artifact is None:
                    self._send({"error": "Study artifact not found"}, 404)
                else:
                    media = mimetypes.guess_type(artifact.name)[0] or "application/octet-stream"
                    self._send(artifact.read_bytes(), media=media, filename=pieces[1])
            elif path == "/api/stats":
                self._send(self.graph["stats"])
            elif path == "/api/verification":
                report = DATA / "verification/native_geometry.json"
                simulations = DATA / "verification/native_simulations.json"
                self._send({"graph": self.graph["validation"], "native": json.loads(report.read_text()) if report.is_file() else None,
                            "native_simulations": json.loads(simulations.read_text()) if simulations.is_file() else None})
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

    def do_POST(self):
        if urlsplit(self.path).path != "/api/simulations":
            self._send({"error": "Unknown API route"}, 404)
            return
        host = self.headers.get("Host", "")
        try:
            parsed = urlsplit("http://" + host)
            if parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.port != self.server.server_address[1]:
                self._send({"error": "Solving requires a local loopback request"}, 403)
                return
            if self.headers.get("Origin") not in {None, "http://" + host}:
                self._send({"error": "Cross-origin solve requests are not allowed"}, 403)
                return
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                self._send({"error": "Send application/json"}, 415)
                return
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 16384:
                raise ValueError("A bounded JSON request body is required")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict) or set(body) - {"case_id", "parameters", "plan_sha256"}:
                raise ValueError("Send case_id, optional typed parameters and frozen plan hash only")
            if self.solver is None:
                self._send({"error": "Native solving is disabled. Start with --enable-solver and a FEMM Docker image."}, 503)
                return
            self._send(self.solver.submit(body["case_id"], body.get("parameters", {}),
                                         expected_plan_sha256=body.get("plan_sha256")), 202)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self._send({"error": str(exc)[:1000]}, 400)


def serve(graph, store, host="127.0.0.1", port=8010, solver=None):
    if solver is not None and host != "localhost" and not ipaddress.ip_address(host).is_loopback:
        raise ValueError("Native web solving must bind to a loopback address")
    server = ThreadingHTTPServer((host, port), partial(Handler, graph=graph, store=store, solver=solver))
    print(f"FEMM Second Brain: http://{host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if solver:
            solver.close()
