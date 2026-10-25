"""Cross-layer and HTTP acceptance checks for the source-backed FEMM brain.

These tests do not invoke FEMM and do not turn documented examples into
numerically verified examples. Native and analytical recipe tests live beside
the geometry module.
"""
from copy import deepcopy
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def indexed(graph, key):
    return {item["id"]: item for item in graph[key]}


def reachable(start, edges):
    neighbors = {}
    for edge in edges:
        if edge["relation"] != "alternative_to":
            neighbors.setdefault(edge["source"], set()).add(edge["target"])
    seen, pending = set(), [start]
    while pending:
        node = pending.pop()
        if node not in seen:
            seen.add(node)
            pending.extend(neighbors.get(node, set()) - seen)
    return seen


class CorpusIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from femm_brain.core import load_graph
        cls.graph = load_graph()
        cls.nodes = indexed(cls.graph, "nodes")

    def test_corpus_has_three_layers_and_traceable_distinct_examples(self):
        from femm_brain.core import validate_graph
        self.assertEqual(validate_graph(self.graph), [])
        self.assertEqual(self.graph.get("consolidation_conflicts", []), [])
        self.assertEqual({node["layer"] for node in self.graph["nodes"]}, {1, 2, 3})
        self.assertGreaterEqual(len(self.graph["examples"]), 25)
        self.assertEqual(len(indexed(self.graph, "examples")), len(self.graph["examples"]))
        sources = indexed(self.graph, "sources")
        for example in self.graph["examples"]:
            with self.subTest(example=example["id"]):
                self.assertTrue(example["source_ids"])
                self.assertTrue(set(example["source_ids"]).issubset(sources))
                self.assertTrue(example["summary"].strip())
                self.assertTrue(example["source_url"].startswith("https://"))
                self.assertIn(example["workflow_id"], self.nodes)
                self.assertEqual(self.nodes[example["workflow_id"]]["layer"], 1)
                self.assertTrue(set(example["node_ids"]).issubset(self.nodes))
                methods = [node_id for node_id in example["node_ids"]
                           if self.nodes[node_id]["layer"] == 2]
                self.assertTrue(methods, "A source example must expose a component or method")
                self.assertTrue(any(any(self.nodes[child]["layer"] == 3
                                       for child in reachable(method, self.graph["edges"]))
                                    for method in methods),
                                "At least one source method must lead to executable primitives")

    def test_induction_methods_consolidate_shared_construction_without_losing_assumptions(self):
        by_source = {}
        for example in self.graph["examples"]:
            for source in example["source_ids"]:
                by_source.setdefault(source, []).append(example)
        self.assertIn("inductionmotorexample", by_source)
        self.assertIn("rapidimanalysis", by_source)
        first = by_source["inductionmotorexample"][0]
        second = by_source["rapidimanalysis"][0]
        self.assertNotEqual(first["id"], second["id"])
        self.assertTrue(first["assumptions"])
        self.assertTrue(second["assumptions"])
        self.assertNotEqual(first["assumptions"], second["assumptions"])
        shared = set(first["node_ids"]) & set(second["node_ids"])
        self.assertTrue(any(self.nodes[node_id]["layer"] == 2 for node_id in shared),
                        "Two motor examples must consolidate actual methods, not just shared API calls")
        alternatives = [edge for edge in self.graph["edges"] if edge["relation"] == "alternative_to"]
        self.assertTrue(alternatives)
        self.assertTrue(all(edge["when"].strip() for edge in alternatives),
                        "An alternative requires a condition of applicability")

    def test_graph_validation_rejects_dangling_edges_sources_and_extra_semantic_layers(self):
        from femm_brain.core import validate_graph
        for mutation in ("edge", "source", "layer"):
            invalid = deepcopy(self.graph)
            if mutation == "edge":
                invalid["edges"][0]["target"] = "missing:operation"
            elif mutation == "source":
                invalid["nodes"][0]["source_ids"] = ["missing-source"]
            else:
                invalid["nodes"][0]["layer"] = 4
            with self.subTest(mutation=mutation):
                self.assertTrue(validate_graph(invalid))

    def test_sqlite_roundtrip_preserves_provenance_and_supports_search(self):
        from femm_brain.store import GraphStore
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "brain.sqlite3"
            store = GraphStore(path)
            store.replace(self.graph)
            restored = GraphStore(path).graph()
            self.assertEqual(indexed(restored, "nodes"), indexed(self.graph, "nodes"))
            self.assertEqual(indexed(restored, "edges"), indexed(self.graph, "edges"))
            self.assertEqual(indexed(restored, "sources"), indexed(self.graph, "sources"))
            hits = store.search("induction", limit=10)
            self.assertTrue(hits)
            self.assertLessEqual(len(hits), 10)
            self.assertFalse(store.search("zzzz_nonexistent_engineering_term_zzzz"))

    def test_consolidation_retains_each_sources_conditions_and_rejects_silent_promotion(self):
        from femm_brain.core import load_graph
        first = deepcopy(next(node for node in self.graph["nodes"] if node["layer"] == 2))
        first.update(id="method:fixture_shared", title="Shared circuit study",
                     source_ids=["source_a"], description="AC identification with induced rotor currents.",
                     constraints=["Linear material assumption for this identification method."],
                     verification={"status": "documented", "note": "Fixture source A."})
        second = deepcopy(first)
        second.update(source_ids=["source_b"], description="Magnetostatic iteration with imposed rotor currents.",
                      constraints=["Rotor current must be supplied by the external iteration."])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source_manifest.json").write_text(json.dumps({"sources": [
                {"id": name, "url": "https://femm.info/" + name} for name in ("manual5", "source_a", "source_b")
            ], "artifacts": []}))
            (root / "machines.json").write_text(json.dumps({"nodes": [first], "edges": [], "examples": []}))
            method_file = root / "methods.json"
            method_file.write_text(json.dumps({"nodes": [second], "edges": [], "examples": []}))
            graph = load_graph(root)
            self.assertFalse(graph["consolidation_conflicts"])
            shared = indexed(graph, "nodes")[first["id"]]
            contexts = {item["source_ids"][0]: item for item in shared["source_contexts"]}
            self.assertEqual(contexts["source_a"]["constraints"], first["constraints"])
            self.assertEqual(contexts["source_b"]["constraints"], second["constraints"])
            self.assertEqual(contexts["source_a"]["description"], first["description"])
            self.assertEqual(contexts["source_b"]["description"], second["description"])
            second["verification"] = {"status": "numerically_checked", "note": "Conflicting fixture status.", "report": "fixture-report"}
            method_file.write_text(json.dumps({"nodes": [second], "edges": [], "examples": []}))
            self.assertTrue(any(issue["code"] == "shared_evidence_conflict"
                                for issue in load_graph(root)["consolidation_conflicts"]))


class HttpIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            cls.port = sock.getsockname()[1]
        cls.url = "http://127.0.0.1:" + str(cls.port)
        cls.log = (cls.folder / "server.log").open("w+")
        cls.process = subprocess.Popen(
            [sys.executable, "-m", "femm_brain", "serve", "--host", "127.0.0.1",
             "--port", str(cls.port), "--db", str(cls.folder / "brain.sqlite3")],
            cwd=ROOT, stdout=cls.log, stderr=subprocess.STDOUT,
        )
        for _ in range(150):
            if cls.process.poll() is not None:
                break
            try:
                with urllib.request.urlopen(cls.url + "/api/graph", timeout=1) as response:
                    if response.status == 200:
                        cls.graph = json.load(response)
                        return
            except (OSError, urllib.error.URLError):
                time.sleep(.1)
        cls.process.terminate()
        cls.process.wait(timeout=5)
        cls.log.flush()
        failure = (cls.folder / "server.log").read_text()[-5000:]
        cls.log.close()
        cls.temp.cleanup()
        raise RuntimeError("Brain server did not start: " + failure)

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        try:
            cls.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.wait(timeout=5)
        cls.log.close()
        cls.temp.cleanup()

    def get(self, path, *, raw=False):
        with urllib.request.urlopen(self.url + path, timeout=10) as response:
            payload = response.read()
            return payload if raw else json.loads(payload)

    def test_graph_search_and_source_details_are_available_offline(self):
        self.assertTrue(self.graph["nodes"])
        self.assertTrue(self.graph["stats"])
        hits = self.get("/api/search?q=induction")
        self.assertTrue(hits["results"])
        node_id = self.graph["nodes"][0]["id"]
        detail = self.get("/api/node/" + urllib.parse.quote(node_id, safe=""))
        self.assertEqual(detail["node"]["id"], node_id)
        for key in ("incoming", "outgoing", "sources"):
            self.assertIn(key, detail)
        case = self.graph["examples"][0]
        detail = self.get("/api/examples/" + urllib.parse.quote(case["id"], safe=""))
        self.assertEqual(detail["id"], case["id"])
        self.assertTrue(detail["nodes"])
        source_data = self.get("/api/sources")
        self.assertTrue(source_data["sources"])
        self.assertIn("coverage", source_data)

    def test_bundled_source_geometry_retains_its_identity(self):
        inventory = self.get("/api/sources")
        previews = [artifact for artifact in inventory["artifacts"] if artifact.get("preview_available")]
        self.assertTrue(previews, "At least one genuine FEMM source model must be inspectable offline")
        source = previews[0]
        geometry = self.get("/api/geometry?artifact=" + urllib.parse.quote(source["id"], safe=""))
        self.assertIn("<svg", geometry["svg"])
        self.assertTrue(geometry["operations"])
        self.assertNotEqual(geometry.get("recipe_id"), "stator_tooth")

    def test_geometry_and_portable_downloads_have_matching_primitive_ids(self):
        recipes = self.get("/api/recipes")["recipes"]
        self.assertTrue(recipes)
        self.assertIn("stator_tooth", {recipe["id"] for recipe in recipes})
        geometry = self.get("/api/geometry?recipe=stator_tooth")
        self.assertIn("<svg", geometry["svg"])
        self.assertTrue(geometry["operations"])
        self.assertTrue(geometry["parameters"])
        for operation in geometry["operations"]:
            for primitive_id in operation.get("geometry_ids", []):
                self.assertIn(primitive_id, geometry["svg"])
        self.assertIn(geometry["verification"]["status"], ("reconstructed", "geometry_checked"))
        for format in ("python", "lua", "fem", "svg"):
            payload = self.get("/api/download?recipe=stator_tooth&format=" + format, raw=True)
            self.assertGreater(len(payload), 20)
            if format == "python":
                compile(payload, "exported_tooth.py", "exec")
                self.assertNotIn(b"from femm_brain", payload)
            elif format == "lua":
                self.assertIn(b"mi_add", payload)
            elif format == "svg":
                self.assertIn(b"<svg", payload)

    def test_unknown_models_invalid_parameters_and_path_traversal_are_rejected(self):
        for path in ("/api/node/missing-node", "/api/geometry?recipe=missing-recipe",
                     "/api/download?recipe=stator_tooth&format=shell",
                     "/../../CONTRACT.md"):
            with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError) as caught:
                self.get(path, raw=True)
            self.assertIn(caught.exception.code, (400, 404, 422))
        query = urllib.parse.urlencode({"recipe": "stator_tooth",
                                       "parameters": json.dumps({"unrecognized_parameter": 1})})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get("/api/geometry?" + query)
        self.assertIn(caught.exception.code, (400, 422))


if __name__ == "__main__":
    unittest.main()
