"""External transfer evidence survives refreshes of the FEMM-only crawl."""
import json
from pathlib import Path
import tempfile
import unittest

from femm_brain.core import load_graph


class ExternalSourceTests(unittest.TestCase):
    def write_fixture(self, directory):
        root = Path(directory)
        primary = {"id": "vendor", "url": "https://vendor.example/tutorial", "sha256": "curated", "status": "fetched"}
        (root / "source_manifest.json").write_text(json.dumps({"sources": [{"id": "manual5"}], "artifacts": []}))
        (root / "external_sources.json").write_text(json.dumps({"sources": [primary], "curated_source_hashes": {"vendor": "curated"}}))
        node = {"id": "workflow:vendor", "title": "Vendor transfer", "description": "Retain a separately curated transfer.", "layer": 1,
                "kind": "workflow", "source_ids": ["vendor"], "verification": {"status": "documented"}}
        (root / "transfer_cases.json").write_text(json.dumps({"nodes": [node]}))
        return root, primary

    def test_femm_manifest_refresh_does_not_remove_external_source_or_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            root, primary = self.write_fixture(directory)
            first = load_graph(root)
            (root / "source_manifest.json").write_text(json.dumps({"sources": [{"id": "manual5"}, {"id": "fresh_femm_page"}], "artifacts": []}))
            refreshed = load_graph(root)
            self.assertEqual(refreshed["validation"]["status"], "passed")
            self.assertIn(primary, refreshed["sources"])
            first_node = next(n for n in first["nodes"] if n["id"] == "workflow:vendor")
            refreshed_node = next(n for n in refreshed["nodes"] if n["id"] == "workflow:vendor")
            self.assertEqual(first_node, refreshed_node)

    def test_external_source_digest_change_requires_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root, primary = self.write_fixture(directory)
            primary["sha256"] = "changed"
            (root / "external_sources.json").write_text(json.dumps({"sources": [primary], "curated_source_hashes": {"vendor": "curated"}}))
            graph = load_graph(root)
            node = next(n for n in graph["nodes"] if n["id"] == "workflow:vendor")
            self.assertEqual(graph["source_changes"], ["vendor"])
            self.assertFalse(node["verification"]["source_current"])
            self.assertEqual(node["source_versions"]["vendor"], "curated")

    def test_collision_does_not_silently_replace_a_crawled_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root, primary = self.write_fixture(directory)
            (root / "source_manifest.json").write_text(json.dumps({"sources": [{"id": "manual5"}, {**primary, "url": "https://different.example/"}], "artifacts": []}))
            graph = load_graph(root)
            self.assertEqual(graph["validation"]["status"], "failed")
            self.assertTrue(any(issue["code"] == "external_source_conflict" for issue in graph["validation"]["issues"]))

    def test_bundled_transfer_retains_boundary_assumption_and_three_layers(self):
        graph = load_graph()
        self.assertEqual(graph["validation"]["status"], "passed")
        self.assertEqual({node["layer"] for node in graph["nodes"]}, {1, 2, 3})
        source = next(source for source in graph["sources"] if source["id"] == "comsol-parallel-wires")
        self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")
        transfer = next(node for node in graph["nodes"] if node["id"] == "method:parallel_wire_exterior_transfer")
        self.assertEqual(transfer["assumptions"][0]["origin"], "numerical_adaptation")
        self.assertEqual(transfer["verification"]["status"], "documented")


if __name__ == "__main__":
    unittest.main()
