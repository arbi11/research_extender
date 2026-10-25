"""Consolidation preserves source-scoped alternatives and fails on ambiguity."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from femm_brain.core import load_graph


class ConsolidationTests(unittest.TestCase):
    def graph(self, first, second, edges_first=(), edges_second=()):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source_manifest.json").write_text(json.dumps({"sources": [{"id": "manual5"}, {"id": "case_a"}, {"id": "case_b"}], "artifacts": []}))
            for name, nodes, edges in (("machines.json", first, edges_first), ("methods.json", second, edges_second)):
                (root / name).write_text(json.dumps({"nodes": nodes, "edges": edges, "examples": []}))
            return load_graph(root)

    def node(self):
        return {"id": "method:test", "title": "Test method", "layer": 2, "kind": "method", "description": "An explicit source-scoped method.",
                "parameters": [{"name": "length", "unit": "mm"}], "source_ids": ["case_a"], "constraints": ["Case A condition"], "verification": {"status": "documented"}}

    def test_parameter_and_verification_conflicts_cannot_be_silently_consolidated(self):
        first = self.node()
        for change in ({"parameters": [{"name": "length", "unit": "m"}]}, {"verification": {"status": "geometry_checked"}}, {"recipe_id": "other"}):
            second = {**deepcopy(first), **change}
            with self.subTest(change=change):
                self.assertEqual(self.graph([first], [second])["validation"]["status"], "failed")

    def test_compatible_shared_method_retains_conditions_with_their_sources(self):
        first = self.node()
        second = {**deepcopy(first), "source_ids": ["case_b"], "constraints": ["Case B condition"]}
        graph = self.graph([first], [second])
        self.assertEqual(graph["validation"]["status"], "passed")
        node = next(n for n in graph["nodes"] if n["id"] == first["id"])
        self.assertEqual(node["source_ids"], ["case_a", "case_b"])
        self.assertEqual([x["constraints"] for x in node["source_contexts"]], [["Case A condition"], ["Case B condition"]])

    def test_differently_conditioned_same_identity_edges_are_rejected(self):
        node = self.node()
        edge = {"id": "edge:test", "source": "stage:geometry", "target": node["id"], "relation": "decomposes_into", "when": "Condition A", "parameter_bindings": {}, "source_ids": ["case_a"]}
        graph = self.graph([node], [], [edge], [{**edge, "when": "Condition B"}])
        self.assertTrue(any(issue["code"] == "edge_conflict" for issue in graph["validation"]["issues"]))

    def test_source_refresh_does_not_silently_renew_a_curated_interpretation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source_manifest.json").write_text(json.dumps({"sources": [{"id": "manual5", "sha256": "unchanged"}, {"id": "case_a", "sha256": "new"}], "artifacts": []}))
            (root / "curation_manifest.json").write_text(json.dumps({"source_hashes": {"manual5": "unchanged", "case_a": "old"}}))
            (root / "methods.json").write_text(json.dumps({"nodes": [self.node()], "edges": [], "examples": []}))
            graph = load_graph(root)
            node = next(n for n in graph["nodes"] if n["id"] == "method:test")
            self.assertEqual(graph["source_changes"], ["case_a"])
            self.assertFalse(node["verification"]["source_current"])
            self.assertEqual(node["source_versions"]["case_a"], "old")


if __name__ == "__main__":
    unittest.main()
