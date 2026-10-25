"""Documented numerical criteria remain tied to their actual source studies."""
import json
from pathlib import Path
import tempfile
import unittest

from femm_brain.core import example_subgraph, load_graph


class BenchmarkMethodTests(unittest.TestCase):
    def test_ei_force_target_cites_companion_force_source(self):
        graph = load_graph()
        self.assertEqual(graph["validation"]["status"], "passed")
        study = example_subgraph(graph, "inductanceexample")
        self.assertIn("acforceexample", study["source_ids"])
        method = next(node for node in study["nodes"] if node["id"] == "method:benchmark_ei_output_checks")
        force = next(item for item in method["published_targets"] if item["metric"] == "force_y_N")
        self.assertEqual(force["source_id"], "acforceexample")
        self.assertEqual(force["value"], 1.11347)
        self.assertEqual(method["verification"]["status"], "documented")

    def test_bearing_contains_bh_and_weighted_force_commands(self):
        study = example_subgraph(load_graph(), "radialmagneticbearing")
        nodes = {node["id"]: node for node in study["nodes"]}
        self.assertIn("primitive:mi_addbhpoint", nodes)
        self.assertIn("primitive:mo_selectblock", nodes)
        self.assertIn("primitive:mo_blockintegral", nodes)
        self.assertIn("method:benchmark_bearing_output_checks", nodes)
        self.assertNotIn("method:benchmark_ei_output_checks", nodes)
        method = nodes["method:benchmark_bearing_output_checks"]
        self.assertEqual(method["published_targets"][0]["relative_tolerance"], 0.05)
        self.assertTrue(any("LamFill" in line for line in method["constraints"]))

    def test_unknown_benchmark_study_is_explicit_graph_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source_manifest.json").write_text(json.dumps({"sources": [{"id": "manual5"}]}))
            (root / "benchmark_methods.json").write_text(json.dumps({"example_bindings": [{"example_id": "missing"}]}))
            graph = load_graph(root)
            self.assertEqual(graph["validation"]["status"], "failed")
            self.assertTrue(any(issue["code"] == "benchmark_example_unknown" for issue in graph["validation"]["issues"]))


if __name__ == "__main__":
    unittest.main()
