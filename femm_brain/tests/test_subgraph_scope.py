"""A source example must not inherit every method attached to shared stages."""
import unittest

from femm_brain.core import example_subgraph, load_graph


class ExampleScopeTests(unittest.TestCase):
    def test_real_aircore_scope_keeps_its_recipe_but_not_unrelated_physics(self):
        graph = load_graph()
        subgraph = example_subgraph(graph, "aircore")
        selected = {node["id"] for node in subgraph["nodes"]}
        example = next(item for item in graph["examples"] if item["id"] == "aircore")
        self.assertTrue(set(example["node_ids"]) <= selected)
        self.assertIn("recipe:coil_section", selected)
        self.assertIn("recipe:coil_section:op:0", selected)
        self.assertIn("primitive:mi_addnode", selected)
        self.assertNotIn("method:methods_thermal_convection_boundary", selected)
        self.assertNotIn("method:methods_axisymmetric_spheres", selected)
        self.assertNotIn("recipe:slotted_stator", selected)
        self.assertLess(len(selected), 60)

    def test_all_examples_keep_their_explicit_bindings_and_source_assets(self):
        graph = load_graph()
        for example in graph["examples"]:
            with self.subTest(example=example["id"]):
                scoped = example_subgraph(graph, example["id"])
                selected = {node["id"] for node in scoped["nodes"]}
                self.assertTrue(set(example["node_ids"]) <= selected)
                self.assertIn(example["workflow_id"], selected)
                self.assertTrue(all(edge["source"] in selected and edge["target"] in selected for edge in scoped["edges"]))
                self.assertTrue(all(set(asset["source_ids"]) & set(example["source_ids"]) for asset in scoped["artifacts"]))

    def test_source_specific_stage_still_expands_with_its_own_context(self):
        graph = {
            "nodes": [{"id": "workflow:a", "kind": "workflow", "layer": 1},
                      {"id": "stage:a:geometry", "kind": "stage", "layer": 1},
                      {"id": "method:a", "kind": "method", "layer": 2},
                      {"id": "method:b", "kind": "method", "layer": 2},
                      {"id": "primitive:a", "kind": "operation", "layer": 3}],
            "examples": [{"id": "a", "workflow_id": "workflow:a", "source_ids": ["a"], "node_ids": ["stage:a:geometry"]}],
            "edges": [{"source": "workflow:a", "target": "stage:a:geometry", "relation": "decomposes_into", "source_ids": ["a"]},
                      {"source": "stage:a:geometry", "target": "method:a", "relation": "decomposes_into", "source_ids": ["a"]},
                      {"source": "stage:a:geometry", "target": "method:b", "relation": "decomposes_into", "source_ids": ["b"]},
                      {"source": "method:a", "target": "primitive:a", "relation": "implemented_by", "source_ids": ["manual"]}],
            "artifacts": []}
        scoped = example_subgraph(graph, "a")
        self.assertEqual({node["id"] for node in scoped["nodes"]}, {"workflow:a", "stage:a:geometry", "method:a", "primitive:a"})

    def test_unselected_common_stage_child_stays_out_even_with_shared_source(self):
        graph = {"nodes": [{"id": "workflow:a", "kind": "workflow", "layer": 1},
                           {"id": "stage:geometry", "kind": "stage", "layer": 1},
                           {"id": "method:a", "kind": "method", "layer": 2},
                           {"id": "method:alternative", "kind": "method", "layer": 2}],
                 "examples": [{"id": "a", "workflow_id": "workflow:a", "source_ids": ["a"], "node_ids": ["stage:geometry", "method:a"]}],
                 "edges": [{"source": "stage:geometry", "target": "method:a", "relation": "decomposes_into", "source_ids": ["a"]},
                           {"source": "stage:geometry", "target": "method:alternative", "relation": "decomposes_into", "source_ids": ["a"]}],
                 "artifacts": []}
        self.assertNotIn("method:alternative", {node["id"] for node in example_subgraph(graph, "a")["nodes"]})


if __name__ == "__main__":
    unittest.main()
