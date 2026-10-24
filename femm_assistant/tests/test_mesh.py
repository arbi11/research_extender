from pathlib import Path
import tempfile
import unittest

from app.mesh import MeshValidationError, validate_mesh
from app.plan import GeometryPlan


class MeshTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def write(self, nodes, elements, *, base="model", edges=None):
        (self.folder / f"{base}.node").write_text(nodes)
        (self.folder / f"{base}.ele").write_text(elements)
        if edges is not None:
            (self.folder / f"{base}.edge").write_text(edges)

    def square(self, zero=False, base="model", edges=None):
        start = 0 if zero else 1
        nodes = "4 2 0 1\n" + "".join(f"{index+start} {x} {y} 0\n" for index, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]))
        elements = f"2 3 1\n{start} {start} {start+1} {start+2} 1\n{start+1} {start} {start+2} {start+3} 1\n"
        self.write(nodes, elements, base=base, edges=edges)
        return nodes, elements

    def grid(self):
        nodes = "16 2 0 0\n" + "".join(f"{y*4+x} {x} {y}\n" for y in range(4) for x in range(4))
        triangles = []
        for y in range(3):
            for x in range(3):
                index = y * 4 + x
                attribute = 2 if (x, y) == (1, 1) else 1
                triangles.extend([(index, index + 1, index + 5, attribute), (index, index + 5, index + 4, attribute)])
        elements = "18 3 1\n" + "".join(f"{index} " + " ".join(map(str, triangle)) + "\n" for index, triangle in enumerate(triangles))
        self.write(nodes, elements)
        plan = GeometryPlan.model_validate(dict(title="Two material faces", depth_mm=1., mesh_mm=.5,
            domain=dict(type="rectangle", x_min=0., x_max=3., y_min=0., y_max=3.),
            air_label=dict(x=.3, y=.5), materials=[], circuits=[],
            regions=[dict(id="inner", material="air", label=dict(x=1.3, y=1.5),
                          shape=dict(type="rectangle", x_min=1., x_max=2., y_min=1., y_max=2.))]))
        return nodes, elements, plan

    def test_actual_mesh_counts_quality_and_svg(self):
        self.square()
        report = validate_mesh(self.folder)
        self.assertEqual((report["node_count"], report["element_count"], report["edge_count"]), (4, 2, 5))
        self.assertAlmostEqual(report["quality"]["min_angle_deg"], 45.)
        self.assertAlmostEqual(report["quality"]["max_aspect_ratio"], 2.)
        self.assertAlmostEqual(report["quality"]["total_area_mm2"], 1.)
        self.assertFalse(report["convergence_verified"])
        self.assertIn("convergence", " ".join(report["warnings"]))
        self.assertEqual((self.folder / "mesh.svg").read_text().count("<polygon"), 2)

    def test_zero_indices_cli_basename_comments_lfs_and_edge_file(self):
        nodes, elements = self.square(zero=True, base="model.1", edges="5 1\n0 0 1 1\n1 1 2 1\n2 0 2 0\n3 2 3 1\n4 0 3 1\n")
        nodes = "# native tangle output\n" + nodes.replace(" 0\n", " 0 .2 # local feature size\n")
        (self.folder / "model.1.node").write_text(nodes)
        report = validate_mesh(self.folder)
        self.assertEqual(report["files"], ["model.1.node", "model.1.ele", "model.1.edge"])

    def test_does_not_accept_ambiguous_or_incomplete_artifacts(self):
        self.square()
        self.square(base="model.1")
        with self.assertRaisesRegex(MeshValidationError, "one complete"):
            validate_mesh(self.folder)
        (self.folder / "model.ele").unlink()
        (self.folder / "model.1.ele").unlink()
        with self.assertRaisesRegex(MeshValidationError, "one complete"):
            validate_mesh(self.folder)

    def test_node_header_counts_and_nonfinite_values(self):
        nodes, elements = self.square()
        for replacement in [nodes.replace("4 2", "250001 2", 1), nodes.replace("4 2", "4 3", 1),
                            nodes.replace("1 0 0 0", "1 nan 0 0"), nodes.replace("1 0 0 0", "1 1001 0 0"),
                            nodes.replace("2 1 0 0", "2 0 0 0"), nodes.replace("2 1 0 0", "1 1 0 0")]:
            with self.subTest(node=replacement):
                self.write(replacement, elements)
                with self.assertRaises(MeshValidationError):
                    validate_mesh(self.folder)

    def test_rejects_truncation_extra_rows_bad_integer_and_nonascii(self):
        nodes, elements = self.square()
        for replacement in ["\n".join(nodes.splitlines()[:-1]), nodes + "5 2 2 0\n", nodes.replace("2 1 0 0", "2.0 1 0 0"), nodes + "é\n"]:
            self.write(replacement, elements)
            with self.subTest(node=replacement), self.assertRaises(MeshValidationError):
                validate_mesh(self.folder)

    def test_invalid_element_indices_attributes_and_duplicates(self):
        nodes, elements = self.square()
        invalid = [elements.replace("2 1 3 4 1", "2 1 3 9 1"), elements.replace("2 1 3 4 1", "2 1 3 3 1"),
                   elements.replace("2 1 3 4 1", "2 3 2 1 1"), elements.replace("2 1 3 4 1", "1 1 3 4 1"),
                   elements.replace("2 1 3 4 1", "2 1 3 4 nan"), elements.replace("2 1 3 4 1", "2 1 3 4 0"),
                   elements.replace("2 1 3 4 1", "2 1 3 4 1.5"), elements.replace("2 3 1", "500001 3 1", 1)]
        for replacement in invalid:
            self.write(nodes, replacement)
            with self.subTest(element=replacement), self.assertRaises(MeshValidationError):
                validate_mesh(self.folder)

    def test_zero_area_and_numerically_degenerate_triangles(self):
        for height in (0, 1e-15):
            self.write(f"3 2 0 0\n1 0 0\n2 1 0\n3 .5 {height}\n", "1 3 0\n1 1 2 3\n")
            with self.subTest(height=height), self.assertRaisesRegex(MeshValidationError, "degenerate"):
                validate_mesh(self.folder)

    def test_adjacent_overlap_and_nonmanifold_edges_rejected(self):
        nodes, _ = self.square()
        self.write(nodes, "2 3 0\n1 1 2 3\n2 1 2 4\n")
        with self.assertRaisesRegex(MeshValidationError, "overlapping"):
            validate_mesh(self.folder)
        self.write("5 2 0 0\n1 0 0\n2 1 0\n3 1 1\n4 0 -1\n5 .5 2\n", "3 3 0\n1 1 2 3\n2 1 2 4\n3 1 2 5\n")
        with self.assertRaisesRegex(MeshValidationError, "non-manifold"):
            validate_mesh(self.folder)

    def test_edge_artifact_must_match_actual_triangles(self):
        self.square(edges="4 0\n1 1 2\n2 2 3\n3 3 4\n4 1 4\n")
        with self.assertRaisesRegex(MeshValidationError, "does not match"):
            validate_mesh(self.folder)

    def test_reversed_winding_is_supported(self):
        nodes, elements = self.square()
        self.write(nodes, elements.replace("2 1 3 4 1", "2 4 3 1 1"))
        self.assertEqual(validate_mesh(self.folder)["element_count"], 2)

    def test_plan_domain_and_all_label_seeds_are_checked(self):
        _, _, plan = self.grid()
        report = validate_mesh(self.folder, plan)
        self.assertEqual(report["region_element_counts"], {"1": 16, "2": 2})
        self.assertAlmostEqual(report["quality"]["total_area_mm2"], 9.)
        self.assertEqual([check["name"] for check in report["checks"]][-2:], ["domain_coverage", "block_labels"])

    def test_plan_rejects_missing_attribute_and_wrong_seed_assignment(self):
        nodes, elements, plan = self.grid()
        self.write(nodes, elements.replace(" 2\n", " 1\n"))
        with self.assertRaisesRegex(MeshValidationError, "assignments"):
            validate_mesh(self.folder, plan)
        self.write(nodes, elements.replace(" 2\n", " 1\n").replace("0 0 1 5 1", "0 0 1 5 2"))
        with self.assertRaisesRegex(MeshValidationError, "seed"):
            validate_mesh(self.folder, plan)
        self.write(nodes, elements.replace(" 2\n", " 3\n"))
        with self.assertRaisesRegex(MeshValidationError, "absent"):
            validate_mesh(self.folder, plan)

    def test_plan_rejects_outside_nodes_holes_and_incomplete_domain(self):
        nodes, elements, plan = self.grid()
        self.write(nodes.replace("0 0 0", "0 -.01 0"), elements)
        with self.assertRaisesRegex(MeshValidationError, "outside"):
            validate_mesh(self.folder, plan)
        lines = elements.splitlines()
        # Remove the two inner-face triangles and renumber remaining elements.
        kept = [line.split()[1:] for line in lines[1:] if line.split()[-1] != "2"]
        self.write(nodes, "16 3 1\n" + "".join(f"{index} " + " ".join(row) + "\n" for index, row in enumerate(kept)))
        with self.assertRaisesRegex(MeshValidationError, "hole"):
            validate_mesh(self.folder, plan)
        self.write("3 2 0 0\n0 0 0\n1 3 0\n2 3 3\n", "1 3 1\n0 0 1 2 1\n")
        with self.assertRaisesRegex(MeshValidationError, "area"):
            validate_mesh(self.folder, plan)

    def test_slender_elements_warn_without_claiming_convergence(self):
        self.write("3 2 0 0\n0 0 0\n1 1 0\n2 .5 .001\n", "1 3 0\n0 0 1 2\n")
        report = validate_mesh(self.folder)
        self.assertEqual(report["status"], "passed")
        self.assertLess(report["quality"]["min_angle_deg"], 1)
        self.assertIn("Slender", " ".join(report["warnings"]))
        self.assertFalse(report["convergence_verified"])


if __name__ == "__main__":
    unittest.main()
