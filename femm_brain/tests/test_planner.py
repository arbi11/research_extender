"""The local catalog route never invents a baseline or discards input data."""
import unittest

from femm_brain.planner import plan_request


def catalog():
    cases = []
    for variant, dimensions in (("short", (1, 2, 1)), ("long", (1, 1.25, 4)), ("flat", (.25, 2, .25))):
        parameters = [{"name": name, "default": value, "unit": "in", "minimum": .001, "maximum": 100}
                      for name, value in zip(("inner_radius", "outer_radius", "length"), dimensions)]
        parameters += [{"name": "turns", "default": 100, "unit": "count", "minimum": 1, "maximum": 10000, "integer": True},
                       {"name": "current", "default": 1, "unit": "A", "minimum": .001, "maximum": 1000},
                       {"name": "mesh_scale", "default": 1, "unit": "1", "minimum": .05, "maximum": 4},
                       {"name": "boundary_scale", "default": 1, "unit": "1", "minimum": 1, "maximum": 8}]
        cases.append({"id": "aircore-" + variant, "title": "Air-core " + variant, "example_id": "aircore", "parameters": parameters})
    for variant in ("planar", "axisymmetric"):
        cases.append({"id": "uniform-" + variant, "title": "Uniform " + variant, "example_id": "uniformfield-" + variant,
                      "parameters": [{"name": "radius", "default": 100, "unit": "mm", "minimum": 1, "maximum": 1000},
                                     {"name": "applied_field_scale", "default": 1, "unit": "1", "minimum": .001, "maximum": 10},
                                     {"name": "mesh_scale", "default": 1, "unit": "1", "minimum": .05, "maximum": 4}]})
    return {"cases": cases}


class LocalPlannerTests(unittest.TestCase):
    def plan(self, request, graph=None):
        return plan_request(request, graph, catalog=catalog())

    def test_exact_named_profile_uses_no_hidden_overrides(self):
        result = self.plan("Please solve aircore-short")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["selected_case_id"], "aircore-short")
        self.assertEqual(result["parameters"], {})
        self.assertEqual(result["llm_calls"], 0)

    def test_alias_and_dimension_conversion(self):
        result = self.plan("Create the tall thin coil with inner_radius=25.4 mm, outer radius=3.175 cm, axial length 0.1016 m, 100 turns and current=1 A")
        self.assertEqual(result["status"], "ready", result)
        self.assertEqual(result["selected_case_id"], "aircore-long")
        for name, expected in (("inner_radius", 1), ("outer_radius", 1.25), ("length", 4)):
            self.assertAlmostEqual(result["parameters"][name], expected)
        self.assertEqual(len(result["parsed_values"]), 5)

    def test_generic_profile_is_ambiguous(self):
        result = self.plan("Solve an air-core coil inductance")
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(len(result["candidates"]), 3)
        self.assertIsNone(result["selected_case_id"])

    def test_full_dimensions_can_identify_one_documented_profile(self):
        result = self.plan('Solve an air core coil, ri=1 in, ro=2 inches, z=1 inch')
        self.assertEqual(result["status"], "ready", result)
        self.assertEqual(result["selected_case_id"], "aircore-short")

    def test_custom_dimensions_do_not_choose_an_arbitrary_source_baseline(self):
        result = self.plan("Solve aircore with ri=2 in, ro=3 in, z=4 in")
        self.assertEqual(result["status"], "ambiguous")

    def test_uniform_coordinate_system_must_be_supplied(self):
        self.assertEqual(self.plan("Solve a uniform magnetic field")["status"], "ambiguous")
        result = self.plan("Solve an axisymmetric uniform field")
        self.assertEqual(result["status"], "ready", result)
        self.assertEqual(result["selected_case_id"], "uniform-axisymmetric")

    def test_uniform_field_value_becomes_explainable_scale(self):
        result = self.plan("uniform-planar with target field=200 mT and radius=10 cm")
        self.assertEqual(result["status"], "ready", result)
        self.assertAlmostEqual(result["parameters"]["applied_field_scale"], .4)
        self.assertEqual(result["parameters"]["radius"], 100)
        self.assertEqual(next(x for x in result["parsed_values"] if x["name"] == "target_field")["unit"], "T")

    def test_common_uniform_field_phrase_has_an_explicit_value(self):
        result = self.plan("uniform planar field of 1 T")
        self.assertEqual(result["status"], "ready", result)
        self.assertEqual(result["selected_case_id"], "uniform-planar")
        self.assertEqual(result["parameters"]["applied_field_scale"], 2)
        self.assertEqual(self.plan("uniform field 2d target field=1 T")["status"], "ready")
        self.assertEqual(self.plan("uniform planar field of 1 T mesh scale=0.5 and radius=100 mm")["status"], "ready")

    def test_missing_units_are_not_guessed(self):
        result = self.plan("aircore-short inner radius=2")
        self.assertEqual(result["status"], "needs_input")
        self.assertIn("unit is required", " ".join(result["messages"]))

    def test_conflicting_equivalent_parameter_aliases_are_rejected(self):
        result = self.plan("aircore-short inner radius=1 in, ri=30 mm")
        self.assertEqual(result["status"], "needs_input")
        self.assertIn("Conflicting values", " ".join(result["messages"]))

    def test_duplicate_equivalent_values_are_accepted(self):
        result = self.plan("aircore-short inner radius=1 in, ri=25.4 mm")
        self.assertEqual(result["status"], "ready", result)

    def test_wrong_unit_does_not_become_a_length(self):
        result = self.plan("aircore-short inner radius=2 A")
        self.assertEqual(result["status"], "needs_input")
        self.assertIn("does not match", " ".join(result["messages"]))

    def test_unknown_numeric_and_textual_geometry_changes_are_not_ignored(self):
        for request in ("aircore-short diameter=20 mm", "aircore-short depth=10 mm", "aircore-short add another winding", "aircore-short with an oval coil", "aircore-short ri1", "uniform-planar B2"):
            with self.subTest(request=request):
                self.assertNotEqual(self.plan(request)["status"], "ready")

    def test_unsupported_physics_is_catalog_scoped(self):
        for request in ("aircore-short at 60 Hz", "aircore-short nonlinear iron", "uniform-planar with permanent magnets", "planar aircore-short"):
            with self.subTest(request=request):
                self.assertEqual(self.plan(request)["status"], "unsupported")

    def test_number_signs_and_scientific_exponents_are_preserved(self):
        self.assertEqual(self.plan("aircore-short inner radius=-1 in")["status"], "needs_input")
        result = self.plan("aircore-short current=1e-2 A")
        self.assertEqual(result["status"], "ready", result)
        self.assertEqual(result["parameters"]["current"], .01)

    def test_fractional_turn_count_is_rejected(self):
        self.assertEqual(self.plan("aircore-short 10.5 turns")["status"], "needs_input")

    def test_two_cases_are_not_silently_collapsed(self):
        self.assertEqual(self.plan("aircore-short and aircore-long")["status"], "ambiguous")

    def test_uniform_mode_and_field_definition_conflicts_are_explicit(self):
        self.assertEqual(self.plan("uniform-planar axisymmetric")["status"], "ambiguous")
        self.assertEqual(self.plan("uniform-planar target field=0.5 T applied_field_scale=1")["status"], "needs_input")

    def test_field_to_scale_conversion_still_enforces_catalog_range(self):
        self.assertEqual(self.plan("uniform-planar target field=100 T")["status"], "needs_input")
        self.assertEqual(self.plan("uniform-planar target field=0 T")["status"], "needs_input")

    def test_actual_backend_catalog_units_and_evidence_are_supported(self):
        result = plan_request("aircore-short mesh_scale=0.5")
        self.assertEqual(result["status"], "ready", result)
        self.assertEqual(result["parameters"]["mesh_scale"], .5)
        self.assertIn("workflow:aircore", result["node_ids"])
        for request, expected in (("uniform-planar target field=0 T", 0),
                                  ("uniform-axisymmetric target field=-200 mT", -.3)):
            with self.subTest(request=request):
                result = plan_request(request)
                self.assertEqual(result["status"], "ready", result)
                self.assertAlmostEqual(result["parameters"]["applied_field_scale"], expected)

    def test_inverted_winding_radii_do_not_produce_a_ready_plan(self):
        result = self.plan("aircore-short inner radius=3 in")
        self.assertEqual(result["status"], "needs_input")
        self.assertIn("outer_radius", result["messages"][0])

    def test_parallel_wires_transfer_alias_units_and_force_are_supported(self):
        result = plan_request("Solve the COMSOL parallel wire force with wire radius=20 cm, center distance=1 m and current=1 A")
        self.assertEqual(result["status"], "ready", result)
        self.assertEqual(result["selected_case_id"], "parallel-wires")
        self.assertAlmostEqual(result["parameters"]["wire_radius"], .2)
        self.assertIn("method:parallel_wire_cross_sections", result["node_ids"])

    def test_parallel_wire_overlap_and_wrong_physics_are_rejected(self):
        self.assertEqual(plan_request("parallel-wires wire radius=1 m")["status"], "needs_input")
        self.assertEqual(plan_request("parallel-wires axisymmetric")["status"], "unsupported")
        self.assertEqual(plan_request("parallel-wires 60 Hz")["status"], "unsupported")
        self.assertEqual(plan_request("parallel-wires equal currents")["status"], "unsupported")
        self.assertEqual(plan_request("parallel-wires same direction")["status"], "unsupported")
        self.assertEqual(plan_request("aircore-short force")["status"], "unsupported")

    def test_evidence_uses_explicit_nodes_without_shared_stage_expansion(self):
        graph = {"examples": [{"id": "aircore", "node_ids": ["workflow:aircore", "stage:geometry", "method:coil"]}],
                 "edges": [{"source": "stage:geometry", "target": "method:unrelated_thermal"}]}
        result = self.plan("aircore-short", graph)
        self.assertEqual(result["node_ids"], ["workflow:aircore", "stage:geometry", "method:coil"])
        self.assertNotIn("method:unrelated_thermal", result["node_ids"])

    def test_empty_and_unknown_requests_remain_reviewable(self):
        self.assertEqual(self.plan("")["status"], "needs_input")
        self.assertEqual(self.plan("Design an IPM motor")["status"], "unsupported")




class EngineeringPlannerTests(unittest.TestCase):
    def test_ei_source_force_energy_and_signed_current_selection(self):
        result = plan_request('gapped EI core force and inductance at 2 A')
        self.assertEqual(result['status'], 'ready', result)
        self.assertEqual(result['selected_case_id'], 'gapped-ei-core')
        self.assertEqual(result['parameters']['current'], 2)
        self.assertEqual(plan_request('gapped EI core current=-1 A')['status'], 'ready')
        self.assertEqual(plan_request('gapped EI core current=0 A')['status'], 'needs_input')

    def test_nonlinear_profile_and_explicit_excitation_scale(self):
        result = plan_request('nonlinear radial bearing current_scale=0.5')
        self.assertEqual(result['status'], 'ready', result)
        self.assertEqual(result['parameters']['current_scale'], .5)
        self.assertEqual(plan_request('eight pole magnetic bearing force')['selected_case_id'], 'radial-bearing')
        self.assertEqual(plan_request('radial-bearing current=12 A')['status'], 'needs_input')

    def test_material_laws_coordinate_modes_and_metrics_are_not_ignored(self):
        for text in ('gapped EI core nonlinear steel', 'radial bearing permanent magnets',
                     'aircore-short iron', 'uniform-planar energy', 'uniform-planar inductance',
                     'gapped EI core axisymmetric', 'radial bearing at 60 Hz'):
            with self.subTest(text=text):
                self.assertEqual(plan_request(text)['status'], 'unsupported')


if __name__ == "__main__":
    unittest.main()
