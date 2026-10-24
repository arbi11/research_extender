"""Executable contracts for hierarchy, physical dependencies and shared compiler."""
import copy
import math
import unittest

from pydantic import ValidationError

from app.assembly import (MachineAssembly, compile_assembly, preview_assembly,
                          apply_assembly_edits, capability_manifest)


def fact(identity, value, unit, role, **extra):
    return {"id": identity, "label": identity, "value": value, "unit": unit,
            "role": role, "status": "user", "source_refs": ["nl_request"], **extra}


def coil_assembly():
    facts = [fact('ri', 1., 'cm', 'length'), fact('ro', 20., 'mm', 'length'),
             fact('zmin', -20., 'mm', 'length'), fact('zmax', 20., 'mm', 'length'),
             fact('mu', 1., '1', 'permeability'), fact('amps', 1., 'A', 'current'),
             fact('turns', 100., 'turns', 'turns')]
    return MachineAssembly.model_validate({"id": "machine", "title": "Finite annular winding",
        "input_mode": "natural_language", "family": "new coil family", "physics": "axisymmetric_dc_magnetics",
        "facts": facts, "materials": [{"id": "winding", "name": "Winding", "model": "linear_isotropic", "relative_permeability_ref": "mu"}],
        "circuits": [{"id": "drive", "current_ref": "amps"}],
        "components": [{"id": "coil", "name": "Winding", "role": "winding", "family": "annular winding",
            "mode": "template", "source_refs": ["nl_request"],
            "template": {"kind": "annular_coil", "feature_id": "body", "material_id": "winding", "circuit_id": "drive", "turns_ref": "turns",
                "parameter_refs": [{"name": name, "fact_id": identity} for name, identity in
                    [('r_inner','ri'),('r_outer','ro'),('z_min','zmin'),('z_max','zmax')]]}}],
        "domain": {"type": "rectangle", "x_min": 0., "x_max": 80., "y_min": -80., "y_max": 80.},
        "air_label": {"x": 60., "y": 60.}, "mesh_mm": 2., "probes": [{"x": 0., "y": 0.}]})


def custom_assembly():
    return MachineAssembly.model_validate({"id": "machine", "title": "Uncatalogued slotted machine",
        "input_mode": "combined", "family": "unclassified experimental machine", "physics": "planar_dc_magnetics",
        "facts": [fact('depth', 5., 'mm', 'length')], "depth_ref": "depth", "materials": [], "circuits": [],
        "components": [{"id": "rotor", "name": "Custom rotor", "role": "rotor", "family": "unknown", "mode": "custom",
            "features": [{"id": "slots", "name": "Repeated slot", "role": "slot", "material_id": "air",
                "shape": {"type": "rectangle", "x_min": 8., "x_max": 10., "y_min": -.5, "y_max": .5},
                "label": {"x": 9., "y": 0.}, "source_refs": ["paper_span_1"],
                "pattern": {"count": 4, "rotate_step_deg": 90.}}]}],
        "domain": {"type": "circle", "center": {"x": 0., "y": 0.}, "radius": 25.},
        "air_label": {"x": 20., "y": 0.}, "mesh_mm": 2.})


class AssemblyTests(unittest.TestCase):
    def test_template_compiles_source_units_and_assignments(self):
        result = compile_assembly(coil_assembly())
        self.assertTrue(result.execution_supported, result.issues)
        self.assertEqual(result.plan.regions[0].shape.x_min, 10.)
        self.assertEqual(result.plan.regions[0].turns, 100)
        self.assertEqual(result.plan.circuits[0].current_a, 1.)
        self.assertEqual(result.expanded_features[0].feature_id, 'coil.body')
        self.assertEqual(result.feature_map['coil.body']['region_id'], result.plan.regions[0].id)
        self.assertIn('fact:amps', result.dependency_graph['feature:coil.body'])
        self.assertIn('fact:mu', result.dependency_graph['feature:coil.body'])

    def test_custom_novel_family_repetition_and_stable_ids(self):
        first = compile_assembly(custom_assembly())
        self.assertTrue(first.execution_supported, first.issues)
        self.assertEqual(len(first.plan.regions), 4)
        ids = [item.region_id for item in first.expanded_features]
        self.assertEqual(list(first.feature_map), ['rotor.slots@000','rotor.slots@001','rotor.slots@002','rotor.slots@003'])
        edited = apply_assembly_edits(custom_assembly(), [{"op":"replace", "path":"/components/rotor/features/slots/pattern/count", "value":3}])
        self.assertEqual([item.region_id for item in edited['compilation'].expanded_features], ids[:3])
        self.assertIn('rotor.slots@003', edited['affected_feature_ids'])

    def test_physics_and_material_law_remain_described_with_preview(self):
        raw = custom_assembly().model_dump()
        raw['required_physics'] = ['permanent_magnets', 'nonlinear_bh']
        result = compile_assembly(raw)
        self.assertFalse(result.execution_supported)
        self.assertIsNone(result.plan)
        self.assertEqual(len(result.expanded_features), 4)
        svg = preview_assembly(result)
        self.assertIn('native execution unavailable', svg)
        self.assertIn('data-feature-id="rotor.slots@000"', svg)
        self.assertEqual(result.assembly.family, 'unclassified experimental machine')
        raw['required_physics'] = []
        raw['materials'] = [{"id":"pm", "name":"Permanent magnet", "model":"permanent_magnet"}]
        raw['components'][0]['features'][0]['material_id'] = 'pm'
        result = compile_assembly(raw)
        self.assertTrue(any(issue.code == 'material_unavailable' for issue in result.issues))
        self.assertEqual(len(result.expanded_features), 4)

    def test_unknown_permeability_is_not_air(self):
        raw = coil_assembly().model_dump()
        raw['facts'][4].update(value=None, status='missing', unit=None)
        result = compile_assembly(raw)
        self.assertFalse(result.execution_supported)
        self.assertIsNone(result.plan)
        self.assertTrue(any(issue.code in {'invalid_feature','material_unavailable'} for issue in result.issues))

    def test_no_zero_thickness_or_unmentioned_dimensions_in_template(self):
        raw = coil_assembly().model_dump()
        raw['facts'][1]['value'] = 10.
        result = compile_assembly(raw)
        self.assertFalse(result.execution_supported)
        self.assertTrue(any(issue.code == 'invalid_component' for issue in result.issues))
        raw = coil_assembly().model_dump()
        raw['components'][0]['template']['parameter_refs'][0]['fact_id'] = 'not_given'
        self.assertFalse(compile_assembly(raw).execution_supported)

    def test_dimensional_roles_and_signed_turns_checked(self):
        raw = coil_assembly().model_dump()
        raw['facts'][0]['unit'] = 'A'
        with self.assertRaises(ValidationError): MachineAssembly.model_validate(raw)
        raw = coil_assembly().model_dump()
        raw['facts'][-1]['value'] = -50.
        self.assertEqual(compile_assembly(raw).plan.regions[0].turns, -50)
        raw['facts'][-1]['value'] = 2.5
        self.assertFalse(compile_assembly(raw).execution_supported)

    def test_derived_current_uses_units_and_exact_typed_arithmetic(self):
        raw = coil_assembly().model_dump()
        raw['facts'] += [fact('density', 1., 'A/mm2', 'other'), fact('area', 100., 'mm2', 'other'),
            fact('amp_turns',100.,'A','current',status='derived',source_refs=[],explanation='J times area',
                 calculation={"op":"multiply","inputs":["density","area"]})]
        raw['facts'][5].update(value=1., status='derived', source_refs=[], explanation='NI divided by turns',
                              calculation={"op":"divide","inputs":["amp_turns","turns"]})
        self.assertTrue(compile_assembly(raw).execution_supported)
        raw['facts'][5]['value'] = 100.
        self.assertTrue(any(issue.code == 'invalid_fact' for issue in compile_assembly(raw).issues))

    def test_closed_arcs_keep_intent_and_bound_sagitta(self):
        raw = custom_assembly().model_dump()
        feature = raw['components'][0]['features'][0]
        feature['pattern'] = None
        feature['shape'] = {"type":"closed_path", "start":{"x":1.,"y":0.},
            "edges":[{"kind":"arc","end":{"x":-1.,"y":0.},"sweep_deg":180.},
                     {"kind":"arc","end":{"x":1.,"y":0.},"sweep_deg":180.}], "tolerance_mm":.01}
        feature['label'] = {"x":0.,"y":0.}
        result = compile_assembly(raw)
        self.assertTrue(result.execution_supported, result.issues)
        self.assertEqual(result.assembly.components[0].features[0].shape.type, 'closed_path')
        approximate = result.expanded_features[0].approximation
        self.assertLessEqual(approximate['max_sagitta_mm'], .01+1e-12)
        self.assertGreater(approximate['vertices'], 4)
        self.assertIn('curve tolerance 0.01 mm', preview_assembly(result))
        raw['components'][0]['features'][0]['shape']['tolerance_mm'] = 1e-20
        result = compile_assembly(raw)
        self.assertFalse(result.execution_supported)
        self.assertTrue(any('no coarser curve' in issue.message for issue in result.issues))

    def test_mirror_rotate_translate_and_parent_transform_order(self):
        raw = custom_assembly().model_dump()
        rotor = raw['components'][0]
        rotor['parent_id'] = 'holder'
        rotor['features'][0]['pattern'] = None
        rotor['features'][0]['transform'] = {"mirror":"x", "rotate_deg":90., "dx_mm":1.}
        raw['components'].append({"id":"holder", "name":"Holder", "role":"assembly", "family":"unknown", "mode":"custom", "features":[],
                                  "transform":{"dx_mm":2.,"dy_mm":1.}})
        result = compile_assembly(raw)
        self.assertTrue(result.execution_supported, result.issues)
        label = result.expanded_features[0].label
        self.assertAlmostEqual(label.x,3.)
        self.assertAlmostEqual(label.y,10.)
        self.assertEqual(result.feature_map['rotor.slots']['parent_component_ids'], ['holder'])
        edited = apply_assembly_edits(raw,[{"op":"replace","path":"/components/holder/transform/dx_mm","value":3.}])
        self.assertIn('rotor.slots',edited['affected_feature_ids'])

    def test_template_modified_uses_same_features_and_explicit_ids(self):
        raw = coil_assembly().model_dump()
        component = raw['components'][0]
        component['mode'] = 'template_modified'
        component['overrides'] = [{"op":"replace","feature_id":"body","feature":{
            "id":"body","name":"Changed winding","role":"winding","shape":{"type":"rectangle","x_min":10.,"x_max":20.,"y_min":-20.,"y_max":20.},
            "material_id":"winding","circuit_id":"drive","turns_ref":"turns","label":{"x":12.,"y":0.},"fact_ids":["ri","ro","zmin","zmax"]}}]
        result = compile_assembly(raw)
        self.assertTrue(result.execution_supported,result.issues)
        self.assertEqual(list(result.feature_map),['coil.body'])
        self.assertEqual(result.plan.regions[0].label.x,12.)

    def test_scoped_fact_edits_invalidate_dependents_and_preserve_original(self):
        original = coil_assembly()
        edit = {"op":"replace","path":"/facts/amps/value","value":2.}
        result = apply_assembly_edits(original,[edit],allowed_paths=['/facts/amps/value'])
        self.assertEqual(original.facts[5].value,1.)
        self.assertEqual(result['compilation'].plan.circuits[0].current_a,2.)
        self.assertEqual(result['affected_feature_ids'],['coil.body'])
        with self.assertRaisesRegex(ValueError,'scope'): apply_assembly_edits(original,[edit],allowed_paths=['/facts/ro'])
        with self.assertRaises(ValueError): apply_assembly_edits(original,[{"op":"replace","path":"/components/coil/id","value":"new"}])

    def test_coordinate_binding_checks_values_and_units(self):
        raw = custom_assembly().model_dump()
        raw['facts'].append(fact('inner',.8,'cm','length'))
        feature = raw['components'][0]['features'][0]
        feature['bindings'] = [{"path":"/shape/x_min","fact_id":"inner","unit":"mm"}]
        self.assertTrue(compile_assembly(raw).execution_supported)
        raw['facts'][-1]['value'] = .9
        result = compile_assembly(raw)
        self.assertFalse(result.execution_supported)
        self.assertTrue(any('coordinate disagrees' in issue.message for issue in result.issues))

    def test_dependency_cycles_and_undefined_feature_dependencies_block(self):
        raw = custom_assembly().model_dump()
        raw['components'][0]['features'][0]['depends_on'] = ['not.real']
        self.assertTrue(any(issue.code=='undefined_dependency' for issue in compile_assembly(raw).issues))
        raw['components'][0]['features'][0]['depends_on'] = ['rotor.slots@000']
        self.assertTrue(any(issue.code=='dependency_cycle' for issue in compile_assembly(raw).issues))
        raw['components'][0]['parent_id'] = 'rotor'
        with self.assertRaises(ValidationError): MachineAssembly.model_validate(raw)

    def test_no_pilot_topology_or_polygon_vertex_caps(self):
        raw = custom_assembly().model_dump()
        feature = raw['components'][0]['features'][0]
        feature['shape'] = {"type":"circle", "center":{"x":50.,"y":0.}, "radius":1.}
        feature['label'] = {"x":50.,"y":0.}
        feature['pattern'] = {"count":40,"rotate_step_deg":9.}
        raw['domain']['radius'] = 100.
        raw['air_label'] = {"x":90.,"y":0.}
        result = compile_assembly(raw)
        self.assertTrue(result.execution_supported,result.issues)
        self.assertEqual(len(result.plan.regions),40)
        feature['pattern'] = None
        feature['shape'] = {"type":"closed_path", "start":{"x":1.,"y":0.},
            "edges":[{"kind":"arc","end":{"x":-1.,"y":0.},"sweep_deg":180.},
                     {"kind":"arc","end":{"x":1.,"y":0.},"sweep_deg":180.}],"tolerance_mm":.0001}
        feature['label'] = {"x":0.,"y":0.}
        result = compile_assembly(raw)
        self.assertTrue(result.execution_supported,result.issues)
        self.assertGreater(len(result.plan.regions[0].shape.points),128)

    def test_feature_edit_invalidation_excludes_unrelated_siblings(self):
        raw = custom_assembly().model_dump()
        first = raw['components'][0]['features'][0]
        first['pattern'] = None
        second = copy.deepcopy(first)
        second['id'] = 'other'
        second['shape'] = {"type":"circle","center":{"x":0.,"y":0.},"radius":1.}
        second['label'] = {"x":0.,"y":0.}
        raw['components'][0]['features'].append(second)
        result = apply_assembly_edits(raw,[{"op":"replace","path":"/components/rotor/features/slots/label/x","value":9.1}])
        self.assertEqual(result['affected_feature_ids'],['rotor.slots'])

    def test_planar_cannot_reinterpret_annular_template(self):
        raw = coil_assembly().model_dump(); raw['physics']='planar_dc_magnetics'
        self.assertTrue(any(issue.code=='template_coordinate_system' for issue in compile_assembly(raw).issues))
        self.assertIn('permanent_magnets',capability_manifest()['unsupported'])


if __name__ == '__main__': unittest.main()
