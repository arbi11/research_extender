"""Registered measured-force targets remain separate and correctly scoped."""
from copy import deepcopy
import hashlib
from pathlib import Path
import unittest
from femm_brain.planner import plan_request
from femm_brain.simulation import plan_simulation, assess_target, _mesh_svg

class VM270IntegrationTests(unittest.TestCase):
    def test_local_units_and_registered_measured_targets(self):
        selection = plan_request('ANSYS VM270 force at 50 mA')
        self.assertEqual(selection['status'], 'ready', selection)
        self.assertEqual(selection['parameters']['current'], .05)
        plan = plan_simulation(selection['selected_case_id'], selection['parameters'])
        self.assertEqual(len(plan['frozen_references']), 2)
        self.assertAlmostEqual(plan['frozen_references'][1]['value'], -.0112815)
        self.assertTrue({'ansys-vm270','ansys-vm270-input'} <= set(plan['source_ids']))
        module = Path(__file__).resolve().parents[1] / 'vm270.py'
        self.assertEqual(plan['vm270_recipe_sha256'], hashlib.sha256(module.read_bytes()).hexdigest())
        self.assertIn('force_coil_z_N', plan['lua'])

    def test_browser_and_cli_numeric_notation_share_the_same_frozen_plan(self):
        for case in ('ansys-vm270','gapped-ei-core','radial-bearing','aircore-short'):
            baseline=plan_simulation(case)
            values={key: int(value) if float(value).is_integer() else float(value)
                    for key,value in baseline['parameters'].items()}
            self.assertEqual(plan_simulation(case,values)['plan_sha256'],baseline['plan_sha256'])
            values={key:float(value) for key,value in values.items()}
            self.assertEqual(plan_simulation(case,values)['plan_sha256'],baseline['plan_sha256'])

    def test_physical_input_edits_do_not_inherit_measured_force(self):
        for parameters in ({'current':0}, {'current':-.05}, {'displacement':0}):
            plan = plan_simulation('ansys-vm270', parameters)
            self.assertTrue(all(not ref['applicable'] and ref.get('reason') for ref in plan['frozen_references']))
            self.assertEqual(assess_target(plan, {})['status'], 'unassessed')
        self.assertTrue(plan_simulation('ansys-vm270', {'domain_radius':40})['frozen_reference']['applicable'])

    def test_secondary_force_reference_cannot_silently_pass(self):
        plan = plan_simulation('ansys-vm270')
        target = plan['frozen_reference']['value']
        metrics = {'force_magnet_z_N':target, 'force_coil_z_N':-target, 'force_coil_wst_z_N':-target}
        self.assertEqual(assess_target(plan, metrics)['status'], 'passed')
        plan['frozen_references'][1]['value'] *= 2
        result = assess_target(plan, metrics)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['targets'][1]['status'], 'failed')

    def test_mesh_preview_bound_does_not_change_native_connectivity(self):
        mesh={'nodes':{0:(0,0),1:(1,0),2:(0,1)},'elements':[[0,1,2] for _ in range(13)]}
        before=deepcopy(mesh)
        svg=_mesh_svg(mesh,max_elements=3)
        self.assertEqual(mesh,before)
        self.assertEqual(svg.count('Z'),3)
        self.assertIn('3 of 13 actual triangles',svg)
        self.assertIn('model.ele',svg)
        with self.assertRaises(ValueError): _mesh_svg(mesh,max_elements=0)
