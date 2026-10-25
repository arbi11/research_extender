"""Source units, independent physical assignments and force signs for VM270."""
from copy import deepcopy
import math
import unittest

from femm_brain.vm270 import (
    DEFAULTS, RELATIVE_PERMEABILITY, TARGET_FORCE_N, TARGET_RELATIVE_TOLERANCE,
    assess_vm270, build_vm270, metric_lines, positions, verify_vm270,
)


def fixture():
    p = dict(DEFAULTS)
    geometry, lines, *_ = build_vm270({}, p)
    plan = {'parameters': p, 'geometry': geometry,
            'frozen_reference': {'metric': 'force_magnet_z_N', 'value': TARGET_FORCE_N,
                                 'applicable': True, 'relative_tolerance': TARGET_RELATIVE_TOLERANCE}}
    actual = deepcopy(geometry)
    actual['metadata']['Frequency'] = '0'
    actual['materials'] = []
    for name, mu, hc in [('air', 1, 0), ('copper', 1, 0), ('SmCo', RELATIVE_PERMEABILITY, 720000)]:
        actual['materials'].append({'name': name, 'properties': {'Mu_x': str(mu), 'Mu_y': str(mu), 'H_c': str(hc), 'J_re': '0', 'J_im': '0', 'Sigma': '0', 'LamType': '0', 'LamFill': '1'}})
    actual['circuits'] = [{'name': 'icoil', 'properties': {'TotalAmps_re': '.05', 'TotalAmps_im': '0', 'CircuitType': '1'}}]
    actual['boundaries'] = [{'name': 'A=0', 'properties': {k: '0' for k in ('BdryType', 'A_0', 'A_1', 'A_2', 'Phi', 'c0', 'c1')}}]
    for node in actual['nodes']:
        node['attributes'] = ['0', '0']
    for seg in actual['segments']:
        seg['attributes'] = ['-1', '0', '0', '0']
    for arc in actual['arcs']:
        arc['attributes'] = ['180', '1', '1', '0', '0', '0']
    for label, attrs in zip(actual['labels'], [['3', '.005', '0', '90', '1', '1'], ['2', '.01', '1', '0', '2', '280'], ['1', '.005', '0', '0', '0', '1'], ['1', '.01', '0', '0', '0', '1'], ['1', '.1', '0', '0', '0', '1']]):
        label['attributes'] = attrs
        label['material_index'] = int(attrs[0])
    return plan, actual, lines


class VM270Tests(unittest.TestCase):
    def test_exact_source_positions_material_formula_and_measured_target(self):
        q = positions(DEFAULTS)
        self.assertAlmostEqual(q['magnet_radius'], 1.4986)
        self.assertAlmostEqual(q['coil_inner_radius'], 1.524)
        self.assertAlmostEqual(q['coil_outer_radius'], 1.9812)
        self.assertAlmostEqual(q['coil_start'], 1.961)
        self.assertAlmostEqual(q['coil_end'], 3.485)
        self.assertEqual(RELATIVE_PERMEABILITY, 1.02/(720000*4*math.pi*1e-7))
        self.assertAlmostEqual(TARGET_FORCE_N, .0112815)
        self.assertEqual(TARGET_RELATIVE_TOLERANCE, .02)

    def test_build_is_typed_and_domain_does_not_coarsen_local_mesh(self):
        a = build_vm270({}, DEFAULTS)
        b = build_vm270({}, dict(DEFAULTS, domain_radius=40))
        self.assertEqual(a[0]['metadata']['LengthUnits'], 'millimeters')
        self.assertEqual(a[-1], .001)
        self.assertEqual(len(a[0]['arcs']), 1)
        self.assertEqual([line for line in a[1] if line.startswith('mi_setblockprop')],
                         [line for line in b[1] if line.startswith('mi_setblockprop')])
        self.assertIn('mi_addcircprop("icoil",0.050000000000000003,1)', a[1])
        self.assertFalse(any('dofile' in line for line in a[1]))

    def test_physics_verifier_catches_wrong_current_magnetization_turns_and_boundary(self):
        plan, actual, _ = fixture()
        self.assertEqual(verify_vm270(plan, actual)['status'], 'passed')
        mutations = [lambda a: a['labels'][0]['attributes'].__setitem__(3, '-90'),
                     lambda a: a['labels'][1]['attributes'].__setitem__(5, '279'),
                     lambda a: a['circuits'][0]['properties'].__setitem__('TotalAmps_re', '.5'),
                     lambda a: a['materials'][2]['properties'].__setitem__('Mu_x', '1.126894'),
                     lambda a: a['arcs'][0]['attributes'].__setitem__(2, '0'),
                     lambda a: a['segments'][0]['attributes'].__setitem__(1, '1'),
                     lambda a: a['boundaries'][0]['properties'].__setitem__('A_0', '1'),
                     lambda a: a['metadata'].__setitem__('ProblemType', 'planar'),
                     lambda a: a['nodes'][0]['attributes'].__setitem__(0, '1')]
        for mutate in mutations:
            changed = deepcopy(actual)
            mutate(changed)
            with self.assertRaises(ValueError):
                verify_vm270(plan, changed)

    def test_force_checks_require_both_estimators_signs_and_action_reaction(self):
        plan, *_ = fixture()
        metrics = {'force_magnet_z_N': TARGET_FORCE_N, 'force_coil_z_N': -TARGET_FORCE_N,
                   'force_coil_wst_z_N': -TARGET_FORCE_N}
        self.assertEqual(assess_vm270(plan, metrics)['status'], 'passed')
        for key in metrics:
            for value in (TARGET_FORCE_N*1.1, -TARGET_FORCE_N*1.1):
                changed = dict(metrics, **{key: value})
                self.assertEqual(assess_vm270(plan, changed)['status'], 'failed')
        changed = dict(metrics, force_magnet_z_N=TARGET_FORCE_N*1.015, force_coil_z_N=-TARGET_FORCE_N*.985)
        self.assertEqual(assess_vm270(plan, changed)['status'], 'failed')
        plan['frozen_reference'].update(applicable=False, reason='Edited displacement')
        self.assertEqual(assess_vm270(plan, {})['status'], 'unassessed')

    def test_saved_mesh_and_material_defaults_must_match_frozen_settings(self):
        plan, actual, _ = fixture()
        for index in range(5):
            changed=deepcopy(actual)
            changed['labels'][index]['attributes'][1]=str(float(changed['labels'][index]['attributes'][1])*100)
            with self.subTest(label=index),self.assertRaisesRegex(ValueError,'region assignment'):
                verify_vm270(plan,changed)
        changed=deepcopy(actual);changed['arcs'][0]['attributes'][1]='90'
        with self.assertRaisesRegex(ValueError,'arc subdivision'):
            verify_vm270(plan,changed)
        for index in range(3):
            for key,value in [('LamType','1'),('LamFill','.5'),('J_im','1')]:
                changed=deepcopy(actual);changed['materials'][index]['properties'][key]=value
                with self.subTest(material=index,property=key),self.assertRaisesRegex(ValueError,'material law'):
                    verify_vm270(plan,changed)
        # Refinement scales the four local regions, while exterior air and
        # the boundary's one-degree subdivision remain fixed by this recipe.
        plan['parameters']['mesh_scale']=.5
        refined=deepcopy(actual)
        for label in refined['labels'][:4]:
            label['attributes'][1]=str(float(label['attributes'][1])*.5)
        self.assertEqual(verify_vm270(plan,refined)['status'],'passed')
        refined['labels'][-1]['attributes'][1]='.05'
        with self.assertRaisesRegex(ValueError,'region assignment'):
            verify_vm270(plan,refined)

    def test_metric_definitions_are_axial_force_not_radial_force(self):
        text = '\n'.join(metric_lines(DEFAULTS))
        self.assertIn('magnet_fz=mo_blockintegral(19)', text)
        self.assertIn('coil_lorentz_fz=mo_blockintegral(12)', text)
        self.assertNotIn('mo_blockintegral(18)', text)
        self.assertIn('force_magnet_z_N', text)

    def test_air_mesh_interfaces_never_cut_copper_at_small_displacement(self):
        for displacement in (0, .01, .1, .234, 5):
            p = dict(DEFAULTS, displacement=displacement)
            geometry, *_ = build_vm270({}, p)
            q = positions(p)
            for seg in geometry['segments']:
                a, b = (geometry['nodes'][seg[k]] for k in ('n0', 'n1'))
                # A horizontal segment that reaches into the winding radial
                # span may only lie on or outside its axial bounds.
                if a['y'] == b['y'] and max(a['x'], b['x']) > q['coil_inner_radius']:
                    self.assertFalse(q['coil_start'] < a['y'] < q['coil_end'])

    def test_frozen_pilot_baseline_lua_unchanged_after_edited_displacement_guard(self):
        geometry, lines, *_ = build_vm270({}, DEFAULTS)
        q = positions(DEFAULTS)
        self.assertTrue(any(abs(n['y']-1.9) < 1e-12 for n in geometry['nodes']))
        self.assertEqual(q['coil_start'], 1.961)
        self.assertEqual([line for line in lines if line.startswith('mi_setblockprop("air"')][-1],
                         'mi_setblockprop("air",0,0.10000000000000001,"<None>",0,0,1)')

    def test_invalid_values_rejected_before_femm(self):
        for p in ({'current': True}, {'mesh_scale': 0}, {'domain_radius': float('nan')}, {'displacement': -1}, {'turns': 5}):
            with self.assertRaises(ValueError):
                build_vm270({}, p)


if __name__ == '__main__':
    unittest.main()
