"""Typed source data must survive reconstruction without losing physical meaning."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from femm_brain.simulation import plan_simulation, verify_native_physics
from femm_brain.source_recipe import compile_source, load_source, verify_source_physics

DATA = Path(__file__).parents[1] / 'data' / 'models'
EI = 'cea7acd3ed0296ea4531f7d8b6693adfa87683dd5c6857f4d76231042963cabb'
BEARING = '85069a0505f8e430f345ea74271def0f8f322f78db8dd9fa8d11a46c3413851e'


class SourceRecipeTests(unittest.TestCase):
    def source(self, ident=EI):
        return json.loads((DATA / (ident + '.json')).read_text())

    def test_frozen_data_digest_blocks_modified_source_before_execution(self):
        raw = (DATA / (EI + '.json')).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.json'; path.write_bytes(raw)
            self.assertEqual(load_source(path, EI, digest)['original_sha256'], EI)
            path.write_bytes(raw + b' ')
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_source(path, EI, digest)
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                load_source(path, 'incorrect-original', digest)

    def test_operations_are_generated_from_typed_entities_not_source_code(self):
        source = self.source()
        source['operations'] = [{'function': 'os.execute', 'arguments': ['malicious']}]
        geometry, lines, expected, units = compile_source(source)
        script = '\n'.join(lines)
        self.assertNotIn('os.execute', script)
        self.assertNotIn('malicious', script)
        self.assertEqual(units, .0254)
        self.assertEqual((len(geometry['nodes']), len(geometry['segments']), len(geometry['labels'])), (20, 22, 5))
        self.assertEqual(verify_source_physics(expected, self.source())['status'], 'passed')

    def test_nonlinear_curve_and_all_signed_windings_are_preserved(self):
        source = self.source(BEARING)
        geometry, lines, expected, _ = compile_source(source)
        self.assertEqual(len(geometry['arcs']), 22)
        self.assertEqual(sum(len(m['bh_points']) for m in expected['materials']), 13)
        self.assertEqual(sum('mi_addbhpoint(' in line for line in lines), 13)
        windings = [l for l in expected['labels'] if l['circuit_name']]
        self.assertEqual(len(windings), 16)
        self.assertEqual(sorted(l['turns'] for l in windings), [-80] * 8 + [80] * 8)
        self.assertEqual(verify_source_physics(expected, source)['nonlinear_bh_points'], 13)
        actual = deepcopy(source); actual['materials'][4]['bh_points'][6][1] += 1
        with self.assertRaisesRegex(ValueError, 'B-H'):
            verify_source_physics(expected, actual)

    def test_saved_physics_check_detects_source_depth_material_boundary_or_winding_corruption(self):
        source = self.source(); _, _, expected, _ = compile_source(source)
        alterations = [
            lambda x: x['metadata'].update(Depth='25.4'),
            lambda x: x['materials'][0]['properties'].update(Mu_x='2499'),
            lambda x: x['materials'][2]['properties'].update(LamType='0'),
            lambda x: x['circuits'][0]['properties'].update(TotalAmps_im='1'),
            lambda x: x['boundaries'][0]['properties'].update(A_0='1'),
            lambda x: x['segments'][14]['attributes'].__setitem__(1, '0'),
            lambda x: x['segments'][0]['attributes'].__setitem__(1, '1'),
            lambda x: x['labels'][2]['attributes'].__setitem__(5, '-66'),
            lambda x: x['labels'][2]['attributes'].__setitem__(2, '0'),
            lambda x: x['labels'][2]['attributes'].__setitem__(1, '.025'),
        ]
        for alteration in alterations:
            actual = deepcopy(source); alteration(actual)
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                verify_source_physics(expected, actual)
        # Coordinate/name binding is independent of saved contour/label ordering.
        reordered = deepcopy(source)
        reordered['segments'].reverse(); reordered['labels'].reverse()
        self.assertEqual(verify_source_physics(expected, reordered)['status'], 'passed')

    def test_compiler_rejects_unhandled_source_physics_instead_of_silently_discarding_it(self):
        alterations = [
            lambda x: x['metadata'].update(Frequency='50'),
            lambda x: x['metadata'].update(Coordinates='polar'),
            lambda x: x['materials'][0]['properties'].update(UnknownLaw='1'),
            lambda x: x['materials'][0]['properties'].update(J_im='1'),
            lambda x: x['boundaries'][0]['properties'].update(BdryType='4'),
            lambda x: x['nodes'][0]['attributes'].__setitem__(0, '1'),
            lambda x: x['labels'][0]['attributes'].__setitem__(6, '1'),
            lambda x: x['labels'][0]['attributes'].__setitem__(0, '999'),
        ]
        for alteration in alterations:
            source = self.source(); alteration(source)
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                compile_source(source)
        source = self.source(BEARING)
        source['materials'][4]['bh_points'][6][1] = float('nan')
        with self.assertRaises(ValueError): compile_source(source)

    def test_parameter_changes_are_frozen_in_independent_expectations(self):
        plan = plan_simulation('gapped-ei-core', {'current': -2, 'mesh_scale': .5})
        expected = plan['expected_source_physics']
        self.assertEqual(expected['circuits'][0]['properties']['TotalAmps_re'], -2)
        self.assertEqual(expected['labels'][0]['mesh'], .025)
        self.assertEqual(expected['labels'][2]['turns'], 66)
        self.assertEqual(expected['labels'][3]['turns'], -66)
        self.assertTrue(plan['source_recipe_sha256'])
        source = self.source()
        with self.assertRaisesRegex(ValueError, 'property differs'):
            verify_native_physics(plan, source)


if __name__ == '__main__': unittest.main()
