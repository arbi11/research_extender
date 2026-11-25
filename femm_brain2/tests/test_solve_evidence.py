"""Integrity and independent solution checks over real saved FEMM solver output."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from femm_brain2.catalogue import load_batch
from femm_brain2.geometry_check import evaluate, verify_evidence
from femm_brain2.solve_check import evaluate_solution
from femm_brain2.source import fingerprint

EVIDENCE = Path(__file__).resolve().parents[1] / 'data/verification/solve-20261125-mesh-count-fixed'


class SolveEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name) / 'solve'
        shutil.copytree(EVIDENCE, self.folder, ignore=shutil.ignore_patterns('binding.lua'))
        self.path = self.folder / 'report.json'
        self.protocol = json.loads((self.folder / 'protocol.json').read_text())

    def test_real_mesh_count_return_matches_solver_nodes_in_every_control(self):
        executions = verify_evidence(self.path, load_batch())
        self.assertEqual(len(executions), 16)
        self.assertTrue(all(r['status'] == 'passed' for r in executions.values()))
        report = json.loads(self.path.read_text())
        self.assertEqual(report['status'], 'passed')
        for check in report['checks']:
            solution = check['observed']['solution']
            values = solution['values']
            self.assertEqual(values['mesh.has_return'], [1.0, 0.0])
            self.assertEqual(values['mesh.created_nodes'], values['mesh.solved_nodes'])
            self.assertEqual(values['mesh.created_nodes'][0], solution['solution_mesh']['nodes'])
            self.assertNotEqual(values['mesh.created_nodes'], values['mesh.solved_elements'])

    def test_original_nil_return_remains_a_recorded_failure(self):
        folder = EVIDENCE.parent / 'solve-20261125-run3'
        report = json.loads((folder / 'report.json').read_text())
        checks = evaluate(folder, json.loads((folder / 'protocol.json').read_text()))
        self.assertEqual(report['status'], 'failed')
        self.assertEqual([c['id'] for c in checks if c['status'] == 'failed'], ['explicit_mesh_return'])

    def test_runtime_build_manifest_tampering_is_rejected(self):
        protocol = json.loads((self.folder / 'protocol.json').read_text())
        filename = protocol['runtime_manifest']['file']
        file = self.folder / filename
        content = json.loads(file.read_text())
        content['femm_headless_sha256'] = '0' * 64
        file.write_text(json.dumps(content))
        report = json.loads(self.path.read_text())
        report['artifacts'][filename] = fingerprint(file.read_bytes())
        self.path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'Runtime build manifest differs'):
            verify_evidence(self.path, load_batch())

    def test_mesh_fix_keeps_original_physical_inputs_and_analytical_targets(self):
        old = json.loads((EVIDENCE.parent / 'solve-20261125-run3/protocol.json').read_text())
        original = {c['id']: c for c in old['cases']}
        self.assertNotEqual(old['reference_image'], self.protocol['reference_image'])
        for case in self.protocol['cases']:
            for field in ['physical_inputs', 'expected_values', 'analytical', 'expected_headers', 'expected_counts']:
                self.assertEqual(case[field], original[case['id']][field])

    def test_field_polarity_and_gauge_offset_controls_match_physics(self):
        checks = {c['id']: c for c in json.loads(self.path.read_text())['checks']}
        values = {n: checks[n]['observed']['solution']['values'] for n in ['positive', 'negative', 'zero']}
        self.assertAlmostEqual(values['positive']['p0.B1'][0], 0.2, delta=1e-6)
        self.assertAlmostEqual(values['negative']['p0.B1'][0], -0.2, delta=1e-6)
        self.assertAlmostEqual(values['zero']['p0.B1'][0], 0, delta=1e-6)
        self.assertAlmostEqual(values['positive']['circuit.linkage'][0], 0.0002, delta=1e-8)
        self.assertAlmostEqual(values['negative']['circuit.linkage'][0], -0.0002, delta=1e-8)
        self.assertAlmostEqual(values['zero']['all.energy'][0], 0, delta=5e-6)

    def test_changed_solver_potential_is_rejected_even_when_query_output_is_unchanged(self):
        case = self.protocol['cases'][0]
        file = self.folder / case['solution_file']
        lines = file.read_text().splitlines()
        marker = next(i for i, line in enumerate(lines) if line.strip().lower() == '[solution]')
        first_node = marker + 2
        fields = lines[first_node].split()
        fields[2] = format(float(fields[2]) + 0.0001, '.17g')
        lines[first_node] = ' '.join(fields)
        file.write_text('\n'.join(lines) + '\n')
        observed, failures = evaluate_solution(self.folder, case)
        self.assertGreater(observed['solution_mesh']['max_potential_error'], case['analytical']['potential_tolerance'])
        self.assertIn('Solver nodal potentials differ from analytical A', failures)
        report = json.loads(self.path.read_text())
        report['artifacts'][file.name] = fingerprint(file.read_bytes())
        self.path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'observations do not agree'):
            verify_evidence(self.path, load_batch())


if __name__ == '__main__':
    unittest.main()
