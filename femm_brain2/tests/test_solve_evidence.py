"""Integrity and independent solution checks over real saved FEMM solver output."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from femm_brain2.catalogue import load_batch
from femm_brain2.geometry_check import verify_evidence
from femm_brain2.solve_check import evaluate_solution
from femm_brain2.source import fingerprint

EVIDENCE = Path(__file__).resolve().parents[1] / 'data/verification/solve-20261125-run3'


class SolveEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name) / 'solve'
        shutil.copytree(EVIDENCE, self.folder, ignore=shutil.ignore_patterns('binding.lua'))
        self.path = self.folder / 'report.json'
        self.protocol = json.loads((self.folder / 'protocol.json').read_text())

    def test_three_analytical_solutions_pass_and_mesh_return_failure_stays_visible(self):
        executions = verify_evidence(self.path, load_batch())
        self.assertEqual(len(executions), 16)
        self.assertEqual([n for n, e in executions.items() if e['status'] != 'passed'], ['mi_createmesh'])
        report = json.loads(self.path.read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual([c['id'] for c in report['checks'] if c['status'] == 'passed'], ['positive', 'negative', 'zero'])

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
