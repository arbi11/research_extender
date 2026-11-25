"""Offline evidence integrity checks over saved, real FEMM output; no solver mocks."""
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest

from femm_brain2.catalogue import load_batch
from femm_brain2.geometry_check import evaluate, verify_evidence
from femm_brain2.source import fingerprint

EVIDENCE = Path(__file__).resolve().parents[1] / 'data' / 'verification'


class NativeEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name) / 'evidence'
        shutil.copytree(EVIDENCE / 'geometry-20261125-field-refresh', self.folder)
        self.path = self.folder / 'report.json'

    def test_verified_geometry_evidence_covers_33_commands_and_no_solve(self):
        executions = verify_evidence(self.path, load_batch())
        self.assertEqual(len(executions), 33)
        self.assertTrue(all(e['status'] == 'passed' for e in executions.values()))
        self.assertNotIn('mi_analyze', executions)

    def test_first_run_remains_a_recorded_14_of_16_failure(self):
        folder = EVIDENCE / 'geometry-20261125-run1'
        report = json.loads((folder / 'report.json').read_text())
        checks = evaluate(folder, json.loads((folder / 'protocol.json').read_text()))
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(sum(c['status'] == 'passed' for c in checks), 14)

    def test_native_model_tampering_is_rejected(self):
        file = self.folder / 'add_primitives.fem'
        file.write_text(file.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
            verify_evidence(self.path, load_batch())

    def test_report_labels_cannot_override_saved_models(self):
        report = json.loads(self.path.read_text())
        report['checks'][0]['status'] = 'failed'
        self.path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'observations do not agree'):
            verify_evidence(self.path, load_batch())

    def test_existing_native_models_must_have_hashes(self):
        report = json.loads(self.path.read_text())
        report['artifacts'].pop('add_primitives.fem')
        self.path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'unhashed model'):
            verify_evidence(self.path, load_batch())

    def test_native_evidence_cannot_be_attached_to_another_batch(self):
        batch = load_batch()
        batch['commands'][0]['purpose'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'another reviewed command batch'):
            verify_evidence(self.path, batch)


class PropertyEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name) / 'properties'
        shutil.copytree(EVIDENCE / 'properties-20261125-field-refresh', self.folder)
        self.path = self.folder / 'report.json'

    def test_native_property_evidence_covers_15_new_commands(self):
        executions = verify_evidence(self.path, load_batch())
        self.assertEqual(len(executions), 15)
        self.assertTrue(all(e['status'] == 'passed' for e in executions.values()))
        self.assertIn('mi_getmaterial', executions)
        self.assertIn('mi_modifycircprop', executions)
        self.assertIn('mi_addpointprop', executions)

    def test_native_observations_include_complex_currents_and_reopened_assignments(self):
        report = json.loads(self.path.read_text())
        checks = {c['id']: c for c in report['checks']}
        circuits = checks['circuit_fields']['observed']['properties']['circuits']
        self.assertTrue(any(c['TotalAmps_im'] != 0 for c in circuits))
        self.assertEqual(checks['assignments']['observed'], checks['reopen_assignments']['observed'])

    def test_changed_property_is_rejected_even_with_a_recomputed_artifact_hash(self):
        model = self.folder / 'material_fields.fem'
        text = model.read_text()
        changed = re.sub(r'(?m)^(\s*<Mu_x>\s*=\s*).+$', r'\g<1>123456', text, count=1)
        self.assertNotEqual(text, changed)
        model.write_text(changed)
        report = json.loads(self.path.read_text())
        report['artifacts'][model.name] = fingerprint(model.read_bytes())
        self.path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'observations do not agree'):
            verify_evidence(self.path, load_batch())


if __name__ == '__main__':
    unittest.main()
