"""Re-read saved real FEMM output; no simulator or physics is mocked."""
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from femm_brain2.catalogue import load_batch
from femm_brain2.knowledge import install, load_knowledge, plan_from_store, search
from femm_brain2.knowledge_run import audit
from femm_brain2.source import fingerprint

PACKAGE=Path(__file__).resolve().parents[1]
EVIDENCE=PACKAGE/'data/verification/knowledge-lubin-2011-20261125-run3'


class KnowledgeStoreTests(unittest.TestCase):
    def test_sqlite_retrieval_resolves_shared_procedures_and_real_api_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'brain.sqlite3'
            with closing(sqlite3.connect(path)) as db, db:
                db.executescript((PACKAGE/'schema.sql').read_text())
                for record in load_batch()['commands']:
                    db.execute('INSERT INTO commands(name,version,binding,physics,category,data) VALUES (?,?,?,?,?,?)',
                               tuple(record[k] for k in ['name','version','binding','physics','category'])+(json.dumps(record),))
                knowledge=load_knowledge();install(db,knowledge)
            self.assertIn('semi_closed_radial_slot',[n['id'] for n in search('semi closed slot',path)['results']])
            plan=plan_from_store(knowledge['papers'][0]['id'],path)
            self.assertEqual(len(plan['steps']),6)
            self.assertEqual(len({s['id'] for s in plan['steps']}),6)
            contracts={a['name']:a for step in plan['steps'] for a in step['api_contracts']}
            self.assertEqual(contracts['mi_addarc']['signature'],'mi_addarc(x1, y1, x2, y2, angle, maxseg)')


class KnowledgeEvidenceTests(unittest.TestCase):
    def test_real_paper_scope_and_all_controls_pass(self):
        result=audit(EVIDENCE/'report.json',load_knowledge())
        self.assertEqual(result['status'],'passed')
        report=json.loads((EVIDENCE/'report.json').read_text())
        self.assertEqual([c['id'] for c in report['checks']],['pilot','baseline','refined_mesh','zero_magnet','gauge_shift'])
        self.assertTrue(all(e['returncode']==0 and e['network']=='none' for e in report['engines']))
        values=result['assessment']['measurements']
        self.assertGreater(values['emf_observed_V'],0.5)
        self.assertLessEqual(values['emf_absolute_error_V'],values['emf_tolerance_V'])
        self.assertLess(values['mesh_relative_change'],0.01)
        self.assertLess(values['zero_source_flux_Wb'],1e-10)
        self.assertEqual(len(result['procedures']),6)

    def test_changed_source_parameter_cannot_reuse_native_evidence(self):
        knowledge=deepcopy(load_knowledge())
        knowledge['papers'][0]['parameters']['r3_cm']['value']+=0.01
        with self.assertRaisesRegex(ValueError,'another|belongs'):
            audit(EVIDENCE/'report.json',knowledge)

    def test_rehashed_node_group_corruption_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'evidence';shutil.copytree(EVIDENCE,folder,ignore=shutil.ignore_patterns('binding.lua'))
            path=folder/'pilot/model-000.fem';lines=path.read_text().splitlines()
            i=next(i for i,line in enumerate(lines) if line.startswith('[NumPoints]'))+1
            fields=lines[i].split();fields[3]='99';lines[i]='\t'.join(fields)
            path.write_text('\n'.join(lines)+'\n')
            report=json.loads((folder/'report.json').read_text())
            report['artifacts']['pilot/model-000.fem']=fingerprint(path.read_bytes())
            (folder/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'do not agree'):
                audit(folder/'report.json',load_knowledge())

    def test_original_failures_and_frozen_physical_targets_are_preserved(self):
        for name in ['run1','run2']:
            folder=EVIDENCE.parent/('knowledge-lubin-2011-20261125-'+name)
            self.assertEqual(json.loads((folder/'report.json').read_text())['status'],'failed')
        previous=json.loads((EVIDENCE.parent/'knowledge-lubin-2011-20261125-run2/knowledge.json').read_text())['papers'][0]
        current=load_knowledge()['papers'][0]
        for key in ['parameters','acceptance','controls','bindings','geometry_expectations']:
            self.assertEqual(previous[key],current[key])
