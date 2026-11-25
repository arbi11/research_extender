"""Offline software tests use synthetic documentation; no physics is mocked or scored."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from femm_brain2.__main__ import main
from femm_brain2.catalogue import build, get, load_batch, search, stats, validate_batch
from femm_brain2.source import fingerprint, lua_definitions, manual_definitions


class CatalogueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / 'cache'
        self.db = self.root / 'catalogue #1.sqlite3'
        manual = ('3.5 Common Commands\n'
                  ' • newdocument(doctype) Synthetic document definition.\n 1\n\f'
                  '3.6 Magnetics Preprocessor\n'
                  ' • mi_addarc(x1,y1,x2,y2,angle,maxseg) Synthetic arc definition.\n'
                  ' • mi_selectnode(x,y) Synthetic pending definition.\n 2\n')
        lua = ('function newdocument(doctype)\n return callfemm("create",doctype)\nend\n'
               'function mi_addarc(x1,y1,x2,y2,angle,maxseg)\n return callfemm("mi_addarc")\nend\n')
        self.sources = []
        definitions = {}
        for identity, role, text in (('manual', 'api_manual', manual), ('lua', 'lua_binding', lua)):
            digest = fingerprint(text.encode())
            folder = self.cache / digest
            folder.mkdir(parents=True)
            (folder / 'source.txt').write_text(text)
            self.sources.append({'id': identity, 'role': role, 'format': 'text', 'filename': 'source.txt',
                                 'sha256': digest, 'title': 'Synthetic test source', 'url': 'https://femm.info/test'})
            definitions[identity] = manual_definitions(text) if role == 'api_manual' else lua_definitions(text)
        self.batch = deepcopy(load_batch())
        self.batch['commands'] = [r for r in self.batch['commands'] if r['name'] in ('newdocument', 'mi_addarc')]
        for record in self.batch['commands']:
            record['references'] = [{'source_id': identity, 'anchor': record['name'],
                                     'pdf_pages': list(values[record['name']].pages),
                                     'definition_sha256': values[record['name']].digest}
                                    for identity, values in definitions.items()]

    def built(self):
        batch_path = self.cache.parent / 'batch.json'
        sources_path = self.cache.parent / 'sources.json'
        batch_path.write_text(json.dumps(self.batch))
        sources_path.write_text(json.dumps({'sources': self.sources}))
        return build(self.db, batch_path=batch_path, sources_path=sources_path, cache=self.cache)

    def test_search_returns_source_linked_matches_and_honours_filters(self):
        self.built()
        rows = search('how do I draw an arc', self.db, physics='magnetics', binding='lua4', version='5.0')
        self.assertEqual([r['name'] for r in rows], ['mi_addarc'])
        self.assertEqual(rows[0]['references'][0]['url'], 'https://femm.info/test')
        self.assertEqual(search('arc', self.db, physics='electrostatics'), [])
        self.assertEqual(search('arc', self.db, version='4.2'), [])

    def test_get_distinguishes_pending_missing_and_other_versions(self):
        self.built()
        self.assertEqual(get('mi_addarc', self.db)['review']['status'], 'source_checked')
        self.assertEqual(get('mi_selectnode', self.db)['status'], 'not_reviewed')
        self.assertEqual(get('invented_command', self.db)['status'], 'not_found')
        self.assertEqual(get('mi_addarc', self.db, version='4.2')['status'], 'not_found')

    def test_lookup_works_without_the_source_cache(self):
        self.built()
        shutil.rmtree(self.cache)
        self.assertEqual(get('mi_addarc', self.db)['name'], 'mi_addarc')
        self.assertEqual(stats(self.db)['execution_tested_commands'], 0)

    def test_source_tampering_is_rejected(self):
        source = self.sources[0]
        (self.cache / source['sha256'] / source['filename']).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Cached source hash mismatch'):
            validate_batch(self.batch, self.sources, self.cache)

    def test_bad_fingerprint_and_locator_are_rejected(self):
        for field, value in (('definition_sha256', '0' * 64), ('pdf_pages', [999])):
            batch = deepcopy(self.batch)
            batch['commands'][0]['references'][0][field] = value
            self.assertEqual(validate_batch(batch, self.sources, self.cache)['status'], 'failed')

    def test_signature_order_and_undocumented_defaults_are_checked(self):
        batch = deepcopy(self.batch)
        arc = next(r for r in batch['commands'] if r['name'] == 'mi_addarc')
        arc['parameters'].reverse()
        self.assertEqual(validate_batch(batch, self.sources, self.cache)['status'], 'failed')
        batch = deepcopy(self.batch)
        batch['commands'][0]['parameters'][0]['default'] = 1
        self.assertEqual(validate_batch(batch, self.sources, self.cache)['status'], 'failed')

    def test_documentation_import_cannot_assert_execution_success(self):
        self.batch['commands'][0]['execution']['status'] = 'passed'
        result = validate_batch(self.batch, self.sources, self.cache)
        self.assertTrue(any('execution success' in error for error in result['errors']))

    def test_failed_rebuild_preserves_the_database(self):
        self.built()
        before = self.db.read_bytes()
        self.batch['commands'][0]['references'][0]['pdf_pages'] = [999]
        with self.assertRaises(ValueError):
            self.built()
        self.assertEqual(self.db.read_bytes(), before)

    def test_unrelated_database_is_never_overwritten(self):
        with sqlite3.connect(self.db) as connection:
            connection.execute('CREATE TABLE other_project (value TEXT)')
        before = self.db.read_bytes()
        with self.assertRaisesRegex(ValueError, 'not a femm_brain2'):
            self.built()
        self.assertEqual(self.db.read_bytes(), before)

    def test_cli_returns_json_and_missing_database_is_not_created(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            status = main(['get', 'mi_addarc', '--db', str(self.db)])
        self.assertEqual(status, 2)
        self.assertEqual(json.loads(stream.getvalue())['status'], 'error')
        self.assertFalse(self.db.exists())
        self.built()
        stream = io.StringIO()
        with redirect_stdout(stream):
            status = main(['search', 'draw arc', '--db', str(self.db)])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stream.getvalue())['results'][0]['name'], 'mi_addarc')

    def test_query_punctuation_does_not_become_sql(self):
        self.built()
        self.assertEqual(get("mi_addarc' OR 1=1 --", self.db)['status'], 'not_found')
        search('arc"; DROP TABLE commands; --', self.db)
        self.assertEqual(stats(self.db)['reviewed_commands'], 2)

    def test_parser_preserves_multpage_definition_and_duplicate_locations(self):
        text = ('3.7 Postprocessing\n • mo_getpointvalues(x,y) Synthetic definition.\n 1\n\f'
                ' Return values continue here.\n • mo_getb(x,y) Next command.\n'
                ' • mo_getb(x,y) Another documented occurrence.\n 2\n')
        definitions = manual_definitions(text)
        self.assertEqual(definitions['mo_getpointvalues'].pages, (1, 2))
        self.assertIn('Return values continue', definitions['mo_getpointvalues'].text)
        self.assertEqual(len(definitions['mo_getb'].additional_pages), 1)

    def test_review_batch_retains_full_point_value_return_order_and_resolves_arc(self):
        records = {r['name']: r for r in load_batch()['commands']}
        self.assertEqual(len(records), 38)
        self.assertEqual([r['name'] for r in records['mo_getpointvalues']['returns']['values']],
                         ['A', 'B1', 'B2', 'Sig', 'E', 'H1', 'H2', 'Je', 'Js', 'Mu1', 'Mu2', 'Pe', 'Ph', 'ff'])
        self.assertEqual(records['mi_addarc']['review']['semantic_status'], 'reviewed')
        self.assertEqual(records['mi_addarc']['parameters'][-1]['units'], 'degrees_per_element')
        self.assertEqual(records['mi_copyrotate']['signature'], 'mi_copyrotate(bx, by, angle, copies)')
        self.assertNotIn('editaction', [p['name'] for p in records['mi_copyrotate']['parameters']])
        self.assertTrue(all(r['execution']['status'] == 'not_tested' for r in records.values()))


if __name__ == '__main__':
    unittest.main()
