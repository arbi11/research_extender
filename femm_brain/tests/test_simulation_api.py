"""HTTP boundaries for offline planning and explicitly enabled local solves."""
from functools import partial
from http.server import ThreadingHTTPServer
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.parse import urlencode

from femm_brain.core import load_graph
from femm_brain.server import Handler
from femm_brain.simulation import plan_simulation
from femm_brain.store import GraphStore


class RecordingSolver:
    def __init__(self):
        self.calls = []

    def submit(self, case_id, parameters, *, expected_plan_sha256=None):
        plan = plan_simulation(case_id, parameters)
        if expected_plan_sha256 and expected_plan_sha256 != plan['plan_sha256']:
            raise ValueError('Frozen plan changed')
        self.calls.append(plan)
        return {'id': 'local-test-job', 'status': 'queued', 'plan_sha256': plan['plan_sha256']}

    def get(self, identifier):
        return None

    def artifact(self, *parts):
        return None


class SimulationApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.graph = load_graph()
        cls.store = GraphStore(Path(cls.temp.name) / 'index.sqlite3')
        cls.store.replace(cls.graph)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.solver = RecordingSolver()
        self.handler = partial(Handler, graph=self.graph, store=self.store, solver=self.solver)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), self.handler)
        self.port = self.server.server_port
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        try:
            connection.request(method, path, body, headers or {})
            response = connection.getresponse()
            raw = response.read()
            return response.status, dict(response.getheaders()), raw
        finally:
            connection.close()

    def get_json(self, path):
        status, _, body = self.request('GET', path)
        self.assertEqual(status, 200, body)
        return json.loads(body)

    def post(self, payload, **headers):
        return self.request('POST', '/api/simulations', json.dumps(payload),
                            {'Content-Type': 'application/json', **headers})

    def test_catalog_and_local_language_plan_export_match(self):
        catalog = self.get_json('/api/simulations')
        self.assertTrue({'gapped-ei-core', 'radial-bearing'} <= {case['id'] for case in catalog['cases']})
        self.assertGreaterEqual(len(catalog['cases']), 8)
        self.assertEqual(catalog['llm_calls'], 0)
        verified = self.get_json('/api/verification')['native_simulations']
        self.assertEqual(len(verified['cases']), 6)
        self.assertEqual(verified['llm_calls'], 0)
        self.assertTrue(all(case['status'] == 'passed' for case in verified['cases']))
        engineering = self.get_json('/api/verification')['engineering_benchmarks']
        self.assertFalse(engineering['audit']['all_published_targets_passed'])
        ei = next(case for case in engineering['cases'] if case['case_id'] == 'gapped-ei-core')
        self.assertEqual(ei['checks']['solve']['status'], 'passed')
        self.assertEqual(ei['checks']['target']['status'], 'failed')
        selection = self.get_json('/api/simulation-request?' + urlencode({'q': 'AirCore short solenoid at 2 A'}))
        self.assertEqual(selection['status'], 'ready')
        query = urlencode({'case': selection['selected_case_id'], 'parameters': json.dumps(selection['parameters'])})
        plan = self.get_json('/api/simulation-plan?' + query)
        self.assertEqual(plan['parameters']['current'], 2)
        self.assertIn('mi_analyze', plan['lua'])
        self.assertTrue(plan['knowledge_sources'])
        status, headers, lua = self.request('GET', '/api/simulation-download?' + query + '&format=lua')
        self.assertEqual(status, 200)
        self.assertEqual(lua.decode(), plan['lua'])
        self.assertIn('attachment', headers['Content-Disposition'])
        self.assertEqual(self.request('GET', '/api/simulation-download?' + query + '&plan_sha256=stale')[0], 400)
        self.assertFalse(self.solver.calls, 'GET planning must never launch FEMM')

    def test_geometry_edit_does_not_inherit_published_inductance_target(self):
        query = urlencode({'case': 'aircore-short', 'parameters': json.dumps({'turns': 200})})
        plan = self.get_json('/api/simulation-plan?' + query)
        self.assertFalse(plan['frozen_reference']['applicable'])
        self.assertIn('turns', plan['frozen_reference']['reason'])

    def test_explicit_local_submission_carries_frozen_plan_hash(self):
        plan = plan_simulation('aircore-short')
        status, _, body = self.post({'case_id': 'aircore-short', 'plan_sha256': plan['plan_sha256']},
                                    Origin='http://127.0.0.1:' + str(self.port))
        self.assertEqual(status, 202, body)
        self.assertEqual(json.loads(body)['plan_sha256'], plan['plan_sha256'])
        self.assertEqual(len(self.solver.calls), 1)
        self.assertEqual(self.post({'case_id': 'aircore-short', 'plan_sha256': 'wrong'})[0], 400)
        self.assertEqual(len(self.solver.calls), 1)

    def test_solver_is_disabled_by_default(self):
        # Same handler default as the ordinary browser-only `serve` command.
        self.server.RequestHandlerClass = partial(Handler, graph=self.graph, store=self.store)
        self.assertFalse(self.get_json('/api/simulations')['solver_enabled'])
        status, _, body = self.post({'case_id': 'aircore-short'})
        self.assertEqual(status, 503, body)
        self.assertFalse(self.solver.calls)

    def test_cross_origin_or_remote_host_cannot_start_a_solve(self):
        payload = {'case_id': 'aircore-short'}
        for headers in ({'Origin': 'https://unrelated.example'},
                        {'Host': 'unrelated.example:' + str(self.port)},
                        {'Host': '127.0.0.1:1'}, {'Host': '[broken'}):
            with self.subTest(headers=headers):
                self.assertIn(self.post(payload, **headers)[0], (400, 403))
        self.assertFalse(self.solver.calls)

    def test_untyped_requests_unknown_cases_and_arbitrary_code_are_rejected(self):
        for payload in ([], {'case_id': 'missing'}, {'case_id': 'aircore-short', 'lua': 'os.execute()'},
                        {'case_id': 'aircore-short', 'parameters': {'current': 'oops'}},
                        {'case_id': 'aircore-short', 'parameters': {'unexpected': 3}}):
            with self.subTest(payload=payload):
                self.assertEqual(self.post(payload)[0], 400)
        self.assertEqual(self.request('POST', '/api/simulations', '{}', {'Content-Type': 'text/plain'})[0], 415)
        self.assertEqual(self.request('POST', '/api/simulations', ' ' * 16385,
                                      {'Content-Type': 'application/json'})[0], 400)
        self.assertFalse(self.solver.calls)

    def test_unknown_jobs_artifacts_exports_and_traversal_fail(self):
        for path in ('/api/simulation-jobs/missing', '/api/simulation-artifacts/missing/report.json',
                     '/api/simulation-artifacts/missing/%2e%2e%2f.env',
                     '/api/simulation-plan?case=missing',
                     '/api/simulation-download?case=aircore-short&format=shell'):
            with self.subTest(path=path):
                self.assertIn(self.request('GET', path)[0], (400, 404))


if __name__ == '__main__':
    unittest.main()
