"""Real-browser regression for paper provenance, targets and stale recovery UI.

The real API persists paper/revision/target/queue records. Explicit fixtures
replace PDF parsing, AI providers and native completion; no paid call or FEMM
process runs. Numerical/PDF correctness has separate native/unit coverage.
Run with the development Python that has Playwright and Chromium installed.
"""
import re
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

from browser_smoke import FIELD_FIXTURE, MESH_FIXTURE, MESH_REPORT, ROOT, TOKEN, wait_for_server

SERVER = '''
import hashlib,sys
from pathlib import Path
import uvicorn
from app import papers
from app.main import Settings,create_app
# Browser fixture isolates the UI/storage boundaries from PDF-library setup.
def document(data, filename):
    digest=hashlib.sha256(data).hexdigest()
    return papers.PaperDocument(source_id='paper_'+digest[:24],sha256=digest,filename=filename,
      page_count=1,pages=(papers.PaperPage(number=1,text='Current 1 A. Current 2 A. Inductance 1.73 mH. Inductance 3.46 mH.'),))
papers.extract_pdf=document
settings=Settings(data_dir=Path(sys.argv[1]),token='browser-pilot',ai_key='browser-fixture',
 ai_model='browser-fixture',ai_base_url='http://127.0.0.1:9/fixture',femm_lua='python3',start_worker=False)
uvicorn.run(create_app(settings),host='127.0.0.1',port=int(sys.argv[2]),log_level='warning')
'''


def exercise(url, folder):
    from playwright.sync_api import sync_playwright, expect
    sys.path.insert(0, str(ROOT))
    from app.jobs import JobStore
    from app.history import ModelRevisionStore
    from app.plan import GeometryPlan, wire_example
    from app.replication import ReplicationStore
    from app import provenance

    jobs = JobStore(folder / 'data')
    history = ModelRevisionStore(folder / 'data')
    papers = ReplicationStore(folder / 'data')
    mode = {'extraction': 'normal', 'late_revision': False, 'late_upload': False}
    revision_requests, solve_requests, recover_requests, browser_errors, planning_requests = [], [], [], [], []
    first_failed_job = None
    original_contract = None
    artifact_dir = Path(os.environ["BROWSER_REPLICATION_ARTIFACTS"]) if os.getenv("BROWSER_REPLICATION_ARTIFACTS") else None
    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    from test_papers import text_pdf
    paper_bytes = text_pdf('Current 1 A. Current 2 A. Inductance 1.73 mH. Inductance 3.46 mH.')

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1450, 'height': 1100})
        page.on('pageerror', lambda error: browser_errors.append(str(error)))
        page.set_default_timeout(10000)

        def upload(route):
            response = route.fetch()
            if mode['late_upload']:
                mode['late_upload'] = False
                page.locator('#paper-file').set_input_files({'name': 'replacement.pdf', 'mimeType': 'application/pdf', 'buffer': paper_bytes+b'new'})
            route.fulfill(response=response)

        def extract(route):
            identity = route.request.url.split('/')[-2]
            doc = papers.get_document(identity)
            citation = {'source_id': doc.source_id, 'page': 1,
                        'quote': doc.pages[0].text}
            evidence = {'source_id': doc.source_id, 'title': 'Browser paper fixture',
                'summary': 'Synthetic values for browser integration, not a scientific reference.',
                'parameters': [], 'targets': [], 'questions': [], 'limitations': []}
            for case, current, inductance in [('baseline', 1, 1.73), ('displaced', 2, 3.46)]:
                evidence['parameters'].append({'id': case+'_current', 'description': 'Circuit current',
                    'value': current, 'unit': 'A', 'evidence_kind': 'explicit', 'citations': [citation], 'case_id': case})
                evidence['targets'].append({'id': case+'_inductance', 'case_id': case,
                    'description': case+' inductance', 'observable': {'quantity': 'inductance', 'circuit_id': 'wire'},
                    'value': inductance, 'unit': 'mH', 'tolerance': {'relative': .02},
                    'reference_kind': 'reported_numerical', 'evidence_kind': 'explicit', 'citations': [citation],
                    'conditions': str(current)+' A, '+case+' geometry'})
            behavior = mode['extraction']; mode['extraction'] = 'normal'
            if behavior == 'error':
                route.fulfill(status=502, json={'detail': {'message': 'Evidence could not be verified.',
                    'audit_id': 'a'*64, 'validation_summary': [{'message': 'The cited unit does not match.'}], 'usage': {'cost': .001}}})
                return
            if behavior == 'partial':
                evidence['questions'] = ['Supply the excluded core dimension before recreating that feature.']
            record = papers.put_evidence(identity, evidence, model='explicit-browser-fixture')
            if behavior == 'partial':
                record = {**record, 'extraction_status': 'partial', 'rejected_claim_count': 1, 'audit_id': 'b'*64,
                    'validation_summary': [{'claim_id': 'excluded_dimension', 'message': 'The dimension unit was not supported by its citation.'}]}
            if behavior == 'clear':
                page.locator('#paper-clear').click()
            route.fulfill(json=record)

        def interpret(route):
            body = route.request.post_data_json
            planning_requests.append(body)
            spec = wire_example().model_dump(mode='json')
            if body['case_id'] == 'displaced':
                spec['title'] = 'Displaced paper case fixture'
                spec['circuits'][0]['current_a'] = 2.
                spec['regions'][0]['shape']['radius'] = 1.5
            planning = provenance.put_planning(papers, spec, body['evidence_id'], body['case_id'], body['prompt'],
                preserve_paper_excitation=body['preserve_paper_excitation'])
            route.fulfill(json={'spec': spec, 'planning_id': planning['planning_id'], 'explanation': 'Source-bound case; verify original conditions.',
                'questions': [], 'attempts': 1, 'function_name': 'create_plan', 'decision_trace': [],
                'paper_excitation_check': {'status': 'skipped', 'message': 'Excitation arithmetic is not applicable to this fixture; verify the cited conditions.'}})

        def revision(route):
            revision_requests.append(route.request.post_data_json)
            response = route.fetch()
            if mode['late_revision']:
                mode['late_revision'] = False
                expected = mode['late_revision_expected']
                assert route.request.post_data_json['spec']['title'] == expected['title'], 'Late-response fixture intercepted another model'
                assert route.request.post_data_json.get('parent_id') == expected['id'], 'Late-response fixture intercepted another model branch'
                replacement = wire_example().model_dump(mode='json')
                replacement['title'] = 'Fresh case while old save is pending'
                page.evaluate('spec => setPlan(spec, {fresh:true})', replacement)
            route.fulfill(response=response)

        def mesh(route):
            response = route.fetch()
            job = response.json()
            jobs.update(job['id'], 'completed', result={**MESH_REPORT, 'runtime': {'value': .01, 'unit': 's'}})
            route.fulfill(response=response)

        def solve(route):
            nonlocal first_failed_job, original_contract
            solve_requests.append(route.request.post_data_json)
            response = route.fetch()
            assert response.status == 202, response.text()
            job = response.json()
            contract = papers.get_contract(job['target_contract_id'])
            if original_contract is None:
                original_contract = contract
            passed = len(solve_requests) > 1
            report = {'status': 'passed' if passed else 'failed', 'checks': [{'target_id': 'displaced_inductance',
                'status': 'passed' if passed else 'failed', 'target': 3.46, 'unit': 'mH',
                'actual': 3.46 if passed else 3., 'permitted_error': .0692, 'error': None}]}
            check = history.record_check(job['revision_id'], stage='target', status=report['status'], job_id=job['id'], details={'fixture': 'Browser worker completion'})
            jobs.annotate(job['id'], target_report=report, check_ids=[check['id']], **({} if passed else {'failure_id': check['id']}))
            jobs.update(job['id'], 'completed', result={'energy': {'value': .001, 'unit': 'J'}, 'runtime': {'value': .01, 'unit': 's'}})
            if not passed:
                first_failed_job = jobs.get(job['id'])
            route.fulfill(response=response)

        def recover(route):
            recover_requests.append(route.request.post_data_json)
            job = jobs.get(route.request.url.split('/')[-2])
            branch = 'browser-recovery'
            restored = history.branch_recovery(job['evidence']['failure_id'], job['revision_id'], branch=branch, diagnosis='Explicit browser mesh-recovery fixture')
            repaired = history.apply_patch(restored['id'], [{'op': 'replace', 'path': '/mesh_mm', 'value': .4}], allowed_paths=['/mesh_mm'], branch=branch)
            repaired = {**repaired, 'branch': branch}
            contract = papers.get_contract(job['target_contract_id'])
            rebound = papers.put_contract(repaired['id'], contract['contract']['case_id'], contract['contract']['targets'],
                GeometryPlan.model_validate(repaired['plan']), evidence_id=contract['evidence_id'])
            mesh_job = jobs.create(GeometryPlan.model_validate(repaired['plan']), kind='mesh', revision_id=repaired['id'])
            jobs.update(mesh_job['id'], 'completed', result={**MESH_REPORT, 'runtime': {'value': .01, 'unit': 's'}})
            route.fulfill(status=202, json={'revision': repaired, 'mesh_job': mesh_job,
                'target_contract_id': rebound['contract_id'], 'proposal': {'explanation': 'Refined mesh; fixed targets retained.'}})

        page.route('**/api/papers', upload)
        page.route('**/api/papers/*/extract', extract)
        page.route('**/api/plan/interpret', interpret)
        page.route('**/api/revisions', revision)
        page.route('**/api/plan/mesh', mesh)
        page.route('**/api/plan/jobs', solve)
        page.route('**/api/jobs/*/recover', recover)
        page.route('**/api/jobs/*/mesh.svg', lambda route: route.fulfill(content_type='image/svg+xml', body=MESH_FIXTURE))
        page.route('**/api/jobs/*/field.svg', lambda route: route.fulfill(content_type='image/svg+xml', body=FIELD_FIXTURE))
        page.goto(url)
        page.locator('#token').fill(TOKEN)
        page.locator('#access-form button').click()
        page.locator('#paper-controls > summary').click()
        page.locator('#paper-file').set_input_files({'name': 'paper.pdf', 'mimeType': 'application/pdf', 'buffer': paper_bytes})
        page.locator('#paper-upload').click()
        expect(page.locator('#paper-status')).to_contain_text('pages read locally')
        page.locator('#paper-extract').click()
        expect(page.locator('#paper-evidence')).to_be_visible()
        expect(page.locator('#paper-preserve-excitation')).to_be_checked()
        expect(page.locator('#paper-target-conditions')).to_contain_text('1 A, baseline geometry')
        expect(page.locator('#compare-targets')).to_be_disabled()
        page.locator('#prompt').fill('Reproduce the selected paper case; preserve original conditions.')
        page.locator('#interpret').click()
        expect(page.locator('#mesh-check')).to_be_enabled()
        expect(page.locator('#compare-targets')).to_be_enabled()
        expect(page.locator('#compare-targets')).to_be_checked()
        assert revision_requests[-1]['planning_id']
        expect(page.locator('#paper-status')).to_contain_text('Model ready for baseline')
        expect(page.locator('#paper-status')).to_contain_text('Excitation arithmetic is not applicable')
        assert planning_requests[-1]['preserve_paper_excitation'] is True
        page.locator('#paper-preserve-excitation').uncheck()
        expect(page.locator('#compare-targets')).to_be_disabled()
        expect(page.locator('#compare-targets')).not_to_be_checked()
        expect(page.locator('#mesh-check')).to_be_disabled()
        page.locator('#request-action').select_option('new')
        page.locator('#prompt').fill('Create a paper-based variation using the excitation in my instructions.')
        page.locator('#interpret').click()
        expect(page.locator('#compare-targets')).to_be_enabled()
        assert planning_requests[-1]['preserve_paper_excitation'] is False
        baseline_revision = page.evaluate('recordedRevision.id')
        page.locator('#paper-preserve-excitation').check()
        expect(page.locator('#compare-targets')).to_be_disabled()

        page.locator('#paper-case').select_option('displaced')
        expect(page.locator('#compare-targets')).to_be_disabled()
        expect(page.locator('#compare-targets')).not_to_be_checked()
        expect(page.locator('#paper-target-conditions')).to_contain_text('2 A, displaced geometry')
        page.locator('#validate-plan').click()
        expect(page.locator('#mesh-check')).to_be_enabled()
        expect(page.locator('#compare-targets')).to_be_disabled()
        assert 'planning_id' not in revision_requests[-1], 'Preview silently retagged old geometry to another paper case'
        assert page.evaluate('recordedRevision.id') == baseline_revision
        page.locator('#request-action').select_option('new')
        page.locator('#prompt').fill('Create the selected displaced operating case from the paper.')
        page.locator('#interpret').click()
        expect(page.locator('#compare-targets')).to_be_enabled()
        assert page.evaluate('recordedRevision.id') != baseline_revision
        page.locator('#mesh-check').click()
        expect(page.locator('#mesh-status')).to_have_text('PASSED')
        page.locator('#solve').click()
        expect(page.locator('#target-status')).to_contain_text('Target mismatch')
        expect(page.locator('#recovery-controls')).to_be_visible()
        if artifact_dir:
            page.screenshot(path=str(artifact_dir/'paper-target-mismatch.png'), full_page=True)
        assert solve_requests[-1]['target_contract_id'], 'Targets were not frozen before queueing'
        page.locator('#recover-model').click()
        expect(page.locator('#target-status')).to_contain_text('Numerical target comparison passed', timeout=15000)
        assert len(solve_requests) == 2 and len(recover_requests) == 1
        recovered_contract_id = solve_requests[-1]['target_contract_id']
        expect(page.locator('#compare-targets')).to_be_checked()
        assert page.locator('#target-json').evaluate('(element) => element.readOnly')
        page.locator('#solve').click()
        expect(page.locator('#job-status')).to_have_text('COMPLETED')
        assert len(solve_requests) == 3
        assert solve_requests[-1]['target_contract_id'] == recovered_contract_id
        rebound = papers.get_contract(solve_requests[-1]['target_contract_id'])
        assert rebound['contract']['targets_sha256'] == original_contract['contract']['targets_sha256']
        assert rebound['evidence_id'] == original_contract['evidence_id']
        if artifact_dir:
            page.screenshot(path=str(artifact_dir/'paper-recovered-target-pass.png'), full_page=True)
        # Loading hides recovery controls synchronously, but its preview and
        # revision save finish later. Wait for that exact persisted revision and
        # its accepted preview before arming the next delayed-response fixture.
        with page.expect_response(lambda response: response.url.endswith('/api/revisions')
                                  and response.request.method == 'POST') as loaded_response:
            page.locator('#load-example').click()
        loaded_revision = loaded_response.value.json()
        expect(page.locator('#revision-summary')).to_contain_text(loaded_revision['id'][:12])
        expect(page.locator('#mesh-check')).to_be_enabled()
        assert page.evaluate('recordedRevision.id') == loaded_revision['id']
        assert 'parent_id' not in loaded_response.value.request.post_data_json
        assert history.get(loaded_revision['id'])['parent_id'] is None
        expect(page.locator('#recovery-controls')).to_be_hidden()
        page.evaluate('job => offerRecovery(job)', first_failed_job)
        expect(page.locator('#recovery-controls')).to_be_hidden()
        assert page.evaluate('recoveryJob') is None

        # Withhold an old revision response until a fresh draft replaces it.
        mode['late_revision_expected'] = {'id': loaded_revision['id'], 'title': loaded_revision['plan']['title']}
        mode['late_revision'] = True
        with page.expect_response(lambda response: response.url.endswith('/api/revisions')
                                  and response.request.method == 'POST') as stale_response:
            page.locator('#validate-plan').click()
        assert stale_response.value.request.post_data_json['parent_id'] == loaded_revision['id']
        assert history.get(stale_response.value.json()['id'])['plan'] == loaded_revision['plan']
        expect(page.locator('#plan-json')).to_have_value(re.compile('Fresh case while old save is pending'))
        assert page.evaluate('recordedRevision') is None, 'Late save reattached the old model branch'
        with page.expect_response(lambda response: response.url.endswith('/api/revisions')
                                  and response.request.method == 'POST'
                                  and response.request.post_data_json['spec']['title'] == 'Fresh case while old save is pending') as fresh_response:
            page.locator('#validate-plan').click()
        fresh_revision = fresh_response.value.json()
        expect(page.locator('#revision-summary')).to_contain_text(fresh_revision['id'][:12])
        expect(page.locator('#mesh-check')).to_be_enabled()
        assert page.evaluate('recordedRevision.id') == fresh_revision['id']
        assert 'parent_id' not in fresh_response.value.request.post_data_json, 'Fresh model inherited an earlier revision'
        assert history.get(fresh_revision['id'])['parent_id'] is None

        # Clearing during extraction and replacing a file during upload discard
        # their late responses; neither can silently restore paper context.
        mode['extraction'] = 'clear'
        page.locator('#paper-extract').click()
        expect(page.locator('#paper-evidence')).to_be_hidden()
        expect(page.locator('#paper-status')).to_contain_text('Paper context removed')
        assert page.evaluate('selectedPaper') is None
        page.locator('#paper-file').set_input_files({'name': 'again.pdf', 'mimeType': 'application/pdf', 'buffer': paper_bytes})
        mode['late_upload'] = True
        page.locator('#paper-upload').click()
        expect(page.locator('#paper-status')).to_contain_text('Read the selected PDF')
        assert page.evaluate('uploadedPaper') is None
        page.locator('#paper-upload').click()
        expect(page.locator('#paper-status')).to_contain_text('pages read locally')
        mode['extraction'] = 'error'
        page.locator('#paper-extract').click()
        expect(page.locator('#paper-status')).to_contain_text('The cited unit does not match.')
        expect(page.locator('#paper-status')).to_contain_text('Audit:')
        mode['extraction'] = 'partial'
        page.locator('#paper-extract').click()
        expect(page.locator('#paper-status')).to_contain_text('Partial extraction: 1 claims excluded')
        expect(page.locator('#paper-questions')).to_contain_text('Supply the excluded core dimension')
        assert all(parameter['id'] != 'excluded_dimension' for parameter in page.evaluate('paperEvidence.parameters'))
        assert all(target['id'] != 'excluded_dimension' for target in page.evaluate('paperEvidence.targets'))
        page.locator('#request-action').select_option('new')
        page.locator('#prompt').fill('Create the verified baseline case; request any missing feature dimension.')
        page.locator('#interpret').click()
        expect(page.locator('#paper-status')).to_contain_text('Model ready for baseline')
        expect(page.locator('#paper-status')).to_contain_text('Partial extraction: 1 claims excluded')
        assert not browser_errors, browser_errors
        browser.close()


def main():
    with tempfile.TemporaryDirectory(prefix='femm-paper-browser-') as directory:
        folder = Path(directory)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0)); port = listener.getsockname()[1]
        url = 'http://127.0.0.1:'+str(port)
        with (folder/'server.log').open('w+') as log:
            process = subprocess.Popen([sys.executable, '-u', '-c', SERVER, str(folder/'data'), str(port)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_for_server(process, url, log)
                exercise(url, folder)
                print('Paper browser regression passed: exact source/case binding, target freeze, fixed-target recovery, stale response guards and structured extraction errors; no paid/native calls.')
            finally:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)


if __name__ == '__main__':
    main()
