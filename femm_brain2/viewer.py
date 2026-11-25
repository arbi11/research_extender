"""Export indexed L1–L5 records and their code as a standalone HTML file."""
from contextlib import closing
import inspect
import json
from pathlib import Path

from .catalogue import DEFAULT_DB, connect_readonly
from .knowledge import construction_plan, descendants, snapshot, validate_knowledge
from .recipes import PROCEDURES

TEMPLATE = Path(__file__).with_name('viewer.html')


def export(paper_id, output, db_path=DEFAULT_DB):
    target = Path(output).resolve()
    if target.suffix.lower() != '.html' or target == TEMPLATE.resolve():
        raise ValueError('Choose an output .html file other than the viewer template')
    knowledge = snapshot(db_path)
    validate_knowledge(knowledge)
    paper = next((p for p in knowledge['papers'] if p['id'] == paper_id), None)
    if paper is None:
        raise ValueError('Unknown paper: ' + paper_id)
    plan = construction_plan(knowledge, paper_id)
    steps = {s['id']: s for s in plan['steps']}
    # Include the paper's procedures and their physical ancestors, not other papers' features.
    scope = set(steps) | {paper['classification']}
    while True:
        parents = {e['parent'] for e in knowledge['edges']
                   if e['relation'] == 'contains' and e['child'] in scope}
        if parents <= scope:
            break
        scope |= parents
    nodes = [n for n in knowledge['nodes'] if n['id'] in scope]
    with closing(connect_readonly(db_path)) as db:
        row = db.execute("SELECT value FROM metadata WHERE key='knowledge_native_evidence'").fetchone()
        evidence = json.loads(row[0]) if row else None
        if evidence and evidence['paper_id'] == paper_id:
            from .knowledge_run import audit
            checked = audit(evidence['report'], knowledge)
            if checked['report_sha256'] != evidence['report_sha256']:
                raise ValueError('Attached report changed; rebuild the index with audited evidence')
            report = json.loads(Path(checked['report']).read_text())
            evidence = {**checked, 'checks': [{'id': c['id'], 'status': c['status']}
                                              for c in report['checks']],
                        'solves': sum(sum(key.startswith('step') and key.endswith('.angle')
                                          for key in c.get('observed', {}).get('values', {}))
                                      for c in report['checks'])}
        else:
            evidence = None
        api = {}
        for name in sorted({a for s in plan['steps'] for a in s['api_commands']}):
            row = db.execute('SELECT data FROM commands WHERE name=? AND version=? AND binding=?',
                             (name, '5.0', 'lua4')).fetchone()
            if row is None:
                raise ValueError('Missing indexed API contract: ' + name)
            record = json.loads(row[0])
            api[name] = {k: record[k] for k in ('signature', 'purpose')}
    for node in nodes:
        node['procedures'] = sorted(descendants(knowledge, node['id']) & steps.keys())
        names = set()
        for procedure in node['procedures']:
            names.update(v[1:] for v in paper['bindings'][procedure].values()
                         if isinstance(v, str) and v.startswith('$') and v[1:] in paper['parameters'])
        pending = list(names)
        while pending:
            for name in paper['parameters'][pending.pop()].get('inputs', []):
                if name not in names:
                    names.add(name)
                    pending.append(name)
        node['parameters'] = {name: {**paper['parameters'][name], 'value': plan['parameters'][name]}
                              for name in sorted(names)}
        if node['id'] in steps:
            node['arguments'] = steps[node['id']]['arguments']
            node['code'] = inspect.getsource(PROCEDURES[node['implementation']['function']])
    payload = {'paper': {k: paper[k] for k in ('id', 'source', 'scope', 'excluded')},
               'nodes': nodes, 'edges': [e for e in knowledge['edges']
                                        if e['parent'] in scope and e['child'] in scope],
               'evidence': evidence, 'api': api}
    # Keep source text/code inert even if it contains an HTML script terminator.
    data = json.dumps(payload, ensure_ascii=True, allow_nan=False)
    data = data.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    page = TEMPLATE.read_text().replace('__BRAIN_DATA__', data)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(page)
    return {'status': 'created', 'html': str(target), 'nodes': len(nodes),
            'paper_id': paper_id, 'bytes': target.stat().st_size,
            'evidence': evidence['status'] if evidence else 'not_attached'}
