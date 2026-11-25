"""Validated L1–L5 records, SQLite retrieval and dependency-ordered construction."""
import ast
from contextlib import closing
import inspect
import json
import math
from pathlib import Path
import re

from .catalogue import DEFAULT_DB, connect_readonly, load_batch
from .recipes import Geometry, PROCEDURES
from .source import DATA, fingerprint

KNOWLEDGE = DATA / 'knowledge.json'


def load_knowledge(path=KNOWLEDGE):
    return json.loads(Path(path).read_text())


def digest(value):
    return fingerprint(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def arithmetic(expression, names):
    """Small arithmetic expressions only; never eval source or agent text."""
    if len(expression) > 500:
        raise ValueError('Parameter expression too long')
    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -walk(node.operand) if isinstance(node.op, ast.USub) else walk(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add): return a+b
            if isinstance(node.op, ast.Sub): return a-b
            if isinstance(node.op, ast.Mult): return a*b
            if isinstance(node.op, ast.Div): return a/b
            if isinstance(node.op, ast.Pow) and abs(b) <= 12: return a**b
        raise ValueError('Unsupported parameter expression')
    try:
        result = walk(ast.parse(expression, mode='eval').body)
        if type(result) not in (int, float) or not math.isfinite(result):
            raise ValueError('Nonfinite parameter expression')
        return result
    except (SyntaxError, KeyError, ZeroDivisionError, OverflowError, TypeError) as exc:
        raise ValueError('Invalid parameter expression: '+expression) from exc


def parameters(paper, overrides=None, rotor_angle=0):
    values = {'pi': math.pi, 'rotor_angle': rotor_angle}
    definitions = paper['parameters']
    overrides = overrides or {}
    if set(overrides)-set(definitions):
        raise ValueError('Unknown overridden parameter')
    for key in overrides:
        if definitions[key]['provenance'] == 'derived':
            raise ValueError('Override source parameters, not derived values')
    remaining = dict(definitions)
    while remaining:
        progressed = False
        for key, parameter in list(remaining.items()):
            if parameter['provenance'] == 'derived':
                if not set(parameter['inputs']) <= values.keys():
                    continue
                values[key] = arithmetic(parameter['expression'], values)
            else:
                values[key] = overrides.get(key, parameter['value'])
            del remaining[key]
            progressed = True
        if not progressed:
            raise ValueError('Cyclic or missing parameter dependencies: '+', '.join(remaining))
    # Numerical mesh controls scale together; they never change physical inputs.
    for key in ('gap_mesh_mm', 'magnet_mesh_mm', 'slot_mesh_mm', 'gap_arc_deg', 'arc_deg'):
        values[key] *= values['mesh_scale']
    return values


def resolve(value, values):
    if isinstance(value, str) and value.startswith('$'):
        if value[1:] not in values:
            raise ValueError('Unknown parameter reference: '+value)
        return values[value[1:]]
    if isinstance(value, dict): return {k: resolve(v, values) for k,v in value.items()}
    if isinstance(value, list): return [resolve(v, values) for v in value]
    return value


def descendants(knowledge, start):
    nodes = {n['id']: n for n in knowledge['nodes']}
    if start not in nodes:
        raise ValueError('Unknown hierarchy entry: '+start)
    found, pending = set(), [start]
    while pending:
        node = pending.pop()
        if node in found: continue
        found.add(node)
        pending.extend(e['child'] for e in knowledge['edges'] if e['relation']=='contains' and e['parent']==node)
    return found


def construction_plan(knowledge, paper_id, overrides=None, rotor_angle=0):
    paper = next((p for p in knowledge['papers'] if p['id']==paper_id), None)
    if paper is None: raise ValueError('Unknown paper: '+paper_id)
    nodes = {n['id']: n for n in knowledge['nodes']}
    reachable = descendants(knowledge, paper['classification'])
    selected = set(paper['bindings'])
    if not selected <= reachable or any(nodes[n]['level'] != 5 for n in selected):
        raise ValueError('Paper binding is not a reachable L5 procedure')
    prerequisites = {n: {e['child'] for e in knowledge['edges']
                          if e['relation']=='requires' and e['parent']==n} for n in selected}
    if any(not requirements <= selected for requirements in prerequisites.values()):
        raise ValueError('Construction is missing a required procedure')
    values = parameters(paper, overrides, rotor_angle)
    ordered, remaining = [], set(selected)
    while remaining:
        ready = sorted(n for n in remaining if not prerequisites[n] & remaining)
        if not ready: raise ValueError('Cyclic construction prerequisites')
        for node_id in ready:
            node = nodes[node_id]
            ordered.append({'id': node_id, 'title': node['title'],
                            'implementation': node['implementation'],
                            'arguments': resolve(paper['bindings'][node_id], values),
                            'api_commands': node['api_commands'],
                            'requires': sorted(prerequisites[node_id])})
            remaining.remove(node_id)
    return {'paper_id': paper_id, 'parameters': values, 'steps': ordered,
            'context_api_commands': paper['context_api_commands'],
            'scope': paper['scope'], 'knowledge_sha256': digest(knowledge)}


def construct(plan):
    geometry = Geometry()
    for step in plan['steps']:
        implementation = step['implementation']
        if implementation['module'] != 'femm_brain2.recipes' or implementation['function'] not in PROCEDURES:
            raise ValueError('Procedure is outside the reviewed implementation registry')
        geometry.owner = step['id']
        function = PROCEDURES[implementation['function']]
        try:
            inspect.signature(function).bind(geometry, **step['arguments'])
        except TypeError as exc:
            raise ValueError('Invalid arguments for '+step['id']+': '+str(exc)) from exc
        function(geometry, **step['arguments'])
        geometry.trace.append(step['id'])
    if not hasattr(geometry, 'reference'):
        raise ValueError('Ideal-iron model requires a potential reference')
    return geometry


def validate_knowledge(knowledge, *, source_check=False):
    nodes = {n['id']: n for n in knowledge['nodes']}
    if knowledge['schema_version'] != 1 or len(nodes) != len(knowledge['nodes']):
        raise ValueError('Unsupported knowledge version or duplicate hierarchy IDs')
    if {n['level'] for n in nodes.values()} != {1,2,3,4,5}:
        raise ValueError('Hierarchy must contain L1 through L5')
    api_names = {c['name'] for c in load_batch()['commands']}
    for edge in knowledge['edges']:
        if edge['parent'] not in nodes or edge['child'] not in nodes:
            raise ValueError('Dangling knowledge link')
        a,b = nodes[edge['parent']], nodes[edge['child']]
        if edge['relation']=='contains':
            if b['level'] != a['level']+1: raise ValueError('Hierarchy link skips a level')
        elif edge['relation']=='requires':
            if a['level'] != 5 or b['level'] != 5: raise ValueError('Prerequisites must connect L5 procedures')
        else: raise ValueError('Unknown knowledge relationship')
    roots = [n['id'] for n in nodes.values() if n['level']==1]
    reached = set().union(*(descendants(knowledge, root) for root in roots))
    if reached != nodes.keys(): raise ValueError('Unreachable hierarchy entries')
    for node in nodes.values():
        if not node.get('references'): raise ValueError('Hierarchy entry has no source: '+node['id'])
        if node['level']==5:
            if node['execution']['status'] != 'not_tested': raise ValueError('Source records cannot assert native success')
            if not set(node['api_commands']) <= api_names: raise ValueError('Unreviewed API link')
    for paper in knowledge['papers']:
        if not set(paper['context_api_commands']) <= api_names: raise ValueError('Unreviewed context API')
        values = parameters(paper)
        for name, parameter in paper['parameters'].items():
            provenance = parameter['provenance']
            required = {'quoted': ['quote','pdf_page','location'], 'derived': ['expression','inputs'],
                        'figure': ['pdf_page','location','uncertainty'], 'assumed': ['reason'],
                        'numerical': ['reason','sensitivity_control']}
            if provenance not in required or any(k not in parameter for k in required[provenance]):
                raise ValueError('Incomplete provenance for '+name)
            if provenance == 'assumed' and not ({'effect','sensitivity_control'} & parameter.keys()):
                raise ValueError('Assumption requires an effect bound or sensitivity control: '+name)
            if provenance == 'derived' and not math.isclose(values[name],parameter['value'],rel_tol=1e-10,abs_tol=1e-10):
                raise ValueError('Stored derived value disagrees with its formula: '+name)
        geometry = construct(construction_plan(knowledge,paper['id']))
        counts = dict(zip(('points','segments','arcs','labels'),map(len,(geometry.nodes,geometry.segments,geometry.arcs,geometry.labels))))
        if counts != paper['geometry_expectations']['counts']:
            raise ValueError('Constructed geometry differs from frozen counts: '+str(counts))
        for control in paper['controls']:
            construct(construction_plan(knowledge,paper['id'],control['overrides']))
    result = {'status':'passed','nodes':len(nodes),'relationships':len(knowledge['edges']),
              'papers':len(knowledge['papers']),'levels':{str(i):sum(n['level']==i for n in nodes.values()) for i in range(1,6)}}
    if source_check:
        from .paper_source import verify_source
        result['source_verification'] = verify_source(knowledge)
    return result


def install(db, knowledge):
    """Called inside catalogue.build's staging transaction, before atomic replacement."""
    db.execute('INSERT INTO metadata VALUES (?,?)',('knowledge_sha256',json.dumps(digest(knowledge))))
    db.execute('INSERT INTO metadata VALUES (?,?)',('knowledge_snapshot',json.dumps(knowledge)))
    for node in knowledge['nodes']:
        cursor = db.execute('INSERT INTO knowledge_nodes(id,level,title,data) VALUES (?,?,?,?)',
                            (node['id'],node['level'],node['title'],json.dumps(node)))
        db.execute('INSERT INTO knowledge_search(rowid,title,details) VALUES (?,?,?)',
                   (cursor.lastrowid,node['title'],json.dumps(node)))
        for command in node.get('api_commands',[]):
            db.execute('INSERT INTO knowledge_api_links VALUES (?,?,?,?)',(node['id'],command,'5.0','lua4'))
    db.executemany('INSERT INTO knowledge_edges VALUES (?,?,?)',[(e['parent'],e['child'],e['relation']) for e in knowledge['edges']])
    db.executemany('INSERT INTO knowledge_papers VALUES (?,?)',[(p['id'],json.dumps(p)) for p in knowledge['papers']])


def snapshot(path=DEFAULT_DB):
    with closing(connect_readonly(path)) as db:
        row = db.execute("SELECT value FROM metadata WHERE key='knowledge_snapshot'").fetchone()
        if row is None: raise ValueError('Hierarchy is not indexed. Run: python3 -m femm_brain2 build --knowledge')
        knowledge = json.loads(row[0])
        expected = json.loads(db.execute("SELECT value FROM metadata WHERE key='knowledge_sha256'").fetchone()[0])
        if digest(knowledge) != expected: raise ValueError('Stored hierarchy fingerprint mismatch')
    return knowledge


def show(node_id='electric_motor', path=DEFAULT_DB):
    knowledge = snapshot(path)
    ids = descendants(knowledge, node_id)
    with closing(connect_readonly(path)) as db:
        row = db.execute("SELECT value FROM metadata WHERE key='knowledge_native_evidence'").fetchone()
    return {'nodes': [n for n in knowledge['nodes'] if n['id'] in ids],
            'relationships': [e for e in knowledge['edges'] if e['parent'] in ids and e['child'] in ids],
            'native_evidence': json.loads(row[0]) if row else {'status': 'not_attached'}}


def tree(path=DEFAULT_DB):
    knowledge = snapshot(path)
    nodes = {n['id']: n for n in knowledge['nodes']}
    lines = []
    def visit(id, indent):
        node = nodes[id]
        implementation = node.get('implementation', {})
        suffix = ' -> ' + implementation['function'] + '()' if implementation else ''
        lines.append('  '*indent + f"L{node['level']} {node['title']} [{id}]" + suffix)
        for edge in knowledge['edges']:
            if edge['parent'] == id and edge['relation'] == 'contains':
                visit(edge['child'], indent+1)
    for node in knowledge['nodes']:
        if node['level'] == 1:
            visit(node['id'], 0)
    return '\n'.join(lines)


def search(query, path=DEFAULT_DB):
    terms = re.findall(r'\w+', query.lower())[:20]
    if not terms:
        return {'results': []}
    expression = ' AND '.join('"'+term+'"*' for term in terms)
    snapshot(path)
    with closing(connect_readonly(path)) as db:
        rows = db.execute('SELECT n.data FROM knowledge_search JOIN knowledge_nodes n ON n.rowid=knowledge_search.rowid '
                          'WHERE knowledge_search MATCH ? ORDER BY bm25(knowledge_search) LIMIT 10', (expression,)).fetchall()
    return {'query': query, 'results': [json.loads(row[0]) for row in rows]}


def plan_from_store(paper_id, path=DEFAULT_DB):
    plan = construction_plan(snapshot(path), paper_id)
    with closing(connect_readonly(path)) as db:
        for step in plan['steps']:
            step['api_contracts'] = []
            for name in step['api_commands']:
                row = db.execute('SELECT data FROM commands WHERE name=? AND version=? AND binding=?', (name, '5.0', 'lua4')).fetchone()
                if row is None:
                    raise ValueError('Missing indexed API contract: '+name)
                record = json.loads(row[0])
                step['api_contracts'].append({'name': name, 'signature': record['signature'], 'review': record['review']})
    return plan
