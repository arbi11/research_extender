"""Generate native Lua from the retrieved L5 construction plan and API contracts."""
import json
import math

from .catalogue import load_batch
from .knowledge import construct, construction_plan, parameters


def lua(value):
    if isinstance(value,str): return json.dumps(value,ensure_ascii=True)
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('Only finite numeric and string API arguments are supported')
    return format(value,'.17g')


class Program:
    def __init__(self, knowledge, paper):
        self.lines, self.trace = [], []
        self.contracts = {c['name']:c for c in load_batch()['commands']}
        self.allowed = {n['id']:set(n['api_commands']) for n in knowledge['nodes'] if n['level']==5}
        self.allowed['model_context'] = set(paper['context_api_commands'])

    def call(self, name, arguments=(), owner='model_context', prefix=''):
        if name not in self.allowed.get(owner,set()):
            raise ValueError('API call lacks a reviewed knowledge link: '+owner+' -> '+name)
        contract = self.contracts[name]
        required = sum(p['required'] for p in contract['parameters'])
        if not required <= len(arguments) <= len(contract['parameters']):
            raise ValueError('Argument count differs from reviewed API contract: '+name)
        self.lines.append(prefix+name+'('+', '.join(lua(a) for a in arguments)+')')
        self.trace.append({'procedure':owner,'api':name,'line':len(self.lines)})


def generate(knowledge, paper_id, control):
    paper = next(p for p in knowledge['papers'] if p['id']==paper_id)
    plan = construction_plan(knowledge,paper_id,control['overrides'])
    g = construct(plan)
    values = parameters(paper,control['overrides'])
    p = Program(knowledge,paper)
    p.lines += ['-- Generated from validated L1-L5 records; expected results are not in this script.',
                'function emit(file, key, value)',
                ' write(file, key .. " " .. format("%.17g", re(value)) .. " " .. format("%.17g", im(value)) .. "\\n")',
                'end']
    p.call('newdocument',[0])
    p.call('mi_probdef',[0,'millimeters','planar',1e-10,values['depth_mm'],30])
    for name,mu,hc in [('Air',1,0),('Coil',1,0),('Magnet',values['magnet_mu_r'],values['coercivity_A_m'])]:
        p.call('mi_addmaterial',[name,mu,mu,hc,0,0,0,0,1,0,0,0,0,0])
    for phase in ('A','B','C'): p.call('mi_addcircprop',[phase,0,1])
    for n in g.nodes:
        owner=n['owner']
        p.call('mi_addnode',[n['x'],n['y']],owner)
        p.call('mi_selectnode',[n['x'],n['y']],owner)
        p.call('mi_setnodeprop',['<None>',n['group']],owner)
        p.call('mi_clearselected',owner=owner)
    for segment in g.segments:
        a,b=[g.nodes[segment[k]] for k in ('a','b')]; owner=segment['owner']
        p.call('mi_addsegment',[a['x'],a['y'],b['x'],b['y']],owner)
        p.call('mi_selectsegment',[(a['x']+b['x'])/2,(a['y']+b['y'])/2],owner)
        p.call('mi_setsegmentprop',['<None>',0,1,0,segment['group']],owner)
        p.call('mi_clearselected',owner=owner)
    for arc in g.arcs:
        a,b=[g.nodes[arc[k]] for k in ('a','b')]; owner=arc['owner']
        p.call('mi_addarc',[a['x'],a['y'],b['x'],b['y'],arc['angle'],arc['max_segment']],owner)
        angle=math.atan2(a['y'],a['x'])+math.radians(arc['angle']/2); radius=math.hypot(a['x'],a['y'])
        p.call('mi_selectarcsegment',[radius*math.cos(angle),radius*math.sin(angle)],owner)
        p.call('mi_setarcsegmentprop',[arc['max_segment'],'<None>',0,arc['group']],owner)
        p.call('mi_clearselected',owner=owner)
    for label in g.labels:
        owner=label['owner']
        p.call('mi_addblocklabel',[label['x'],label['y']],owner)
        p.call('mi_selectlabel',[label['x'],label['y']],owner)
        p.call('mi_setblockprop',[label['material'],0,label['mesh'],label['circuit'],label['magdir'],label['group'],label['turns']],owner)
        p.call('mi_clearselected',owner=owner)
    reference=g.reference; owner=reference['owner']
    p.call('mi_addpointprop',['ReferenceA',reference['value'],0],owner)
    p.call('mi_selectnode',reference['at'],owner)
    p.call('mi_setnodeprop',['ReferenceA',0],owner)
    p.call('mi_clearselected',owner=owner)
    p.lines += ['local file = openfile("/work/observations.obs", "w")']
    count=control['samples']; step=360/values['pole_pairs']/values['samples']
    for i in range(count):
        key=f'step{i:03d}'
        if i:
            owner='build_radial_magnet_ring'
            p.call('mi_clearselected',owner=owner)
            p.call('mi_selectgroup',[1],owner)
            p.call('mi_moverotate',[0,0,step],owner)
            p.call('mi_clearselected',owner=owner)
            # Geometry cleanup in the pinned runtime can clear node group tags.
            # Restore rotor membership spatially; no geometry or material is changed.
            p.call('mi_selectcircle',[0,0,values['r2_mm']+values['air_gap_mm']/4,4],owner)
            p.call('mi_setgroup',[1],owner)
            p.call('mi_clearselected',owner=owner)
        # Retain first and final models/solutions; all position results remain in observations.obs.
        stem='model-000' if i==0 else 'model-last'
        p.call('mi_saveas',['/work/'+stem+'.fem'])
        p.call('mi_createmesh',prefix='local created = ' if i==0 else 'created = ')
        p.call('mi_analyze',[1])
        p.call('mi_loadsolution')
        p.lines += [f'emit(file, "{key}.angle", {lua(i*step)})', f'emit(file, "{key}.created_nodes", created)']
        p.call('mo_numnodes',prefix='local nodes = ' if i==0 else 'nodes = ')
        p.call('mo_numelements',prefix='local elements = ' if i==0 else 'elements = ')
        p.lines += [f'emit(file, "{key}.nodes", nodes)',f'emit(file, "{key}.elements", elements)']
        for phase in ('A','B','C'):
            p.call('mo_getcircuitproperties',[phase],prefix='local current, volts, linkage = ' if i==0 and phase=='A' else 'current, volts, linkage = ')
            p.lines += [f'emit(file, "{key}.{phase}.current", current)',f'emit(file, "{key}.{phase}.voltage", volts)',f'emit(file, "{key}.{phase}.linkage", linkage)']
        if i==0:
            radius=(values['r2_mm']+values['r3_mm'])/2
            p.call('mo_getpointvalues',[radius/math.sqrt(2),radius/math.sqrt(2)],prefix='local A, B1, B2 = ')
            p.lines += ['emit(file,"probe.B1",B1)','emit(file,"probe.B2",B2)']
        p.call('mo_close')
        if i+1 < count and (i+1) % 12 == 0:
            # Bound FEMM process memory without changing any physical input.
            # pkill runs only in this container's private PID namespace.
            p.call('mi_close')
            p.lines += ['closefemm()', 'execute("/usr/bin/pkill -x femm_headless")',
                        'execute("sleep 0.1")', 'openfemm(1)']
            p.call('open',['/work/'+stem+'.fem'])
    p.call('mi_close')
    p.lines += ['emit(file,"completed",1)','closefile(file)','closefemm()']
    return '\n'.join(p.lines)+'\n', p.trace, g
