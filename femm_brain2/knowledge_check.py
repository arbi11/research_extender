"""Read native model/solver files and score the frozen paper scope independently."""
import math
from pathlib import Path
import re

from .geometry_check import read_model
from .knowledge import construct, construction_plan, parameters
from .property_check import read_properties
from .solve_check import read_observations


def _rows(text, heading):
    lines=text.splitlines()
    i=next(i for i,s in enumerate(lines) if re.match(r'^\['+heading+r'\]\s*=',s))
    count=int(lines[i].split('=')[1])
    return [[float(x) for x in s.split()] for s in lines[i+1:i+1+count]]


def read_mesh(path):
    """FKN writes .ans node coordinates in the original model units (here mm)."""
    text=Path(path).read_text()
    if '[LengthUnits] = millimeters' not in text:
        raise ValueError('Expected millimeter solution coordinates')
    marker=re.search(r'(?mi)^\[Solution\]\s*$',text)
    if not marker: raise ValueError('No solver solution section')
    rows=text[marker.end():].strip().splitlines()
    count=int(rows[0]); nodes=[[float(v) for v in row.split()[:3]] for row in rows[1:count+1]]
    elements=int(rows[count+1]); triangles=[[int(v) for v in row.split()[:4]] for row in rows[count+2:count+2+elements]]
    if count<3 or elements<1 or len(nodes)!=count or len(triangles)!=elements:
        raise ValueError('Incomplete native solver mesh')
    if any(len(n)!=3 or not all(math.isfinite(x) for x in n) for n in nodes):
        raise ValueError('Invalid solver potentials/coordinates')
    areas={}; max_field=0
    for triangle in triangles:
        if len(triangle)!=4 or any(i<0 or i>=count for i in triangle[:3]):
            raise ValueError('Invalid solver element')
        (x0,y0,a0),(x1,y1,a1),(x2,y2,a2)=[nodes[i] for i in triangle[:3]]
        twice=(x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)
        if abs(twice)<1e-20: raise ValueError('Degenerate solver element')
        area=abs(twice)/2
        areas[triangle[3]]=areas.get(triangle[3],0)+area
        # A is Wb/m; coordinates above are mm, hence derivative times 1000.
        bx=1000*((x1-x0)*(a2-a0)-(x2-x0)*(a1-a0))/twice
        by=-1000*((a1-a0)*(y2-y0)-(a2-a0)*(y1-y0))/twice
        max_field=max(max_field,math.hypot(bx,by))
    return {'nodes':count,'elements':elements,'area_mm2':sum(areas.values()),
            'region_areas_mm2':{str(k):v for k,v in sorted(areas.items())},
            'max_element_field_T':max_field,'potential_min':min(n[2] for n in nodes),
            'potential_max':max(n[2] for n in nodes)}


def check_geometry(folder,knowledge,paper,control,stem,angle):
    path=Path(folder)/(stem+'.fem');text=path.read_text()
    model=read_model(path);holes=_rows(text,'NumHoles')
    expected=construct(construction_plan(knowledge,paper['id'],control['overrides'],angle))
    values=parameters(paper,control['overrides'],angle)
    tolerance=paper['acceptance']['geometry_absolute_tolerance_mm']
    errors=[]
    counts={k:len(v) for k,v in model.items()}; counts['holes']=len(holes)
    target={**paper['geometry_expectations']['counts'],'labels':sum(l['material']!='<No Mesh>' for l in expected.labels),'holes':sum(l['material']=='<No Mesh>' for l in expected.labels)}
    if counts!=target: errors.append('Native object counts differ from frozen construction')
    # Compare independently saved coordinates and assignments, not just entity counts.
    unmatched=list(model['points'])
    for n in expected.nodes:
        i=next((i for i,row in enumerate(unmatched) if abs(row[0]-n['x'])<=tolerance and abs(row[1]-n['y'])<=tolerance and int(row[3])==n['group']),None)
        if i is None: errors.append('Missing native node/group');break
        unmatched.pop(i)
    props=read_properties(path)
    materials={p['BlockName']:i+1 for i,p in enumerate(props['materials'])}
    circuits={p['CircuitName']:i+1 for i,p in enumerate(props['circuits'])}
    if set(materials)!={'Air','Coil','Magnet'} or set(circuits)!={'A','B','C'}:
        errors.append('Unexpected native materials or circuits')
    for material in props['materials']:
        mu=values['magnet_mu_r'] if material['BlockName']=='Magnet' else 1
        hc=values['coercivity_A_m'] if material['BlockName']=='Magnet' else 0
        if abs(material.get('Mu_x',0)-mu)>1e-10 or abs(material.get('Mu_y',0)-mu)>1e-10 or abs(material.get('H_c',0)-hc)>1e-6:
            errors.append('Native magnetic material differs from frozen inputs')
    if any(abs(c.get('TotalAmps_re',c.get('TotalAmps',0)))>1e-12 or abs(c.get('TotalAmps_im',0))>1e-12 for c in props['circuits']):
        errors.append('No-load circuit current must be zero')
    for label in expected.labels:
        if label['material']=='<No Mesh>':
            if not any(abs(row[0]-label['x'])<=tolerance and abs(row[1]-label['y'])<=tolerance and int(row[2])==label['group'] for row in holes):
                errors.append('Missing ideal-iron hole/group')
            continue
        row=next((r for r in model['labels'] if abs(r[0]-label['x'])<=tolerance and abs(r[1]-label['y'])<=tolerance),None)
        if row is None:
            errors.append('Missing native region label');continue
        if int(row[2])!=materials.get(label['material']) or int(row[4])!=circuits.get(label['circuit'],0) or int(row[6])!=label['group'] or int(row[7])!=label['turns']:
            errors.append('Native label material/circuit/turns/group mismatch')
        if label['material']=='Magnet' and label['magdir'] not in text:
            errors.append('Native radial magnetization expression is missing')
    for key,want in [('Depth',values['depth_mm']),('Frequency',0),('Precision',1e-10)]:
        m=re.search(r'(?m)^\['+key+r'\]\s*=\s*(.*?)\s*$',text)
        if not m or float(m.group(1))!=want:errors.append('Frozen model header mismatch: '+key)
    for key,want in [('ProblemType','planar'),('LengthUnits','millimeters')]:
        if not re.search(r'(?m)^\['+key+r'\]\s*=\s*"?'+want+r'"?\s*$',text):errors.append('Model '+key+' differs')
    mesh=read_mesh(Path(folder)/(stem+'.ans'))
    target_area=paper['geometry_expectations']['expected_mesh_area_mm2']
    if abs(mesh['area_mm2']-target_area)>paper['acceptance']['geometry_area_relative_tolerance']*target_area:
        errors.append('Meshed area differs from paper geometry')
    return {'counts':counts,'mesh':mesh,'material_names':list(materials),'circuit_names':list(circuits)},errors


def evaluate_control(folder,knowledge,paper,control,engine):
    errors=[]; observed={}
    try:
        obs=read_observations(Path(folder)/'observations.obs')
        values=parameters(paper,control['overrides']); step=360/values['pole_pairs']/values['samples']
        expected={'completed','probe.B1','probe.B2'}
        for i in range(control['samples']):
            prefix=f'step{i:03d}'
            expected|={prefix+'.'+key for key in ['angle','created_nodes','nodes','elements']}
            expected|={prefix+'.'+phase+'.'+key for phase in ['A','B','C'] for key in ['current','voltage','linkage']}
            if obs.get(prefix+'.angle')!=[i*step,0]:errors.append('Unexpected native rotor angle')
            if obs.get(prefix+'.created_nodes')!=obs.get(prefix+'.nodes'):errors.append('Native mesh-return mismatch')
            for key in ['nodes','elements']:
                if obs.get(prefix+'.'+key,[0])[0]<1:errors.append('No native '+key)
            for phase in ['A','B','C']:
                if obs.get(prefix+'.'+phase+'.current')!=[0,0]:errors.append('Nonzero no-load current')
                if obs.get(prefix+'.'+phase+'.voltage')!=[0,0]:errors.append('Unexpected voltage in static snapshot')
                if abs(obs.get(prefix+'.'+phase+'.linkage',[0,1])[1])>1e-12:errors.append('Unexpected complex DC flux')
        if set(obs)!=expected or obs.get('completed')!=[1,0]:errors.append('Incomplete or unexpected native observations')
        for stem,index in [('model-000',0)]+([('model-last',control['samples']-1)] if control['samples']>1 else []):
            geometry,failures=check_geometry(folder,knowledge,paper,control,stem,index*step)
            observed[stem]=geometry;errors+=failures
            prefix=f'step{index:03d}'
            if geometry['mesh']['nodes']!=obs[prefix+'.nodes'][0] or geometry['mesh']['elements']!=obs[prefix+'.elements'][0]:
                errors.append('Postprocessor count differs from native solver file')
        observed['values']=obs
        if control['samples']>1:
            flux=[obs[f'step{i:03d}.A.linkage'][0] for i in range(control['samples'])]
            n=len(flux)
            c=2/n*sum(v*math.cos(2*math.pi*i/n) for i,v in enumerate(flux))
            s=2/n*sum(v*math.sin(2*math.pi*i/n) for i,v in enumerate(flux))
            observed['phase_A_flux_fundamental_Wb']=math.hypot(c,s)
            observed['emf_fundamental_peak_V']=2*math.pi*values['electrical_frequency_Hz']*math.hypot(c,s)
            half_wave=max(abs(flux[i]+flux[i+n//2]) for i in range(n//2))
            observed['half_wave_residual_Wb']=half_wave
            if half_wave>paper['acceptance']['symmetry_flux_max_Wb']:errors.append('Half-wave symmetry check failed')
        if engine['returncode']!=0: errors.append('Native FEMM process failed or timed out')
    except (OSError,ValueError,KeyError,IndexError,StopIteration) as exc:
        errors.append(str(exc))
    return {'id':control['id'],'status':'failed' if errors else 'passed','errors':errors,'observed':observed}


def evaluate_summary(paper,checks):
    by_id={c['id']:c for c in checks};errors=[];measurements={}
    if len(checks)!=len(paper['controls']) or any(c['status']!='passed' for c in checks):
        errors.append('Not every frozen native control passed')
    try:
        base=by_id['baseline']['observed'];fine=by_id['refined_mesh']['observed']
        actual=base['emf_fundamental_peak_V']; target=paper['acceptance']['emf_fundamental_peak_V']
        tolerance=paper['acceptance']['relative_tolerance']*abs(target)+paper['acceptance']['absolute_tolerance']
        measurements.update(emf_target_V=target,emf_observed_V=actual,emf_absolute_error_V=abs(actual-target),emf_tolerance_V=tolerance)
        if abs(actual-target)>tolerance:errors.append('No-load EMF differs from frozen paper target')
        change=abs(fine['emf_fundamental_peak_V']-actual)/abs(actual)
        measurements['mesh_relative_change']=change
        if change>paper['acceptance']['mesh_relative_change_max']:errors.append('Mesh sensitivity exceeds frozen bound')
        gauge=by_id['gauge_shift']['observed']['values'];zero=by_id['zero_magnet']['observed']['values']
        gauge_error=max(abs(gauge['step000.'+p+'.linkage'][0]-base['values']['step000.'+p+'.linkage'][0]) for p in ['A','B','C'])
        zero_flux=max(abs(zero['step000.'+p+'.linkage'][0]) for p in ['A','B','C'])
        measurements.update(gauge_flux_difference_Wb=gauge_error,zero_source_flux_Wb=zero_flux)
        if gauge_error>paper['acceptance']['gauge_flux_difference_max_Wb']:errors.append('Gauge-invariance control failed')
        if zero_flux>paper['acceptance']['zero_field_flux_max_Wb']:errors.append('Zero-source flux control failed')
        if math.hypot(zero['probe.B1'][0],zero['probe.B2'][0])>paper['acceptance']['zero_field_B_max_T']:
            errors.append('Zero-source field control failed')
    except (KeyError,ZeroDivisionError) as exc:
        errors.append('Missing physical comparison: '+str(exc))
    return {'status':'passed' if not errors else 'failed','scope':'Paper Example 1 geometry and no-load EMF only; not whole-paper replication',
            'errors':errors,'measurements':measurements}
