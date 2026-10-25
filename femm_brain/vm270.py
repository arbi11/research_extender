"""Fresh typed reconstruction of the public ANSYS VM270 / TEAM 23 case.

The measured force is retained independently of construction and solving. No
downloaded APDL is opened or executed. Source infinite elements are explicitly
replaced with a finite A=0 semicircle, with independent mesh/domain parameters.
"""
from __future__ import annotations

import math

from .geometry import _Drawing, arc_geometry
from .native_check import geometry_signature

MU0 = 4 * math.pi * 1e-7
REMANENCE = 1.02
COERCIVITY = 720000.
RELATIVE_PERMEABILITY = REMANENCE / (COERCIVITY * MU0)
TARGET_FORCE_N = 1.150 * 9.81e-3
TARGET_RELATIVE_TOLERANCE = .02
DEFAULTS = {"current": .05, "displacement": .234, "domain_radius": 20., "mesh_scale": 1.}
PARAMETERS = [
    {"name": "current", "unit": "A", "description": "Signed winding DC current; the measured force target applies only to 0.05 A", "minimum": -.5, "maximum": .5},
    {"name": "displacement", "unit": "mm", "description": "Source axial displacement beyond the 0.127 mm spool end; source target applies to 0.234 mm", "minimum": 0., "maximum": 5.},
    {"name": "domain_radius", "unit": "mm", "description": "Radius of finite A=0 exterior adaptation; air mesh lengths are independent of radius", "minimum": 10., "maximum": 100.},
    {"name": "mesh_scale", "unit": "ratio", "description": "Multiplier on body and local-air mesh lengths; exterior air remains fixed at 0.1 mm", "minimum": .25, "maximum": 4.},
]


def _parameters(p):
    if not isinstance(p, dict):
        raise ValueError("VM270 parameters must be an object")
    if set(p) - set(DEFAULTS):
        raise ValueError("Unknown VM270 parameter")
    result = dict(DEFAULTS, **p)
    for spec in PARAMETERS:
        value = result[spec['name']]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("VM270 parameter must be finite: " + spec['name'])
        if not spec['minimum'] <= value <= spec['maximum']:
            raise ValueError("VM270 parameter outside supported range: " + spec['name'])
    return result


def positions(p):
    """Source exact radii, lengths and label locations, in millimeters."""
    p = _parameters(p)
    start = 1.6 + .127 + p['displacement']
    return {'magnet_radius': 2.9972 / 2, 'magnet_length': 1.6,
            'coil_inner_radius': 3.048 / 2, 'coil_outer_radius': 3.9624 / 2,
            'coil_start': start, 'coil_end': start + 1.524,
            'magnet_label': (2.9972 / 4, .8),
            'coil_label': ((3.048 + 3.9624) / 4, start + 1.524 / 2),
            'boundary_center_z': 1.6 / 3}


def build_vm270(case, p):
    """Return geometry, construction Lua lines, probes, extent, radius, m/unit."""
    # Local imports permit simulation's builder dispatch to import this module.
    from .simulation import _block, _call, _draw_script
    p = _parameters(p)
    q = positions(p)
    radius, center = p['domain_radius'], q['boundary_center_z']
    d = _Drawing()
    d.polygon([(0, 0), (q['magnet_radius'], 0),
               (q['magnet_radius'], q['magnet_length']), (0, q['magnet_length'])])
    d.polygon([(q['coil_inner_radius'], q['coil_start']), (q['coil_outer_radius'], q['coil_start']),
               (q['coil_outer_radius'], q['coil_end']), (q['coil_inner_radius'], q['coil_end'])])
    # This artificial interface separates identical air material solely to
    # resolve Maxwell stress near the permanent magnet. It is not a physical
    # boundary and carries no constraint; exterior air can remain much coarser.
    pocket_top = q['coil_end'] + 1.
    d.line((0., -1.), (4., -1.))
    d.line((4., -1.), (4., pocket_top))
    d.line((4., pocket_top), (0., pocket_top))
    envelope_r = q['magnet_radius'] + .3
    # Keep the artificial mesh interface below the winding even when a user
    # requests zero source displacement; it must never cut the copper region.
    envelope_bottom = -.3
    envelope_top = min(q['magnet_length'] + .3, q['coil_start'] - .03)
    d.line((0., envelope_bottom), (envelope_r, envelope_bottom))
    d.line((envelope_r, envelope_bottom), (envelope_r, envelope_top))
    d.line((envelope_r, envelope_top), (0., envelope_top))
    d.arc((0, center-radius), (0, center+radius), 180)
    # The existing magnet axis edge supplies the middle segment, avoiding an
    # overlapping long axis edge that native FEMM would implicitly split.
    axis_points = (center-radius, -1., envelope_bottom, 0., q['magnet_length'], envelope_top, pocket_top, center+radius)
    for a, b in zip(axis_points, axis_points[1:]):
        d.line((0, a), (0, b))
    d.label(*q['magnet_label'], 'SmCo magnet')
    d.label(*q['coil_label'], '280-turn copper coil')
    d.label(q['magnet_radius']+.15, .8, 'fine air envelope')
    d.label(.1, -.5, 'fine local air')
    d.label(5., center, 'air and nonmagnetic spool')
    geometry = d.result()
    geometry['metadata'] = {'LengthUnits': 'millimeters', 'ProblemType': 'axisymmetric'}
    lines = ['newdocument(0)', _call('mi_probdef', 0, 'millimeters', 'axi', 1e-9, 1, 30),
             *_draw_script(geometry), _call('mi_addmaterial', 'air', 1, 1, 0, 0, 0),
             _call('mi_addmaterial', 'copper', 1, 1, 0, 0, 0),
             _call('mi_addmaterial', 'SmCo', RELATIVE_PERMEABILITY, RELATIVE_PERMEABILITY, COERCIVITY, 0, 0),
             _call('mi_addcircprop', 'icoil', p['current'], 1)]
    _block(lines, *q['magnet_label'], 'SmCo', .005*p['mesh_scale'], direction=90, group=1, turns=1)
    _block(lines, *q['coil_label'], 'copper', .01*p['mesh_scale'], 'icoil', group=2, turns=280)
    _block(lines, q['magnet_radius']+.15, .8, 'air', .005*p['mesh_scale'], turns=1)
    _block(lines, .1, -.5, 'air', .01*p['mesh_scale'], turns=1)
    # Fixed absolute air size makes domain and local mesh changes independent.
    _block(lines, 5., center, 'air', .1, turns=1)
    lines += [_call('mi_addboundprop', 'A=0', 0, 0, 0, 0, 0, 0, 0, 0, 0),
              _call('mi_selectarcsegment', radius, center),
              _call('mi_setarcsegmentprop', 1, 'A=0', 0, 0), 'mi_clearselected()']
    probes = [(0.1, .8), (1., (q['coil_start']+q['magnet_length'])/2)]
    return geometry, lines, probes, (0., 4., -1., q['coil_end']+1.), radius, .001


def metric_lines(p):
    """Lua metric statements, after solution is loaded and metrics CSV opened."""
    from .simulation import _call
    p = _parameters(p)
    q = positions(p)
    return [
        _call('mo_selectblock', *q['magnet_label']),
        'magnet_fz=mo_blockintegral(19)', 'mo_clearblock()',
        _call('mo_selectblock', *q['coil_label']),
        'coil_lorentz_fz=mo_blockintegral(12)', 'coil_wst_fz=mo_blockintegral(19)', 'mo_clearblock()',
        'amps,volts,linkage=mo_getcircuitproperties("icoil")',
        'write(out,"force_magnet_z_N,",magnet_fz,",N\\n")',
        'write(out,"force_coil_z_N,",coil_lorentz_fz,",N\\n")',
        'write(out,"force_coil_wst_z_N,",coil_wst_fz,",N\\n")',
        'write(out,"current_A,",amps,",A\\n")',
        'write(out,"flux_linkage_Wb_turn,",linkage,",Wb-turn\\n")',
    ]


def assess_vm270(plan, metrics):
    """Compare both independent force estimators and signs to a frozen target."""
    reference = plan['frozen_reference']
    if not reference['applicable']:
        return {'status': 'unassessed', 'reference': reference, 'reason': reference.get('reason')}
    observed = {key: metrics[key] for key in ('force_magnet_z_N', 'force_coil_z_N', 'force_coil_wst_z_N')}
    if not all(math.isfinite(v) for v in observed.values()):
        raise ValueError('Native VM270 force metrics must be finite')
    target = reference['value']
    tolerance = abs(target)*reference['relative_tolerance']
    primary_error = abs(observed['force_magnet_z_N']-target)
    lorentz_error = abs(observed['force_coil_z_N']+target)
    balance = abs(observed['force_magnet_z_N']+observed['force_coil_z_N'])
    crosscheck = abs(observed['force_coil_wst_z_N']-observed['force_coil_z_N'])
    passed = (primary_error <= tolerance and lorentz_error <= tolerance and balance <= tolerance
              and crosscheck <= tolerance and observed['force_magnet_z_N'] > 0
              and observed['force_coil_z_N'] < 0 and observed['force_coil_wst_z_N'] < 0)
    from .simulation import _compare_reference
    references = plan.get('frozen_references') or [reference, {**reference, 'metric': 'force_coil_z_N', 'value': -target}]
    comparisons = [_compare_reference(item, metrics) for item in references]
    passed = passed and all(item['status'] == 'passed' for item in comparisons)
    return {'status': 'passed' if passed else 'failed', 'metric': 'force_magnet_z_N',
            'observed': observed['force_magnet_z_N'], 'expected': target,
            'absolute_error': primary_error, 'relative_error': primary_error/abs(target),
            'absolute_tolerance': tolerance, 'coil_lorentz_relative_error': lorentz_error/abs(target),
            'action_reaction_imbalance_N': balance, 'coil_estimator_difference_N': crosscheck,
            'reference': reference, 'targets': comparisons,
            'consistency': {'status': 'passed' if balance <= tolerance and crosscheck <= tolerance else 'failed',
                            'action_reaction_imbalance_N': balance, 'coil_estimator_difference_N': crosscheck,
                            'absolute_tolerance_N': tolerance}}


def verify_vm270(plan, actual):
    """Inspect native assignments and every exterior attachment independently."""
    p = _parameters(plan['parameters'])
    q = positions(p)
    if geometry_signature(plan['geometry']) != geometry_signature(actual):
        raise ValueError('Native VM270 geometry differs from the frozen typed geometry')
    def close(a, b):
        return math.isclose(float(a), b, rel_tol=1e-11, abs_tol=1e-12)
    meta = actual['metadata']
    if meta.get('LengthUnits') != 'millimeters' or meta.get('ProblemType') != 'axisymmetric' or not close(meta.get('Frequency', -1), 0):
        raise ValueError('VM270 requires millimeter, axisymmetric DC physics')
    if len(actual['materials']) != 3:
        raise ValueError('VM270 requires exactly air, copper and SmCo materials')
    material_index = {}
    expected = {'air': (1., 0.), 'copper': (1., 0.), 'SmCo': (RELATIVE_PERMEABILITY, COERCIVITY)}
    for i, material in enumerate(actual['materials'], 1):
        name, props = material['name'], material['properties']
        if name not in expected or name in material_index:
            raise ValueError('VM270 has an unexpected or duplicate material')
        mu, hc = expected[name]
        if material.get('bh_points') or not all(close(props.get(k, float('nan')), v) for k, v in (
                ('Mu_x', mu), ('Mu_y', mu), ('H_c', hc), ('J_re', 0), ('J_im', 0),
                ('Sigma', 0), ('LamType', 0), ('LamFill', 1))):
            raise ValueError('VM270 native material law differs: ' + name)
        material_index[name] = i
    if len(actual['circuits']) != 1:
        raise ValueError('VM270 requires exactly one series circuit')
    circuit = actual['circuits'][0]
    if circuit['name'] != 'icoil' or not all(close(circuit['properties'].get(k, float('nan')), v) for k, v in (
            ('TotalAmps_re', p['current']), ('TotalAmps_im', 0), ('CircuitType', 1))):
        raise ValueError('VM270 source winding current or circuit differs')
    if len(actual['labels']) != 5:
        raise ValueError('VM270 requires magnet, coil and three air mesh regions')
    for name, pos, circuit_id, angle, group, turns, mesh in (
            ('SmCo', q['magnet_label'], 0, 90, 1, 1, .005*p['mesh_scale']),
            ('copper', q['coil_label'], 1, 0, 2, 280, .01*p['mesh_scale']),
            ('air', (q['magnet_radius']+.15, .8), 0, 0, 0, 1, .005*p['mesh_scale']),
            ('air', (.1, -.5), 0, 0, 0, 1, .01*p['mesh_scale']),
            ('air', (5., q['boundary_center_z']), 0, 0, 0, 1, .1)):
        labels = [label for label in actual['labels'] if close(label['x'], pos[0]) and close(label['y'], pos[1])]
        if len(labels) != 1:
            raise ValueError('VM270 label position differs: ' + name)
        attrs = labels[0].get('attributes', [])
        if len(attrs) < 6 or not all(close(attrs[k], value) for k, value in (
                (0, material_index[name]), (1, mesh), (2, circuit_id), (3, angle), (4, group), (5, turns))):
            raise ValueError('VM270 native region assignment differs: ' + name)
    if len(actual.get('boundaries', [])) != 1:
        raise ValueError('VM270 requires one finite A=0 boundary')
    boundary = actual['boundaries'][0]
    if boundary['name'] != 'A=0' or not all(close(boundary['properties'].get(k, float('nan')), 0) for k in (
            'BdryType', 'A_0', 'A_1', 'A_2', 'Phi', 'c0', 'c1')):
        raise ValueError('VM270 exterior must retain finite A=0 adaptation')
    if len(actual['arcs']) != 1:
        raise ValueError('VM270 requires one exterior semicircular arc')
    arc = actual['arcs'][0]
    a, b = (actual['nodes'][arc[k]] for k in ('n0', 'n1'))
    circle = arc_geometry((a['x'], a['y']), (b['x'], b['y']), arc['angle'])
    attrs = arc.get('attributes', [])
    if not close(circle['radius'], p['domain_radius']) or not close(circle['center'][0], 0) or not close(circle['center'][1], q['boundary_center_z']) or len(attrs) < 3 or not close(attrs[2], 1) or not close(attrs[1], 1):
        raise ValueError('VM270 A=0 boundary attachment or one-degree arc subdivision differs')
    if any(len(seg.get('attributes', [])) < 2 or not close(seg['attributes'][1], 0) for seg in actual['segments']):
        raise ValueError('VM270 axis/interior contours must not receive the exterior boundary')
    if any(len(node.get('attributes', [])) < 1 or not close(node['attributes'][0], 0) for node in actual['nodes']):
        raise ValueError('VM270 must not acquire unexpected point boundary conditions')
    return {'status': 'passed', 'coordinate_system': 'axisymmetric', 'length_units': 'millimeters',
            'materials': 3, 'turns': 280, 'circuit_count': 1, 'magnetization_degrees': 90,
            'relative_permeability': RELATIVE_PERMEABILITY,
            'boundary_attachments': {'status': 'passed', 'assigned_exterior_arcs': 1,
                                     'unassigned_segments': len(actual['segments'])}}
