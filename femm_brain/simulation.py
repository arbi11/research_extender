"""Source-backed, deterministic geometry generation and native magnetic solves.

No downloaded code is executed, and no AI service is called. Published FEMM
numbers and analytic references are frozen before each run; modified geometry
never inherits a published baseline target. Native mesh, solver and numerical
checks are reported separately, including failures and unassessed targets.
"""
from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from .core import load_graph
from .geometry import _Drawing, _svg, arc_geometry, parse_fem
from .native_check import _run, _version, geometry_signature
from .source_recipe import compile_source, load_source, verify_source_physics

DATA = Path(__file__).parent / "data"
MU0 = 4e-7 * math.pi
PARAMETERS = {
    "gapped_ei": [
        {"name":"current", "unit":"A", "description":"Signed DC series winding current; zero is excluded because linkage/current defines inductance", "minimum":-1000, "maximum":1000},
        {"name":"mesh_scale", "unit":"ratio", "description":"Multiplier on source block/contour mesh lengths; smaller values refine the mesh", "minimum":0.25, "maximum":4}
    ],
    "radial_bearing": [
        {"name":"current_scale", "unit":"ratio", "description":"Multiplier on all four source circuits (6,12,6,0 A); nonlinear force is not assumed to scale quadratically", "minimum":-2, "maximum":2},
        {"name":"mesh_scale", "unit":"ratio", "description":"Multiplier on source block/contour mesh lengths; smaller values refine the mesh", "minimum":0.25, "maximum":4}
    ],
    "aircore": [
        {"name":"inner_radius", "unit":"in", "description":"Inner winding radius", "minimum":0.001, "maximum":100},
        {"name":"outer_radius", "unit":"in", "description":"Outer winding radius", "minimum":0.002, "maximum":100},
        {"name":"length", "unit":"in", "description":"Axial winding length", "minimum":0.001, "maximum":100},
        {"name":"turns", "unit":"count", "description":"Series winding turns", "minimum":1, "maximum":100000, "integer":True},
        {"name":"current", "unit":"A", "description":"DC circuit current (inductance is current-independent in this linear model)", "minimum":0.000001, "maximum":100000},
        {"name":"mesh_scale", "unit":"ratio", "description":"Mesh length multiplier; smaller values refine the mesh", "minimum":0.25, "maximum":4},
        {"name":"boundary_scale", "unit":"ratio", "description":"Multiplier on the source domain radius 2*max(ri,ro,length)", "minimum":1, "maximum":4}
    ],
    "parallel_wires": [
        {"name":"wire_radius","unit":"m","description":"Radius of each circular wire","minimum":0.001,"maximum":5},
        {"name":"center_distance","unit":"m","description":"Center-to-center separation of the two wires","minimum":0.01,"maximum":100},
        {"name":"current","unit":"A","description":"Magnitude of equal and opposite DC currents","minimum":0.000001,"maximum":100000},
        {"name":"domain_radius","unit":"m","description":"Finite circular A=0 boundary radius; differs from source infinite elements","minimum":0.1,"maximum":1000},
        {"name":"mesh_scale","unit":"ratio","description":"Mesh length multiplier; smaller values refine the mesh","minimum":0.25,"maximum":4}
    ],
    "uniform": [
        {"name":"radius", "unit":"mm", "description":"Interior circular/spherical region radius", "minimum":0.1, "maximum":10000},
        {"name":"applied_field_scale", "unit":"ratio", "description":"Multiplier on source shell coercivity; zero and negative fields are supported", "minimum":-10, "maximum":10},
        {"name":"mesh_scale", "unit":"ratio", "description":"Mesh length multiplier; smaller values refine the mesh", "minimum":0.25, "maximum":4}
    ]
}


def _cases():
    return json.loads((DATA / "simulations.json").read_text())["cases"]


def simulation_catalog(graph=None):
    """Return editable simulations with genuine knowledge-graph bindings."""
    graph = load_graph() if graph is None else graph
    examples = {item["id"]: item for item in graph["examples"]}
    node_ids = {item["id"] for item in graph["nodes"]}
    source_index = {item["id"]:item for item in graph["sources"]}
    artifact_index = {item["sha256"]:item for item in graph.get("artifacts",[]) if item.get("sha256")}
    result = []
    for case in _cases():
        item = deepcopy(case)
        example = examples.get(case["example_id"])
        if example is None:
            raise ValueError("Simulation has no corresponding knowledge example: " + case["example_id"])
        item["knowledge_node_ids"] = list(example["node_ids"])
        if any(ident not in source_index for ident in case["source_ids"]):
            raise ValueError("Simulation references an unavailable knowledge source")
        if set(case["source_ids"]) & set(graph.get("source_changes",[])):
            raise ValueError("Simulation knowledge sources changed after curation; review before execution")
        item["knowledge_sources"] = [{k:source_index[ident].get(k) for k in ("id","url","sha256")}
                                     for ident in case["source_ids"]]
        item["knowledge_artifacts"] = []
        if case.get("source_model_sha256"):
            asset=artifact_index.get(case["source_model_sha256"])
            if asset is None or not asset.get("geometry_available"):
                raise ValueError("Simulation's source model is unavailable in the knowledge graph")
            item["knowledge_artifacts"] = [{k:asset.get(k) for k in ("id","url","sha256","geometry_model_file")}]

        if any(ident not in node_ids for ident in item["knowledge_node_ids"]):
            raise ValueError("Simulation references an unavailable knowledge node")
        item["parameters"] = [dict(p, default=case["defaults"][p["name"]], integer=p.get("integer", False))
                              for p in PARAMETERS[case.get("parameter_family",case["family"])]]
        item["physics"] = list(example.get("physics", []))
        item["requires_ai"] = False
        item["dimension_source"] = "curated_original_native_model" if case["family"] == "native_source" else "normalized_source_construction" if case["family"] == "uniform" else "published_parameters"
        result.append(item)
    return result


def _validate_parameters(case, overrides):
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("Parameters must be a JSON object")
    overrides = overrides or {}
    specs = {p["name"]: p for p in case["parameters"]}
    unknown = set(overrides) - set(specs)
    if unknown:
        raise ValueError("Unknown simulation parameters: " + ", ".join(sorted(unknown)))
    parameters = dict(case["defaults"], **overrides)
    for name, value in parameters.items():
        spec = specs[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(name + " must be a finite number")
        if not spec["minimum"] <= value <= spec["maximum"]:
            raise ValueError(f"{name} must be between {spec['minimum']} and {spec['maximum']} {spec['unit']}")
        if spec.get("integer") and not float(value).is_integer():
            raise ValueError(name + " must be an integer")
    if case["family"] == "aircore" and parameters["outer_radius"] <= parameters["inner_radius"]:
        raise ValueError("outer_radius must be greater than inner_radius")
    if case["family"] == "parallel_wires":
        if 2*parameters['wire_radius'] >= parameters['center_distance']:
            raise ValueError("Parallel wires must not touch or overlap")
        if parameters['domain_radius'] < 2*(parameters['center_distance']/2+parameters['wire_radius']):
            raise ValueError("Exterior domain must enclose the wires with at least their bounding radius of clearance")
    if case["id"] == "gapped-ei-core" and abs(parameters["current"]) < 1e-6:
        raise ValueError("EI current magnitude must be at least 1e-6 A to define linkage/current inductance")
    return parameters


def _call(name, *values):
    def lua(value):
        return json.dumps(value) if isinstance(value, str) else format(value, ".17g")
    return name + "(" + ",".join(lua(v) for v in values) + ")"


def _draw_script(drawing):
    return [_call(item["function"], *item["arguments"]) for item in drawing["operations"]]


def _block(lines, x, y, material, mesh, circuit="<None>", direction=0, group=0, turns=0):
    lines.extend([_call("mi_selectlabel", x, y),
                  _call("mi_setblockprop", material, 0, mesh, circuit, direction, group, turns),
                  "mi_clearselected()"])


def _aircore(case, p):
    ri, ro, length = p["inner_radius"], p["outer_radius"], p["length"]
    source_radius = 2 * max(ri, ro, length)
    radius = p["boundary_scale"] * source_radius
    d = _Drawing()
    d.polygon([(ri,-length/2),(ro,-length/2),(ro,length/2),(ri,length/2)])
    d.arc((0,-radius),(0,radius),180)
    d.line((0,-radius),(0,radius))
    d.label((ri+ro)/2,0,"coil")
    d.label(.75*radius,0,"air")
    geometry = d.result()
    geometry["metadata"] = {"LengthUnits":"inches", "ProblemType":"axisymmetric"}
    lines = ['newdocument(0)', 'mi_probdef(0,"inches","axi",1e-8,0,30)', *_draw_script(geometry),
             'mi_addmaterial("coil",1,1,0,0,0)', 'mi_addmaterial("air",1,1,0,0,0)',
             _call("mi_addcircprop", "icoil", p["current"], 1)]
    _block(lines,(ri+ro)/2,0,"coil",radius/20*p["mesh_scale"],"icoil",turns=p["turns"])
    _block(lines,.75*radius,0,"air",radius/100*p["mesh_scale"])
    lines.extend([_call("mi_addboundprop","abc",0,0,0,0,0,0,1/(radius*.0254*MU0),0,2),
                  _call("mi_selectarcsegment",radius,0), _call("mi_setarcsegmentprop",5,"abc",0,0),
                  "mi_clearselected()"])
    probes = [(0,0)]
    field_extent = (0, radius*.95, -radius*.95, radius*.95)
    return geometry, lines, probes, field_extent, radius, .0254


def _uniform(case, p):
    # The graph's passive original model is data only: extract source shell
    # permeability coefficients, never execute its source operations or scripts.
    model_path = DATA / "models" / (case["source_model_sha256"] + ".json")
    source_model = json.loads(model_path.read_text())
    if source_model.get("original_sha256") != case["source_model_sha256"]:
        raise ValueError("Source model provenance does not match its catalog binding")
    shells = source_model["materials"][:7]
    coefficients = [float(m["properties"]["Mu_x"]) for m in shells]
    if len(coefficients)!=7 or any(not math.isfinite(mu) or mu<=0 for mu in coefficients):
        raise ValueError("Source IABC shell coefficients are unavailable or invalid")
    if any(float(m["properties"]["Mu_y"]) != mu or m.get("bh_points") for m,mu in zip(shells,coefficients)):
        raise ValueError("Source IABC recipe requires isotropic linear shell permeability")
    axisymmetric = case["id"] == "uniform-axisymmetric"
    radius = p["radius"]
    d = _Drawing()
    for index in range(8):
        r = radius*(1+.1*index/7)
        if axisymmetric:
            d.arc((0,-r),(0,r),180)
        else:
            d.arc((0,-r),(0,r),180)
            d.arc((0,r),(0,-r),180)
    if axisymmetric:
        # Split the axis at shell intersections so native and intended topology agree.
        yy = sorted({point["y"] for point in d.nodes})
        for a,b in zip(yy,yy[1:]):
            d.line((0,a),(0,b))
    d.label(radius*.25,0,"air")
    for index in range(7):
        d.label(radius*(1+.1*(index+.5)/7),0,"shell_"+str(index+1))
    geometry = d.result()
    geometry["metadata"] = {"LengthUnits":"millimeters", "ProblemType":"axisymmetric" if axisymmetric else "planar"}
    # Original IABC shell arc segmentation is one degree.
    for item in geometry["operations"]:
        if item["function"] == "mi_addarc": item["arguments"][-1] = 1
    for item in geometry["arcs"]: item["max_segment_degrees"] = 1
    lines = ['newdocument(0)', _call("mi_probdef",0,"millimeters","axi" if axisymmetric else "planar",1e-8,1,30),
             *_draw_script(geometry), 'mi_addmaterial("air",1,1,0,0,0)']
    _block(lines,radius*.25,0,"air",radius/25*p["mesh_scale"])
    for index, mu in enumerate(coefficients):
        name = "shell_"+str(index+1)
        lines.append(_call("mi_addmaterial",name,mu,mu,p["applied_field_scale"]/MU0,0,0))
        _block(lines,radius*(1+.1*(index+.5)/7),0,name,radius/100*p["mesh_scale"],direction=90 if axisymmetric else 0)
    lines.append(_call("mi_addboundprop","A=0",0,0,0,0,0,0,0,0,0))
    lines.extend([_call("mi_selectarcsegment",1.1*radius,0), _call("mi_setarcsegmentprop",1,"A=0",0,0), "mi_clearselected()"])
    if not axisymmetric:
        lines.extend([_call("mi_selectarcsegment",-1.1*radius,0),_call("mi_setarcsegmentprop",1,"A=0",0,0),"mi_clearselected()"])
    # A center probe and four interior points check spatial uniformity.
    probes = [(radius*.01,0), (radius*.5,0), (radius*.25,radius*.5), (radius*.25,-radius*.5)]
    if not axisymmetric: probes.append((-radius*.5,0))
    field_extent = ((0 if axisymmetric else -radius*.9),radius*.9,-radius*.9,radius*.9)
    return geometry, lines, probes, field_extent, radius*1.1, .001



def _parallel_wires(case,p):
    radius,distance,domain=p['wire_radius'],p['center_distance'],p['domain_radius']
    d=_Drawing()
    d.circle(radius,-distance/2,0)
    d.circle(radius,distance/2,0)
    d.circle(domain)
    d.label(-distance/2,0,'left_wire')
    d.label(distance/2,0,'right_wire')
    d.label(0,domain*.5,'air')
    geometry=d.result()
    geometry['metadata']={'LengthUnits':'meters','ProblemType':'planar'}
    lines=['newdocument(0)',_call('mi_probdef',0,'meters','planar',1e-8,1,30),*_draw_script(geometry),
           _call('mi_addmaterial','wire',1,1,0,0,0),_call('mi_addmaterial','air',1,1,0,0,0),
           _call('mi_addcircprop','left_current',p['current'],1),_call('mi_addcircprop','right_current',-p['current'],1)]
    _block(lines,-distance/2,0,'wire',radius/20*p['mesh_scale'],'left_current',group=1,turns=1)
    _block(lines,distance/2,0,'wire',radius/20*p['mesh_scale'],'right_current',group=2,turns=1)
    _block(lines,0,domain*.5,'air',domain/40*p['mesh_scale'])
    lines.append(_call('mi_addboundprop','A=0',0,0,0,0,0,0,0,0,0))
    for angle in (45,135,225,315):
        x,y=domain*math.cos(math.radians(angle)),domain*math.sin(math.radians(angle))
        lines += [_call('mi_selectarcsegment',x,y),_call('mi_setarcsegmentprop',2,'A=0',0,0),'mi_clearselected()']
    probes=[(0,0)]
    extent=(-distance*1.5,distance*1.5,-distance*1.5,distance*1.5)
    return geometry,lines,probes,extent,domain,1.


def _native_source(case, p):
    source = load_source(DATA / "models" / (case["source_model_sha256"] + ".json"),
                         case["source_model_sha256"], case["source_model_data_sha256"])
    currents = ({"icoil": p["current"]} if case["id"] == "gapped-ei-core" else
                {c["name"]: float(c["properties"]["TotalAmps_re"]) * p["current_scale"] for c in source["circuits"]})
    geometry, lines, expected, metres = compile_source(source, mesh_scale=p["mesh_scale"], circuit_currents=currents)
    geometry["expected_source_physics"] = expected
    if case["id"] == "gapped-ei-core":
        probes = [(0, -.0125), (0, .5)]
        extent = (-1.2, 1.2, -.7, 1.2)
    else:
        # Visual diagnostics only; no point-value targets are claimed without
        # source-specified locations. Rotor iron is not the center air region.
        probes = [(0, 0), (.287013, .69291)]
        extent = (-2.45, 2.45, -2.45, 2.45)
    radius = max(math.hypot(n["x"], n["y"]) for n in geometry["nodes"]) * 1.01
    return geometry, lines, probes, extent, radius, metres

def _finish_script(lines, case, p, probes, extent, radius):
    lines += ['mi_saveas("model.fem")', 'out=openfile("geometry-completed.txt","w")',
              'write(out,"geometry saved\\n")', 'closefile(out)', 'mi_createmesh()',
              'out=openfile("mesh-completed.txt","w")', 'write(out,"mesh completed\\n")', 'closefile(out)',
              'mi_analyze(1)', 'mi_loadsolution()', 'out=openfile("metrics.csv","w")', 'write(out,"metric,value,unit\\n")']
    if case["family"] == "aircore":
        lines += ['amps,volts,linkage=mo_getcircuitproperties("icoil")',
                  'write(out,"current_A,",amps,",A\\n")', 'write(out,"flux_linkage_Wb_turn,",linkage,",Wb-turn\\n")',
                  'write(out,"inductance_H,",linkage/amps,",H\\n")']
    elif case['id']=='gapped-ei-core':
        lines += ['amps,volts,linkage=mo_getcircuitproperties("icoil")',
                  'write(out,"current_A,",amps,",A\\n")', 'write(out,"flux_linkage_Wb_turn,",linkage,",Wb-turn\\n")',
                  'write(out,"inductance_H,",linkage/amps,",H\\n")',
                  'mo_groupselectblock()', 'energy=mo_blockintegral(2)', 'mo_clearblock()',
                  'write(out,"energy_J,",energy,",J\\n")',
                  'write(out,"energy_inductance_H,",2*energy/(amps*amps),",H\\n")',
                  _call('mo_selectblock',0,-.125), 'fx=mo_blockintegral(18)', 'fy=mo_blockintegral(19)', 'mo_clearblock()',
                  'write(out,"force_I_x_N,",fx,",N\\n")', 'write(out,"force_I_y_N,",fy,",N\\n")']
    elif case['id']=='radial-bearing':
        lines += [_call('mo_selectblock',.287013,.69291), 'fx=mo_blockintegral(18)', 'fy=mo_blockintegral(19)', 'mo_clearblock()',
                  'write(out,"force_rotor_x_N,",fx,",N\\n")', 'write(out,"force_rotor_y_N,",fy,",N\\n")',
                  'mo_groupselectblock()', 'energy=mo_blockintegral(2)', 'mo_clearblock()',
                  'write(out,"energy_J,",energy,",J\\n")']
        for circuit in ('i1','i2','i3','i4'):
            lines += [_call('amps,volts,linkage=mo_getcircuitproperties',circuit),
                      f'write(out,"{circuit}_current_A,",amps,",A\\n")',
                      f'write(out,"{circuit}_flux_linkage_Wb_turn,",linkage,",Wb-turn\\n")']
    elif case['family']=='parallel_wires':
        distance=p['center_distance']
        lines += [_call('mo_selectblock',-distance/2,0),'left_fx=mo_blockintegral(18)','left_fy=mo_blockintegral(19)',
                  'left_lorentz_fx=mo_blockintegral(11)','left_lorentz_fy=mo_blockintegral(12)','mo_clearblock()',
                  _call('mo_selectblock',distance/2,0),'right_fx=mo_blockintegral(18)','right_fy=mo_blockintegral(19)',
                  'right_lorentz_fx=mo_blockintegral(11)','right_lorentz_fy=mo_blockintegral(12)','mo_clearblock()',
                  'write(out,"force_left_x_N,",left_fx,",N\\n")','write(out,"force_left_y_N,",left_fy,",N\\n")',
                  'write(out,"force_right_x_N,",right_fx,",N\\n")','write(out,"force_right_y_N,",right_fy,",N\\n")',
                  'write(out,"repulsive_force_N_per_m,",(right_fx-left_fx)/2,",N/m\\n")',
                  'write(out,"lorentz_left_x_N,",left_lorentz_fx,",N\\n")','write(out,"lorentz_left_y_N,",left_lorentz_fy,",N\\n")',
                  'write(out,"lorentz_right_x_N,",right_lorentz_fx,",N\\n")','write(out,"lorentz_right_y_N,",right_lorentz_fy,",N\\n")',
                  'write(out,"lorentz_repulsive_force_N_per_m,",(right_lorentz_fx-left_lorentz_fx)/2,",N/m\\n")']
    for index,(x,y) in enumerate(probes):
        lines += [_call("a,bx,by=mo_getpointvalues",x,y),
                  f'write(out,"probe_{index}_field_x_T,",bx,",T\\n")',
                  f'write(out,"probe_{index}_field_y_T,",by,",T\\n")',
                  f'write(out,"probe_{index}_field_magnitude_T,",sqrt(bx*bx+by*by),",T\\n")']
        if index == 0:
            lines += ['write(out,"field_x_T,",bx,",T\\n")','write(out,"field_y_T,",by,",T\\n")','write(out,"field_magnitude_T,",sqrt(bx*bx+by*by),",T\\n")']
    lines += ['closefile(out)', 'out=openfile("field.csv","w")', 'write(out,"x,y,bx,by\\n")']
    xmin,xmax,ymin,ymax=extent
    # Explicit finite sampling bounds are visual diagnostics, not solver settings.
    lines += ['for row=0,24 do', f'y={ymin:.17g}+row*{(ymax-ymin)/24:.17g}', 'for col=0,24 do',
              f'x={xmin:.17g}+col*{(xmax-xmin)/24:.17g}', f'if x*x+y*y < {(radius*.97)**2:.17g} then',
              'a,bx,by=mo_getpointvalues(x,y)', 'write(out,x,",",y,",",bx,",",by,"\\n")', 'end','end','end',
              'closefile(out)', 'out=openfile("solve-completed.txt","w")', 'write(out,"native field solve completed\\n")',
              'closefile(out)', 'mo_close()', 'mi_close()']
    return "\n".join(lines)+"\n"


def plan_simulation(case_id, parameters=None, graph=None):
    """Freeze provenance, geometry, script and numerical criteria before execution."""
    cases = {item["id"]:item for item in simulation_catalog(graph)}
    if case_id not in cases:
        raise ValueError("Unknown simulation: " + str(case_id))
    case = cases[case_id]
    p = _validate_parameters(case,parameters)
    builder={'aircore':_aircore,'uniform':_uniform,'parallel_wires':_parallel_wires,'native_source':_native_source}[case['family']]
    geometry,lines,probes,extent,radius,metres_per_unit = builder(case,p)
    reference = deepcopy(case["reference"])
    reference["applicable"] = True
    reference["tolerance_origin"] = "Predeclared benchmark acceptance tolerance; not a tolerance certified by the source"
    assumptions = ["DC magnetostatics; source material laws are retained.", "No external model or downloaded source script is executed."]
    limits = ["A passing target check is not a general convergence study or certification of a machine design."]
    comparators = []
    if case["family"] == "aircore":
        changed = [name for name in ("inner_radius","outer_radius","length","turns") if p[name] != case["defaults"][name]]
        if changed:
            reference["applicable"] = False
            reference["reason"] = "Published baseline does not apply after changing: " + ", ".join(changed)
        assumptions += ["All winding dimensions are inches, as in the source Octave function.",
                        "The source single mixed asymptotic outer boundary is retained; its coefficient follows domain radius.",
                        "Inductance is flux linkage divided by current, in this linear air-core model."]
        comparators = [{"kind":"published_wheeler_empirical", "value":case["wheeler_mH"]*.001,"unit":"H",
                        "applicable":not changed,"used_for_acceptance":False,
                        "note":"The source empirical formula has a different accuracy and geometry applicability from FEMM."}]
    elif case['family']=='parallel_wires':
        reference['value']=MU0*p['current']**2/(2*math.pi*p['center_distance'])
        reference['formula']='mu0 * current^2 / (2*pi*center_distance)'
        assumptions += ['Two nonoverlapping circular cross-sections carry equal and opposite uniform DC current; both have relative permeability one.',
                        'The planar out-of-plane depth is exactly one meter; native weighted stress tensor force in newtons therefore equals force per unit length numerically.',
                        'COMSOL reports Maxwell surface stress and volumetric Lorentz force. This transfer reports FEMM weighted stress as its primary estimator and FEMM Lorentz integration as a separate cross-check.',
                        case['boundary_adaptation']]
        limits += ['Transfer validation requires domain-radius and mesh-size sensitivity; this individual run does not establish infinite-domain convergence.']
    elif case['family']=='native_source':
        assumptions += ['All geometry, signed winding assignments, material properties, B-H points and boundary attachments are reconstructed from pinned passive native model data.',
                        'The original source model is never opened or executed; a fresh FEMM model is generated from typed entities.',
                        'Source coordinates and planar depth are inches; reported force and energy are SI quantities.']
        if case['id']=='gapped-ei-core':
            assumptions += ['Constant relative permeability 2500 and the source 66-turn series winding are retained.',
                            'Linear current scaling preserves inductance and scales stored energy and force by current squared.',
                            'Stored energy is integrated over every domain region; I-section force uses weighted stress tensor integrals 18/19 on label (0,-0.125) inches.']
        else:
            if p['current_scale'] != case['defaults']['current_scale']:
                reference.update(applicable=False,reason='The published nonlinear bearing force applies only to the original four circuit currents; no quadratic scaling is assumed.')
            assumptions += [case['source_discrepancy'],
                            'The 13-point source M-19 B-H curve is retained; no linear force or inductance scaling is assumed.',
                            'Rotor force selects the M-19 label (0.287013,0.69291) inches. The center (0,0) is air and is not used as the rotor selection.']
            limits += ['The source gives rounded force and field results. Only rotor force is an assessed published target because precise field-probe locations are not specified.']
    else:
        reference["value"] *= abs(p["applied_field_scale"])
        assumptions += ["Seven IABC shell permeabilities are read as data from the graph's cited original FEMM model.",
                        "The source circular/spherical construction is recentered at the origin and scaled to an editable radius in millimeters.",
                        "Uniform-field similarity and linear coercivity scaling justify the analytic magnitude target for edited radius or field scale.",
                        "Original shell magnetization directions are retained; signed field components are reported, while source targets are magnitudes."]
        comparators = [{"kind":"published_femm_result", "value":case["published_femm_value"]*abs(p["applied_field_scale"]),"unit":"T",
                        "used_for_acceptance":False,"note":"Published rounded numerical magnitude, separate from the analytic target."}]
    geometry["parameters"] = p
    geometry["source_ids"] = case["source_ids"]
    references=[reference]
    for item in case.get('additional_references',[]):
        extra=deepcopy(item)
        extra.update(applicable=reference['applicable'],tolerance_origin=reference['tolerance_origin'])
        if case['id']=='gapped-ei-core':
            extra['value'] *= (p['current']/case['defaults']['current'])**2
            extra['scaling'] = 'Linear source material law: baseline value multiplied by (current / baseline_current)^2'
        references.append(extra)
    plan = {"schema_version":1,"case_id":case_id,"title":case["title"],"family":case["family"],"parameters":p,
            "parameter_definitions":case["parameters"],"physics":case["physics"],"source_ids":case["source_ids"],
            "source_url":case["source_url"],"knowledge_node_ids":case["knowledge_node_ids"],
            "knowledge_sources":case["knowledge_sources"],"knowledge_artifacts":case["knowledge_artifacts"],
            "recipe_bindings":{"geometry":"method:methods_axisymmetric_coil_section" if case["family"]=="aircore" else
                              "method:methods_gapped_ei_core" if case_id=='gapped-ei-core' else
                              "method:machine_bearing_geometry" if case_id=='radial-bearing' else
                              "method:parallel_wire_cross_sections" if case['family']=='parallel_wires' else
                              "method:methods_uniform_field_iabc_axi" if case_id=="uniform-axisymmetric" else "method:methods_uniform_field_iabc_planar",
                              "solve":"method:methods_magnetic_solve_snapshot",
                              "postprocess":"method:methods_weighted_stress_force" if case["family"]=="parallel_wires" or case_id=='radial-bearing' else "method:methods_linear_flux_inductance" if case["family"]=="aircore" or case_id=='gapped-ei-core' else "method:methods_field_probe_or_contour"},
            "source_model_sha256":case.get("source_model_sha256"),
            "source_model_data_sha256":hashlib.sha256((DATA/"models"/(case["source_model_sha256"]+".json")).read_bytes()).hexdigest() if case.get("source_model_sha256") else None,
            "assumptions":assumptions,"limits":limits,
            "frozen_reference":reference,"frozen_references":references,"comparators":comparators,"geometry":geometry,"preview_svg":_svg(geometry),
            "lua":_finish_script(lines,case,p,probes,extent,radius),"probes":probes,"metres_per_coordinate_unit":metres_per_unit,
            "requires_ai":False,"numerical_convergence_verified":False,
            "compiler_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "catalog_sha256":hashlib.sha256((DATA/"simulations.json").read_bytes()).hexdigest()}
    if case['family']=='native_source':
        plan['source_recipe_sha256']=hashlib.sha256(Path(__file__).with_name('source_recipe.py').read_bytes()).hexdigest()
        plan['expected_source_physics']=geometry.pop('expected_source_physics')
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return plan


def read_native_mesh(folder, metres_per_unit):
    """Validate native connectivity and report geometric area in square mm."""
    def rows(name):
        path=Path(folder)/name
        if not path.is_file() or path.stat().st_size > 64*1024*1024:
            raise ValueError("Missing or oversized mesh artifact: "+name)
        return [line.split("#",1)[0].split() for line in path.read_text().splitlines() if line.split("#",1)[0].strip()]
    nr,er=rows("model.node"),rows("model.ele")
    nh,eh=[int(v) for v in nr[0]],[int(v) for v in er[0]]
    if len(nh)!=4 or nh[1]!=2 or nh[0]!=len(nr)-1 or nh[0]<3:
        raise ValueError("Native mesh node count or dimensionality is invalid")
    if len(eh)!=3 or eh[1:]!=[3,1] or eh[0]!=len(er)-1 or eh[0]<1:
        raise ValueError("Native mesh element count or topology is invalid")
    nodes={}
    for row in nr[1:]:
        i,x,y=int(row[0]),float(row[1]),float(row[2])
        if i in nodes or not math.isfinite(x) or not math.isfinite(y):raise ValueError("Invalid mesh node")
        nodes[i]=(x,y)
    elements=[]; areas=[]
    for row in er[1:]:
        vertex=[int(v) for v in row[1:4]]
        if len(set(vertex))!=3 or any(i not in nodes for i in vertex):raise ValueError("Invalid mesh connectivity")
        a,b,c=(nodes[i] for i in vertex)
        area=abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2
        region=float(row[4])
        if not math.isfinite(area) or area<=0 or not math.isfinite(region) or region<1 or not region.is_integer():
            raise ValueError("Invalid mesh element area or region")
        elements.append(vertex);areas.append(area)
    return {"status":"passed","node_count":len(nodes),"element_count":len(elements),
            "minimum_element_area_mm2":min(areas)*(metres_per_unit*1000)**2,
            "total_cross_section_area_mm2":sum(areas)*(metres_per_unit*1000)**2,
            "area_note":"Geometric cross-section area, not axisymmetric revolved volume", "nodes":nodes,"elements":elements}


def _mesh_svg(mesh, max_elements=100000):
    nodes=mesh["nodes"]
    xmin,xmax=min(p[0] for p in nodes.values()),max(p[0] for p in nodes.values())
    ymin,ymax=min(p[1] for p in nodes.values()),max(p[1] for p in nodes.values())
    span=max(xmax-xmin,ymax-ymin)
    paths=[]
    if not isinstance(max_elements, int) or max_elements < 1:
        raise ValueError("Mesh preview limit must be a positive integer")
    stride=max(1,math.ceil(len(mesh["elements"])/max_elements))
    shown=mesh["elements"][::stride]
    for v in shown:
        a,b,c=(nodes[i] for i in v)
        paths.append(f'M{a[0]:.7g},{-a[1]:.7g}L{b[0]:.7g},{-b[1]:.7g}L{c[0]:.7g},{-c[1]:.7g}Z')
    title=f'Native mesh preview: {len(shown)} of {len(mesh["elements"])} actual triangles. Full connectivity is retained in model.ele.'
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{xmin} {-ymax} {xmax-xmin} {ymax-ymin}"><title>{title}</title><path d="'+''.join(paths)+f'" fill="none" stroke="#457d79" stroke-width="{span/1800}"/></svg>'


def _field_svg(folder,geometry):
    """Render actual solved samples at their own extent, with clipped context.

    Native mesh geometry can span a very large exterior domain. Using that
    entire domain as the field viewport hides interior vectors; this view uses
    only the finite sampled coordinates and displays its actual magnitude range.
    Vector length is normalized and opacity carries magnitude information.
    """
    points=[]
    with (folder/"field.csv").open() as stream:
        for row in csv.DictReader(stream):
            vals=[float(row[k]) for k in ("x","y","bx","by")]
            if not all(math.isfinite(v) for v in vals):raise ValueError("Nonfinite native field sample")
            points.append(vals)
    if not points:raise ValueError("Native field sample set is empty")
    xmin,xmax=min(p[0] for p in points),max(p[0] for p in points)
    ymin,ymax=min(p[1] for p in points),max(p[1] for p in points)
    span=max(xmax-xmin,ymax-ymin,1e-9)
    # Retain a useful viewport even for a single diagnostic sample or one row.
    if xmin==xmax:xmin-=span*.05;xmax+=span*.05
    if ymin==ymax:ymin-=span*.05;ymax+=span*.05
    pad=span*.045
    plot_x,plot_y=xmin-pad,-ymax-pad
    plot_width,plot_height=xmax-xmin+2*pad,ymax-ymin+2*pad
    font=span/29
    legend_height=font*3.4
    magnitudes=[math.hypot(p[2],p[3]) for p in points]
    minimum,maximum=min(magnitudes),max(magnitudes)
    description=f"Native solved field vectors; sampled |B| range {minimum:.5g} to {maximum:.5g} T"
    # Reuse the exact line/arc geometry group without full-domain node/label
    # markers, and scale its stroke for this sampled viewport.
    geometry_svg=_svg(geometry)
    context=geometry_svg[geometry_svg.index('<g '):geometry_svg.index('</g>')+4]
    width_start=context.index('stroke-width="')+len('stroke-width="')
    width_end=context.index('"',width_start)
    context=context[:width_start]+format(span/650,'.12g')+context[width_end:]
    arrows=[]
    for (x,y,bx,by),mag in zip(points,magnitudes):
        if mag==0:continue
        length=span*.024
        dx,dy=bx/mag*length,by/mag*length
        opacity=.3+.7*(mag/maximum)**.35
        arrows.append(f'<path d="M{x-dx/2:.8g},{-y+dy/2:.8g}l{dx:.8g},{-dy:.8g}m{-dx*.28-dy*.2:.8g},{dy*.28-dx*.2:.8g}l{dx*.28+dy*.2:.8g},{-dy*.28+dx*.2:.8g}l{-dx*.28+dy*.2:.8g},{dy*.28+dx*.2:.8g}" opacity="{opacity:.4g}"/>')
    legend=f'|B|: {minimum:.5g} to {maximum:.5g} T'
    note='Zero field: no vectors.' if maximum==0 else 'Vector lengths normalized; opacity shows |B|.'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{plot_x:.12g} {plot_y:.12g} {plot_width:.12g} {plot_height+legend_height:.12g}" role="img" aria-label="Native solved field vectors">'
            f'<title>{description}</title><defs><clipPath id="field-sample-clip"><rect x="{plot_x:.12g}" y="{plot_y:.12g}" width="{plot_width:.12g}" height="{plot_height:.12g}"/></clipPath></defs>'
            f'<rect x="{plot_x:.12g}" y="{plot_y:.12g}" width="{plot_width:.12g}" height="{plot_height+legend_height:.12g}" fill="#fff"/>'
            f'<g clip-path="url(#field-sample-clip)"><g opacity=".65">{context}</g>'
            f'<g class="native-field-vectors" fill="none" stroke="#a13c10" stroke-width="{span/650:.12g}" stroke-linecap="round">'+''.join(arrows)+'</g></g>'
            f'<g fill="#233f3a" font-family="sans-serif" font-size="{font:.12g}">'
            f'<text x="{plot_x+pad*.4:.12g}" y="{plot_y+plot_height+font*1.25:.12g}">{legend}</text>'
            f'<text x="{plot_x+pad*.4:.12g}" y="{plot_y+plot_height+font*2.6:.12g}" font-size="{font*.75:.12g}">{note}</text></g></svg>')


def _compare_reference(ref, metrics):
    if not ref["applicable"]:
        return {"status":"unassessed","metric":ref['metric'],"reason":ref.get("reason"),"reference":ref}
    observed=metrics.get(ref["metric"])
    if observed is None or not math.isfinite(observed):
        raise ValueError("Native target metric is missing or nonfinite")
    absolute_error=abs(observed-ref["value"])
    # Zero-source runs have a strict absolute residual check instead of division by zero.
    relative_error=absolute_error/abs(ref["value"]) if ref["value"] else None
    tolerance=abs(ref["value"])*ref["relative_tolerance"] if ref["value"] else 1e-10
    return {"status":"passed" if absolute_error<=tolerance else "failed","metric":ref["metric"],
            "observed":observed,"expected":ref["value"],"absolute_error":absolute_error,
            "relative_error":relative_error,"absolute_tolerance":tolerance,"reference":ref}


def assess_target(plan,metrics):
    """Compare actual native values only against the already frozen criteria."""
    ref=plan["frozen_reference"]
    result=_compare_reference(ref,metrics)
    if plan['family']=='native_source':
        targets=[_compare_reference(item,metrics) for item in plan['frozen_references']]
        consistency={}
        if plan['case_id']=='gapped-ei-core':
            linkage=metrics['inductance_H'];energy_l=metrics['energy_inductance_H']
            if linkage<=0 or energy_l<=0 or metrics['energy_J']<=0:
                raise ValueError('Native EI energy and inductances must be positive')
            mismatch=abs(linkage-energy_l)/abs(linkage)
            cross=abs(metrics['force_I_x_N'])/max(abs(metrics['force_I_y_N']),1e-30)
            consistency={'status':'passed' if mismatch<=.001 and cross<=.001 and metrics['force_I_y_N']>0 else 'failed',
                         'linkage_energy_relative_difference':mismatch,'linkage_energy_relative_tolerance':.001,
                         'transverse_force_ratio':cross,'transverse_force_ratio_tolerance':.001,
                         'force_direction':'upward' if metrics['force_I_y_N']>0 else 'unexpected'}
        elif plan['parameters']['current_scale']==plan['parameter_definitions'][0]['default']:
            cross=abs(metrics['force_rotor_x_N'])/max(abs(metrics['force_rotor_y_N']),1e-30)
            consistency={'status':'passed' if cross<=.01 and metrics['force_rotor_y_N']>0 else 'failed',
                         'transverse_force_ratio':cross,'transverse_force_ratio_tolerance':.01,
                         'force_direction':'upward' if metrics['force_rotor_y_N']>0 else 'unexpected'}
        if any(t['status']=='failed' for t in targets) or consistency.get('status')=='failed':result['status']='failed'
        result.update(targets=targets,consistency=consistency)
        return result
    if result['status']=='unassessed':return result
    tolerance=result['absolute_tolerance'];observed=result['observed']
    if plan["family"]=="uniform":
        axis='y' if plan['case_id']=='uniform-axisymmetric' else 'x'
        transverse='x' if axis=='y' else 'y'
        deviation=max(abs(v-ref['value']) for k,v in metrics.items() if k.startswith('probe_') and k.endswith('field_magnitude_T'))
        cross=max(abs(v) for k,v in metrics.items() if k.startswith('probe_') and k.endswith('field_'+transverse+'_T'))
        uniformity=max(abs(v-metrics['field_'+axis+'_T']) for k,v in metrics.items() if k.startswith('probe_') and k.endswith('field_'+axis+'_T'))
        result.update(maximum_interior_magnitude_error_T=deviation,maximum_transverse_component_T=cross,maximum_signed_component_variation_T=uniformity)
        if deviation>tolerance or cross>max(tolerance,1e-10) or uniformity>max(tolerance,1e-10):result['status']='failed'
    if plan['family']=='parallel_wires':
        balance=abs(metrics['force_left_x_N']+metrics['force_right_x_N'])
        transverse=max(abs(metrics['force_left_y_N']),abs(metrics['force_right_y_N']))
        result.update(longitudinal_force_imbalance_N=balance,maximum_transverse_force_N=transverse,
                      lorentz_repulsive_force_N_per_m=metrics['lorentz_repulsive_force_N_per_m'],
                      lorentz_relative_error=abs(metrics['lorentz_repulsive_force_N_per_m']-ref['value'])/ref['value'],
                      force_estimator_difference_N_per_m=abs(metrics['lorentz_repulsive_force_N_per_m']-observed),
                      lorentz_status='passed' if abs(metrics['lorentz_repulsive_force_N_per_m']-ref['value'])<=tolerance else 'failed')
        if metrics['force_left_x_N']>=0 or metrics['force_right_x_N']<=0 or balance>tolerance or transverse>tolerance or result['lorentz_status']=='failed':
            result['status']='failed'

    return result




def verify_boundary_attachments(plan, actual):
    """Check every native contour assignment against the frozen exterior role.

    A boundary property definition alone does not constrain any geometry. All
    supported recipes attach their single boundary to the outer circular arc(s)
    only. Coil contours, IABC interfaces and axis segments remain unassigned.
    Coordinate keys make the check independent of native entity ordering.
    """
    if plan['family']=='native_source':
        return verify_source_physics(plan['expected_source_physics'],actual)['boundary_attachments']
    if len(actual.get('boundaries',[]))!=1:
        raise ValueError("Native recipe requires exactly one exterior boundary property")
    p=plan['parameters']
    if plan['family']=='aircore':
        outer_radius=2*max(p['inner_radius'],p['outer_radius'],p['length'])*p['boundary_scale']
    elif plan['family']=='parallel_wires':
        outer_radius=p['domain_radius']
    else:
        outer_radius=1.1*p['radius']
    def key(geometry,arc):
        points=[(round(geometry['nodes'][arc[n]]['x'],9),round(geometry['nodes'][arc[n]]['y'],9)) for n in ('n0','n1')]
        angle=arc['angle']
        if angle<0:points.reverse();angle=-angle
        return tuple(points)+(round(angle,9),)
    frozen=plan['geometry']; expected={}
    for arc in frozen['arcs']:
        a,b=(frozen['nodes'][arc[n]] for n in ('n0','n1'))
        circle=arc_geometry((a['x'],a['y']),(b['x'],b['y']),arc['angle'])
        outer=(math.isclose(circle['radius'],outer_radius,rel_tol=1e-11,abs_tol=1e-10)
               and math.hypot(*circle['center'])<=1e-10*max(1,outer_radius))
        expected[key(frozen,arc)]=1 if outer else 0
    seen=set()
    for arc in actual.get('arcs',[]):
        ident=key(actual,arc)
        attrs=arc.get('attributes',[])
        if ident not in expected or ident in seen or len(attrs)<3 or float(attrs[2])!=expected[ident]:
            raise ValueError("Native arc boundary attachment differs from frozen exterior/interior assignments")
        seen.add(ident)
    if seen!=set(expected):
        raise ValueError("Native exterior/interior arc assignments are incomplete")
    if len(actual.get('segments',[]))!=len(frozen['segments']):
        raise ValueError("Native axis/interior segment assignments are incomplete")
    for segment in actual.get('segments',[]):
        attrs=segment.get('attributes',[])
        if len(attrs)<2 or float(attrs[1])!=0:
            raise ValueError("Native axis/interior segment must have no boundary property attached")
    return {'status':'passed','assigned_exterior_arcs':sum(expected.values()),
            'unassigned_interior_arcs':len(expected)-sum(expected.values()),
            'unassigned_segments':len(frozen['segments'])}

def verify_native_physics(plan, actual):
    """Inspect the saved native physical assignments, independently of Lua success."""
    if plan['family']=='native_source':
        return verify_source_physics(plan['expected_source_physics'],actual)
    attachments=verify_boundary_attachments(plan,actual)
    expected=plan['geometry']['metadata']
    if any(actual['metadata'].get(k)!=v for k,v in expected.items()) or float(actual['metadata'].get('Frequency',-1))!=0:
        raise ValueError("Native coordinate system, units or DC problem definition differs from the frozen plan")
    materials={m['name']:m['properties'] for m in actual['materials']}
    def close(value,wanted):
        return math.isclose(float(value),wanted,rel_tol=1e-11,abs_tol=1e-12)
    def material(name,mux,muy,hc=0):
        m=materials.get(name,{})
        if not all(k in m and close(m[k],v) for k,v in [('Mu_x',mux),('Mu_y',muy),('H_c',hc)]):
            raise ValueError("Native material law differs from frozen recipe: "+name)
    index={i+1:m['name'] for i,m in enumerate(actual['materials'])}
    labels={index.get(l['material_index']):l for l in actual['labels']}
    p=plan['parameters']
    if plan['family']=='aircore':
        material('coil',1,1);material('air',1,1)
        if len(actual['circuits'])!=1:raise ValueError("Native winding must have exactly one circuit")
        circuit=actual['circuits'][0]
        if circuit['name']!='icoil' or not close(circuit['properties']['TotalAmps_re'],p['current']) or not close(circuit['properties']['CircuitType'],1):
            raise ValueError("Native winding circuit/current differs from frozen recipe")
        attrs=labels.get('coil',{}).get('attributes',[])
        if len(attrs)<6 or not close(attrs[2],1) or not close(attrs[5],p['turns']):
            raise ValueError("Native winding turns/circuit binding differs from frozen recipe")
        radius=2*max(p['inner_radius'],p['outer_radius'],p['length'])*p['boundary_scale']
        boundary=actual['boundaries'][0]['properties']
        if not close(boundary['BdryType'],2) or not close(boundary['c0'],1/(radius*.0254*MU0)) or not close(boundary['c1'],0):
            raise ValueError("Native asymptotic boundary differs from frozen recipe")
    elif plan['family']=='parallel_wires':
        material('wire',1,1);material('air',1,1)
        if not close(actual['metadata'].get('Depth',0),1):raise ValueError("Native transfer depth must be exactly one meter")
        circuits={c['name']:(i+1,c['properties']) for i,c in enumerate(actual['circuits'])}
        if len(circuits)!=2:raise ValueError("Parallel-wire transfer must have two circuits")
        for name,expected_current,xpos in [('left_current',p['current'],-p['center_distance']/2),('right_current',-p['current'],p['center_distance']/2)]:
            if name not in circuits:raise ValueError("Missing transfer wire circuit")
            ci,props=circuits[name]
            if not close(props['TotalAmps_re'],expected_current) or not close(props['CircuitType'],1):
                raise ValueError("Native parallel-wire current differs from frozen transfer")
            wire=next((l for l in actual['labels'] if close(l['x'],xpos) and close(l['y'],0)),None)
            if wire is None or not close(wire['attributes'][2],ci) or not close(wire['attributes'][5],1):
                raise ValueError("Native parallel-wire region/circuit binding differs from frozen transfer")
        boundary=actual['boundaries'][0]['properties']
        if not close(boundary['BdryType'],0) or not close(boundary['A_0'],0):raise ValueError("Native finite exterior adaptation must use A=0")
    else:
        material('air',1,1)
        model=json.loads((DATA/'models'/(plan['source_model_sha256']+'.json')).read_text())
        for i,source in enumerate(model['materials'][:7]):
            name='shell_'+str(i+1);mu=float(source['properties']['Mu_x'])
            material(name,mu,mu,p['applied_field_scale']/MU0)
            attrs=labels.get(name,{}).get('attributes',[])
            direction=90 if plan['case_id']=='uniform-axisymmetric' else 0
            if len(attrs)<4 or not close(attrs[3],direction):
                raise ValueError("Native shell magnetization direction differs from frozen recipe")
        if actual['circuits']:raise ValueError("Uniform-field construction must not have a winding circuit")
        boundary=actual['boundaries'][0]['properties']
        if not close(boundary['BdryType'],0) or not close(boundary['A_0'],0):
            raise ValueError("Native exterior zero-potential boundary differs from frozen recipe")
    return {"status":"passed","coordinate_system":expected['ProblemType'],"length_units":expected['LengthUnits'],
            "materials":len(materials),"circuit_count":len(actual['circuits']),"boundary_attachments":attachments}

def run_simulation(case_id, output, parameters=None, command=None, timeout=180, graph=None):
    """Build, mesh and solve one fresh case, retaining every stage and artifact."""
    executable=shutil.which(command or os.getenv("FEMM_LUA","femm-lua"))
    if executable is None:raise ValueError("Native FEMM 5 is required; set FEMM_LUA or pass command")
    if not isinstance(timeout,(int,float)) or isinstance(timeout,bool) or not math.isfinite(timeout) or timeout<=0:
        raise ValueError("Native timeout must be positive")
    plan=plan_simulation(case_id,parameters,graph)
    folder=Path(output).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    if any(folder.iterdir()):raise ValueError("Simulation output must be an empty directory; existing evidence is never overwritten")
    (folder/"plan.json").write_text(json.dumps(plan,indent=2,allow_nan=False)+"\n")
    (folder/"geometry.svg").write_text(plan["preview_svg"])
    script=folder/"model.lua";script.write_text(plan["lua"])
    report={"schema_version":1,"case_id":case_id,"title":plan["title"],"parameters":plan["parameters"],
            "plan_sha256":plan["plan_sha256"],"source_ids":plan["source_ids"],"source_url":plan["source_url"],
            "knowledge_node_ids":plan["knowledge_node_ids"],"knowledge_sources":plan["knowledge_sources"],
            "knowledge_artifacts":plan["knowledge_artifacts"],"frozen_reference":plan["frozen_reference"],
            "frozen_references":plan['frozen_references'],
            "started_at":datetime.now(timezone.utc).isoformat(),"engine":_version(executable),
            "checks":{name:{"status":"not_run"} for name in ("geometry","mesh","solve","target")},
            "requires_ai":False,"numerical_convergence_verified":False,"status":"failed","metrics":{}}
    started=time.monotonic();execution_error=None
    try:_run(executable,script,folder,timeout)
    except (ValueError,OSError,RuntimeError,subprocess.SubprocessError) as exc:execution_error=str(exc)
    stage='geometry'
    try:
        if not (folder/"geometry-completed.txt").is_file():raise ValueError("Native construction completion marker is missing")
        actual=parse_fem((folder/"model.fem").read_text())
        if geometry_signature(actual)!=geometry_signature(plan["geometry"]):raise ValueError("Native geometry differs from the frozen generated primitives")
        report['checks']['geometry']={"status":"passed","counts":{k:len(actual[k]) for k in ('nodes','segments','arcs','labels')},"physical_assignments":verify_native_physics(plan,actual)}
        stage='mesh'
        if not (folder/"mesh-completed.txt").is_file():raise ValueError("Native mesh completion marker is missing")
        mesh=read_native_mesh(folder,plan['metres_per_coordinate_unit'])
        (folder/'mesh.svg').write_text(_mesh_svg(mesh))
        report['checks']['mesh']={k:v for k,v in mesh.items() if k not in ('nodes','elements')}
        stage='solve'
        if not (folder/'solve-completed.txt').is_file() or not (folder/'model.ans').is_file():raise ValueError("Native field solution or completion marker is missing")
        metrics={}
        with (folder/'metrics.csv').open() as stream:
            for row in csv.DictReader(stream):
                value=float(row['value'])
                if row['metric'] in metrics or not math.isfinite(value):raise ValueError("Duplicate or nonfinite native metric")
                metrics[row['metric']]=value
        if not metrics:raise ValueError("Native metrics are empty")
        if (plan['family']=='aircore' or plan['case_id']=='gapped-ei-core') and metrics.get('inductance_H',0)<=0:raise ValueError("Native winding inductance is not positive")
        (folder/'field.svg').write_text(_field_svg(folder,plan['geometry']))
        report['metrics']=metrics
        report['checks']['solve']={"status":"passed","solution_bytes":(folder/'model.ans').stat().st_size}
        stage='target';report['checks']['target']=assess_target(plan,metrics)
        report['status']='passed' if report['checks']['target']['status']=='passed' else 'solved_unassessed' if report['checks']['target']['status']=='unassessed' else 'failed'
    except (ValueError,OSError,KeyError,IndexError,csv.Error) as exc:
        report['checks'][stage]={"status":"failed","error":str(exc)}
    if execution_error:
        report['execution_error']=execution_error
        report['status']='failed'
    report['duration_seconds']=round(time.monotonic()-started,4)
    report['completed_at']=datetime.now(timezone.utc).isoformat()
    report['artifacts']=[{"filename":p.name,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"size_bytes":p.stat().st_size}
                         for p in sorted(folder.iterdir()) if p.is_file()]
    (folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return report
