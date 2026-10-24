"""Independent native re-reading of saved OpenRouter/MCP trial results."""
import json
import math
from pathlib import Path
import sys

from femm_agent.femfile import parse_file
from femm_agent.server import build_server
from femm_agent.sessions import SessionManager
from verify_natural_models import verify_plan,verify_capacitor

source=Path(sys.argv[1]);output=Path(sys.argv[2])
assert not output.exists()
report=json.loads(source.read_text())
cases=json.loads(Path('/app/evaluation/prompt_cases.json').read_text())
expected_base=next(case['expected'] for case in cases if case['id']=='asymmetric_c_core')
manager=SessionManager(default_work_dir=output.parent/'audit-sessions')
_,ctx=build_server(manager)
tools=ctx.extras['tools']
result={'source_report':source.name,'checks':[],'upstream_modified':False}
def call(tool_name,**arguments):
    response=tools[tool_name](**arguments)
    if not response['ok']:
        raise RuntimeError(str(response['error']))
    return response['result']
try:
    for case in report['cases']:
        for trial in case['prompts']:
            item={'case':case['id'],'prompt_number':trial['number']}
            result['checks'].append(item)
            model=Path(trial['native_model'])
            state=next(state for state in trial['native_sessions']['result'] if state.get('document'))
            native=Path(state['document'])
            parsed=parse_file(model)
            if native.suffix in {'.ans','.res','.anh','.anc'}:
                item['agent_session_document_extension']=native.suffix
                native=native.with_suffix({'magnetics':'.fem','electrostatics':'.fee','heatflow':'.feh','currentflow':'.fec'}[parsed.physics])
            call('femm_open',path=str(native))
            call('femm_load_solution')
            if case['id']=='asymmetric':
                expected=json.loads(json.dumps(expected_base))
                if trial['number']==2:
                    for region in expected['regions']:
                        if region['shape']['type']=='polygon':
                            region['shape']['points'][5]['x']=14
                            region['shape']['points'][6]['x']=14
                item['saved_model']=verify_plan(expected,model)
                item['probe']=call('femm_point_values',points=[[3,0]])
                labels=[[expected['air_label']['x'],expected['air_label']['y']]]+[[region['label']['x'],region['label']['y']] for region in expected['regions']]
                integral=call('femm_block_integral',points=labels,codes=[2])
                item['all_regions_energy']=integral
                actual_energy=next(iter(integral['results'].values()))
                circuit=call('femm_circuit_results',name=parsed.circuits()[0]['CircuitName'])
                item['circuit']=circuit
                item['energy_flux_identity_relative_error']=abs(actual_energy-.5*circuit['current_A']*circuit['flux_linkage_Wb'])/actual_energy
                energy_calls=[tool for tool in trial['tool_calls'] if tool['name']=='femm_block_integral' and tool['response'].get('ok') and '2: magnetic field energy' in tool['response'].get('result',{}).get('results',{})]
                last=energy_calls[-1]
                reported=last['response']['result']['results']['2: magnetic field energy']
                item['agent_energy_integration_arguments']=last['arguments']
                item['agent_reported_total_energy_J']=reported
                item['actual_total_energy_J']=actual_energy
                item['energy_claim_correct']=math.isclose(reported,actual_energy,rel_tol=.01)
                item['energy_claim_relative_error']=abs(reported-actual_energy)/actual_energy
                item['geometry_assignments_mesh_probe_passed']=(item['saved_model']['geometry_and_assignments_passed'] is True and item['saved_model']['mesh_constraint_passed'] is True)
                item['request_passed']=item['geometry_assignments_mesh_probe_passed'] and item['energy_claim_correct']
            elif case['id']=='capacitor':
                item['saved_model']=verify_capacitor(model)
                item['probe']=call('femm_point_values',points=[[5,.5]])
                item['hot']=call('femm_circuit_results',name='hot')
                item['energy']=call('femm_block_integral',points=[[5,.5]],codes=[0])
                expected_capacitance=8.8541878128e-12*4*(.01*.1)/.001
                measured_capacitance=item['hot']['charge_C']/10
                point=item['probe']['values'][0]
                item['analytic_checks']={'centre_voltage':math.isclose(point['V'],5,rel_tol=.01),
                    'electric_field':math.isclose(point['|E|'],10000,rel_tol=.01),
                    'capacitance':math.isclose(measured_capacitance,expected_capacitance,rel_tol=.01)}
                energy=next(iter(item['energy']['results'].values()))
                if isinstance(energy,list):
                    energy=energy[0]
                item['analytic_checks']['stored_energy']=math.isclose(energy,.5*expected_capacitance*100,rel_tol=.01)
                item['explicit_mesh_limit_passed']=bool(parsed.block_labels()) and all(0<label['mesh_size']<=.1+1e-10 for label in parsed.block_labels())
                item['expected_capacitance_F']=expected_capacitance
                item['measured_capacitance_F']=measured_capacitance
                item['request_passed']=item['saved_model']['geometry_and_assignments_passed'] is True and item['explicit_mesh_limit_passed'] and all(item['analytic_checks'].values())
            else:
                points=parsed.points()
                outer_arcs=[arc for arc in parsed.geometry.get('NumArcSegments',[]) if all(math.isclose(math.hypot(*points[int(index)]),50,rel_tol=1e-7) for index in arc[:2])]
                boundaries=parsed.boundaries()
                def prescribed_zero(arc):
                    index=int(arc[4])
                    if not 0<index<=len(boundaries):
                        return False
                    boundary=boundaries[index-1]
                    return int(boundary.get('BdryType',-1))==0 and float(boundary.get('A_0',0))==0
                item['outer_arc_count']=len(outer_arcs)
                item['outer_arc_A0_count']=sum(prescribed_zero(arc) for arc in outer_arcs)
                item['entire_outer_boundary_A0_passed']=bool(outer_arcs) and all(prescribed_zero(arc) for arc in outer_arcs)
                item['outer_arcs']=outer_arcs
                sweeps=[tool['response']['result'] for tool in trial['tool_calls'] if tool['name']=='femm_experiment_ac_sweep' and tool['response'].get('ok')]
                item['actual_ac_sweep']=sweeps[-1] if sweeps else None
                item['request_passed']=item['entire_outer_boundary_A0_passed'] and bool(sweeps)
finally:
    manager.close_all()
output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'output':str(output),'outcomes':[{key:item.get(key) for key in ['case','prompt_number','request_passed','actual_total_energy_J','agent_reported_total_energy_J','outer_arc_count','outer_arc_A0_count']} for item in result['checks']]}))
