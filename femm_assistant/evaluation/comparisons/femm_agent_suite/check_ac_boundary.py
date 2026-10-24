"""Controlled correction on a COPY of the agent's AC wire model; no LLM call."""
import json
import math
from pathlib import Path
import sys
from femm_agent import estimate
from femm_agent.server import build_server
from femm_agent.sessions import SessionManager
from femm_agent.femfile import parse_file

source=Path(sys.argv[1]);output=Path(sys.argv[2]);assert not output.exists()
report=json.loads(source.read_text())
trial=next(case for case in report['cases'] if case['id']=='ac_wire')['prompts'][0]
state=next(state for state in trial['native_sessions']['result'] if state.get('document'))
original=Path(state['document'])
work=output.parent/'controlled-ac-boundary'
work.mkdir(exist_ok=True)
manager=SessionManager(default_work_dir=work/'sessions')
_,context=build_server(manager)
tools=context.extras['tools'];calls=[]
def call(tool_name,**arguments):
    response=tools[tool_name](**arguments);calls.append({'tool':tool_name,'arguments':arguments,'response':response})
    assert response['ok'],response.get('error')
    return response['result']
try:
    parsed=parse_file(original)
    name=next(boundary['BdryName'] for boundary in parsed.boundaries() if int(boundary['BdryType'])==0)
    call('femm_open',path=str(original))
    corrected=work/'full-boundary.fem'
    call('femm_save',path=str(corrected))
    points=[[50*math.cos(math.pi/4+index*math.pi/2),50*math.sin(math.pi/4+index*math.pi/2)] for index in range(4)]
    call('femm_set_arc_prop',points=points,boundary=name,max_seg_deg=1)
    call('femm_save')
    sweep=call('femm_experiment_ac_sweep',frequencies=[10,1000,5000],circuits=['wire'],current=1)
    comparison=[]
    for row in sweep['rows']:
        frequency=row[0]
        impedance=estimate.round_wire_internal_impedance(.005,58e6,frequency)+2j*math.pi*frequency*(4e-7*math.pi)/(2*math.pi)*math.log(10)
        comparison.append({'frequency_Hz':frequency,'native_R_ohm':row[4],'native_L_H':row[5],
                           'reference_R_ohm':impedance.real,'reference_L_H':impedance.imag/(2*math.pi*frequency),
                           'R_relative_error':abs(row[4]-impedance.real)/abs(impedance.real),
                           'L_relative_error':abs(row[5]-impedance.imag/(2*math.pi*frequency))/abs(impedance.imag/(2*math.pi*frequency))})
    result={'source':'controlled native rerun after assigning A=0 to all four outer arcs on a copy',
            'original_untouched':str(original),'corrected_model':str(corrected),'calls':calls,
            'comparison':comparison,'scope':'This corrects an agent boundary-selection error; it is not an original prompt pass.'}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(comparison))
finally:
    manager.close_all()
