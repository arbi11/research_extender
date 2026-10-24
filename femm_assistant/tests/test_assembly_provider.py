"""No paid calls: staged strict contracts, typed graph recovery and source checks."""
import copy
import json
import re
import unittest
from unittest.mock import patch

import httpx

from app.assembly import MachineAssembly
from app.assembly_provider import (AssemblyProviderError, MachineInventory,
    discover_assembly, detail_assembly, plan_assembly)
from app import assembly_provider, structured
from tests.test_assembly import coil_assembly, custom_assembly


PROMPT = ('Create the finite annular coil. Inner radius 10 mm, outer radius 20 mm, '
          'axial bounds -20 mm and 20 mm, relative permeability 1, 100 turns, current 1 A.')


def inventory(assembly=None, *, scopes=None):
    assembly = assembly or coil_assembly()
    scopes = scopes or {}
    return MachineInventory.model_validate({
        'id':assembly.id, 'title':assembly.title, 'input_mode':assembly.input_mode,
        'family':assembly.family, 'physics':assembly.physics, 'required_physics':assembly.required_physics,
        'explanation':'Discover intent before filling facts and geometry.',
        'components':[{'id':c.id, 'name':c.name, 'role':c.role, 'family':c.family,
            'mode':c.mode, 'parent_id':c.parent_id, 'template_kind':c.template.kind if c.template else None,
            'scope':scopes.get(c.id,'region'),
            'source_refs':c.source_refs, 'required_physics':c.required_physics, 'intent':c.description or c.name}
            for c in assembly.components]})


def inventory_wire(assembly=None):
    return inventory(assembly).model_dump(mode='json',exclude={'identity_map'})


def selector_binding(binding, shape):
    path=binding['path'];index=None
    targets={'/label/x':'label_x','/label/y':'label_y','/mesh_mm':'mesh_mm',
        '/shape/center/x':'center_x','/shape/center/y':'center_y',
        '/shape/radius':'radius','/shape/start/x':'start_x','/shape/start/y':'start_y',
        '/shape/tolerance_mm':'tolerance_mm',**{'/shape/'+key:key for key in ('x_min','x_max','y_min','y_max')}}
    if path in targets:
        target=targets[path]
    elif match:=re.fullmatch(r'/shape/points/(\d+)/([xy])',path):
        index=int(match[1]);target='point_'+match[2]
    elif match:=re.fullmatch(r'/shape/edges/(\d+)/end/([xy])',path):
        index=int(match[1]);target='edge_end_'+match[2]
    elif match:=re.fullmatch(r'/shape/edges/(\d+)/sweep_deg',path):
        index=int(match[1]);target='edge_sweep_deg'
    else:
        raise ValueError('Test fixture binding has no typed selector: '+path)
    result={'fact_id':binding['fact_id'],'target':target}
    if shape and shape['type'] in {'polygon','closed_path'}: result['index']=index
    return result


def staged_replies(assembly=None, *, include_discovery=False, inventory_model=None):
    """Mock wire transcript in public stage order; also used by runner contract tests."""
    assembly = assembly or coil_assembly()
    inv=inventory_model or inventory(assembly)
    scopes={c.id:c.scope for c in inv.components}
    replies = [inv.model_dump(mode='json',exclude={'identity_map'})] if include_discovery else []
    facts = []
    for f in assembly.facts:
        raw=f.model_dump(mode='json');value=raw.pop('value');calculation=raw.pop('calculation')
        raw.update(value_number=value if isinstance(value,(int,float)) else None,
            value_text=value if isinstance(value,str) else None,
            calculation_op=calculation['op'] if calculation else None,
            calculation_inputs=calculation['inputs'] if calculation else [],
            calculation_factor=calculation['factor'] if calculation else None)
        facts.append(raw)
    replies.append({'facts':facts, 'materials':[x.model_dump(mode='json') for x in assembly.materials],
        'circuits':[x.model_dump(mode='json') for x in assembly.circuits],
        'required_physics':assembly.required_physics, 'depth_ref':assembly.depth_ref,
        'domain_kind':assembly.domain.type if assembly.domain else 'unresolved',
        'air_label':assembly.air_label.model_dump() if assembly.air_label else None,
        'mesh_mm':assembly.mesh_mm, 'probes':[p.model_dump() for p in assembly.probes],
        'force_feature_id':assembly.force_feature_id, 'numerical_notes':assembly.numerical_notes,
        'questions':assembly.questions, 'explanation':'Source facts and disclosed numerical settings.'})
    if assembly.domain:
        replies.append(assembly.domain.model_dump(mode='json'))
    for c in assembly.components:
        if scopes[c.id] in {'background','measurement'}: continue
        intents=[];details=[]
        for op,identity,f in [('feature',f.id,f) for f in c.features]+[(o.op,o.feature_id,o.feature) for o in c.overrides]:
            intents.append({'id':identity,'name':f.name if f else identity,'role':f.role if f else 'removed_feature',
                'operation':op, 'shape_kind':f.shape.type if f and f.shape else 'unresolved'})
            if f is not None:
                raw=f.model_dump(mode='json')
                for key in ('id','name','role'): raw.pop(key)
                raw['bindings']=[selector_binding(binding,raw['shape']) for binding in raw['bindings']]
                if raw['shape'] and raw['shape']['type']=='closed_path':
                    for edge in raw['shape']['edges']:
                        edge.setdefault('sweep_deg',None)
                details.append(raw)
        replies.append({'depends_on':c.depends_on,'transform':c.transform.model_dump(mode='json'),
            'template':c.template.model_dump(mode='json') if c.template else None,
            'feature_intents':intents,'source_refs':c.source_refs,'required_physics':c.required_physics,
            'description':c.description,'questions':[], 'explanation':'Fill this component checkpoint.'})
        replies.extend(details)
    return replies


class AssemblyProviderTests(unittest.IsolatedAsyncioTestCase):
    def test_fact_units_are_an_explicit_provider_vocabulary_and_keep_roles(self):
        from pydantic import ValidationError
        from app.assembly import capability_manifest
        schema = structured.strict_json_schema(assembly_provider.FactDraft)
        variants = schema["properties"]["unit"]["anyOf"]
        units = next(item["enum"] for item in variants if "enum" in item)
        self.assertIn("1", units)
        self.assertIn("turns", units)
        self.assertNotIn("dimensionless", units)
        self.assertEqual(capability_manifest()["fact_units_by_role"]["permeability"], ["1"])
        common = dict(id="test", label="test", role="turns", value_number=3, value_text=None,
            unit="turns", status="user", source_refs=["nl_request"], depends_on=[], calculation_op=None,
            calculation_inputs=[], calculation_factor=None, source_parameter_id=None, explanation="User input")
        self.assertEqual(assembly_provider.FactDraft.model_validate(common).to_fact().value, 3)
        with self.assertRaises(ValidationError):
            assembly_provider.FactDraft.model_validate({**common, "unit": "1"})
        with self.assertRaises(ValidationError):
            assembly_provider.FactDraft.model_validate({**common, "unit": "dimensionless"})
        self.assertEqual(assembly_provider.FactDraft.model_validate({**common, "role": "permeability", "unit": "1"}).unit, "1")


    def fake(self, replies, *, diagnostic_reply=None):
        original = httpx.AsyncClient
        self.requests, self.options, self.decision_requests = [], [], []
        async def handle(request):
            if request.url.path.endswith('/decisions'):
                self.decision_requests.append(json.loads(request.content))
                return diagnostic_reply or httpx.Response(200,json={
                    'id':'diagnostic-test','model':'typesafe/jev-1.14',
                    'answers':{'arguments_match':{'type':'noul','noul':.3}},
                    'usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12,'cost':.0001}})
            self.requests.append(json.loads(request.content))
            reply = replies[len(self.requests)-1]
            if isinstance(reply, httpx.Response):
                return reply
            return httpx.Response(200,json={'id':'generation-test', 'provider':'test-provider', 'model':'test-model',
                'choices':[{'finish_reason':'stop', 'message':{'content':json.dumps(reply)}}],
                'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120,'cost':.001,
                    'completion_tokens_details':{'reasoning_tokens':7}}})
        def client(*args, **kwargs):
            self.options.append(kwargs)
            return original(*args,transport=httpx.MockTransport(handle),**kwargs)
        return patch.object(assembly_provider.httpx, 'AsyncClient', side_effect=client)

    def payload(self,index):
        return json.loads(self.requests[index]['messages'][1]['content'])

    async def test_small_strict_stages_and_no_artificial_output_or_read_limits(self):
        with self.fake(staged_replies(include_discovery=True)):
            result = await plan_assembly(PROMPT,'test-key','https://provider.test/v1','test-model')
        self.assertEqual(result.provider_calls,4)
        self.assertTrue(result.compilation.execution_supported,result.compilation.issues)
        self.assertEqual(result.assembly.model_dump(),coil_assembly().model_dump())
        self.assertEqual(len(result.usage),4)
        names=[]
        for request,options in zip(self.requests,self.options):
            response=request['response_format']
            self.assertEqual(response['type'],'json_schema')
            self.assertTrue(response['json_schema']['strict'])
            self.assertTrue(request['provider']['require_parameters'])
            names.append(response['json_schema']['name'])
            for key in ('max_tokens','max_completion_tokens','temperature'): self.assertNotIn(key,request)
            self.assertIsNone(options['timeout'].read)
            self.assertNotIn('MachineAssembly',json.dumps(response))
            self.assertNotIn('GeometryFeature',json.dumps(response))
        self.assertEqual(names,['MachineInventory','AssemblySettingsDraft','Rectangle','ComponentDraft'])
        self.assertEqual(result.provider_metadata[0]['id'],'generation-test')
        self.assertEqual(result.usage[0]['completion_tokens_details']['reasoning_tokens'],7)
        self.assertIn('schema_sha256',result.provider_metadata[0]['contract'])
        self.assertIn('capabilities',self.payload(0))
        self.assertEqual(self.payload(1)['inventory'],inventory().model_dump(mode='json'))

    async def test_persisted_paper_inventory_skips_discovery_only(self):
        with self.fake(staged_replies()):
            result=await plan_assembly(PROMPT,'key','https://provider.test','model',
                evidence_context={'assembly_inventory':inventory().model_dump(mode='json')})
        self.assertEqual(result.provider_calls,3)
        self.assertEqual(len(self.requests),3)

    async def test_discovery_has_open_families_and_retains_unknown_components(self):
        raw=inventory_wire();raw['family']='previously undescribed machine'
        raw['components'][0].update(mode='custom',template_kind=None,family='unknown bespoke winding')
        with self.fake([raw]):
            result=await discover_assembly(PROMPT,'key','https://provider.test','model')
        self.assertEqual(result.inventory.family,'previously undescribed machine')
        self.assertEqual(result.inventory.components[0].mode,'custom')

    async def test_actual_wire_error_gets_one_strict_same_stage_repair(self):
        correct=inventory_wire();bad=copy.deepcopy(correct);del bad['questions']
        with self.fake([bad,correct]):
            result=await discover_assembly(PROMPT,'key','https://provider.test','model')
        self.assertEqual(len(self.requests),2)
        self.assertEqual(result.provider_metadata['provider_calls'],3)
        self.assertEqual(result.usage['total_tokens'],252)
        recovery=self.payload(1)['validation_recovery']
        self.assertTrue(any(x['path']=='/questions' for x in recovery['issues']))
        self.assertEqual(recovery['previous_candidate'],bad)
        self.assertEqual(self.requests[0]['response_format'],self.requests[1]['response_format'])
        self.assertEqual(result.provider_metadata['attempts'][0]['outcome'],'validation_failed')

    async def test_second_invalid_response_stops_without_ordinary_json_fallback(self):
        raw=inventory_wire();raw['unrecognized']='value'
        with self.fake([raw,raw]):
            with self.assertRaises(AssemblyProviderError) as caught:
                await discover_assembly(PROMPT,'key','https://provider.test','model')
        self.assertEqual(caught.exception.provider_calls,3)
        self.assertEqual(len(caught.exception.usage),3)
        self.assertTrue(all(r['response_format']['type']=='json_schema' for r in self.requests))

    async def test_source_fact_repair_preserves_inventory_and_settings_checkpoint(self):
        replies=staged_replies(include_discovery=True)
        good=copy.deepcopy(replies[1]);fact=good['facts'][0]
        fact.update(status='paper',source_parameter_id='inner',source_refs=['paper_span_1'])
        bad=copy.deepcopy(good);bad['facts'][0]['value_number']=1.1
        replies[1:2]=[bad,good]
        with self.fake(replies):
            result=await plan_assembly(PROMPT,'key','https://provider.test','model',
                evidence_context={'parameters':[{'id':'inner','value':10.,'unit':'mm'}]})
        self.assertEqual(result.provider_calls,6)
        self.assertEqual([r['response_format']['json_schema']['name'] for r in self.requests],
            ['MachineInventory','AssemblySettingsDraft','AssemblySettingsDraft','Rectangle','ComponentDraft'])
        self.assertEqual(result.assembly.facts[0].value,1.)
        self.assertEqual(len(result.usage),6)
        self.assertEqual(result.provider_metadata[1]['outcome'],'validation_failed')
        self.assertIn('changed its verified paper',self.payload(2)['validation_recovery']['issues'][0]['message'])

    async def test_unknown_paper_parameter_remains_strict(self):
        bad=staged_replies()[0]
        bad['facts'][0].update(status='paper',source_parameter_id='fabricated')
        with self.fake([bad,bad]):
            with self.assertRaises(AssemblyProviderError) as caught:
                await detail_assembly('Create an unspecified coil.','key','https://provider.test','model',inventory=inventory())
        self.assertEqual(caught.exception.provider_calls,3)
        self.assertIn('existing verified source',caught.exception.message)

    async def test_nl_fact_cannot_claim_a_source_parameter_id(self):
        bad=staged_replies()[0];bad['facts'][0]['source_parameter_id']='pretend_paper'
        with self.fake([bad,bad]):
            with self.assertRaises(AssemblyProviderError):
                await detail_assembly(PROMPT,'key','https://provider.test','model',inventory=inventory())
        self.assertTrue(any('/facts/0' in issue['path'] for issue in self.payload(1)['validation_recovery']['issues']))

    async def test_unsupported_physics_graph_remains_available(self):
        raw=coil_assembly().model_dump();raw['required_physics']=['permanent_magnets']
        assembly=MachineAssembly.model_validate(raw)
        with self.fake(staged_replies(assembly)):
            result=await plan_assembly(PROMPT,'key','https://provider.test','model',inventory=inventory(assembly))
        self.assertIsNotNone(result.assembly)
        self.assertFalse(result.compilation.execution_supported)
        self.assertEqual(len(result.compilation.expanded_features),1)
        self.assertTrue(any(issue.code=='unsupported_physics' for issue in result.compilation.issues))

    async def test_provider_grammar_rejection_has_no_geometry_retry_and_preserves_cost(self):
        replies=staged_replies();replies[1]=httpx.Response(400,json={'error':'secret-key','usage':{'cost':.002}})
        with self.fake(replies):
            with self.assertRaises(AssemblyProviderError) as caught:
                await detail_assembly(PROMPT,'secret-key','https://provider.test','model',inventory=inventory())
        self.assertEqual(len(self.requests),2)
        self.assertEqual(caught.exception.provider_calls,2)
        self.assertEqual(caught.exception.usage[0]['cost'],.001)
        self.assertEqual(caught.exception.usage[1]['cost'],.002)
        self.assertEqual(caught.exception.attempts[-1]['outcome'],'provider_rejected')
        self.assertEqual(self.decision_requests,[])
        self.assertNotIn('secret-key',caught.exception.message)

    async def test_stage_specific_custom_geometry_preserves_patterns_transforms_and_arcs(self):
        raw=custom_assembly().model_dump(mode='json');base=raw['components'][0]['features'][0]
        base['transform'].update(mirror='y',dx_mm=1,rotate_deg=15)
        for identity,shape,label in [
            ('disc',{'type':'circle','center':{'x':-5.,'y':-5.},'radius':1.},{'x':-5.,'y':-5.}),
            ('triangle',{'type':'polygon','points':[{'x':-9.,'y':0.},{'x':-7.,'y':0.},{'x':-8.,'y':2.}]},{'x':-8.,'y':.5}),
            ('curved',{'type':'closed_path','start':{'x':1.,'y':0.},'edges':[
                {'kind':'arc','end':{'x':-1.,'y':0.},'sweep_deg':180.},
                {'kind':'arc','end':{'x':1.,'y':0.},'sweep_deg':180.}],'tolerance_mm':.01},{'x':0.,'y':0.})]:
            f=copy.deepcopy(base);f.update(id=identity,shape=shape,label=label,pattern=None)
            f['transform'].update(mirror='none',dx_mm=0,rotate_deg=0)
            raw['components'][0]['features'].append(f)
        assembly=MachineAssembly.model_validate(raw)
        with self.fake(staged_replies(assembly)):
            result=await plan_assembly('Create a novel machine of depth 5 mm.','key','https://provider.test','model',inventory=inventory(assembly))
        self.assertEqual(result.assembly.model_dump(),assembly.model_dump())
        names=[r['response_format']['json_schema']['name'] for r in self.requests]
        self.assertEqual(names,['AssemblySettingsDraft','Circle','ComponentDraft','RectangleFeatureDraft',
            'CircleFeatureDraft','PolygonFeatureDraft','ClosedPathFeatureDraft'])
        self.assertEqual(result.assembly.components[0].features[0].pattern.count,4)
        self.assertEqual(result.assembly.components[0].features[-1].shape.edges[0].sweep_deg,180.)

    async def test_feature_validation_retries_only_failed_shape(self):
        assembly=custom_assembly();replies=staged_replies(assembly)
        bad=copy.deepcopy(replies[-1]);bad['shape']['x_max']=7.
        replies[-1:]=[bad,replies[-1]]
        with self.fake(replies):
            result=await plan_assembly('Create a machine of depth 5 mm.','key','https://provider.test','model',inventory=inventory(assembly))
        self.assertEqual(result.provider_calls,6)
        self.assertEqual([m['stage'] for m in result.provider_metadata],
            ['facts_and_settings','exterior_geometry','component:rotor','feature:rotor.slots',
             'diagnostic:detail_feature','feature:rotor.slots'])
        self.assertEqual(self.payload(3)['component_checkpoint'],self.payload(4)['component_checkpoint'])
        self.assertIn('validation_recovery',self.payload(4))

    async def test_inconsistent_selected_case_stops_before_provider(self):
        raw=inventory_wire();raw.update(cases=[{'id':'first','label':'first'}],selected_case_id='first')
        with self.fake([]):
            with self.assertRaisesRegex(AssemblyProviderError,'operating case'):
                await detail_assembly(PROMPT,'key','https://provider.test','model',inventory=raw,evidence_context={'case_id':'another'})
        self.assertEqual(self.requests,[])

    async def test_nl_paraphrase_and_number_words_are_advisory_only(self):
        raw=custom_assembly().model_dump(mode='json')
        raw['facts'] += [
            {'id':'count','label':'Repetition count','role':'count','value':4.,'unit':'1',
             'status':'user','source_refs':['nl_request']},
            {'id':'outer_bc','label':'Outer boundary annotation','role':'other',
             'value':'homogeneous Dirichlet vector potential','unit':'text',
             'status':'user','source_refs':['nl_request']}]
        raw['components'][0]['features'][0]['pattern']['fact_ids']=['count']
        assembly=MachineAssembly.model_validate(raw)
        prompt='Create four repeated air regions, depth five millimeters. Ground the outer boundary with A=0.'
        with self.fake(staged_replies(assembly)):
            result=await plan_assembly(prompt,'key','https://provider.test','model',inventory=inventory(assembly))
        self.assertTrue(result.compilation.execution_supported,result.compilation.issues)
        self.assertEqual(len(result.compilation.expanded_features),4)
        presence=result.provider_metadata[0]['validation_advisories']['nl_source_presence']
        self.assertEqual(presence['policy'],'advisory_only')
        self.assertEqual(presence['raw_prompt'],prompt)
        self.assertEqual({n['fact_id'] for n in presence['notes']},{'depth','count','outer_bc'})
        self.assertEqual(self.decision_requests,[])

    async def test_text_annotation_cannot_supply_numeric_current(self):
        raw=coil_assembly().model_dump(mode='json')
        raw['facts'][5].update(value='one ampere',unit='text')
        assembly=MachineAssembly.model_validate(raw)
        with self.fake(staged_replies(assembly)):
            result=await plan_assembly('Use one ampere.','key','https://provider.test','model',inventory=inventory(assembly))
        self.assertIsNone(result.compilation.plan)
        self.assertTrue(any(i.code=='circuit_unavailable' for i in result.compilation.issues))
        self.assertEqual(result.assembly.facts[5].value,'one ampere')

    async def test_missing_permeability_cannot_become_air_after_advisory_change(self):
        raw=coil_assembly().model_dump(mode='json')
        raw['facts'][4].update(value=None,unit=None,status='missing',source_refs=[])
        assembly=MachineAssembly.model_validate(raw)
        with self.fake(staged_replies(assembly)):
            result=await plan_assembly('The core permeability is unknown; do not assume it.',
                'key','https://provider.test','model',inventory=inventory(assembly))
        self.assertIsNone(result.assembly.facts[4].value)
        self.assertEqual(result.assembly.facts[4].status,'missing')
        self.assertIsNone(result.compilation.plan)
        self.assertTrue(any(i.code=='material_unavailable' for i in result.compilation.issues))

    async def test_background_and_measurement_route_to_settings_without_geometry_calls(self):
        raw=coil_assembly().model_dump(mode='json')
        for identity in ('exterior','sampling'):
            raw['components'].append({'id':identity,'name':identity,'role':'auxiliary','family':'unknown',
                'mode':'custom','source_refs':['nl_request'],'description':'Intent retained in settings.'})
        assembly=MachineAssembly.model_validate(raw)
        inv=inventory(assembly,scopes={'exterior':'background','sampling':'measurement'})
        with self.fake(staged_replies(assembly,inventory_model=inv)):
            result=await plan_assembly(PROMPT,'key','https://provider.test','model',inventory=inv)
        self.assertEqual(len(self.requests),3)
        self.assertEqual(len(result.assembly.components),3)
        self.assertEqual(len(result.compilation.plan.regions),1)
        self.assertEqual(result.assembly.domain,assembly.domain)
        self.assertEqual(result.assembly.probes,assembly.probes)
        route={c['component_id']:c for c in result.provider_metadata[0]['component_routing']}
        self.assertEqual(route['exterior']['settings_refs'],['/domain','/air_label'])
        self.assertEqual(route['sampling']['settings_refs'],['/probes','/force_feature_id'])
        self.assertFalse(route['exterior']['component_provider_call'])
        self.assertEqual(result.assembly.components[1].features,[])

    async def test_interior_air_cavity_remains_a_region_regardless_of_name(self):
        raw=custom_assembly().model_dump(mode='json');raw['components'][0]['name']='Air background cavity'
        assembly=MachineAssembly.model_validate(raw);inv=inventory(assembly)
        with self.fake(staged_replies(assembly,inventory_model=inv)):
            result=await plan_assembly('Use four interior air cavities and depth five millimeters.',
                'key','https://provider.test','model',inventory=inv)
        self.assertEqual(len(result.compilation.plan.regions),4)
        self.assertEqual(self.requests[-1]['response_format']['json_schema']['name'],'RectangleFeatureDraft')

    async def test_group_header_retains_transforms_and_has_no_owned_regions(self):
        raw=custom_assembly().model_dump(mode='json')
        raw['components'][0]['parent_id']='frame'
        raw['components'].insert(0,{'id':'frame','name':'Frame','role':'group','family':'novel',
            'mode':'custom','transform':{'dx_mm':2.,'rotate_deg':15.},'source_refs':['nl_request']})
        assembly=MachineAssembly.model_validate(raw);inv=inventory(assembly,scopes={'frame':'group'})
        with self.fake(staged_replies(assembly,inventory_model=inv)):
            result=await plan_assembly('Create a grouped machine with depth 5 mm.','key','https://provider.test','model',inventory=inv)
        self.assertEqual(result.assembly.components[0].transform,assembly.components[0].transform)
        self.assertEqual(result.assembly.components[0].features,[])
        self.assertEqual(len(result.compilation.expanded_features),4)
        self.assertTrue(all(f.parent_component_ids==['frame'] for f in result.compilation.expanded_features))

    async def test_non_region_header_cannot_silently_emit_or_drop_material_features(self):
        raw=custom_assembly().model_dump(mode='json');raw['components'][0]['features']=[]
        assembly=MachineAssembly.model_validate(raw);inv=inventory(assembly,scopes={'rotor':'group'})
        replies=staged_replies(assembly,inventory_model=inv)
        bad=copy.deepcopy(replies[-1]);bad['feature_intents']=[{
            'id':'unexpected','name':'Unexpected material face','role':'solid','operation':'feature','shape_kind':'circle'}]
        replies[-1:]=[bad,replies[-1]]
        with self.fake(replies):
            result=await plan_assembly('Depth 5 mm.','key','https://provider.test','model',inventory=inv)
        self.assertEqual(result.assembly.components[0].features,[])
        self.assertEqual(self.payload(3)['validation_recovery']['issues'][0]['code'],'non_region_scope')
        self.assertEqual(len(self.decision_requests),1)

    async def test_binding_value_failure_repairs_feature_before_compilation(self):
        raw=custom_assembly().model_dump(mode='json')
        raw['facts'].append({'id':'left','label':'Left bound','role':'length','value':8.,'unit':'mm',
                            'status':'user','source_refs':['nl_request']})
        raw['components'][0]['features'][0]['bindings']=[{'path':'/shape/x_min','fact_id':'left','unit':'mm'}]
        assembly=MachineAssembly.model_validate(raw);replies=staged_replies(assembly)
        bad=copy.deepcopy(replies[-1]);bad['bindings'][0]['target']='x_max'
        replies[-1:]=[bad,replies[-1]]
        with self.fake(replies):
            result=await plan_assembly('Depth 5 mm; left bound 8 mm.','key','https://provider.test','model',inventory=inventory(assembly))
        self.assertTrue(result.compilation.execution_supported,result.compilation.issues)
        error=self.payload(4)['validation_recovery']['issues'][0]
        self.assertEqual(error['code'],'invalid_binding')
        self.assertEqual(error['path'],'/bindings/0')
        self.assertEqual(error['coordinate_path'],'/shape/x_max')
        self.assertEqual(result.assembly.components[0].features[0].bindings[0].path,'/shape/x_min')

    async def test_free_binding_path_is_rejected_by_shape_specific_wire_contract(self):
        raw=custom_assembly().model_dump(mode='json')
        raw['components'][0]['features'][0]['bindings']=[{'path':'/shape/y_max','fact_id':'depth','unit':'mm'}]
        # Use an explicit coordinate fact for an exact fixture binding.
        raw['facts'].append({'id':'top','label':'Top bound','role':'length','value':.5,'unit':'mm',
                            'status':'user','source_refs':['nl_request']})
        raw['components'][0]['features'][0]['bindings'][0]['fact_id']='top'
        assembly=MachineAssembly.model_validate(raw);replies=staged_replies(assembly)
        bad=copy.deepcopy(replies[-1]);bad['bindings']=[{'path':'/shape/y_max','fact_id':'top','unit':'mm'}]
        replies[-1:]=[bad,replies[-1]]
        with self.fake(replies):
            result=await plan_assembly('Depth five millimeters, top bound half a millimeter.',
                'key','https://provider.test','model',inventory=inventory(assembly))
        self.assertTrue(result.compilation.execution_supported,result.compilation.issues)
        self.assertTrue(any(e['code']=='wire_additional_property' for e in self.payload(4)['validation_recovery']['issues']))
        schema=self.requests[-1]['response_format']['json_schema']['schema']
        self.assertNotIn('path',schema['$defs']['RectangleBinding']['properties'])

    async def test_polygon_binding_index_is_checked_before_feature_acceptance(self):
        raw=custom_assembly().model_dump(mode='json');feature=raw['components'][0]['features'][0]
        feature.update(shape={'type':'polygon','points':[{'x':8.,'y':0.},{'x':10.,'y':0.},{'x':9.,'y':1.}]},
            label={'x':9.,'y':.25},pattern=None,
            bindings=[{'path':'/shape/points/0/x','fact_id':'left','unit':'mm'}])
        raw['facts'].append({'id':'left','label':'Left point','role':'length','value':8.,'unit':'mm',
                            'status':'user','source_refs':['nl_request']})
        assembly=MachineAssembly.model_validate(raw);replies=staged_replies(assembly)
        bad=copy.deepcopy(replies[-1]);bad['bindings'][0]['index']=99
        replies[-1:]=[bad,replies[-1]]
        with self.fake(replies):
            result=await plan_assembly('Depth 5 mm and left point 8 mm.','key','https://provider.test','model',inventory=inventory(assembly))
        self.assertTrue(result.compilation.execution_supported,result.compilation.issues)
        self.assertEqual(self.payload(4)['validation_recovery']['issues'][0]['code'],'invalid_binding')

    async def test_template_modified_preserves_explicit_removal_and_addition(self):
        raw=coil_assembly().model_dump(mode='json');component=raw['components'][0]
        component['mode']='template_modified'
        component['overrides']=[{'op':'remove','feature_id':'body','feature':None},
            {'op':'add','feature_id':'replacement','feature':{'id':'replacement','name':'Replacement winding',
             'role':'winding','shape':{'type':'rectangle','x_min':10.,'x_max':20.,'y_min':-20.,'y_max':20.},
             'material_id':'winding','circuit_id':'drive','turns_ref':'turns','label':{'x':15.,'y':0.},
             'bindings':[{'path':'/shape/x_min','fact_id':'ri','unit':'mm'}]}}]
        assembly=MachineAssembly.model_validate(raw)
        with self.fake(staged_replies(assembly)):
            result=await plan_assembly(PROMPT,'key','https://provider.test','model',inventory=inventory(assembly))
        self.assertEqual(result.assembly.model_dump(),assembly.model_dump())
        self.assertTrue(result.compilation.execution_supported,result.compilation.issues)
        self.assertEqual([o.op for o in result.assembly.components[0].overrides],['remove','add'])
        self.assertEqual(len(self.requests),4) # A removal has no geometry call.

    async def test_unresolved_geometry_keeps_novel_component_and_questions(self):
        raw=custom_assembly().model_dump(mode='json')
        raw['components'][0]['features'][0].update(shape=None,pattern=None,
            description='Custom contour is not yet specified.')
        raw['questions']=['What is the custom contour?']
        assembly=MachineAssembly.model_validate(raw)
        with self.fake(staged_replies(assembly)):
            result=await plan_assembly('Create a machine of depth 5 mm.','key','https://provider.test','model',inventory=inventory(assembly))
        self.assertEqual(result.assembly.family,assembly.family)
        self.assertIsNone(result.assembly.components[0].features[0].shape)
        self.assertIn('What is the custom contour?',result.questions)
        self.assertIsNone(result.compilation.plan)
        self.assertEqual(self.requests[-1]['response_format']['json_schema']['name'],'UnresolvedFeatureDraft')

    async def test_aggregate_component_explanations_are_not_truncated(self):
        raw=coil_assembly().model_dump(mode='json')
        raw['components']=[{**raw['components'][0],'id':'coil'+str(index)} for index in range(3)]
        assembly=MachineAssembly.model_validate(raw);replies=staged_replies(assembly)
        replies[0]['explanation']='Settings explanation. '*80
        for reply in replies[2:]: reply['explanation']='Component explanation. '*80
        with self.fake(replies):
            result,_,_=await detail_assembly(PROMPT,'key','https://provider.test','model',inventory=inventory(assembly))
        self.assertGreater(len(result.explanation),5000)
        self.assertEqual(len(result.assembly.components),3)

    async def test_diagnostic_failure_does_not_block_validation_repair(self):
        correct=inventory_wire();bad=copy.deepcopy(correct);del bad['questions']
        with self.fake([bad,correct],diagnostic_reply=httpx.Response(503,json={'error':'private-provider-details'})):
            result=await discover_assembly(PROMPT,'key','https://provider.test','model',
                decision_model='custom/decision',decision_base_url='https://decision.test/v1')
        self.assertEqual(result.provider_metadata['provider_calls'],3)
        diagnostic=result.provider_metadata['attempts'][1]
        self.assertEqual(diagnostic['outcome'],'diagnostic_unavailable')
        self.assertEqual(diagnostic['requested_model'],'custom/decision')
        self.assertNotIn('private-provider-details',json.dumps(diagnostic))
        self.assertEqual(self.decision_requests[0]['state']['function']['name'],'discover_assembly')
        self.assertEqual(self.decision_requests[0]['state']['arguments']['candidate'],bad)

    async def test_jev_match_cannot_approve_a_second_invalid_response(self):
        bad=inventory_wire();del bad['questions']
        match=httpx.Response(200,json={'answers':{'arguments_match':{'type':'noul','noul':.99}},
            'usage':{'cost':.0002}})
        with self.fake([bad,bad],diagnostic_reply=match):
            with self.assertRaises(AssemblyProviderError) as caught:
                await discover_assembly(PROMPT,'key','https://provider.test','model')
        self.assertEqual(caught.exception.attempts[1]['arguments_match'],True)
        self.assertEqual(caught.exception.attempts[-1]['outcome'],'validation_failed')
        self.assertEqual(caught.exception.usage[1]['cost'],.0002)
        self.assertIn('/questions',caught.exception.message)

    async def test_derived_arithmetic_repair_uses_actual_fact_validator(self):
        raw=coil_assembly().model_dump(mode='json');current=raw['facts'][5]
        raw['facts'].append({**current,'id':'source_current','value':2.})
        current.update(status='derived',source_refs=[],explanation='Half the supplied current.',
            calculation={'op':'scale','inputs':['source_current'],'factor':.5})
        assembly=MachineAssembly.model_validate(raw);replies=staged_replies(assembly)
        bad=copy.deepcopy(replies[0]);bad['facts'][5]['value_number']=2.
        replies[0:1]=[bad,replies[0]]
        with self.fake(replies):
            result=await plan_assembly(PROMPT+' Source current is 2 A.','key','https://provider.test','model',inventory=inventory(assembly))
        self.assertEqual(result.assembly.facts[5].value,1.)
        errors=self.payload(1)['validation_recovery']['issues']
        self.assertTrue(any(x['code']=='invalid_fact' and x['path']=='/facts/amps' for x in errors))
        self.assertEqual(len(self.decision_requests),1)

    async def test_component_dependency_cycle_repairs_only_current_component(self):
        raw=custom_assembly().model_dump(mode='json')
        raw['components']=[{**raw['components'][0],'id':'first','features':[],'depends_on':['second']},
            {**raw['components'][0],'id':'second','features':[]}]
        assembly=MachineAssembly.model_validate(raw)
        inv=inventory(assembly,scopes={'first':'group','second':'group'})
        replies=staged_replies(assembly,inventory_model=inv)
        bad=copy.deepcopy(replies[-1]);bad['depends_on']=['first']
        replies[-1:]=[bad,replies[-1]]
        with self.fake(replies):
            result=await plan_assembly('Create a machine of depth 5 mm.','key','https://provider.test','model',inventory=inv)
        self.assertEqual(result.assembly.model_dump(),assembly.model_dump())
        self.assertEqual([r['response_format']['json_schema']['name'] for r in self.requests],
            ['AssemblySettingsDraft','Circle','ComponentDraft','ComponentDraft','ComponentDraft'])
        self.assertEqual(self.payload(3)['completed_components'],self.payload(4)['completed_components'])
        self.assertEqual(self.payload(4)['validation_recovery']['issues'][0]['code'],'dependency_cycle')

    def test_provider_errors_are_actionable_without_raw_bodies(self):
        for status,word in [(401,'key'),(403,'access'),(402,'spending limit'),(429,'rate'),
                            (400,'strict'),(422,'strict'),(404,'identifier')]:
            message=assembly_provider._provider_failure(status,{'error':{'message':'private-body'}})
            self.assertIn(word,message)
            self.assertNotIn('private-body',message)
        self.assertIn('grammar',assembly_provider._provider_failure(400,{'error':{'message':'compiled grammar too large'}}))

    async def test_opaque_inventory_ids_and_references_are_normalized_without_source_changes(self):
        raw=inventory_wire()
        machine='An experimental machine identity longer than thirty two characters'
        parent='Assembly parent / frame'
        child='Winding component (left side) / novel'
        operating='Case 1: fully specified steady operating point'
        raw['id']=machine
        raw['components'][0].update(id=child,parent_id=parent,source_refs=[child,'immutable_span:page_1'])
        group=assembly_provider.InventoryComponent(id=parent,name='Parent frame',role='group',family='novel',
            scope='group',mode='custom',intent='Retain the parent hierarchy.').model_dump(mode='json')
        raw['components'].insert(0,group)
        raw['cases']=[assembly_provider.InventoryCase(id=operating,label='Operating case',
            source_refs=['immutable_span:page_1']).model_dump(mode='json')]
        raw['selected_case_id']=operating
        original=copy.deepcopy(raw)
        with self.fake([raw]):
            result=await discover_assembly(PROMPT,'key','https://provider.test','model',
                evidence_context={'case_id':operating})
        mapping={(x['kind'],x['raw']):x['canonical'] for x in result.inventory.id_map}
        self.assertEqual(result.inventory.id,mapping['machine',machine])
        self.assertEqual(result.inventory.components[1].parent_id,mapping['component',parent])
        self.assertEqual(result.inventory.selected_case_id,mapping['case',operating])
        self.assertEqual(result.inventory.components[1].source_refs,[child,'immutable_span:page_1'])
        self.assertEqual(result.inventory.cases[0].source_refs,['immutable_span:page_1'])
        self.assertEqual(raw,original)
        self.assertTrue(all(re.fullmatch(r'[a-z][a-z0-9_]{0,31}',value) for value in mapping.values()))
        self.assertEqual(result.provider_metadata['inventory_id_map'],result.inventory.id_map)
        repeated=MachineInventory.model_validate(original)
        self.assertEqual(repeated.id_map,result.inventory.id_map)
        restored=assembly_provider._inventory(json.loads(result.inventory.model_dump_json()))
        self.assertEqual(restored.id_map,result.inventory.id_map)
        assembly_provider._validate_case(restored,{'case_id':operating})
        schema=self.requests[0]['response_format']['json_schema']['schema']
        self.assertEqual(schema['properties']['id'],{'type':'string'})

    async def test_later_component_and_feature_refs_use_server_canonical_ids(self):
        raw_child='Custom region / alpha';raw_parent='Frame / main'
        raw=inventory_wire(custom_assembly())
        raw['id']='Machine / arbitrary internal label'
        raw['components'][0].update(id=raw_child,parent_id=raw_parent,source_refs=[raw_child])
        raw['components'].insert(0,assembly_provider.InventoryComponent(id=raw_parent,name='Frame',role='group',
            family='unknown',scope='group',mode='custom',intent='Grouping.').model_dump(mode='json'))
        inv=MachineInventory.model_validate(raw)
        inv=assembly_provider._inventory(json.loads(inv.model_dump_json()))
        parent,child=[c.id for c in inv.components]
        assembly_raw=custom_assembly().model_dump(mode='json');assembly_raw['id']=inv.id
        component=assembly_raw['components'][0];component.update(id=child,parent_id=parent,depends_on=[parent],source_refs=[raw_child])
        feature=component['features'][0];feature['depends_on']=[child+'.body']
        body=copy.deepcopy(feature);body.update(id='body',name='Body',pattern=None,depends_on=[],
            shape={'type':'circle','center':{'x':0.,'y':0.},'radius':1.},label={'x':0.,'y':0.})
        component['features'].append(body)
        assembly_raw['components'].insert(0,{'id':parent,'name':'Frame','role':'group','family':'unknown','mode':'custom'})
        assembly_raw['force_feature_id']=child+'.slots@000'
        assembly=MachineAssembly.model_validate(assembly_raw)
        replies=staged_replies(assembly,inventory_model=inv)
        replies[0]['force_feature_id']=raw_child+'.slots@000'
        replies[3]['depends_on']=[raw_parent]
        replies[4]['depends_on']=[raw_child+'.body']
        with self.fake(replies):
            result=await plan_assembly('Create an assembly with depth 5 mm.','key','https://provider.test','model',inventory=inv)
        self.assertEqual(result.assembly.id,inv.id)
        self.assertEqual(result.assembly.components[1].parent_id,parent)
        self.assertEqual(result.assembly.components[1].depends_on,[parent])
        self.assertEqual(result.assembly.components[1].features[0].depends_on,[child+'.body'])
        self.assertEqual(result.assembly.force_feature_id,child+'.slots@000')
        self.assertEqual(result.assembly.components[1].source_refs,[raw_child])
        self.assertTrue(result.compilation.execution_supported,result.compilation.issues)
        self.assertTrue(any(x['raw']==raw_child for x in result.provider_metadata[0]['inventory_id_map']))

    async def test_server_identity_map_is_not_a_provider_response_field(self):
        correct=inventory_wire();bad=copy.deepcopy(correct)
        bad['identity_map']=inventory().id_map
        with self.fake([bad,correct]):
            result=await discover_assembly(PROMPT,'key','https://provider.test','model')
        schema=self.requests[0]['response_format']['json_schema']['schema']
        self.assertNotIn('identity_map',schema['properties'])
        self.assertNotIn('InventoryIdentity',schema.get('$defs',{}))
        self.assertTrue(any(issue['path']=='/identity_map' and issue['code']=='wire_additional_property'
            for issue in self.payload(1)['validation_recovery']['issues']))
        self.assertEqual(result.inventory.id_map,inventory().id_map)

    def test_persisted_identity_map_cannot_forge_aliases_or_merge_identities(self):
        raw=inventory_wire();raw['id']='A machine / opaque'
        raw['components'][0]['id']='A winding / opaque'
        inv=MachineInventory.model_validate(raw);persisted=inv.model_dump(mode='json')
        forged=copy.deepcopy(persisted);forged['identity_map'][0]['raw']='Different raw identity'
        with self.assertRaisesRegex(ValueError,'deterministic'):
            assembly_provider._inventory(forged)
        incomplete=copy.deepcopy(persisted);incomplete['identity_map'].pop()
        with self.assertRaisesRegex(ValueError,'exact canonical'):
            assembly_provider._inventory(incomplete)
        duplicated=copy.deepcopy(persisted);duplicated['identity_map'].append(copy.deepcopy(duplicated['identity_map'][0]))
        with self.assertRaisesRegex(ValueError,'unique'):
            assembly_provider._inventory(duplicated)

    async def test_hyphenated_paper_parameter_identity_is_exact_and_never_normalized(self):
        source_id='Coil-radius-Inner'
        raw=coil_assembly().model_dump(mode='json')
        raw['facts'][0].update(status='paper',source_parameter_id=source_id,source_refs=['paper_span_1'])
        assembly=MachineAssembly.model_validate(raw)
        source={'parameters':[{'id':source_id,'value':10.,'unit':'mm'}]}
        with self.fake(staged_replies(assembly)):
            result=await plan_assembly(PROMPT,'key','https://provider.test','model',inventory=inventory(assembly),evidence_context=source)
        self.assertTrue(result.compilation.execution_supported,result.compilation.issues)
        self.assertEqual(result.assembly.facts[0].source_parameter_id,source_id)
        self.assertEqual(self.decision_requests,[])
        bad_settings=staged_replies(assembly)[0]
        with self.fake([bad_settings,bad_settings]):
            with self.assertRaisesRegex(AssemblyProviderError,'existing verified source'):
                await plan_assembly(PROMPT,'key','https://provider.test','model',inventory=inventory(assembly),
                    evidence_context={'parameters':[{'id':'Coil_radius_Inner','value':10.,'unit':'mm'}]})

    def test_normalization_preserves_distinct_ids_and_handles_reserved_hash_collision(self):
        mapping=assembly_provider._canonical_ids(['coil-a','coil a','coil_a'],'component')
        self.assertEqual(mapping['coil_a'],'coil_a')
        self.assertEqual(len(set(mapping.values())),3)
        reserved=mapping['coil-a']
        collision=assembly_provider._canonical_ids(['coil-a',reserved],'component')
        self.assertEqual(collision[reserved],reserved)
        self.assertNotEqual(collision['coil-a'],reserved)
        reversed_map=assembly_provider._canonical_ids([reserved,'coil-a'],'component')
        self.assertEqual(collision,reversed_map)
        raw=inventory_wire()
        raw['components']*=2
        with self.assertRaisesRegex(ValueError,'unique'):
            MachineInventory.model_validate(raw)

    async def test_legacy_inventory_scope_and_persistent_assembly_remain_compatible(self):
        assembly=coil_assembly();before=assembly.model_dump(mode='json')
        raw=inventory_wire(assembly);del raw['components'][0]['scope']
        with self.fake(staged_replies(assembly)):
            result=await plan_assembly(PROMPT,'key','https://provider.test','model',inventory=raw)
        self.assertEqual(result.assembly.model_dump(mode='json'),before)
        self.assertNotIn('scope',result.assembly.components[0].model_dump())
        self.assertEqual(result.provider_metadata[0]['legacy_scope_defaults'],['coil'])

    async def test_intent_scope_is_retained_without_fabricated_native_substitute(self):
        raw=coil_assembly().model_dump(mode='json')
        raw['components'].append({'id':'unknown','name':'Unknown custom component','role':'novel',
            'family':'previously undescribed','mode':'custom','description':'Contour not supplied.'})
        assembly=MachineAssembly.model_validate(raw);inv=inventory(assembly,scopes={'unknown':'intent'})
        with self.fake(staged_replies(assembly,inventory_model=inv)):
            result=await plan_assembly(PROMPT,'key','https://provider.test','model',inventory=inv)
        self.assertEqual(result.assembly.components[-1].features,[])
        self.assertTrue(any('intent only' in question for question in result.questions))
        self.assertIsNone(result.compilation.plan)

    def test_small_contracts_have_closed_required_fields_and_no_topology_caps(self):
        contracts=[assembly_provider.AssemblySettingsDraft,assembly_provider.ComponentDraft,
            *assembly_provider.FEATURE_CONTRACTS.values()]
        for model in contracts:
            schema=structured.strict_json_schema(model)
            self.assertEqual(schema['required'],list(schema['properties']))
            self.assertFalse(schema['additionalProperties'])
            self.assertNotIn('maxItems',json.dumps(schema))
            self.assertNotIn('MachineAssembly',schema.get('$defs',{}))
            self.assertNotIn('GeometryFeature',schema.get('$defs',{}))
            self.assertLess(len(json.dumps(schema)),len(json.dumps(structured.strict_json_schema(assembly_provider.AssemblyDraft))))


if __name__ == '__main__': unittest.main()
