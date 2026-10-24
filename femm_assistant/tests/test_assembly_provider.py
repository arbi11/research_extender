"""No paid calls: staged strict contracts, typed graph recovery and source checks."""
import copy
import json
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


def inventory(assembly=None):
    assembly = assembly or coil_assembly()
    return MachineInventory.model_validate({
        'id':assembly.id, 'title':assembly.title, 'input_mode':assembly.input_mode,
        'family':assembly.family, 'physics':assembly.physics, 'required_physics':assembly.required_physics,
        'explanation':'Discover intent before filling facts and geometry.',
        'components':[{'id':c.id, 'name':c.name, 'role':c.role, 'family':c.family,
            'mode':c.mode, 'parent_id':c.parent_id, 'template_kind':c.template.kind if c.template else None,
            'source_refs':c.source_refs, 'required_physics':c.required_physics, 'intent':c.description or c.name}
            for c in assembly.components]})


def staged_replies(assembly=None, *, include_discovery=False):
    """Mock wire transcript in public stage order; also used by runner contract tests."""
    assembly = assembly or coil_assembly()
    replies = [inventory(assembly).model_dump(mode='json')] if include_discovery else []
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
        intents=[];details=[]
        for op,identity,f in [('feature',f.id,f) for f in c.features]+[(o.op,o.feature_id,o.feature) for o in c.overrides]:
            intents.append({'id':identity,'name':f.name if f else identity,'role':f.role if f else 'removed_feature',
                'operation':op, 'shape_kind':f.shape.type if f and f.shape else 'unresolved'})
            if f is not None:
                raw=f.model_dump(mode='json')
                for key in ('id','name','role'): raw.pop(key)
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
        raw=inventory().model_dump(mode='json');raw['family']='previously undescribed machine'
        raw['components'][0].update(mode='custom',template_kind=None,family='unknown bespoke winding')
        with self.fake([raw]):
            result=await discover_assembly(PROMPT,'key','https://provider.test','model')
        self.assertEqual(result.inventory.family,'previously undescribed machine')
        self.assertEqual(result.inventory.components[0].mode,'custom')

    async def test_actual_wire_error_gets_one_strict_same_stage_repair(self):
        correct=inventory().model_dump(mode='json');bad=copy.deepcopy(correct);del bad['questions']
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
        raw=inventory().model_dump(mode='json');raw['unrecognized']='value'
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

    async def test_unknown_paper_parameter_and_unstated_user_fact_are_not_invented(self):
        for source_mode in ('paper','user'):
            replies=staged_replies();bad=copy.deepcopy(replies[0])
            if source_mode=='paper':
                bad['facts'][0].update(status='paper',source_parameter_id='fabricated')
            with self.fake([bad,bad]):
                with self.assertRaises(AssemblyProviderError) as caught:
                    await detail_assembly('Create an unspecified coil.','key','https://provider.test','model',inventory=inventory())
            self.assertEqual(caught.exception.provider_calls,3)
            self.assertIn('existing verified source' if source_mode=='paper' else 'not present',caught.exception.message)

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
        raw=inventory().model_dump();raw.update(cases=[{'id':'first','label':'first'}],selected_case_id='first')
        with self.fake([]):
            with self.assertRaisesRegex(AssemblyProviderError,'operating case'):
                await detail_assembly(PROMPT,'key','https://provider.test','model',inventory=raw,evidence_context={'case_id':'another'})
        self.assertEqual(self.requests,[])

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
        correct=inventory().model_dump(mode='json');bad=copy.deepcopy(correct);del bad['questions']
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
        bad=inventory().model_dump(mode='json');del bad['questions']
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
        assembly=MachineAssembly.model_validate(raw);replies=staged_replies(assembly)
        bad=copy.deepcopy(replies[-1]);bad['depends_on']=['first']
        replies[-1:]=[bad,replies[-1]]
        with self.fake(replies):
            result=await plan_assembly('Create a machine of depth 5 mm.','key','https://provider.test','model',inventory=inventory(assembly))
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
