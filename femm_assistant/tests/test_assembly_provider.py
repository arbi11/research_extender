"""No provider calls: strict wire, uncapped requests and two-stage source binding."""
import json
import unittest
from unittest.mock import patch

import httpx

from app.assembly_provider import (AssemblyDraft, AssemblyProviderError, MachineInventory,
    discover_assembly, detail_assembly, plan_assembly)
from app import assembly_provider, structured
from tests.test_assembly import coil_assembly


PROMPT = ('Create the finite annular coil. Inner radius 10 mm, outer radius 20 mm, '
          'axial bounds -20 mm and 20 mm, relative permeability 1, 100 turns, current 1 A.')


def inventory():
    return MachineInventory.model_validate({"id":"machine", "title":"Finite annular winding", "input_mode":"natural_language",
        "family":"new coil family", "physics":"axisymmetric_dc_magnetics", "explanation":"Finite winding, not a filament.",
        "components":[{"id":"coil","name":"Winding","role":"winding","family":"annular winding", "mode":"template",
            "template_kind":"annular_coil","intent":"Finite annular coil section."}]})


class AssemblyProviderTests(unittest.IsolatedAsyncioTestCase):
    def fake(self, replies):
        original = httpx.AsyncClient
        self.requests, self.options = [], []
        async def handle(request):
            self.requests.append(json.loads(request.content))
            reply = replies[len(self.requests)-1]
            if isinstance(reply, httpx.Response):
                return reply
            return httpx.Response(200,json={"id":"generation-test", "provider":"test-provider", "model":"test-model",
                "choices":[{"finish_reason":"stop", "message":{"content":json.dumps(reply)}}],
                "usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"cost":.001,"completion_tokens_details":{"reasoning_tokens":7}}})
        def client(*args, **kwargs):
            self.options.append(kwargs)
            return original(*args,transport=httpx.MockTransport(handle),**kwargs)
        return patch.object(assembly_provider.httpx, 'AsyncClient', side_effect=client)

    def draft(self):
        return AssemblyDraft(assembly=coil_assembly(),explanation='Compiled from explicit facts.',questions=[]).model_dump(mode='json')

    async def test_two_strict_stages_and_no_artificial_output_or_read_limits(self):
        with self.fake([inventory().model_dump(mode='json'),self.draft()]):
            result = await plan_assembly(PROMPT,'test-key','https://provider.test/v1','test-model')
        self.assertEqual(result.provider_calls,2)
        self.assertTrue(result.compilation.execution_supported,result.compilation.issues)
        self.assertEqual(len(result.usage),2)
        for request,options in zip(self.requests,self.options):
            self.assertEqual(request['response_format']['type'],'json_schema')
            self.assertTrue(request['response_format']['json_schema']['strict'])
            self.assertTrue(request['provider']['require_parameters'])
            self.assertNotIn('max_tokens',request)
            self.assertNotIn('max_completion_tokens',request)
            self.assertNotIn('temperature',request)
            self.assertIsNone(options['timeout'].read)
        self.assertEqual(result.provider_metadata[0]['id'],'generation-test')
        self.assertEqual(result.usage[0]['completion_tokens_details']['reasoning_tokens'],7)
        self.assertIn('schema_sha256',result.provider_metadata[0]['contract'])
        first_payload=json.loads(self.requests[0]['messages'][1]['content'])
        self.assertIn('capabilities',first_payload)
        self.assertEqual(json.loads(self.requests[1]['messages'][1]['content'])['inventory'], inventory().model_dump(mode='json'))

    async def test_persisted_paper_inventory_skips_discovery(self):
        with self.fake([self.draft()]):
            result=await plan_assembly(PROMPT,'key','https://provider.test','model',
                                      evidence_context={'assembly_inventory':inventory().model_dump(mode='json')})
        self.assertEqual(result.provider_calls,1)
        self.assertEqual(len(self.requests),1)

    async def test_discovery_has_open_families_and_retains_unknown_components(self):
        raw=inventory().model_dump(mode='json');raw['family']='previously undescribed machine'
        raw['components'][0].update(mode='custom',template_kind=None,family='unknown bespoke winding')
        with self.fake([raw]):
            result=await discover_assembly(PROMPT,'key','https://provider.test','model')
        self.assertEqual(result.inventory.family,'previously undescribed machine')
        self.assertEqual(result.inventory.components[0].mode,'custom')

    async def test_no_silent_wire_missing_field_or_extra_key_repair(self):
        raw=inventory().model_dump(mode='json');del raw['questions']
        with self.fake([raw]):
            with self.assertRaises(AssemblyProviderError): await discover_assembly(PROMPT,'key','https://provider.test','model')
        raw=inventory().model_dump(mode='json');raw['unrecognized']='value'
        with self.fake([raw]):
            with self.assertRaises(AssemblyProviderError): await discover_assembly(PROMPT,'key','https://provider.test','model')

    async def test_paper_facts_bind_existing_source_parameters_and_units(self):
        draft=self.draft()
        fact=draft['assembly']['facts'][0]
        fact.update(status='paper',source_parameter_id='coil_inner_radius',source_refs=['paper_span_1'])
        source={'parameters':[{'id':'coil_inner_radius','value':10.,'unit':'mm'}]}
        with self.fake([draft]):
            result,_,_=await detail_assembly(PROMPT,'key','https://provider.test','model',inventory=inventory(),evidence_context=source)
        self.assertEqual(result.assembly.facts[0].value,1.) # 1 cm equals sourced 10 mm
        source['parameters'][0]['value']=11.
        with self.fake([draft]):
            with self.assertRaisesRegex(AssemblyProviderError,'changed its verified paper'):
                await detail_assembly(PROMPT,'key','https://provider.test','model',inventory=inventory(),evidence_context=source)
        draft['assembly']['facts'][0]['source_parameter_id']='fabricated'
        with self.fake([draft]):
            with self.assertRaisesRegex(AssemblyProviderError,'existing verified source'):
                await detail_assembly(PROMPT,'key','https://provider.test','model',inventory=inventory(),evidence_context=source)

    async def test_unstated_physical_value_cannot_be_a_user_fact(self):
        with self.fake([self.draft()]):
            with self.assertRaisesRegex(AssemblyProviderError,'not present'):
                await detail_assembly('Create an unspecified coil.','key','https://provider.test','model',inventory=inventory())

    async def test_detail_cannot_replace_inventory_machine_or_component_ids(self):
        draft=self.draft();draft['assembly']['components'][0]['id']='other'
        with self.fake([draft]):
            with self.assertRaisesRegex(AssemblyProviderError,'stable hierarchy'):
                await detail_assembly(PROMPT,'key','https://provider.test','model',inventory=inventory())
        draft=self.draft();draft['assembly']['family']='a different device'
        with self.fake([draft]):
            with self.assertRaisesRegex(AssemblyProviderError,'family or physics'):
                await detail_assembly(PROMPT,'key','https://provider.test','model',inventory=inventory())

    async def test_unsupported_physics_graph_remains_available(self):
        draft=self.draft();draft['assembly']['required_physics']=['permanent_magnets']
        with self.fake([draft]):
            result=await plan_assembly(PROMPT,'key','https://provider.test','model',inventory=inventory())
        self.assertIsNotNone(result.assembly)
        self.assertFalse(result.compilation.execution_supported)
        self.assertEqual(len(result.compilation.expanded_features),1)
        self.assertTrue(any(issue.code=='unsupported_physics' for issue in result.compilation.issues))

    async def test_provider_failure_is_not_retried_and_no_key_exposed(self):
        with self.fake([httpx.Response(429,json={'error':'secret-key'})]):
            with self.assertRaises(AssemblyProviderError) as caught:
                await discover_assembly(PROMPT,'secret-key','https://provider.test','model')
        self.assertEqual(len(self.requests),1)
        self.assertEqual(caught.exception.status_code,429)
        self.assertNotIn('secret-key',caught.exception.message)

    def test_schema_includes_optional_nulls_and_no_open_objects(self):
        contract=structured.strict_json_schema(AssemblyDraft)
        self.assertEqual(contract['required'],list(contract['properties']))
        self.assertFalse(contract['additionalProperties'])
        self.assertIn('closed_path',json.dumps(contract))
        self.assertNotIn('maxItems',json.dumps(contract))


if __name__ == '__main__': unittest.main()
