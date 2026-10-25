"""Source-backed planning and native artifact checks, without network or AI."""
from copy import deepcopy
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from femm_brain.core import load_graph
from femm_brain.simulation import (
    _field_svg, assess_target, plan_simulation, read_native_mesh, run_simulation,
    simulation_catalog, verify_boundary_attachments, verify_native_physics,
)


class SimulationTests(unittest.TestCase):
    def test_catalog_bound_to_existing_source_workflows_and_methods(self):
        graph=load_graph()
        catalog=simulation_catalog(graph)
        self.assertEqual(len(catalog),6)
        node_ids={n['id'] for n in graph['nodes']}
        sources={s['id'] for s in graph['sources']}
        for case in catalog:
            self.assertFalse(case['requires_ai'])
            self.assertTrue(set(case['knowledge_node_ids'])<=node_ids)
            self.assertTrue(set(case['source_ids'])<=sources)
            self.assertIn('workflow:'+case['example_id'],case['knowledge_node_ids'])
            self.assertTrue(case['reference']['source_locator'])

    def test_stale_sources_or_missing_artifacts_block_silent_reuse(self):
        graph=load_graph();graph['source_changes']=['aircore']
        with self.assertRaises(ValueError):simulation_catalog(graph)
        graph=load_graph();graph['artifacts']=[]
        with self.assertRaises(ValueError):simulation_catalog(graph)
        catalog=simulation_catalog()
        for case in catalog:
            self.assertTrue(all(s['sha256'] for s in case['knowledge_sources']))
            if case['family']=='uniform':
                self.assertEqual(case['knowledge_artifacts'][0]['sha256'],case['source_model_sha256'])

    def test_plan_is_deterministic_and_does_not_mutate_catalog(self):
        before=simulation_catalog()
        first=plan_simulation('aircore-short',{'current':2})
        second=plan_simulation('aircore-short',{'current':2})
        self.assertEqual(first,second)
        self.assertEqual(before,simulation_catalog())
        self.assertEqual(first['parameters']['current'],2)
        self.assertEqual(first['frozen_reference']['value'],.00064707)
        self.assertTrue(first['frozen_reference']['applicable'])
        self.assertFalse(first['numerical_convergence_verified'])

    def test_geometry_or_turn_edits_never_inherit_published_target(self):
        for parameters in ({'turns':200},{'length':1.5},{'inner_radius':.8},{'outer_radius':2.5}):
            with self.subTest(parameters=parameters):
                plan=plan_simulation('aircore-short',parameters)
                self.assertFalse(plan['frozen_reference']['applicable'])
                self.assertEqual(assess_target(plan,{'inductance_H':.00064707})['status'],'unassessed')
                self.assertFalse(plan['comparators'][0]['used_for_acceptance'])
                self.assertFalse(plan['comparators'][0]['applicable'])

    def test_source_units_circuit_and_mixed_boundary_are_preserved(self):
        plan=plan_simulation('aircore-short')
        self.assertEqual(plan['metres_per_coordinate_unit'],.0254)
        self.assertIn('mi_probdef(0,"inches","axi",1e-8,0,30)',plan['lua'])
        self.assertIn('mi_setblockprop("coil",0,0.20000000000000001,"icoil",0,0,100)',plan['lua'])
        self.assertIn('mi_addcircprop("icoil",1,1)',plan['lua'])
        self.assertEqual(plan['lua'].count('mi_addboundprop'),1)
        self.assertIn('mi_analyze(1)',plan['lua'])
        self.assertIn('mo_getcircuitproperties',plan['lua'])
        self.assertIn('mi_loadsolution()',plan['lua'])
        self.assertEqual(plan['geometry']['metadata']['ProblemType'],'axisymmetric')
        self.assertNotIn('dofile',plan['lua'])

    def test_unknown_nonfinite_boolean_or_impossible_parameters_rejected(self):
        for params in ({'outer_radius':.5},{'turns':1.5},{'turns':True},{'length':float('nan')},
                       {'current':0},{'mesh_scale':0},{'outer_radius':'2'},{'bogus':1}):
            with self.subTest(params=params),self.assertRaises(ValueError):
                plan_simulation('aircore-short',params)
        with self.assertRaises(ValueError):plan_simulation('missing')
        with self.assertRaises(ValueError):plan_simulation('aircore-short',[])

    def test_uniform_analytic_target_scales_by_magnitude_and_radius_invariance(self):
        for case,baseline in [('uniform-planar',.5),('uniform-axisymmetric',2/3)]:
            p=plan_simulation(case,{'radius':42,'applied_field_scale':-2})
            self.assertEqual(p['frozen_reference']['metric'],'field_magnitude_T')
            self.assertEqual(p['frozen_reference']['value'],2*baseline)
            self.assertTrue(p['frozen_reference']['applicable'])
            self.assertEqual(p['frozen_reference']['kind'],'analytic_uniform_field')
            self.assertTrue(p['source_model_data_sha256'])
            self.assertIn('shell_7',p['lua'])
            self.assertTrue(any('recentered' in a for a in p['assumptions']))

    def test_uniform_target_checks_spatial_uniformity_and_transverse_component(self):
        plan=plan_simulation('uniform-planar')
        metrics={'field_magnitude_T':.5,'field_x_T':-.5,'field_y_T':0}
        for i in range(5):
            metrics.update({f'probe_{i}_field_magnitude_T':.5,f'probe_{i}_field_x_T':-.5,f'probe_{i}_field_y_T':0})
        self.assertEqual(assess_target(plan,metrics)['status'],'passed')
        metrics['probe_3_field_x_T']=.5
        self.assertEqual(assess_target(plan,metrics)['status'],'failed')
        metrics['probe_3_field_x_T']=-.5
        metrics['probe_3_field_y_T']=.1
        self.assertEqual(assess_target(plan,metrics)['status'],'failed')

    def test_zero_field_target_uses_absolute_residual_and_reports_components(self):
        plan=plan_simulation('uniform-planar',{'applied_field_scale':0})
        metrics={'field_magnitude_T':0,'field_x_T':0,'field_y_T':0,
                 'probe_0_field_magnitude_T':0,'probe_0_field_x_T':0,'probe_0_field_y_T':0}
        result=assess_target(plan,metrics)
        self.assertEqual(result['status'],'passed')
        self.assertIsNone(result['relative_error'])
        self.assertEqual(result['absolute_tolerance'],1e-10)

    def test_commercial_transfer_preserves_source_wires_and_discloses_boundary_adaptation(self):
        plan=plan_simulation('parallel-wires')
        self.assertAlmostEqual(plan['frozen_reference']['value'],2e-7,delta=1e-20)
        self.assertEqual(plan['frozen_reference']['relative_tolerance'],.05)
        self.assertEqual(plan['geometry']['metadata']['LengthUnits'],'meters')
        self.assertEqual(plan['parameters']['wire_radius'],.2)
        self.assertEqual(plan['parameters']['center_distance'],1)
        self.assertIn('mi_addcircprop("left_current",1,1)',plan['lua'])
        self.assertIn('mi_addcircprop("right_current",-1,1)',plan['lua'])
        self.assertIn('mo_blockintegral(18)',plan['lua'])
        self.assertIn('mo_blockintegral(11)',plan['lua'])
        self.assertTrue(any('infinite elements' in a for a in plan['assumptions']))
        edited=plan_simulation('parallel-wires',{'current':2,'center_distance':2})
        self.assertAlmostEqual(edited['frozen_reference']['value'],4e-7,delta=1e-20)
        for invalid in ({'wire_radius':.5},{'domain_radius':1}):
            with self.assertRaises(ValueError):plan_simulation('parallel-wires',invalid)

    def test_force_target_rejects_attraction_unbalanced_or_transverse_results(self):
        plan=plan_simulation('parallel-wires')
        metrics={'repulsive_force_N_per_m':2e-7,'force_left_x_N':-2e-7,'force_right_x_N':2e-7,
                 'force_left_y_N':0,'force_right_y_N':0,'lorentz_repulsive_force_N_per_m':2e-7}
        self.assertEqual(assess_target(plan,metrics)['status'],'passed')
        for key,value in [('force_left_x_N',2e-7),('force_right_x_N',3e-7),('force_left_y_N',1e-7)]:
            changed=dict(metrics);changed[key]=value
            self.assertEqual(assess_target(plan,changed)['status'],'failed')

    def test_native_mesh_reader_normalizes_inches_to_square_mm(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            (folder/'model.node').write_text('3 2 0 0\n0 0 0\n1 1 0\n2 0 1\n')
            (folder/'model.ele').write_text('1 3 1\n0 0 1 2 1\n')
            mesh=read_native_mesh(folder,.0254)
            self.assertAlmostEqual(mesh['total_cross_section_area_mm2'],.5*25.4**2)
            for bad in ('0 0 0 2 1','0 0 1 4 1','0 0 1 2 nan','0 0 1 2 0'):
                (folder/'model.ele').write_text('1 3 1\n'+bad+'\n')
                with self.subTest(bad=bad),self.assertRaises(ValueError):read_native_mesh(folder,.0254)

    def test_native_physics_verification_catches_current_turns_and_unit_changes(self):
        from femm_brain.simulation import MU0
        plan=plan_simulation('aircore-short')
        actual={'metadata':{'LengthUnits':'inches','ProblemType':'axisymmetric','Frequency':'0'},
                'materials':[{'name':name,'properties':{'Mu_x':'1','Mu_y':'1','H_c':'0'}} for name in ('coil','air')],
                'circuits':[{'name':'icoil','properties':{'TotalAmps_re':'1','CircuitType':'1'}}],
                'labels':[{'material_index':1,'attributes':['1','.2','1','0','0','100','0']},
                          {'material_index':2,'attributes':['2','.04','0','0','0','1','0']}],
                'boundaries':[{'properties':{'BdryType':'2','c0':str(1/(4*.0254*MU0)),'c1':'0'}}]}
        actual['nodes']=deepcopy(plan['geometry']['nodes'])
        actual['segments']=[dict(deepcopy(segment),attributes=['-1','0','0','0']) for segment in plan['geometry']['segments']]
        actual['arcs']=[dict(deepcopy(arc),attributes=['180','5','1','0','0','0']) for arc in plan['geometry']['arcs']]
        self.assertEqual(verify_native_physics(plan,actual)['status'],'passed')
        alterations=[('units',lambda x:x['metadata'].update(LengthUnits='millimeters')),
                     ('turns',lambda x:x['labels'][0]['attributes'].__setitem__(5,'99')),
                     ('current',lambda x:x['circuits'][0]['properties'].update(TotalAmps_re='2')),
                     ('boundary',lambda x:x['boundaries'][0]['properties'].update(c0='0')),
                     ('material',lambda x:x['materials'][0]['properties'].update(Mu_x='1000'))]
        for name,alter in alterations:
            changed=deepcopy(actual);alter(changed)
            with self.subTest(name=name),self.assertRaises(ValueError):verify_native_physics(plan,changed)

    def test_boundary_attachments_reject_removed_exterior_and_added_interior_or_axis_conditions(self):
        for case in simulation_catalog():
            plan=plan_simulation(case['id'])
            actual=deepcopy(plan['geometry']);actual['boundaries']=[{}]
            p=plan['parameters']
            radius=(2*max(p['inner_radius'],p['outer_radius'],p['length'])*p['boundary_scale'] if case['family']=='aircore'
                    else p['domain_radius'] if case['family']=='parallel_wires' else 1.1*p['radius'])
            # Independent fixture classification uses endpoint distances, while
            # production checks the frozen circular center and radius.
            for arc in actual['arcs']:
                outer=all(math.isclose(math.hypot(actual['nodes'][arc[n]]['x'],actual['nodes'][arc[n]]['y']),radius,rel_tol=1e-11) for n in ('n0','n1'))
                arc['attributes']=[str(arc['angle']),'5','1' if outer else '0','0','0','0']
            for segment in actual['segments']:segment['attributes']=['-1','0','0','0']
            with self.subTest(case=case['id']):
                result=verify_boundary_attachments(plan,actual)
                self.assertEqual(result['status'],'passed')
                self.assertGreater(result['assigned_exterior_arcs'],0)
                stripped=deepcopy(actual)
                for arc in stripped['arcs']:arc['attributes'][2]='0'
                with self.assertRaisesRegex(ValueError,'arc boundary attachment'):verify_boundary_attachments(plan,stripped)
                outer=next(a for a in actual['arcs'] if a['attributes'][2]=='1')
                wrong_index=deepcopy(actual)
                wrong_index['arcs'][actual['arcs'].index(outer)]['attributes'][2]='2'
                with self.assertRaises(ValueError):verify_boundary_attachments(plan,wrong_index)
                interior=next((a for a in actual['arcs'] if a['attributes'][2]=='0'),None)
                if interior:
                    changed=deepcopy(actual);changed['arcs'][actual['arcs'].index(interior)]['attributes'][2]='1'
                    with self.assertRaisesRegex(ValueError,'arc boundary attachment'):verify_boundary_attachments(plan,changed)
                if actual['segments']:
                    changed=deepcopy(actual);changed['segments'][0]['attributes'][1]='1'
                    with self.assertRaisesRegex(ValueError,'segment must have no boundary'):verify_boundary_attachments(plan,changed)
                reordered=deepcopy(actual)
                reordered['arcs'].reverse();reordered['segments'].reverse()
                self.assertEqual(verify_boundary_attachments(plan,reordered)['status'],'passed')
                changed=deepcopy(actual);changed['boundaries'].append({})
                with self.assertRaises(ValueError):verify_boundary_attachments(plan,changed)

    def test_field_viewport_focuses_actual_samples_and_records_magnitude_legend(self):
        import xml.etree.ElementTree as ET
        geometry=plan_simulation('parallel-wires',{'domain_radius':20})['geometry']
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            (folder/'field.csv').write_text('x,y,bx,by\n-1.5,-1.5,1e-7,0\n1.5,1.5,0,2e-7\n0,0,0,0\n')
            svg=_field_svg(folder,geometry)
        root=ET.fromstring(svg);ns={'s':'http://www.w3.org/2000/svg'}
        x,y,width,height=[float(v) for v in root.attrib['viewBox'].split()]
        self.assertLess(width,4)  # The outer geometry is 40 m wide.
        self.assertLess(height,5)
        self.assertLessEqual(x,-1.5);self.assertLessEqual(y,-1.5)
        self.assertGreaterEqual(x+width,1.5);self.assertGreaterEqual(y+height,1.5)
        self.assertIn('sampled |B| range 0 to 2e-07 T',root.find('s:title',ns).text)
        self.assertIn('|B|: 0 to 2e-07 T',svg)
        self.assertIn('Vector lengths normalized',svg)
        self.assertIsNotNone(root.find('.//s:clipPath',ns))
        vectors=root.find('.//s:g[@class="native-field-vectors"]',ns)
        self.assertEqual(len(vectors.findall('s:path',ns)),2)
        self.assertTrue(root.findall('.//s:g[@clip-path="url(#field-sample-clip)"]',ns))

    def test_zero_field_plot_has_visible_extent_and_truthful_zero_legend(self):
        import xml.etree.ElementTree as ET
        geometry=plan_simulation('uniform-planar',{'applied_field_scale':0})['geometry']
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            (folder/'field.csv').write_text('x,y,bx,by\n-1,-1,0,0\n1,1,0,0\n')
            svg=_field_svg(folder,geometry)
        root=ET.fromstring(svg);ns={'s':'http://www.w3.org/2000/svg'}
        self.assertTrue(all(math.isfinite(float(v)) for v in root.attrib['viewBox'].split()))
        self.assertIn('|B|: 0 to 0 T',svg)
        self.assertIn('Zero field: no vectors.',svg)
        self.assertEqual(len(root.find('.//s:g[@class="native-field-vectors"]',ns)),0)

    def test_no_native_launcher_cannot_manufacture_a_passing_report(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):run_simulation('aircore-short',folder,command='/no/native/femm-lua')
            self.assertFalse((Path(folder)/'report.json').exists())

    def test_existing_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            retained=Path(folder)/'original.txt';retained.write_text('keep')
            with patch('femm_brain.simulation.shutil.which',return_value='/mock/femm-lua'):
                with self.assertRaises(ValueError):run_simulation('aircore-short',folder)
            self.assertEqual(retained.read_text(),'keep')
            self.assertFalse((Path(folder)/'plan.json').exists())

    def test_missing_native_stage_retains_frozen_plan_and_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('femm_brain.simulation.shutil.which',return_value='/mock/femm-lua'), \
                 patch('femm_brain.simulation._version',return_value={}), \
                 patch('femm_brain.simulation._run',side_effect=RuntimeError('actual native error')):
                result=run_simulation('aircore-short',folder)
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['checks']['geometry']['status'],'failed')
            self.assertEqual(result['checks']['mesh']['status'],'not_run')
            self.assertEqual(result['execution_error'],'actual native error')
            self.assertTrue((Path(folder)/'plan.json').is_file())
            self.assertTrue((Path(folder)/'manifest.json').is_file())


@unittest.skipUnless(os.getenv('FEMM_BRAIN_NATIVE')=='1','Set FEMM_BRAIN_NATIVE=1 with FEMM 5 installed')
class NativeSimulationTests(unittest.TestCase):
    def test_all_source_cases_generate_mesh_and_solve(self):
        with tempfile.TemporaryDirectory() as folder:
            for case in simulation_catalog():
                with self.subTest(case=case['id']):
                    report=run_simulation(case['id'],Path(folder)/case['id'])
                    self.assertEqual(report['status'],'passed',report)
                    self.assertTrue(all(c['status']=='passed' for c in report['checks'].values()))
                    self.assertGreater(report['checks']['mesh']['element_count'],0)
                    self.assertFalse(report['requires_ai'])
                    self.assertFalse(report['numerical_convergence_verified'])
                    self.assertTrue((Path(folder)/case['id']/'model.ans').is_file())


if __name__=='__main__':unittest.main()
