"""Graph and parameter contracts; physics results require native evidence."""
from copy import deepcopy
import unittest

from femm_brain2.knowledge import arithmetic, construct, construction_plan, load_knowledge, validate_knowledge
from femm_brain2.knowledge_compile import generate


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.knowledge=deepcopy(load_knowledge())
        self.paper=self.knowledge['papers'][0]

    def test_records_form_the_agreed_five_levels(self):
        result=validate_knowledge(self.knowledge)
        self.assertEqual(result['status'],'passed')
        self.assertEqual(set(result['levels']),{'1','2','3','4','5'})
        plan=construction_plan(self.knowledge,self.paper['id'])
        positions={s['id']:i for i,s in enumerate(plan['steps'])}
        self.assertLess(positions['build_semi_closed_stator'],positions['assign_concentrated_winding'])

    def test_missing_or_cyclic_construction_prerequisites_are_rejected(self):
        del self.paper['bindings']['build_semi_closed_stator']
        with self.assertRaisesRegex(ValueError,'missing a required'):
            construction_plan(self.knowledge,self.paper['id'])
        self.setUp()
        self.knowledge['edges'].append({'parent':'build_semi_closed_stator','child':'assign_concentrated_winding','relation':'requires'})
        with self.assertRaisesRegex(ValueError,'yclic'):
            construction_plan(self.knowledge,self.paper['id'])

    def test_unreviewed_api_and_skipped_hierarchy_levels_are_rejected(self):
        node=next(n for n in self.knowledge['nodes'] if n['level']==5)
        node['api_commands'].append('mi_invented_command')
        with self.assertRaisesRegex(ValueError,'Unreviewed API'):
            validate_knowledge(self.knowledge)
        self.setUp()
        self.knowledge['edges'].append({'parent':'electric_motor','child':'stator','relation':'contains'})
        with self.assertRaisesRegex(ValueError,'skips a level'):
            validate_knowledge(self.knowledge)

    def test_derived_value_tampering_and_code_expressions_are_rejected(self):
        self.paper['parameters']['air_gap_mm']['value']=2
        with self.assertRaisesRegex(ValueError,'derived value'):
            validate_knowledge(self.knowledge)
        with self.assertRaisesRegex(ValueError,'Unsupported'):
            arithmetic('__import__("os").system("false")',{})

    def test_impossible_geometry_is_refused_before_femm(self):
        with self.assertRaisesRegex(ValueError,'Air gap must be positive'):
            construct(construction_plan(self.knowledge,self.paper['id'],{'r3_cm':2.6}))

    def test_api_trace_binds_every_generated_call_and_excludes_expected_results(self):
        control=next(c for c in self.paper['controls'] if c['id']=='baseline')
        script,trace,geometry=generate(self.knowledge,self.paper['id'],control)
        self.assertTrue(trace)
        self.assertEqual(set(geometry.trace),set(self.paper['bindings']))
        self.assertNotIn(str(self.paper['acceptance']['emf_fundamental_peak_V']),script)
        self.assertIn('mi_moverotate(',script)
        self.assertIn('mo_getcircuitproperties(',script)


if __name__=='__main__':
    unittest.main()
