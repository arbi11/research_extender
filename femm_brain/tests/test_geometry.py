"""Executable geometry recipes and passive native-model reconstruction."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

from femm_brain.geometry import arc_geometry, construct, export_recipe, parse_fem, recipe_catalog


NATIVE = '''[Format] = 4.0
[Frequency] = 50
[Depth] = 100
[LengthUnits] = millimeters
[ProblemType] = planar
[PointProps] = 0
[BdryProps] = 1
<BeginBdry>
<BdryName> = "periodic edge"
<BdryType> = 4
<EndBdry>
[BlockProps] = 1
<BeginBlock>
<BlockName> = "Example steel"
<Mu_x> = 1000
<BHPoints> = 2
0 0
1.5 1000
<EndBlock>
[CircuitProps] = 1
<BeginCircuit>
<CircuitName> = "Phase A"
<TotalAmps_re> = 2
<CircuitType> = 1
<EndCircuit>
[NumPoints] = 4
0 0 0 2
10 0 0 2
10 10 0 2
0 10 0 2
[NumSegments] = 3
0 1 -1 1 0 2
1 2 -1 0 0 2
2 3 -1 0 0 2
[NumArcSegments] = 1
3 0 180 5 0 0 2
[NumHoles] = 0
[NumBlockLabels] = 1
5 5 1 1 1 0 2 10 0 0
'''


class GeometryTests(unittest.TestCase):
    def test_every_recipe_is_closed_and_every_visible_primitive_links_to_an_operation(self):
        for recipe in recipe_catalog():
            with self.subTest(recipe=recipe['id']):
                result = construct(recipe['id'])
                self.assertEqual(result['validation']['status'], 'passed')
                self.assertFalse(result['validation']['native_mesh_verified'])
                self.assertFalse(result['validation']['numerical_convergence_verified'])
                self.assertEqual(result['metadata']['dimension_source'], 'illustrative_parameters')
                xml = ET.fromstring(result['svg'])
                operations = {op['id']: op for op in result['operations']}
                primitives = [element for element in xml.iter() if element.tag.endswith(('line', 'path'))]
                self.assertEqual(len(primitives), len(result['segments'])+len(result['arcs']))
                for primitive in primitives:
                    op = operations[primitive.attrib['data-operation-id']]
                    self.assertIn(primitive.attrib['id'], op['geometry_ids'])
                visible = {element.attrib['id']: element for element in xml.iter() if 'id' in element.attrib}
                for operation in result['operations']:
                    for geometry_id in operation['geometry_ids']:
                        self.assertIn(geometry_id, visible)
                        self.assertEqual(visible[geometry_id].attrib['data-operation-id'], operation['id'])
                        self.assertEqual(visible[geometry_id].attrib['data-geometry-id'], geometry_id)
                self.assertEqual(len({(round(p['x'],11),round(p['y'],11)) for p in result['nodes']}), len(result['nodes']))

    def test_stator_is_a_connected_yoke_with_one_continuous_slotted_inner_boundary(self):
        result = construct('slotted_stator', {'slot_count':24,'slot_opening':1.5,'slot_width':4})
        neighbors = {i:set() for i in range(len(result['nodes']))}
        for primitive in result['segments']+result['arcs']:
            a,b = primitive['n0'],primitive['n1']
            neighbors[a].add(b); neighbors[b].add(a)
        remaining = set(neighbors); components=[]
        while remaining:
            reached=set(); todo=[next(iter(remaining))]
            while todo:
                at=todo.pop()
                if at not in reached:
                    reached.add(at); todo.extend(neighbors[at]-reached)
            remaining-=reached; components.append(reached)
        self.assertEqual(sorted(map(len,components)), [4,144])
        self.assertTrue(all(len(value)==2 for value in neighbors.values()))
        # Outer circle consists of exact quarter arcs; every slot contributes
        # one curved root and one bore-face arc.
        self.assertEqual(len(result['arcs']), 4+2*24)
        self.assertEqual(len(result['segments']), 4*24)

    def test_parameter_changes_rebuild_the_actual_geometry_and_roundtrip_native_file(self):
        base = construct('stator_tooth')
        changed = construct('stator_tooth',{'bore_radius':30,'tip_width':9,'body_width':6,'tip_height':2,'height':12})
        self.assertNotEqual(base['nodes'],changed['nodes'])
        for recipe in recipe_catalog():
            with self.subTest(recipe=recipe['id']):
                original = construct(recipe['id'])
                parsed = parse_fem(export_recipe(recipe['id'],format='fem'))
                self.assertEqual(len(parsed['nodes']),len(original['nodes']))
                self.assertEqual(len(parsed['segments']),len(original['segments']))
                self.assertEqual(len(parsed['arcs']),len(original['arcs']))
                for first,second in zip(parsed['nodes'],original['nodes']):
                    self.assertAlmostEqual(first['x'],second['x'],places=11)
                    self.assertAlmostEqual(first['y'],second['y'],places=11)

    def test_exact_arc_centers_and_major_arc_preview_bounds(self):
        quarter = arc_geometry((1,0),(0,1),90)
        self.assertAlmostEqual(quarter['center'][0],0,places=12)
        self.assertAlmostEqual(quarter['center'][1],0,places=12)
        self.assertAlmostEqual(quarter['radius'],1,places=12)
        major = arc_geometry((1,0),(0,-1),270)
        self.assertAlmostEqual(major['center'][0],0,places=12)
        self.assertAlmostEqual(major['center'][1],0,places=12)
        native = '[NumPoints] = 2\n1 0\n0 -1\n[NumSegments] = 0\n[NumArcSegments] = 1\n0 1 270 5\n'
        result=parse_fem(native)
        self.assertIn('0 1 0',result['svg'])
        bounds=list(map(float,ET.fromstring(result['svg']).attrib['viewBox'].split()))
        self.assertLess(bounds[0],-1)
        self.assertGreater(bounds[2],2)
        clockwise=arc_geometry((1,0),(0,-1),-90)
        self.assertAlmostEqual(clockwise['center'][0],0,places=12)
        self.assertAlmostEqual(clockwise['center'][1],0,places=12)

    def test_invalid_physical_parameters_do_not_generate_geometry(self):
        cases = [
            ('circle',{'radius':0}),('circle',{'radius':float('nan')}),
            ('circle',{'radius':'10'}),('circle',{'radius':True}),
            ('circle',{'unknown':3}),('rectangle',{'height':-1}),
            ('stator_tooth',{'body_width':9,'tip_width':7}),
            ('stator_tooth',{'tip_height':10}),
            ('stator_tooth',{'tip_width':40}),
            ('stator_tooth',{'tip_height':.02}),
            ('slotted_stator',{'slot_count':12.5}),
            ('slotted_stator',{'slot_count':96}),
            ('slotted_stator',{'outer_radius':25}),
            ('slotted_stator',{'tip_depth':9}),
            ('circular_airgap',{'outer_radius':20.5}),
            ('rotor_bar',{'length':2,'width':3}),
            ('flux_barrier',{'inner_radius':1,'thickness':2}),
            ('flux_barrier',{'span':1,'thickness':3}),
        ]
        for recipe, parameters in cases:
            with self.subTest(recipe=recipe,parameters=parameters),self.assertRaises(ValueError):
                construct(recipe,parameters)
        with self.assertRaises(ValueError): construct('../anything')
        with self.assertRaises(ValueError): construct('circle',[])

    def test_native_parser_preserves_material_boundaries_circuit_and_source_locations(self):
        result=parse_fem(NATIVE)
        self.assertEqual(result['metadata']['Frequency'],'50')
        self.assertEqual(result['materials'][0]['name'],'Example steel')
        self.assertEqual(result['materials'][0]['bh_points'],[[0.,0.],[1.5,1000.]])
        self.assertEqual(result['boundaries'][0]['name'],'periodic edge')
        self.assertEqual(result['circuits'][0]['name'],'Phase A')
        self.assertEqual(result['nodes'][0]['group'],2)
        self.assertEqual(result['segments'][0]['group'],2)
        self.assertEqual(result['arcs'][0]['group'],2)
        self.assertEqual(result['labels'][0]['material_index'],1)
        self.assertGreater(result['nodes'][0]['source_line'],1)
        self.assertTrue(all(op['origin']=='native_file_reconstruction' for op in result['operations']))
        self.assertIn('not the original authoring history',result['verification']['note'])
        self.assertEqual(result['metadata']['physics'],'magnetics')
        visible = {element.attrib['id']: element for element in ET.fromstring(result['svg']).iter() if 'id' in element.attrib}
        for operation in result['operations']:
            for geometry_id in operation['geometry_ids']:
                self.assertEqual(visible[geometry_id].attrib['data-operation-id'],operation['id'])

    def test_native_physics_namespaces_match_fee_feh_and_fec_files(self):
        # Native geometric sections are shared; their reconstructed calls must
        # use the preprocessor belonging to the original file's physics.
        for selector, canonical, prefix in [
                ('electrostatics','electrostatics','ei'),('fee','electrostatics','ei'),
                ('heatflow','heatflow','hi'),('.feh','heatflow','hi'),
                ('currentflow','currentflow','ci'),('fec','currentflow','ci'),
                ('mi','magnetics','mi')]:
            with self.subTest(physics=selector):
                result=parse_fem(NATIVE,physics=selector)
                self.assertEqual(result['metadata']['physics'],canonical)
                self.assertEqual({op['function'] for op in result['operations']},
                                 {prefix+'_addnode',prefix+'_addsegment',prefix+'_addarc',prefix+'_addblocklabel'})
        with self.assertRaises(ValueError): parse_fem(NATIVE,physics='unknown')
        with self.assertRaises(ValueError): parse_fem(NATIVE,physics=None)

    def test_malformed_native_files_are_rejected_without_execution(self):
        invalid = [
            'not FEMM',
            '[NumPoints] = 2\n0 0\n[NumSegments] = 0',
            '[NumPoints] = -1',
            '[NumPoints] = 50001',
            '[NumPoints] = 1\nnan 0',
            '[NumPoints] = 2\n0 0\n1 0\n[NumSegments] = 1\n0 2',
            '[NumPoints] = 2\n0 0\n1 0\n[NumArcSegments] = 1\n0 1 360 5',
            NATIVE.replace('1.5 1000','bad 1000'),
            NATIVE.replace('1.5 1000','nan 1000'),
            NATIVE.replace('<BHPoints> = 2','<BHPoints> = -1'),
        ]
        for value in invalid:
            with self.subTest(value=value[:80]), self.assertRaises(ValueError): parse_fem(value)

    def test_exports_are_self_contained_and_execute_without_repository_imports(self):
        source=export_recipe('slotted_stator',{'slot_count':9,'slot_opening':2,'slot_width':5},'python')
        self.assertNotIn('from femm_brain',source)
        compile(source,'geometry_model.py','exec')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); script=root/'geometry_model.py'; script.write_text(source)
            params=root/'parameters.json'; params.write_text(json.dumps({'slot_count':6,'slot_opening':2,'slot_width':5}))
            result=subprocess.run([sys.executable,str(script),'--write-only','--parameters',str(params),'--output',str(root/'out')],cwd=directory,capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)
            parsed=parse_fem((root/'out/geometry.fem').read_text())
            self.assertEqual(len(parsed['segments']),4*6)
            lua=(root/'out/geometry.lua').read_text()
            self.assertIn('mi_addarc',lua)
            self.assertNotIn('mi_analyze',lua)
            self.assertNotIn('mo_blockintegral',lua)
            ET.fromstring((root/'out/geometry.svg').read_text())
        with self.assertRaises(ValueError): export_recipe('circle',format='shell')


if __name__=='__main__':
    unittest.main()
