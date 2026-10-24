import copy
import math
import unittest

from pydantic import ValidationError

from app.model import CCoreSpec
from app.plan import (Circle, GeometryPlan, Point, Polygon, bounds, contains,
                      contours_intersect, from_ccore, point_location, wire_example)


class GeometryPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = wire_example().model_dump()

    def invalid(self, value, message=None):
        with self.assertRaises(ValidationError) as caught:
            GeometryPlan.model_validate(value)
        if message:
            self.assertIn(message, str(caught.exception))

    def test_round_wire_and_ccore_are_different_valid_plans(self):
        wire = GeometryPlan.model_validate_json(wire_example().model_dump_json())
        core = from_ccore(CCoreSpec())
        self.assertEqual(wire.regions[0].shape.type, "circle")
        self.assertEqual(core.regions[0].shape.type, "polygon")
        self.assertEqual(core.force_region, "armature")
        self.assertEqual(core.regions[-1].turns, -500)
        self.assertEqual(bounds(wire.domain), (-10, 10, -10, 10))

    def test_finite_typed_numeric_values_and_unknown_operations(self):
        for bad in (float("nan"), float("inf"), True, "2"):
            value = copy.deepcopy(self.plan)
            value["depth_mm"] = bad
            self.invalid(value)
        for field, value in (("lua", "os.execute('anything')"), ("physics", "3d_ac"), ("units", "meters")):
            data = copy.deepcopy(self.plan)
            data[field] = value
            self.invalid(data)
        for name in ('wire\";execute()', '../wire', 'wire\n', ''):
            value = copy.deepcopy(self.plan)
            value['regions'][0]['id'] = name
            self.invalid(value)

    def test_self_intersection_touching_and_backtracking_polygons(self):
        for points in (
            [(0, 0), (2, 2), (0, 2), (2, 0)],
            [(0, 0), (2, 0), (2, 2), (1, 0), (0, 2)],
            [(0, 0), (2, 0), (1, 0), (1, 2), (0, 2)],
            [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)],
        ):
            with self.subTest(points=points), self.assertRaises(ValidationError):
                Polygon(type='polygon', points=[Point(x=x, y=y) for x, y in points])
        # Straight-through collinear subdivisions are legitimate contour nodes.
        Polygon(type='polygon', points=[Point(x=x, y=y) for x, y in [(0, 0), (1, 0), (2, 0), (2, 2), (0, 2)]])

    def test_exact_circle_topology_catches_crossing_between_preview_vertices(self):
        first = Circle(type='circle', center=Point(x=0, y=0), radius=1)
        angle = math.pi / 64  # half a preview segment away from a cardinal node
        second = Circle(type='circle', center=Point(x=1.9999 * math.cos(angle), y=1.9999 * math.sin(angle)), radius=1)
        self.assertTrue(contours_intersect(first, second))
        tangent = Circle(type='circle', center=Point(x=2, y=0), radius=1)
        self.assertTrue(contours_intersect(first, tangent))
        concentric = Circle(type='circle', center=Point(x=0, y=0), radius=2)
        self.assertFalse(contours_intersect(first, concentric))
        self.assertTrue(contours_intersect(first, first))
        self.assertEqual(point_location(first, Point(x=1, y=0)), 0)
        self.assertFalse(contains(first, Point(x=1, y=0)))

    def test_circle_line_intersection_and_tangency(self):
        circle = Circle(type='circle', center=Point(x=0, y=0), radius=1)
        polygon = lambda points: Polygon(type='polygon', points=[Point(x=x, y=y) for x, y in points])
        self.assertTrue(contours_intersect(circle, polygon([(1, -2), (2, -2), (2, 2), (1, 2)])))
        self.assertTrue(contours_intersect(circle, polygon([(0, -2), (2, -2), (2, 2), (0, 2)])))
        self.assertFalse(contours_intersect(circle, polygon([(2, -2), (3, -2), (3, 2), (2, 2)])))
        self.assertFalse(contours_intersect(circle, polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)])))

    def test_regions_cannot_touch_cross_or_escape_domain(self):
        for center_x, radius in ((9, 1), (10, 1), (0, 11)):
            value = copy.deepcopy(self.plan)
            value['regions'][0]['shape'].update(center={'x': center_x, 'y': 0}, radius=radius)
            value['regions'][0]['label'] = {'x': center_x, 'y': 0}
            self.invalid(value, 'strictly inside the domain')
        value = copy.deepcopy(self.plan)
        value['regions'].append({'id': 'another', 'shape': {'type': 'circle', 'center': {'x': 2, 'y': 0}, 'radius': 1},
                                 'label': {'x': 2, 'y': 0}, 'material': 'air'})
        self.invalid(value, 'cross or touch')

    def test_concave_regions_require_labels_in_the_actual_face(self):
        shape = {'type': 'polygon', 'points': [{'x': x, 'y': y} for x, y in
                  [(-2, -2), (2, -2), (2, -1), (-1, -1), (-1, 1), (2, 1), (2, 2), (-2, 2)]]}
        value = copy.deepcopy(self.plan)
        value['regions'][0]['shape'] = shape
        self.invalid(value, 'strictly inside its contour')
        value['regions'][0]['label'] = {'x': -1.5, 'y': 0}
        value['probes'] = [{'x': 0, 'y': 0}]
        GeometryPlan.model_validate(value)

    def test_nested_contours_define_material_faces_and_holes(self):
        value = copy.deepcopy(self.plan)
        value['regions'].append({'id': 'shell', 'shape': {'type': 'circle', 'center': {'x': 0, 'y': 0}, 'radius': 3},
                                 'label': {'x': 2, 'y': 0}, 'material': 'air'})
        GeometryPlan.model_validate(value)
        value['regions'][1]['label'] = {'x': 0.5, 'y': 0}
        self.invalid(value, 'nested contour')
        value['regions'][1]['label'] = {'x': 1, 'y': 0}
        self.invalid(value, 'nested contour')

    def test_air_label_cannot_silently_label_a_hole_or_solid(self):
        for point in ({'x': 0, 'y': 0}, {'x': 1, 'y': 0}, {'x': 10, 'y': 0}, {'x': 11, 'y': 0}):
            value = copy.deepcopy(self.plan)
            value['air_label'] = point
            self.invalid(value, 'air label')

    def test_material_circuit_references_and_signed_turns(self):
        for field, wrong in (('material', 'unknown'), ('circuit', 'unknown'), ('turns', 0), ('turns', True), ('turns', 1.5)):
            value = copy.deepcopy(self.plan)
            value['regions'][0][field] = wrong
            self.invalid(value)
        value = copy.deepcopy(self.plan)
        value['circuits'].append({'id': 'unused', 'current_a': 1})
        self.invalid(value, 'assigned')
        value = copy.deepcopy(self.plan)
        value['materials'].append({'id': 'air', 'relative_permeability': 1})
        self.invalid(value, 'reserved')
        value = copy.deepcopy(self.plan)
        value['regions'][0]['turns'] = -1
        GeometryPlan.model_validate(value)

    def test_duplicate_ids_and_larger_valid_domains(self):
        for collection in ('materials', 'circuits', 'regions'):
            value = copy.deepcopy(self.plan)
            value[collection].append(copy.deepcopy(value[collection][0]))
            self.invalid(value, 'unique')
        value = copy.deepcopy(self.plan)
        value['domain']['radius'] = 1000
        self.assertEqual(GeometryPlan.model_validate(value).domain.radius, 1000)

    def test_force_region_requires_an_air_surrounding_face(self):
        value = copy.deepcopy(self.plan)
        value['force_region'] = 'wire'
        GeometryPlan.model_validate(value)
        value['force_region'] = 'missing'
        self.invalid(value, 'existing region')
        value['force_region'] = 'wire'
        value['regions'].append({'id': 'iron_shell', 'shape': {'type': 'circle', 'center': {'x': 0, 'y': 0}, 'radius': 3},
                                 'label': {'x': 2, 'y': 0}, 'material': 'conductor'})
        self.invalid(value, 'surrounded by air')

    def test_field_probe_positions_avoid_domain_and_material_boundaries(self):
        for point in ({'x': 10, 'y': 0}, {'x': 11, 'y': 0}, {'x': 1, 'y': 0}):
            value = copy.deepcopy(self.plan)
            value['probes'] = [point]
            self.invalid(value, 'Field probes')


if __name__ == '__main__':
    unittest.main()
