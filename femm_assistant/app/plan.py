"""Typed, bounded geometry plans for the deterministic FEMM harness.

Contours may be nested or disjoint. Crossings, touching contours and Boolean
operations are intentionally rejected: they need an explicit split/merge before
they can describe unambiguous FEMM faces. Geometry tests use exact circles.
"""
from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


EPS = 1e-7
MIN_FEATURE_MM = 0.01
Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")]
Coordinate = Annotated[float, Field(ge=-1000, le=1000)]


class PlanValue(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class Point(PlanValue):
    x: Coordinate
    y: Coordinate


class Rectangle(PlanValue):
    type: Literal["rectangle"]
    x_min: Coordinate
    y_min: Coordinate
    x_max: Coordinate
    y_max: Coordinate

    @model_validator(mode="after")
    def positive_dimensions(self):
        if min(self.x_max - self.x_min, self.y_max - self.y_min) < MIN_FEATURE_MM:
            raise ValueError("Rectangle width and height must each be at least 0.01 mm.")
        return self


class Circle(PlanValue):
    type: Literal["circle"]
    center: Point
    radius: float = Field(ge=MIN_FEATURE_MM, le=1000)

    @model_validator(mode="after")
    def bounded_extent(self):
        if max(abs(self.center.x), abs(self.center.y)) + self.radius > 1000:
            raise ValueError("The complete circle must fit within +/-1000 mm.")
        return self


class Polygon(PlanValue):
    type: Literal["polygon"]
    points: list[Point] = Field(min_length=3, max_length=128)

    @model_validator(mode="after")
    def simple_polygon(self):
        points = self.points
        n = len(points)
        for i, a in enumerate(points):
            b, c = points[(i + 1) % n], points[(i + 2) % n]
            if distance(a, b) < MIN_FEATURE_MM:
                raise ValueError("Polygon edges must be at least 0.01 mm; omit a repeated closing point.")
            if abs(cross(a, b, c)) <= EPS * max(distance(a, b), distance(b, c), 1):
                if (a.x - b.x) * (c.x - b.x) + (a.y - b.y) * (c.y - b.y) > 0:
                    raise ValueError("Adjacent polygon edges may not overlap or backtrack.")
        edges = list(pairs(points))
        for i, (a, b) in enumerate(edges):
            for j in range(i + 1, n):
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue
                if segments_intersect(a, b, *edges[j]):
                    raise ValueError("Polygon must be simple: non-adjacent edges may not cross or touch.")
        if polygon_area(points) < MIN_FEATURE_MM ** 2:
            raise ValueError("Polygon area must be at least 0.0001 square mm.")
        return self


Shape = Annotated[Rectangle | Circle | Polygon, Field(discriminator="type")]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def cross(a: Point, b: Point, c: Point) -> float:
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def pairs(points):
    return zip(points, points[1:] + points[:1])


def polygon_area(points) -> float:
    return abs(sum(a.x * b.y - b.x * a.y for a, b in pairs(points))) / 2


def segment_distance(point: Point, a: Point, b: Point) -> float:
    dx, dy = b.x - a.x, b.y - a.y
    norm = dx * dx + dy * dy
    if norm == 0:
        return distance(point, a)
    t = max(0., min(1., ((point.x - a.x) * dx + (point.y - a.y) * dy) / norm))
    return math.hypot(point.x - a.x - t * dx, point.y - a.y - t * dy)


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    if any(segment_distance(p, x, y) <= EPS for p, x, y in ((a, c, d), (b, c, d), (c, a, b), (d, a, b))):
        return True
    return ((cross(a, b, c) > 0) != (cross(a, b, d) > 0)
            and (cross(c, d, a) > 0) != (cross(c, d, b) > 0))


def shape_points(shape: Rectangle | Circle | Polygon) -> list[Point]:
    """Preview vertices only; topology and circle containment use exact tests."""
    if isinstance(shape, Rectangle):
        return [Point(x=shape.x_min, y=shape.y_min), Point(x=shape.x_max, y=shape.y_min),
                Point(x=shape.x_max, y=shape.y_max), Point(x=shape.x_min, y=shape.y_max)]
    if isinstance(shape, Circle):
        return [Point(x=shape.center.x + shape.radius * math.cos(i * math.tau / 64),
                      y=shape.center.y + shape.radius * math.sin(i * math.tau / 64)) for i in range(64)]
    return shape.points


def bounds(shape: Rectangle | Circle | Polygon) -> tuple[float, float, float, float]:
    if isinstance(shape, Rectangle):
        return shape.x_min, shape.x_max, shape.y_min, shape.y_max
    if isinstance(shape, Circle):
        return shape.center.x - shape.radius, shape.center.x + shape.radius, shape.center.y - shape.radius, shape.center.y + shape.radius
    points = shape.points
    return min(p.x for p in points), max(p.x for p in points), min(p.y for p in points), max(p.y for p in points)


def area(shape: Rectangle | Circle | Polygon) -> float:
    if isinstance(shape, Circle):
        return math.pi * shape.radius ** 2
    if isinstance(shape, Rectangle):
        return (shape.x_max - shape.x_min) * (shape.y_max - shape.y_min)
    return polygon_area(shape.points)


def point_location(shape: Rectangle | Circle | Polygon, point: Point) -> int:
    """Return +1 inside, 0 on the boundary, -1 outside a closed contour."""
    if isinstance(shape, Circle):
        delta = distance(point, shape.center) - shape.radius
        return 0 if abs(delta) <= EPS else (1 if delta < 0 else -1)
    points = shape_points(shape)
    if any(segment_distance(point, a, b) <= EPS for a, b in pairs(points)):
        return 0
    inside = False
    for a, b in pairs(points):
        if (a.y > point.y) != (b.y > point.y):
            if point.x < (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x:
                inside = not inside
    return 1 if inside else -1


def contains(shape: Rectangle | Circle | Polygon, point: Point, include_boundary=False) -> bool:
    return point_location(shape, point) >= (0 if include_boundary else 1)


def contours_intersect(first: Rectangle | Circle | Polygon, second: Rectangle | Circle | Polygon) -> bool:
    if isinstance(first, Circle) and isinstance(second, Circle):
        d = distance(first.center, second.center)
        return abs(first.radius - second.radius) - EPS <= d <= first.radius + second.radius + EPS
    if isinstance(first, Circle) or isinstance(second, Circle):
        circle, other = (first, second) if isinstance(first, Circle) else (second, first)
        return any(segment_distance(circle.center, a, b) <= circle.radius + EPS
                   and max(distance(circle.center, a), distance(circle.center, b)) >= circle.radius - EPS
                   for a, b in pairs(shape_points(other)))
    return any(segments_intersect(a, b, c, d) for a, b in pairs(shape_points(first)) for c, d in pairs(shape_points(second)))


def witness(shape: Rectangle | Circle | Polygon) -> Point:
    """A contour point used after the contours have been proved nonintersecting."""
    if isinstance(shape, Circle):
        return Point(x=shape.center.x + shape.radius, y=shape.center.y)
    return shape_points(shape)[0]


class Material(PlanValue):
    id: Identifier
    relative_permeability: float = Field(ge=0.01, le=100000)


class Circuit(PlanValue):
    id: Identifier
    current_a: float = Field(ge=-1000, le=1000)


class Region(PlanValue):
    id: Identifier
    shape: Shape
    material: Identifier
    label: Point
    circuit: Identifier | None = None
    turns: int = Field(default=0, ge=-10000, le=10000, strict=True)
    mesh_mm: float | None = Field(default=None, ge=0.05, le=25)

    @model_validator(mode="after")
    def excitation_matches_circuit(self):
        if self.circuit is None and self.turns != 0:
            raise ValueError("Nonzero turns require a named circuit.")
        if self.circuit is not None and self.turns == 0:
            raise ValueError("A circuit region requires nonzero signed turns.")
        return self


def axis_interior(shape: Rectangle | Circle | Polygon, point: Point) -> bool:
    """Whether a point is on the interior of a straight r=0 symmetry edge.

    The axis is a coordinate boundary, not a material interface. Its endpoints
    and any non-axis boundary remain excluded from field probes.
    """
    if point.x != 0 or isinstance(shape, Circle):
        return False
    return any(a.x == b.x == 0 and min(a.y, b.y) < point.y < max(a.y, b.y)
               for a, b in pairs(shape_points(shape)))


def contours_intersect_off_axis(first, second) -> bool:
    """Reject all contour intersections except contacts on the symmetry axis."""
    if isinstance(first, Circle) or isinstance(second, Circle):
        # Axis-tangent circles are rejected separately, so every circle contact
        # in a supported axisymmetric model occurs at positive radius.
        return contours_intersect(first, second)
    for a, b in pairs(shape_points(first)):
        for c, d in pairs(shape_points(second)):
            if not segments_intersect(a, b, c, d):
                continue
            denominator = (b.x-a.x)*(d.y-c.y) - (b.y-a.y)*(d.x-c.x)
            if abs(denominator) <= EPS:
                # Collinear overlap away from the axis remains ambiguous.
                if min(max(a.x, b.x), max(c.x, d.x)) > EPS:
                    return True
            else:
                t = ((c.x-a.x)*(d.y-c.y) - (c.y-a.y)*(d.x-c.x)) / denominator
                if a.x + t*(b.x-a.x) > EPS:
                    return True
    return False


def radial_witness(shape):
    """A contour point away from the axis for nesting and containment tests."""
    return max(shape_points(shape), key=lambda point: point.x)


class GeometryPlan(PlanValue):
    kind: Literal["magnetostatic_plan"] = "magnetostatic_plan"
    title: str = Field(min_length=1, max_length=160, pattern=r"^[^\x00-\x1f\x7f]+$")
    units: Literal["millimeters"] = "millimeters"
    physics: Literal["planar_dc_magnetics", "axisymmetric_dc_magnetics"] = "planar_dc_magnetics"
    depth_mm: float | None = Field(default=None, ge=0.1, le=1000)
    mesh_mm: float = Field(ge=0.05, le=25)
    domain: Shape
    air_label: Point
    materials: list[Material] = Field(max_length=16)
    circuits: list[Circuit] = Field(max_length=8)
    regions: list[Region] = Field(min_length=1, max_length=32)
    force_region: Identifier | None = None
    probes: list[Point] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def well_defined_faces(self):
        axisymmetric = self.physics == "axisymmetric_dc_magnetics"
        if axisymmetric:
            if self.depth_mm is not None:
                raise ValueError("Axisymmetric models use the full 360-degree volume; depth_mm must be null or omitted.")
            if isinstance(self.domain, Circle) or bounds(self.domain)[0] != 0:
                raise ValueError("An axisymmetric domain must be a rectangle or polygon with a straight boundary at r=0.")
            for shape in [self.domain] + [region.shape for region in self.regions]:
                if bounds(shape)[0] < 0:
                    raise ValueError("Axisymmetric coordinates use x=r, y=z in millimeters; every radius must be nonnegative.")
                if bounds(shape)[0] == 0:
                    if isinstance(shape, Circle) or not any(a.x == b.x == 0 for a, b in pairs(shape_points(shape))):
                        raise ValueError("A contour reaching r=0 must have a straight axis edge; isolated axis tangencies are unsupported.")
            if any(point.x < 0 for point in self.probes):
                raise ValueError("Axisymmetric field probes require nonnegative radius.")
        elif self.depth_mm is None:
            raise ValueError("Planar models require a positive out-of-plane depth_mm.")
        intersects = contours_intersect_off_axis if axisymmetric else contours_intersect
        contour_witness = radial_witness if axisymmetric else witness

        def unique(items, name):
            ids = [item.id for item in items]
            if len(set(ids)) != len(ids):
                raise ValueError(f"{name} ids must be unique.")
            return set(ids)

        material_ids = unique(self.materials, "Material")
        if "air" in material_ids:
            raise ValueError("The material id 'air' is reserved for the built-in background air.")
        material_ids.add("air")
        circuit_ids = unique(self.circuits, "Circuit")
        region_ids = unique(self.regions, "Region")
        shapes = [self.domain] + [r.shape for r in self.regions]
        vertex_count = sum(4 if isinstance(shape, Circle) else len(shape_points(shape)) for shape in shapes)
        if vertex_count > 512:
            raise ValueError("The plan exceeds the pilot limit of 512 contour control vertices.")
        for region in self.regions:
            if region.material not in material_ids:
                raise ValueError(f"Region '{region.id}' refers to an undefined material.")
            if region.circuit is not None and region.circuit not in circuit_ids:
                raise ValueError(f"Region '{region.id}' refers to an undefined circuit.")
            if intersects(self.domain, region.shape) or not contains(self.domain, contour_witness(region.shape)):
                raise ValueError(f"Region '{region.id}' must lie strictly inside the domain without touching its outer boundary (axisymmetric regions may share r=0).")
            if not contains(region.shape, region.label):
                raise ValueError(f"Label for region '{region.id}' must lie strictly inside its contour.")
        if circuit_ids != {r.circuit for r in self.regions if r.circuit is not None}:
            raise ValueError("Every defined circuit must be assigned to at least one region.")

        children = {r.id: [] for r in self.regions}
        for i, first in enumerate(self.regions):
            for second in self.regions[i + 1:]:
                if intersects(first.shape, second.shape):
                    raise ValueError(f"Contours '{first.id}' and '{second.id}' cross or touch. Split or merge them explicitly.")
                if contains(first.shape, contour_witness(second.shape)):
                    children[first.id].append(second)
                elif contains(second.shape, contour_witness(first.shape)):
                    children[second.id].append(first)
        for region in self.regions:
            if any(contains(child.shape, region.label, include_boundary=True) for child in children[region.id]):
                raise ValueError(f"Label for region '{region.id}' falls inside or on a nested contour instead of its own face.")
        if not contains(self.domain, self.air_label) or any(contains(r.shape, self.air_label, include_boundary=True) for r in self.regions):
            raise ValueError("The air label must be inside the domain and outside every region contour.")

        if self.force_region is not None:
            if self.force_region not in region_ids:
                raise ValueError("force_region must name an existing region.")
            selected = next(r for r in self.regions if r.id == self.force_region)
            parents = [r for r in self.regions if contains(r.shape, contour_witness(selected.shape)) and r.id != selected.id]
            parent = min(parents, key=lambda r: area(r.shape)) if parents else None
            if children[selected.id] or (parent is not None and parent.material != "air"):
                raise ValueError("Force extraction requires a single region without nested contours and surrounded by air.")

        for point in self.probes:
            in_domain = contains(self.domain, point) or (axisymmetric and axis_interior(self.domain, point))
            on_interface = any(point_location(region.shape, point) == 0
                               and not (axisymmetric and axis_interior(region.shape, point))
                               for region in self.regions)
            if not in_domain or on_interface:
                raise ValueError("Field probes must be inside the domain and away from material interfaces; r=0 symmetry-axis probes are allowed.")
        # Conservative area-over-size planning bound, not a promise of a final
        # element count. Native solver timeout and memory caps remain necessary.
        cells = area(self.domain) / self.mesh_mm ** 2
        cells += sum(area(r.shape) / (r.mesh_mm or self.mesh_mm) ** 2 for r in self.regions)
        if cells > 200000:
            raise ValueError("Requested mesh exceeds the pilot budget of approximately 200000 cells. Increase mesh_mm or reduce the domain.")
        return self


def wire_example() -> GeometryPlan:
    return GeometryPlan(
        title="Round conductor in air", depth_mm=20, mesh_mm=0.5,
        domain=Circle(type="circle", center=Point(x=0, y=0), radius=10),
        air_label=Point(x=5, y=0), materials=[Material(id="conductor", relative_permeability=1)],
        circuits=[Circuit(id="wire", current_a=1)],
        regions=[Region(id="wire", shape=Circle(type="circle", center=Point(x=0, y=0), radius=1),
                        material="conductor", label=Point(x=0, y=0), circuit="wire", turns=1)],
        probes=[Point(x=2, y=0)])


def from_ccore(spec) -> GeometryPlan:
    """Represent the existing example through the general plan vocabulary.

    The original C-core endpoint retains its established edge-refinement rules;
    this conversion is a starting point for generic geometry editing.
    """
    from .generate import geometry

    w, h, t, g = spec.width_mm, spec.height_mm, spec.limb_mm, spec.gap_mm
    margin = 2 * max(w, h)
    xmin, xmax, ymin, ymax = -t - margin, w + g + t + margin, -margin, h + margin
    labels = {"core": Point(x=t / 2, y=h / 2), "armature": Point(x=w + g + t / 2, y=h / 2),
              "coil_positive": Point(x=1.3 * t, y=h / 2), "coil_negative": Point(x=-.3 * t, y=h / 2)}
    return GeometryPlan(
        title="C-core actuator", depth_mm=spec.depth_mm, mesh_mm=4 * spec.mesh_mm,
        domain=Rectangle(type="rectangle", x_min=xmin, y_min=ymin, x_max=xmax, y_max=ymax),
        air_label=Point(x=xmin + margin / 2, y=ymin + margin / 2),
        materials=[Material(id="core", relative_permeability=spec.relative_permeability), Material(id="winding", relative_permeability=1)],
        circuits=[Circuit(id="coil", current_a=spec.current_a)],
        regions=[Region(id=name, shape=Polygon(type="polygon", points=[Point(x=x, y=y) for x, y in points]),
                        material="winding" if name.startswith("coil_") else "core", label=labels[name],
                        circuit="coil" if name.startswith("coil_") else None,
                        turns=spec.turns if name == "coil_positive" else (-spec.turns if name == "coil_negative" else 0),
                        mesh_mm=spec.mesh_mm) for name, points in geometry(spec).items()],
        force_region="armature")
