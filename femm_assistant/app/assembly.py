"""Editable machine/component/feature graph compiled to the existing FEMM plan.

Families are descriptive strings, never a closed catalogue. Templates are ordinary
feature generators; custom and modified components use the same graph/compiler.
This module executes no model-generated code. Unsupported physics remain visible
in the assembly and preview, while native execution stays unavailable.
"""
from __future__ import annotations

import copy
import hashlib
import html
import json
import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .plan import Circle, GeometryPlan, Identifier, Point, Polygon, Rectangle, Shape, bounds, shape_points
from .targets import _UNITS, convert_units, unit_dimension


class AssemblyValue(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


FactId = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")]
SourceParameterId = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")]
SourceId = Annotated[str, Field(min_length=1)]


def capability_manifest() -> dict:
    return {"version": 1, "execution_physics": ["planar_dc_magnetics", "axisymmetric_dc_magnetics"],
        "material_laws": ["linear_isotropic"], "coordinate_units": "millimeters",
        "fact_units_by_role": {"permeability": ["1"], "count": ["1"], "turns": ["turn", "turns"],
            "length": ["m", "cm", "mm", "um", "in", "inch", "inches"],
            "mesh": ["m", "cm", "mm", "um", "in", "inch", "inches"],
            "current": ["A", "mA"], "angle": ["deg", "rad"], "text": ["text"]},
        "planar_depth": "explicit positive source/user value required",
        "axisymmetric": "r>=0, full revolution, finite winding cross-section; no planar depth",
        "geometry": ["rectangle", "circle", "simple_polygon", "closed_path"],
        "closed_path": "line/arc intent retained; arcs converted to polygons using an explicit sagitta tolerance",
        "relationships": ["disjoint", "nested", "shared_symmetry_axis"],
        "unsupported": ["zero_thickness_winding", "permanent_magnets", "nonlinear_bh", "ac", "full_3d", "touching_nonaxis_contours", "boolean_operations"],
        "resources": "No fixed topology-count or coordinate pilot caps; execution resources are reported by the running solver.",
        "templates": {"block": ["x_min", "x_max", "y_min", "y_max"],
                      "annular_coil": ["r_inner", "r_outer", "z_min", "z_max"]},
        "family_policy": "Unknown or novel families may contain custom components. A family label never establishes physical support.",
        "component_scopes": {"region": "Material face, including a genuine interior air cavity",
            "background": "The single exterior domain/background stored in model settings",
            "measurement": "Probe/force intent stored in model settings",
            "group": "Hierarchy and transforms without an owned material face",
            "intent": "Unresolved or unsupported intent retained without invented geometry"},
        "checks": ["fact provenance and dimensions", "feature dependencies", "deterministic geometry", "native mesh", "native solver", "frozen targets"]}


class Calculation(AssemblyValue):
    op: Literal["add", "subtract", "multiply", "divide", "negate", "scale"]
    inputs: list[FactId] = Field(min_length=1)
    factor: float | None = None


class PhysicalFact(AssemblyValue):
    id: FactId
    label: str = Field(min_length=1)
    role: Literal["length", "permeability", "current", "turns", "count", "angle", "mesh", "domain", "label", "other"]
    value: float | str | None
    unit: str | None = Field(default=None, max_length=24)
    status: Literal["paper", "user", "derived", "numerical_assumption", "missing", "conflict"]
    source_refs: list[SourceId] = Field(default_factory=list)
    depends_on: list[FactId] = Field(default_factory=list)
    calculation: Calculation | None = None
    source_parameter_id: SourceParameterId | None = None
    explanation: str = Field(default="", max_length=1200)

    @field_validator("value", mode="before")
    @classmethod
    def no_boolean(cls, value):
        if isinstance(value, bool):
            raise ValueError("A physical fact is not a boolean.")
        return value

    @model_validator(mode="after")
    def provenance(self):
        if self.status in {"missing", "conflict"}:
            if self.value is not None:
                raise ValueError("Missing or conflicting facts retain value=null; do not choose a value silently.")
            return self
        if self.value is None or self.unit is None:
            raise ValueError("A resolved fact requires an explicit value and unit.")
        if self.status in {"paper", "user"} and not self.source_refs:
            raise ValueError("Paper/user facts require source references.")
        if self.status == "numerical_assumption" and (self.role not in {"mesh", "domain", "label"} or not self.explanation):
            raise ValueError("Only disclosed mesh/domain/label choices may be numerical assumptions.")
        if self.status == "derived" and (self.calculation is None or not self.explanation):
            raise ValueError("Derived facts require a typed calculation and explanation.")
        if self.calculation is not None and self.status != "derived":
            raise ValueError("A calculation must be explicitly marked derived.")
        if isinstance(self.value, str):
            if self.unit != "text":
                raise ValueError("Text facts require unit=text.")
            return self
        dimension = unit_dimension(self.unit)
        expected = {"length": "length", "permeability": "dimensionless", "current": "current", "turns": "turns",
                    "count": "dimensionless", "angle": "angle", "mesh": "length", "domain": "length", "label": "length"}.get(self.role)
        if expected is not None and dimension != expected:
            raise ValueError("Fact unit does not match its physical role.")
        return self


class Transform(AssemblyValue):
    origin: Point = Field(default_factory=lambda: Point(x=0, y=0))
    rotate_deg: float = 0
    mirror: Literal["none", "x", "y"] = "none"
    dx_mm: float = 0
    dy_mm: float = 0
    fact_ids: list[FactId] = Field(default_factory=list)


class Pattern(AssemblyValue):
    count: int = Field(ge=1, strict=True)
    rotate_step_deg: float = 0
    dx_step_mm: float = 0
    dy_step_mm: float = 0
    origin: Point = Field(default_factory=lambda: Point(x=0, y=0))
    fact_ids: list[FactId] = Field(default_factory=list)


class LineEdge(AssemblyValue):
    kind: Literal["line"]
    end: Point


class ArcEdge(AssemblyValue):
    kind: Literal["arc"]
    end: Point
    sweep_deg: float = Field(gt=-360, lt=360)

    @model_validator(mode="after")
    def nonzero(self):
        if abs(self.sweep_deg) < 1e-8:
            raise ValueError("An arc requires a nonzero signed sweep.")
        return self


class ClosedPath(AssemblyValue):
    type: Literal["closed_path"]
    start: Point
    edges: list[Annotated[LineEdge | ArcEdge, Field(discriminator="kind")]] = Field(min_length=2)
    tolerance_mm: float = Field(gt=0)

    @model_validator(mode="after")
    def closed(self):
        if math.hypot(self.edges[-1].end.x-self.start.x, self.edges[-1].end.y-self.start.y) > 1e-7:
            raise ValueError("A closed path must explicitly end at its start.")
        return self


FeatureShape = Annotated[Rectangle | Circle | Polygon | ClosedPath, Field(discriminator="type")]


class FieldBinding(AssemblyValue):
    path: str = Field(pattern=r"^/(?:shape|label|mesh_mm)(?:/[A-Za-z0-9_]+)*$")
    fact_id: FactId
    unit: Literal["mm", "deg"]


class GeometryFeature(AssemblyValue):
    id: Identifier
    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    shape: FeatureShape | None
    material_id: Identifier | None
    circuit_id: Identifier | None = None
    turns_ref: FactId | None = None
    label: Point | None = None
    mesh_mm: float | None = Field(default=None, gt=0)
    transform: Transform = Field(default_factory=Transform)
    pattern: Pattern | None = None
    fact_ids: list[FactId] = Field(default_factory=list)
    source_refs: list[SourceId] = Field(default_factory=list)
    bindings: list[FieldBinding] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    required_physics: list[str] = Field(default_factory=list)
    description: str = Field(default="", max_length=1200)


class ParameterReference(AssemblyValue):
    name: str = Field(min_length=1)
    fact_id: FactId


class TemplateSpec(AssemblyValue):
    kind: Literal["block", "annular_coil"]
    feature_id: Identifier
    parameter_refs: list[ParameterReference] = Field(min_length=4, max_length=4)
    material_id: Identifier
    circuit_id: Identifier | None = None
    turns_ref: FactId | None = None
    mesh_ref: FactId | None = None


class FeatureOverride(AssemblyValue):
    op: Literal["add", "replace", "remove"]
    feature_id: Identifier
    feature: GeometryFeature | None = None

    @model_validator(mode="after")
    def matches(self):
        if self.op == "remove" and self.feature is not None:
            raise ValueError("A removal has no replacement feature.")
        if self.op != "remove" and (self.feature is None or self.feature.id != self.feature_id):
            raise ValueError("An override must preserve the addressed feature id.")
        return self


class AssemblyComponent(AssemblyValue):
    id: Identifier
    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    family: str = Field(min_length=1)
    mode: Literal["template", "custom", "template_modified"]
    parent_id: Identifier | None = None
    depends_on: list[Identifier] = Field(default_factory=list)
    transform: Transform = Field(default_factory=Transform)
    template: TemplateSpec | None = None
    features: list[GeometryFeature] = Field(default_factory=list)
    overrides: list[FeatureOverride] = Field(default_factory=list)
    source_refs: list[SourceId] = Field(default_factory=list)
    required_physics: list[str] = Field(default_factory=list)
    description: str = Field(default="", max_length=1200)

    @model_validator(mode="after")
    def mode_contract(self):
        if self.mode == "custom":
            if self.template is not None or self.overrides:
                raise ValueError("A custom component contains features, not a hidden template.")
        elif self.template is None or self.features:
            raise ValueError("Template components require a template; changes use explicit feature overrides.")
        if self.mode == "template" and self.overrides:
            raise ValueError("Template overrides require mode=template_modified.")
        if len({feature.id for feature in self.features}) != len(self.features):
            raise ValueError("Component feature ids must be unique.")
        return self


class AssemblyMaterial(AssemblyValue):
    id: Identifier
    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    relative_permeability_ref: FactId | None = None
    property_refs: list[FactId] = Field(default_factory=list)
    source_refs: list[SourceId] = Field(default_factory=list)


class AssemblyCircuit(AssemblyValue):
    id: Identifier
    current_ref: FactId | None
    source_refs: list[SourceId] = Field(default_factory=list)


class MachineAssembly(AssemblyValue):
    version: Literal[1] = 1
    id: Identifier
    title: str = Field(min_length=1)
    input_mode: Literal["natural_language", "paper", "combined"]
    family: str = Field(min_length=1)
    physics: str = Field(min_length=1)
    required_physics: list[str] = Field(default_factory=list)
    facts: list[PhysicalFact] = Field(default_factory=list)
    components: list[AssemblyComponent] = Field(min_length=1)
    materials: list[AssemblyMaterial] = Field(default_factory=list)
    circuits: list[AssemblyCircuit] = Field(default_factory=list)
    depth_ref: FactId | None = None
    domain: Shape | None = None
    air_label: Point | None = None
    mesh_mm: float | None = Field(default=None, gt=0)
    probes: list[Point] = Field(default_factory=list)
    force_feature_id: str | None = Field(default=None, max_length=100)
    numerical_notes: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def stable_ids_and_parents(self):
        for items in (self.facts, self.components, self.materials, self.circuits):
            if len({item.id for item in items}) != len(items):
                raise ValueError("Assembly ids must be unique within their collection.")
        components = {item.id: item for item in self.components}
        def visit(identity, ancestry):
            if identity in ancestry:
                raise ValueError("Component parent/dependency relationships must be acyclic.")
            component = components.get(identity)
            if component is None:
                raise ValueError("Component dependencies must name existing components.")
            for parent in ([component.parent_id] if component.parent_id else []) + component.depends_on:
                visit(parent, ancestry | {identity})
        for identity in components:
            visit(identity, set())
        return self


class AssemblyIssue(AssemblyValue):
    code: str
    message: str
    path: str
    severity: Literal["error", "warning"] = "error"


class ExpandedFeature(AssemblyValue):
    feature_id: str
    component_id: str
    local_feature_id: str
    region_id: str
    shape: Shape
    material_id: str | None
    circuit_id: str | None
    turns: int
    label: Point | None
    mesh_mm: float | None
    fact_ids: list[str]
    source_refs: list[str]
    depends_on: list[str]
    parent_component_ids: list[str]
    approximation: dict | None = None


class AssemblyCompilation(AssemblyValue):
    assembly: MachineAssembly
    expanded_features: list[ExpandedFeature]
    plan: GeometryPlan | None
    issues: list[AssemblyIssue]
    dependency_graph: dict[str, list[str]]
    feature_map: dict[str, dict]
    execution_supported: bool


_DIMENSIONS = {"dimensionless": (), "turns": (), "length": (("L", 1),), "area": (("L", 2),),
 "current": (("A", 1),), "current_density": (("A", 1), ("L", -2)), "angle": (("angle", 1),)}


def _combine(left, right, sign=1):
    result = dict(left)
    for key, power in right:
        result[key] = result.get(key, 0) + sign*power
    return tuple(sorted((key, value) for key, value in result.items() if value))


def _resolved_facts(assembly, issues):
    facts = {fact.id: fact for fact in assembly.facts}
    resolved, active = {}, set()
    def resolve(identity):
        if identity in resolved:
            return resolved[identity]
        if identity in active:
            raise ValueError("Derived fact dependency cycle.")
        if identity not in facts:
            raise ValueError("Undefined physical fact: " + identity)
        fact = facts[identity]
        if fact.status in {"missing", "conflict"} or not isinstance(fact.value, (int, float)):
            raise ValueError("Physical fact is unresolved or not numeric: " + identity)
        active.add(identity)
        dimension, factor = _UNITS[fact.unit]
        dims = _DIMENSIONS.get(dimension, ((dimension, 1),))
        value = fact.value*factor
        for dependency in fact.depends_on:
            resolve(dependency)
        if fact.calculation:
            calc = fact.calculation
            inputs = [resolve(item) for item in calc.inputs]
            if calc.op in {"add", "subtract"}:
                if any(item[1] != inputs[0][1] for item in inputs):
                    raise ValueError("A sum/difference requires matching physical dimensions.")
                if calc.op == "subtract" and len(inputs) != 2:
                    raise ValueError("A difference requires two inputs.")
                actual = sum(item[0] for item in inputs) if calc.op == "add" else inputs[0][0]-inputs[1][0]
                actual_dims = inputs[0][1]
            elif calc.op in {"negate", "scale"}:
                if len(inputs) != 1 or (calc.op == "scale" and calc.factor is None):
                    raise ValueError("A scale/negation requires one input and an explicit scale factor.")
                actual = inputs[0][0] * (-1 if calc.op == "negate" else calc.factor)
                actual_dims = inputs[0][1]
            elif calc.op == "multiply":
                actual, actual_dims = 1., ()
                for number, units in inputs:
                    actual *= number
                    actual_dims = _combine(actual_dims, units)
            else:
                if len(inputs) != 2 or inputs[1][0] == 0:
                    raise ValueError("A ratio requires two inputs and a nonzero denominator.")
                actual, actual_dims = inputs[0][0]/inputs[1][0], _combine(inputs[0][1], inputs[1][1], -1)
            if actual_dims != dims or not math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("Derived fact value/unit disagrees with its typed calculation: " + identity)
        active.remove(identity)
        resolved[identity] = (value, dims)
        return resolved[identity]
    for fact in assembly.facts:
        if isinstance(fact.value, (int, float)):
            try:
                resolve(fact.id)
            except (ValueError, KeyError) as error:
                issues.append(AssemblyIssue(code="invalid_fact", path="/facts/"+fact.id, message=str(error)))
                active.clear()
    return facts, resolved


def _number(facts, resolved, identity, unit, role=None):
    if identity is None or identity not in facts or identity not in resolved:
        raise ValueError("An explicit resolved physical fact is required: " + str(identity))
    fact = facts[identity]
    if role is not None and fact.role != role:
        raise ValueError("Fact '"+identity+"' has the wrong role; expected " + role)
    return convert_units(fact.value, fact.unit, unit)


def _template_features(component, facts, resolved):
    template = component.template
    refs = {item.name: item.fact_id for item in template.parameter_refs}
    expected = capability_manifest()["templates"][template.kind]
    if len(refs) != 4 or set(refs) != set(expected):
        raise ValueError("Template parameter names must be exactly: " + ", ".join(expected))
    values = [_number(facts, resolved, refs[name], "mm", "length") for name in expected]
    if template.kind == "annular_coil":
        if values[0] < 0:
            raise ValueError("An annular winding's inner radius must be nonnegative.")
        xmin, xmax, ymin, ymax = values
    else:
        xmin, xmax, ymin, ymax = values
    shape = Rectangle(type="rectangle", x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax)
    all_refs = list(refs.values())
    if template.turns_ref:
        all_refs.append(template.turns_ref)
    mesh = _number(facts, resolved, template.mesh_ref, "mm", "mesh") if template.mesh_ref else None
    return [GeometryFeature(id=template.feature_id, name=component.name, role=component.role, shape=shape,
        material_id=template.material_id, circuit_id=template.circuit_id, turns_ref=template.turns_ref,
        label=Point(x=(xmin+xmax)/2, y=(ymin+ymax)/2), mesh_mm=mesh,
        fact_ids=all_refs + ([template.mesh_ref] if template.mesh_ref else []), source_refs=component.source_refs)]


def _arc_polygon(path):
    points = [path.start]
    current = path.start
    max_sagitta = 0.
    has_arc = False
    for edge in path.edges:
        if edge.kind == "line":
            points.append(edge.end)
        else:
            has_arc = True
            dx, dy = edge.end.x-current.x, edge.end.y-current.y
            chord = math.hypot(dx, dy)
            theta = math.radians(edge.sweep_deg)
            if chord < 1e-9:
                raise ValueError("Arc endpoints must differ; use multiple arcs for a full circle.")
            offset = chord/(2*math.tan(theta/2))
            cx = (current.x+edge.end.x)/2 - dy/chord*offset
            cy = (current.y+edge.end.y)/2 + dx/chord*offset
            radius = math.hypot(current.x-cx, current.y-cy)
            if 1-path.tolerance_mm/radius == 1:
                raise ValueError("Curve tolerance is below floating-point resolution for this radius; no coarser curve was substituted.")
            max_step = 2*math.acos(max(-1., min(1., 1-path.tolerance_mm/radius)))
            count = max(1, math.ceil(abs(theta)/max(max_step, 1e-12)))
            angle = math.atan2(current.y-cy, current.x-cx)
            for index in range(1, count+1):
                if index == count:
                    points.append(edge.end)
                else:
                    a = angle+theta*index/count
                    points.append(Point(x=cx+radius*math.cos(a), y=cy+radius*math.sin(a)))
            max_sagitta = max(max_sagitta, radius*(1-math.cos(abs(theta)/(2*count))))
        current = edge.end
    if math.hypot(points[-1].x-points[0].x, points[-1].y-points[0].y) <= 1e-7:
        points.pop()
    shape = Polygon(type="polygon", points=points)
    approximation = {"kind": "arc_polyline", "tolerance_mm": path.tolerance_mm,
                     "max_sagitta_mm": max_sagitta, "vertices": len(points)} if has_arc else None
    return shape, approximation


def _point(point, transform):
    x, y = point.x-transform.origin.x, point.y-transform.origin.y
    if transform.mirror == "x":
        y = -y
    elif transform.mirror == "y":
        x = -x
    angle = math.radians(transform.rotate_deg)
    return Point(x=transform.origin.x+x*math.cos(angle)-y*math.sin(angle)+transform.dx_mm,
                 y=transform.origin.y+x*math.sin(angle)+y*math.cos(angle)+transform.dy_mm)


def _transform_shape(shape, transforms):
    def apply(point):
        for transform in transforms:
            point = _point(point, transform)
        return point
    if isinstance(shape, Circle):
        return Circle(type="circle", center=apply(shape.center), radius=shape.radius)
    points = [apply(point) for point in shape_points(shape)]
    if isinstance(shape, Rectangle) and all(abs(transform.rotate_deg % 90) < 1e-9 for transform in transforms):
        return Rectangle(type="rectangle", x_min=min(point.x for point in points), x_max=max(point.x for point in points),
                         y_min=min(point.y for point in points), y_max=max(point.y for point in points))
    return Polygon(type="polygon", points=points)


def _component_ancestors(component, components):
    result = []
    while component.parent_id:
        component = components[component.parent_id]
        result.append(component)
    return result


def _get_path(value, path):
    for key in path.strip("/").split("/"):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


class FeatureBindingError(ValueError):
    """One deterministic binding error, shared by staging and compilation."""
    def __init__(self, message, index, path):
        super().__init__(message)
        self.index, self.path = index, path


def validate_feature_bindings(feature, facts, resolved):
    """Check the addressed numeric field and exact converted source value.

    This is independent of native meshing and requires no semantic approval.
    Keeping it here lets a staged feature fail before accepting its checkpoint
    while retaining the same checks for saved or manually edited assemblies.
    """
    raw = feature.model_dump(mode="json")
    for index, binding in enumerate(feature.bindings):
        try:
            expected = _number(facts, resolved, binding.fact_id, binding.unit)
            actual = _get_path(raw, binding.path)
            if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Feature coordinate disagrees with its source fact binding: " + binding.path)
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise FeatureBindingError(str(error), index, binding.path) from None


def compile_assembly(value: MachineAssembly | dict) -> AssemblyCompilation:
    assembly = value if isinstance(value, MachineAssembly) else MachineAssembly.model_validate(value)
    issues, expanded, graph, feature_map = [], [], {}, {}
    facts, resolved = _resolved_facts(assembly, issues)
    components = {component.id: component for component in assembly.components}
    material_lookup = {item.id: item for item in assembly.materials}
    circuit_lookup = {item.id: item for item in assembly.circuits}
    supported = set(capability_manifest()["execution_physics"] + capability_manifest()["material_laws"])
    required = set(assembly.required_physics) | {assembly.physics}
    for component in assembly.components:
        required.update(component.required_physics)
        graph["component:"+component.id] = ["component:"+item for item in component.depends_on + ([component.parent_id] if component.parent_id else [])]
    for fact in assembly.facts:
        graph["fact:"+fact.id] = ["fact:"+item for item in fact.depends_on + (fact.calculation.inputs if fact.calculation else [])]
    for component in assembly.components:
        try:
            features = list(component.features) if component.mode == "custom" else _template_features(component, facts, resolved)
            by_id = {feature.id: feature for feature in features}
            for override in component.overrides:
                exists = override.feature_id in by_id
                if (override.op == "add" and exists) or (override.op != "add" and not exists):
                    raise ValueError("Feature override does not match an existing/addable stable id: " + override.feature_id)
                if override.op == "remove":
                    del by_id[override.feature_id]
                else:
                    by_id[override.feature_id] = override.feature
            features = list(by_id.values())
            ancestors = _component_ancestors(component, components)
            for feature in features:
                required.update(feature.required_physics)
                identity = component.id+"."+feature.id
                try:
                    if feature.shape is None:
                        raise ValueError("Feature geometry is unresolved; its intent remains in the assembly.")
                    assignment_refs = []
                    material = material_lookup.get(feature.material_id)
                    if material:
                        assignment_refs += material.property_refs + ([material.relative_permeability_ref] if material.relative_permeability_ref else [])
                    circuit = circuit_lookup.get(feature.circuit_id)
                    if circuit and circuit.current_ref:
                        assignment_refs.append(circuit.current_ref)
                    fact_ids = list(dict.fromkeys(feature.fact_ids + assignment_refs + feature.transform.fact_ids + component.transform.fact_ids
                        + [item for ancestor in ancestors for item in ancestor.transform.fact_ids]
                        + (feature.pattern.fact_ids if feature.pattern else []) + [binding.fact_id for binding in feature.bindings]
                        + ([feature.turns_ref] if feature.turns_ref else [])))
                    for fact_id in fact_ids:
                        if fact_id not in resolved:
                            raise ValueError("Feature refers to an unresolved/undefined physical fact: " + fact_id)
                    if not fact_ids and not feature.source_refs and not component.source_refs:
                        raise ValueError("A feature needs physical fact or source references; geometry has no memory defaults.")
                    validate_feature_bindings(feature, facts, resolved)
                    shape, approximation = _arc_polygon(feature.shape) if isinstance(feature.shape, ClosedPath) else (feature.shape, None)
                    turns = _number(facts, resolved, feature.turns_ref, "turns", "turns") if feature.circuit_id else 0
                    if feature.turns_ref and not feature.circuit_id:
                        raise ValueError("Winding turns require a circuit.")
                    if not float(turns).is_integer():
                        raise ValueError("Current FEMM execution requires signed integer turns.")
                    if feature.pattern:
                        graph["feature:"+identity] = ["component:"+component.id] + ["fact:"+item for item in fact_ids] + ["feature:"+item for item in feature.depends_on]
                    for index in range(feature.pattern.count if feature.pattern else 1):
                        fid = identity + ("@"+f"{index:03d}" if feature.pattern else "")
                        transforms = [feature.transform]
                        if feature.pattern:
                            transforms.append(Transform(origin=feature.pattern.origin, rotate_deg=index*feature.pattern.rotate_step_deg,
                                dx_mm=index*feature.pattern.dx_step_mm, dy_mm=index*feature.pattern.dy_step_mm))
                        transforms += [component.transform] + [ancestor.transform for ancestor in ancestors]
                        transformed = _transform_shape(shape, transforms)
                        label = feature.label
                        if label is not None:
                            for transform in transforms:
                                label = _point(label, transform)
                        rid = "r_"+hashlib.sha256(fid.encode()).hexdigest()[:22]
                        dependencies = (["feature:"+identity] if feature.pattern else ["component:"+component.id] + ["fact:"+item for item in fact_ids] + ["feature:"+item for item in feature.depends_on])
                        graph["feature:"+fid] = dependencies
                        item = ExpandedFeature(feature_id=fid, component_id=component.id, local_feature_id=feature.id,
                            region_id=rid, shape=transformed, material_id=feature.material_id, circuit_id=feature.circuit_id,
                            turns=int(turns), label=label, mesh_mm=feature.mesh_mm, fact_ids=fact_ids,
                            source_refs=list(dict.fromkeys(component.source_refs+feature.source_refs)), depends_on=dependencies,
                            parent_component_ids=[ancestor.id for ancestor in ancestors], approximation=approximation)
                        expanded.append(item)
                        feature_map[fid] = {key: val for key, val in item.model_dump(mode="json").items() if key not in {"shape", "material_id", "circuit_id", "turns", "label", "mesh_mm"}}
                except (ValueError, KeyError, IndexError, TypeError) as error:
                    issues.append(AssemblyIssue(code="invalid_feature", path="/components/"+component.id+"/features/"+feature.id, message=str(error)))
        except (ValueError, KeyError) as error:
            issues.append(AssemblyIssue(code="invalid_component", path="/components/"+component.id, message=str(error)))
    for physics in sorted(required-supported):
        issues.append(AssemblyIssue(code="unsupported_physics", path="/required_physics", message="Assembly describes '"+physics+"', which current native execution does not support."))
    for component in assembly.components:
        if component.template and component.template.kind == "annular_coil" and assembly.physics != "axisymmetric_dc_magnetics":
            issues.append(AssemblyIssue(code="template_coordinate_system", path="/components/"+component.id, message="annular_coil is an axisymmetric winding cross-section; it cannot silently become a planar extrusion."))
    for node, dependencies in list(graph.items()):
        for dependency in dependencies:
            if dependency not in graph:
                issues.append(AssemblyIssue(code="undefined_dependency", path=node, message="Undefined dependency: "+dependency))
    active, visited = set(), set()
    def check_cycle(node):
        if node in active:
            raise ValueError("Feature/fact dependency graph contains a cycle: "+node)
        if node in visited or node not in graph:
            return
        active.add(node)
        for dependency in graph[node]:
            check_cycle(dependency)
        active.remove(node)
        visited.add(node)
    try:
        for node in graph:
            check_cycle(node)
    except ValueError as error:
        issues.append(AssemblyIssue(code="dependency_cycle", path="/components", message=str(error)))
    materials, circuits = [], []
    for material in assembly.materials:
        try:
            if material.model != "linear_isotropic":
                raise ValueError("Material law '"+material.model+"' is described but is unsupported by current execution.")
            mu = _number(facts, resolved, material.relative_permeability_ref, "1", "permeability")
            materials.append({"id": material.id, "relative_permeability": mu})
        except ValueError as error:
            issues.append(AssemblyIssue(code="material_unavailable", path="/materials/"+material.id, message=str(error)))
    for circuit in assembly.circuits:
        try:
            circuits.append({"id": circuit.id, "current_a": _number(facts, resolved, circuit.current_ref, "A", "current")})
        except ValueError as error:
            issues.append(AssemblyIssue(code="circuit_unavailable", path="/circuits/"+circuit.id, message=str(error)))
    for item in expanded:
        for prefix, identities in (("material", [material.id for material in assembly.materials]+["air"]), ("circuit", [circuit.id for circuit in assembly.circuits])):
            identity = getattr(item, prefix+"_id")
            if prefix == "material" and identity is None or identity is not None and identity not in identities:
                issues.append(AssemblyIssue(code="undefined_assignment", path=item.feature_id, message="Undefined "+prefix+" assignment: "+str(identity)))
        if item.label is None:
            issues.append(AssemblyIssue(code="missing_label", path=item.feature_id, message="A region label must be explicitly supplied or deterministically generated by its template."))
    for question in assembly.questions:
        issues.append(AssemblyIssue(code="clarification_required", path="/questions", message=question))
    plan = None
    if not any(issue.severity == "error" for issue in issues):
        try:
            depth = _number(facts, resolved, assembly.depth_ref, "mm", "length") if assembly.physics == "planar_dc_magnetics" else None
            if assembly.physics == "axisymmetric_dc_magnetics" and assembly.depth_ref is not None:
                raise ValueError("Axisymmetric geometry uses full revolution; depth_ref must be null.")
            if assembly.domain is None or assembly.air_label is None or assembly.mesh_mm is None:
                raise ValueError("A declared numerical domain, air label and mesh size are required before native execution.")
            force = assembly.force_feature_id
            if force is not None:
                if force not in feature_map:
                    raise ValueError("force_feature_id must select one exact expanded feature instance.")
                force = feature_map[force]["region_id"]
            plan = GeometryPlan.model_validate({"title": assembly.title, "physics": assembly.physics, "depth_mm": depth,
                "mesh_mm": assembly.mesh_mm, "domain": assembly.domain.model_dump(mode="json"), "air_label": assembly.air_label.model_dump(),
                "materials": materials, "circuits": circuits,
                "regions": [{"id": item.region_id, "shape": item.shape.model_dump(mode="json"), "material": item.material_id,
                    "circuit": item.circuit_id, "turns": item.turns, "label": item.label.model_dump(), "mesh_mm": item.mesh_mm} for item in expanded],
                "probes": [point.model_dump() for point in assembly.probes], "force_region": force})
        except (ValueError, ValidationError) as error:
            issues.append(AssemblyIssue(code="native_plan_invalid", path="/", message=str(error)[:4000]))
    return AssemblyCompilation(assembly=assembly, expanded_features=expanded, plan=plan, issues=issues,
        dependency_graph=graph, feature_map=feature_map, execution_supported=plan is not None)


def preview_assembly(value: MachineAssembly | AssemblyCompilation | dict) -> str:
    compilation = value if isinstance(value, AssemblyCompilation) else compile_assembly(value)
    items = compilation.expanded_features
    if not items:
        return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 100"><text x="10" y="30">Assembly retained; feature coordinates are unresolved.</text></svg>'
    extent = [bounds(item.shape) for item in items]
    xmin, xmax = min(x[0] for x in extent), max(x[1] for x in extent)
    ymin, ymax = min(x[2] for x in extent), max(x[3] for x in extent)
    margin = max(xmax-xmin, ymax-ymin, 1)*.08
    width, height = xmax-xmin+2*margin, ymax-ymin+2*margin
    esc = lambda value: html.escape(str(value), quote=True)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{xmin-margin} {-ymax-margin} {width} {height+margin}" role="img">',
             '<title>'+esc(compilation.assembly.title)+' — assembly geometry preview</title>',
             '<desc>Preview includes unsupported components. Native execution status is reported separately. Curved paths retain their original intent in the assembly and use the declared polygon approximation tolerance here.</desc>']
    colors = ["#b4d4ca", "#e6c09a", "#aabfd9", "#d0b4d6"]
    for index, item in enumerate(items):
        shape = item.shape
        attrs = f'fill="{colors[index%len(colors)]}" fill-opacity="0.55" stroke="#214943" stroke-width="{max(width,height)/500}" data-feature-id="{esc(item.feature_id)}"'
        title = '<title>'+esc(item.feature_id)+((' (curve tolerance '+str(item.approximation['tolerance_mm'])+' mm)') if item.approximation else '')+'</title>'
        if isinstance(shape, Circle):
            parts.append(f'<circle cx="{shape.center.x}" cy="{-shape.center.y}" r="{shape.radius}" {attrs}>{title}</circle>')
        else:
            points = " ".join(f"{point.x},{-point.y}" for point in shape_points(shape))
            parts.append(f'<polygon points="{points}" {attrs}>{title}</polygon>')
    label = "Native execution available" if compilation.execution_supported else "Geometry preview — native execution unavailable; inspect assembly issues"
    parts.append(f'<text x="{xmin-margin}" y="{-ymin+margin*1.5}" font-size="{max(width,height)/40}" fill="#214943">{esc(label)}</text></svg>')
    return "".join(parts)


class AssemblyEdit(AssemblyValue):
    op: Literal["replace", "add", "remove"]
    path: str = Field(pattern=r"^/[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)*$")
    value: Any = None

    @model_validator(mode="after")
    def explicit_value(self):
        if self.op != "remove" and "value" not in self.model_fields_set:
            raise ValueError("Assembly add/replace requires an explicit value.")
        if self.op == "remove" and self.value is not None:
            raise ValueError("An assembly removal does not accept a replacement value.")
        if "id" in self.path.strip("/").split("/"):
            raise ValueError("Stable assembly ids are immutable.")
        return self


def apply_assembly_edits(value, edits, *, allowed_paths=None):
    """Apply scoped stable-id edits; return graph plus exact affected feature ids.

    No mutation of the caller's assembly. Facts/components/features are addressed
    by id, never collection index; point/edge coordinates retain numeric indices.
    """
    before = value if isinstance(value, MachineAssembly) else MachineAssembly.model_validate(value)
    raw = copy.deepcopy(before.model_dump(mode="json"))
    changes = [item if isinstance(item, AssemblyEdit) else AssemblyEdit.model_validate(item) for item in edits]
    paths = [item.path for item in changes]
    if len(paths) != len(set(paths)) or any(a != b and (a.startswith(b+"/") or b.startswith(a+"/")) for a in paths for b in paths):
        raise ValueError("Assembly edits must not overlap.")
    def key_for(collection, name, add=False):
        if isinstance(collection, list):
            if (not collection and not name.isdigit()) or (collection and isinstance(collection[0], dict) and ("id" in collection[0] or "name" in collection[0])):
                field = "id" if not collection or "id" in collection[0] else "name"
                found = [index for index, item in enumerate(collection) if item.get(field) == name]
                if not found:
                    if add:
                        return len(collection)
                    raise ValueError("Unknown stable assembly id: "+name)
                return found[0]
            if not name.isdigit():
                raise ValueError("Coordinate/edge lists use explicit numeric indices.")
            return int(name)
        return name
    for change in changes:
        if allowed_paths is not None and not any(change.path == path or change.path.startswith(path+"/") for path in allowed_paths):
            raise ValueError("Assembly edit is outside the permitted recovery scope.")
        keys = change.path.strip("/").split("/")
        if keys[-1] in {"facts", "components", "materials", "circuits", "features"}:
            raise ValueError("Address named assembly members by stable id rather than replacing a whole collection.")
        current = raw
        for name in keys[:-1]:
            current = current[key_for(current, name)]
        key = key_for(current, keys[-1], change.op == "add")
        if change.op == "remove":
            del current[key]
        elif isinstance(current, list) and key == len(current):
            if change.op != "add" or not isinstance(change.value, dict) or change.value.get("id", change.value.get("name")) != keys[-1]:
                raise ValueError("An added named feature must match its addressed stable id.")
            current.append(change.value)
        else:
            if change.op == "add":
                raise ValueError("Add cannot overwrite an existing assembly member.")
            if isinstance(current, list) and isinstance(current[key], dict) and "id" in current[key] and (not isinstance(change.value, dict) or change.value.get("id") != current[key]["id"]):
                raise ValueError("A replacement must preserve its stable assembly id.")
            current[key] = change.value
    after = MachineAssembly.model_validate(raw)
    old, compiled = compile_assembly(before), compile_assembly(after)
    dirty = set()
    for path in paths:
        bits = path.strip("/").split("/")
        if bits[0] == "components" and len(bits) > 3 and bits[2] == "features":
            dirty.add("feature:"+bits[1]+"."+bits[3])
        elif bits[0] in {"facts", "components"} and len(bits) > 1:
            dirty.add(("fact:" if bits[0] == "facts" else "component:")+bits[1])
        else:
            dirty.update("feature:"+key for key in old.feature_map)
    graph = {**old.dependency_graph, **compiled.dependency_graph}
    while True:
        more = {node for node, deps in graph.items() if dirty.intersection(deps)}
        if more <= dirty:
            break
        dirty.update(more)
    return {"assembly": after, "compilation": compiled, "changed_paths": paths,
            "affected_feature_ids": sorted(node[len("feature:"):] for node in dirty if node.startswith("feature:"))}
