"""Small, explicit starting models for the browser; no provider is required."""
from .plan import GeometryPlan


def axisymmetric_example():
    return GeometryPlan.model_validate({
        "kind": "magnetostatic_plan", "title": "Axisymmetric air coil",
        "physics": "axisymmetric_dc_magnetics", "units": "millimeters", "depth_mm": None,
        "mesh_mm": 4,
        "domain": {"type": "rectangle", "x_min": 0, "x_max": 160, "y_min": -160, "y_max": 160},
        "air_label": {"x": 120, "y": 120},
        "materials": [{"id": "winding", "relative_permeability": 1}],
        "circuits": [{"id": "coil", "current_a": 1}],
        "regions": [{"id": "coil", "shape": {"type": "rectangle", "x_min": 20,
            "x_max": 30, "y_min": -20, "y_max": 20}, "label": {"x": 25, "y": 0},
            "material": "winding", "circuit": "coil", "turns": 200, "mesh_mm": .5}],
        "probes": [{"x": 0, "y": z} for z in (0, 20, 50)], "force_region": None})
