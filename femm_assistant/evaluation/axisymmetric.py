"""Independent references for linear axisymmetric DC magnetostatics.

The winding is a uniformly filled rectangular r-z cross-section, revolved
through 360 degrees. Its on-axis field integrates the Biot-Savart circular-loop
field analytically over both r and z. This is not the infinite/thin solenoid
approximation and does not reuse any FEMM field or fitting parameters.

Circular-loop derivation: OpenStax University Physics 2, section 12.4:
https://openstax.org/books/university-physics-volume-2/pages/12-4-magnetic-field-of-a-current-loop
FEMM coordinate/integral semantics:
https://www.femm.info/doku/doku.php?id=axisymmetricformulation
"""
from __future__ import annotations

import math

from app.plan import GeometryPlan

MU0 = 4 * math.pi * 1e-7


def rectangular_winding_axis_bz(*, r_inner_mm: float, r_outer_mm: float,
                                 z_min_mm: float, z_max_mm: float, turns: int,
                                 current_a: float, z_mm: float) -> float:
    """Exact Bz in tesla on the axis of an air winding of uniform turn density.

    Each differential turn contributes mu0*I*r^2/[2*(r^2+dz^2)^1.5].
    Integrating z first gives u/sqrt(r^2+u^2), and integrating radius gives
    u*asinh(r/abs(u)). The u=0 limit is exactly zero.
    """
    r1, r2, z1, z2, z = [float(value) * 1e-3 for value in
                         (r_inner_mm, r_outer_mm, z_min_mm, z_max_mm, z_mm)]
    if not all(math.isfinite(value) for value in (r1, r2, z1, z2, z, current_a)):
        raise ValueError("Reference parameters must be finite")
    if not 0 < r1 < r2 or not z1 < z2:
        raise ValueError("Reference winding needs 0 < inner radius < outer radius and positive axial length")

    def primitive(u):
        return 0. if u == 0 else u * (math.asinh(r2 / abs(u)) - math.asinh(r1 / abs(u)))

    return MU0 * turns * current_a * (primitive(z-z1)-primitive(z-z2)) / (2*(r2-r1)*(z2-z1))


def air_solenoid_plan(*, current_a: float = 1, mesh_mm: float = 4,
                      winding_mesh_mm: float = .5, domain_mm: float = 160,
                      nearfield_mesh_mm: float | None = None) -> GeometryPlan:
    """Finite air solenoid, with a visible r=0 axis and three on-axis probes.

    Optional nested air contours refine the probe neighbourhood without
    introducing a material interface or changing the current distribution.
    The exterior domain remains an independent convergence parameter.
    """
    data = {
        "kind": "magnetostatic_plan", "title": "Finite axisymmetric air solenoid",
        "units": "millimeters", "physics": "axisymmetric_dc_magnetics", "depth_mm": None,
        "mesh_mm": mesh_mm,
        "domain": {"type": "rectangle", "x_min": 0, "x_max": domain_mm,
                   "y_min": -domain_mm, "y_max": domain_mm},
        "air_label": {"x": domain_mm*.75, "y": domain_mm*.75},
        "materials": [{"id": "winding", "relative_permeability": 1}],
        "circuits": [{"id": "coil", "current_a": current_a}],
        "regions": [{"id": "coil", "shape": {"type": "rectangle", "x_min": 20,
                     "x_max": 30, "y_min": -20, "y_max": 20}, "material": "winding",
                     "label": {"x": 25, "y": 0}, "circuit": "coil", "turns": 200,
                     "mesh_mm": winding_mesh_mm}],
        "force_region": None,
        "probes": [{"x": 0, "y": z} for z in (0, 20, 50)]}
    if nearfield_mesh_mm is not None:
        data["regions"].append({"id": "nearfield_air", "shape": {"type": "rectangle", "x_min": 0,
                                "x_max": 60, "y_min": -80, "y_max": 80}, "material": "air",
                                "label": {"x": 50, "y": 60}, "mesh_mm": nearfield_mesh_mm})
    return GeometryPlan.model_validate(data)


def air_solenoid_reference(z_mm: float, current_a: float = 1) -> float:
    return rectangular_winding_axis_bz(r_inner_mm=20, r_outer_mm=30,
                                       z_min_mm=-20, z_max_mm=20,
                                       turns=200, current_a=current_a, z_mm=z_mm)
