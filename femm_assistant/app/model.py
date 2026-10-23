"""Validated SI/metric inputs for the first supported model family."""
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal


class CCoreSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    geometry: Literal["c_core"] = "c_core"
    width_mm: float = Field(default=60, ge=10, le=200)
    height_mm: float = Field(default=60, ge=10, le=200)
    limb_mm: float = Field(default=10, ge=2, le=40)
    gap_mm: float = Field(default=1, ge=0.2, le=20)
    depth_mm: float = Field(default=20, ge=1, le=200)
    turns: int = Field(default=500, ge=1, le=10000, strict=True)
    current_a: float = Field(default=1, ge=0, le=100)
    relative_permeability: float = Field(default=1000, ge=1, le=10000)
    mesh_mm: float = Field(default=2, ge=0.5, le=5)

    @model_validator(mode="after")
    def feasible_geometry(self):
        if self.width_mm < 4 * self.limb_mm:
            raise ValueError("Width must be at least four limb thicknesses.")
        if self.height_mm < 4 * self.limb_mm:
            raise ValueError("Height must be at least four limb thicknesses.")
        return self


ASSUMPTIONS = [
    "2D planar, DC magnetostatics; depth is the out-of-plane stack length.",
    "Core and armature use a constant relative permeability: saturation, hysteresis, and losses are not modeled.",
    "The air gap is present at both pole faces; the winding has two opposite current cross-sections.",
    "Coil width is 0.4 × limb thickness; insulation clearance is 0.1 × limb thickness.",
    "A = 0 on an air box extending two device dimensions beyond the geometry.",
    "Results require mesh and outer-boundary convergence checks before engineering use.",
]
