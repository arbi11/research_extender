from femm_solver import FEMMSolver
import numpy as np
import pandas as pd
from scipy.stats import qmc
from dataclasses import dataclass
import femm
from pathlib import Path

SAMPLES_TO_COLLECT = 500
SAVE_INTERVAL = 100

@dataclass
class CCoreParams:
    coil_current: float
    armature_gap: float
    core_width: float
    coil_turns: int


# Generate LHS samples
sampler = qmc.LatinHypercube(d=4, seed=42)
lhs_samples = sampler.random(n=SAMPLES_TO_COLLECT)

# Map to physical parameter ranges
c_core_params = []
for u in lhs_samples:
    params = CCoreParams(
        coil_current=-50 + (50 - (-50)) * u[0],
        armature_gap=0.5 + (5.0 - 0.5) * u[1],
        core_width=30 + (45 - 30) * u[2],
        coil_turns=int(300 + (700 - 300) * u[3])
    )
    c_core_params.append(params)

# Run FEMM simulations
femm.openfemm(1)  # 1 = hide FEMM window
solver = FEMMSolver()
data = []
current_dir = Path(__file__).parent.absolute()
output_file = current_dir / 'c_core_force_dataset.csv'

for i, params in enumerate(c_core_params):
    print(f"[{i+1}/{SAMPLES_TO_COLLECT}] I={params.coil_current:.1f}A, gap={params.armature_gap:.2f}mm, w={params.core_width:.1f}mm, N={params.coil_turns}")

    mesh_data = solver.analyze_c_core_armature(
        coil_current=params.coil_current,
        armature_gap=params.armature_gap,
        core_width=params.core_width,
        coil_turns=params.coil_turns
    )

    if mesh_data.success and mesh_data.forces is not None:
        data.append({
            'coil_current_a': params.coil_current,
            'armature_gap_mm': params.armature_gap,
            'core_width_mm': params.core_width,
            'coil_turns': params.coil_turns,
            'force_magnitude_n': mesh_data.forces['magnitude']
        })
        print(f"  → F={mesh_data.forces['magnitude']:.4f} N")
    else:
        print(f"  → Failed")

    # Intermittent save every SAVE_INTERVAL samples
    if (i + 1) % SAVE_INTERVAL == 0:
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"\n[CHECKPOINT] Saved {len(df)} samples to {output_file.name}\n")

solver.close_femm()

# Final save
df = pd.DataFrame(data)
df.to_csv(output_file, index=False)
print(f"\n[COMPLETE] Saved {len(df)} samples to {output_file.name}")
print(f"Force range: [{df['force_magnitude_n'].min():.4f}, {df['force_magnitude_n'].max():.4f}] N")
