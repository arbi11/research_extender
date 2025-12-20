import femm
import numpy as np
from scipy.stats import qmc
import os

# --- 1. Setup ---
N_SAMPLES = 100
AIRBOX_SIZE = 80  # Half-width/height, from -80 to 80 mm
LOWER_BOUNDS = [3, 5, -1, -1]  # R (mm), I (A), xc_norm, yc_norm
UPPER_BOUNDS = [15, 15, 1, 1]  # R (mm), I (A), xc_norm, yc_norm
DIMS = len(LOWER_BOUNDS)

# --- 2. Generate samples using Latin Hypercube Sampling ---
print(f"Generating {N_SAMPLES} samples using Latin Hypercube Sampling...")
sampler = qmc.LatinHypercube(d=DIMS, seed=42)
samples_norm = sampler.random(n=N_SAMPLES)
samples = qmc.scale(samples_norm, LOWER_BOUNDS, UPPER_BOUNDS)
print("Sample generation complete.")

# --- 3. Run FEMM for each sample ---
print("\nStarting FEMM simulation loop...")
femm.openfemm()

for i, sample in enumerate(samples):
    # Unpack and scale sample parameters
    R, I, xc_norm, yc_norm = sample
    max_c = AIRBOX_SIZE - R
    xc = xc_norm * max_c
    yc = yc_norm * max_c

    print(f"Running sample {i+1}/{N_SAMPLES}: R={R:.2f}, I={I:.2f}, xc={xc:.2f}, yc={yc:.2f}")

    # Create a new magnetics problem
    femm.newdocument(0)
    femm.mi_probdef(0, 'millimeters', 'planar', 1e-8, 0, 30)

    # --- Define Materials ---
    femm.mi_addmaterial('Air', 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    femm.mi_addmaterial('Copper', 1, 1, 0, 0, 58, 0, 0, 1, 3, 0, 0)

    # --- Define Circuit ---
    femm.mi_addcircprop('CoilCircuit', I, 1) # 1 for series

    # --- Draw Geometry ---
    # Airbox
    femm.mi_drawrectangle(-AIRBOX_SIZE, -AIRBOX_SIZE, AIRBOX_SIZE, AIRBOX_SIZE)

    # Coil (circle)
    femm.mi_addnode(xc + R, yc)
    femm.mi_addnode(xc - R, yc)
    femm.mi_addarc(xc + R, yc, xc - R, yc, 180, 10)
    femm.mi_addarc(xc - R, yc, xc + R, yc, 180, 10)

    # --- Add Block Labels ---
    # Air region (placed in a corner to avoid the coil)
    femm.mi_addblocklabel(-AIRBOX_SIZE*0.9, -AIRBOX_SIZE*0.9)
    femm.mi_setblockprop('Air', 1, 0, '<None>', 0, 0, 0)

    # Coil region
    femm.mi_addblocklabel(xc, yc)
    femm.mi_setblockprop('Copper', 1, 0, 'CoilCircuit', 0, 0, 1) # 1 turn

    # --- Boundary Conditions ---
    femm.mi_addboundprop('ZeroB', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    femm.mi_selectsegment(AIRBOX_SIZE, 0)
    femm.mi_selectsegment(-AIRBOX_SIZE, 0)
    femm.mi_selectsegment(0, AIRBOX_SIZE)
    femm.mi_selectsegment(0, -AIRBOX_SIZE)
    femm.mi_setsegmentprop('ZeroB', 0, 1, 0, 0, 0)
    femm.mi_clearselected()
    
    # --- Analyze ---
    filename = f"temp_coil_sample_{i:03d}.fem"
    femm.mi_saveas(os.path.join(os.getcwd(), filename))
    femm.mi_analyze(1) # 1 to hide analysis window
    femm.mi_loadsolution()

    print(f"  -> Solved and saved to {filename.replace('.fem', '.ans')}")
    
    femm.mo_close()
    femm.mi_close()

femm.closefemm()

print(f"\nFinished generating {N_SAMPLES} samples.")
