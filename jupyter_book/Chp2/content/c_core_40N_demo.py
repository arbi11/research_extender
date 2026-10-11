"""
C-core electromagnet demo: armature pulled with ~40 N.

Runs FEMMSolver.analyze_c_core_armature at a design point found by bisection
on coil current (FEMM 5, Linux):
    500 turns, 5 mm armature gap, 150 mm core width, 16.50 A  ->  |F| ~ 40 N
    (steel: M-19 14 mil)

The model is planar with 1 mm depth, so the force is per 1 mm of stack depth.

Usage:
    python c_core_40N_demo.py            # headless solve, saves c_core_40N.png
    python c_core_40N_demo.py --gui      # also shows the model in the FEMM GUI
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from femm_solver import FEMMSolver

COIL_CURRENT = 16.50   # A
ARMATURE_GAP = 5.0     # mm
CORE_WIDTH = 150.0     # mm
COIL_TURNS = 500

gui = '--gui' in sys.argv
solver = FEMMSolver(debug_mode=gui)
solver.open_femm(hidden=not gui)
mesh = solver.analyze_c_core_armature(
    coil_current=COIL_CURRENT,
    armature_gap=ARMATURE_GAP,
    core_width=CORE_WIDTH,
    coil_turns=COIL_TURNS
)
if gui:
    # Reopen the solution so it stays visible in the GUI
    import femm
    femm.mi_loadsolution()
    femm.mo_showdensityplot(1, 0, 2.0, 0.0, 'bmag')
    input("FEMM GUI open - press Enter to close...")
solver.close_femm()

forces = mesh.forces
print(f"\nNI = {COIL_CURRENT * COIL_TURNS:.0f} A-turns")
print(f"Fx = {forces['Fx']:.2f} N, Fy = {forces['Fy']:.2f} N, |F| = {forces['magnitude']:.2f} N")

# Plot |B| and flux lines around the core
x, y = mesh.node_coords[:, 0], mesh.node_coords[:, 1]
A = mesh.node_fields[:, 0]
B = np.hypot(mesh.node_fields[:, 1], mesh.node_fields[:, 2])

fig, ax = plt.subplots(figsize=(8, 6.5))
cf = ax.tricontourf(x, y, mesh.elements, B, levels=40, cmap='viridis')
ax.tricontour(x, y, mesh.elements, A, levels=25, colors='white', linewidths=0.5)
fig.colorbar(cf, ax=ax, label='|B| [T]')

# Geometry outlines (same as femm_solver.analyze_c_core_armature)
arm_x = CORE_WIDTH + ARMATURE_GAP
outlines = [
    (0, 0, 50, 200), (50, 150, CORE_WIDTH - 50, 50), (50, 0, CORE_WIDTH - 50, 50),
    (arm_x, 0, 50, 200), (-20, 80, 20, 40), (50, 80, 20, 40)
]
for x0, y0, w, h in outlines:
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor='red', linewidth=1))

ax.annotate(f"|F| = {forces['magnitude']:.1f} N", xy=(arm_x + 25, 100),
            xytext=(arm_x - 90, 100), color='white', fontsize=12, ha='center',
            arrowprops=dict(arrowstyle='<-', color='white', lw=2))
ax.set_xlim(-60, arm_x + 100)
ax.set_ylim(-40, 240)
ax.set_aspect('equal')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_title(f"C-core: {COIL_TURNS} turns x {COIL_CURRENT} A, gap {ARMATURE_GAP} mm")
fig.tight_layout()
fig.savefig('c_core_40N.png', dpi=150)
print("Saved c_core_40N.png")
