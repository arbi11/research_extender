# Pot core with its winding

Build this in FEMM 5 and save it as `submission/model.fem`: a magnetics problem at 0 Hz, axisymmetric about the z axis (r >= 0), lengths in millimetres.

## Geometry (r and z in mm)

- The core is `Ferrite`: the rectangle 0 <= r <= 14, -13.1 <= z <= 13.1, minus its winding window
  3.3 <= r <= 9.6, -10.6 <= z <= 10.6.
- Inside the window the winding is `Copper`: 3.8 <= r <= 9.1, -10.1 <= z <= 10.1.
  The rest of the window is `Air`.
- `Air` fills the rest of the region 0 <= r <= 28, -26.2 <= z <= 26.2.

## Materials (linear, isotropic, not laminated)

- `Ferrite`: relative permeability 2000, conductivity 0 MS/m
- `Copper`: relative permeability 1, conductivity 58 MS/m
- `Air`: relative permeability 1, conductivity 0 MS/m

## Boundary

A = 0 on the edges r = 28 and z = +/-26.2. The axis r = 0 needs no condition.

## How it is scored

The scorer opens your saved model, solves it with no currents, and checks which material lies
where at a few thousand points inside that region. Materials are recognised by their relative
permeability and conductivity, so give each one exactly the values above; names do not matter.
A solve that runs longer than 3 minutes scores 0.
