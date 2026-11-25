# Pot core with its winding

Build this in FEMM 5 and save it as `submission/model.fem`: a magnetics problem at 0 Hz, axisymmetric about the z axis (r >= 0), lengths in millimetres.

## Geometry (r and z in mm)

- The core is `Ferrite`: the rectangle 0 <= r <= 19.6, -12.1 <= z <= 12.1, minus its winding window
  9.7 <= r <= 17.2, -8.3 <= z <= 8.3.
- Inside the window the winding is `Copper`: 10.1 <= r <= 16.8, -7.9 <= z <= 7.9.
  The rest of the window is `Air`.
- `Air` fills the rest of the region 0 <= r <= 39.2, -24.2 <= z <= 24.2.

## Materials (linear, isotropic, not laminated)

- `Ferrite`: relative permeability 2000, conductivity 0 MS/m
- `Copper`: relative permeability 1, conductivity 58 MS/m
- `Air`: relative permeability 1, conductivity 0 MS/m

## Boundary

A = 0 on the edges r = 39.2 and z = +/-24.2. The axis r = 0 needs no condition.

## How it is scored

The scorer opens your saved model, solves it with no currents, and checks which material lies
where at a few thousand points inside that region. Materials are recognised by their relative
permeability and conductivity, so give each one exactly the values above; names do not matter.
A solve that runs longer than 3 minutes scores 0.
