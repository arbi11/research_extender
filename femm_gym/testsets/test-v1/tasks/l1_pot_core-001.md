# Pot core with its winding

Build this in FEMM 5 and save it as `submission/model.fem`: a magnetics problem at 0 Hz, axisymmetric about the z axis (r >= 0), lengths in millimetres.

## Geometry (r and z in mm)

- The core is `Ferrite`: the rectangle 0 <= r <= 14.5, -14.2 <= z <= 14.2, minus its winding window
  3.8 <= r <= 11.7, -9.8 <= z <= 9.8.
- Inside the window the winding is `Copper`: 4.5 <= r <= 11, -9.1 <= z <= 9.1.
  The rest of the window is `Air`.
- `Air` fills the rest of the region 0 <= r <= 29, -28.4 <= z <= 28.4.

## Materials (linear, isotropic, not laminated)

- `Ferrite`: relative permeability 2000, conductivity 0 MS/m
- `Copper`: relative permeability 1, conductivity 58 MS/m
- `Air`: relative permeability 1, conductivity 0 MS/m

## Boundary

A = 0 on the edges r = 29 and z = +/-28.4. The axis r = 0 needs no condition.

## How it is scored

The scorer opens your saved model, solves it with no currents, and checks which material lies
where at a few thousand points inside that region. Materials are recognised by their relative
permeability and conductivity, so give each one exactly the values above; names do not matter.
A solve that runs longer than 3 minutes scores 0.
