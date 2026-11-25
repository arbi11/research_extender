# Surface-magnet rotor with 8 arc magnets

Build this in FEMM 5 and save it as `submission/model.fem`: a magnetics problem at 0 Hz, planar, lengths in millimetres, depth 50 mm.

## Geometry

- The rotor core is a solid `Steel` disc of radius 28.5 mm centred on the origin.
- 8 arc magnets of `Magnet` sit on it, each from r = 28.5 mm to r = 31.3 mm and
  spanning 28.5 degrees. Magnet k (k = 0, 1, ..., 7) is centred on the ray at
  k x 45 degrees from the +x axis.
- Everything else inside the circle r = 32.7 mm is `Air`: the gaps between the magnets and
  the ring above them.

## Materials (linear, isotropic, not laminated)

- `Steel`: relative permeability 1000, conductivity 0 MS/m
- `Magnet`: relative permeability 1.05, conductivity 0.667 MS/m, coercivity 900000 A/m
- `Air`: relative permeability 1, conductivity 0 MS/m

Magnetize the magnets radially, alternating outward and inward; the direction is not scored here.

## Boundary

A = 0 on the circle r = 32.7 mm. Nothing lies outside it.

## How it is scored

The scorer opens your saved model, solves it with no currents, and checks which material lies
where at a few thousand points inside the circle r = 32.7 mm. Materials are recognised by their relative
permeability and conductivity, so give each one exactly the values above; names do not matter.
A solve that runs longer than 3 minutes scores 0.
