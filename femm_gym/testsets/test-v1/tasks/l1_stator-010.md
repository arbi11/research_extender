# Stator lamination with 18 trapezoidal slots

Build this in FEMM 5 and save it as `submission/model.fem`: a magnetics problem at 0 Hz, planar, lengths in millimetres, depth 50 mm.

## Geometry

- The lamination is a ring of `Steel` centred on the origin, from the bore circle (radius
  54.5 mm) to the outer circle (radius 92.9 mm).
- It has 18 identical slots. Slot k (k = 0, 1, ..., 17) is symmetric about the
  ray at (k + 1/2) x 20 degrees from the +x axis, so a tooth is centred on the +x axis.
- In each slot's own frame (x outward along its centre line from the origin, y across it):
  - the opening is the strip |y| <= 2.25 mm from the bore circle out to
    x = 55.5 mm. It is air, part of the bore;
  - the slot body is the trapezoid with corners (x, y) = (55.5, +/-3.5) mm and
    (76.1, +/-5.15) mm: straight sides and a flat bottom. It is `Copper`.
- Inside the bore circle is `Air`.

## Materials (linear, isotropic, not laminated)

- `Steel`: relative permeability 2500, conductivity 0 MS/m
- `Copper`: relative permeability 1, conductivity 58 MS/m
- `Air`: relative permeability 1, conductivity 0 MS/m

## Boundary

A = 0 on the outer circle. Nothing lies outside it.

## How it is scored

The scorer opens your saved model, solves it with no currents, and checks which material lies
where at a few thousand points inside the outer circle. Materials are recognised by their relative
permeability and conductivity, so give each one exactly the values above; names do not matter.
A solve that runs longer than 3 minutes scores 0.
