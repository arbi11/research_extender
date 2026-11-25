# Inductance of a gapped C-core inductor

Build this in FEMM 5 and save it as `submission/model.fem`: a magnetics problem at 0 Hz, planar, lengths in millimetres, depth 31 mm.

## Device (x and y in mm)

- The core is `Steel`: the rectangle |x| <= 50, |y| <= 22.5, minus its window |x| <= 37.5,
  |y| <= 10, and minus an air gap that cuts the right limb: 37.5 <= x <= 50,
  |y| <= 1.45.
- A coil of 180 turns is wound on the left limb. Its two sides are `Copper` rectangles with
  |y| <= 6.7: one inside the window, -36.5 <= x <= -7.9, and one
  outside the core, -79.6 <= x <= -51. In FEMM terms the inside side carries +180
  turns of circuit `coil` and the outside side -180.
- Everything else is `Air`. Model the surroundings as open (unbounded) space, by a method of your choice.

## Materials (linear, isotropic, not laminated)

- `Steel`: relative permeability 2000, conductivity 0 MS/m
- `Copper`: relative permeability 1, conductivity 58 MS/m
- `Air`: relative permeability 1, conductivity 0 MS/m

## Scored

The coil's self-inductance, in henries.

## What to save

- `submission/model.fem`: your model. Name the winding's circuit `coil` (series circuit).
- `submission/answer.json` (optional, not scored): `{"value": <your result>, "unit": "H"}`.

## How it is scored

The scorer opens your saved model, sets circuit `coil` to 1 A and every other circuit
to 0 A, solves it with the mesh you saved, and measures the inductance from the circuit's flux linkage. Your own number is never
used, so the result depends on the model, including how fine its mesh is.
A solve that runs longer than 3 minutes scores 0.
