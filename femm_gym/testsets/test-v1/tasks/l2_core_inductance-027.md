# Inductance of a gapped C-core inductor

Build this in FEMM 5 and save it as `submission/model.fem`: a magnetics problem at 0 Hz, planar, lengths in millimetres, depth 41 mm.

## Device (x and y in mm)

- The core is `Steel`: the rectangle |x| <= 49, |y| <= 31, minus its window |x| <= 34.5,
  |y| <= 16.5, and minus an air gap that cuts the right limb: 34.5 <= x <= 49,
  |y| <= 0.35.
- A coil of 140 turns is wound on the left limb. Its two sides are `Copper` rectangles with
  |y| <= 9.7: one inside the window, -33.5 <= x <= -9.2, and one
  outside the core, -74.3 <= x <= -50. In FEMM terms the inside side carries +140
  turns of circuit `coil` and the outside side -140.
- Everything else is `Air`. Model the surroundings as open (unbounded) space, by a method of your choice.

## Materials (linear, isotropic, not laminated)

- `Steel`: relative permeability 500, conductivity 0 MS/m
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
