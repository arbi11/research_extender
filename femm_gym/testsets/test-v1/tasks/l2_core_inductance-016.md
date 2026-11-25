# Inductance of a gapped C-core inductor

Build this in FEMM 5 and save it as `submission/model.fem`: a magnetics problem at 0 Hz, planar, lengths in millimetres, depth 17 mm.

## Device (x and y in mm)

- The core is `Steel`: the rectangle |x| <= 46, |y| <= 50.5, minus its window |x| <= 30,
  |y| <= 34.5, and minus an air gap that cuts the right limb: 30 <= x <= 46,
  |y| <= 1.3.
- A coil of 310 turns is wound on the left limb. Its two sides are `Copper` rectangles with
  |y| <= 23.55: one inside the window, -29 <= x <= -5.5, and one
  outside the core, -70.5 <= x <= -47. In FEMM terms the inside side carries +310
  turns of circuit `coil` and the outside side -310.
- Everything else is `Air`. Model the surroundings as open (unbounded) space, by a method of your choice.

## Materials (linear, isotropic, not laminated)

- `Steel`: relative permeability 5000, conductivity 0 MS/m
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
