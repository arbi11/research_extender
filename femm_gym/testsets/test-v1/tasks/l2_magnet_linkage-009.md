# Flux linkage of a coil beside a ring magnet

Build this in FEMM 5 and save it as `submission/model.fem`: a magnetics problem at 0 Hz, axisymmetric about the z axis (r >= 0), lengths in millimetres.

## Device (r and z in mm)

- A ring magnet of `NdFeB`: 5.1 <= r <= 15.1,
  -2.1 <= z <= 2.1, magnetized along +z (remanence 1.15 T).
- A coil of `Copper` above it: 5.6 <= r <= 13,
  7.9 <= z <= 19.2, with 260 turns (positive).
- Everything else is `Air`. Model the surroundings as open (unbounded) space, by a method of your choice.

## Materials (linear, isotropic)

- `NdFeB`: relative permeability 1.05, conductivity 0 MS/m, coercivity 871563 A/m
- `Copper`: relative permeability 1, conductivity 58 MS/m
- `Air`: relative permeability 1, conductivity 0 MS/m

## Scored

The coil's flux linkage, in webers, with no current in the coil (the magnet's field alone).

## What to save

- `submission/model.fem`: your model. Name the winding's circuit `coil` (series circuit).
- `submission/answer.json` (optional, not scored): `{"value": <your result>, "unit": "Wb"}`.

## How it is scored

The scorer opens your saved model, sets circuit `coil` to 0 A and every other circuit
to 0 A, solves it with the mesh you saved, and measures the circuit's flux linkage. Your own number is never
used, so the result depends on the model, including how fine its mesh is.
A solve that runs longer than 3 minutes scores 0.
