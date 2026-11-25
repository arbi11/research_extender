# Axial force between a ring magnet and a coil

Build this in FEMM 5 and save it as `submission/model.fem`: a magnetics problem at 0 Hz, axisymmetric about the z axis (r >= 0), lengths in millimetres.

## Device (r and z in mm)

- A ring magnet of `NdFeB`: 4.9 <= r <= 15.7,
  -4.7 <= z <= 4.7, magnetized along +z (remanence 1.16 T).
- A coil of `Copper` above it: 14 <= r <= 27.8,
  12.6 <= z <= 16.1, with 430 turns (positive).
- Everything else is `Air`. Model the surroundings as open (unbounded) space, by a method of your choice.

## Materials (linear, isotropic)

- `NdFeB`: relative permeability 1.05, conductivity 0 MS/m, coercivity 879142 A/m
- `Copper`: relative permeability 1, conductivity 58 MS/m
- `Air`: relative permeability 1, conductivity 0 MS/m

## Scored

The axial (z) force on the magnet, in newtons, positive along +z, with 4.5 A in the coil.

## What to save

- `submission/model.fem`: your model. Name the winding's circuit `coil` (series circuit).
- `submission/answer.json` (optional, not scored): `{"value": <your result>, "unit": "N"}`.

## How it is scored

The scorer opens your saved model, sets circuit `coil` to 4.5 A and every other circuit
to 0 A, solves it with the mesh you saved, and measures the force on the magnet as minus the Lorentz force on the model's currents (the coil's). Your own number is never
used, so the result depends on the model, including how fine its mesh is.
A solve that runs longer than 3 minutes scores 0.
