# Finite-element analysis of a 20-pole surface-magnet machine with a 24-slot concentrated winding

## Abstract

This paper describes a 24-slot, 20-pole surface permanent-magnet machine with a double-layer concentrated winding for an industrial servo drive. It gives the machine's dimensions, winding and materials, and two-dimensional finite-element results for the no-load back-EMF and the torque at the rated current.

## 1. Introduction

Fractional-slot concentrated windings give short end windings and low cogging torque, and they suit machines with many poles. Surface-mounted magnets keep the rotor simple. This paper records one such design in enough detail for its finite-element results to be reproduced.

## 2. Machine geometry

The stator has 24 parallel-sided teeth, each ending in a tooth tip whose underside is flat and at right angles to the tooth centre line; the slots between them have an arc-shaped bottom. The slot opening between adjacent tips is air, and each slot holds two coil sides side by side. The rotor carries 20 arc-shaped magnets bonded to the surface of a cylindrical rotor core, with no retaining sleeve and air between neighbouring magnets. Table 1 gives the main dimensions.

The magnets are 4.1 mm thick.

Table 1. Main dimensions and ratings

| Quantity | Value |
|---|---|
| Number of slots | 24 |
| Number of poles | 20 |
| Stator outer radius | 87 mm |
| Stator inner (bore) radius | 49.2 mm |
| Stator yoke thickness | 4.1 mm |
| Tooth width (parallel-sided) | 7 mm |
| Slot opening | 1.6 mm |
| Tooth-tip depth (bore to tip underside, on the tooth centre line) | 1.7 mm |
| Rotor outer radius (over the magnets) | 48.2 mm |
| Rotor core outer radius | 44.1 mm |
| Pole-arc to pole-pitch ratio | 0.82 |
| Shaft radius | 13.5 mm |
| Stack length | 75 mm |
| Turns per coil | 13 |
| Parallel paths per phase | 2 |
| Rated current | 85 A (rms) |
| Rated speed | 1000 r/min |

## 3. Winding

The winding is a double-layer concentrated winding: each of the 24 teeth carries one coil, so every slot holds one coil side from each of its two teeth. The coils are connected into three phases by the star of slots, for the highest fundamental winding factor. All coils have the same number of turns, and the 8 coils of each phase form 2 identical parallel paths.

## 4. Materials

The stator core is stacked from M-45 18 mil laminations. The rotor core is laminated. It is mounted on a non-magnetic shaft. The magnets are sintered NdFeB of grade N30. Each magnet is magnetized radially, alternately outward and inward.

## 5. Results

A two-dimensional magnetostatic finite-element model of the whole machine was solved. At 1000 r/min the no-load phase back-EMF has a fundamental of 24.5 V (rms). With the rated current in each phase, in phase with the back-EMF, the average torque is 48.1 N m. The slot fill factor is 0.47, and the rated current density is 6.9 A/mm2 (rms).

## 6. Conclusion

The machine meets its torque requirement at the rated current with a moderate current density. Its dimensions and materials are listed above.
