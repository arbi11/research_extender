# Design of a 18-slot, 12-pole surface permanent-magnet machine for a light electric vehicle

## Abstract

This paper describes a 18-slot, 12-pole surface permanent-magnet machine with a double-layer concentrated winding for a light electric vehicle. It gives the machine's dimensions, winding and materials, and two-dimensional finite-element results for the no-load back-EMF and the torque at the rated current.

## 1. Introduction

Fractional-slot concentrated windings give short end windings and low cogging torque, and they suit machines with many poles. Surface-mounted magnets keep the rotor simple. This paper records one such design in enough detail for its finite-element results to be reproduced.

## 2. Machine geometry

The stator has 18 parallel-sided teeth, each ending in a tooth tip whose underside is flat and at right angles to the tooth centre line; the slots between them have an arc-shaped bottom. The slot opening between adjacent tips is air, and each slot holds two coil sides side by side. The rotor carries 12 arc-shaped magnets bonded to the surface of a cylindrical rotor core, with no retaining sleeve and air between neighbouring magnets. Table 1 gives the main dimensions.

The stator yoke is 5.3 mm thick. The teeth are parallel-sided, 8.1 mm wide. The air gap between the magnet surface and the stator bore is 1.45 mm. The magnets are 4.3 mm thick.

Table 1. Main dimensions and ratings

| Quantity | Value |
|---|---|
| Number of slots | 18 |
| Number of poles | 12 |
| Stator outer diameter | 133 mm |
| Stator bore diameter | 84.6 mm |
| Slot opening | 2.8 mm |
| Tooth-tip depth (bore to tip underside, on the tooth centre line) | 1.1 mm |
| Pole-arc to pole-pitch ratio | 0.86 |
| Stack length | 105 mm |
| Turns per coil | 25 |
| Parallel paths per phase | 1 |
| Rated current | 10.5 A (rms) |
| Rated speed | 2000 r/min |

## 3. Winding

The winding is a double-layer concentrated winding: each of the 18 teeth carries one coil, so every slot holds one coil side from each of its two teeth. Table 2 lists the coil on each tooth, counter-clockwise from tooth 1. A coil marked + is wound so that positive phase current drives flux radially inward through its tooth, towards the rotor; a coil marked - is wound the other way. All coils have the same number of turns, and the 6 coils of each phase are connected in series.

Table 2. Coil sequence

| Tooth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Coil | +A | +B | +C | +A | +B | +C | +A | +B | +C | +A | +B | +C | +A | +B | +C | +A | +B | +C |

## 4. Materials

The stator core is stacked from M-19 14 mil laminations. The rotor core is made of M-19 14 mil. It is mounted on a non-magnetic shaft. The magnets have a remanence of 1.14 T and a relative recoil permeability of 1.03. Each magnet is magnetized in parallel, along the centre line of its pole, alternately outward and inward.

## 5. Results

A two-dimensional magnetostatic finite-element model of the whole machine was solved. At 2000 r/min the no-load phase back-EMF has a fundamental of 158 V (rms). With the rated current in each phase, in phase with the back-EMF, the average torque is 23.7 N m. The slot fill factor is 0.46, and the rated current density is 6.3 A/mm2 (rms).

## 6. Conclusion

The machine meets its torque requirement at the rated current with a moderate current density. Its dimensions and materials are listed above.
