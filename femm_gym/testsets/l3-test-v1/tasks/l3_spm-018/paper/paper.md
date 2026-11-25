# Finite-element analysis of a 12-pole surface-magnet machine with a 18-slot concentrated winding

## Abstract

This paper describes a 18-slot, 12-pole surface permanent-magnet machine with a double-layer concentrated winding for an industrial servo drive. It gives the machine's dimensions, winding and materials, and two-dimensional finite-element results for the no-load back-EMF and the torque at the rated current.

## 1. Introduction

Fractional-slot concentrated windings give short end windings and low cogging torque, and they suit machines with many poles. Surface-mounted magnets keep the rotor simple. This paper records one such design in enough detail for its finite-element results to be reproduced.

## 2. Machine geometry

The stator has 18 parallel-sided teeth, each ending in a tooth tip whose underside is flat and at right angles to the tooth centre line; the slots between them have an arc-shaped bottom. The slot opening between adjacent tips is air, and each slot holds two coil sides side by side. The rotor carries 12 arc-shaped magnets bonded to the surface of a cylindrical rotor core, with no retaining sleeve and air between neighbouring magnets. Table 1 gives the main dimensions.

The teeth are parallel-sided, 6.4 mm wide. The underside of each tooth tip lies 2.1 mm beyond the bore, measured along the tooth centre line. The active stack length is 105 mm.

Table 1. Main dimensions and ratings

| Quantity | Value |
|---|---|
| Number of slots | 18 |
| Number of poles | 12 |
| Stator outer diameter | 128 mm |
| Stator inner (bore) radius | 38.7 mm |
| Stator yoke thickness | 4.8 mm |
| Slot opening | 2.3 mm |
| Rotor outer diameter (over the magnets) | 75.7 mm |
| Magnet thickness | 3.5 mm |
| Pole-arc to pole-pitch ratio | 0.87 |
| Turns per coil | 40 |
| Parallel paths per phase | 2 |
| Rated current | 16.2 A (rms) |
| Rated speed | 750 r/min |

## 3. Winding

The winding is a double-layer concentrated winding: each of the 18 teeth carries one coil, so every slot holds one coil side from each of its two teeth. Table 2 lists the coil on each tooth, counter-clockwise from tooth 1. A coil marked + is wound so that positive phase current drives flux radially inward through its tooth, towards the rotor; a coil marked - is wound the other way. All coils have the same number of turns, and the 6 coils of each phase form 2 identical parallel paths.

Table 2. Coil sequence

| Tooth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Coil | +A | +B | +C | +A | +B | +C | +A | +B | +C | +A | +B | +C | +A | +B | +C | +A | +B | +C |

## 4. Materials

The stator core is stacked from M-15 14 mil laminations. The rotor core is laminated. It is mounted on a non-magnetic shaft. The magnets are sintered NdFeB of grade N38. Each magnet is magnetized radially, alternately outward and inward.

## 5. Results

A two-dimensional magnetostatic finite-element model of the whole machine was solved. At 750 r/min the no-load phase back-EMF has a fundamental of 46.7 V (rms). With the rated current in each phase, in phase with the back-EMF, the average torque is 27.7 N m. The slot fill factor is 0.46, and the rated current density is 6.9 A/mm2 (rms).

## 6. Conclusion

The machine meets its torque requirement at the rated current with a moderate current density. Its dimensions and materials are listed above.
