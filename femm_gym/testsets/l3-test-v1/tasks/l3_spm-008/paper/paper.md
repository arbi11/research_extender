# Design of a 12-slot, 14-pole surface permanent-magnet machine for an electric bicycle

## Abstract

This paper describes a 12-slot, 14-pole surface permanent-magnet machine with a double-layer concentrated winding for an electric bicycle. It gives the machine's dimensions, winding and materials, and two-dimensional finite-element results for the no-load back-EMF and the torque at the rated current.

## 1. Introduction

Fractional-slot concentrated windings give short end windings and low cogging torque, and they suit machines with many poles. Surface-mounted magnets keep the rotor simple. This paper records one such design in enough detail for its finite-element results to be reproduced.

## 2. Machine geometry

The stator has 12 parallel-sided teeth, each ending in a tooth tip whose underside is flat and at right angles to the tooth centre line; the slots between them have an arc-shaped bottom. The slot opening between adjacent tips is air, and each slot holds two coil sides side by side. The rotor carries 14 arc-shaped magnets bonded to the surface of a cylindrical rotor core, with no retaining sleeve and air between neighbouring magnets. Table 1 gives the main dimensions.

The slot bottom is an arc of radius 60.8 mm. The underside of each tooth tip lies 1.5 mm beyond the bore, measured along the tooth centre line. The active stack length is 110 mm.

Table 1. Main dimensions and ratings

| Quantity | Value |
|---|---|
| Number of slots | 12 |
| Number of poles | 14 |
| Stator outer radius | 67 mm |
| Stator bore diameter | 77 mm |
| Tooth width (parallel-sided) | 9.2 mm |
| Slot opening | 3.5 mm |
| Rotor outer diameter (over the magnets) | 75 mm |
| Magnet thickness | 2.1 mm |
| Magnet arc (mechanical) | 19.03 deg |
| Turns per coil | 11 |
| Parallel paths per phase | 2 |
| Rated current | 81.3 A (rms) |
| Rated speed | 2000 r/min |

## 3. Winding

The winding is a double-layer concentrated winding: each of the 12 teeth carries one coil, so every slot holds one coil side from each of its two teeth. Table 2 lists the coil on each tooth, counter-clockwise from tooth 1. A coil marked + is wound so that positive phase current drives flux radially inward through its tooth, towards the rotor; a coil marked - is wound the other way. All coils have the same number of turns, and the 4 coils of each phase form 2 identical parallel paths.

Table 2. Coil sequence

| Tooth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Coil | +A | +C | -C | -B | +B | +A | -A | -C | +C | +B | -B | -A |

## 4. Materials

The stator core is stacked from M-27 14 mil laminations. The rotor core is made of M-27 14 mil. It is mounted on a non-magnetic shaft. The magnets have a remanence of 1.12 T and a relative recoil permeability of 1.05. Each magnet is magnetized in parallel, along the centre line of its pole, alternately outward and inward.

## 5. Results

A two-dimensional magnetostatic finite-element model of the whole machine was solved. At 2000 r/min the no-load phase back-EMF has a fundamental of 20.3 V (rms). With the rated current in each phase, in phase with the back-EMF, the average torque is 22.9 N m. The slot fill factor is 0.4, and the rated current density is 6.2 A/mm2 (rms).

## 6. Conclusion

The machine meets its torque requirement at the rated current with a moderate current density. Its dimensions and materials are listed above.
