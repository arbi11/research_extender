# Design of a 12-slot, 10-pole surface permanent-magnet machine for a small wind turbine

## Abstract

This paper describes a 12-slot, 10-pole surface permanent-magnet machine with a double-layer concentrated winding for a small wind turbine. It gives the machine's dimensions, winding and materials, and two-dimensional finite-element results for the no-load back-EMF and the torque at the rated current.

## 1. Introduction

Fractional-slot concentrated windings give short end windings and low cogging torque, and they suit machines with many poles. Surface-mounted magnets keep the rotor simple. This paper records one such design in enough detail for its finite-element results to be reproduced.

## 2. Machine geometry

The stator has 12 parallel-sided teeth, each ending in a tooth tip whose underside is flat and at right angles to the tooth centre line; the slots between them have an arc-shaped bottom. The slot opening between adjacent tips is air, and each slot holds two coil sides side by side. The rotor carries 10 arc-shaped magnets bonded to the surface of a cylindrical rotor core, with no retaining sleeve and air between neighbouring magnets. Table 1 gives the main dimensions.

The stator yoke is 9.1 mm thick. The teeth are parallel-sided, 10.9 mm wide. The air gap between the magnet surface and the stator bore is 1.2 mm. The magnets are 3.3 mm thick.

Table 1. Main dimensions and ratings

| Quantity | Value |
|---|---|
| Number of slots | 12 |
| Number of poles | 10 |
| Stator outer radius | 73.5 mm |
| Stator inner (bore) radius | 44 mm |
| Slot opening | 1.5 mm |
| Tooth-tip depth (bore to tip underside, on the tooth centre line) | 1.2 mm |
| Magnet arc (mechanical) | 32.76 deg |
| Stack length | 115 mm |
| Series turns per phase | 84 |
| Parallel paths per phase | 2 |
| Rated current | 24.6 A (rms) |
| Rated speed | 2000 r/min |

## 3. Winding

The winding is a double-layer concentrated winding: each of the 12 teeth carries one coil, so every slot holds one coil side from each of its two teeth. Table 2 lists the coil on each tooth, counter-clockwise from tooth 1. A coil marked + is wound so that positive phase current drives flux radially inward through its tooth, towards the rotor; a coil marked - is wound the other way. All coils have the same number of turns, and the 4 coils of each phase form 2 identical parallel paths; Table 1 gives the turns in series in one path.

Table 2. Coil sequence

| Tooth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Coil | +A | -A | -B | +B | +C | -C | -A | +A | +B | -B | -C | +C |

## 4. Materials

The stator core is stacked from M-36 14 mil laminations. The rotor core is made of 1020 Steel (hot rolled). It is mounted on a non-magnetic shaft. The magnets are sintered NdFeB of grade N33. Each magnet is magnetized radially, alternately outward and inward.

## 5. Results

A two-dimensional magnetostatic finite-element model of the whole machine was solved. At 2000 r/min the no-load phase back-EMF has a fundamental of 102 V (rms). With the rated current in each phase, in phase with the back-EMF, the average torque is 35.6 N m. The slot fill factor is 0.45, and the rated current density is 6.8 A/mm2 (rms).

## 6. Conclusion

The machine meets its torque requirement at the rated current with a moderate current density. Its dimensions and materials are listed above.
