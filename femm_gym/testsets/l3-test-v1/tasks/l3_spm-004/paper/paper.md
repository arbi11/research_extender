# Design of a 12-slot, 8-pole surface permanent-magnet machine for a robot joint

## Abstract

This paper describes a 12-slot, 8-pole surface permanent-magnet machine with a double-layer concentrated winding for a robot joint. It gives the machine's dimensions, winding and materials, and two-dimensional finite-element results for the no-load back-EMF and the torque at the rated current.

## 1. Introduction

Fractional-slot concentrated windings give short end windings and low cogging torque, and they suit machines with many poles. Surface-mounted magnets keep the rotor simple. This paper records one such design in enough detail for its finite-element results to be reproduced.

## 2. Machine geometry

The stator has 12 parallel-sided teeth, each ending in a tooth tip whose underside is flat and at right angles to the tooth centre line; the slots between them have an arc-shaped bottom. The slot opening between adjacent tips is air, and each slot holds two coil sides side by side. The rotor carries 8 arc-shaped magnets bonded to the surface of a cylindrical rotor core, with no retaining sleeve and air between neighbouring magnets. Table 1 gives the main dimensions.

The stator yoke is 10.4 mm thick. The teeth are parallel-sided, 14.3 mm wide. The underside of each tooth tip lies 1.8 mm beyond the bore, measured along the tooth centre line. The magnets are 2.8 mm thick.

Table 1. Main dimensions and ratings

| Quantity | Value |
|---|---|
| Number of slots | 12 |
| Number of poles | 8 |
| Stator outer diameter | 175 mm |
| Stator inner (bore) radius | 50.8 mm |
| Slot opening | 3.4 mm |
| Air gap | 1.4 mm |
| Rotor core outer radius | 46.6 mm |
| Magnet arc (mechanical) | 36.45 deg |
| Shaft radius | 19.5 mm |
| Stack length | 40 mm |
| Turns per coil | 46 |
| Parallel paths per phase | 1 |
| Rated current | 12.4 A (rms) |
| Rated speed | 1000 r/min |

## 3. Winding

The winding is a double-layer concentrated winding: each of the 12 teeth carries one coil, so every slot holds one coil side from each of its two teeth. The coils are connected into three phases by the star of slots, for the highest fundamental winding factor. All coils have the same number of turns, and the 4 coils of each phase are connected in series.

## 4. Materials

The stator core is stacked from M-15 14 mil laminations. The rotor core is made of ARMCO Pure Iron. It is mounted on a non-magnetic shaft. The magnets are sintered NdFeB of grade N35. Each magnet is magnetized radially, alternately outward and inward.

## 5. Results

A two-dimensional magnetostatic finite-element model of the whole machine was solved. At 1000 r/min the no-load phase back-EMF has a fundamental of 58.9 V (peak). With the rated current in each phase, in phase with the back-EMF, the average torque is 14.5 N m. The slot fill factor is 0.41, and the rated current density is 5.8 A/mm2 (rms).

## 6. Conclusion

The machine meets its torque requirement at the rated current with a moderate current density. Its dimensions and materials are listed above.
