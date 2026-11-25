# Design of a 12-slot, 14-pole surface permanent-magnet machine for a robot joint

## Abstract

This paper describes a 12-slot, 14-pole surface permanent-magnet machine with a double-layer concentrated winding for a robot joint. It gives the machine's dimensions, winding and materials, and two-dimensional finite-element results for the no-load back-EMF and the torque at the rated current.

## 1. Introduction

Fractional-slot concentrated windings give short end windings and low cogging torque, and they suit machines with many poles. Surface-mounted magnets keep the rotor simple. This paper records one such design in enough detail for its finite-element results to be reproduced.

## 2. Machine geometry

The stator has 12 parallel-sided teeth, each ending in a tooth tip whose underside is flat and at right angles to the tooth centre line; the slots between them have an arc-shaped bottom. The slot opening between adjacent tips is air, and each slot holds two coil sides side by side. The rotor carries 14 arc-shaped magnets bonded to the surface of a cylindrical rotor core, with no retaining sleeve and air between neighbouring magnets. Table 1 gives the main dimensions.

The teeth are parallel-sided, 13.2 mm wide. Adjacent tooth tips are 2.6 mm apart along the bore. The air gap between the magnet surface and the stator bore is 1.35 mm. The magnets are 3.5 mm thick.

Table 1. Main dimensions and ratings

| Quantity | Value |
|---|---|
| Number of slots | 12 |
| Number of poles | 14 |
| Stator outer radius | 79.5 mm |
| Stator inner (bore) radius | 45.3 mm |
| Slot bottom radius | 71.7 mm |
| Tooth-tip depth (bore to tip underside, on the tooth centre line) | 2.1 mm |
| Rotor core outer radius | 40.45 mm |
| Magnet arc (mechanical) | 18.77 deg |
| Stack length | 110 mm |
| Series turns per phase | 56 |
| Parallel paths per phase | 1 |
| Rated current | 41.2 A (rms) |
| Rated speed | 1500 r/min |

## 3. Winding

The winding is a double-layer concentrated winding: each of the 12 teeth carries one coil, so every slot holds one coil side from each of its two teeth. The coils are connected into three phases by the star of slots, for the highest fundamental winding factor. All coils have the same number of turns, and the 4 coils of each phase are connected in series; Table 1 gives the turns in series in one path.

## 4. Materials

The stator core is stacked from M-43 18 mil laminations. The rotor core is solid steel. It is mounted on a non-magnetic shaft. The magnets have a remanence of 1.28 T and a relative recoil permeability of 1.04. Each magnet is magnetized in parallel, along the centre line of its pole, alternately outward and inward.

## 5. Results

A two-dimensional magnetostatic finite-element model of the whole machine was solved. With the rated current in each phase, in phase with the back-EMF, the average torque is 41.2 N m. At 1500 r/min the no-load phase back-EMF has a fundamental of 77.4 V (peak). The slot fill factor is 0.44, and the rated current density is 6.1 A/mm2 (rms).

## 6. Conclusion

The machine meets its torque requirement at the rated current with a moderate current density. Its dimensions and materials are listed above.
