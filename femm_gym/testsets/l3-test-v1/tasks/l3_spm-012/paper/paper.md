# Finite-element analysis of a 10-pole surface-magnet machine with a 9-slot concentrated winding

## Abstract

This paper describes a 9-slot, 10-pole surface permanent-magnet machine with a double-layer concentrated winding for an electric bicycle. It gives the machine's dimensions, winding and materials, and two-dimensional finite-element results for the no-load back-EMF and the torque at the rated current.

## 1. Introduction

Fractional-slot concentrated windings give short end windings and low cogging torque, and they suit machines with many poles. Surface-mounted magnets keep the rotor simple. This paper records one such design in enough detail for its finite-element results to be reproduced.

## 2. Machine geometry

The stator has 9 parallel-sided teeth, each ending in a tooth tip whose underside is flat and at right angles to the tooth centre line; the slots between them have an arc-shaped bottom. The slot opening between adjacent tips is air, and each slot holds two coil sides side by side. The rotor carries 10 arc-shaped magnets bonded to the surface of a cylindrical rotor core, with no retaining sleeve and air between neighbouring magnets. Table 1 gives the main dimensions.

The slot bottom is an arc of radius 49.8 mm. The air gap between the magnet surface and the stator bore is 1.5 mm.

Table 1. Main dimensions and ratings

| Quantity | Value |
|---|---|
| Number of slots | 9 |
| Number of poles | 10 |
| Stator outer radius | 59.5 mm |
| Stator inner (bore) radius | 37.3 mm |
| Tooth width (parallel-sided) | 14.6 mm |
| Slot opening | 1.9 mm |
| Tooth-tip depth (bore to tip underside, on the tooth centre line) | 1.1 mm |
| Magnet thickness | 3.6 mm |
| Pole-arc to pole-pitch ratio | 0.85 |
| Stack length | 45 mm |
| Turns per coil | 11 |
| Parallel paths per phase | 1 |
| Rated current | 21 A (rms) |
| Rated speed | 3000 r/min |

## 3. Winding

The winding is a double-layer concentrated winding: each of the 9 teeth carries one coil, so every slot holds one coil side from each of its two teeth. The coils are connected into three phases by the star of slots, for the highest fundamental winding factor. All coils have the same number of turns, and the 3 coils of each phase are connected in series.

## 4. Materials

The stator core is stacked from M-19 14 mil laminations. The rotor core is laminated. It is mounted on a non-magnetic shaft. The magnets are sintered NdFeB of grade N35. Each magnet is magnetized radially, alternately outward and inward.

## 5. Results

A two-dimensional magnetostatic finite-element model of the whole machine was solved. At 3000 r/min the no-load phase back-EMF has a fundamental of 28.5 V (peak). With the rated current in each phase, in phase with the back-EMF, the average torque is 4.03 N m. The slot fill factor is 0.45, and the rated current density is 5.6 A/mm2 (rms).

## 6. Conclusion

The machine meets its torque requirement at the rated current with a moderate current density. Its dimensions and materials are listed above.
