"""Parameterized L5 construction procedures; no paper IDs or paper dimensions.

Procedures build a small geometry representation. Lua generation is a separate
step, so a construction can be inspected before anything is sent to FEMM.
"""
import math


def polar(radius, angle):
    a = math.radians(angle % 360)
    return (round(radius * math.cos(a), 10), round(radius * math.sin(a), 10))


class Geometry:
    def __init__(self):
        self.nodes, self.segments, self.arcs, self.labels = [], [], [], []
        self._points = {}
        self.owner = None
        self.trace = []

    def node(self, point, group=0):
        point = tuple(point)
        if point not in self._points:
            self._points[point] = len(self.nodes)
            self.nodes.append({"x": point[0], "y": point[1], "group": group, "owner": self.owner})
        return self._points[point]

    def segment(self, a, b, group=0):
        a, b = self.node(a, group), self.node(b, group)
        if a == b or any({s['a'], s['b']} == {a, b} for s in self.segments):
            raise ValueError("Degenerate or duplicate segment")
        self.segments.append({"a": a, "b": b, "group": group, "owner": self.owner})

    def arc(self, radius, start, span, max_segment, group=0):
        if not 0 < span <= 180 or not 0 < max_segment <= 10:
            raise ValueError("Arc span must be in (0,180], discretization in (0,10]")
        a, b = self.node(polar(radius, start), group), self.node(polar(radius, start + span), group)
        if a == b or any(s['a'] == a and s['b'] == b and abs(s['angle']-span) < 1e-9 for s in self.arcs):
            raise ValueError("Degenerate or duplicate arc")
        self.arcs.append({"a": a, "b": b, "angle": span, "max_segment": max_segment,
                          "radius": radius, "start": start % 360, "group": group, "owner": self.owner})

    def label(self, at, material, mesh=0, group=0, circuit="<None>", turns=0, magdir=0):
        self.labels.append({"x": at[0], "y": at[1], "material": material, "mesh": mesh,
                            "group": group, "circuit": circuit, "turns": turns,
                            "magdir": magdir, "owner": self.owner})


def semi_closed_radial_stator(g, *, slots, bore_radius, opening_radius, bottom_radius,
                              opening_angle, slot_angle, gap_arc, arc_step, slot_mesh):
    """Radial-sided slots in ideal iron, with shoulders and separate neck regions.

    Iron is excluded using <No Mesh> labels: its interfaces have the natural
    H_t=0 condition. This procedure is inapplicable to finite-permeability iron.
    Each interface is drawn once; there are no overlapping complete circles.
    """
    pitch = 360 / slots
    if int(slots) != slots or slots < 2 or not 0 < opening_angle < slot_angle < pitch:
        raise ValueError("Require integer slots and 0 < opening angle < slot angle < slot pitch")
    if not 0 < bore_radius < opening_radius < bottom_radius:
        raise ValueError("Stator radii must increase")
    for k in range(slots):
        c = k * pitch
        for sign in (-1, 1):
            g.segment(polar(bore_radius, c + sign*opening_angle/2),
                      polar(opening_radius, c + sign*opening_angle/2))
            g.segment(polar(opening_radius, c + sign*slot_angle/2),
                      polar(bottom_radius, c + sign*slot_angle/2))
        # Mesh partition only: air on both sides, no physical boundary condition.
        g.arc(bore_radius, c-opening_angle/2, opening_angle, gap_arc)
        g.arc(bore_radius, c+opening_angle/2, pitch-opening_angle, gap_arc)
        g.arc(opening_radius, c-slot_angle/2, (slot_angle-opening_angle)/2, arc_step)
        g.arc(opening_radius, c-opening_angle/2, opening_angle, arc_step)
        g.arc(opening_radius, c+opening_angle/2, (slot_angle-opening_angle)/2, arc_step)
        g.arc(bottom_radius, c-slot_angle/2, slot_angle, arc_step)
        g.arc(bottom_radius, c+slot_angle/2, pitch-slot_angle, arc_step)
        g.label(polar((bore_radius+opening_radius)/2, c), "Air", slot_mesh)
        g.label(polar((opening_radius+bottom_radius)/2, c+pitch/2), "<No Mesh>")


def radial_surface_magnet_ring(g, *, poles, inner_radius, outer_radius, pole_arc_ratio,
                               origin_angle, rotor_angle, mesh, gap_arc):
    """Alternating radial magnets and interpolar air around an ideal rotor core."""
    if int(poles) != poles or poles < 2 or poles % 2 or not 0 < pole_arc_ratio < 1:
        raise ValueError("Require positive even poles and 0 < pole arc ratio < 1")
    if not 0 < inner_radius < outer_radius:
        raise ValueError("Rotor radii must increase")
    pitch, width = 360/poles, 360/poles*pole_arc_ratio
    for k in range(poles):
        centre = origin_angle + rotor_angle + (k+0.5)*pitch
        start = centre-width/2
        for r in (inner_radius, outer_radius):
            g.arc(r, start, width, gap_arc, 1)
            g.arc(r, start+width, pitch-width, gap_arc, 1)
        for a in (start, start+width):
            g.segment(polar(inner_radius, a), polar(outer_radius, a), 1)
        g.label(polar((inner_radius+outer_radius)/2, centre), "Magnet", mesh, 1,
                magdir="theta" if k % 2 == 0 else "theta+180")
        g.label(polar((inner_radius+outer_radius)/2, centre+pitch/2), "Air", mesh, 1)


def slot_circuit_labels(g, *, slots, radius, winding, mesh):
    """Explicit signed slot-to-phase mapping; winding layout is never guessed."""
    if sorted(w['slot'] for w in winding) != list(range(slots)):
        raise ValueError("Winding must assign every slot exactly once")
    phases = {w['phase'] for w in winding}
    for phase in phases:
        if sum(w['turns'] for w in winding if w['phase'] == phase) != 0:
            raise ValueError("Each phase needs balanced go and return turns")
    for w in winding:
        if w['phase'] not in ('A', 'B', 'C') or not isinstance(w['turns'], int) or w['turns'] == 0:
            raise ValueError("Use named three-phase circuits and nonzero signed integer turns")
        g.label(polar(radius, w['slot']*360/slots), "Coil", mesh,
                circuit=w['phase'], turns=w['turns'])


def uniform_air_gap(g, *, rotor_radius, bore_radius, mesh):
    """Label the annulus between existing rotor and stator surfaces; draw neither again."""
    if not 0 < rotor_radius < bore_radius:
        raise ValueError("Air gap must be positive")
    g.label(polar((rotor_radius+bore_radius)/2, 0), "Air", mesh)


def ideal_iron_core(g, *, radius, group):
    """Exclude the circular ideal core already bounded by the magnet-ring procedure."""
    if radius <= 0 or not any(abs(math.hypot(n['x'], n['y'])-radius) < 1e-8 for n in g.nodes):
        raise ValueError("An existing core interface is required")
    g.label((0, 0), "<No Mesh>", group=group)


def point_potential_reference(g, *, radius, angle, value):
    """Reference A at an existing boundary node; no extra geometry is introduced."""
    at = polar(radius, angle)
    if at not in g._points:
        raise ValueError("Reference potential must use an existing node")
    g.reference = {"at": at, "value": value, "owner": g.owner}


PROCEDURES = {
    f.__name__: f for f in (semi_closed_radial_stator, radial_surface_magnet_ring,
                            slot_circuit_labels, uniform_air_gap,
                            point_potential_reference, ideal_iron_core)
}
