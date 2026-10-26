"""Closed-form references that official examples compare FEMM results against.

Each assessor uses only article-stated physical inputs and the native values
already measured by the frozen plan. Standard-library implementations keep the
references available without numerical packages; tests cross-check them.
"""
import cmath
import math

EULER_GAMMA = 0.57721566490153286061
MU0 = 4e-7 * math.pi


def bessel_i(order, z, terms=80):
    """Modified Bessel function of the first kind for a nonnegative integer order, complex z (power series)."""
    if not isinstance(order, int) or order < 0:
        raise ValueError("The order must be a nonnegative integer")
    quarter = z * z / 4
    term = (z / 2) ** order / math.factorial(order)
    total = term
    for k in range(1, terms):
        term *= quarter / (k * (k + order))
        total += term
    return total


def bessel_k(order, z, terms=80):
    """Modified Bessel function of the second kind for order 0 or 1 (A&S 9.6.13 and 9.6.11)."""
    quarter = z * z / 4
    log_half = cmath.log(z / 2)
    if order == 0:
        term, harmonic, total = 1 + 0j, 0.0, 0j
        for k in range(1, terms):
            term *= quarter / (k * k)
            harmonic += 1 / k
            total += term * harmonic
        return -(log_half + EULER_GAMMA) * bessel_i(0, z, terms) + total
    if order == 1:
        digamma = lambda n: -EULER_GAMMA + sum(1 / j for j in range(1, n))
        term, total = 1 + 0j, 0j
        for k in range(terms):
            if k:
                term *= quarter / (k * (k + 1))
            total += (digamma(k + 1) + digamma(k + 2)) * term
        return 1 / z + log_half * bessel_i(1, z, terms) - z / 4 * total
    raise ValueError("Only orders 0 and 1 are implemented")


def rotating_cylinder_torque(order, current_density, inner_radius, outer_radius, radius, conductivity, slip, depth):
    """Torque on a non-magnetic conducting cylinder from one travelling wave of an annular winding in free space.

    The winding's current density Re[J e^(j(omega t - order theta))] (complex J,
    order nonzero) fills inner_radius < r < outer_radius. Alone, it gives the
    potential alpha r^n inside the winding (n = |order|), with alpha =
    mu0 J G / (2n) and G the integral of rho^(1-n) over the winding. A cylinder of
    the given radius and conductivity, seeing the wave at slip angular frequency
    s = omega - order Omega, holds A = E I_n(k r) with k^2 = j mu0 sigma s and
    reflects D r^-n, D = alpha R^2n (n - f)/(n + f), f = kR I_n'(kR)/I_n(kR).
    The gap's Maxwell stress then gives T = -2 pi L order n |alpha|^2 Im(D/alpha) / mu0,
    the same at every radius between the cylinder and the winding, positive
    counterclockwise. SI units; depth is the axial length.
    """
    n = abs(order)
    if not n or not 0 < radius < inner_radius < outer_radius:
        raise ValueError("Need a nonzero order and radius < inner radius < outer radius")
    if slip == 0:
        return 0.0
    spread = math.log(outer_radius / inner_radius) if n == 2 else (outer_radius ** (2 - n) - inner_radius ** (2 - n)) / (2 - n)
    alpha = MU0 * current_density * spread / (2 * n)
    z = cmath.sqrt(1j * MU0 * conductivity * slip) * radius
    f = n + z * bessel_i(n + 1, z) / bessel_i(n, z)
    reflection = radius ** (2 * n) * (n - f) / (n + f)
    return -2 * math.pi * depth * order * n * abs(alpha) ** 2 * reflection.imag / MU0


def tube_solution(inner_radius, outer_radius, relative_permeability, conductivity, frequency, current):
    """Stoll-type solution for eddy currents in a tube carrying zero net current around a wire."""
    delta = math.sqrt(2 / (2 * math.pi * frequency * relative_permeability * MU0 * conductivity))
    alpha = (1 + 1j) / delta
    a = [[bessel_i(1, alpha * r), bessel_k(1, alpha * r)] for r in (inner_radius, outer_radius)]
    b = [current / (2 * math.pi * r) for r in (inner_radius, outer_radius)]
    determinant = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    c0 = (b[0] * a[1][1] - a[0][1] * b[1]) / determinant
    c1 = (a[0][0] * b[1] - b[0] * a[1][0]) / determinant
    return {"skin_depth_m": delta, "alpha": alpha, "c0": c0, "c1": c1,
            "current_density": lambda r: alpha * (c0 * bessel_i(0, alpha * r) - c1 * bessel_k(0, alpha * r))}


def _tube(assessor, case, raw, metrics, model):
    p = assessor["parameters"]
    solution = tube_solution(p["inner_radius_m"], p["outer_radius_m"], p["relative_permeability"],
                             p["conductivity_S_per_m"], p["frequency_Hz"], p["current_A"])
    measurement = next(m for m in case["measurements"] if m["id"] == assessor["measurement"])
    scale = assessor["metres_per_unit"]
    step = (case.get("steps") or [{"label": "base"}])[0]["label"]
    errors, exact_values, rows = [], [], []
    for index, (x, y) in enumerate(measurement["points"]):
        radius = math.hypot(x, y) * scale
        fea = (raw[(step, measurement["id"] + ":Je", index)] + raw[(step, measurement["id"] + ":Js", index)]) * 1e6
        exact = solution["current_density"](radius)
        errors.append(abs(fea - exact) ** 2)
        exact_values.append(abs(exact) ** 2)
        rows.append({"radius_m": radius, "fea_A_per_m2": [fea.real, fea.imag], "exact_A_per_m2": [exact.real, exact.imag]})
    rms_error = math.sqrt(sum(errors) / len(errors))
    rms_exact = math.sqrt(sum(exact_values) / len(exact_values))
    ratio = rms_error / rms_exact
    # Published constants verify this reference implementation independently of
    # FEMM. Each comparison names the component it checks, so a printed sign
    # discrepancy is reported rather than silently absorbed.
    parts = {"complex": lambda v: v, "real": lambda v: v.real, "imag": lambda v: v.imag, "abs": abs}
    constant_checks = []
    for item in assessor.get("published_constants", []):
        computed = complex(solution[item["name"]])
        wanted = complex(*item["value"]) if isinstance(item["value"], list) else complex(item["value"])
        a, b = parts[item["compare"]](computed), parts[item["compare"]](wanted)
        difference = abs(a - b) / abs(b)
        constant_checks.append(dict(item, computed=[computed.real, computed.imag], relative_difference=difference,
                                    passed=difference <= assessor.get("constant_relative_tolerance", 1e-3)))
    constants_ok = all(c["passed"] for c in constant_checks if c.get("used_for_acceptance", True))
    metrics["current_density_rms_error_ratio"] = ratio
    status = "passed" if ratio <= assessor["maximum_rms_error_ratio"] and constants_ok else "failed"
    return {"status": status, "assessor": "tube_bessel", "rms_error_A_per_m2": rms_error, "rms_exact_A_per_m2": rms_exact,
            "rms_error_ratio": ratio, "maximum_rms_error_ratio": assessor["maximum_rms_error_ratio"],
            "published_rms_error_ratio": assessor.get("published_rms_error_ratio"),
            "analytic_constants": constant_checks, "analytic_constants_match_article": constants_ok, "points": rows,
            "target": {"used_for_acceptance": True, "kind": "analytic", "quote": assessor.get("quote")}}


def kelvin_applied_field(assessor, model):
    """Applied field predicted for FEMM's periodically linked Kelvin disk, independent of any solve.

    FEMM's mesher orients each periodic chain with its region on the left and
    reverses the second chain (tangle.cpp, buildPbcTwinFromCDT), so the exterior
    rim angle phi' is linked to the domain angle 180 - phi. The disk's uniform
    magnetization at angle alpha is a rim current M sin(phi' - alpha), which in
    the domain is M sin(phi + alpha): a uniform field at -alpha with magnitude
    mu0*Hc/2.
    """
    rule = assessor["expected_field"]
    if rule["rule"] != "kelvin_mirror":
        raise ValueError("Unknown applied-field rule: " + rule["rule"])
    exterior = next(label for label in model["labels"] if label.get("material") == rule["exterior_material"])
    material = next(m for m in model["materials"] if m["name"] == rule["exterior_material"])
    direction = -exterior["magdir"]
    magnitude = MU0 * material["props"]["H_c"] / 2
    return magnitude * math.cos(math.radians(direction)), magnitude * math.sin(math.radians(direction)), direction


def _angle_difference(a, b):
    return abs((a - b + 180) % 360 - 180)


def _dipole_torque(assessor, case, raw, metrics, model):
    """Torque m x B against the applied field predicted from the construction, plus a measured-field check."""
    field = assessor["field_step"]
    bx = metrics[assessor["field_metrics"][0] + "@" + field]
    by = metrics[assessor["field_metrics"][1] + "@" + field]
    magnitude = math.hypot(bx, by)
    ex, ey, direction = kelvin_applied_field(assessor, model)
    predicted = math.hypot(ex, ey)
    measured_direction = math.degrees(math.atan2(by, bx))
    direction_ok = _angle_difference(measured_direction, direction) <= assessor["expected_field"]["direction_tolerance_deg"]
    magnitude_ok = abs(magnitude - predicted) <= assessor["relative_tolerance"] * predicted
    declared_ok = abs(predicted - assessor["applied_field_T"]) <= 1e-9 * assessor["applied_field_T"]
    magnet = next(label for label in model["labels"] if label.get("material") == assessor["magnet_material"])
    base = magnet["magdir"]
    moment = assessor["moment_A_m2"]
    tolerance = assessor["relative_tolerance"] * moment * predicted
    rows, ok = [], True
    for step, angle in assessor["rotation_steps"].items():
        rotated = math.radians(base + angle)
        expected = moment * (math.cos(rotated) * ey - math.sin(rotated) * ex)
        with_measured = moment * (math.cos(rotated) * by - math.sin(rotated) * bx)
        observed = metrics[assessor["torque_metric"] + "@" + step]
        error = abs(observed - expected)
        rows.append({"step": step, "rotor_angle_deg": angle, "expected_N_m": expected, "observed_N_m": observed,
                     "expected_from_measured_field_N_m": with_measured, "absolute_error": error, "passed": error <= tolerance})
        ok &= error <= tolerance
    passed = ok and direction_ok and magnitude_ok and declared_ok
    return {"status": "passed" if passed else "failed", "assessor": "dipole_torque",
            "predicted_applied_field_T": [ex, ey], "predicted_field_direction_deg": direction,
            "measured_applied_field_T": [bx, by], "measured_field_direction_deg": measured_direction,
            "applied_field_magnitude_T": magnitude, "field_direction_check": direction_ok, "field_magnitude_check": magnitude_ok,
            "declared_field_matches_construction": declared_ok,
            "magnet_direction_deg": base, "absolute_tolerance_N_m": tolerance, "rows": rows,
            "convention": "Rotor angle rotates the inner air-gap side counterclockwise; torque is m x B about +z with the predicted field.",
            "target": {"used_for_acceptance": True, "kind": "analytic", "quote": assessor.get("quote")}}


def _applied_field(assessor, case, raw, metrics, model):
    """A Kelvin-imposed field is uniform, at the predicted direction and magnitude, at every probe."""
    ex, ey, direction = kelvin_applied_field(assessor, model)
    predicted = math.hypot(ex, ey)
    step = (case.get("steps") or [{"label": "base"}])[0]["label"]
    measurement = next(m for m in case["measurements"] if m["id"] == assessor["measurement"])
    rows, ok = [], True
    for index, point in enumerate(measurement["points"]):
        bx = raw[(step, measurement["id"] + ":B1", index)].real
        by = raw[(step, measurement["id"] + ":B2", index)].real
        error = math.hypot(bx - ex, by - ey) / predicted
        angle = math.degrees(math.atan2(by, bx))
        passed = error <= assessor["relative_tolerance"] and _angle_difference(angle, direction) <= assessor["expected_field"]["direction_tolerance_deg"]
        rows.append({"point": point, "measured_T": [bx, by], "direction_deg": angle, "relative_vector_error": error, "passed": passed})
        ok &= passed
    return {"status": "passed" if ok else "failed", "assessor": "applied_field", "predicted_applied_field_T": [ex, ey],
            "predicted_field_direction_deg": direction, "rows": rows,
            "target": {"used_for_acceptance": True, "kind": "analytic", "quote": assessor.get("quote")}}


def _polygon_centroid(polygon):
    area = cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        cross = x1 * y2 - x2 * y1
        area, cx, cy = area + cross, cx + (x1 + x2) * cross, cy + (y1 + y2) * cross
    return cx / (3 * area), cy / (3 * area)


def linkage_prediction(model, pole_pairs, circuits, rotor_group):
    """Fundamental flux-linkage phasor of each circuit, from the model's construction alone.

    Each coil side contributes its signed turns at the angle of its region's
    centroid, and each magnet its radial magnetization (positive outward) at its
    region's centroid. Magnetization continues radially across the air gap, so
    with the rotor's fundamental phasor R = sum(m_r e^(i p theta)) the vector
    potential at the conductors varies as sin(p theta - arg R), and a circuit with
    S = sum(turns e^(i p theta)) links Im(S e^(-i arg R)). Turning the rotor
    counterclockwise by alpha advances arg R by p alpha.
    """
    from .reproduce import _face_cycles, _region_of
    faces = _face_cycles(model)
    coercivity = {m["name"]: m["props"].get("H_c", 0) for m in model["materials"]}

    def centroid_angle(label):
        index = _region_of((label["x"], label["y"]), faces)
        if index is None:
            raise ValueError(f"Label at ({label['x']}, {label['y']}) lies outside every closed contour")
        x, y = _polygon_centroid(faces[index])
        return math.atan2(y, x)

    windings = {name: 0j for name in circuits}
    rotor = 0j
    for label in model["labels"]:
        if label.get("circuit") in windings:
            windings[label["circuit"]] += label["turns"] * cmath.exp(1j * pole_pairs * centroid_angle(label))
        elif label.get("group") == rotor_group and coercivity.get(label["material"]):
            theta = centroid_angle(label)
            function = (label.get("magdir_function") or "").replace(" ", "")
            if function not in ("", "theta", "theta+180"):
                raise ValueError("Unsupported magnetization function: " + function)
            direction = (math.degrees(theta) + (180 if function == "theta+180" else 0)) if function else label["magdir"]
            radial = math.cos(math.radians(direction) - theta)
            rotor += coercivity[label["material"]] * radial * cmath.exp(1j * pole_pairs * theta)
    if not rotor or not all(windings.values()):
        raise ValueError("The model has no rotor magnetization fundamental or a circuit links nothing")
    return windings, rotor


def _rotor_samples(assessor, case):
    """Rotor angle of each step from the case's rotate_group steps, keeping one even sweep of the declared span.

    With ``rotor_band`` the angle is instead the named sliding band's inner angle
    (boundary property 10), which turns the rotor side without remeshing.

    The span is one electrical period, or half of one when the assessor declares
    half-wave symmetry (a pole pitch negates every linkage). ``step_prefix``
    keeps only the sweep's steps when others (an inductance step) follow it.
    """
    group, band = assessor["rotor_group"], assessor.get("rotor_band")
    period = 360 / assessor["pole_pairs"]
    span = period / 2 if assessor.get("half_wave_symmetry") else period
    alpha, angles = 0.0, {}
    for step in case["steps"]:
        for change in step.get("modify", []):
            if band is None and change["function"] == "rotate_group" and change["group"] == group:
                alpha += change["angle"]
            elif band is not None and change["function"] == "modifyboundprop" and change["name"] == band and change["property"] == 10:
                alpha = change["value"]  # a sliding band's inner angle is absolute
        if step["label"].startswith(assessor.get("step_prefix", "")):
            angles.setdefault(step["label"], alpha)
    samples = sorted((a, label) for label, a in angles.items() if -1e-9 <= a < span - 1e-9)
    count = len(samples)
    if count < 8 or any(abs(a - k * span / count) > 1e-6 * span for k, (a, _) in enumerate(samples)):
        raise ValueError("Rotor positions must sample the declared span evenly from zero")
    return samples, span


def _waveform_statistic(assessor, case, raw, metrics, model):
    """A statistic of one metric over the rotor sweep, against a value frozen in the assessor.

    ``fundamental_rms`` and ``fundamental_peak`` take the component at
    pole_pairs times the rotor angle (extending a half period by the declared
    half-wave symmetry), and ``harmonic_peak`` the component of electrical
    ``order`` n at n times that; ``peak_to_peak``, ``mean`` and ``maximum`` (the
    largest sample, as a paper's "maximum of the curve") use the samples as they
    are, so a maximum needs the peak on a sample or dense samples near it.

    A harmonic can be reported as the time derivative of the waveform at a given
    electrical frequency (``time_derivative_frequency_hz``: an EMF's harmonic from
    a flux linkage's, n times 2 pi f times its amplitude), and any statistic times
    a stated ``scale`` (a convention factor, such as the square root of 3/2 that
    takes a phase flux-linkage amplitude to a power-invariant d-axis value).

    ``rms`` and ``peak`` describe the whole waveform, every harmonic the samples
    resolve: the root mean square over a period and the largest absolute value
    of the band-limited (trigonometric) interpolation through the samples. With
    a time derivative they describe the derivative, as a measured EMF's rms and
    peak (an oscilloscope's readings, harmonics included) follow from a sampled
    flux linkage.
    """
    statistic = assessor["statistic"]
    samples, span = _rotor_samples(assessor, case)
    p = assessor["pole_pairs"]
    values = [(a, metrics[assessor["metric"] + "@" + label]) for a, label in samples]
    order = assessor.get("order", 1) if statistic == "harmonic_peak" else 1
    if assessor.get("time_derivative_frequency_hz") and statistic not in ("fundamental_rms", "fundamental_peak", "harmonic_peak",
                                                                           "rms", "peak"):
        raise ValueError("A time derivative applies to a harmonic statistic, rms or peak")
    if statistic in ("rms", "peak"):
        observed = _band_limited_statistic(statistic, values, span, p, assessor.get("half_wave_symmetry"),
                                           assessor.get("time_derivative_frequency_hz"))
    elif statistic in ("fundamental_rms", "fundamental_peak", "harmonic_peak"):
        if not (isinstance(order, int) and order >= 1):
            raise ValueError("A harmonic order is a positive integer")
        points = values + ([(a + span, -v) for a, v in values] if assessor.get("half_wave_symmetry") else [])
        amplitude = abs(2 / len(points) * sum(v * cmath.exp(1j * math.radians(order * p * a)) for a, v in points))
        observed = amplitude / math.sqrt(2) if statistic == "fundamental_rms" else amplitude
        if assessor.get("time_derivative_frequency_hz"):
            observed *= 2 * math.pi * assessor["time_derivative_frequency_hz"] * order
    elif statistic == "peak_to_peak":
        observed = max(v for _, v in values) - min(v for _, v in values)
    elif statistic == "mean":
        observed = sum(v for _, v in values) / len(values)
    elif statistic == "maximum":
        observed = max(v for _, v in values)
    else:
        raise ValueError("Unknown waveform statistic: " + statistic)
    observed *= assessor.get("scale", 1)
    target = {key: assessor[key] for key in ("value", "relative_tolerance", "absolute_tolerance", "quote", "location",
                                              "used_for_acceptance", "allowance") if key in assessor}
    target.update(kind=assessor.get("value_kind"), metric=assessor_metric(assessor))
    target.setdefault("used_for_acceptance", True)
    error = abs(observed - assessor["value"])
    tolerance = max(abs(assessor["value"]) * assessor.get("relative_tolerance", 0), assessor.get("absolute_tolerance", 0))
    return {"status": "passed" if error <= tolerance else "failed", "assessor": "waveform_statistic",
            "metric": assessor_metric(assessor), "statistic": statistic + (f" {order}" if statistic == "harmonic_peak" else ""),
            "observed": observed,
            "expected": assessor["value"], "absolute_error": error,
            "relative_error": error / abs(assessor["value"]) if assessor["value"] else None,
            "absolute_tolerance": tolerance, "samples": len(values), "target": target}


def _band_limited_statistic(statistic, values, span, pole_pairs, half_wave, frequency_hz=None):
    """rms or peak of the trigonometric interpolation through samples evenly spread over one electrical period.

    ``values`` are (mechanical angle, value) pairs over ``span`` (a period, or half
    of one with ``half_wave`` symmetry, which supplies the negated other half).
    With ``frequency_hz`` the statistic is of the time derivative at that
    electrical frequency; the derivative drops the mean and the Nyquist term,
    which a derivative of samples cannot resolve.
    """
    points = list(values) + ([(a + span, -v) for a, v in values] if half_wave else [])
    count = len(points)
    if count < 4:
        raise ValueError("A waveform statistic needs at least four samples per period")
    phases = [math.radians(pole_pairs * a) for a, _ in points]
    top = count // 2
    # Complex coefficients c_n of v(theta) = sum c_n exp(j n theta), for n = 0..top (negative orders are conjugates).
    coefficients = [sum(v * cmath.exp(-1j * n * t) for (_, v), t in zip(points, phases)) / count for n in range(top + 1)]
    weights = [1] + [2] * (top - 1) + ([1] if count % 2 == 0 else [2])
    if frequency_hz:
        omega = 2 * math.pi * frequency_hz
        coefficients = [1j * n * omega * c for n, c in enumerate(coefficients)]
        coefficients[0] = 0
        if count % 2 == 0:
            coefficients[top] = 0
    if statistic == "rms":
        return math.sqrt(sum(w * abs(c) ** 2 for w, c in zip(weights, coefficients)))
    dense = 32 * count
    peak = 0.0
    for k in range(dense):
        theta = 2 * math.pi * k / dense
        value = sum((w * c * cmath.exp(1j * n * theta)).real for n, (w, c) in enumerate(zip(weights, coefficients)))
        peak = max(peak, abs(value))
    return peak


def _spatial_harmonic(assessor, case, raw, metrics, model):
    """One spatial harmonic's amplitude of a field component sampled evenly along one period, against a frozen value.

    The points of the named point_values measurement must lie on a line along
    ``axis`` (0 for x or r, 1 for y or z) and sample exactly one ``period``
    evenly; the amplitude of order n is |2/N sum v exp(-2 pi i n s / period)|.
    """
    measurement = next(m for m in case["measurements"] if m["id"] == assessor["measurement"])
    points, axis, period = measurement["points"], assessor["axis"], assessor["period"]
    along = [p[axis] for p in points]
    across = {round(p[1 - axis], 9) for p in points}
    count = len(points)
    spacing = period / count
    if len(across) != 1 or any(abs((s - along[0]) - k * spacing) > 1e-9 * period for k, s in enumerate(along)):
        raise ValueError("The points must sample one period evenly along a line")
    step = assessor.get("step") or (case.get("steps") or [{"label": "base"}])[0]["label"]
    values = [raw[(step, f"{measurement['id']}:{assessor['field']}", k)].real for k in range(count)]
    order = assessor["order"]
    amplitude = abs(2 / count * sum(v * cmath.exp(-2j * math.pi * order * (s - along[0]) / period) for v, s in zip(values, along)))
    metric = assessor_metric(assessor)
    target = {key: assessor[key] for key in ("value", "relative_tolerance", "absolute_tolerance", "quote", "location",
                                              "used_for_acceptance", "allowance") if key in assessor}
    target.update(kind=assessor.get("value_kind"), metric=metric)
    target.setdefault("used_for_acceptance", True)
    error = abs(amplitude - assessor["value"])
    tolerance = max(abs(assessor["value"]) * assessor.get("relative_tolerance", 0), assessor.get("absolute_tolerance", 0))
    return {"status": "passed" if error <= tolerance else "failed", "assessor": "spatial_harmonic", "metric": metric,
            "statistic": f"harmonic {order}", "observed": amplitude, "expected": assessor["value"], "absolute_error": error,
            "relative_error": error / abs(assessor["value"]) if assessor["value"] else None,
            "absolute_tolerance": tolerance, "samples": count, "target": target}


def _phase_linkage(assessor, case, raw, metrics, model):
    """Winding sign and phase linkage: each phase's flux-linkage fundamental against the construction.

    Rotor angles come from the case's rotate_group steps, or from a sliding band's
    inner angle when the assessor names ``rotor_band``. Samples must cover one
    electrical period evenly, or half of one when the declared half-wave symmetry
    (a pole pitch negates every linkage) extends them.
    """
    p, sequence, group = assessor["pole_pairs"], assessor["expected_sequence"], assessor["rotor_group"]
    windings, rotor = linkage_prediction(model, p, sequence, group)
    samples, span = _rotor_samples(assessor, case)
    count = len(samples)
    rows, phases_ok = [], True
    predicted, measured = {}, {}
    for name in sequence:
        points = [(a, metrics[assessor["flux_metrics"][name] + "@" + label]) for a, label in samples]
        if assessor.get("half_wave_symmetry"):
            points += [(a + span, -value) for a, value in points]
        fundamental = 2 / len(points) * sum(value * cmath.exp(1j * math.radians(p * a)) for a, value in points)
        measured[name] = (math.degrees(cmath.phase(1j * fundamental)), abs(fundamental))
        predicted[name] = math.degrees(cmath.phase(windings[name] / rotor))
        error = _angle_difference(measured[name][0], predicted[name])
        ok = error <= assessor["phase_tolerance_deg"]
        phases_ok &= ok
        rows.append({"phase": name, "predicted_phase_deg": predicted[name], "measured_phase_deg": measured[name][0],
                     "phase_error_deg": error, "fundamental_Wb": measured[name][1], "passed": ok})
    amplitudes = [measured[name][1] for name in sequence]
    balance = (max(amplitudes) - min(amplitudes)) / (sum(amplitudes) / len(amplitudes))
    links = [abs(windings[name]) for name in sequence]
    winding_balance = (max(links) - min(links)) / (sum(links) / len(links))
    step_angle = 360 / len(sequence)
    # Counterclockwise rotation brings each phase to its peak one phase step after the previous one. The prediction
    # comes from region centroids, which a drawing's coordinate noise moves (0.1 micrometre in the IEEJ D model's DXF
    # moved them by 5e-5 degrees), so its order is checked to 1e-3 degrees.
    predicted_order = all(_angle_difference(predicted[name] - predicted[sequence[0]], k * step_angle) <= 1e-3
                          for k, name in enumerate(sequence))
    measured_order = all(_angle_difference(measured[name][0] - measured[sequence[0]][0], k * step_angle)
                         <= assessor["phase_tolerance_deg"] for k, name in enumerate(sequence))
    passed = (phases_ok and balance <= assessor["amplitude_balance_tolerance"] and winding_balance <= 1e-6
              and predicted_order and measured_order)
    return {"status": "passed" if passed else "failed", "assessor": "phase_linkage", "rows": rows,
            "rotor_positions": count * (2 if assessor.get("half_wave_symmetry") else 1),
            "amplitude_imbalance": balance, "winding_imbalance": winding_balance, "predicted_sequence_check": predicted_order,
            "measured_sequence_check": measured_order, "expected_sequence": sequence,
            "convention": "Phases listed in the order they peak as the rotor turns counterclockwise; angles are electrical.",
            "target": {"used_for_acceptance": True, "kind": "analytic", "quote": assessor.get("quote")}}


def _harmonic_steps(case, sweep, speed):
    """{order: (step label, conjugate)} for one operating point of a harmonic-superposition sweep."""
    found = {}
    for step in case.get("steps") or []:
        for item in step.get("harmonics", []):
            if item["sweep"] == sweep and math.isclose(item["speed_rad_s"], speed, rel_tol=1e-12, abs_tol=1e-9):
                if item["order"] in found:
                    raise ValueError(f"Order {item['order']} of {sweep} at {speed} rad/s is solved twice")
                found[item["order"]] = (step["label"], bool(step.get("conjugate")))
    if not found:
        raise ValueError(f"No steps serve {sweep} at {speed} rad/s")
    return found


def _circle(case, identifier):
    from .reproduce import circle_points
    measurement = next(m for m in case["measurements"] if m["id"] == identifier)
    if measurement["op"] != "circle_values":
        raise ValueError(f"{identifier} is not a circle_values measurement")
    points = circle_points(measurement)
    return measurement, [math.atan2(y - measurement.get("center", (0, 0))[1], x - measurement.get("center", (0, 0))[0])
                         for x, y in points]


def _circle_coefficient(raw, case, identifier, field, step, conjugate, order):
    """The stator-frame coefficient of exp(-j m theta) of a field sampled on a circle.

    A step solved with conjugated excitation (its rotor-frame frequency was
    negative) carries the order's response as the conjugate of its -m component.
    Br and Btheta are formed from B1 and B2 before the transform.
    """
    measurement, angles = _circle(case, identifier)
    def sample(k, theta):
        if field in ("Br", "Btheta"):
            b1, b2 = raw[(step, f"{identifier}:B1", k)], raw[(step, f"{identifier}:B2", k)]
            return b1 * math.cos(theta) + b2 * math.sin(theta) if field == "Br" else -b1 * math.sin(theta) + b2 * math.cos(theta)
        return raw[(step, f"{identifier}:{field}", k)]
    k_order = -order if conjugate else order
    value = sum(sample(k, t) * cmath.exp(1j * k_order * t) for k, t in enumerate(angles)) / len(angles)
    return value.conjugate() if conjugate else value


def _gap_torque(order, a_in, r_in, a_out, r_out, depth):
    """Time-average torque of one order from the potential on two circles of a current-free gap.

    In the gap a(r) = alpha r^n + beta r^-n; B_r = -j m a / r and B_theta = -a'(r),
    and the stress tensor gives pi r^2 depth Re(B_r conj(B_theta)) / mu0.
    """
    n = abs(order)
    det = r_in ** n * r_out ** -n - r_in ** -n * r_out ** n
    alpha = (a_in * r_out ** -n - a_out * r_in ** -n) / det
    beta = (r_in ** n * a_out - r_out ** n * a_in) / det
    radial = -1j * order * a_in / r_in
    tangential = -n * (alpha * r_in ** (n - 1) - beta * r_in ** (-n - 1))
    return math.pi * r_in ** 2 * depth / MU0 * (radial * tangential.conjugate()).real


def _harmonic_superposition(assessor, case, raw, metrics, model):
    """A quantity of a linear problem with a rotationally uniform moving part, summed over spatial orders.

    Each spatial order m of the excitation reaches the moving part at its own
    frequency, omega - m Omega; the frozen steps solve each order at that
    frequency (conjugated when it is negative) and record circle samples. This
    sums one quantity over the orders of one operating point: the torque from
    the potential on two gap circles, the moving part's loss from each order's
    slip power, a conductor's loss from eddy-current samples on rings, a
    winding's voltage per turn from potential samples on rings, or a field
    component at a point.
    """
    from .reproduce import UNITS
    problem = model["problem"]
    scale = UNITS[problem["units"]]
    depth = problem.get("depth", 1) * scale
    omega = 2 * math.pi * assessor["supply_frequency_hz"]
    speed = assessor["speed_rad_s"]
    steps = _harmonic_steps(case, assessor["sweep"], speed)
    quantity = assessor["quantity"]
    rows, total = [], 0j
    for order in sorted(steps, key=lambda m: (abs(m), m)):
        step, conjugate = steps[order]
        coefficient = lambda identifier, field: _circle_coefficient(raw, case, identifier, field, step, conjugate, order)
        if quantity in ("torque", "moving_loss"):
            (inner, outer) = assessor["gap_circles"]
            r_in, r_out = (_circle(case, i)[0]["radius"] * scale for i in (inner, outer))
            torque = _gap_torque(order, coefficient(inner, "A"), r_in, coefficient(outer, "A"), r_out, depth)
            value = torque if quantity == "torque" else torque * (omega - order * speed) / order
        elif quantity == "conductor_loss":
            # Rings carry radial quadrature weights for the integral of f(r) r dr, in square metres, and may carry
            # their own conductivity (a rotor of two conductors); FEMM reports eddy-current density in MA/m^2.
            value = math.pi * depth * sum(
                ring["weight"] * abs(coefficient(ring["measurement"], "Je") * 1e6) ** 2
                / ring.get("conductivity_S_per_m", assessor.get("conductivity_S_per_m")) for ring in assessor["rings"])
        elif quantity == "voltage":
            winding = assessor["winding"]
            radial = sum(ring["weight"] * coefficient(ring["measurement"], "A") for ring in winding["rings"])
            inner, outer = winding["inner_radius"] * scale, winding["outer_radius"] * scale
            flux = 0j
            for start, end, sign in winding["belts"]:
                a, b = math.radians(start), math.radians(end)
                area = (b - a) / 2 * (outer ** 2 - inner ** 2)
                angular = (cmath.exp(-1j * order * b) - cmath.exp(-1j * order * a)) / (-1j * order)
                flux += sign * radial * angular / area
            value = 1j * omega * flux * depth
        elif quantity == "point_field":
            point = assessor["point"]
            component = coefficient(point["measurement"], point["component"] if point["component"] != "Htheta" else "Btheta")
            if point["component"] == "Htheta":
                component /= MU0 * point.get("relative_permeability", 1)
            value = component * cmath.exp(-1j * order * math.radians(point.get("angle_deg", 0)))
        else:
            raise ValueError("Unknown superposition quantity: " + str(quantity))
        total += value
        rows.append({"order": order, "step": step, "conjugate": conjugate,
                     "rotor_frame_frequency_hz": (omega - order * speed) / (2 * math.pi), "re": value.real, "im": value.imag})
    part = assessor.get("part", "abs" if quantity == "voltage" else "re")
    observed = {"re": total.real, "im": total.imag, "abs": abs(total)}[part]
    metric = assessor_metric(assessor)
    target = {key: assessor[key] for key in ("value", "relative_tolerance", "absolute_tolerance", "quote", "location",
                                              "used_for_acceptance", "allowance") if key in assessor}
    target.update(kind=assessor.get("value_kind"), metric=metric)
    target.setdefault("used_for_acceptance", True)
    error = abs(observed - assessor["value"])
    tolerance = max(abs(assessor["value"]) * assessor.get("relative_tolerance", 0), assessor.get("absolute_tolerance", 0))
    return {"status": "passed" if error <= tolerance else "failed", "assessor": "harmonic_superposition", "metric": metric,
            "statistic": f"sum over {len(rows)} orders", "observed": observed, "expected": assessor["value"],
            "absolute_error": error, "relative_error": error / abs(assessor["value"]) if assessor["value"] else None,
            "absolute_tolerance": tolerance, "samples": len(rows), "rows": rows, "target": target}


def _dq_linkage(assessor, case, raw, metrics, model):
    """d- and q-axis flux linkage or torque of one operating point, from measured phase flux linkages.

    Each step of the point records its currents and the electrical angle of the
    d-axis (the rotor's north-pole axis) from phase A's magnetic axis, measured
    in the direction of the phase sequence: {"dq": {"point", "id", "iq",
    "electrical_angle_deg"}}. The amplitude-invariant Park transform gives
    psi_d = 2/3 sum psi_k cos(theta - k 120 deg) and psi_q = -2/3 sum psi_k
    sin(theta - k 120 deg); torque is 3/2 p (psi_d i_q - psi_q i_d), or the
    stress-tensor torque of a block-integral or air-gap measurement turned into
    the direction of the phase sequence. A model of some of the poles scales the
    flux linkages and the measured torque up to the machine (flux_scale,
    torque_scale). Values are averaged over the point's rotor positions, or
    taken at its first position.

    ``L_d`` and ``L_q`` are inductances from single-axis currents, as benchmarks
    separate them: (mean psi_d at the point - mean psi_d at ``reference_point``,
    usually no load) / i_d, and the same for q; they do not depend on the
    transform's scaling. ``transform: power_invariant`` reports psi_d and psi_q
    on the power-invariant scale (the square root of 3/2 times the
    amplitude-invariant values), as some benchmarks state magnet flux.
    """
    point, quantity = assessor["point"], assessor["quantity"]
    if assessor.get("transform", "amplitude_invariant") not in ("amplitude_invariant", "power_invariant"):
        raise ValueError("The dq transform is amplitude_invariant or power_invariant")

    def point_rows(name):
        steps = [s for s in case.get("steps") or [] if (s.get("dq") or {}).get("point") == name]
        if not steps:
            raise ValueError(f"No steps carry the operating point {name}")
        if assessor.get("average", "positions") == "first":
            steps = steps[:1]
        phases, out = assessor["phases"], []
        for step in steps:
            dq = step["dq"]
            theta = math.radians(dq["electrical_angle_deg"])
            psi = [assessor.get("flux_scale", 1) * raw[(step["label"], phases[k], 2)].real for k in range(3)]
            d = 2 / 3 * sum(psi[k] * math.cos(theta - 2 * math.pi * k / 3) for k in range(3))
            q = -2 / 3 * sum(psi[k] * math.sin(theta - 2 * math.pi * k / 3) for k in range(3))
            out.append((step, dq, d, q))
        return out

    scale = math.sqrt(1.5) if assessor.get("transform") == "power_invariant" else 1
    rows = []
    for step, dq, d, q in point_rows(point):
        if quantity == "psi_d":
            value = scale * d
        elif quantity == "psi_q":
            value = scale * q
        elif quantity == "torque":
            value = 1.5 * assessor["pole_pairs"] * (d * dq["iq"] - q * dq["id"])
        elif quantity == "stress_torque":
            value = assessor["sequence_direction"] * assessor.get("torque_scale", 1) * raw[(step["label"], assessor["torque_measurement"], 0)].real
        elif quantity in ("L_d", "L_q"):
            value = d if quantity == "L_d" else q
        else:
            raise ValueError("Unknown dq quantity: " + str(quantity))
        rows.append({"step": step["label"], "electrical_angle_deg": dq["electrical_angle_deg"], "psi_d": d, "psi_q": q, "value": value})
    observed = sum(r["value"] for r in rows) / len(rows)
    if quantity in ("L_d", "L_q"):
        axis = "id" if quantity == "L_d" else "iq"
        currents = {dq[axis] for _, dq, _, _ in point_rows(point)}
        if len(currents) != 1 or not next(iter(currents)):
            raise ValueError(f"{quantity} needs one nonzero {axis} at every position of the point")
        reference = point_rows(assessor["reference_point"])
        base = sum(d if quantity == "L_d" else q for _, _, d, q in reference) / len(reference)
        observed = (observed - base) / next(iter(currents))
    metric = assessor_metric(assessor)
    target = {key: assessor[key] for key in ("value", "relative_tolerance", "absolute_tolerance", "quote", "location",
                                              "used_for_acceptance", "allowance") if key in assessor}
    target.update(kind=assessor.get("value_kind"), metric=metric)
    target.setdefault("used_for_acceptance", True)
    error = abs(observed - assessor["value"])
    tolerance = max(abs(assessor["value"]) * assessor.get("relative_tolerance", 0), assessor.get("absolute_tolerance", 0))
    spread = max(r["value"] for r in rows) - min(r["value"] for r in rows)
    return {"status": "passed" if error <= tolerance else "failed", "assessor": "dq_linkage", "metric": metric,
            "statistic": f"mean over {len(rows)} rotor positions", "observed": observed, "expected": assessor["value"],
            "absolute_error": error, "relative_error": error / abs(assessor["value"]) if assessor["value"] else None,
            "absolute_tolerance": tolerance, "samples": len(rows), "position_spread": spread, "rows": rows, "target": target}


def _metric_relation(assessor, case, raw, metrics, model):
    """Values that a symmetry of the construction makes equal: metric@step = factor * reference@step for each pair.

    The observed value is the largest deviation, |value - factor * reference|,
    relative to the largest |factor * reference|; the frozen value is zero.
    """
    pairs = assessor["pairs"]
    if not pairs:
        raise ValueError("A metric relation needs at least one pair")
    rows = []
    for pair in pairs:
        value, reference = metrics[pair["metric"]], metrics[pair["reference"]]
        rows.append({"metric": pair["metric"], "reference": pair["reference"], "factor": pair.get("factor", 1),
                     "value": value, "expected_value": pair.get("factor", 1) * reference})
    scale = max(abs(r["expected_value"]) for r in rows)
    if not scale:
        raise ValueError("Every reference value is zero")
    for r in rows:
        r["deviation"] = abs(r["value"] - r["expected_value"]) / scale
    observed = max(r["deviation"] for r in rows)
    target = {key: assessor[key] for key in ("value", "absolute_tolerance", "quote", "location", "used_for_acceptance") if key in assessor}
    target.update(kind=assessor.get("value_kind"), metric=assessor_metric(assessor))
    target.setdefault("used_for_acceptance", True)
    tolerance = assessor["absolute_tolerance"]
    return {"status": "passed" if abs(observed - assessor.get("value", 0)) <= tolerance else "failed", "assessor": "metric_relation",
            "metric": assessor_metric(assessor), "statistic": f"largest relative deviation over {len(rows)} pairs",
            "observed": observed, "expected": assessor.get("value", 0), "absolute_error": observed, "relative_error": None,
            "absolute_tolerance": tolerance, "samples": len(rows), "rows": rows, "target": target}


def _parallel_branches(assessor, case, raw, metrics, model):
    """Coupled conductors connected in parallel across one voltage, from solves that drive one branch each.

    Step k carries current only in branch k (the others open, eddy currents
    allowed); the circuit measurements give Z[j][k] = V_j / I_k. A common voltage
    U then drives I = U Z^-1 1, and the total current's part in phase with U, or
    lagging it, is compared with the frozen value. In a linear problem this is
    the parallel connection exactly (double-cage bars, parallel strands).
    """
    circuits, steps = assessor["circuits"], assessor["steps"]
    n = len(circuits)
    if n < 2 or len(steps) != n:
        raise ValueError("Give two or more circuits and one driving step per circuit")
    z = [[raw[(steps[k], circuits[j], 1)] / raw[(steps[k], circuits[k], 0)] for k in range(n)] for j in range(n)]
    # Gaussian elimination with partial pivoting on [Z | U 1].
    voltage = assessor["voltage"]
    rows = [z[j][:] + [complex(voltage)] for j in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if not abs(rows[pivot][col]):
            raise ValueError("The impedance matrix is singular")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(n):
            if r != col:
                factor = rows[r][col] / rows[col][col]
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[col])]
    currents = [rows[j][n] / rows[j][j] for j in range(n)]
    total = sum(currents)
    part = assessor.get("part", "in_phase")
    observed = {"in_phase": total.real, "lagging": -total.imag, "abs": abs(total)}[part]
    target = {key: assessor[key] for key in ("value", "relative_tolerance", "absolute_tolerance", "quote", "location",
                                              "used_for_acceptance") if key in assessor}
    target.update(kind=assessor.get("value_kind"), metric=assessor_metric(assessor))
    target.setdefault("used_for_acceptance", True)
    error = abs(observed - assessor["value"])
    tolerance = max(abs(assessor["value"]) * assessor.get("relative_tolerance", 0), assessor.get("absolute_tolerance", 0))
    return {"status": "passed" if error <= tolerance else "failed", "assessor": "parallel_branches", "metric": assessor_metric(assessor),
            "statistic": f"{part} part of the total current at {voltage} V", "observed": observed, "expected": assessor["value"],
            "absolute_error": error, "relative_error": error / abs(assessor["value"]) if assessor["value"] else None,
            "absolute_tolerance": tolerance, "samples": n,
            "rows": [{"circuit": c, "current_re": i.real, "current_im": i.imag} for c, i in zip(circuits, currents)], "target": target}


def _step_statistic(assessor, case, raw, metrics, model):
    """A statistic of one metric over the steps whose labels start with ``step_prefix``, against a frozen value.

    For sweeps that are not an even rotor period, such as a locked-rotor test
    at the angles the paper used: ``maximum`` (a curve's peak, as the paper
    reads it from the same samples), ``minimum`` or ``mean``.
    """
    prefix, statistic = assessor["step_prefix"], assessor["statistic"]
    labels = [step["label"] for step in case.get("steps") or [] if step["label"].startswith(prefix)]
    if not prefix or not labels:
        raise ValueError(f"No steps start with {prefix!r}")
    values = [metrics[assessor["metric"] + "@" + label] for label in labels]
    if statistic == "maximum":
        index = max(range(len(values)), key=values.__getitem__)
    elif statistic == "minimum":
        index = min(range(len(values)), key=values.__getitem__)
    elif statistic != "mean":
        raise ValueError("Unknown step statistic: " + statistic)
    observed = sum(values) / len(values) if statistic == "mean" else values[index]
    observed *= assessor.get("scale", 1)
    target = {key: assessor[key] for key in ("value", "relative_tolerance", "absolute_tolerance", "quote", "location",
                                              "used_for_acceptance", "allowance") if key in assessor}
    target.update(kind=assessor.get("value_kind"), metric=assessor_metric(assessor))
    target.setdefault("used_for_acceptance", True)
    error = abs(observed - assessor["value"])
    tolerance = max(abs(assessor["value"]) * assessor.get("relative_tolerance", 0), assessor.get("absolute_tolerance", 0))
    return {"status": "passed" if error <= tolerance else "failed", "assessor": "step_statistic",
            "metric": assessor_metric(assessor), "statistic": f"{statistic} over {len(values)} steps",
            "observed": observed, "expected": assessor["value"], "absolute_error": error,
            "relative_error": error / abs(assessor["value"]) if assessor["value"] else None,
            "absolute_tolerance": tolerance, "samples": len(values),
            **({"at_step": labels[index]} if statistic != "mean" else {}), "target": target}


def assessor_metric(assessor):
    """The metric name an assessor reports its result under, or None for kinds that report only their kind."""
    kind = assessor.get("kind")
    # A label names an assessor apart from others of the same quantity (a paper's FE value and its measurement).
    label = f":{assessor['label']}" if assessor.get("label") else ""
    if kind == "waveform_statistic":
        order = f":h{assessor.get('order', 1)}" if assessor["statistic"] == "harmonic_peak" else ""
        return f"{assessor['metric']}:{assessor['statistic']}{order}{label}"
    if kind == "step_statistic":
        return f"{assessor['metric']}:{assessor['statistic']}:{assessor['step_prefix']}{label}"
    if kind == "spatial_harmonic":
        return f"{assessor['measurement']}:{assessor['field']}:h{assessor['order']}{label}"
    if kind == "harmonic_superposition":
        quantity = assessor["quantity"]
        part = assessor.get("part", "abs" if quantity == "voltage" else "re")
        label = f":{assessor['point']['label']}" if quantity == "point_field" else ""
        return f"{assessor['sweep']}@{assessor['speed_rad_s']:g}:{quantity}{label}:{part}"
    if kind in ("metric_relation", "parallel_branches"):
        return assessor["name"]
    if kind == "dq_linkage":
        return f"{assessor['point']}:{assessor['quantity']}" + (":first_position" if assessor.get("average") == "first" else "") + label
    return None


ASSESSORS = {"tube_bessel": _tube, "dipole_torque": _dipole_torque, "applied_field": _applied_field,
             "phase_linkage": _phase_linkage, "waveform_statistic": _waveform_statistic, "spatial_harmonic": _spatial_harmonic,
             "harmonic_superposition": _harmonic_superposition, "dq_linkage": _dq_linkage,
             "metric_relation": _metric_relation, "parallel_branches": _parallel_branches, "step_statistic": _step_statistic}


def run_assessor(assessor, case, raw, metrics, model):
    try:
        return ASSESSORS[assessor["kind"]](assessor, case, raw, metrics, model)
    except (KeyError, ValueError, ZeroDivisionError, StopIteration) as exc:
        return {"status": "failed", "assessor": assessor.get("kind"), "error": f"{type(exc).__name__}: {exc}",
                "target": {"used_for_acceptance": True}}
