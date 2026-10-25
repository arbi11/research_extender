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
    """Modified Bessel function of the first kind for order 0 or 1, complex z."""
    if order not in (0, 1):
        raise ValueError("Only orders 0 and 1 are implemented")
    quarter = z * z / 4
    term = (z / 2) ** order
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

    The span is one electrical period, or half of one when the assessor declares
    half-wave symmetry (a pole pitch negates every linkage). ``step_prefix``
    keeps only the sweep's steps when others (an inductance step) follow it.
    """
    group = assessor["rotor_group"]
    period = 360 / assessor["pole_pairs"]
    span = period / 2 if assessor.get("half_wave_symmetry") else period
    alpha, angles = 0.0, {}
    for step in case["steps"]:
        alpha += sum(c["angle"] for c in step.get("modify", []) if c["function"] == "rotate_group" and c["group"] == group)
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
    half-wave symmetry); ``peak_to_peak`` and ``mean`` use the samples as they are.
    """
    statistic = assessor["statistic"]
    samples, span = _rotor_samples(assessor, case)
    p = assessor["pole_pairs"]
    values = [(a, metrics[assessor["metric"] + "@" + label]) for a, label in samples]
    if statistic in ("fundamental_rms", "fundamental_peak"):
        points = values + ([(a + span, -v) for a, v in values] if assessor.get("half_wave_symmetry") else [])
        amplitude = abs(2 / len(points) * sum(v * cmath.exp(1j * math.radians(p * a)) for a, v in points))
        observed = amplitude / math.sqrt(2) if statistic == "fundamental_rms" else amplitude
    elif statistic == "peak_to_peak":
        observed = max(v for _, v in values) - min(v for _, v in values)
    elif statistic == "mean":
        observed = sum(v for _, v in values) / len(values)
    else:
        raise ValueError("Unknown waveform statistic: " + statistic)
    target = {key: assessor[key] for key in ("value", "relative_tolerance", "absolute_tolerance", "quote", "location",
                                              "used_for_acceptance", "allowance") if key in assessor}
    target.update(kind=assessor.get("value_kind"), metric=f"{assessor['metric']}:{statistic}")
    target.setdefault("used_for_acceptance", True)
    error = abs(observed - assessor["value"])
    tolerance = max(abs(assessor["value"]) * assessor.get("relative_tolerance", 0), assessor.get("absolute_tolerance", 0))
    return {"status": "passed" if error <= tolerance else "failed", "assessor": "waveform_statistic",
            "metric": f"{assessor['metric']}:{statistic}", "statistic": statistic, "observed": observed,
            "expected": assessor["value"], "absolute_error": error,
            "relative_error": error / abs(assessor["value"]) if assessor["value"] else None,
            "absolute_tolerance": tolerance, "samples": len(values), "target": target}


def _phase_linkage(assessor, case, raw, metrics, model):
    """Winding sign and phase linkage: each phase's flux-linkage fundamental against the construction.

    Rotor angles come from the case's rotate_group steps. Samples must cover one
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
    # Counterclockwise rotation brings each phase to its peak one phase step after the previous one.
    predicted_order = all(_angle_difference(predicted[name] - predicted[sequence[0]], k * step_angle) <= 1e-6
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


ASSESSORS = {"tube_bessel": _tube, "dipole_torque": _dipole_torque, "applied_field": _applied_field,
             "phase_linkage": _phase_linkage, "waveform_statistic": _waveform_statistic}


def run_assessor(assessor, case, raw, metrics, model):
    try:
        return ASSESSORS[assessor["kind"]](assessor, case, raw, metrics, model)
    except (KeyError, ValueError, ZeroDivisionError, StopIteration) as exc:
        return {"status": "failed", "assessor": assessor.get("kind"), "error": f"{type(exc).__name__}: {exc}",
                "target": {"used_for_acceptance": True}}
