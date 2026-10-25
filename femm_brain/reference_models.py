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


ASSESSORS = {"tube_bessel": _tube, "dipole_torque": _dipole_torque, "applied_field": _applied_field}


def run_assessor(assessor, case, raw, metrics, model):
    try:
        return ASSESSORS[assessor["kind"]](assessor, case, raw, metrics, model)
    except (KeyError, ValueError, ZeroDivisionError, StopIteration) as exc:
        return {"status": "failed", "assessor": assessor.get("kind"), "error": f"{type(exc).__name__}: {exc}",
                "target": {"used_for_acceptance": True}}
