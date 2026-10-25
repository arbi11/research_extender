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


def _dipole_torque(assessor, case, raw, metrics, model):
    """Torque on a rigid dipole, m x B, from the native field measured with the magnet removed."""
    field = assessor["field_step"]
    bx = metrics[assessor["field_metrics"][0] + "@" + field]
    by = metrics[assessor["field_metrics"][1] + "@" + field]
    magnitude = math.hypot(bx, by)
    magnet = next(label for label in model["labels"] if label.get("material") == assessor["magnet_material"])
    base = magnet["magdir"]
    moment = assessor["moment_A_m2"]
    tolerance = assessor["relative_tolerance"] * moment * magnitude
    rows, ok = [], True
    for step, angle in assessor["rotation_steps"].items():
        direction = math.radians(base + angle)
        expected = moment * (math.cos(direction) * by - math.sin(direction) * bx)
        observed = metrics[assessor["torque_metric"] + "@" + step]
        error = abs(observed - expected)
        rows.append({"step": step, "rotor_angle_deg": angle, "expected_N_m": expected, "observed_N_m": observed,
                     "absolute_error": error, "passed": error <= tolerance})
        ok &= error <= tolerance
    field_ok = abs(magnitude - assessor["applied_field_T"]) <= assessor["relative_tolerance"] * assessor["applied_field_T"]
    return {"status": "passed" if ok and field_ok else "failed", "assessor": "dipole_torque",
            "measured_applied_field_T": [bx, by], "applied_field_magnitude_T": magnitude, "applied_field_check": field_ok,
            "magnet_direction_deg": base, "absolute_tolerance_N_m": tolerance, "rows": rows,
            "convention": "Rotor angle rotates the inner air-gap side counterclockwise; torque is m x B about +z.",
            "target": {"used_for_acceptance": True, "kind": "analytic", "quote": assessor.get("quote")}}


ASSESSORS = {"tube_bessel": _tube, "dipole_torque": _dipole_torque}


def run_assessor(assessor, case, raw, metrics, model):
    try:
        return ASSESSORS[assessor["kind"]](assessor, case, raw, metrics, model)
    except (KeyError, ValueError, ZeroDivisionError, StopIteration) as exc:
        return {"status": "failed", "assessor": assessor.get("kind"), "error": f"{type(exc).__name__}: {exc}",
                "target": {"used_for_acceptance": True}}
