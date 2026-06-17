"""Result validation and reference comparison.

Guards two things BUILD2's acceptance criteria call out:
  * a tiny *smoke-test* result must never be silently reported as agreeing with
    experiment (compare_to_reference refuses to call it a benchmark);
  * every comparison carries the sign convention and run mode.
"""

import math

from .free_energy import FreeEnergyResult


def is_finite_result(result):
    """True if the result's delta_g is a finite number."""
    return math.isfinite(result.delta_g_kcal_mol)


def check_sign_convention(result):
    """Raise if a result is missing its sign-convention provenance."""
    if not result.sign_convention:
        raise ValueError(f"result {result.name!r} has no sign_convention set")
    return True


def compare_to_reference(result, reference, smoke_test=False):
    """Compare a computed FreeEnergyResult to a reference (value, uncertainty).

    Returns a dict with computed/reference/error/uncertainties and the run mode.
    If smoke_test is True the 'agrees' field is None and 'mode' is 'smoke_test'
    -- a short pipeline run is a code-path check, not an experimental comparison.
    """
    check_sign_convention(result)
    ref_val = reference.value if hasattr(reference, "value") else float(reference[0])
    ref_unc = (reference.uncertainty if hasattr(reference, "uncertainty")
               else (float(reference[1]) if len(reference) > 1 else float("nan")))

    error = result.delta_g_kcal_mol - ref_val
    out = {
        "name": result.name,
        "leg_type": result.leg_type,
        "method": result.method,
        "sign_convention": result.sign_convention,
        "computed_kcal_mol": result.delta_g_kcal_mol,
        "computed_uncertainty": result.uncertainty_kcal_mol,
        "reference_kcal_mol": ref_val,
        "reference_uncertainty": ref_unc,
        "error_kcal_mol": error,
        "mode": "smoke_test" if smoke_test else "benchmark",
        "finite": is_finite_result(result),
    }
    if smoke_test:
        out["agrees"] = None  # never claim experimental agreement for a smoke test
    else:
        tol = max(1.0, 2.0 * (ref_unc if math.isfinite(ref_unc) else 0.0))
        out["agrees"] = bool(math.isfinite(error) and abs(error) <= tol)
    return out


def summarize_comparison(cmp):
    """One-line human summary of a compare_to_reference dict."""
    tag = "[SMOKE TEST - not a benchmark]" if cmp["mode"] == "smoke_test" else ""
    agree = "" if cmp["agrees"] is None else f"  agrees={cmp['agrees']}"
    return (f"{cmp['name']} ({cmp['leg_type']}): computed "
            f"{cmp['computed_kcal_mol']:+.2f}  ref {cmp['reference_kcal_mol']:+.2f}"
            f"  err {cmp['error_kcal_mol']:+.2f} kcal/mol{agree} {tag}".rstrip())
