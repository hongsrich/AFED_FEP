"""MBAR analysis of the fixed-window HFE data.

Sign convention
---------------
The lambda schedule runs from the fully *coupled* solute (state 0:
lambda_elec=lambda_steric=1) to the fully *decoupled* solute (last state:
lambda_elec=lambda_steric=0). MBAR therefore reports

    Delta G_decouple = G(decoupled) - G(coupled)         [free energy F[0] -> F[-1]]

i.e. the free energy of *removing* the solute from solvent. The hydration free
energy is the reverse process (inserting the solute into water):

    Delta G_hyd = -Delta G_decouple .

A favorable (negative) hydration free energy thus corresponds to a positive
decoupling free energy. This is asserted in tests/test_mbar_analysis.py.
"""

from dataclasses import dataclass

import numpy as np
from openmm import unit

from .units import KJ_PER_KCAL, kT


@dataclass
class MBARResult:
    dg_decouple_kT: float          # F[-1] - F[0], in units of kT
    ddg_decouple_kT: float
    dg_decouple_kJ: float
    dg_hyd_kJ: float
    dg_hyd_kcal: float
    ddg_kJ: float
    ddg_kcal: float
    free_energies_kT: np.ndarray   # per-state F relative to state 0
    overlap: np.ndarray            # KxK overlap matrix (may be None)


def _flatten_u_kln(u_kln, N_k):
    """Convert (K,K,n) u_kln into MBAR's u_kn (K, sum N_k) layout."""
    K = u_kln.shape[0]
    N_total = int(np.sum(N_k))
    u_kn = np.zeros((K, N_total), dtype=np.float64)
    col = 0
    for k in range(K):
        nk = int(N_k[k])
        # samples from state k, evaluated in every state l
        u_kn[:, col:col + nk] = u_kln[k, :, :nk]
        col += nk
    return u_kn


def run_mbar(u_kln, N_k, temperature):
    """Run MBAR and return an MBARResult. Works with pymbar 3.x and 4.x."""
    from pymbar import MBAR

    N_k = np.asarray(N_k, dtype=int)
    u_kn = _flatten_u_kln(u_kln, N_k)

    mbar = MBAR(u_kn, N_k)

    # --- free energy differences (API differs between pymbar versions) -------
    # When states overlap extremely well (or sampling is short) pymbar 4's
    # asymptotic-covariance check can raise ParameterError. The free energies
    # themselves are still valid, so we retry without uncertainties and report
    # the uncertainty as NaN rather than crashing.
    if hasattr(mbar, "compute_free_energy_differences"):     # pymbar >= 4
        try:
            res = mbar.compute_free_energy_differences()
            Deltaf = np.asarray(res["Delta_f"])
            dDeltaf = np.asarray(res["dDelta_f"])
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            print(f"[analysis] uncertainty estimate unavailable ({exc}); "
                  f"reporting free energies without error bars.")
            res = mbar.compute_free_energy_differences(compute_uncertainty=False)
            Deltaf = np.asarray(res["Delta_f"])
            dDeltaf = np.full_like(Deltaf, np.nan)
    else:                                                     # pymbar 3.x
        Deltaf, dDeltaf = mbar.getFreeEnergyDifferences()[:2]
        Deltaf = np.asarray(Deltaf)
        dDeltaf = np.asarray(dDeltaf)

    # F relative to state 0, and the full coupled->decoupled difference.
    free_energies_kT = Deltaf[0, :]
    dg_decouple_kT = float(Deltaf[0, -1])
    ddg_decouple_kT = float(dDeltaf[0, -1])

    # --- overlap matrix -------------------------------------------------------
    overlap = None
    try:
        if hasattr(mbar, "compute_overlap"):
            overlap = np.asarray(mbar.compute_overlap()["matrix"])
        else:
            overlap = np.asarray(mbar.computeOverlap()["matrix"])
    except Exception:  # noqa: BLE001 - overlap is diagnostic, not essential
        overlap = None

    kT_kJ = kT(temperature).value_in_unit(unit.kilojoule_per_mole)
    dg_decouple_kJ = dg_decouple_kT * kT_kJ
    ddg_kJ = ddg_decouple_kT * kT_kJ

    # Hydration free energy = -(decoupling free energy).
    dg_hyd_kJ = -dg_decouple_kJ

    return MBARResult(
        dg_decouple_kT=dg_decouple_kT,
        ddg_decouple_kT=ddg_decouple_kT,
        dg_decouple_kJ=dg_decouple_kJ,
        dg_hyd_kJ=dg_hyd_kJ,
        dg_hyd_kcal=dg_hyd_kJ / KJ_PER_KCAL,
        ddg_kJ=ddg_kJ,
        ddg_kcal=ddg_kJ / KJ_PER_KCAL,
        free_energies_kT=free_energies_kT,
        overlap=overlap,
    )


def per_window_dg_kT(result):
    """Per-window free energy increments F[k+1]-F[k] in kT."""
    f = result.free_energies_kT
    return np.diff(f)


def summary_string(result):
    r = result
    lines = [
        "MBAR hydration free energy summary",
        "----------------------------------",
        f"Delta G_decouple = {r.dg_decouple_kJ:8.3f} +/- {r.ddg_kJ:.3f} kJ/mol "
        f"({r.dg_decouple_kJ / KJ_PER_KCAL:7.3f} kcal/mol)",
        f"Delta G_hyd      = {r.dg_hyd_kJ:8.3f} +/- {r.ddg_kJ:.3f} kJ/mol "
        f"({r.dg_hyd_kcal:7.3f} +/- {r.ddg_kcal:.3f} kcal/mol)",
        "  (sign: Delta G_hyd = -Delta G_decouple)",
    ]
    return "\n".join(lines)
