"""Free-energy result container and reusable thermodynamic-cycle math.

This module centralizes the *sign conventions* for the whole package so that the
absolute-hydration, relative-hydration, and relative-binding pipelines all agree
and are testable in isolation (see tests/test_relative_cycle_signs.py).

Sign conventions (stated once, used everywhere)
-----------------------------------------------
Absolute hydration (gas -> water), from a decoupling simulation:
    dG_hyd(A) = -dG_decouple_water(A)
A favorable (negative) hydration free energy <=> positive decoupling free energy.

Relative hydration A -> B in water:
    ddG_hyd(A->B) = dG_hyd(B) - dG_hyd(A)

Relative binding A -> B (double-decoupling / alchemical cycle):
    ddG_bind(A->B) = dG_complex(A->B) - dG_solvent(A->B)
where each leg is the alchemical A->B transformation in that environment.
"""

from dataclasses import dataclass, field
import math

from .units import KJ_PER_KCAL

# Canonical sign-convention strings (stored in every result for provenance).
SIGN_HYDRATION = "dG_hyd = -dG_decouple_water"
SIGN_RELATIVE_HYDRATION = "ddG_hyd(A->B) = dG_hyd(B) - dG_hyd(A)"
SIGN_RELATIVE_BINDING = "ddG_bind(A->B) = dG_complex(A->B) - dG_solvent(A->B)"

# Allowed leg types.
LEG_TYPES = ("hydration", "relative_hydration", "complex", "solvent", "rbfe")


@dataclass
class FreeEnergyResult:
    """A single free-energy estimate plus full provenance.

    delta_g_kcal_mol / uncertainty_kcal_mol are the headline numbers; everything
    else records *how* it was produced so a result is self-describing in CSV/JSON.
    """

    name: str
    leg_type: str                         # one of LEG_TYPES
    delta_g_kcal_mol: float
    uncertainty_kcal_mol: float = float("nan")
    sign_convention: str = ""
    method: str = "MBAR"                   # MBAR | TI | dynamic_lambda | reference | derived
    n_windows: int = 0
    n_samples: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.leg_type not in LEG_TYPES:
            raise ValueError(
                f"leg_type {self.leg_type!r} not in {LEG_TYPES}")

    @property
    def delta_g_kj_mol(self):
        return self.delta_g_kcal_mol * KJ_PER_KCAL

    def as_dict(self):
        d = dict(self.__dict__)
        d["delta_g_kj_mol"] = self.delta_g_kj_mol
        return d

    def __str__(self):
        return (f"{self.name} [{self.leg_type}/{self.method}] "
                f"dG = {self.delta_g_kcal_mol:+.2f} +/- "
                f"{self.uncertainty_kcal_mol:.2f} kcal/mol")


def _combine_uncertainty(*uncertainties):
    """Propagate independent uncertainties in quadrature; NaNs are ignored."""
    vals = [u for u in uncertainties if u is not None and not math.isnan(u)]
    if not vals:
        return float("nan")
    return math.sqrt(sum(u * u for u in vals))


# --------------------------------------------------------------------------- #
# Sign-convention transforms (the single source of truth for the cycles)
# --------------------------------------------------------------------------- #

def hydration_from_decouple(dg_decouple_kcal, uncertainty_kcal=float("nan"),
                            name="molecule", method="MBAR", **meta):
    """dG_hyd = -dG_decouple_water. Returns a 'hydration' FreeEnergyResult."""
    return FreeEnergyResult(
        name=name,
        leg_type="hydration",
        delta_g_kcal_mol=-float(dg_decouple_kcal),
        uncertainty_kcal_mol=float(uncertainty_kcal),
        sign_convention=SIGN_HYDRATION,
        method=method,
        metadata=dict(dg_decouple_kcal_mol=float(dg_decouple_kcal), **meta),
    )


def relative_hydration_from_absolute(result_a, result_b, name=None):
    """ddG_hyd(A->B) = dG_hyd(B) - dG_hyd(A) from two absolute hydration results.

    Accepts FreeEnergyResult objects or plain (value, uncertainty) floats.
    """
    ga, ua, na = _unpack(result_a)
    gb, ub, nb = _unpack(result_b)
    if name is None:
        name = f"{na}->{nb}"
    return FreeEnergyResult(
        name=name,
        leg_type="relative_hydration",
        delta_g_kcal_mol=gb - ga,
        uncertainty_kcal_mol=_combine_uncertainty(ua, ub),
        sign_convention=SIGN_RELATIVE_HYDRATION,
        method="derived",
        metadata=dict(dg_hyd_A=ga, dg_hyd_B=gb, ligand_a=na, ligand_b=nb),
    )


def relative_binding_from_legs(complex_leg, solvent_leg, name=None):
    """ddG_bind(A->B) = dG_complex(A->B) - dG_solvent(A->B).

    Each argument is the A->B alchemical transformation free energy in that
    environment (FreeEnergyResult or (value, uncertainty) float).
    """
    gc, uc, nc = _unpack(complex_leg)
    gs, us, ns = _unpack(solvent_leg)
    if name is None:
        name = nc if nc != "molecule" else "rbfe"
    return FreeEnergyResult(
        name=name,
        leg_type="rbfe",
        delta_g_kcal_mol=gc - gs,
        uncertainty_kcal_mol=_combine_uncertainty(uc, us),
        sign_convention=SIGN_RELATIVE_BINDING,
        method="derived",
        metadata=dict(dg_complex=gc, dg_solvent=gs),
    )


def _unpack(x):
    """Return (value, uncertainty, name) from a FreeEnergyResult or a tuple."""
    if isinstance(x, FreeEnergyResult):
        return x.delta_g_kcal_mol, x.uncertainty_kcal_mol, x.name
    if isinstance(x, (tuple, list)):
        val = float(x[0])
        unc = float(x[1]) if len(x) > 1 else float("nan")
        return val, unc, "molecule"
    return float(x), float("nan"), "molecule"
