"""Unit helpers and thermodynamic constants.

We standardize on OpenMM units (kJ/mol, nm, ps, kelvin) internally and convert
to kcal/mol only for human-readable reporting.
"""

from openmm import unit

# Boltzmann constant in kJ/mol/K (OpenMM molar gas constant).
KB = unit.MOLAR_GAS_CONSTANT_R  # 0.0083145 kJ/mol/K

KJ_PER_KCAL = 4.184


def kT(temperature):
    """Return kB*T as an OpenMM Quantity in kJ/mol.

    temperature may be a plain float (kelvin) or an OpenMM Quantity.
    """
    if unit.is_quantity(temperature):
        T = temperature
    else:
        T = temperature * unit.kelvin
    return KB * T


def beta(temperature):
    """Return 1/(kB*T) as an OpenMM Quantity in mol/kJ."""
    return 1.0 / kT(temperature)


def kj_to_kcal(x):
    """Convert a number in kJ/mol to kcal/mol (plain float)."""
    return float(x) / KJ_PER_KCAL


def kcal_to_kj(x):
    """Convert a number in kcal/mol to kJ/mol (plain float)."""
    return float(x) * KJ_PER_KCAL
