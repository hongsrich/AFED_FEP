"""Hard-coded experimental reference data and derived relative references.

All values in kcal/mol. Sources are recorded per entry. Derived relative values
are computed from the absolutes using the same sign conventions as
free_energy.py, so tests can check the cycle math against these.

Sources
-------
- Hydration free energies: FreeSolv database (Mobley & Guthrie, J Comput Aided
  Mol Des 2014; doi:10.1007/s10822-014-9747-x).
- SAMPL6 host-guest (octa-acid OA) binding: SAMPL6 SAMPLing challenge reference
  values (Rizzi et al., J Comput Aided Mol Des 2020; doi:10.1007/s10822-020-00290-5).
- T4 lysozyme L99A binding: experimental values collated in the alchemical
  binding literature (e.g. Mobley et al., J Mol Biol 2007; Gilson/Boyce datasets).
"""

from collections import namedtuple
import math

Reference = namedtuple("Reference", ["value", "uncertainty", "source"])

_FREESOLV = "FreeSolv (Mobley & Guthrie 2014)"
_SAMPL6 = "SAMPL6 SAMPLing (Rizzi et al. 2020)"
_T4 = "T4 L99A experimental (alchemical binding literature)"

# --- absolute hydration free energies (kcal/mol) ------------------------------
HYDRATION = {
    "methane": Reference(+2.00, 0.20, _FREESOLV),
    "methanol": Reference(-5.10, 0.60, _FREESOLV),
    "benzene": Reference(-0.90, 0.20, _FREESOLV),
    "toluene": Reference(-0.90, 0.20, _FREESOLV),
    "ethylbenzene": Reference(-0.79, 0.60, _FREESOLV),
    # extras validated in-house against the same FreeSolv set (not required by
    # BUILD2 but handy as additional checks).
    "ethane": Reference(+1.83, 0.20, _FREESOLV),
    "acetone": Reference(-3.85, 0.20, _FREESOLV),
    "phenol": Reference(-6.62, 0.20, _FREESOLV),
}

# --- absolute binding free energies (kcal/mol), keyed by (system, ligand) -----
BINDING = {
    ("sampl6_oa", "OA-G3"): Reference(-5.18, 0.02, _SAMPL6),
    ("sampl6_oa", "OA-G6"): Reference(-4.97, 0.02, _SAMPL6),
    ("t4_l99a", "benzene"): Reference(-5.19, 0.16, _T4),
    ("t4_l99a", "toluene"): Reference(-5.52, 0.06, _T4),
    ("t4_l99a", "ethylbenzene"): Reference(-5.76, 0.07, _T4),
}


def _combine(ua, ub):
    return math.sqrt(ua * ua + ub * ub)


def get_hydration_reference(name):
    """Return the Reference(value, uncertainty, source) for a molecule's HFE."""
    key = name.lower()
    if key not in HYDRATION:
        raise KeyError(f"no hydration reference for {name!r}; "
                       f"available: {sorted(HYDRATION)}")
    return HYDRATION[key]


def get_binding_reference(system, ligand):
    """Return the Reference for an absolute binding free energy."""
    key = (system.lower(), ligand)
    if key not in BINDING:
        raise KeyError(f"no binding reference for {key!r}; "
                       f"available: {sorted(BINDING)}")
    return BINDING[key]


def get_relative_hydration_reference(ligand_a, ligand_b):
    """ddG_hyd(A->B) = dG_hyd(B) - dG_hyd(A), uncertainty in quadrature."""
    a = get_hydration_reference(ligand_a)
    b = get_hydration_reference(ligand_b)
    return Reference(b.value - a.value, _combine(a.uncertainty, b.uncertainty),
                     f"derived from {_FREESOLV}")


def get_relative_binding_reference(system, ligand_a, ligand_b):
    """ddG_bind(A->B) = dG_bind(B) - dG_bind(A), uncertainty in quadrature."""
    a = get_binding_reference(system, ligand_a)
    b = get_binding_reference(system, ligand_b)
    src = a.source
    return Reference(b.value - a.value, _combine(a.uncertainty, b.uncertainty),
                     f"derived from {src}")


def list_available_references():
    """Return a dict summarizing all available reference keys."""
    return {
        "hydration": sorted(HYDRATION.keys()),
        "binding": sorted(f"{s}:{l}" for (s, l) in BINDING.keys()),
        "relative_hydration_examples": [
            "methane->methanol", "benzene->toluene", "toluene->ethylbenzene",
        ],
        "relative_binding_examples": [
            "sampl6_oa: OA-G3->OA-G6",
            "t4_l99a: benzene->toluene",
            "t4_l99a: toluene->ethylbenzene",
        ],
    }
