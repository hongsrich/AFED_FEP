"""Center-of-mass restraints for host-guest / protein-ligand complex legs.

When a guest/ligand is decoupled it would otherwise drift out of the binding
site, making the complex leg diverge. A harmonic (or flat-bottom) COM restraint
between the host/protein and the guest keeps it localized; the analytical
restraint free energy is added back so the cycle stays exact.

Used by the RBFE complex leg. Kept dependency-light (pure OpenMM CustomCentroid
BondForce) so no extra packages are needed.
"""

import math

import openmm
from openmm import unit


def add_harmonic_com_restraint(system, host_atoms, guest_atoms,
                               k_kcal_per_mol_per_A2=10.0, r0_nm=0.0):
    """Add a harmonic COM-COM distance restraint; returns the Force index.

    k in kcal/mol/A^2 (converted internally), r0 the equilibrium COM separation
    (nm). 0.5*k*(d - r0)^2 between the two group centroids (mass-weighted).
    """
    k = (k_kcal_per_mol_per_A2 * 4.184 * 100.0)  # kcal/mol/A^2 -> kJ/mol/nm^2
    force = openmm.CustomCentroidBondForce(2, "0.5*k*(distance(g1,g2)-r0)^2")
    force.addGlobalParameter("k", k)
    force.addGlobalParameter("r0", r0_nm)
    force.addGroup([int(i) for i in host_atoms])
    force.addGroup([int(i) for i in guest_atoms])
    force.addBond([0, 1], [])
    system.addForce(force)
    return system.getNumForces() - 1


def add_flat_bottom_com_restraint(system, host_atoms, guest_atoms,
                                  k_kcal_per_mol_per_A2=10.0, tol_nm=0.5):
    """Flat-bottom COM restraint: zero force within tol_nm, harmonic beyond.

    Lets the guest explore the pocket freely up to tol_nm before being pushed
    back -- gentler than a stiff harmonic for binding-site sampling.
    """
    k = (k_kcal_per_mol_per_A2 * 4.184 * 100.0)
    expr = "0.5*k*step(d-tol)*(d-tol)^2; d=distance(g1,g2)"
    force = openmm.CustomCentroidBondForce(2, expr)
    force.addGlobalParameter("k", k)
    force.addGlobalParameter("tol", tol_nm)
    force.addGroup([int(i) for i in host_atoms])
    force.addGroup([int(i) for i in guest_atoms])
    force.addBond([0, 1], [])
    system.addForce(force)
    return system.getNumForces() - 1


def harmonic_restraint_free_energy(k_kcal_per_mol_per_A2, temperature=298.15):
    """Standard-state free energy of releasing a harmonic COM restraint (kcal/mol).

    dG_restraint = -kT ln( (V_standard) / (2 pi kT / k)^(3/2) ) for a 3D harmonic
    well; used to add the restraint contribution back into the binding cycle.
    Returns kcal/mol. (Approximate: assumes an isotropic 3D harmonic well.)
    """
    kB_kcal = 0.0019872041  # kcal/mol/K
    kT = kB_kcal * temperature
    k = k_kcal_per_mol_per_A2  # kcal/mol/A^2
    # well volume (A^3) of a 3D Gaussian; V_standard = 1660.5 A^3 (1 M).
    v_well = (2.0 * math.pi * kT / k) ** 1.5
    v_standard = 1660.5
    return -kT * math.log(v_standard / v_well)
