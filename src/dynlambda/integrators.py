"""Integrator construction: plain Langevin or a RESPA/MTS Langevin scheme.

Multiple-time-step (RESPA) integration evaluates the expensive nonbonded forces
(PME reciprocal space, LJ, the alchemical CustomNonbondedForces) once per OUTER
step while the cheap bonded forces are evaluated several times per outer step on
a short inner timestep. Combined with hydrogen mass repartitioning (see
molecule_setup.build_from_smiles) this lets the outer timestep reach 3-4 fs --
roughly doubling throughput -- without the fast bonded/angle motions blowing up.

This mirrors AToM-OpenMM's scheme (Gallicchio-Lab/AToM-OpenMM, ommsystem.py):
    bonded_frequency = round(outer_dt / 1 fs)
    fgroups = [(bonded_group, bonded_frequency), (nonbonded_group, 1)]
with a ~1 fs inner step.
"""

import warnings

import openmm
from openmm import unit

# Force groups used by the MTS split.
FAST_GROUP = 0   # bonded: HarmonicBond/Angle, PeriodicTorsion, alchemical bonds
SLOW_GROUP = 1   # nonbonded: NonbondedForce + alchemical CustomNonbondedForces


def assign_mts_force_groups(system, slow_group=SLOW_GROUP, fast_group=FAST_GROUP):
    """Split forces into a slow (nonbonded) and fast (everything else) group.

    Any force whose class name contains 'Nonbonded' (NonbondedForce and the
    alchemical CustomNonbondedForces) goes to the slow group; all others
    (bonded, torsions, alchemical CustomBondForces, CMMotionRemover) go to the
    fast group. Returns (slow_group, fast_group). Mutates ``system`` in place, so
    call BEFORE building the Context.
    """
    for force in system.getForces():
        if "Nonbonded" in force.__class__.__name__:
            force.setForceGroup(slow_group)
        else:
            force.setForceGroup(fast_group)
    return slow_group, fast_group


def make_integrator(system, temperature, friction_per_ps, timestep_fs,
                    kind="langevin", mts_inner_steps=None):
    """Build the MD integrator.

    kind="langevin": LangevinMiddleIntegrator at ``timestep_fs``.
    kind="mts":      MTSLangevinIntegrator with the nonbonded forces evaluated
                     once per outer step and bonded forces ``mts_inner_steps``
                     times (inner step = timestep_fs / mts_inner_steps, ~1 fs by
                     default, following AToM). Assigns force groups on ``system``.

    For kind="mts" the outer ``timestep_fs`` is the large step (use 3-4 fs with
    HMR). Returns the integrator.
    """
    T = temperature * unit.kelvin
    gamma = friction_per_ps / unit.picosecond

    if kind == "langevin":
        if timestep_fs > 2.5:
            warnings.warn(
                f"timestep {timestep_fs} fs with a plain Langevin integrator and "
                "no MTS is aggressive; use kind='mts' and HMR for >2.5 fs.",
                RuntimeWarning,
            )
        return openmm.LangevinMiddleIntegrator(
            T, gamma, timestep_fs * unit.femtosecond)

    if kind == "mts":
        if mts_inner_steps is None:
            # ~1 fs inner step, AToM-style (bonded_frequency = round(dt/1fs)).
            mts_inner_steps = max(1, int(round(timestep_fs)))
        slow_group, fast_group = assign_mts_force_groups(system)
        # groups ordered slowest (substeps=1) -> fastest; nonbonded once per outer
        # step, bonded mts_inner_steps times.
        groups = [(slow_group, 1), (fast_group, int(mts_inner_steps))]
        return openmm.MTSLangevinIntegrator(
            T, gamma, timestep_fs * unit.femtosecond, groups)

    raise ValueError(f"unknown integrator kind {kind!r}")
