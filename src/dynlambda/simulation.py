"""Simulation drivers: contexts, minimization, and the fixed-window HFE loop.

The fixed-window driver runs one short MD simulation per lambda window and, for
each saved sample, evaluates the reduced potential in *every* window. That
matrix (u_kln) is exactly what MBAR consumes.
"""

import numpy as np
import openmm
from openmm import unit

from . import alchemy
from .units import beta


def make_langevin_context(system, platform, properties, temperature=298.15,
                          timestep_fs=2.0, friction_per_ps=1.0,
                          integrator_kind="langevin", mts_inner_steps=None):
    """Create an MD Context (plain Langevin or RESPA/MTS Langevin).

    integrator_kind="langevin" -> LangevinMiddleIntegrator (BAOAB).
    integrator_kind="mts"      -> MTSLangevinIntegrator; nonbonded forces are
                                  evaluated once per outer step and bonded forces
                                  mts_inner_steps times (~1 fs inner). Combine
                                  with HMR (hydrogen_mass_amu) for a 3-4 fs outer
                                  step. NOTE: this splits ``system`` into force
                                  groups in place, so the Context must be built
                                  from the same (mutated) system -- which it is.
    """
    from .integrators import make_integrator
    integrator = make_integrator(
        system, temperature, friction_per_ps, timestep_fs,
        kind=integrator_kind, mts_inner_steps=mts_inner_steps)
    context = openmm.Context(system, integrator, platform, properties)
    return context, integrator


def minimize(context, positions, max_iterations=200):
    context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=max_iterations)
    return context.getState(getPositions=True).getPositions()


def _reduced_potentials_all_states(context, alchemical_state, schedule, beta_value):
    """Reduced potentials of the current configuration in every lambda window.

    Returns a 1D array u_l of length len(schedule).
    """
    out = np.empty(len(schedule), dtype=np.float64)
    for l, (le, ls) in enumerate(schedule):
        alchemy.apply_lambdas(context, alchemical_state, le, ls)
        energy = context.getState(getEnergy=True).getPotentialEnergy()
        out[l] = beta_value * energy.value_in_unit(unit.kilojoule_per_mole)
    return out


def run_fixed_window_hfe(
    solvated,
    alchemical_system,
    alchemical_state,
    schedule,
    platform,
    properties,
    temperature=298.15,
    timestep_fs=2.0,
    friction_per_ps=1.0,
    equil_steps=2500,       # ~5 ps at 2 fs
    prod_steps=10000,       # ~20 ps at 2 fs
    sample_interval=500,    # collect every 1 ps
    minimize_first=True,
    seed=None,
    integrator_kind="langevin",
    mts_inner_steps=None,
    progress=True,
):
    """Run every lambda window and assemble the MBAR u_kln matrix.

    Returns dict with:
        u_kln  : (K, K, max_n) reduced potentials (state sampled, state evaluated)
        N_k    : (K,) number of samples collected per window
        schedule, temperature
    """
    K = len(schedule)
    beta_value = beta(temperature).value_in_unit(unit.mole / unit.kilojoule)

    context, integrator = make_langevin_context(
        alchemical_system, platform, properties,
        temperature=temperature, timestep_fs=timestep_fs,
        friction_per_ps=friction_per_ps,
        integrator_kind=integrator_kind, mts_inner_steps=mts_inner_steps,
    )
    if seed is not None:
        integrator.setRandomNumberSeed(int(seed))

    n_samples = prod_steps // sample_interval
    u_kln = np.zeros((K, K, n_samples), dtype=np.float64)
    N_k = np.zeros(K, dtype=int)

    positions = solvated.positions
    if minimize_first:
        # Minimize once at the fully coupled end state.
        alchemy.apply_lambdas(context, alchemical_state, 1.0, 1.0)
        positions = minimize(context, positions)

    for k, (le, ls) in enumerate(schedule):
        if progress:
            print(f"  window {k+1}/{K}  (elec={le:.3f}, steric={ls:.3f})")
        # Start each window from the previous configuration (sequential warmup).
        context.setPositions(positions)
        alchemy.apply_lambdas(context, alchemical_state, le, ls)
        # Relax any clash introduced by the lambda change before assigning
        # velocities; this keeps short equilibrations numerically stable.
        openmm.LocalEnergyMinimizer.minimize(context, maxIterations=100)
        context.setVelocitiesToTemperature(temperature * unit.kelvin)

        integrator.step(equil_steps)

        for s in range(n_samples):
            integrator.step(sample_interval)
            # Reduced potentials in all states for this configuration.
            u_kln[k, :, s] = _reduced_potentials_all_states(
                context, alchemical_state, schedule, beta_value
            )
            # Restore this window's lambda: _reduced_potentials_all_states leaves
            # the context at the final schedule state, and the next MD block must
            # propagate at THIS window's lambda, not the decoupled endpoint.
            alchemy.apply_lambdas(context, alchemical_state, le, ls)
            N_k[k] += 1
        # Hand off configuration to next window.
        positions = context.getState(getPositions=True).getPositions()

    del context, integrator
    return {
        "u_kln": u_kln,
        "N_k": N_k,
        "schedule": np.array(schedule),
        "temperature": temperature,
    }
