"""Alchemical system construction and lambda schedules.

Thermodynamic setup
-------------------
We perform *absolute* alchemical decoupling of a solute in explicit water using
OpenMMTools' AbsoluteAlchemicalFactory with softcore sterics. Two context
parameters control the perturbation:

    lambda_electrostatics : 1 -> fully coupled charges, 0 -> uncharged solute
    lambda_sterics        : 1 -> fully coupled LJ,      0 -> non-interacting

Decoupling order (charges off before sterics) avoids the well-known singularity
of turning off LJ while charges remain on.

Sign convention (see analysis.py for the full statement):
    We compute Delta G_decouple = G(decoupled) - G(coupled), the free energy of
    removing solute-solvent interactions. The hydration free energy is
        Delta G_hyd = -Delta G_decouple .
"""

import numpy as np


def make_lambda_schedule(n_elec=6, n_steric=12):
    """Build the staged (lambda_elec, lambda_steric) schedule.

    Stage 1: electrostatics 1 -> 0 over n_elec windows, sterics held at 1.
    Stage 2: sterics 1 -> 0 over n_steric windows, electrostatics held at 0.

    Stage 1 contributes n_elec windows running from (1, 1) down to
    (1/n_elec, 1); stage 2 contributes n_steric windows running from (0, 1)
    down to (0, 0). The total is n_elec + n_steric windows (default 18), with
    endpoints (1, 1) fully coupled and (0, 0) fully decoupled.

    Returns a list of (lambda_electrostatics, lambda_sterics) tuples.
    """
    # Stage 1 electrostatics: drop the final 0.0 (it belongs to stage 2's start).
    elec = np.linspace(1.0, 0.0, n_elec + 1)[:-1]   # n_elec values: 1.0 ... 1/n_elec
    steric = np.linspace(1.0, 0.0, n_steric)        # n_steric values: 1.0 ... 0.0

    schedule = []
    # Stage 1: discharge while sterics stay fully on.
    for e in elec:
        schedule.append((float(e), 1.0))
    # Stage 2: decouple sterics, charges already off.
    for s in steric:
        schedule.append((0.0, float(s)))
    return schedule


def smoothstep(x):
    """Smooth 0->1 ramp with zero slope at both ends (3x^2 - 2x^3).

    Used by the dynamic-lambda toy to map one master lambda onto the
    electrostatics/sterics context parameters.
    """
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_alchemical_system(reference_system, alchemical_atoms, softcore=True,
                            pme_treatment="direct-space",
                            annihilate_electrostatics=False):
    """Wrap a reference System in an OpenMMTools alchemical System.

    Returns (alchemical_system, alchemical_state) where alchemical_state is an
    openmmtools.alchemy.AlchemicalState whose .lambda_electrostatics and
    .lambda_sterics can be applied to a Context.

    pme_treatment: "direct-space" (fast, scales only the short-range part of the
    solute electrostatics -- approximate for polar solutes) or "exact" (scales
    the solute charges in both the direct and reciprocal PME sums via
    NonbondedForce parameter offsets -- correct, slightly slower). Use "exact"
    for polar/charged solutes where the long-range electrostatic solvation
    matters.
    """
    from openmmtools.alchemy import (
        AbsoluteAlchemicalFactory,
        AlchemicalRegion,
        AlchemicalState,
    )

    factory = AbsoluteAlchemicalFactory(
        consistent_exceptions=False,
        alchemical_pme_treatment=pme_treatment,
    )
    # annihilate_electrostatics=True turns off the solute's INTRAMOLECULAR (and
    # 1-4) electrostatics as well as solute-solvent; that is only correct for an
    # absolute HFE if a separate gas-phase recharging leg is added back. This
    # pipeline is a single solution-phase decoupling, so the correct choice for
    # polar solutes is DECOUPLE (annihilate_electrostatics=False): keep
    # intramolecular electrostatics, remove only solute-solvent.
    region = AlchemicalRegion(
        alchemical_atoms=alchemical_atoms,
        softcore_alpha=0.5 if softcore else 0.0,
        softcore_beta=0.0,
        annihilate_electrostatics=annihilate_electrostatics,
        annihilate_sterics=False,
    )
    alchemical_system = factory.create_alchemical_system(reference_system, region)
    alchemical_state = AlchemicalState.from_system(alchemical_system)
    return alchemical_system, alchemical_state


def apply_lambdas(context, alchemical_state, lambda_electrostatics, lambda_sterics):
    """Set both alchemical context parameters on a live Context."""
    alchemical_state.lambda_electrostatics = float(lambda_electrostatics)
    alchemical_state.lambda_sterics = float(lambda_sterics)
    alchemical_state.apply_to_context(context)
