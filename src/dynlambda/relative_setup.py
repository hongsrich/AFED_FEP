"""Internal alchemical leg setup + the generic fixed-window leg runner.

Two responsibilities:

1. ``run_fixed_window_leg`` -- the reusable engine that turns ANY prepared
   alchemical LegSetup (system + state + schedule) into a FreeEnergyResult via
   the existing fixed-window sampler (simulation.run_fixed_window_hfe) and MBAR
   (analysis.run_mbar). FixedWindowLegRunner (rbfe.py) calls this. It works for
   absolute decoupling legs and for the dual-topology relative legs below.

2. Builders that produce LegSetups:
   * ``build_decoupling_leg`` -- absolute decoupling of a solute (reuses the
     validated molecule_setup + alchemy.build_alchemical_system path; decouple,
     not annihilate -- see methanol fix).
   * ``build_relative_hydration_system`` -- DUAL-TOPOLOGY A->B: both ligands in
     one solvent box, unique atoms of B grown while unique atoms of A vanish,
     mapped core shared in space. This is the internal hybrid-topology engine.

No adaptive bias anywhere; a future dynamic-lambda leg reuses the AFED histogram
estimator (see rbfe.DynamicLambdaLegRunner).
"""

from dataclasses import dataclass, field

from .free_energy import FreeEnergyResult, SIGN_HYDRATION


@dataclass
class LegSetup:
    """Everything run_fixed_window_leg needs to sample one alchemical leg."""
    solvated: object                       # has .positions, .system, .topology
    alchemical_system: object
    alchemical_state: object
    schedule: list                         # list of (lambda_elec, lambda_steric)
    name: str = "leg"
    temperature: float = 298.15
    metadata: dict = field(default_factory=dict)


def run_fixed_window_leg(system_setup, lambda_schedule, output_dir,
                         temperature=None, equil_steps=2500, prod_steps=10000,
                         sample_interval=500, leg_type="complex", seed=None,
                         prefer_gpu=True, integrator_kind="langevin",
                         mts_inner_steps=None, progress=True):
    """Sample a prepared LegSetup over fixed windows and analyze with MBAR.

    Returns a FreeEnergyResult. ``lambda_schedule`` overrides the LegSetup's own
    schedule if given. The reported delta_g is the leg's coupled->decoupled (or
    A->B) free energy in kcal/mol with the recorded sign convention.
    """
    import os
    import numpy as np
    from .platform import get_fastest_platform
    from .simulation import run_fixed_window_hfe
    from .analysis import run_mbar

    setup = system_setup
    schedule = lambda_schedule or setup.schedule
    T = temperature or setup.temperature
    platform, properties = get_fastest_platform(prefer_gpu=prefer_gpu)

    data = run_fixed_window_hfe(
        setup.solvated, setup.alchemical_system, setup.alchemical_state,
        schedule, platform, properties, temperature=T,
        equil_steps=equil_steps, prod_steps=prod_steps,
        sample_interval=sample_interval, seed=seed,
        integrator_kind=integrator_kind, mts_inner_steps=mts_inner_steps,
        progress=progress)

    mbar = run_mbar(data["u_kln"], data["N_k"], T)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.savez(os.path.join(output_dir, "leg_mbar.npz"),
                 u_kln=data["u_kln"], N_k=data["N_k"],
                 schedule=data["schedule"], dg_kcal=mbar.dg_decouple_kJ / 4.184)

    # For an absolute hydration/decoupling leg the headline is dG_hyd; for a
    # relative (complex/solvent) leg the headline is the A->B transformation free
    # energy (= the schedule's state0->stateN free energy = dg_decouple sense).
    if leg_type == "hydration":
        dg = mbar.dg_hyd_kcal
        sign = SIGN_HYDRATION
    else:
        dg = mbar.dg_decouple_kJ / 4.184   # state0 -> stateN along the schedule
        sign = "leg dG = F[stateN] - F[state0] along the A->B schedule"

    return FreeEnergyResult(
        name=getattr(setup, "name", output_dir or leg_type),
        leg_type=leg_type,
        delta_g_kcal_mol=dg,
        uncertainty_kcal_mol=mbar.ddg_kcal,
        sign_convention=sign,
        method="MBAR",
        n_windows=len(schedule),
        n_samples=int(np.sum(data["N_k"])),
        metadata={"overlap_available": mbar.overlap is not None},
    )


def build_decoupling_leg(smiles, name="molecule", padding_nm=1.0,
                         water_model="tip3p", n_elec=6, n_steric=12,
                         hydrogen_mass_amu=None, temperature=298.15):
    """Absolute decoupling LegSetup for a solute (reuses the validated path)."""
    from .molecule_setup import setup_molecule
    from .alchemy import build_alchemical_system, make_lambda_schedule

    solvated = setup_molecule(smiles=smiles, name=name, prefer_openff=True,
                              padding=padding_nm, water_model=water_model,
                              hydrogen_mass_amu=hydrogen_mass_amu)
    alch_system, alch_state = build_alchemical_system(
        solvated.system, solvated.alchemical_atoms)   # decouple default
    schedule = make_lambda_schedule(n_elec=n_elec, n_steric=n_steric)
    return LegSetup(solvated=solvated, alchemical_system=alch_system,
                    alchemical_state=alch_state, schedule=schedule,
                    name=name, temperature=temperature)


@dataclass
class TwoLigandSetup:
    """A dual-topology A->B relative system: both ligands in one box.

    The two ligands are mutually non-interacting (A-B nonbonded exclusions) and
    alchemically anti-correlated: the paired schedule couples B up as A is
    decoupled, so state 0 = "A solvated / B ghost" and the final state =
    "A ghost / B solvated". The free energy along the schedule is ddG_hyd(A->B).
    """
    system: object
    lambda_parameters: list             # global parameter names (lambda_*_A/_B)
    positions: object
    topology: object
    atoms_a: list
    atoms_b: list
    schedule: list                      # list of dicts of region lambdas
    name: str = "A->B"
    temperature: float = 298.15
    metadata: dict = field(default_factory=dict)


def two_region_schedule(n_elec=6, n_steric=12):
    """Paired A(decouple)/B(couple) schedule as a list of region-lambda dicts.

    Endpoint 0: A fully on, B off. Endpoint -1: A off, B fully on.

    BOTH sides are staged to avoid a Coulomb catastrophe (a partial charge on an
    atom with no LJ repulsion lets the solvent collapse onto it -> NaN):
      * A decouples discharge-FIRST (remove electrostatics while LJ is on, then
        remove LJ): the (e,1) discharge block then the (0,s) decouple block.
      * B couples in the time-REVERSED order, i.e. grow STERICS FIRST, then
        charge. B's path is therefore exactly reversed(A's path), so at every
        window neither region ever has electrostatics on with sterics off. This
        matches the d-AFED master map (master_to_two_region), which also grows B's
        sterics before its charge.
    """
    import numpy as np
    elec = np.linspace(1.0, 0.0, n_elec + 1)[:-1]
    ster = np.linspace(1.0, 0.0, n_steric)
    a_seq = ([(float(e), 1.0) for e in elec]          # discharge (LJ on)
             + [(0.0, float(s)) for s in ster])       # then decouple LJ
    b_seq = list(reversed(a_seq))                      # grow LJ first, then charge
    return [
        {"lambda_electrostatics_A": ae, "lambda_sterics_A": as_,
         "lambda_electrostatics_B": be, "lambda_sterics_B": bs}
        for (ae, as_), (be, bs) in zip(a_seq, b_seq)
    ]


def build_two_ligand_box(smiles_a, smiles_b, mapping=None, name="A->B",
                         padding_nm=1.0, water_model="tip3p", solvate=True,
                         separation_nm=1.2, n_elec=6, n_steric=12):
    """Build a dual-topology A->B system (the internal hybrid-topology engine).

    Steps: parametrize both ligands (OpenFF), place B separated from A so they
    do not clash, (optionally) solvate, add A-B nonbonded exclusions so the two
    ligands never interact, then define two named alchemical regions (decouple,
    not annihilate) and a paired schedule. Returns a TwoLigandSetup.

    The two-region MD sampling loop (run_two_region_leg) is the remaining step;
    this builder + its state are unit-tested without MD.
    """
    import numpy as np
    import openmm
    from openmm import app, unit
    from openff.toolkit import Molecule, Topology
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    from .molecule_setup import _rdkit_mol_from_smiles, DEFAULT_SMALL_MOLECULE_FF
    from .mapping import map_ligands
    if mapping is None:
        mapping = map_ligands(smiles_a, smiles_b)

    molA = Molecule.from_rdkit(_rdkit_mol_from_smiles(smiles_a),
                               allow_undefined_stereo=True)
    molB = Molecule.from_rdkit(_rdkit_mol_from_smiles(smiles_b),
                               allow_undefined_stereo=True)

    posA = molA.conformers[0].to_openmm().value_in_unit(unit.nanometer)
    posB = molB.conformers[0].to_openmm().value_in_unit(unit.nanometer)
    posA = np.array(posA)
    posB = np.array(posB) + np.array([separation_nm, 0.0, 0.0])  # separate B from A

    off_top = Topology.from_molecules([molA, molB])
    omm_top = off_top.to_openmm()
    positions = np.vstack([posA, posB]) * unit.nanometer

    smirnoff = SMIRNOFFTemplateGenerator(
        molecules=[molA, molB], forcefield=DEFAULT_SMALL_MOLECULE_FF)
    forcefield = app.ForceField("amber/tip3p_standard.xml")
    forcefield.registerTemplateGenerator(smirnoff.generator)

    modeller = app.Modeller(omm_top, positions)
    if solvate:
        modeller.addSolvent(forcefield, model=water_model,
                            padding=padding_nm * unit.nanometer)
        nonbonded = app.PME
    else:
        nonbonded = app.NoCutoff

    system = forcefield.createSystem(
        modeller.topology, nonbondedMethod=nonbonded,
        nonbondedCutoff=1.0 * unit.nanometer, constraints=app.HBonds,
        rigidWater=True)

    nA = molA.n_atoms
    nB = molB.n_atoms
    atoms_a = list(range(0, nA))
    atoms_b = list(range(nA, nA + nB))

    _add_interligand_exclusions(system, atoms_a, atoms_b)

    alch_system, lambda_params = _build_two_region_alchemical(
        system, atoms_a, atoms_b)
    schedule = two_region_schedule(n_elec=n_elec, n_steric=n_steric)

    return TwoLigandSetup(
        system=alch_system, lambda_parameters=lambda_params,
        positions=modeller.positions, topology=modeller.topology,
        atoms_a=atoms_a, atoms_b=atoms_b, schedule=schedule, name=name,
        metadata={"smiles_a": smiles_a, "smiles_b": smiles_b,
                  "mcs_mapped": mapping.n_mapped, "solvated": solvate})


def _add_interligand_exclusions(system, atoms_a, atoms_b):
    """Add zeroed NonbondedForce exceptions for every A-B atom pair.

    Makes ligands A and B mutually non-interacting (dual-topology requirement):
    each is an alternative end state, not a co-solute.
    """
    import openmm
    from openmm import unit
    nbforces = [f for f in system.getForces()
                if isinstance(f, openmm.NonbondedForce)]
    if not nbforces:
        return 0
    nb = nbforces[0]
    existing = set()
    for k in range(nb.getNumExceptions()):
        i, j, *_ = nb.getExceptionParameters(k)
        existing.add((min(i, j), max(i, j)))
    added = 0
    setb = set(atoms_b)
    for i in atoms_a:
        for j in setb:
            key = (min(i, j), max(i, j))
            if key in existing:
                continue
            nb.addException(i, j, 0.0 * unit.elementary_charge ** 2,
                            0.1 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
            added += 1
    return added


def _build_two_region_alchemical(system, atoms_a, atoms_b):
    """Two named alchemical regions (A, B), decoupled (not annihilated).

    Returns (alchemical_system, [global parameter names]). We drive the region
    lambdas by setting the suffixed global parameters (lambda_sterics_A, etc.)
    directly on the Context -- openmmtools' base AlchemicalState cannot
    introspect multiple named regions, but the parameters live on the forces.
    """
    from openmmtools.alchemy import AbsoluteAlchemicalFactory, AlchemicalRegion
    factory = AbsoluteAlchemicalFactory(
        consistent_exceptions=False, alchemical_pme_treatment="direct-space")
    region_a = AlchemicalRegion(
        alchemical_atoms=atoms_a, softcore_alpha=0.5, softcore_beta=0.0,
        annihilate_electrostatics=False, annihilate_sterics=False, name="A")
    region_b = AlchemicalRegion(
        alchemical_atoms=atoms_b, softcore_alpha=0.5, softcore_beta=0.0,
        annihilate_electrostatics=False, annihilate_sterics=False, name="B")
    alch_system = factory.create_alchemical_system(system, [region_a, region_b])

    params = set()
    for f in alch_system.getForces():
        if hasattr(f, "getNumGlobalParameters"):
            for i in range(f.getNumGlobalParameters()):
                n = f.getGlobalParameterName(i)
                if n.lower().startswith("lambda_"):
                    params.add(n)
    return alch_system, sorted(params)


def apply_two_region_lambdas(context, params):
    """Apply a paired-schedule dict (lambda_*_A / lambda_*_B) to a context."""
    for key, val in params.items():
        context.setParameter(key, float(val))


def build_relative_hydration_system(smiles_a, smiles_b, mapping=None,
                                    padding_nm=1.0, water_model="tip3p",
                                    solvate=True):
    """Build the dual-topology A->B relative-hydration system (see TwoLigandSetup)."""
    return build_two_ligand_box(
        smiles_a, smiles_b, mapping=mapping, padding_nm=padding_nm,
        water_model=water_model, solvate=solvate)


def _reduced_potentials_two_region(context, schedule, beta_value):
    """Reduced potentials of the current configuration in every paired window.

    Returns a 1D array of length len(schedule). Mirrors
    simulation._reduced_potentials_all_states but applies the paired region
    lambda dicts (lambda_*_A / lambda_*_B) instead of a single (elec, steric).
    """
    import numpy as np
    from openmm import unit
    out = np.empty(len(schedule), dtype=np.float64)
    for l, params in enumerate(schedule):
        apply_two_region_lambdas(context, params)
        energy = context.getState(getEnergy=True).getPotentialEnergy()
        out[l] = beta_value * energy.value_in_unit(unit.kilojoule_per_mole)
    return out


def run_two_region_leg(setup, output_dir=None, temperature=None,
                       timestep_fs=2.0, friction_per_ps=1.0,
                       equil_steps=2500, prod_steps=10000, sample_interval=500,
                       minimize_first=True, seed=None, prefer_gpu=True,
                       integrator_kind="langevin", mts_inner_steps=None,
                       progress=True):
    """Sample a TwoLigandSetup over its paired A->B schedule and analyze w/ MBAR.

    Mirrors simulation.run_fixed_window_hfe, but the schedule is a list of paired
    region-lambda dicts (apply_two_region_lambdas) rather than (elec, steric)
    tuples, and the alchemical globals live directly on the Custom forces (no
    openmmtools AlchemicalState -- see build_two_ligand_box).

    State 0 = "A solvated / B ghost", final state = "A ghost / B solvated", so
    the schedule free energy F[last] - F[0] is exactly

        ddG_hyd(A->B) = dG_hyd(B) - dG_hyd(A)

    (the ghost molecule contributes only its intramolecular/gas free energy, which
    cancels). Returned as a 'relative_hydration' FreeEnergyResult in kcal/mol.
    """
    import os
    import numpy as np
    import openmm
    from openmm import unit

    from .platform import get_fastest_platform
    from .integrators import make_integrator
    from .analysis import run_mbar
    from .units import beta
    from .free_energy import FreeEnergyResult, SIGN_RELATIVE_HYDRATION

    schedule = setup.schedule
    K = len(schedule)
    T = temperature if temperature is not None else setup.temperature
    beta_value = beta(T).value_in_unit(unit.mole / unit.kilojoule)
    platform, properties = get_fastest_platform(prefer_gpu=prefer_gpu)

    # make_integrator(kind='mts') assigns force groups on setup.system in place,
    # so it must run before the Context is built.
    integrator = make_integrator(
        setup.system, T, friction_per_ps, timestep_fs,
        kind=integrator_kind, mts_inner_steps=mts_inner_steps)
    if seed is not None:
        integrator.setRandomNumberSeed(int(seed))
    context = openmm.Context(setup.system, integrator, platform, properties)

    n_samples = prod_steps // sample_interval
    u_kln = np.zeros((K, K, n_samples), dtype=np.float64)
    N_k = np.zeros(K, dtype=int)

    positions = setup.positions
    if minimize_first:
        # Minimize once at endpoint 0 (A fully coupled, B ghost).
        apply_two_region_lambdas(context, schedule[0])
        context.setPositions(positions)
        openmm.LocalEnergyMinimizer.minimize(context, maxIterations=200)
        positions = context.getState(getPositions=True).getPositions()

    for k, params in enumerate(schedule):
        if progress:
            print(f"  window {k+1}/{K}  {params}")
        # Sequential warmup from the previous window's configuration.
        context.setPositions(positions)
        apply_two_region_lambdas(context, params)
        # Relax any clash the lambda change introduced before velocities.
        openmm.LocalEnergyMinimizer.minimize(context, maxIterations=100)
        context.setVelocitiesToTemperature(T * unit.kelvin)

        integrator.step(equil_steps)

        for s in range(n_samples):
            integrator.step(sample_interval)
            u_kln[k, :, s] = _reduced_potentials_two_region(
                context, schedule, beta_value)
            # Restore THIS window's lambdas: the reduced-potential sweep left the
            # context at the final schedule state; the next MD block must
            # propagate at this window's lambdas.
            apply_two_region_lambdas(context, params)
            N_k[k] += 1
        positions = context.getState(getPositions=True).getPositions()

    del context, integrator

    mbar = run_mbar(u_kln, N_k, T)
    ddg_kcal = mbar.dg_decouple_kJ / 4.184   # F[last] - F[0] = ddG_hyd(A->B)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.savez(os.path.join(output_dir, "two_region_mbar.npz"),
                 u_kln=u_kln, N_k=N_k, ddg_kcal=ddg_kcal)

    return FreeEnergyResult(
        name=setup.name,
        leg_type="relative_hydration",
        delta_g_kcal_mol=ddg_kcal,
        uncertainty_kcal_mol=mbar.ddg_kcal,
        sign_convention=SIGN_RELATIVE_HYDRATION,
        method="MBAR",
        n_windows=K,
        n_samples=int(np.sum(N_k)),
        metadata={"engine": "dual_topology_two_region",
                  "overlap_available": mbar.overlap is not None,
                  **getattr(setup, "metadata", {})},
    )
