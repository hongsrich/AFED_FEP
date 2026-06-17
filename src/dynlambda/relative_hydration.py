"""Relative hydration free energy: from-absolute (validated) and direct paths.

ddG_hyd(A->B) = dG_hyd(B) - dG_hyd(A)   (free_energy.SIGN_RELATIVE_HYDRATION)

Two modes:
  * relative_hydration_from_absolute -- run (or load) two absolute decoupling
    legs and subtract. Reuses the validated decouple+MBAR engine, so it is the
    reliable path and the cleanest sign-convention check.
  * relative_hydration_direct -- one-box dual-topology A->B (relative_setup);
    the two-region MD runner is the engine's documented extension point.
"""

from .free_energy import (
    FreeEnergyResult, relative_hydration_from_absolute as _combine_from_absolute,
)


def compute_absolute_hydration(smiles, name, config=None, output_dir=None):
    """Run one absolute hydration leg and return a 'hydration' FreeEnergyResult."""
    from .relative_setup import build_decoupling_leg, run_fixed_window_leg
    config = config or {}
    leg = build_decoupling_leg(
        smiles, name=name,
        padding_nm=config.get("padding_nm", 1.0),
        water_model=config.get("water_model", "tip3p"),
        n_elec=config.get("n_elec_windows", 6),
        n_steric=config.get("n_steric_windows", 12),
        hydrogen_mass_amu=config.get("hydrogen_mass_amu"),
        temperature=config.get("temperature", 298.15))
    return run_fixed_window_leg(
        leg, leg.schedule, output_dir,
        temperature=config.get("temperature", 298.15),
        equil_steps=config.get("equil_steps", 2500),
        prod_steps=config.get("prod_steps", 10000),
        sample_interval=config.get("sample_interval", 500),
        leg_type="hydration", seed=config.get("seed"),
        prefer_gpu=config.get("prefer_gpu", True),
        integrator_kind=config.get("integrator", "langevin"),
        mts_inner_steps=config.get("mts_inner_steps"),
        progress=config.get("progress", True))


def _resolve_absolute(spec, name, config, output_dir):
    """spec may be a SMILES (compute), a FreeEnergyResult, or (value, unc)."""
    if isinstance(spec, FreeEnergyResult):
        return spec
    if isinstance(spec, (tuple, list)):
        return FreeEnergyResult(name, "hydration", float(spec[0]),
                                float(spec[1]) if len(spec) > 1 else float("nan"),
                                method="loaded")
    return compute_absolute_hydration(spec, name, config, output_dir)


def relative_hydration_from_absolute(ligand_a, ligand_b, config=None,
                                     output_dir=None, name=None):
    """ddG_hyd(A->B) by running/loading two absolute legs and subtracting.

    ligand_a/ligand_b may each be a SMILES string (computed), a FreeEnergyResult,
    or a (value, uncertainty) pair (loaded). Returns a relative_hydration result
    with the per-leg absolutes attached in metadata.
    """
    config = config or {}
    import os
    odir_a = os.path.join(output_dir, "ligand_a") if output_dir else None
    odir_b = os.path.join(output_dir, "ligand_b") if output_dir else None

    res_a = _resolve_absolute(ligand_a, "ligand_a", config, odir_a)
    res_b = _resolve_absolute(ligand_b, "ligand_b", config, odir_b)

    rel = _combine_from_absolute(res_a, res_b, name=name)
    rel.metadata["absolute_a"] = res_a.as_dict()
    rel.metadata["absolute_b"] = res_b.as_dict()
    return rel


def _run_two_region_mbar(setup, config, output_dir):
    """Drive a two-region / hybrid TwoLigandSetup with fixed-window MBAR.

    Shared by the dual-topology and single-topology MBAR entry points -- both
    produce the same TwoLigandSetup, so the runner is identical.
    """
    from .relative_setup import run_two_region_leg
    return run_two_region_leg(
        setup, output_dir=output_dir,
        temperature=config.get("temperature", 298.15),
        timestep_fs=config.get("timestep_fs", 2.0),
        equil_steps=config.get("equil_steps", 2500),
        prod_steps=config.get("prod_steps", 10000),
        sample_interval=config.get("sample_interval", 500),
        seed=config.get("seed"),
        prefer_gpu=config.get("prefer_gpu", True),
        integrator_kind=config.get("integrator", "langevin"),
        mts_inner_steps=config.get("mts_inner_steps"),
        progress=config.get("progress", True))


def _run_two_region_dafed(setup, config, output_dir):
    """Drive a two-region / hybrid TwoLigandSetup with d-AFED + reweighting.

    Shared by the dual-topology and single-topology d-AFED entry points. Recovers
    ddG_hyd(A->B) = A(tau=1) - A(tau=0) from the reweighted tau histogram (no
    adaptive bias; a calibrated double-well barrier separates the end states).
    """
    import os
    import numpy as np
    from .platform import get_fastest_platform
    from .dynamic_lambda import (
        run_dynamic_lambda_two_region, reweight_free_energy, basin_delta_f,
        BarrierPotential, AdaptiveBias, NullBias)
    from .units import KB
    from openmm import unit
    from .free_energy import FreeEnergyResult, SIGN_RELATIVE_HYDRATION

    dyn = config.get("dynamic", {})
    platform, properties = get_fastest_platform(
        prefer_gpu=config.get("prefer_gpu", True))
    lambda_temperature = dyn.get("lambda_temperature", 2000.0)
    barrier = BarrierPotential.from_height(
        height=dyn.get("barrier_height", 10.0),
        sigma0=dyn.get("barrier_sigma0", 0.02))
    kT_s_kJ = (KB * (lambda_temperature * unit.kelvin)).value_in_unit(
        unit.kilojoule_per_mole)
    bias = (AdaptiveBias(kT_kJ=kT_s_kJ) if dyn.get("use_adaptive_bias", False)
            else NullBias())

    # Optional SIN(R)/regulated large-timestep integrator (config integrator=
    # 'regulated'). Needs a flexible/unconstrained system (build the hybrid with
    # flexible=True) and the softcore long-range correction OFF (it does not
    # converge for the alchemical CustomNonbondedForce and cancels in the FE).
    integrator_factory = None
    regulated_minimize_first = True
    if config.get("integrator") == "regulated":
        from .regulated import make_regulated_integrator, init_regulated_momenta
        import openmm as _mm
        reg = config.get("regulated", {})
        ts = config.get("timestep_fs", 48.0)
        L = reg.get("L", 2.0)
        respa = reg.get("respa", [3, 16])
        ctime = reg.get("characteristic_time_fs", 10.0)
        T = config.get("temperature", 298.15)
        for f in setup.system.getForces():
            if isinstance(f, _mm.CustomNonbondedForce):
                f.setUseLongRangeCorrection(False)
            elif isinstance(f, _mm.NonbondedForce):
                f.setUseDispersionCorrection(False)

        def integrator_factory(system, ts=ts, L=L, respa=respa, ctime=ctime, T=T):
            return (make_regulated_integrator(
                temperature=T, timestep_fs=ts, L=L, respa=respa,
                characteristic_time_fs=ctime, system=system),
                init_regulated_momenta)

        # Pre-equilibrate at a small timestep. Starting big SIN(R) steps straight
        # from minimization NaNs on large explicit-solvent systems: the ~10^4
        # flexible waters are cold/unrelaxed and one blows up. A short Langevin
        # 1 fs warmup fixes it (verified: 10 ps -> SIN(R) 48 fs stable on the 37k-
        # atom T4 complex). Equilibrate at the schedule midpoint, then skip the
        # runner's re-minimization so we keep the relaxed configuration.
        from .relative_setup import apply_two_region_lambdas
        eq_ps = reg.get("equil_ps", 15)
        if eq_ps:
            eqi = _mm.LangevinMiddleIntegrator(
                T * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picosecond)
            eqc = _mm.Context(setup.system, eqi, platform, properties)
            eqc.setPositions(setup.positions)
            apply_two_region_lambdas(eqc, setup.schedule[len(setup.schedule) // 2])
            _mm.LocalEnergyMinimizer.minimize(eqc, maxIterations=500)
            eqc.setVelocitiesToTemperature(T * unit.kelvin)
            eqi.step(int(eq_ps * 1000))
            setup.positions = eqc.getState(getPositions=True).getPositions()
            del eqc, eqi
            regulated_minimize_first = False
        else:
            regulated_minimize_first = True

    # 3-scale MTS "4/4/8": bonded + direct-space nonbonded + alchemical softcore
    # on the fast (inner) step, the smooth/expensive PME RECIPROCAL space on the
    # slow (outer) step. With HMR + HBonds + rigid water this runs a 4 fs inner /
    # 8 fs outer step stably -- the robust large-step scheme for the big explicit-
    # solvent complex (where SIN(R)'s flexible water is too fragile/slow). Set
    # config integrator='mts448', timestep_fs=8.0 (outer), mts_inner_steps=2
    # (8/2 = 4 fs inner), hydrogen_mass_amu~3.0, flexible=False.
    elif config.get("integrator") == "mts448":
        import openmm as _mm
        T = config.get("temperature", 298.15)
        fric = config.get("friction_per_ps", 1.0)
        outer_fs = config.get("timestep_fs", 8.0)
        nsub = int(config.get("mts_inner_steps", 2))   # outer/nsub = inner step
        for f in setup.system.getForces():
            if isinstance(f, _mm.CustomNonbondedForce):
                f.setUseLongRangeCorrection(False)
            elif isinstance(f, _mm.NonbondedForce):
                f.setUseDispersionCorrection(False)

        def integrator_factory(system, T=T, fric=fric, outer_fs=outer_fs, nsub=nsub):
            for f in system.getForces():
                if isinstance(f, _mm.NonbondedForce):
                    f.setForceGroup(0)                 # direct space -> inner
                    f.setReciprocalSpaceForceGroup(2)  # PME reciprocal -> outer
                else:
                    f.setForceGroup(0)                 # bonded + softcore -> inner
            integ = _mm.MTSLangevinIntegrator(
                T * unit.kelvin, fric / unit.picosecond,
                outer_fs * unit.femtosecond, [(2, 1), (0, nsub)])
            return integ, None

    result = run_dynamic_lambda_two_region(
        setup, platform, properties,
        temperature=config.get("temperature", 298.15),
        lambda_temperature=lambda_temperature,
        timestep_fs=config.get("timestep_fs", 2.0),
        friction_per_ps=config.get("friction_per_ps", 1.0),
        md_steps_per_block=dyn.get("md_steps_per_block", 50),
        n_blocks=dyn.get("n_blocks", 2000),
        lambda_mass=dyn.get("lambda_mass", 200.0),
        lambda_friction=dyn.get("lambda_friction", 5.0),
        mode=dyn.get("mode", "staged"),
        bias=bias, barrier=barrier,
        tau0=dyn.get("tau0", 0.5),
        minimize_first=regulated_minimize_first,
        seed=config.get("seed"),
        integrator_type=("langevin" if integrator_factory
                         else config.get("integrator", "langevin")),
        mts_inner_steps=config.get("mts_inner_steps"),
        integrator_factory=integrator_factory,
        progress=config.get("progress", True))

    centers, A_kJ = reweight_free_energy(
        result.lambdas, lambda_temperature,
        nbins=dyn.get("pmf_nbins", 25),
        barrier=result.barrier, bias=result.bias,
        drop_frac=dyn.get("reweight_drop_frac", 0.2))
    # End states are tau windows [0, basin_edge] (lambda=0) and [1-basin_edge, 1]
    # (lambda=1), Boltzmann-summed -- NOT the single outermost bin.
    ddg_kcal = float(basin_delta_f(
        centers, A_kJ, config.get("temperature", 298.15),
        edge=dyn.get("basin_edge", 0.2)) / 4.184)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.savez(os.path.join(output_dir, "two_region_dafed.npz"),
                 tau=result.lambdas, times_ps=result.times_ps,
                 dUdlambda=result.dUdlambda,
                 pmf_centers=centers, pmf_kJ=A_kJ,
                 ddg_kcal=ddg_kcal, lambda_temperature=lambda_temperature)
        # Plain-text raw lambda(tau)-vs-time for this leg (no NumPy needed).
        np.savetxt(
            os.path.join(output_dir, "lambda_vs_time.csv"),
            np.column_stack([result.times_ps, result.lambdas, result.dUdlambda]),
            delimiter=",", header="time_ps,tau,dUdtau_kJ_per_mol", comments="")

    return FreeEnergyResult(
        name=setup.name, leg_type="relative_hydration",
        delta_g_kcal_mol=ddg_kcal, uncertainty_kcal_mol=float("nan"),
        sign_convention=SIGN_RELATIVE_HYDRATION, method="dynamic_lambda",
        n_windows=dyn.get("pmf_nbins", 25), n_samples=int(result.lambdas.size),
        metadata={"lambda_temperature": lambda_temperature,
                  "realized_barrier_kJ": barrier.realized_height(),
                  **getattr(setup, "metadata", {})})


def relative_hydration_direct(smiles_a, smiles_b, mapping=None, config=None,
                              output_dir=None):
    """Direct A->B relative hydration via the internal DUAL-topology engine (MBAR).

    Builds a one-box dual-topology system (both whole ligands) and samples its
    paired A->B schedule with MBAR. NOTE: the whole-molecule insertion of B is
    under-sampled at short MD and biases ddG positive; prefer
    relative_hydration_single_topology for congeneric pairs, or
    relative_hydration_from_absolute otherwise.
    """
    from .relative_setup import build_relative_hydration_system
    config = config or {}
    setup = build_relative_hydration_system(
        smiles_a, smiles_b, mapping=mapping,
        padding_nm=config.get("padding_nm", 1.0),
        water_model=config.get("water_model", "tip3p"))
    return _run_two_region_mbar(setup, config, output_dir)


def relative_hydration_direct_dafed(smiles_a, smiles_b, mapping=None,
                                    config=None, output_dir=None):
    """Direct A->B relative hydration via the d-AFED (dynamic-lambda) estimator.

    Same dual-topology box as relative_hydration_direct, but a single master tau
    is propagated as an adiabatic, high-temperature extended-system coordinate
    (run_dynamic_lambda_two_region) instead of sampling fixed windows + MBAR. The
    free energy is recovered by histogram reweighting (reweight_free_energy):
    ddG_hyd(A->B) = A(tau=1) - A(tau=0). No adaptive bias (per project decision);
    a calibrated double-well barrier separates the two physical end states.

    Returns a relative_hydration FreeEnergyResult with method="dynamic_lambda".
    """
    from .relative_setup import build_relative_hydration_system
    config = config or {}
    setup = build_relative_hydration_system(
        smiles_a, smiles_b, mapping=mapping,
        padding_nm=config.get("padding_nm", 1.0),
        water_model=config.get("water_model", "tip3p"))
    return _run_two_region_dafed(setup, config, output_dir)


def _single_topology_two_leg(smiles_a, smiles_b, config, output_dir, runner,
                             method_label):
    """Single-topology relative hydration = transform(solvent) - transform(vacuum).

    The hybrid only mutates the R-group difference, but the unique fragment's
    interactions with its OWN core are treated as 'environment' by the region
    alchemy, so a single solvent leg includes the intramolecular morphing work.
    Running the SAME transform in vacuum isolates that intramolecular part;
    subtracting it leaves the pure solute-solvent (hydration) difference, exactly
    the standard two-leg single-topology relative hydration cycle:

        ddG_hyd(A->B) = dG_transform_solvent(A->B) - dG_transform_vacuum(A->B).

    ``runner`` is _run_two_region_mbar or _run_two_region_dafed.
    """
    import os
    from .single_topology import build_hybrid_single_topology
    from .free_energy import (
        FreeEnergyResult, SIGN_RELATIVE_HYDRATION, _combine_uncertainty)

    pad = config.get("padding_nm", 1.0)
    wm = config.get("water_model", "tip3p")
    # The vacuum leg MUST use the same electrostatics as the solvent leg (PME in a
    # periodic box), or the grown fragment's intramolecular Coulomb is treated
    # differently in the two legs and the cancellation fails by ~+2 kcal/mol
    # (confirmed: NoCutoff vacuum gave toluene->ethylbenzene +2.5 / phenol->cresol
    # +2.3; PME vacuum gave -0.15 / +0.18 vs exp +0.10 / +0.48). Box edge must
    # exceed 2x the 1 nm cutoff; 2.5 nm is ample for a small fragment in vacuum.
    vac_box = config.get("vacuum_pme_box_nm", 2.5)
    hmr = config.get("hydrogen_mass_amu")     # HMR (use with integrator='mts' + 3-4 fs)
    solv = build_hybrid_single_topology(smiles_a, smiles_b, padding_nm=pad,
                                        water_model=wm, solvate=True,
                                        hydrogen_mass_amu=hmr)
    vac = build_hybrid_single_topology(smiles_a, smiles_b, padding_nm=pad,
                                       water_model=wm, solvate=False,
                                       vacuum_pme_box_nm=vac_box,
                                       hydrogen_mass_amu=hmr)
    odir_s = os.path.join(output_dir, "solvent") if output_dir else None
    odir_v = os.path.join(output_dir, "vacuum") if output_dir else None
    res_s = runner(solv, config, odir_s)
    res_v = runner(vac, config, odir_v)

    ddg = res_s.delta_g_kcal_mol - res_v.delta_g_kcal_mol
    unc = _combine_uncertainty(res_s.uncertainty_kcal_mol,
                               res_v.uncertainty_kcal_mol)
    return FreeEnergyResult(
        name=f"{smiles_a}->{smiles_b} (single-top)",
        leg_type="relative_hydration",
        delta_g_kcal_mol=ddg, uncertainty_kcal_mol=unc,
        sign_convention=SIGN_RELATIVE_HYDRATION, method=method_label,
        n_windows=res_s.n_windows,
        n_samples=res_s.n_samples + res_v.n_samples,
        metadata={"engine": "single_topology_two_leg",
                  "solvent_leg_kcal": res_s.delta_g_kcal_mol,
                  "vacuum_leg_kcal": res_v.delta_g_kcal_mol,
                  "solvent_unc": res_s.uncertainty_kcal_mol,
                  "vacuum_unc": res_v.uncertainty_kcal_mol})


def relative_hydration_single_topology(smiles_a, smiles_b, config=None,
                                       output_dir=None):
    """Relative hydration via the HYBRID SINGLE-topology engine (MBAR), two legs.

    Only the R-group difference between A and B is alchemically mutated; the
    shared maximum-common-substructure core stays fully coupled, removing the
    whole-molecule insertion bias of the dual-topology engine. Computes the
    solvent and vacuum transforms and subtracts (see _single_topology_two_leg).
    Congeneric pairs only (neutral, no ring make/break, no net-charge change) --
    see single_topology.build_hybrid_single_topology.
    """
    return _single_topology_two_leg(
        smiles_a, smiles_b, config or {}, output_dir, _run_two_region_mbar,
        "MBAR (single-topology, solvent-vacuum)")


def relative_hydration_single_topology_dafed(smiles_a, smiles_b, config=None,
                                             output_dir=None):
    """Relative hydration via the HYBRID SINGLE-topology engine (d-AFED), two legs.

    Same two-leg cycle as relative_hydration_single_topology, sampled with the
    dynamic-lambda (d-AFED) estimator instead of fixed-window MBAR.
    """
    return _single_topology_two_leg(
        smiles_a, smiles_b, config or {}, output_dir, _run_two_region_dafed,
        "dynamic_lambda (single-topology, solvent-vacuum)")
