"""Relative binding free energy (RBFE) via the hybrid single-topology engine.

ddG_bind(A->B) = dG_transform(complex) - dG_transform(solvent)
              = [dG_bind(B) - dG_bind(A)]

The SAME A->B single-topology transform (only the R-group difference mutates; the
common scaffold stays fully coupled) is run twice: once with the ligand in the
protein binding pocket (complex leg, complex_setup.build_hybrid_complex) and once
in bulk water (solvent leg, build_hybrid_single_topology(solvate=True)). Their
difference is the relative binding free energy -- the protein-ligand analogue of
the solvent-vs-vacuum cycle used for relative hydration. The intramolecular
morphing of the fragment against its own core is identical in both legs and
cancels, so only the (complex - solvent) environment difference survives.

Reuses the two-region runners from relative_hydration unchanged (MBAR or d-AFED),
since the complex hybrid is just another TwoLigandSetup with the ligand atoms
first and protein+water as fully-coupled environment.

Scope (v1): neutral congeneric ligand pairs, single R-group swap, no net-charge
change, crystal structure already holding the bound reference ligand. Validation
target: T4 lysozyme L99A benzene->toluene (PDB 181L), ref ddG_bind ~ -0.33 kcal/mol.
"""

import os

from .free_energy import relative_binding_from_legs
from .relative_hydration import _run_two_region_mbar, _run_two_region_dafed


def _single_topology_rbfe(smiles_a, smiles_b, pdb_path, ligand_resname,
                          config, output_dir, runner, method_label):
    """ddG_bind(A->B) = transform(complex) - transform(solvent), single topology."""
    from .single_topology import (
        build_hybrid_single_topology, build_hybrid_complex)

    pad = config.get("padding_nm", 1.0)
    wm = config.get("water_model", "tip3p")
    hmr = config.get("hydrogen_mass_amu")        # HMR (use with integrator='mts')
    flex = config.get("flexible", False)         # flexible/unconstrained (SIN(R))

    complex_setup = build_hybrid_complex(
        smiles_a, smiles_b, pdb_path, ligand_resname=ligand_resname,
        padding_nm=pad, water_model=wm, hydrogen_mass_amu=hmr, flexible=flex)
    solvent_setup = build_hybrid_single_topology(
        smiles_a, smiles_b, padding_nm=pad, water_model=wm, solvate=True,
        hydrogen_mass_amu=hmr, flexible=flex)

    odir_c = os.path.join(output_dir, "complex") if output_dir else None
    odir_s = os.path.join(output_dir, "solvent") if output_dir else None
    res_complex = runner(complex_setup, config, odir_c)
    res_solvent = runner(solvent_setup, config, odir_s)

    result = relative_binding_from_legs(
        res_complex, res_solvent, name=f"{smiles_a}->{smiles_b} (single-top RBFE)")
    result.method = method_label
    result.metadata.update({
        "engine": "single_topology_rbfe",
        "pdb": pdb_path, "ligand_resname": ligand_resname,
        "complex_leg_kcal": res_complex.delta_g_kcal_mol,
        "solvent_leg_kcal": res_solvent.delta_g_kcal_mol,
        "complex_unc": res_complex.uncertainty_kcal_mol,
        "solvent_unc": res_solvent.uncertainty_kcal_mol})
    return result


def relative_binding_mapped(rdmol_a, rdmol_b, a_to_b, pdb_path, config=None,
                            output_dir=None, name="A->B"):
    """RBFE for one perturbation-map edge from PRE-ALIGNED SDF mols + a given map.

    ddG_bind = dG_transform(complex) - dG_transform(solvent), built with
    mapped_single_topology (the SDF poses + provided atom mapping are used
    directly -- no MCS, no re-alignment). Sampled with d-AFED (the runner reads
    config['integrator'], default 'mts448'). Returns an 'rbfe' FreeEnergyResult.
    """
    from .mapped_single_topology import build_hybrid_from_mapped_mols
    from .complex_setup import build_complex_from_rdmol

    cfg = config or {}
    pad = cfg.get("padding_nm", 1.0)
    wm = cfg.get("water_model", "tip3p")
    hmr = cfg.get("hydrogen_mass_amu")
    flex = cfg.get("flexible", False)

    base = build_complex_from_rdmol(
        rdmol_b, pdb_path, padding_nm=pad, water_model=wm,
        hydrogen_mass_amu=hmr, flexible=flex)
    complex_setup = build_hybrid_from_mapped_mols(
        rdmol_a, rdmol_b, a_to_b, base=base, hydrogen_mass_amu=hmr,
        flexible=flex, name=f"{name} complex")
    solvent_setup = build_hybrid_from_mapped_mols(
        rdmol_a, rdmol_b, a_to_b, solvate=True, padding_nm=pad, water_model=wm,
        hydrogen_mass_amu=hmr, flexible=flex, name=f"{name} solvent")

    odir_c = os.path.join(output_dir, "complex") if output_dir else None
    odir_s = os.path.join(output_dir, "solvent") if output_dir else None
    res_complex = _run_two_region_dafed(complex_setup, cfg, odir_c)
    res_solvent = _run_two_region_dafed(solvent_setup, cfg, odir_s)

    result = relative_binding_from_legs(res_complex, res_solvent, name=name)
    result.method = "dynamic_lambda (mapped single-topology RBFE)"
    result.metadata.update({
        "engine": "mapped_single_topology_rbfe", "pdb": pdb_path,
        "complex_leg_kcal": res_complex.delta_g_kcal_mol,
        "solvent_leg_kcal": res_solvent.delta_g_kcal_mol,
        "n_unique_a": complex_setup.metadata["n_unique_a"],
        "n_unique_b": complex_setup.metadata["n_unique_b"]})
    return result


def relative_binding_single_topology(smiles_a, smiles_b, pdb_path,
                                     ligand_resname="BNZ", config=None,
                                     output_dir=None):
    """Relative binding free energy via the hybrid single-topology engine (MBAR).

    Runs the A->B transform in the protein complex and in solvent with fixed-
    window MBAR, then ddG_bind = dG_complex - dG_solvent. Congeneric pairs only.
    """
    return _single_topology_rbfe(
        smiles_a, smiles_b, pdb_path, ligand_resname, config or {}, output_dir,
        _run_two_region_mbar, "MBAR (single-topology RBFE, complex-solvent)")


def relative_binding_single_topology_dafed(smiles_a, smiles_b, pdb_path,
                                           ligand_resname="BNZ", config=None,
                                           output_dir=None):
    """Relative binding free energy via the hybrid single-topology engine (d-AFED).

    Same complex-minus-solvent cycle as relative_binding_single_topology, but each
    transform is sampled with the dynamic-lambda (d-AFED) estimator (a master tau
    propagated as an adiabatic high-T extended coordinate, free energy from the
    reweighted tau histogram). Pairs with the regulated/SIN(R) integrator via
    config['flexible']=True + an integrator_factory in the d-AFED runner.
    """
    return _single_topology_rbfe(
        smiles_a, smiles_b, pdb_path, ligand_resname, config or {}, output_dir,
        _run_two_region_dafed, "dynamic_lambda (single-topology RBFE, complex-solvent)")
