"""Protein-ligand complex setup for relative binding free energy (RBFE).

Builds a solvated protein-ligand complex with the ligand parametrized by OpenFF
and the protein by Amber ff14SB, arranged so the LIGAND ATOMS COME FIRST (ligand
-> protein -> water). That ordering is the load-bearing trick: the hybrid
single-topology merge in single_topology.build_hybrid_single_topology only ever
appends/edits the ligand block (indices 0..n_ligand-1) and treats everything
after it as fixed "environment" -- exactly as water was treated in the hydration
case. So feeding this complex in as the `base` of the hybrid builder turns the
solvent-leg engine into a complex-leg engine with no change to the merge, the
two-region alchemy, or the d-AFED / MBAR drivers.

The relative binding cycle is then

    ddG_bind(A->B) = dG_transform(complex) - dG_transform(solvent)

i.e. the SAME A->B single-topology transform run with the ligand in the protein
pocket vs. in bulk water; the intramolecular morphing cancels between the two
legs just like solvent-vs-vacuum did for relative hydration.

Scope (v1): a crystal structure that already contains the bound reference ligand
(e.g. T4 L99A 181L with benzene = resname BNZ). The OpenFF ligand B is placed by
rigid-aligning its ring onto the crystal ligand's heavy atoms; minimization in
the hybrid builder / leg runner relaxes the rest. Neutral congeneric ligands.
"""

import numpy as np
import openmm
from openmm import app, unit

from .molecule_setup import (
    SolvatedSystem, _rdkit_mol_from_smiles, DEFAULT_SMALL_MOLECULE_FF)

# Amber ff14SB protein + TIP3P water (matches molecule_setup's water choice).
_PROTEIN_FF = ("amber/protein.ff14SB.xml", "amber/tip3p_standard.xml",
               "amber/tip3p_HFE_multivalent.xml")


def _crystal_ligand_heavy_coords(pdb_path, ligand_resname):
    """Heavy-atom coordinates (nm, Nx3) of the bound crystal ligand, by resname."""
    pdb = app.PDBFile(pdb_path)
    pos = np.array(pdb.positions.value_in_unit(unit.nanometer))
    coords = []
    for atom in pdb.topology.atoms():
        if atom.residue.name == ligand_resname and atom.element is not None \
                and atom.element.symbol != "H":
            coords.append(pos[atom.index])
    if not coords:
        raise ValueError(
            f"no heavy atoms for ligand resname {ligand_resname!r} in {pdb_path}")
    return np.array(coords)


def _prep_protein(pdb_path, ph=7.0):
    """PDBFixer: strip heterogens/waters, add missing atoms + hydrogens.

    Returns (topology, positions). The input crystal structure is assumed to be
    the desired construct already (e.g. 181L is the L99A mutant), so no mutation
    is applied here.
    """
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)   # drop ligand, ions, crystal waters
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    return fixer.topology, fixer.positions


def _ligand_ring_indices(rdmol, n_target):
    """Indices of a heavy-atom ring to align onto the crystal ligand.

    Picks the largest ring; if it has more atoms than the crystal reference
    (n_target), trims to the first n_target ring atoms. Falls back to the first
    n_target heavy atoms if no ring is found.
    """
    rings = rdmol.GetRingInfo().AtomRings()
    if rings:
        ring = max(rings, key=len)
        ring = [i for i in ring
                if rdmol.GetAtomWithIdx(i).GetAtomicNum() > 1]
        if len(ring) >= n_target:
            return list(ring[:n_target])
    heavy = [a.GetIdx() for a in rdmol.GetAtoms() if a.GetAtomicNum() > 1]
    return heavy[:n_target]


def _kabsch(a_coords, b_coords):
    """Rigid transform (R, t) mapping a_coords onto b_coords (least squares)."""
    ca = a_coords.mean(axis=0)
    cb = b_coords.mean(axis=0)
    H = (a_coords - ca).T @ (b_coords - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cb - R @ ca
    return R, t


def build_complex_from_smiles(ligand_smiles, pdb_path, ligand_resname="BNZ",
                              name="complex", padding_nm=1.0,
                              water_model="tip3p", ionic_strength=0.0,
                              hydrogen_mass_amu=None, flexible=False,
                              constraints="default"):
    """Solvated protein-ligand complex with the LIGAND ATOMS FIRST.

    ``ligand_smiles`` is parametrized by OpenFF (same atom order as
    molecule_setup._rdkit_mol_from_smiles, so it lines up with the hybrid
    partition); the protein from ``pdb_path`` by ff14SB. The ligand is rigid-
    aligned onto the ``ligand_resname`` heavy atoms in the crystal structure,
    then the two are combined (ligand first), solvated in TIP3P, and built into a
    PME System. ``flexible``/``constraints`` mirror molecule_setup for the
    regulated (SIN(R)) integrator. Returns a SolvatedSystem whose
    ``alchemical_atoms`` are the ligand indices (0..n_ligand-1).
    """
    from openff.toolkit import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    # (1) Ligand: OpenFF params + a conformer, in the canonical atom order.
    rdmol = _rdkit_mol_from_smiles(ligand_smiles)
    offmol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)
    lig_top = offmol.to_topology().to_openmm()
    lig_pos = np.array(offmol.conformers[0].to_openmm().value_in_unit(unit.nanometer))

    # (2) Place the ligand at the crystal ligand pose (rigid ring alignment).
    xtal = _crystal_ligand_heavy_coords(pdb_path, ligand_resname)
    ring = _ligand_ring_indices(rdmol, len(xtal))
    R, t = _kabsch(lig_pos[ring], xtal[:len(ring)])
    lig_pos = (R @ lig_pos.T).T + t

    # (3) Protein from the crystal structure (ff14SB-ready, hydrogens added).
    prot_top, prot_pos = _prep_protein(pdb_path)

    # (4) Force field: protein ff14SB + TIP3P + the ligand SMIRNOFF template.
    smirnoff = SMIRNOFFTemplateGenerator(
        molecules=offmol, forcefield=DEFAULT_SMALL_MOLECULE_FF)
    forcefield = app.ForceField(*_PROTEIN_FF)
    forcefield.registerTemplateGenerator(smirnoff.generator)

    # (5) Combine LIGAND FIRST, then protein, then solvate.
    modeller = app.Modeller(lig_top, lig_pos * unit.nanometer)
    modeller.add(prot_top, prot_pos)
    modeller.addSolvent(forcefield, model=water_model,
                        padding=padding_nm * unit.nanometer,
                        ionicStrength=ionic_strength * unit.molar)

    system_kwargs = dict(nonbondedMethod=app.PME,
                         nonbondedCutoff=1.0 * unit.nanometer,
                         constraints=app.HBonds, rigidWater=True)
    if hydrogen_mass_amu is not None:
        system_kwargs["hydrogenMass"] = hydrogen_mass_amu * unit.amu
    if flexible:
        system_kwargs["constraints"] = None
        system_kwargs["rigidWater"] = False
    if constraints != "default":
        system_kwargs["constraints"] = constraints
        if constraints is None:
            system_kwargs["rigidWater"] = False
    system = forcefield.createSystem(modeller.topology, **system_kwargs)

    n_ligand = lig_top.getNumAtoms()
    return SolvatedSystem(
        system=system, topology=modeller.topology,
        positions=modeller.positions,
        alchemical_atoms=list(range(n_ligand)), name=name)


def build_complex_from_rdmol(rdmol, pdb_path, name="complex", padding_nm=1.0,
                             water_model="tip3p", ionic_strength=0.0,
                             hydrogen_mass_amu=None, flexible=False,
                             constraints="default"):
    """Solvated protein-ligand complex from a PRE-POSED RDKit ligand, ligand FIRST.

    For an already-aligned congeneric series (e.g. an FEP+ benchmark SDF): the
    ligand's own conformer is the binding pose, so it is used DIRECTLY -- no
    crystal-ligand alignment. The protein from ``pdb_path`` is prepped with ff14SB
    (heterogens/waters stripped, hydrogens added). Atom order is ligand-first so
    the mapped single-topology merge applies unchanged. Returns a SolvatedSystem
    whose ``alchemical_atoms`` are the ligand indices (0..n_ligand-1).
    """
    from openff.toolkit import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    offmol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)
    lig_top = offmol.to_topology().to_openmm()
    lig_pos = offmol.conformers[0].to_openmm()          # the aligned SDF pose

    prot_top, prot_pos = _prep_protein(pdb_path)
    smirnoff = SMIRNOFFTemplateGenerator(
        molecules=offmol, forcefield=DEFAULT_SMALL_MOLECULE_FF)
    forcefield = app.ForceField(*_PROTEIN_FF)
    forcefield.registerTemplateGenerator(smirnoff.generator)

    modeller = app.Modeller(lig_top, lig_pos)
    modeller.add(prot_top, prot_pos)
    modeller.addSolvent(forcefield, model=water_model,
                        padding=padding_nm * unit.nanometer,
                        ionicStrength=ionic_strength * unit.molar)

    system_kwargs = dict(nonbondedMethod=app.PME,
                         nonbondedCutoff=1.0 * unit.nanometer,
                         constraints=app.HBonds, rigidWater=True)
    if hydrogen_mass_amu is not None:
        system_kwargs["hydrogenMass"] = hydrogen_mass_amu * unit.amu
    if flexible:
        system_kwargs["constraints"] = None
        system_kwargs["rigidWater"] = False
    if constraints != "default":
        system_kwargs["constraints"] = constraints
        if constraints is None:
            system_kwargs["rigidWater"] = False
    system = forcefield.createSystem(modeller.topology, **system_kwargs)

    return SolvatedSystem(
        system=system, topology=modeller.topology, positions=modeller.positions,
        alchemical_atoms=list(range(lig_top.getNumAtoms())), name=name)
