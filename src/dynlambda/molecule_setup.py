"""Small-molecule setup: SMILES -> parametrized, solvated OpenMM System.

Preferred path
--------------
1. RDKit builds a 3D conformer from SMILES and adds hydrogens.
2. OpenFF (via openmmforcefields' SMIRNOFFTemplateGenerator) supplies GAFF/SMIRNOFF
   parameters for the solute.
3. OpenMM Modeller solvates the solute in TIP3P water.
4. We build a System with PME, a 1 nm cutoff, and HBonds constraints.

Fallback path
-------------
If OpenFF/openmmforcefields are unavailable, fall back to an openmmtools
TestSystem (a solute-in-water box) so the rest of the pipeline still runs.

The returned object always exposes:
    system        : openmm.System
    topology      : openmm.app.Topology
    positions     : openmm.unit.Quantity (Nx3)
    alchemical_atoms : list[int]  (solute atom indices)
"""

from dataclasses import dataclass, field

import numpy as np
import openmm
from openmm import app, unit


@dataclass
class SolvatedSystem:
    system: openmm.System
    topology: app.Topology
    positions: object
    alchemical_atoms: list = field(default_factory=list)
    name: str = "molecule"


# --- common System creation options -------------------------------------------

_SYSTEM_KWARGS = dict(
    nonbondedMethod=app.PME,
    nonbondedCutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,
    rigidWater=True,
)


def _rdkit_mol_from_smiles(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(mol, params) != 0:
        # Retry without strict ETKDG if embedding failed.
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


# Default OpenFF small-molecule force field (Sage). Override via build_from_smiles.
DEFAULT_SMALL_MOLECULE_FF = "openff-2.1.0"


def build_from_smiles(
    smiles,
    name="molecule",
    padding=1.0,
    water_model="tip3p",
    ionic_strength=0.0,
    small_molecule_ff=DEFAULT_SMALL_MOLECULE_FF,
    hydrogen_mass_amu=None,
    flexible=False,
    constraints="default",
):
    """Preferred path: parametrize with OpenFF and solvate in TIP3P.

    padding is the solvent padding in nanometers around the solute.
    hydrogen_mass_amu enables hydrogen mass repartitioning (HMR): OpenMM raises
    each hydrogen to this mass and subtracts it from the bonded heavy atom, which
    slows the fastest bond/angle motions so a 3-4 fs timestep stays stable. Needs
    the HBonds constraints + rigidWater already in _SYSTEM_KWARGS. Amber-style
    values: ~3 amu for 4 fs, ~2 amu for 3 fs. None -> physical masses (no HMR).
    Returns a SolvatedSystem. Raises on failure (caller may fall back).
    """
    from openff.toolkit import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    rdmol = _rdkit_mol_from_smiles(smiles)
    offmol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)

    # SMIRNOFF small-molecule template + standard TIP3P water/ions.
    smirnoff = SMIRNOFFTemplateGenerator(molecules=offmol, forcefield=small_molecule_ff)
    forcefield = app.ForceField("amber/tip3p_standard.xml", "amber/tip3p_HFE_multivalent.xml")
    forcefield.registerTemplateGenerator(smirnoff.generator)

    # Solute topology/positions from the OpenFF molecule.
    off_topology = offmol.to_topology()
    solute_top = off_topology.to_openmm()
    solute_pos = offmol.conformers[0].to_openmm()

    modeller = app.Modeller(solute_top, solute_pos)
    modeller.addSolvent(
        forcefield,
        model=water_model,
        padding=padding * unit.nanometer,
        ionicStrength=ionic_strength * unit.molar,
    )

    system_kwargs = dict(_SYSTEM_KWARGS)
    if hydrogen_mass_amu is not None:
        system_kwargs["hydrogenMass"] = hydrogen_mass_amu * unit.amu
    if flexible:
        # Fully flexible, UNCONSTRAINED system (for the regulated/SIN(R) integrator,
        # which does not handle constraints). Water becomes flexible TIP3P.
        system_kwargs["constraints"] = None
        system_kwargs["rigidWater"] = False
    if constraints != "default":
        # Override the constraint scheme (e.g. app.AllBonds to constrain every bond,
        # enabling a plain 4 fs step without MTS). None == fully flexible.
        system_kwargs["constraints"] = constraints
        if constraints is None:
            system_kwargs["rigidWater"] = False
    system = forcefield.createSystem(modeller.topology, **system_kwargs)

    n_solute = solute_top.getNumAtoms()
    alchemical_atoms = list(range(n_solute))

    return SolvatedSystem(
        system=system,
        topology=modeller.topology,
        positions=modeller.positions,
        alchemical_atoms=alchemical_atoms,
        name=name,
    )


def build_fallback(name="methane", nmol=1):
    """Fallback path: an openmmtools test system (solute in water).

    Uses a small, well-behaved system so the alchemy/MBAR/dynamic-lambda code
    paths can be exercised even without a working OpenFF install.
    """
    from openmmtools import testsystems

    # A tiny WaterBox plus a Lennard-Jones-like solute is overkill; instead use
    # the AlanineDipeptideExplicit-free, lightweight option: a single particle
    # of a host-guest-free test. We use the DischargedWaterBox's solute-style
    # systems via the simplest available: a methane-in-water analog built from
    # the HostGuest-free 'WaterBox' is not alchemy-ready, so we use the
    # purpose-built 'AlchemicalWater'-style: TestSystem with a tagged solute.
    ts = testsystems.WaterBox(box_edge=2.0 * unit.nanometer, cutoff=0.9 * unit.nanometer)

    # Tag the first water molecule's atoms as the "alchemical solute" so the
    # pipeline is exercised end-to-end. This is a proof-of-concept stand-in.
    alchemical_atoms = [0, 1, 2]
    return SolvatedSystem(
        system=ts.system,
        topology=ts.topology,
        positions=ts.positions,
        alchemical_atoms=alchemical_atoms,
        name=name,
    )


def setup_molecule(smiles=None, name="molecule", prefer_openff=True, **kwargs):
    """Top-level entry point used by scripts.

    Tries the OpenFF path; on any failure (and if allowed) falls back to an
    openmmtools test system. Prints which path was taken.
    """
    if smiles is not None and prefer_openff:
        try:
            sysobj = build_from_smiles(smiles, name=name, **kwargs)
            print(f"[molecule_setup] OpenFF path: {name} ({smiles}), "
                  f"{sysobj.system.getNumParticles()} atoms, "
                  f"{len(sysobj.alchemical_atoms)} alchemical")
            return sysobj
        except Exception as exc:  # noqa: BLE001 - we want a graceful fallback
            print(f"[molecule_setup] OpenFF path failed ({exc!r}); using fallback.")

    sysobj = build_fallback(name=name)
    print(f"[molecule_setup] Fallback path: {sysobj.system.getNumParticles()} atoms, "
          f"{len(sysobj.alchemical_atoms)} alchemical")
    return sysobj
