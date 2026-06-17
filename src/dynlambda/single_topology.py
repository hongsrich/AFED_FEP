"""Hybrid single-topology builder for relative hydration (A -> B).

Why
---
The dual-topology engine (relative_setup.build_two_ligand_box) puts both whole
ligands in the box, so going A->B *inserts a whole B and deletes a whole A* --
a large perturbation whose insertion leg is badly under-sampled at short MD,
biasing ddG (benzene->toluene came out ~+2.9 instead of ~0).

Single (hybrid) topology shares the maximum-common substructure ("core") as ONE
set of atoms that stays fully coupled the entire time, and alchemically mutates
only the small R-group difference: atoms unique to A vanish while atoms unique to
B grow. The core's (large) hydration contribution is identical in both end states
and cancels exactly, so only the cheap, well-overlapping difference is sampled.

Construction (fixed-core, v1)
-----------------------------
1. Parametrize + solvate B (molecule_setup.build_from_smiles): the core and the
   B-unique atoms are present and fully parametrized; B's solute atoms come first.
2. Parametrize A alone in the gas phase (OpenFF) to read A-unique nonbonded /
   bonded parameters.
3. Partition atoms (mapping.hydrogen_aware_partition) into core (a_idx->b_idx),
   unique_a, unique_b -- over explicit-H indices, which match the System order.
4. Append the A-unique atoms to B's solvated system (so existing indices don't
   move): add their particles + nonbonded params (from A), the bonded terms /
   constraints / exceptions that involve them (from A, remapping core indices
   A->merged), and zeroed A-unique <-> B-unique exceptions (the two alternate
   end-state groups must never interact). The shared core keeps B's parameters
   (the v1 "fixed core" approximation: core charge differences, tiny for neutral
   congeneric pairs, are neglected).
5. Place the A-unique atoms by Kabsch-aligning A's core onto B's core.
6. Build two alchemical regions on the UNIQUE subsets only (the core is left
   non-alchemical -> always coupled) and return a relative_setup.TwoLigandSetup,
   so run_two_region_leg (MBAR) and run_dynamic_lambda_two_region (d-AFED) drive
   it unchanged. At lambda=0: A-unique on, B-unique ghost = ligand A; at
   lambda=1: reversed = ligand B.

IMPORTANT: a single leg here is the A->B *transformation* free energy, which
includes the intramolecular morphing of the fragment against its own core (the
region alchemy sees the core as 'environment'). The relative HYDRATION free
energy needs TWO legs -- solvent minus vacuum -- so that intramolecular part
cancels: ddG_hyd(A->B) = dG_transform_solvent - dG_transform_vacuum. The wrappers
relative_hydration.relative_hydration_single_topology[_dafed] do both legs; build
the box with solvate=True for the solvent leg and solvate=False for the vacuum leg.

Scope (v1): neutral congeneric pairs, no net-charge change, no ring make/break.
"""

import numpy as np
import openmm
from openmm import app, unit

# Global parameter that interpolates the shared-core partial charges:
# 1 -> ligand A's charges, 0 -> ligand B's. Driven 1->0 along the A->B schedule.
CORE_MORPH_PARAM = "lambda_core_a"


def _force(system, cls):
    for f in system.getForces():
        if isinstance(f, cls):
            return f
    return None


def _parametrize_gas(smiles, hydrogen_mass_amu=None, flexible=False,
                     constraints="default"):
    """OpenFF-parametrize a single molecule in the gas phase (NoCutoff).

    Atom order matches mapping.hydrogen_aware_partition (both come from
    Chem.AddHs(MolFromSmiles)), so indices line up with the partition.
    hydrogen_mass_amu applies hydrogen mass repartitioning (HMR) so the A-unique
    masses (and the vacuum-leg system) match the solvent leg's HMR.
    """
    from openff.toolkit import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    from .molecule_setup import _rdkit_mol_from_smiles, DEFAULT_SMALL_MOLECULE_FF

    rdmol = _rdkit_mol_from_smiles(smiles)
    offmol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)
    smirnoff = SMIRNOFFTemplateGenerator(
        molecules=offmol, forcefield=DEFAULT_SMALL_MOLECULE_FF)
    forcefield = app.ForceField()
    forcefield.registerTemplateGenerator(smirnoff.generator)
    top = offmol.to_topology().to_openmm()
    pos = np.array(offmol.conformers[0].to_openmm().value_in_unit(unit.nanometer))
    cons = (None if flexible else app.HBonds)
    if constraints != "default":
        cons = constraints
    kwargs = dict(nonbondedMethod=app.NoCutoff, constraints=cons,
                  rigidWater=(cons is not None))
    if hydrogen_mass_amu is not None:
        kwargs["hydrogenMass"] = hydrogen_mass_amu * unit.amu
    system = forcefield.createSystem(top, **kwargs)
    return system, top, pos


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


def build_hybrid_single_topology(smiles_a, smiles_b, partition=None,
                                 padding_nm=1.0, water_model="tip3p",
                                 solvate=True, vacuum_pme_box_nm=None,
                                 hydrogen_mass_amu=None, flexible=False,
                                 constraints="default", base=None):
    """Build a hybrid single-topology A->B system as a TwoLigandSetup.

    See module docstring. Returns relative_setup.TwoLigandSetup with atoms_a =
    merged indices of the A-unique atoms (decouple as lambda 0->1) and atoms_b =
    indices of the B-unique atoms (couple), the shared core left fully coupled.

    ``base`` (optional): a pre-built SolvatedSystem whose ligand-B atoms come
    FIRST, in the same atom order as molecule_setup._rdkit_mol_from_smiles(B)
    (e.g. complex_setup.build_complex_from_smiles). When given it replaces the
    internal solvate/gas construction of B's environment, so the SAME merge turns
    this into a complex leg (protein+water environment) instead of a solvent leg.
    """
    from .mapping import hydrogen_aware_partition
    from .molecule_setup import build_from_smiles
    from .relative_setup import (
        TwoLigandSetup, _build_two_region_alchemical, two_region_schedule)

    if partition is None:
        partition = hydrogen_aware_partition(smiles_a, smiles_b)
    core_map = partition.core_map           # a_idx -> b_idx (explicit-H)
    unique_a = list(partition.unique_a)
    unique_b = list(partition.unique_b)
    if not unique_a and not unique_b:
        raise ValueError("A and B are identical under the mapping; nothing to "
                         "transform.")

    # (1) Solvate B: solute first, then water. B solute index == mol_b index.
    # hydrogen_mass_amu applies HMR (heavier H, lighter bonded heavy atom) so a
    # larger MD timestep (3-4 fs) is stable -- use with integrator='mts'.
    if base is not None:
        # Pre-built environment (e.g. protein-ligand complex) with ligand B first.
        merged = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(base.system))
        merged_top = base.topology
        posB_nm = np.array(base.positions.value_in_unit(unit.nanometer))
    elif not solvate:
        # gas-phase B (vacuum leg / fast build tests): reparametrize without solvent.
        sysB, topB, posB = _parametrize_gas(smiles_b,
                                            hydrogen_mass_amu=hydrogen_mass_amu,
                                            flexible=flexible, constraints=constraints)
        merged = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(sysB))
        merged_top = topB
        posB_nm = np.array(posB)
    else:
        solvatedB = build_from_smiles(
            smiles_b, name="B", padding=padding_nm, water_model=water_model,
            hydrogen_mass_amu=hydrogen_mass_amu, flexible=flexible, constraints=constraints)
        merged = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(solvatedB.system))
        merged_top = solvatedB.topology
        posB_nm = np.array(
            solvatedB.positions.value_in_unit(unit.nanometer))

    n_b_total = merged.getNumParticles()

    # (2) Gas-phase A to read A-unique parameters/terms (HMR-matched to B).
    sysA, topA, posA_nm = _parametrize_gas(smiles_a,
                                           hydrogen_mass_amu=hydrogen_mass_amu,
                                           flexible=flexible, constraints=constraints)

    # (3) Remap: a_idx -> merged index. Core a -> its B index; unique a -> append.
    a_to_merged = dict(core_map)            # core atoms reuse B's particles
    nbB = _force(merged, openmm.NonbondedForce)
    nbA = _force(sysA, openmm.NonbondedForce)

    # (4) Append A-unique particles to the system AND the NonbondedForce in step.
    # Masses come from sysA, which already carries HMR if requested. To keep HMR
    # mass-conserving, the repartitioned excess of an A-unique H must be subtracted
    # from the CORE heavy atom it bonds to (B's createSystem only knew about B's
    # hydrogens, not this appended one).
    for u in unique_a:
        a_to_merged[u] = merged.addParticle(sysA.getParticleMass(u))
        q, sig, eps = nbA.getParticleParameters(u)
        nbB.addParticle(q, sig, eps)
        if hydrogen_mass_amu is not None:
            m_u = sysA.getParticleMass(u).value_in_unit(unit.dalton)
            excess = m_u - 1.008                       # repartitioned onto this H
            if excess > 1e-6:
                for nbr in partition.mol_a.GetAtomWithIdx(u).GetNeighbors():
                    j = nbr.GetIdx()
                    if j in core_map:                  # the core atom it attaches to
                        mi = core_map[j]
                        m_core = merged.getParticleMass(mi).value_in_unit(unit.dalton)
                        merged.setParticleMass(mi, (m_core - excess) * unit.dalton)
                        break

    unique_a_set = set(unique_a)

    def involves_unique_a(atoms):
        return any(a in unique_a_set for a in atoms)

    # Copy A's bonded terms that touch an A-unique atom (remapped to merged idx).
    bondA = _force(sysA, openmm.HarmonicBondForce)
    bondM = _force(merged, openmm.HarmonicBondForce)
    if bondA and bondM:
        for k in range(bondA.getNumBonds()):
            i, j, length, kk = bondA.getBondParameters(k)
            if involves_unique_a((i, j)):
                bondM.addBond(a_to_merged[i], a_to_merged[j], length, kk)

    angA = _force(sysA, openmm.HarmonicAngleForce)
    angM = _force(merged, openmm.HarmonicAngleForce)
    if angA and angM:
        for k in range(angA.getNumAngles()):
            i, j, l, angle, kk = angA.getAngleParameters(k)
            if involves_unique_a((i, j, l)):
                angM.addAngle(a_to_merged[i], a_to_merged[j], a_to_merged[l],
                              angle, kk)

    torA = _force(sysA, openmm.PeriodicTorsionForce)
    torM = _force(merged, openmm.PeriodicTorsionForce)
    if torA and torM:
        for k in range(torA.getNumTorsions()):
            i, j, l, m, per, phase, kk = torA.getTorsionParameters(k)
            if involves_unique_a((i, j, l, m)):
                torM.addTorsion(a_to_merged[i], a_to_merged[j], a_to_merged[l],
                                a_to_merged[m], per, phase, kk)

    # Constraints touching an A-unique atom (e.g. its X-H bond under HBonds).
    for k in range(sysA.getNumConstraints()):
        i, j, dist = sysA.getConstraintParameters(k)
        if involves_unique_a((i, j)):
            merged.addConstraint(a_to_merged[i], a_to_merged[j], dist)

    # A-unique intramolecular nonbonded exceptions (1-2/1-3 excluded, 1-4 scaled),
    # remapped from A. These never duplicate B's exceptions (B has no A-unique).
    for k in range(nbA.getNumExceptions()):
        i, j, qprod, sig, eps = nbA.getExceptionParameters(k)
        if involves_unique_a((i, j)):
            nbB.addException(a_to_merged[i], a_to_merged[j], qprod, sig, eps)

    # (5) A-unique <-> B-unique: zeroed exceptions (alternate end states, must not
    # interact -- they often overlap in space sharing the same attachment atom).
    existing = set()
    for k in range(nbB.getNumExceptions()):
        i, j, *_ = nbB.getExceptionParameters(k)
        existing.add((min(i, j), max(i, j)))
    for u in unique_a:
        mu = a_to_merged[u]
        for v in unique_b:                  # B-unique merged index == mol_b index
            key = (min(mu, v), max(mu, v))
            if key not in existing:
                nbB.addException(mu, v, 0.0 * unit.elementary_charge ** 2,
                                 0.1 * unit.nanometer,
                                 0.0 * unit.kilojoule_per_mole)
                existing.add(key)

    # (6) Positions: align A's core onto B's core, place A-unique atoms.
    core_a_order = list(core_map.keys())
    A_core = np.array([posA_nm[a] for a in core_a_order])
    B_core = np.array([posB_nm[core_map[a]] for a in core_a_order])
    R, t = _kabsch(A_core, B_core)
    a_unique_pos = np.array([R @ posA_nm[u] + t for u in unique_a]) \
        if unique_a else np.zeros((0, 3))
    merged_pos = np.vstack([posB_nm, a_unique_pos]) * unit.nanometer

    # Topology bookkeeping: append the A-unique atoms as a new residue at the END
    # (they are appended last in the System too, so topology order == System
    # order). They cannot join B's solute residue -- in the solvated topology that
    # residue is followed by water, and a residue's atoms must stay contiguous.
    if unique_a:
        chain = merged_top.addChain()
        res = merged_top.addResidue("DUA", chain)   # A-unique dummy atoms
        for u in unique_a:
            sym = partition.mol_a.GetAtomWithIdx(u).GetSymbol()
            merged_top.addAtom(f"{sym}x", app.Element.getBySymbol(sym), res)

    # (7) Core charge morph: each shared-core atom's partial charge differs between
    # A and B (AM1-BCC sees the whole molecule), so a neighbouring R-group change
    # shifts the core electrostatics. Capture dq = q_A - q_B per core atom now
    # (nbA = gas A, nbB = merged/B); they are applied as NonbondedForce parameter
    # offsets after the factory (step 8) so the core charge interpolates q_A (at
    # lambda_core_a=1) -> q_B (at 0), exactly, in both direct and reciprocal PME.
    core_charge_dq = {}                      # merged core index -> (q_A - q_B) in e
    max_dq = 0.0
    for a, b in core_map.items():
        qa = nbA.getParticleParameters(a)[0].value_in_unit(unit.elementary_charge)
        qb = nbB.getParticleParameters(b)[0].value_in_unit(unit.elementary_charge)
        if abs(qa - qb) > 1e-6:
            core_charge_dq[b] = qa - qb
            max_dq = max(max_dq, abs(qa - qb))

    # Optional diagnostic: give the VACUUM leg the same electrostatics as the
    # solvent leg (PME in a periodic box) instead of NoCutoff, to test whether the
    # two-leg cancellation residual comes from the solvent(PME)-vs-vacuum(NoCutoff)
    # treatment mismatch of the grown fragment's intramolecular Coulomb.
    if (not solvate) and vacuum_pme_box_nm:
        L = float(vacuum_pme_box_nm)
        merged.setDefaultPeriodicBoxVectors(
            openmm.Vec3(L, 0.0, 0.0) * unit.nanometer,
            openmm.Vec3(0.0, L, 0.0) * unit.nanometer,
            openmm.Vec3(0.0, 0.0, L) * unit.nanometer)
        nbm = _force(merged, openmm.NonbondedForce)
        nbm.setNonbondedMethod(openmm.NonbondedForce.PME)
        nbm.setCutoffDistance(1.0 * unit.nanometer)
        pos_nm = np.asarray(merged_pos.value_in_unit(unit.nanometer))
        merged_pos = (pos_nm - pos_nm.mean(axis=0)
                      + np.array([L / 2, L / 2, L / 2])) * unit.nanometer

    # (8) Two alchemical regions on the UNIQUE atoms; core stays coupled.
    atoms_a = [a_to_merged[u] for u in unique_a]
    atoms_b = list(unique_b)                # B-unique merged index == mol_b index
    alch_system, lambda_params = _build_two_region_alchemical(
        merged, atoms_a, atoms_b)

    # (9) Install the core charge-morph offsets on the post-factory NonbondedForce
    # (the non-alchemical core atoms still live there) and add lambda_core_a to the
    # schedule, driven 1 (=A) -> 0 (=B) across the A->B windows.
    schedule = two_region_schedule()
    if core_charge_dq:
        nb = _force(alch_system, openmm.NonbondedForce)
        nb.addGlobalParameter(CORE_MORPH_PARAM, 1.0)   # 1 -> A charges, 0 -> B
        for idx, dq in core_charge_dq.items():
            nb.addParticleParameterOffset(CORE_MORPH_PARAM, idx, dq, 0.0, 0.0)
        lambda_params = sorted(set(lambda_params) | {CORE_MORPH_PARAM})
        K = len(schedule)
        for k, window in enumerate(schedule):
            window[CORE_MORPH_PARAM] = 1.0 - k / (K - 1)

    return TwoLigandSetup(
        system=alch_system, lambda_parameters=lambda_params,
        positions=merged_pos, topology=merged_top,
        atoms_a=atoms_a, atoms_b=atoms_b,
        schedule=schedule, name=f"{smiles_a}->{smiles_b} (single-top)",
        metadata={"engine": "single_topology", "smiles_a": smiles_a,
                  "smiles_b": smiles_b, "n_core": len(core_map),
                  "n_unique_a": len(unique_a), "n_unique_b": len(unique_b),
                  "core_charge_morph": bool(core_charge_dq),
                  "max_core_dq_e": round(max_dq, 4), "solvated": solvate})


def build_hybrid_complex(smiles_a, smiles_b, pdb_path, ligand_resname="BNZ",
                         partition=None, padding_nm=1.0, water_model="tip3p",
                         hydrogen_mass_amu=None, flexible=False,
                         constraints="default"):
    """Hybrid single-topology A->B embedded in a solvated protein-ligand complex.

    The COMPLEX analogue of build_hybrid_single_topology(solvate=True): builds the
    protein-ligand box (ligand B first) via complex_setup.build_complex_from_smiles
    and runs the identical A->B merge on the ligand block, leaving protein+water as
    fully-coupled environment. The returned TwoLigandSetup is the "complex" leg of
    the relative binding cycle ddG_bind = dG_transform(complex) - dG_transform(solvent).
    """
    from .complex_setup import build_complex_from_smiles

    base = build_complex_from_smiles(
        smiles_b, pdb_path, ligand_resname=ligand_resname, name="complexB",
        padding_nm=padding_nm, water_model=water_model,
        hydrogen_mass_amu=hydrogen_mass_amu, flexible=flexible,
        constraints=constraints)
    setup = build_hybrid_single_topology(
        smiles_a, smiles_b, partition=partition,
        hydrogen_mass_amu=hydrogen_mass_amu, flexible=flexible,
        constraints=constraints, base=base)
    setup.name = f"{smiles_a}->{smiles_b} (single-top complex)"
    setup.metadata["leg"] = "complex"
    setup.metadata["pdb"] = pdb_path
    return setup

