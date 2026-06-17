"""Single-topology hybrid builder driven by PRE-ALIGNED SDF poses + a GIVEN atom map.

This is the input mode a real RBFE campaign uses (e.g. the FEP+ JACS benchmarks):
  1. a protein PDB,
  2. a congeneric ligand series as aligned 3D poses (one SDF, shared frame),
  3. a perturbation map whose edges carry a precomputed atom mapping (Kartograf/
     Lomap), i.e. ligand_a_index -> ligand_b_index over the SDF atom order.

Versus single_topology.build_hybrid_single_topology (which takes two SMILES,
computes its own MCS, generates conformers, and Kabsch-aligns), here we:
  * parametrize each ligand from its SDF RDKit mol (explicit H, real pose),
  * use the GIVEN mapping as the shared core (no MCS),
  * place every atom at its SDF coordinate -- because the series is pre-aligned,
    ligand A's unique atoms are already in the binding-site frame, so NO
    alignment is needed (the A-core and B-core nearly superimpose by construction).

The merge itself (append A-unique particles + their bonded terms / exceptions,
zero A-unique<->B-unique interactions, morph the shared-core charges) is the same
physics as the SMILES builder. Produces a relative_setup.TwoLigandSetup, so the
two-region d-AFED / MBAR runners drive it unchanged.

Scope (v1, same as the SMILES engine): neutral pairs, no NET-charge change, and
the mapping must define a clean core + unique partition (atom add/delete or single
R-group swap). Edges that are ELEMENT CHANGES at a mapped core atom (e.g. H->Br
mapped 1:1, 0 unique atoms) are NOT handled -- the fixed core keeps B's vdW for
that atom in both end states. Skip / fall back for those.
"""

import numpy as np
import openmm
from openmm import app, unit

from .single_topology import _force, CORE_MORPH_PARAM


def partition_from_mapping(rdmol_a, rdmol_b, a_to_b, break_element_changes=True):
    """core_map / unique_a / unique_b from a given A->B atom map (SDF indices).

    ``break_element_changes`` (default True): any mapped pair whose elements differ
    (e.g. an H mapped 1:1 to a Br) is UN-mapped -- the A atom becomes A-unique and
    the B atom B-unique -- so the softcore decouple/couple handles it instead of
    the fixed core (which cannot morph a core atom's vdW/bond). This turns element
    changes into ordinary single-topology R-group swaps. Requires the overall pair
    to stay neutral (no net-charge change), which holds for halogen/H substituent
    scans. Set False to keep the raw mapping (then 0-unique element edges are out
    of scope).
    """
    core_map = {int(a): int(b) for a, b in a_to_b.items()}
    if break_element_changes:
        for a in list(core_map):
            za = rdmol_a.GetAtomWithIdx(a).GetAtomicNum()
            zb = rdmol_b.GetAtomWithIdx(core_map[a]).GetAtomicNum()
            if za != zb:
                del core_map[a]                 # -> a in unique_a, b in unique_b
    mapped_a = set(core_map)
    mapped_b = set(core_map.values())
    unique_a = [i for i in range(rdmol_a.GetNumAtoms()) if i not in mapped_a]
    unique_b = [i for i in range(rdmol_b.GetNumAtoms()) if i not in mapped_b]
    return core_map, unique_a, unique_b


def _offmol(rdmol):
    from openff.toolkit import Molecule
    return Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)


def _parametrize_mol_gas(rdmol, hydrogen_mass_amu=None, flexible=False):
    """OpenFF gas-phase (NoCutoff) System for an RDKit mol, atom order preserved."""
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    from .molecule_setup import DEFAULT_SMALL_MOLECULE_FF

    offmol = _offmol(rdmol)
    smirnoff = SMIRNOFFTemplateGenerator(
        molecules=offmol, forcefield=DEFAULT_SMALL_MOLECULE_FF)
    ff = app.ForceField()
    ff.registerTemplateGenerator(smirnoff.generator)
    top = offmol.to_topology().to_openmm()
    pos = np.array(offmol.conformers[0].to_openmm().value_in_unit(unit.nanometer))
    cons = None if flexible else app.HBonds
    kwargs = dict(nonbondedMethod=app.NoCutoff, constraints=cons,
                  rigidWater=(cons is not None))
    if hydrogen_mass_amu is not None:
        kwargs["hydrogenMass"] = hydrogen_mass_amu * unit.amu
    return ff.createSystem(top, **kwargs), top, pos


def _solvate_mol(rdmol, padding_nm=1.0, water_model="tip3p",
                 hydrogen_mass_amu=None, flexible=False):
    """Solvate an RDKit mol (ligand-first) -> (system, topology, positions_nm)."""
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    from .molecule_setup import DEFAULT_SMALL_MOLECULE_FF

    offmol = _offmol(rdmol)
    smirnoff = SMIRNOFFTemplateGenerator(
        molecules=offmol, forcefield=DEFAULT_SMALL_MOLECULE_FF)
    ff = app.ForceField("amber/tip3p_standard.xml", "amber/tip3p_HFE_multivalent.xml")
    ff.registerTemplateGenerator(smirnoff.generator)
    top = offmol.to_topology().to_openmm()
    pos = offmol.conformers[0].to_openmm()
    modeller = app.Modeller(top, pos)
    modeller.addSolvent(ff, model=water_model, padding=padding_nm * unit.nanometer)
    cons = None if flexible else app.HBonds
    kwargs = dict(nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
                  constraints=cons, rigidWater=(cons is not None))
    if hydrogen_mass_amu is not None:
        kwargs["hydrogenMass"] = hydrogen_mass_amu * unit.amu
    if flexible:
        kwargs["rigidWater"] = False
    system = ff.createSystem(modeller.topology, **kwargs)
    pos_nm = np.array(modeller.positions.value_in_unit(unit.nanometer))
    return system, modeller.topology, pos_nm


def build_hybrid_from_mapped_mols(rdmol_a, rdmol_b, a_to_b, base=None,
                                  solvate=True, padding_nm=1.0,
                                  water_model="tip3p", hydrogen_mass_amu=None,
                                  flexible=False, name="A->B",
                                  break_element_changes=True):
    """Hybrid single-topology A->B from SDF mols + a given mapping (SDF indices).

    ``base`` (optional): a complex_setup SolvatedSystem with ligand B FIRST at its
    SDF pose (the complex leg). Otherwise B is solvated here (solvent leg). A-unique
    atoms are appended at THEIR SDF coordinates (the series is pre-aligned). Returns
    a relative_setup.TwoLigandSetup.
    """
    from .relative_setup import (
        TwoLigandSetup, _build_two_region_alchemical, two_region_schedule)

    core_map, unique_a, unique_b = partition_from_mapping(
        rdmol_a, rdmol_b, a_to_b, break_element_changes=break_element_changes)
    if not unique_a and not unique_b:
        raise ValueError(
            "mapping has 0 unique atoms and break_element_changes=False; nothing "
            "to transform (element change at a mapped core atom is out of scope).")

    # (1) Base environment with ligand B first (complex) or solvate B here (solvent).
    if base is not None:
        merged = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(base.system))
        merged_top = base.topology
        posB_nm = np.array(base.positions.value_in_unit(unit.nanometer))
    else:
        sysB, merged_top, posB_nm = _solvate_mol(
            rdmol_b, padding_nm=padding_nm, water_model=water_model,
            hydrogen_mass_amu=hydrogen_mass_amu, flexible=flexible)
        merged = sysB

    # (2) Gas-phase A for its unique params / bonded terms (atom order = SDF).
    sysA, _, posA_nm = _parametrize_mol_gas(
        rdmol_a, hydrogen_mass_amu=hydrogen_mass_amu, flexible=flexible)

    nbB = _force(merged, openmm.NonbondedForce)
    nbA = _force(sysA, openmm.NonbondedForce)
    a_to_merged = dict(core_map)              # core A atoms reuse B's particles

    # (3) Append A-unique particles + nonbonded params (HMR mass-conserving).
    for u in unique_a:
        a_to_merged[u] = merged.addParticle(sysA.getParticleMass(u))
        q, sig, eps = nbA.getParticleParameters(u)
        nbB.addParticle(q, sig, eps)
        if hydrogen_mass_amu is not None:
            m_u = sysA.getParticleMass(u).value_in_unit(unit.dalton)
            excess = m_u - 1.008
            if excess > 1e-6:
                for nbr in rdmol_a.GetAtomWithIdx(u).GetNeighbors():
                    j = nbr.GetIdx()
                    if j in core_map:
                        mi = core_map[j]
                        m_core = merged.getParticleMass(mi).value_in_unit(unit.dalton)
                        merged.setParticleMass(mi, (m_core - excess) * unit.dalton)
                        break

    unique_a_set = set(unique_a)

    def touches_a(atoms):
        return any(a in unique_a_set for a in atoms)

    # (4) Copy A's bonded terms / constraints / exceptions that touch an A-unique atom.
    bA, bM = _force(sysA, openmm.HarmonicBondForce), _force(merged, openmm.HarmonicBondForce)
    if bA and bM:
        for k in range(bA.getNumBonds()):
            i, j, length, kk = bA.getBondParameters(k)
            if touches_a((i, j)):
                bM.addBond(a_to_merged[i], a_to_merged[j], length, kk)
    aA, aM = _force(sysA, openmm.HarmonicAngleForce), _force(merged, openmm.HarmonicAngleForce)
    if aA and aM:
        for k in range(aA.getNumAngles()):
            i, j, l, ang, kk = aA.getAngleParameters(k)
            if touches_a((i, j, l)):
                aM.addAngle(a_to_merged[i], a_to_merged[j], a_to_merged[l], ang, kk)
    tA, tM = _force(sysA, openmm.PeriodicTorsionForce), _force(merged, openmm.PeriodicTorsionForce)
    if tA and tM:
        for k in range(tA.getNumTorsions()):
            i, j, l, m, per, ph, kk = tA.getTorsionParameters(k)
            if touches_a((i, j, l, m)):
                tM.addTorsion(a_to_merged[i], a_to_merged[j], a_to_merged[l],
                              a_to_merged[m], per, ph, kk)
    for k in range(sysA.getNumConstraints()):
        i, j, dist = sysA.getConstraintParameters(k)
        if touches_a((i, j)):
            merged.addConstraint(a_to_merged[i], a_to_merged[j], dist)
    for k in range(nbA.getNumExceptions()):
        i, j, qp, sig, eps = nbA.getExceptionParameters(k)
        if touches_a((i, j)):
            nbB.addException(a_to_merged[i], a_to_merged[j], qp, sig, eps)

    # (5) Zero A-unique <-> B-unique interactions (alternate end states).
    existing = set()
    for k in range(nbB.getNumExceptions()):
        i, j, *_ = nbB.getExceptionParameters(k)
        existing.add((min(i, j), max(i, j)))
    for u in unique_a:
        mu = a_to_merged[u]
        for v in unique_b:
            key = (min(mu, v), max(mu, v))
            if key not in existing:
                nbB.addException(mu, v, 0.0 * unit.elementary_charge ** 2,
                                 0.1 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
                existing.add(key)

    # (6) Positions: A-unique atoms at THEIR SDF coordinates (series pre-aligned).
    a_unique_pos = np.array([posA_nm[u] for u in unique_a]) if unique_a \
        else np.zeros((0, 3))
    merged_pos = np.vstack([posB_nm, a_unique_pos]) * unit.nanometer
    if unique_a:
        chain = merged_top.addChain()
        res = merged_top.addResidue("DUA", chain)
        for u in unique_a:
            sym = rdmol_a.GetAtomWithIdx(u).GetSymbol()
            merged_top.addAtom(f"{sym}x", app.Element.getBySymbol(sym), res)

    # (7) Shared-core charge morph dq = q_A - q_B (captured now, applied as offsets).
    core_dq, max_dq = {}, 0.0
    for a, b in core_map.items():
        qa = nbA.getParticleParameters(a)[0].value_in_unit(unit.elementary_charge)
        qb = nbB.getParticleParameters(b)[0].value_in_unit(unit.elementary_charge)
        if abs(qa - qb) > 1e-6:
            core_dq[b] = qa - qb
            max_dq = max(max_dq, abs(qa - qb))

    # (8) Two alchemical regions on the unique atoms; core stays coupled.
    atoms_a = [a_to_merged[u] for u in unique_a]
    atoms_b = list(unique_b)
    alch_system, lambda_params = _build_two_region_alchemical(merged, atoms_a, atoms_b)

    # (9) Install core charge-morph offsets + schedule (1=A -> 0=B).
    schedule = two_region_schedule()
    if core_dq:
        nb = _force(alch_system, openmm.NonbondedForce)
        nb.addGlobalParameter(CORE_MORPH_PARAM, 1.0)
        for idx, dq in core_dq.items():
            nb.addParticleParameterOffset(CORE_MORPH_PARAM, idx, dq, 0.0, 0.0)
        lambda_params = sorted(set(lambda_params) | {CORE_MORPH_PARAM})
        K = len(schedule)
        for k, window in enumerate(schedule):
            window[CORE_MORPH_PARAM] = 1.0 - k / (K - 1)

    return TwoLigandSetup(
        system=alch_system, lambda_parameters=lambda_params,
        positions=merged_pos, topology=merged_top,
        atoms_a=atoms_a, atoms_b=atoms_b, schedule=schedule, name=name,
        metadata={"engine": "mapped_single_topology",
                  "n_core": len(core_map), "n_unique_a": len(unique_a),
                  "n_unique_b": len(unique_b), "core_charge_morph": bool(core_dq),
                  "max_core_dq_e": round(max_dq, 4), "solvated": solvate,
                  "leg": "solvent" if base is None else "complex"})
