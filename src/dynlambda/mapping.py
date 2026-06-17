"""Atom mapping between two ligands via an RDKit maximum-common-substructure.

Used by the internal relative free-energy engine (relative_setup.py) to decide
which atoms form the shared single-topology core (mapped) and which are unique
to one end state (grown/shrunk through softcore). We use RDKit's rdFMCS so the
package needs no Lomap/Kartograf/OpenFE to produce a mapping; those remain
optional backends.
"""

from dataclasses import dataclass, field


@dataclass
class LigandMapping:
    smiles_a: str
    smiles_b: str
    mol_a: object                       # RDKit Mol (heavy atoms)
    mol_b: object
    atom_map: dict = field(default_factory=dict)   # a_idx -> b_idx (shared core)
    unique_a: list = field(default_factory=list)   # a indices not in core
    unique_b: list = field(default_factory=list)
    mcs_smarts: str = ""

    @property
    def n_mapped(self):
        return len(self.atom_map)

    def mapped_atomic_numbers(self):
        """Atomic numbers of the mapped core atoms (from mol_a)."""
        return [self.mol_a.GetAtomWithIdx(i).GetAtomicNum()
                for i in self.atom_map]

    def n_mapped_aromatic_carbons(self):
        n = 0
        for i in self.atom_map:
            a = self.mol_a.GetAtomWithIdx(i)
            if a.GetAtomicNum() == 6 and a.GetIsAromatic():
                n += 1
        return n


def _mol_from_smiles(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    return mol


@dataclass
class HybridPartition:
    """Explicit-H atom partition for a single-topology A->B transformation.

    Indices are over the explicit-hydrogen RDKit molecules (Chem.AddHs of the
    SMILES), which is the SAME atom order OpenFF/OpenMM uses, so these indices
    map straight onto the built System's particles.

    core_map  : {a_idx -> b_idx} atoms shared by A and B (heavy MCS + matched H).
                These stay fully coupled (non-alchemical) in the hybrid.
    unique_a  : A indices NOT in the core (vanish as lambda 0->1).
    unique_b  : B indices NOT in the core (grow as lambda 0->1).
    mol_a/mol_b : the explicit-H RDKit mols (for bonds/positions during the merge).
    """
    mol_a: object
    mol_b: object
    core_map: dict = field(default_factory=dict)
    unique_a: list = field(default_factory=list)
    unique_b: list = field(default_factory=list)

    @property
    def core_a(self):
        return list(self.core_map.keys())

    @property
    def core_b(self):
        return list(self.core_map.values())


def _explicit_h_mol(smiles):
    """Chem.AddHs(MolFromSmiles) -- explicit-H mol in OpenFF/OpenMM atom order."""
    from rdkit import Chem
    return Chem.AddHs(_mol_from_smiles(smiles))


def hydrogen_aware_partition(smiles_a, smiles_b, ring_matches_ring_only=True,
                             complete_rings_only=True, timeout=30):
    """Full atom-level (heavy + H) core/unique partition for A->B.

    1. Heavy-atom MCS (element- and ring-aware) gives the shared heavy core.
    2. For each mapped heavy pair, pair up their hydrogens: min(nH_a, nH_b) of
       them join the core (H's on one atom are interchangeable), any surplus H on
       the A side becomes unique_a and on the B side unique_b. So an H that A
       carries where B has a substituent (e.g. benzene's H at the methyl position
       of toluene) is correctly flagged unique_a.
    3. Atoms in neither core list are unique (the swapped R-group + its H's).

    Returns a HybridPartition over explicit-H indices. Raises if there is no
    common substructure.
    """
    from rdkit.Chem import rdFMCS

    mol_a = _explicit_h_mol(smiles_a)
    mol_b = _explicit_h_mol(smiles_b)

    mcs = rdFMCS.FindMCS(
        [mol_a, mol_b],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=ring_matches_ring_only,
        completeRingsOnly=complete_rings_only,
        timeout=timeout,
    )
    if mcs.numAtoms == 0:
        raise ValueError(f"no common substructure between {smiles_a!r} and "
                         f"{smiles_b!r}")

    from rdkit import Chem
    patt = Chem.MolFromSmarts(mcs.smartsString)
    match_a = mol_a.GetSubstructMatch(patt)
    match_b = mol_b.GetSubstructMatch(patt)
    if not match_a or not match_b:
        raise ValueError("MCS pattern did not match both ligands")

    core_map = {a: b for a, b in zip(match_a, match_b)}

    # Pair hydrogens hanging off each mapped heavy atom.
    def h_neighbors(mol, idx):
        return [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
                if nbr.GetAtomicNum() == 1]

    for a_heavy, b_heavy in list(core_map.items()):
        if mol_a.GetAtomWithIdx(a_heavy).GetAtomicNum() == 1:
            continue  # heavy-MCS already paired this (shouldn't be H)
        ha = [h for h in h_neighbors(mol_a, a_heavy) if h not in core_map]
        hb = [h for h in h_neighbors(mol_b, b_heavy) if h not in core_map.values()]
        for h_a, h_b in zip(ha, hb):     # min(len) pairs
            core_map[h_a] = h_b

    core_a = set(core_map.keys())
    core_b = set(core_map.values())
    unique_a = [i for i in range(mol_a.GetNumAtoms()) if i not in core_a]
    unique_b = [i for i in range(mol_b.GetNumAtoms()) if i not in core_b]

    return HybridPartition(mol_a=mol_a, mol_b=mol_b, core_map=core_map,
                           unique_a=unique_a, unique_b=unique_b)


def map_ligands(smiles_a, smiles_b, ring_matches_ring_only=True,
                complete_rings_only=True, timeout=30):
    """Return a LigandMapping for A->B from their MCS (heavy atoms).

    The MCS core is the set of atoms shared by both ligands; atoms outside it are
    'unique' (the part that is alchemically transformed). Element-matched,
    ring-aware MCS so aromatic rings map to aromatic rings.
    """
    from rdkit.Chem import rdFMCS

    mol_a = _mol_from_smiles(smiles_a)
    mol_b = _mol_from_smiles(smiles_b)

    mcs = rdFMCS.FindMCS(
        [mol_a, mol_b],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=ring_matches_ring_only,
        completeRingsOnly=complete_rings_only,
        timeout=timeout,
    )
    if mcs.numAtoms == 0:
        raise ValueError(f"no common substructure between {smiles_a!r} and "
                         f"{smiles_b!r}")

    from rdkit import Chem
    patt = Chem.MolFromSmarts(mcs.smartsString)
    match_a = mol_a.GetSubstructMatch(patt)
    match_b = mol_b.GetSubstructMatch(patt)

    atom_map = {a: b for a, b in zip(match_a, match_b)}
    unique_a = [i for i in range(mol_a.GetNumAtoms()) if i not in match_a]
    unique_b = [i for i in range(mol_b.GetNumAtoms()) if i not in match_b]

    return LigandMapping(
        smiles_a=smiles_a, smiles_b=smiles_b, mol_a=mol_a, mol_b=mol_b,
        atom_map=atom_map, unique_a=unique_a, unique_b=unique_b,
        mcs_smarts=mcs.smartsString,
    )
