"""Download/locate external benchmark structures (SAMPL6 host-guest, T4 L99A).

Per the build decision we ATTEMPT real downloads, but always degrade to clear
instructions if the network is unavailable so the rest of the pipeline (and the
smoke tests) still run. Nothing here fabricates data.
"""

import os
import urllib.request

# SAMPL6 host-guest (octa-acid) inputs live in the SAMPL6 GitHub repo.
SAMPL6_REPO = "https://github.com/MobleyLab/SAMPL6"
SAMPL6_OA_HINT = (
    "SAMPL6 octa-acid (OA) host-guest inputs (OA-G3, OA-G6) are in the MobleyLab "
    "SAMPL6 repo under host_guest/OctaAcidsAndGuests/. Clone it:\n"
    f"    git clone {SAMPL6_REPO}\n"
    "and point the config 'data_dir' at the OA-G3 / OA-G6 mol2/sdf + host files."
)

# T4 lysozyme L99A: use RCSB PDB entries (benzene-bound 181L; others available).
T4_PDB_IDS = {"benzene": "181L", "toluene": "3DMX", "ethylbenzene": "1NHB"}
_RCSB = "https://files.rcsb.org/download/{pdbid}.pdb"


def _try_download(url, dest):
    """Fetch url -> dest. Returns True on success, False (with a note) otherwise."""
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as exc:  # noqa: BLE001 - network optional
        print(f"[data_download] could not fetch {url}: {exc}")
        return False


def download_t4_l99a_pdb(ligand, dest_dir):
    """Download the T4 L99A PDB for a given ligand-bound structure.

    Returns the local path on success, else None (and prints how to get it).
    """
    pdbid = T4_PDB_IDS.get(ligand)
    if pdbid is None:
        raise KeyError(f"no T4 L99A PDB id known for ligand {ligand!r}; "
                       f"known: {sorted(T4_PDB_IDS)}")
    dest = os.path.join(dest_dir, f"{pdbid}.pdb")
    if os.path.exists(dest):
        return dest
    if _try_download(_RCSB.format(pdbid=pdbid), dest):
        print(f"[data_download] T4 L99A {ligand} -> {dest} (PDB {pdbid})")
        return dest
    print(f"[data_download] manual: download https://www.rcsb.org/structure/{pdbid}")
    return None


def locate_sampl6_oa(data_dir):
    """Return data_dir if it already holds SAMPL6 OA files, else print the hint."""
    if data_dir and os.path.isdir(data_dir) and os.listdir(data_dir):
        return data_dir
    print("[data_download] SAMPL6 OA inputs not found locally.")
    print(SAMPL6_OA_HINT)
    return None
