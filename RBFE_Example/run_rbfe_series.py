#!/usr/bin/env python
"""Run a series of relative binding free energies (RBFE) over a perturbation map.

Inputs (the three things a real RBFE campaign needs):
  1. --pdb      a prepared protein structure (apo or with crystal ligand; the
                ligand/heterogens/waters are stripped and hydrogens added).
  2. --ligands  an SDF of the congeneric series as ALIGNED 3D poses (shared frame).
  3. --edges    a perturbation map (OpenFE/Kartograf/Lomap YAML) whose edges carry
                ligand_a, ligand_b and a precomputed atom mapping (A_idx -> B_idx
                over the SDF atom order).

For each edge it builds a hybrid single-topology system (shared core fully coupled,
only the mapped R-group difference alchemically mutated) in the protein complex and
in water, runs d-AFED on each (default integrator HMR-MTS "4/4/8"), and reports

    ddG_bind(A->B) = dG_transform(complex) - dG_transform(solvent)

against experiment (if --exp ligands.yml with IC50/Ki is given). Writes a results
CSV + a per-edge tau(t) npz; plot with scripts/plot_rbfe_lambda_time.py.

Scope (engine v1): neutral congeneric pairs, no NET-charge change, the mapping must
define a clean core + unique partition. Edges that are pure ELEMENT CHANGES at a
mapped core atom (0 unique atoms, e.g. H<->Br mapped 1:1) are skipped with a note.

Example (the bundled CDK2 / Wang 2015 JACS FEP+ set):
  python run_rbfe_series.py \
      --pdb data/cdk2/protein.pdb --ligands data/cdk2/ligands.sdf \
      --edges data/cdk2/edges.yml --exp data/cdk2/ligands.yml \
      --mode smoke --out outputs/cdk2
"""
import argparse
import csv
import math
import os
import sys

# Make the dynlambda package importable when run from this directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

import yaml
from rdkit import Chem

from dynlambda.relative_binding import relative_binding_mapped
from dynlambda.mapped_single_topology import partition_from_mapping

RT_KCAL = 0.0019872041 * 298.15      # kcal/mol at 298.15 K


def load_ligands(sdf_path):
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    return {m.GetProp("_Name"): m for m in suppl if m is not None}


def load_edges(edges_path):
    doc = yaml.safe_load(open(edges_path))
    out = []
    for name, e in doc["edges"].items():
        out.append((str(e["ligand_a"]), str(e["ligand_b"]), e["atom mapping"]))
    return out


def load_exp_dg(exp_path):
    """ligands.yml -> per-ligand experimental dG (kcal/mol).

    Handles both layouts: OpenFF `measurement: {value, unit, type}` IC50/Ki
    (dG = RT ln value[M]) and the jacs_set `dg_exp_kcal:` direct dG.
    """
    if not exp_path or not os.path.exists(exp_path):
        return {}
    doc = yaml.safe_load(open(exp_path))
    units = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12}
    dg = {}
    for name, d in doc.items():
        if not isinstance(d, dict):
            continue
        if d.get("dg_exp_kcal") not in (None, ""):
            dg[str(name)] = float(d["dg_exp_kcal"])
            continue
        m = d.get("measurement", {})
        val, unit = m.get("value"), m.get("unit")
        if val and unit in units:
            dg[str(name)] = RT_KCAL * math.log(val * units[unit])
    return dg


MD_STEPS_PER_BLOCK = 12          # x 8 fs outer = 96 fs of MD per d-AFED block
OUTER_FS = 8.0


def sim_time_to_blocks(sim_time_ns):
    """Convert per-leg simulation time (ns) -> number of d-AFED blocks."""
    return max(1, round(sim_time_ns * 1e6 / (MD_STEPS_PER_BLOCK * OUTER_FS)))


def make_config(sim_time_ns=5.0, lambda_ts=2000.0, lambda_mass=200.0):
    """d-AFED + HMR-MTS 4/4/8 config.

    sim_time_ns is the MD time PER LEG (complex and solvent each); lambda_ts /
    lambda_mass are the d-AFED extended-system temperature (K) and fictitious mass.
    """
    return dict(
        temperature=298.15, padding_nm=1.0,
        integrator="mts448", timestep_fs=OUTER_FS, mts_inner_steps=2,
        hydrogen_mass_amu=3.0, flexible=False, prefer_gpu=True, seed=1,
        progress=False,
        dynamic=dict(n_blocks=sim_time_to_blocks(sim_time_ns),
                     md_steps_per_block=MD_STEPS_PER_BLOCK,
                     lambda_temperature=lambda_ts, lambda_mass=lambda_mass,
                     barrier_height=12.0, tau0=0.5, basin_edge=0.2,
                     reweight_drop_frac=0.3))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--ligands", required=True, help="aligned-pose SDF")
    ap.add_argument("--edges", required=True, help="perturbation-map YAML")
    ap.add_argument("--exp", default=None, help="ligands.yml with IC50/Ki (optional)")
    ap.add_argument("--out", default="outputs/rbfe_series")
    ap.add_argument("--sim-time-ns", type=float, default=5.0,
                    help="MD time PER LEG, ns (complex and solvent each); default 5")
    ap.add_argument("--lambda-ts", type=float, default=2000.0,
                    help="d-AFED extended-system temperature T_s (K); default 2000")
    ap.add_argument("--lambda-mass", type=float, default=200.0,
                    help="d-AFED fictitious lambda mass; default 200")
    ap.add_argument("--edge", default=None,
                    help="run only this edge, 'ligA:ligB' (default: all)")
    ap.add_argument("--max-edges", type=int, default=None)
    ap.add_argument("--n-blocks", type=int, default=None,
                    help="override d-AFED blocks/leg (else set by --sim-time-ns)")
    args = ap.parse_args()

    mols = load_ligands(args.ligands)
    edges = load_edges(args.edges)
    exp_dg = load_exp_dg(args.exp)
    cfg = make_config(args.sim_time_ns, args.lambda_ts, args.lambda_mass)
    if args.n_blocks:
        cfg["dynamic"]["n_blocks"] = args.n_blocks
    os.makedirs(args.out, exist_ok=True)

    if args.edge:
        a, b = args.edge.split(":")
        edges = [e for e in edges if e[0] == a and e[1] == b]

    nb = cfg["dynamic"]["n_blocks"]
    ns = nb * MD_STEPS_PER_BLOCK * OUTER_FS / 1e6
    inner = cfg["timestep_fs"] / cfg["mts_inner_steps"]
    print(f"RBFE series: {len(edges)} edge(s), {ns:.2f} ns/leg ({nb} blocks), "
          f"integrator=mts448 ({inner:.0f}/{inner:.0f}/{cfg['timestep_fs']:.0f} fs), "
          f"T_s={cfg['dynamic']['lambda_temperature']:.0f} K, "
          f"lambda_mass={cfg['dynamic']['lambda_mass']:.0f}")
    print("cycle: ddG_bind = dG_transform(complex) - dG_transform(solvent)\n")

    rows = []
    n_done = 0
    for a, b, amap in edges:
        if args.max_edges and n_done >= args.max_edges:
            break
        if a not in mols or b not in mols:
            print(f"  {a}->{b}: SKIP (ligand not in SDF)")
            continue
        # break_element_changes=True (default) un-maps element-mismatched pairs
        # (e.g. H<->Br) into unique atoms, so element-change edges run as ordinary
        # softcore swaps. Only a truly identical pair (0 changes) is skipped.
        _, ua, ub = partition_from_mapping(mols[a], mols[b], amap)
        if not ua and not ub:
            print(f"  {a}->{b}: SKIP (identical under the mapping; nothing to morph)")
            continue

        ddg_exp = (exp_dg[b] - exp_dg[a]) if (a in exp_dg and b in exp_dg) else None
        print(f"  {a}->{b}: uniqueA={len(ua)} uniqueB={len(ub)} ... running")
        try:
            res = relative_binding_mapped(
                mols[a], mols[b], amap, args.pdb, config=cfg,
                output_dir=os.path.join(args.out, f"{a}__{b}"), name=f"{a}->{b}")
        except Exception as exc:  # noqa: BLE001 -- report, keep going
            print(f"    FAILED: {type(exc).__name__}: {str(exc)[:120]}")
            rows.append((a, b, "", ddg_exp if ddg_exp is not None else "", "FAILED"))
            continue

        ddg = res.delta_g_kcal_mol
        err = (ddg - ddg_exp) if ddg_exp is not None else None
        print(f"    ddG_bind = {ddg:+.2f} kcal/mol"
              + (f"  (exp {ddg_exp:+.2f}, err {err:+.2f})" if ddg_exp is not None else "")
              + f"  [complex {res.metadata['complex_leg_kcal']:+.2f} - "
                f"solvent {res.metadata['solvent_leg_kcal']:+.2f}]")
        rows.append((a, b, round(ddg, 2),
                     round(ddg_exp, 2) if ddg_exp is not None else "",
                     round(err, 2) if err is not None else ""))
        n_done += 1

    csv_path = os.path.join(args.out, "rbfe_results.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ligand_a", "ligand_b", "ddG_calc_kcal", "ddG_exp_kcal", "error_kcal"])
        w.writerows(rows)
    print(f"\nwrote {csv_path}")
    if args.sim_time_ns < 1.0:
        print(f"NOTE: {args.sim_time_ns} ns/leg is short -- likely NOT converged.")


if __name__ == "__main__":
    main()
