# RBFE_Example — relative binding free energies over a perturbation map

A self-contained example of running a **series of relative binding free energies
(RBFE)** with the `dynlambda` single-topology d-AFED engine, driven by the three
inputs a real campaign uses:

1. **a protein PDB** — `--pdb` (apo, or with a crystal ligand that gets stripped);
2. **an aligned congeneric ligand series** — `--ligands` an SDF of 3D poses in one
   shared frame (already docked/aligned into the binding site);
3. **a perturbation map** — `--edges` an OpenFE/Kartograf/Lomap YAML whose edges
   carry `ligand_a`, `ligand_b`, and a precomputed **atom mapping**
   (`A_index -> B_index` over the SDF atom order).

For each edge the driver builds a **hybrid single-topology** system — the shared
core (the mapped atoms) stays fully coupled, only the R-group difference is
alchemically mutated — in the **protein complex** and in **water**, runs d-AFED on
each, and takes

```
ddG_bind(A->B) = dG_transform(complex) - dG_transform(solvent)
```

The intramolecular morphing of the fragment cancels between the two legs, leaving
the relative binding free energy. If `--exp` (a `ligands.yml` with IC50/Ki) is
given, it reports calc-vs-experiment per edge.

## Benchmark targets (bundled)

`data/<target>/` holds the four files per target. These are the full **FEP+ JACS**
benchmark set (Wang et al. 2015, *JACS* 137, 2695; “Accurate and Reliable
Prediction of Relative Ligand Binding Potency…”) with published FEP+ reference
performance (ΔΔG RMSE ≈ 0.7–1.2 kcal/mol per target; ref `10.1021/ja512751q`).

Source: **OpenFE benchmarks** (`OpenFreeEnergy/openfe-benchmarks`, jacs_set) — the
full literature ligand sets with curated **Kartograf** atom mappings and
experimental ΔG bundled in the OpenFE `LigandNetwork`. Ligand counts match the
literature exactly:

| target | ligands | edges | | target | ligands | edges |
|---|---|---|---|---|---|---|
| bace | 36 | 49 | | p38 | 34 | 51 |
| cdk2 | 16 | 24 | | ptp1b | 23 | 34 |
| jnk1 | 21 | 27 | | thrombin | 11 | 14 |
| mcl1 | 42 | 62 | | tyk2 | 16 | 22 |

Re-download / add targets with the bundled preparer:

```bash
python data/prepare_targets.py                 # all 8
python data/prepare_targets.py tyk2 cdk2       # selected
```

## Quick start

```bash
# one edge, short pipeline check (NOT converged)
python run_rbfe_series.py \
    --pdb data/cdk2/protein.pdb --ligands data/cdk2/ligands.sdf \
    --edges data/cdk2/edges.yml --exp data/cdk2/ligands.yml \
    --edge lig_21:lig_22 --sim-time-ns 0.1 --out outputs/cdk2

# the whole map at the default 5 ns/leg (complex ~98k atoms; run on a GPU/cluster)
python run_rbfe_series.py \
    --pdb data/cdk2/protein.pdb --ligands data/cdk2/ligands.sdf \
    --edges data/cdk2/edges.yml --exp data/cdk2/ligands.yml \
    --out outputs/cdk2
```

## Tuning: simulation time and λ-dynamics

| flag | default | meaning |
|---|---|---|
| `--sim-time-ns` | **5.0** | MD time **per leg** (complex and solvent each) |
| `--lambda-ts` | **2000** | d-AFED extended-system temperature T_s (K) |
| `--lambda-mass` | **200** | d-AFED fictitious λ mass |
| `--n-blocks` | — | explicit override of d-AFED blocks/leg |

5 ns/leg = 52 083 blocks (12 × 8 fs MD per block). Lower T_s / heavier λ-mass make
τ diffuse more slowly (gentler, more adiabatic); the defaults are the validated
hydration/RBFE values.

## Cluster submission (SLURM GPU queue)

`submit_slurm.py` emits one **job array per target** (one array task per edge, 1
GPU each), auto-detects a GPU partition from `sinfo`, and submits with `sbatch`.
On a machine without SLURM it still writes all the scripts and tells you how to
submit them on the cluster — so you can prepare everything from a laptop.

```bash
# auto-detect + submit the whole benchmark at 5 ns/leg
python submit_slurm.py --sim-time-ns 5 --conda-env openmm_dynlambda

# selected targets, explicit partition/account, cap concurrency, just write (no submit)
python submit_slurm.py --targets tyk2 cdk2 --partition gpu --account myacct \
    --time 24:00:00 --max-array 20 --dry-run
```

Scripts land in `outputs/slurm_jobs/` (`<target>.sbatch` + `<target>_edges.txt` +
`logs/`). Submit later with: `for f in outputs/slurm_jobs/*.sbatch; do sbatch "$f"; done`.

Outputs (under `--out`):
- `rbfe_results.csv` — `ligand_a, ligand_b, ddG_calc_kcal, ddG_exp_kcal, error_kcal`
- `<A>__<B>/{complex,solvent}/lambda_vs_time.csv` — raw τ(t) per leg
  (`time_ps, tau, dUdtau_kJ_per_mol`, one row/d-AFED block), auto-written
- `<A>__<B>/{complex,solvent}/two_region_dafed.npz` — same τ(t) + PMF + ddG (NumPy)
- plot any edge’s λ(τ)-vs-time:
  `python ../scripts/plot_rbfe_lambda_time.py outputs/cdk2/lig_21__lig_22`

## Method / settings

- **Sampling:** d-AFED — one master coordinate τ drives the A→B mutation as an
  adiabatic, high-temperature (T_s = 2000 K) extended variable; the free energy
  comes from the reweighted τ histogram (basins τ≤0.2 = A, τ≥0.8 = B). No adaptive
  bias; a double-well barrier separates the end states.
- **Integrator (default):** HMR-MTS **“4/4/8”** — PME reciprocal space on an 8 fs
  outer step, bonded + direct-space + softcore on a 4 fs inner step, hydrogen-mass
  repartitioning (3 amu) + rigid water. Robust and fast on large solvated complexes
  (~3.4 ns/hour for a 37k-atom box; ~1.3 ns/hour at CDK2’s ~98k atoms). The
  large-timestep SIN(R) integrator is intentionally **not** used here — it is
  fragile on big flexible-water complexes.
- **Force field:** protein Amber **ff14SB**, ligands **OpenFF 2.1.0 (Sage)** +
  AM1-BCC, **TIP3P** water, PME, 1.0 nm cutoff.
- **Placement:** because the series is pre-aligned, each ligand is used at its SDF
  pose directly (no docking/alignment); A-unique atoms are placed at their SDF
  coordinates.

## Scope and limitations (engine v1)

- Neutral, congeneric pairs with **no net-charge change**; the mapping must define
  a clean **core + unique** partition (atom add/delete or single R-group swap).
- **Element changes are handled** (default `break_element_changes=True`): any
  mapped pair whose elements differ (e.g. an H mapped 1:1 to a Br) is automatically
  **un-mapped** so the A atom decouples while the B atom grows — the standard
  softcore treatment of a substituent swap. (The fixed core itself still cannot
  *morph* a core atom’s vdW/bond, so we route element changes through the
  difference region instead. A pair must stay neutral overall, which holds for
  halogen/H scans.) For maximal overlap on big element changes you would instead
  interpolate the core σ/ε (NonbondedForce offsets) and bonded terms (Custom*Force
  with λ) — heavier, not needed for halogen scans.
- `--mode smoke` is a short, **un-converged** pipeline check. Even `converged`
  (~1.2 ns/leg) is limited by **slow protein/pocket relaxation** around the growing
  group, not by τ-sampling — expect real campaigns to need much longer complex
  sampling and/or enhanced sampling of pocket side chains.

## Files

| file | purpose |
|---|---|
| `run_rbfe_series.py` | the driver (parse inputs → loop edges → RBFE → CSV) |
| `submit_slurm.py` | emit + submit SLURM GPU job-arrays (one task/edge) |
| `data/prepare_targets.py` | download + prepare the 8 targets into the layout |
| `data/<target>/` | protein.pdb, ligands.sdf, edges.yml, ligands.yml |
| `../src/dynlambda/mapped_single_topology.py` | hybrid build from SDF mols + atom map |
| `../src/dynlambda/complex_setup.py` | ligand-first solvated protein complex |
| `../src/dynlambda/relative_binding.py` | `relative_binding_mapped` (complex − solvent) |
