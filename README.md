# dynamic_lambda_hfe

A from-scratch **OpenMM** toolkit for alchemical free energies with a **dynamic-λ
(d-AFED)** sampler: one master coordinate τ drives the alchemical change as an
adiabatic, high-temperature extended variable, and the free energy is recovered by
reweighting the τ histogram. Runs on CUDA, OpenCL (Apple Silicon), or CPU.

**Capabilities**
- **Absolute hydration free energy (HFE)** — decouple a solute from TIP3P water;
  estimate ΔG with fixed-window **MBAR** or **d-AFED**.
- **Relative hydration** — hybrid **single-topology** A→B (shared core stays
  coupled; only the R-group difference is mutated), `solvent − vacuum`.
- **Relative binding (RBFE)** — the same single-topology engine as
  `complex − solvent`, for protein–ligand systems.
- **Large-timestep integrators** — HMR + 3-scale MTS ("4/4/8"), and a SIN(R) /
  regulated-dynamics integrator (Abreu–Tuckerman).
- **Benchmark harness** — run a whole perturbation map (FEP+ JACS sets) and submit
  it to a SLURM GPU queue: see [`RBFE_Example/`](RBFE_Example/).

> Research / methods code. The d-AFED estimator is under active development; treat
> absolute numbers as method-development output, not production free energies.

## Install

```bash
conda env create -f environment.yml
conda activate openmm_dynlambda
pytest -q -m "not slow"          # quick check of the built-in test suite
```

## Quick start

```bash
# absolute hydration free energy of a molecule (fixed-window MBAR)
python scripts/02_run_fixed_lambda_hfe.py --config examples/methane.yaml

# relative binding free energy over a perturbation-map edge (CDK2 FEP+ set)
cd RBFE_Example
python data/prepare_targets.py cdk2          # fetch the benchmark inputs once
python run_rbfe_series.py \
    --pdb data/cdk2/protein.pdb --ligands data/cdk2/ligands.sdf \
    --edges data/cdk2/edges.yml --exp data/cdk2/ligands.yml \
    --edge 17:1oiu --sim-time-ns 5 --out outputs/cdk2
```

See [`RBFE_Example/README.md`](RBFE_Example/README.md) for the full RBFE workflow
(8 bundled FEP+ JACS targets, λ T_s / mass knobs, and SLURM auto-submission).

## Layout

```
src/dynlambda/   core library (alchemy, MBAR, d-AFED, single-topology, integrators)
scripts/         numbered CLI workflows (HFE, relative hydration, RBFE)
examples/        example YAML configs
RBFE_Example/    RBFE benchmark harness + FEP+ JACS data + SLURM submission
tests/           pytest suite  (pytest -q;  add -m "not slow" to skip MD)
```

## Method notes

- **Sign convention:** ΔG_hyd = −ΔG_decouple; electrostatics are switched off
  before sterics (softcore) to avoid the LJ/Coulomb singularity.
- **RBFE cycle:** ddG_bind(A→B) = dG_transform(complex) − dG_transform(solvent);
  the shared scaffold cancels, so only the R-group difference is sampled.
- **Force field:** protein Amber **ff14SB**, ligands **OpenFF 2.x** + AM1-BCC,
  **TIP3P** water, PME.

## Data & references

Benchmark inputs are downloaded by `RBFE_Example/data/prepare_targets.py` from the
[OpenFE benchmarks](https://github.com/OpenFreeEnergy/openfe-benchmarks) (FEP+ JACS
set; Wang et al. *JACS* 2015, `10.1021/ja512751q`). The SIN(R) integrator follows
Abreu & Tuckerman (*Eur. Phys. J. B* 2021, 94, 231).
