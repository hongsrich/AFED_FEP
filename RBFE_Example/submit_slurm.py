#!/usr/bin/env python
"""Submit the RBFE series to a SLURM GPU queue (one array task per edge).

For each target it writes (under <out>/slurm_jobs/):
  * <target>_edges.txt   -- one "ligandA:ligandB" per line (the array index map),
  * <target>.sbatch      -- a SLURM array job (1 GPU/task) that runs
        run_rbfe_series.py --edge $EDGE ... for the array index's edge.

Behaviour:
  * If `sbatch` is on PATH, it auto-detects a GPU partition (from `sinfo`, unless
    --partition is given) and SUBMITS each target's array job (unless --dry-run).
  * If SLURM is NOT present (e.g. a laptop), it still writes all the scripts and
    prints exactly how to submit them on a cluster -- nothing is lost.

Examples:
  # whole benchmark, 5 ns/leg, default d-AFED T_s/mass; auto-detect everything
  python submit_slurm.py --sim-time-ns 5 --conda-env openmm_dynlambda

  # just two targets, custom lambda settings, force a partition
  python submit_slurm.py --targets tyk2 cdk2 --lambda-ts 2000 --lambda-mass 200 \
      --partition gpu --time 24:00:00
"""
import argparse
import os
import shutil
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TARGETS = ["cdk2", "mcl1", "p38", "ptp1b", "thrombin", "tyk2",
                   "bace", "jnk1"]


def detect_slurm():
    return shutil.which("sbatch") is not None and shutil.which("sinfo") is not None


def detect_gpu_partition():
    """First partition that advertises a GPU GRES (else one named '*gpu*')."""
    try:
        out = subprocess.run(["sinfo", "-h", "-o", "%P %G"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    named = None
    for line in out.splitlines():
        parts = line.split(None, 1)
        if not parts:
            continue
        part = parts[0].rstrip("*")
        gres = parts[1] if len(parts) > 1 else ""
        if "gpu" in gres.lower():
            return part
        if "gpu" in part.lower() and named is None:
            named = part
    return named


def runnable_edges(target, data_dir):
    """List of 'A:B' edges for a target (both ligands present is checked at run)."""
    epath = os.path.join(data_dir, target, "edges.yml")
    doc = yaml.safe_load(open(epath))
    return [f"{e['ligand_a']}:{e['ligand_b']}" for e in doc["edges"].values()]


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=rbfe_{target}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
#SBATCH --array=0-{last}{maxarray}
#SBATCH --output={logdir}/{target}_%a.out
{account}
set -euo pipefail
cd {rundir}

# activate the environment that has OpenMM + dynlambda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate {conda_env}

EDGE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {edges_file})
echo "task $SLURM_ARRAY_TASK_ID -> edge $EDGE on $(hostname) gpu ${{CUDA_VISIBLE_DEVICES:-?}}"

python run_rbfe_series.py \\
    --pdb data/{target}/protein.pdb \\
    --ligands data/{target}/ligands.sdf \\
    --edges data/{target}/edges.yml \\
    --exp data/{target}/ligands.yml \\
    --edge "$EDGE" \\
    --sim-time-ns {sim_time_ns} --lambda-ts {lambda_ts} --lambda-mass {lambda_mass} \\
    --out {out}/{target}
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--sim-time-ns", type=float, default=5.0)
    ap.add_argument("--lambda-ts", type=float, default=2000.0)
    ap.add_argument("--lambda-mass", type=float, default=200.0)
    ap.add_argument("--partition", default=None, help="GPU partition (else auto)")
    ap.add_argument("--account", default=None, help="SLURM account (optional)")
    ap.add_argument("--time", default="24:00:00", help="walltime per edge")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--max-array", type=int, default=None,
                    help="cap simultaneous array tasks, e.g. 20 -> %%20")
    ap.add_argument("--conda-env", default="openmm_dynlambda")
    ap.add_argument("--dry-run", action="store_true",
                    help="write scripts but do not call sbatch")
    args = ap.parse_args()

    have_slurm = detect_slurm()
    partition = args.partition or (detect_gpu_partition() if have_slurm else None) \
        or "gpu"
    if have_slurm:
        print(f"SLURM detected. GPU partition: {partition}"
              + (" (auto-detected)" if not args.partition else ""))
    else:
        print("SLURM NOT detected (no sbatch/sinfo) -- writing job scripts only.")
        print(f"  using placeholder partition '{partition}' "
              "(set --partition for your cluster).")

    jobdir = os.path.join(HERE, args.out, "slurm_jobs")
    logdir = os.path.join(jobdir, "logs")
    os.makedirs(logdir, exist_ok=True)
    account = f"#SBATCH --account={args.account}" if args.account else ""

    written = []
    for target in args.targets:
        if not os.path.exists(os.path.join(args.data_dir, target, "edges.yml")):
            print(f"  {target}: no data/{target}/edges.yml -- skipping")
            continue
        edges = runnable_edges(target, args.data_dir)
        edges_file = os.path.join(jobdir, f"{target}_edges.txt")
        with open(edges_file, "w") as fh:
            fh.write("\n".join(edges) + "\n")
        sbatch = SBATCH_TEMPLATE.format(
            target=target, partition=partition, gpus=args.gpus, cpus=args.cpus,
            time=args.time, last=len(edges) - 1,
            maxarray=(f"%{args.max_array}" if args.max_array else ""),
            logdir=logdir, account=account, rundir=HERE,
            conda_env=args.conda_env, edges_file=edges_file,
            sim_time_ns=args.sim_time_ns, lambda_ts=args.lambda_ts,
            lambda_mass=args.lambda_mass, out=args.out)
        spath = os.path.join(jobdir, f"{target}.sbatch")
        with open(spath, "w") as fh:
            fh.write(sbatch)
        written.append((target, spath, len(edges)))
        print(f"  {target}: {len(edges)} edges -> {os.path.relpath(spath, HERE)}")

    total = sum(n for _, _, n in written)
    print(f"\n{len(written)} target(s), {total} edges (= array tasks) total.")

    if have_slurm and not args.dry_run:
        print("submitting...")
        for target, spath, _ in written:
            r = subprocess.run(["sbatch", spath], capture_output=True, text=True)
            print(f"  {target}: {r.stdout.strip() or r.stderr.strip()}")
    else:
        why = "dry-run" if (have_slurm and args.dry_run) else "no SLURM here"
        print(f"not submitting ({why}). On the cluster, submit with:")
        print(f"    for f in {os.path.relpath(jobdir, HERE)}/*.sbatch; do sbatch \"$f\"; done")


if __name__ == "__main__":
    main()
