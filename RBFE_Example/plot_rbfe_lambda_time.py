#!/usr/bin/env python
"""Plot the d-AFED master coordinate tau(t) for the two RBFE legs (complex, solvent).

Reads the two_region_dafed.npz saved by each leg under <out>/complex and
<out>/solvent and draws tau vs time, with the lambda=0 (A) and lambda=1 (B) basins
shaded. Usage: python scripts/plot_rbfe_lambda_time.py outputs/rbfe_t4_mts448
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "outputs/rbfe_t4_mts448"
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, leg, color in zip(axes, ("complex", "solvent"), ("#1f77b4", "#2ca02c")):
        npz = os.path.join(out, leg, "two_region_dafed.npz")
        if not os.path.exists(npz):
            ax.text(0.5, 0.5, f"missing {npz}", ha="center", va="center")
            continue
        d = np.load(npz)
        t = d["times_ps"] / 1000.0          # ns
        tau = d["tau"]
        frac0 = float(np.mean(tau <= 0.2)) * 100
        frac1 = float(np.mean(tau >= 0.8)) * 100
        ax.axhspan(0.0, 0.2, color="grey", alpha=0.15)
        ax.axhspan(0.8, 1.0, color="grey", alpha=0.15)
        ax.plot(t, tau, lw=0.4, color=color)
        ax.set_ylim(-0.55, 1.55)
        ax.set_ylabel("tau (master lambda)")
        ax.set_title(f"{leg} leg  -  basin A(tau<=0.2): {frac0:.0f}%   "
                     f"basin B(tau>=0.8): {frac1:.0f}%")
    axes[-1].set_xlabel("time (ns)")
    fig.suptitle(f"RBFE d-AFED tau(t) per leg: {os.path.basename(os.path.normpath(out))}")
    fig.tight_layout()
    png = os.path.join(out, "lambda_vs_time.png")
    fig.savefig(png, dpi=130)
    print("wrote", png)


if __name__ == "__main__":
    main()
