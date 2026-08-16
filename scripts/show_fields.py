"""
Eyeball check: base flow should be a symmetric recirculation bubble, the
saturated state a von Karman street.  If these two pictures look right the
solver is doing its job.

Run:  python -u show_fields.py --data data_smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def curl(ux, uy, h=1.0):
    return np.gradient(uy, h, axis=-1) - np.gradient(ux, h, axis=-2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--fig", type=Path, default=Path("fig_fields.png"))
    a = p.parse_args()

    files = sorted(a.data.glob("Re*.npz"), key=lambda f: float(f.stem[2:]))
    fig, ax = plt.subplots(len(files), 3, figsize=(13, 2.9 * len(files)),
                           squeeze=False)

    for r, f in enumerate(files):
        d = np.load(f)
        Re, U, D = float(d["Re"]), float(d["u_in"]), int(d["D"])
        h = 2.0 / D                      # stride 2, in diameters
        bx, by = d["base_ux"], d["base_uy"]
        wb = curl(bx, by, h) / U

        k = len(d["t_out"]) - 1
        fx, fy = bx + d["out_ux"][k], by + d["out_uy"][k]
        wf = curl(fx, fy, h) / U
        pert = curl(d["out_ux"][k], d["out_uy"][k], h) / U

        for cix, (w, lab) in enumerate(
                ((wb, "base flow (steady, symmetric)"),
                 (wf, f"final state, t={d['t_out'][k]:.0f} D/U"),
                 (pert, "perturbation about base flow"))):
            v = max(np.abs(w).max(), 1e-9) * 0.8
            ax[r, cix].imshow(w, cmap="RdBu_r", vmin=-v, vmax=v,
                              origin="lower", aspect="equal")
            ax[r, cix].set_xticks([]), ax[r, cix].set_yticks([])
            ax[r, cix].set_title(f"Re={Re:g}  {lab}", fontsize=8)

    fig.tight_layout()
    fig.savefig(a.fig, dpi=130, bbox_inches="tight")
    print(f"wrote {a.fig}")


if __name__ == "__main__":
    main()
