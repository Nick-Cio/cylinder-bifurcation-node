"""
Does the generated sweep actually contain a clean Hopf, and how wide is the ROM?

Produces, from the raw sweep and nothing else:

  sigma(Re)   linear growth rate of the wake mode.  Below onset it is measured
              from the tail of the large-kick run (a perturbation decaying back
              to the fixed point decays at the linear rate); above onset from
              the early phase of the small-kick run, before saturation.  Where
              sigma crosses zero IS the bifurcation point, and because the two
              branches come from independent runs the crossing is not fitted in.
  A(Re)       saturated limit-cycle amplitude.  A^2 linear in Re is the
              signature of a supercritical Hopf; the intercept must agree with
              the sigma = 0 crossing or something is wrong.
  St(Re)      shedding frequency.
  POD ranks   per-Re and pooled-across-Re, which is the number that decides how
              wide the latent state of a parametric NODE has to be.

Run:  python -u analyse.py --data data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def envelope(y):
    return np.abs(hilbert(y - y.mean()))


def fit_rate(t, y, lo, hi, min_pts=6):
    """Slope of log|envelope| over the window where it lies in (lo, hi).

    min_pts is deliberately small: far above onset the mode grows so fast that
    it crosses any sensible fit window in a couple of convective times, giving
    only a handful of samples.  Those points matter least -- the Re_c estimate
    comes from the near-onset cases, where growth is slow and samples are
    plentiful -- so it is better to return a noisy number there than a NaN.
    """
    e = envelope(y)
    m = (e > lo) & (e < hi) & np.isfinite(e)
    if m.sum() < min_pts:
        return np.nan
    return float(np.polyfit(t[m], np.log(e[m]), 1)[0])


def strouhal(t, y):
    if len(t) < 32:
        return np.nan
    s = y - y.mean()
    Y = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    f = np.fft.rfftfreq(len(s), t[1] - t[0])     # cycles per convective time
    return float(f[np.argmax(Y[1:]) + 1])


def rank_for(s, fracs=(0.99, 0.999, 0.9999)):
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    return [int(np.searchsorted(e, f) + 1) for f in fracs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--fig", type=Path, default=Path("fig_sweep.png"))
    a = p.parse_args()

    files = sorted(a.data.glob("Re*.npz"), key=lambda f: float(f.stem[2:]))
    if not files:
        raise SystemExit(f"no Re*.npz in {a.data}")

    rows, blocks = [], []
    print(f"{'Re':>6} {'A/U':>10} {'sigma':>9} {'St':>7} "
          f"{'r99':>5} {'r99.9':>6} {'snaps':>6}")
    for f in files:
        d = np.load(f)
        Re, U = float(d["Re"]), float(d["u_in"])
        to, po = d["t_out"], d["probe_out"] / U
        ti, pi = d["t_in"], d["probe_in"] / U

        A = 0.5 * np.ptp(po[len(po) * 3 // 4:])
        A_in = 0.5 * np.ptp(pi[len(pi) * 3 // 4:])
        A = max(A, A_in)
        unstable = A > 3e-3

        if unstable:                       # growth, before saturation
            sigma = fit_rate(to, po, 1.5 * envelope(po)[:3].mean(), 0.45 * A)
            st = strouhal(to[len(to) // 2:], po[len(po) // 2:])
        else:                              # decay of the large kick
            sigma = fit_rate(ti, pi, 1e-4, 0.25 * envelope(pi).max())
            st = strouhal(ti[: len(ti) // 3], pi[: len(ti) // 3])

        X = np.concatenate([
            np.concatenate([d["out_ux"].reshape(len(to), -1),
                            d["out_uy"].reshape(len(to), -1)], 1),
            np.concatenate([d["in_ux"].reshape(len(ti), -1),
                            d["in_uy"].reshape(len(ti), -1)], 1)])
        s = np.linalg.svd(X, compute_uv=False)
        r99, r999, _ = rank_for(s)
        n = np.linalg.norm(X)
        blocks.append(X / n if n > 0 else X)

        rows.append((Re, A, sigma, st, r99, r999))
        print(f"{Re:>6g} {A:>10.4e} {sigma:>+9.4f} {st:>7.4f} "
              f"{r99:>5d} {r999:>6d} {len(to)+len(ti):>6d}")

    R = np.array([r[0] for r in rows])
    A = np.array([r[1] for r in rows])
    S = np.array([r[2] for r in rows])
    T = np.array([r[3] for r in rows])

    # ---- Re_c from the sigma = 0 crossing (linear near onset) -----------
    near = np.isfinite(S) & (np.abs(S) < 0.12)
    Re_c = slope = np.nan
    if near.sum() >= 3:
        slope, c0 = np.polyfit(R[near], S[near], 1)
        Re_c = -c0 / slope
        print(f"\nsigma = 0 at Re_c = {Re_c:.2f}   (dsigma/dRe = {slope:.5f})")

    on = A > 3e-3
    if on.sum() >= 3:
        k, c1 = np.polyfit(R[on], A[on] ** 2, 1)
        print(f"A^2 extrapolates to zero at Re = {-c1 / k:.2f}  "
              f"-> {'supercritical' if abs(-c1/k - Re_c) < 6 else 'CHECK: mismatch'}")

    # ---- pooled POD -----------------------------------------------------
    P = np.concatenate(blocks, 0)
    sp = np.linalg.svd(P, compute_uv=False)
    r99, r999, r9999 = rank_for(sp)
    print(f"\npooled snapshot matrix {P.shape[0]} x {P.shape[1]}")
    print(f"pooled POD rank:  99% -> {r99}    99.9% -> {r999}    "
          f"99.99% -> {r9999}")
    print(f"per-Re median:    99% -> {int(np.median([r[4] for r in rows]))}"
          f"    99.9% -> {int(np.median([r[5] for r in rows]))}")
    np.save(a.data / "pooled_sv.npy", sp[:120])

    # ---- figure ---------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.subplots_adjust(hspace=0.32, wspace=0.28)

    ax[0, 0].plot(R, A, "o-", color="#c0392b")
    ax[0, 0].plot(R, -A, "o-", color="#c0392b")
    ax[0, 0].axhline(0, color="0.7", lw=0.8)
    if np.isfinite(Re_c):
        ax[0, 0].axvline(Re_c, ls="--", color="#2980b9",
                         label=f"$Re_c$ = {Re_c:.1f}")
        ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_xlabel("Re"), ax[0, 0].set_ylabel("LCO amplitude  $v'/U$")
    ax[0, 0].set_title("bifurcation diagram", fontsize=10)

    ax[0, 1].plot(R, S, "o-", color="#8e44ad")
    ax[0, 1].axhline(0, color="0.7", lw=0.8)
    if np.isfinite(Re_c):
        ax[0, 1].plot(R, slope * (R - Re_c), "--", color="#2980b9", lw=1)
    ax[0, 1].set_xlabel("Re"), ax[0, 1].set_ylabel(r"growth rate $\sigma\,D/U$")
    ax[0, 1].set_title("linear growth rate (decay below, growth above)",
                       fontsize=10)

    if on.sum() >= 3:
        ax[1, 0].plot(R[on], A[on] ** 2, "o", color="#16a085")
        xx = np.linspace(R[on].min(), R[on].max(), 50)
        ax[1, 0].plot(xx, k * xx + c1, "--", color="0.4", lw=1,
                      label="Stuart-Landau  $A^2\\propto Re-Re_c$")
        ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_xlabel("Re"), ax[1, 0].set_ylabel("$A^2$")
    ax[1, 0].set_title("supercritical scaling test", fontsize=10)

    axb = ax[1, 1]
    e = np.cumsum(sp ** 2) / np.sum(sp ** 2)
    axb.semilogy(np.arange(1, min(60, len(sp)) + 1),
                 (1 - e)[:60] + 1e-16, "o-", ms=3, color="#d35400")
    for lvl, lab in ((1e-2, "99%"), (1e-3, "99.9%")):
        axb.axhline(lvl, ls=":", color="0.6")
        axb.text(52, lvl * 1.4, lab, fontsize=7, color="0.4")
    axb.set_xlabel("POD modes retained")
    axb.set_ylabel("unresolved energy fraction")
    axb.set_title(f"pooled basis: r99 = {r99}, r99.9 = {r999}", fontsize=10)

    fig.savefig(a.fig, dpi=140, bbox_inches="tight")
    print(f"\nwrote {a.fig}")


if __name__ == "__main__":
    main()
