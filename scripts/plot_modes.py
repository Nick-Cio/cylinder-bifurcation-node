"""
One directory per Reynolds number, one figure per ROM coordinate.

    plots/
      Re030/  a1.png a2.png a3.png a4.png  _overview.png
      Re036/  ...
      ...
      _bifurcation.png          summary across the whole sweep

Each per-coordinate figure has two panels.  The upper one is the coordinate
against time for both trajectories -- the outward run that starts at the fixed
point and the inward run that starts outside the limit cycle.  The lower one is
the envelope on a logarithmic axis, and that is the panel to read for stability:

    straight line sloping up    exponential growth   (Re above onset)
    straight line sloping down  exponential decay    (Re below onset)
    flat plateau                the limit cycle

The slope of the straight part IS the growth rate.  Reading stability off the
raw time series is much harder; on a log envelope it is immediate, and the
change of sign as Re crosses the bifurcation is visible by flicking between
directories.

Note the envelope of the outward run is NOT monotonic above onset: the initial
kick is a localised blob that projects weakly onto the global wake mode, so it
first decays as it advects away, reaches a minimum, and only then grows
exponentially. The dip is physical, not a numerical artefact -- and it is the
reason a growth rate must be fitted after the minimum, never across it.

Run:  python -u plot_modes.py --data data --out plots --modes 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_C, IN_C = "#c0392b", "#2471a3"


def envelope(a):
    a = np.asarray(a, float)
    if np.ptp(a) < 1e-30:
        return np.full_like(a, 1e-30)
    return np.abs(hilbert(a - a.mean()))


def project(d, Phi, which):
    n = len(d[f"t_{which}"])
    X = np.concatenate([d[f"{which}_ux"].reshape(n, -1),
                        d[f"{which}_uy"].reshape(n, -1)], axis=1)
    return X @ Phi.T, d[f"t_{which}"]


def classify(A, dt, r):
    """Label each pooled-basis coordinate as the shift mode or an oscillation.

    The character of a coordinate is a property of the shared basis, not of any
    one case, so this is done once on the most strongly supercritical case,
    where the limit cycle is cleanest, and the labels are reused everywhere.

    Mode 1 is the shift mode here.  The snapshots are perturbations about the
    base flow and the SVD is deliberately NOT mean-centred -- centring would
    move the origin off the fixed point, which the whole design depends on.  But
    on the limit cycle the time-mean of that perturbation IS the mean-flow
    deformation, large and common to every Reynolds number, so an un-centred SVD
    ranks it first.  It is slow, not oscillatory, and its amplitude measures how
    far the state has moved onto the attractor.
    """
    S = A[len(A) // 2:]
    freqs, ptps = [], []
    for i in range(r):
        s = S[:, i] - S[:, i].mean()
        if np.ptp(s) < 1e-12:
            freqs.append(0.0); ptps.append(0.0); continue
        Y = np.abs(np.fft.rfft(s * np.hanning(len(s))))
        f = np.fft.rfftfreq(len(s), dt)
        freqs.append(float(f[np.argmax(Y[1:]) + 1])); ptps.append(float(np.ptp(s)))
    f0 = freqs[int(np.argmax(ptps))]

    out = []
    for i in range(r):
        if f0 <= 0 or freqs[i] < 0.3 * f0:
            out.append(("shift", "shift mode — mean-flow deformation (slow)"))
        else:
            h = int(round(freqs[i] / f0))
            name = {1: "fundamental", 2: "2nd harmonic", 3: "3rd harmonic"}.get(
                h, f"harmonic {h}")
            out.append(("osc", f"oscillatory, {name}  (f = {freqs[i]:.4f} U/D)"))
    return out, [i for i, (k, _) in enumerate(out) if k == "osc"]


def stability_label(t, a):
    """Describe what the envelope does, without over-claiming a rate."""
    e = envelope(a)
    n = len(e)
    if n < 20:
        return "too short to judge"
    tail = np.median(e[int(0.9 * n):])
    i0 = int(np.argmin(e[: n // 2]))
    floor = max(e[i0], 1e-30)
    if tail > 3 * floor:
        return f"grows to a limit cycle  (x{tail / floor:.0f} from the minimum)"
    head = np.median(e[: max(3, n // 20)])
    if tail < 0.5 * head:
        return f"decays  (x{head / max(tail, 1e-30):.0f} over the run)"
    return "approximately steady"


def per_mode_figure(i, t_o, a_o, t_i, a_i, Re, energy, path, kind=""):
    fig, ax = plt.subplots(2, 1, figsize=(8.4, 5.6),
                           gridspec_kw=dict(height_ratios=[1, 1], hspace=0.34))

    ax[0].plot(t_o, a_o, lw=0.8, color=OUT_C,
               label="from the fixed point" if a_i is not None else None)
    if a_i is not None:
        ax[0].plot(t_i, a_i, lw=0.7, color=IN_C, label="from outside the cycle")
    ax[0].axhline(0, color="0.75", lw=0.7)
    ax[0].set_ylabel(f"$a_{{{i}}}$")
    ax[0].set_xlabel("t   [convective times $D/U$]")
    if a_i is not None:
        ax[0].legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ttl = f"Re = {Re:g}   —   POD coordinate $a_{{{i}}}$"
    if energy is not None:
        ttl += f"   ({100 * energy:.2f}% of sweep energy)"
    ax[0].set_title(ttl + (f"\n{kind}" if kind else ""), fontsize=10.5)

    ax[1].semilogy(t_o, np.maximum(envelope(a_o), 1e-14), lw=0.9, color=OUT_C)
    if a_i is not None:
        ax[1].semilogy(t_i, np.maximum(envelope(a_i), 1e-14), lw=0.9, color=IN_C)
    ax[1].set_ylabel("envelope  $|a_i|$")
    ax[1].set_xlabel("t   [convective times $D/U$]")
    ax[1].set_title("log envelope — straight up = growth, straight down = decay, "
                    "flat = limit cycle", fontsize=9)
    ax[1].grid(True, which="both", alpha=0.25, lw=0.5)

    fig.text(0.5, -0.015, f"outward run: {stability_label(t_o, a_o)}",
             ha="center", fontsize=8.5, color="#444")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def overview_figure(A_o, t_o, A_i, t_i, Re, r, path):
    ncol = 2
    nrow = int(np.ceil(r / ncol)) + 1
    fig = plt.figure(figsize=(10.5, 2.5 * nrow))
    gs = fig.add_gridspec(nrow, ncol, hspace=0.62, wspace=0.26)

    for i in range(r):
        a = fig.add_subplot(gs[i // ncol, i % ncol])
        a.plot(t_o, A_o[:, i], lw=0.7, color=OUT_C)
        if A_i is not None:
            a.plot(t_i, A_i[:, i], lw=0.6, color=IN_C)
        a.axhline(0, color="0.8", lw=0.6)
        a.set_title(f"$a_{{{i+1}}}$", fontsize=9)
        a.tick_params(labelsize=7)
        if i // ncol == nrow - 2:
            a.set_xlabel("t  [D/U]", fontsize=8)

    ph = fig.add_subplot(gs[nrow - 1, 0])
    ph.plot(A_o[:, 0], A_o[:, 1], lw=0.5, color=OUT_C)
    if A_i is not None:
        ph.plot(A_i[:, 0], A_i[:, 1], lw=0.5, color=IN_C)
    ph.set_xlabel("$a_1$", fontsize=8), ph.set_ylabel("$a_2$", fontsize=8)
    ph.set_aspect("equal")
    ph.tick_params(labelsize=7)
    ph.set_title("phase portrait $(a_1, a_2)$", fontsize=9)

    en = fig.add_subplot(gs[nrow - 1, 1])
    for i in range(r):
        en.semilogy(t_o, np.maximum(envelope(A_o[:, i]), 1e-14), lw=0.8,
                    label=f"$a_{{{i+1}}}$")
    en.set_xlabel("t  [D/U]", fontsize=8)
    en.set_title("envelopes, outward run", fontsize=9)
    en.legend(fontsize=6.5, ncol=2)
    en.grid(True, which="both", alpha=0.25, lw=0.5)
    en.tick_params(labelsize=7)

    fig.suptitle(f"Re = {Re:g} — all {r} ROM coordinates", fontsize=12, y=0.995)
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("plots"))
    p.add_argument("--modes", type=int, default=4)
    p.add_argument("--which", choices=("out", "in", "both"), default="out",
                   help="which trajectory to plot; 'out' is THE signal -- it "
                        "starts at the fixed point and runs to the attractor")
    a = p.parse_args()

    basis_path = a.data / "pod_basis.npz"
    if not basis_path.exists():
        raise SystemExit(f"{basis_path} not found — run project.py first")
    b = np.load(basis_path)
    Phi, energy = b["modes"][: a.modes], b["energy"]
    print(f"basis: {Phi.shape[0]} modes of {b['modes'].shape[0]} stored; "
          f"r={a.modes} captures {100 * energy[a.modes - 1]:.2f}% of sweep energy")

    files = sorted(a.data.glob("Re*.npz"), key=lambda f: float(f.stem[2:]))
    files = [f for f in files if f.suffix == ".npz" and f.stem[2:].replace(
        ".", "").isdigit()]
    a.out.mkdir(parents=True, exist_ok=True)

    # classify the coordinates once, on the most strongly supercritical case
    dref = np.load(files[-1])
    Aref, tref = project(dref, Phi, "out")
    kinds, osc = classify(Aref, float(np.median(np.diff(tref))), a.modes)
    print("coordinate character (shared basis, so the same at every Re):")
    for i, (_, lab) in enumerate(kinds):
        print(f"  a{i+1}: {lab}")
    if len(osc) < 2:
        raise SystemExit("no oscillating pair within the requested modes — "
                         "increase --modes")
    p1, p2 = osc[0], osc[1]
    print(f"using the leading oscillating pair (a{p1+1}, a{p2+1}) for amplitude\n")

    summary = []
    for f in files:
        d = np.load(f)
        Re = float(d["Re"])
        A_o, t_o = project(d, Phi, "out")
        A_i, t_i = project(d, Phi, "in")
        if a.which == "out":
            A_i, t_i = None, None

        sub = a.out / f"Re{Re:05.1f}".replace(".", "p")
        sub.mkdir(exist_ok=True)
        for i in range(a.modes):
            per_mode_figure(i + 1, t_o, A_o[:, i], t_i,
                            None if A_i is None else A_i[:, i], Re,
                            float(energy[i]) if i == 0 else
                            float(energy[i] - energy[i - 1]),
                            sub / f"a{i+1}.png", kind=kinds[i][1])
        overview_figure(A_o, t_o, A_i, t_i, Re, a.modes, sub / "_overview.png")

        # Oscillation amplitude = norm of the FLUCTUATING part of the whole
        # retained state, not the radius in the leading pair.  The spatial shape
        # of the oscillation deforms with Re, so its energy leaks out of any
        # fixed pair: measured on the leading pair alone the amplitude peaks
        # near Re = 100 and then falls, while the true amplitude keeps rising.
        # Summing over the retained modes is monotone from r = 8 upwards.
        tail = A_o[len(A_o) * 3 // 4:]
        fluc = tail - tail.mean(0)
        amp = float(np.sqrt((fluc ** 2).sum(1)).mean())
        summary.append((Re, amp, stability_label(t_o, A_o[:, p1])))
        print(f"  {sub.name}/  {a.modes} coordinate plots + overview   "
              f"fluct. norm = {amp:.4f}   {summary[-1][2]}")

    # ---- sweep-level summary -------------------------------------------
    R = np.array([s[0] for s in summary])
    M = np.array([s[1] for s in summary])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(R, M, "o-", color=OUT_C)
    ax[0].plot(R, -M, "o-", color=OUT_C)
    ax[0].axhline(0, color="0.75", lw=0.8)
    ax[0].set_xlabel("Re"), ax[0].set_ylabel("fluctuation norm on the attractor")
    ax[0].set_title(f"bifurcation diagram in ROM coordinates (r = {a.modes})",
                    fontsize=10)
    on = M > 1e-3
    if on.sum() >= 3:
        k, c = np.polyfit(R[on], M[on] ** 2, 1)
        ax[1].plot(R[on], M[on] ** 2, "o", color="#16a085")
        xx = np.linspace(R[on].min(), R[on].max(), 40)
        ax[1].plot(xx, k * xx + c, "--", color="0.45", lw=1)
        ax[1].set_title(f"norm$^2$ vs Re — zero crossing at Re = {-c/k:.1f}",
                        fontsize=10)
    ax[1].set_xlabel("Re"), ax[1].set_ylabel("norm$^2$")
    fig.tight_layout()
    fig.savefig(a.out / "_bifurcation.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    idx = ["# ROM coordinate plots", "",
           f"One directory per Reynolds number, {a.modes} coordinate plots in each.",
           "Read `aN.png` lower panel for stability: straight up = growth,",
           "straight down = decay, flat = limit cycle.", "",
           "| Re | dir | fluctuation norm | outward run |",
           "|---|---|---|---|"]
    for (Re, amp, lab), f in zip(summary, files):
        idx.append(f"| {Re:g} | `Re{Re:05.1f}`".replace(".", "p", 1) +
                   f" | {amp:.4f} | {lab} |")
    (a.out / "README.md").write_text("\n".join(idx) + "\n")
    print(f"\nwrote {len(files)} directories under {a.out}/ "
          f"+ _bifurcation.png + README.md")


if __name__ == "__main__":
    main()
