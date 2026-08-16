"""
Growth rate sigma(Re) and the critical Reynolds number, done properly.

This replaces an earlier envelope-fitting attempt that was wrong in three ways,
each of which is worth knowing about because they are the standard traps.

  1. `hilbert(y - y.mean())`.  After saturation the probe has a large non-zero
     mean (-0.09 U at Re = 160) from mean-flow distortion.  Subtracting the
     record-wide mean injects that DC into the *linear* phase, so the envelope
     starts at 0.1 instead of 4e-4 and the growth phase vanishes entirely.
     One-period complex demodulation annihilates DC and the second harmonic
     exactly while leaving the fundamental untouched.

  2. Selecting the fit window by amplitude threshold.  Above onset the envelope
     is NOT monotonic: the localised kick projects weakly onto the global mode,
     decays as it advects away, and only then grows.  A threshold mask picks up
     points from both phases and the fitted slope is meaningless.

  3. Extrapolating A_sat^2 vs Re to find Re_c.  It does not converge.  On this
     data it gives 44.6, 44.0, 42.4, 39.2, 36.8 as points out to Re = 48, 54,
     62, 73, 80 are included, because dA^2/dRe falls by a factor 6.7 across the
     sweep.  Any number it produces is a coordinate of where you truncated.

The method here fits the saturation instead of hunting for a clean window.  For
a supercritical Hopf the amplitude obeys the Landau equation, whose solution is

    A^2(t) = A_sat^2 / (1 + C exp(-2 sigma (t - t0)))

so one fit over the whole record recovers sigma and A_sat together, with no
window selection at all.  At Re = 45 the fitted sigma is insensitive to the
start time to five significant figures over t0 = 30..100.

Re_c comes from a quadratic fit of sigma(Re) through the bracketing points:
dsigma/dRe changes by 25% over Re = 41..51, so a two-point secant is worth
+-0.25 in Re_c depending which pair you pick.

Run:  python -u growth.py --data data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit, least_squares


def dominant_frequency(y, dt):
    """Shedding frequency, from the running-rms-normalised signal.

    Normalising first matters: the envelope spans four to six decades, so an
    unnormalised spectrum is dominated by whichever end of the record happens
    to be largest rather than by the oscillation.
    """
    y = np.asarray(y, float)
    w = max(8, len(y) // 50)
    k = np.ones(w) / w
    rms = np.sqrt(np.convolve(y ** 2, k, mode="same")) + 1e-30
    s = (y - y.mean()) / rms
    Y = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    f = np.fft.rfftfreq(len(s), dt)
    return float(f[np.argmax(Y[1:]) + 1])


def demodulate(y, t, f0):
    """Amplitude envelope by one-period complex demodulation.

    A(t) = 2 |<y e^{-2 pi i f0 t}>| over a centred window of exactly one
    period.  The one-period width is what kills DC and the even harmonics
    exactly; it also avoids the FFT wrap-around that makes a Hilbert envelope
    turn spuriously upward at the end of a strongly decaying record.
    """
    dt = float(np.median(np.diff(t)))
    w = max(4, int(round(1.0 / (f0 * dt))))
    z = np.asarray(y, float) * np.exp(-2j * np.pi * f0 * np.asarray(t, float))
    k = np.ones(w) / w
    a = (np.convolve(z.real, k, mode="same")
         + 1j * np.convolve(z.imag, k, mode="same"))
    A = 2.0 * np.abs(a)
    A[: w // 2] = np.nan          # windows that overhang the record
    A[-(w // 2 or 1):] = np.nan
    return A


def _landau(t, A_sat, C, sigma, t0):
    return A_sat ** 2 / (1.0 + C * np.exp(-2.0 * sigma * (t - t0)))


def fit_growing(t, A, periods):
    """Global Landau fit; returns sigma, A_sat, relative residual."""
    m = np.isfinite(A) & (A > 0)
    t, A = t[m], A[m]
    if len(t) < 40:
        return np.nan, np.nan, np.nan
    i0 = int(np.argmin(A[: len(A) // 2])) + periods      # one period past the dip
    if i0 >= len(t) - 20:
        i0 = 0
    tt, AA = t[i0:], A[i0:]
    t0 = float(tt[0])
    A_sat0 = float(np.median(AA[-max(5, len(AA) // 20):]))

    # Fit in LOG amplitude, not in A^2.  A least-squares fit to A^2 is dominated
    # by the saturated tail, where every candidate sigma looks alike, and is
    # nearly blind to the early exponential that actually determines sigma --
    # which is why the unweighted version converged to a 40% residual at Re = 45,
    # the one case where the answer matters most.  In log amplitude every decade
    # carries equal weight.  Bounds plus a multi-start over sigma remove the
    # local minima that curve_fit was falling into.
    def resid(p):
        A_sat, C, sigma = p
        model = A_sat ** 2 / (1.0 + C * np.exp(
            -2.0 * sigma * np.clip(tt - t0, -700 / max(2 * sigma, 1e-9),
                                   700 / max(2 * sigma, 1e-9))))
        return np.log(np.maximum(model, 1e-300)) / 2.0 - np.log(AA)

    best = None
    for s0 in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        p0 = [A_sat0, max((A_sat0 / max(AA[0], 1e-12)) ** 2 - 1, 1.0), s0]
        try:
            out = least_squares(resid, p0, bounds=([1e-9, 1e-9, 1e-5],
                                                   [1e3, 1e18, 5.0]),
                                max_nfev=20000)
        except Exception:
            continue
        r = float(np.sqrt(np.mean(out.fun ** 2)))
        if best is None or r < best[0]:
            best = (r, out.x)
    if best is None:
        return np.nan, np.nan, np.nan
    res, (A_sat, _C, sigma) = best
    return abs(float(sigma)), abs(float(A_sat)), float(res)


def fit_decaying(t, A, floor_mult=6.0):
    """Log-linear fit over the clean part of a decaying record."""
    m = np.isfinite(A) & (A > 0)
    t, A = t[m], A[m]
    if len(t) < 30:
        return np.nan, np.nan
    floor = float(np.min(A[-max(5, len(A) // 20):]))
    ok = (t > 0.4 * t[-1]) & (A > floor_mult * floor)
    if ok.sum() < 15:
        ok = (t > 0.3 * t[-1]) & (A > 2 * floor)
    if ok.sum() < 10:
        return np.nan, np.nan
    c = np.polyfit(t[ok], np.log(A[ok]), 1)
    res = float(np.sqrt(np.mean((np.polyval(c, t[ok]) - np.log(A[ok])) ** 2)))
    return float(c[0]), res


def analyse_case(path):
    d = np.load(path)
    Re, U = float(d["Re"]), float(d["u_in"])
    t, y = d["t_out"].astype(float), d["probe_out"].astype(float) / U
    dt = float(np.median(np.diff(t)))
    f0 = dominant_frequency(y, dt)
    period = 1.0 / f0
    A = demodulate(y, t, f0)

    finite = A[np.isfinite(A)]
    A_end = float(np.median(finite[-max(5, len(finite) // 20):]))
    A_min = float(np.nanmin(A[: len(A) // 2]))
    growing = A_end > 3.0 * A_min

    if growing:
        sigma, A_sat, res = fit_growing(t, A, int(round(period / dt)))
        # unresolved if the linear phase is shorter than 3 shedding periods
        i0 = int(np.nanargmin(A[: len(A) // 2]))
        lin = (t[-1] - t[i0]) if not np.isfinite(sigma) else \
            np.log(max(A_sat / max(A[i0], 1e-12), 1.001)) / max(sigma, 1e-9)
        resolved = lin > 3.0 * period
    else:
        sigma, res = fit_decaying(t, A)
        A_sat, resolved = 0.0, (np.isfinite(res) and res < 0.05)

    return dict(Re=Re, f0=f0, St=f0, sigma=sigma, A_sat=A_sat,
                residual=res, growing=growing, resolved=bool(resolved))


def critical_reynolds(rows, half_width=6.0):
    """Weighted quadratic fit of sigma(Re) through the bracketing points."""
    ok = [r for r in rows if np.isfinite(r["sigma"]) and r["resolved"]]
    if len(ok) < 4:
        return np.nan, np.nan, []
    R = np.array([r["Re"] for r in ok])
    S = np.array([r["sigma"] * (1 if r["growing"] else 1) for r in ok])
    S = np.array([r["sigma"] if r["growing"] else -abs(r["sigma"]) for r in ok])

    neg = R[S < 0].max() if (S < 0).any() else R.min()
    pos = R[S > 0].min() if (S > 0).any() else R.max()
    mid = 0.5 * (neg + pos)
    sel = np.abs(R - mid) <= half_width + 0.5 * (pos - neg)
    if sel.sum() < 4:
        sel = np.abs(R - mid) <= 2 * half_width
    c = np.polyfit(R[sel], S[sel], 2)
    roots = np.roots(c)
    roots = [float(x.real) for x in roots
             if abs(x.imag) < 1e-9 and neg - 6 < x.real < pos + 6]
    Re_c = min(roots, key=lambda x: abs(x - mid)) if roots else np.nan
    slope = float(np.polyval(np.polyder(c), Re_c)) if np.isfinite(Re_c) else np.nan
    return Re_c, slope, list(R[sel])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--drop", type=float, nargs="*", default=[30.0],
                   help="Re values to exclude (default 30: two beating damped "
                        "modes and the signal reaches the float32 floor)")
    a = p.parse_args()

    files = sorted(a.data.glob("Re*.npz"), key=lambda f: float(f.stem[2:]))
    files = [f for f in files if f.suffix == ".npz"]
    rows = [analyse_case(f) for f in files]
    rows = [r for r in rows if r["Re"] not in a.drop]

    print(f"{'Re':>5} {'St':>7} {'sigma':>10} {'A_sat':>9} {'resid':>7} "
          f"{'state':>9}  note")
    for r in rows:
        s = r["sigma"] if r["growing"] else -abs(r["sigma"])
        note = "" if r["resolved"] else "UNRESOLVED (linear phase < 3 periods)"
        print(f"{r['Re']:>5g} {r['St']:>7.4f} {s:>+10.5f} {r['A_sat']:>9.4f} "
              f"{r['residual']:>7.4f} {'growing' if r['growing'] else 'decaying':>9}"
              f"  {note}")

    Re_c, slope, used = critical_reynolds(rows)
    print(f"\nsigma = 0 at  Re_c = {Re_c:.2f}"
          f"   dsigma/dRe = {slope:.5f} per unit Re")
    print(f"quadratic fit through Re = {[f'{x:g}' for x in used]}")

    # diagnostics only -- deliberately NOT used for Re_c
    g = [r for r in rows if r["growing"] and r["A_sat"] > 0 and r["resolved"]]
    if len(g) >= 3:
        print("\nLandau coefficient l = sigma / A_sat^2 (diagnostic; a clean "
              "Hopf has this smooth):")
        print("  " + "  ".join(f"Re{r['Re']:g}:{r['sigma']/r['A_sat']**2:.2f}"
                               for r in g))
    return rows, Re_c


if __name__ == "__main__":
    main()
