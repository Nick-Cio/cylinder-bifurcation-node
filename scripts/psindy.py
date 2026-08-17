"""
Parametric SINDy on the cylinder sweep:   z' = A(Re) z + Q(z, z)

Why this exact library and not a general one
--------------------------------------------
Galerkin projection of the incompressible Navier-Stokes equations onto ANY
basis gives dynamics that are exactly linear plus quadratic -- the convective
term is bilinear and nothing of higher order exists.  The two parts carry
different parameter dependence, and that asymmetry is the whole point:

  Q  is Re-INDEPENDENT.  The nonlinearity div(u' u') contains no Reynolds
     number.  One constant tensor serves the entire sweep.
  A  carries all of it: the viscous term contributes proportional to 1/Re, and
     advection by the base flow contributes a part that depends on Re through
     u_s(Re) itself.  Modelled here as a low-order polynomial in 1/Re.

There is no constant term.  The snapshots are perturbations about each Reynolds
number's own steady base flow, so z = 0 is an exact equilibrium for every Re and
that must hold identically, not approximately.  Omitting the constant column
enforces it by construction.

Weighting
---------
The amplitudes span three decades across the sweep, so an unweighted regression
is decided almost entirely by the high-Re cases -- and the LINEAR coefficients,
which are what determine stability and therefore Re_c, are precisely what the
small-amplitude near-onset data constrains.  Each Reynolds number is therefore
normalised to equal weight by the rms of its own z-dot, exactly as each case was
normalised to equal weight in the pooled POD.

Run:  python -u psindy.py --r 8 --split onset
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

SPLITS = {
    "interp": [41.0, 62.0, 90.0, 138.0],
    "onset": [45.0, 48.0, 51.0, 54.0],
    "extrap": [80.0, 90.0, 102.0, 118.0, 138.0, 160.0],
}


def re_basis(Re, order, T=None):
    """Orthonormalised polynomials in 1/Re.

    The raw basis [1, 1/Re, 1/Re^2] is catastrophically collinear over this
    sweep: 1/Re spans only 0.006 to 0.033, so the three columns have a condition
    number in the millions and A0, A1, A2 are individually unidentifiable even
    though their sum is well determined.  The fit then reproduces z-dot beautifully
    while placing the Jacobian at the origin almost anywhere.  Orthonormalising
    against the training distribution fixes the conditioning; `T` carries the
    transform so test points use exactly the same basis.
    """
    raw = np.stack([Re ** (-p) for p in range(order + 1)], axis=-1)
    if T is None:
        q, rr = np.linalg.qr(raw)
        T = np.linalg.inv(rr) * np.sqrt(len(raw))
    return raw @ T, T


def build_library(z, Re, order, T=None):
    """Columns: {phi_p(Re) * z_j} then {z_j z_k, j<=k}.  No constant column."""
    n, r = z.shape
    phi, T = re_basis(Re, order, T)                # (n, order+1)
    lin = (phi[:, :, None] * z[:, None, :]).reshape(n, -1)
    pairs = list(itertools.combinations_with_replacement(range(r), 2))
    quad = np.stack([z[:, j] * z[:, k] for j, k in pairs], axis=1)
    names = ([f"phi{p}*z{j}" for p in range(order + 1) for j in range(r)]
             + [f"z{j}z{k}" for j, k in pairs])
    return np.concatenate([lin, quad], axis=1), names, pairs, T


def stlsq(Theta, y, w, thresh, n_iter=12, ridge=1e-8):
    """Sequentially thresholded least squares on column-normalised features."""
    scale = np.linalg.norm(Theta * w[:, None], axis=0)
    scale[scale == 0] = 1.0
    Tn = Theta / scale
    Tw, yw = Tn * w[:, None], y * w

    def solve(cols):
        if not cols.any():
            return np.zeros(Tn.shape[1])
        A = Tw[:, cols]
        xi = np.linalg.solve(A.T @ A + ridge * np.eye(cols.sum()), A.T @ yw)
        out = np.zeros(Tn.shape[1])
        out[cols] = xi
        return out

    active = np.ones(Tn.shape[1], bool)
    xi = solve(active)
    for _ in range(n_iter):
        new = active & (np.abs(xi) > thresh)
        if new.sum() == 0 or (new == active).all():
            break
        active = new
        xi = solve(active)
    return xi / scale


def fit(z, dz, Re, order, thresh, floor=0.02):
    Theta, names, pairs, T = build_library(z, Re, order)

    # Weight by INVERSE AMPLITUDE, not uniformly.  Amplitudes span three decades,
    # so an unweighted fit is decided by the saturated large-|z| samples where the
    # quadratic term dominates -- and the linear coefficients, which ARE the
    # stability and therefore Re_c, are then fitted as a small correction to that.
    # Weighting by 1/|z| makes every decade of amplitude count equally, so the
    # near-origin data actually constrains the Jacobian at the origin.  A floor at
    # `floor` x the case maximum stops the numerical noise floor being amplified.
    # (This is the same failure, and the same fix, as fitting a growth rate in
    # log-amplitude rather than in A^2.)
    nrm = np.linalg.norm(z, axis=1)
    w = np.zeros(len(z))
    for R in np.unique(Re):
        m = Re == R
        eps = floor * nrm[m].max()
        wi = 1.0 / (nrm[m] + eps)
        w[m] = wi / (np.linalg.norm(wi) + 1e-30)      # equal weight per Re
    w /= w.mean()
    Xi = np.stack([stlsq(Theta, dz[:, i], w, thresh) for i in range(z.shape[1])])
    return Xi, names, pairs, T


def predict(Xi, z, Re, order, T):
    Theta, _, _, _ = build_library(z, Re, order, T)
    return Theta @ Xi.T


def linear_operator(Xi, Re, r, order, T):
    """A(Re): the Jacobian at z = 0, which is where stability lives."""
    phi = re_basis(np.atleast_1d(float(Re)), order, T)[0][0]
    A = np.zeros((r, r))
    for p in range(order + 1):
        A += phi[p] * Xi[:, p * r:(p + 1) * r]
    return A


def leading_eig(Xi, Re, r, order, T):
    """Leading OSCILLATORY eigenvalue.

    A Hopf bifurcation is a complex pair crossing the axis, so the mode that
    matters always has a nonzero imaginary part.  Purely real eigenvalues of the
    fitted operator routinely sit above the oscillatory pair -- they belong to
    the shift mode / mean-flow distortion, not to the vortex shedding -- and
    taking argmax over the whole spectrum picks those instead.
    """
    ev = np.linalg.eigvals(linear_operator(Xi, Re, r, order, T))
    osc = np.abs(ev.imag) > 1e-6
    cand = np.where(osc)[0] if osc.any() else np.arange(len(ev))
    k = int(cand[np.argmax(ev[cand].real)])
    return float(ev[k].real), float(abs(ev[k].imag)) / (2 * np.pi)


def critical_Re(Xi, r, order, T, lo=25.0, hi=200.0):
    """Where the leading growth rate of the LEARNED operator crosses zero."""
    grid = np.linspace(lo, hi, 3500)
    s = np.array([leading_eig(Xi, R, r, order, T)[0] for R in grid])
    sign = np.where(np.diff(np.sign(s)) > 0)[0]
    if len(sign) == 0:
        return np.nan, np.nan
    i = sign[0]
    frac = -s[i] / (s[i + 1] - s[i])
    Re_c = grid[i] + frac * (grid[i + 1] - grid[i])
    return float(Re_c), float(leading_eig(Xi, Re_c, r, order, T)[1])


def load(data, r):
    d = np.load(data / f"latent_r{r}.npz")
    return (d["a"].astype(float), d["adot"].astype(float),
            d["Re"].astype(float), d["traj"].astype(int), d["t"].astype(float))


def r2_per_case(Xi, z, dz, Re, order, T):
    out = {}
    for R in np.unique(Re):
        m = Re == R
        p = predict(Xi, z[m], Re[m], order, T)
        ss = np.sum((dz[m] - p) ** 2)
        tt = np.sum((dz[m] - dz[m].mean(0)) ** 2)
        out[float(R)] = 1.0 - ss / max(tt, 1e-30)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--split", choices=list(SPLITS) + ["all"], default="onset")
    ap.add_argument("--order", type=int, default=2, help="polynomial order in 1/Re")
    ap.add_argument("--thresh", type=float, default=0.02)
    a = ap.parse_args()

    z, dz, Re, traj, t = load(a.data, a.r)
    splits = list(SPLITS) if a.split == "all" else [a.split]

    for name in splits:
        held = SPLITS[name]
        te = np.isin(Re, held)
        tr = ~te
        Xi, names, pairs, T = fit(z[tr], dz[tr], Re[tr], a.order, a.thresh)
        nnz = int((np.abs(Xi) > 0).sum())

        print(f"\n{'='*74}\nSPLIT '{name}'   held out Re = {held}")
        print(f"r = {a.r}, order = {a.order}, threshold = {a.thresh}, "
              f"{nnz}/{Xi.size} coefficients active "
              f"({100*nnz/Xi.size:.0f}% dense)")
        print(f"train {tr.sum():,} samples over {len(np.unique(Re[tr]))} Re   |   "
              f"test {te.sum():,} over {len(held)} Re")

        r2tr = r2_per_case(Xi, z[tr], dz[tr], Re[tr], a.order, T)
        r2te = r2_per_case(Xi, z[te], dz[te], Re[te], a.order, T)
        print(f"\nderivative R^2   train median {np.median(list(r2tr.values())):.4f}"
              f"   |   held-out: " +
              "  ".join(f"Re{k:g}:{v:.4f}" for k, v in sorted(r2te.items())))

        Re_c, St_c = critical_Re(Xi, a.r, a.order, T)
        print(f"\nLEARNED bifurcation:  Re_c = {Re_c:.2f}   "
              f"St at onset = {St_c:.4f}      (measured: 44.2 +/- 0.3, ~0.13)")

        print(f"\n{'Re':>6} {'sigma learned':>14} {'sigma measured':>15}  held out?")
        meas = {36: -0.046, 41: -0.0158, 45: 0.0040, 48: 0.0170, 51: 0.0286,
                54: 0.0393, 58: 0.0523, 62: 0.0639, 67: 0.0769, 73: 0.0903,
                80: 0.1031, 90: 0.1177}
        for R in sorted(np.unique(Re)):
            s, _ = leading_eig(Xi, R, a.r, a.order, T)
            m = meas.get(int(R))
            print(f"{R:>6g} {s:>+14.5f} "
                  f"{('%+.5f' % m) if m is not None else '  (unresolved)':>15}"
                  f"  {'** HELD OUT **' if R in held else ''}")


if __name__ == "__main__":
    main()
