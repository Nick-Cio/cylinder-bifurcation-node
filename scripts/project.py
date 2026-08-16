"""
Turn the raw sweep into the table a parametric neural ODE actually trains on.

Builds ONE common POD basis for the whole sweep and projects every trajectory
onto it, producing

    a      (n_snapshots, r)   latent coordinates
    adot   (n_snapshots, r)   time derivative, finite-differenced
    Re     (n_snapshots,)     the parameter
    traj   (n_snapshots,)     trajectory id, so windows never straddle a run
    t      (n_snapshots,)     time in convective times

A single shared basis is the whole point.  Compute POD per Reynolds number and
a_3 at Re=60 is a different physical direction from a_3 at Re=150, so
f(z, Re) would be asked to interpolate between coordinate systems rather than
between vector fields.  Pooling also has to be weighted: fluctuation energy
grows steeply with Re, so an unweighted SVD lets the top of the sweep own the
basis and resolves the near-onset cases -- the ones that actually carry the
bifurcation -- with whatever is left over.  Each case is normalised to unit
Frobenius norm before stacking.

The mean is NOT subtracted before the SVD.  Snapshots are already perturbations
about each Reynolds number's own base flow, so the origin of the coordinate
system is the fixed point.  Subtracting the sweep mean would move it, and the
whole design is built around the fixed point sitting at z = 0 for every Re.

Run:  python -u project.py --data data --modes 3 8 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def case_matrix(d, which):
    """(n_t, 2*ncell) perturbation snapshots for one trajectory."""
    ux, uy = d[f"{which}_ux"], d[f"{which}_uy"]
    n = len(ux)
    return np.concatenate([ux.reshape(n, -1), uy.reshape(n, -1)], axis=1)


def derivative(a, t):
    """d/dt by second-order central differences on a uniform grid."""
    return np.gradient(a, t, axis=0, edge_order=2)


def write_tensor(out, r, Phi, segs, window, energy):
    """Rectangular (p, m, r) tensor -- the classical snapshot-tensor layout.

    The runs are ragged in length because critical slowing down stretches the
    near-onset transients by ~4x, so a fixed m has to come from a window rather
    than from the runs themselves.  The sampling interval is already uniform at
    0.4 D/U everywhere, so cropping to a common length is lossless in dt: the
    result is a genuine constant-rate tensor, not a resampling.

    window="last"  keeps the approach to and settling on the attractor -- the
                   consistent choice across Re, and what DMD or operator
                   inference wants.
    window="first" keeps the initial linear phase, for growth-rate work.

    The `out` and `in` families are written separately: they have different
    lengths by design, and stacking them would imply they are interchangeable
    when one starts at the fixed point and the other outside the limit cycle.
    """
    fams = {}
    for which in ("out", "in"):
        sel = [(Re, X, t) for Re, w, X, t in segs if w == which]
        if not sel:
            continue
        m = min(len(X) for _, X, _ in sel)
        A = np.stack([(X[-m:] if window == "last" else X[:m]) @ Phi.T
                      for _, X, _ in sel]).astype(np.float32)
        fams[which] = (A, np.array([Re for Re, _, _ in sel], np.float32), m)

    dt = float(np.median(np.diff(segs[0][3])))
    payload, msg = {"dt": dt, "energy": float(energy), "window": window}, []
    for which, (A, Re, m) in fams.items():
        payload[f"a_{which}"] = A
        payload[f"Re_{which}"] = Re
        msg.append(f"a_{which}{A.shape}")
    path = out / f"tensor_r{r}.npz"
    np.savez_compressed(path, **payload)
    print(f"      -> {path.name}  {'  '.join(msg)}  (p, m, r), dt={dt:g} D/U")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--modes", type=int, nargs="+", default=[3, 8, 16])
    p.add_argument("--basis-modes", type=int, default=64,
                   help="how many POD modes to store in the basis file")
    p.add_argument("--tensor", action="store_true",
                   help="also write the rectangular (p, m, r) tensor")
    p.add_argument("--window", choices=("last", "first"), default="last",
                   help="which fixed-length slice of each run to keep")
    a = p.parse_args()
    out = a.out or a.data

    files = sorted(a.data.glob("Re*.npz"), key=lambda f: float(f.stem[2:]))
    if not files:
        raise SystemExit(f"no Re*.npz in {a.data}")
    print(f"{len(files)} cases: {[float(f.stem[2:]) for f in files]}")

    # ---- assemble the pooled, equally-weighted snapshot matrix ----------
    segs = []                     # (Re, which, X, t)
    for f in files:
        d = np.load(f)
        Re = float(d["Re"])
        for which in ("out", "in"):
            X = case_matrix(d, which)
            if len(X) > 4:
                segs.append((Re, which, X, d[f"t_{which}"].astype(np.float64)))

    weights = [1.0 / max(np.linalg.norm(X), 1e-30) for _, _, X, _ in segs]
    P = np.concatenate([w * X for w, (_, _, X, _) in zip(weights, segs)], 0)
    print(f"pooled matrix {P.shape[0]} x {P.shape[1]} "
          f"({P.nbytes / 1e9:.2f} GB)")

    # economy SVD: right singular vectors are the POD modes
    _, s, Vt = np.linalg.svd(P, full_matrices=False)
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    for q in (0.9, 0.99, 0.999, 0.9999):
        print(f"  r{q * 100:g}% = {int(np.searchsorted(e, q) + 1)}")
    for r in a.modes:
        print(f"  energy captured by r={r:<3d}: {100 * e[r - 1]:.3f}%")

    nb = min(a.basis_modes, Vt.shape[0])
    np.savez_compressed(out / "pod_basis.npz", modes=Vt[:nb].astype(np.float32),
                        sv=s[:nb], energy=e[:nb],
                        crop=str(np.load(files[0])["crop"]))
    print(f"\nwrote {out / 'pod_basis.npz'}  ({nb} modes)")

    # ---- project every trajectory onto the shared basis -----------------
    for r in a.modes:
        Phi = Vt[:r]                                   # (r, ncell*2)
        A, AD, RE, TR, T = [], [], [], [], []
        for k, (Re, which, X, t) in enumerate(segs):
            z = X @ Phi.T                              # (n_t, r)
            A.append(z)
            AD.append(derivative(z, t))
            RE.append(np.full(len(z), Re))
            TR.append(np.full(len(z), k))
            T.append(t)
        A = np.concatenate(A).astype(np.float32)
        path = out / f"latent_r{r}.npz"
        np.savez_compressed(
            path, a=A, adot=np.concatenate(AD).astype(np.float32),
            Re=np.concatenate(RE).astype(np.float32),
            traj=np.concatenate(TR).astype(np.int32),
            t=np.concatenate(T).astype(np.float32),
            energy=float(e[r - 1]),
            note=json.dumps({
                "state": "POD coefficients of velocity perturbation about the "
                         "per-Re steady base flow; z = 0 is the fixed point",
                "parameter": "Re",
                "traj": "trajectory id -- do not build training windows across "
                        "a change in this value",
            }))
        print(f"wrote {path}  a{A.shape}  "
              f"{100 * e[r - 1]:.2f}% energy  {path.stat().st_size / 1e6:.1f} MB")

        if a.tensor:
            write_tensor(out, r, Phi, segs, a.window, e[r - 1])


if __name__ == "__main__":
    main()
