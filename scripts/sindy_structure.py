"""Does adding structure to the SINDy library fix sigma?

Three things are compared against the plain linear+quadratic fit.

  cubic      Adds z_i z_j z_k.  The Hopf normal form is cubic in Cartesian
             coordinates (Stuart-Landau), so if the POD truncation does not
             represent the shift mode well, cubic terms are the standard way to
             get amplitude saturation.  Costs a lot of columns.

  energy     Constrains the quadratic tensor to be energy preserving.  The
             convective term of Navier-Stokes does no net work on the
             perturbation, so z . Q(z,z) = 0 identically.  With Q symmetric in
             its last two indices that is q_ijk + q_jki + q_kij = 0 for every
             unordered triple.  This is a HARD linear constraint, not a penalty,
             and it is the constraint of Loiseau & Brunton (JFM 2018).

             It matters here for a specific reason.  sigma depends only on A, and
             the trouble with the plain fit is that A and Q are not separately
             identifiable -- many splits reproduce zdot equally well.  Removing
             degrees of freedom from Q is exactly what should sharpen A.

  both       Cubic library with the quadratic block still constrained.

The constraint couples different output rows, so unlike ordinary SINDy this
cannot be solved one mode at a time.  The whole coefficient set is solved jointly
by projecting onto the nullspace of the constraint.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MEASURED = {36.0: -0.04589, 73.0: 0.09030}


def osc_eig(A):
    ev = np.linalg.eigvals(A)
    o = ev[np.abs(ev.imag) > 1e-6]
    if len(o) == 0:
        return np.nan, np.nan
    k = o[np.argmax(o.real)]
    return float(k.real), float(abs(k.imag) / (2 * np.pi))


def library(z, cubic):
    r = z.shape[1]
    pairs = list(itertools.combinations_with_replacement(range(r), 2))
    cols = [z] + [np.stack([z[:, j] * z[:, k] for j, k in pairs], axis=1)]
    trips = []
    if cubic:
        trips = list(itertools.combinations_with_replacement(range(r), 3))
        cols.append(np.stack([z[:, i] * z[:, j] * z[:, k] for i, j, k in trips], axis=1))
    return np.concatenate(cols, axis=1), pairs, trips


def energy_constraint(r, ncol):
    """Rows of C acting on vec(Xi), Xi shape (r, ncol), row-major.

    q_ijk + q_jki + q_kij = 0 for every unordered triple, where the coefficient
    on column (j,k) of row i equals q_ijk doubled when j != k.
    """
    pairs = list(itertools.combinations_with_replacement(range(r), 2))
    pidx = {p: n for n, p in enumerate(pairs)}

    def col_of(j, k):
        return r + pidx[(min(j, k), max(j, k))]

    rows = []
    for i, j, k in itertools.combinations_with_replacement(range(r), 3):
        row = np.zeros(r * ncol)
        for a, b, c in ((i, j, k), (j, k, i), (k, i, j)):
            row[a * ncol + col_of(b, c)] += 1.0 if b == c else 0.5
        rows.append(row)
    return np.array(rows)


def fit(z, dz, power, cubic=False, energy=False, floor=0.02, ridge=1e-10):
    Theta, pairs, trips = library(z, cubic)
    n, ncol = Theta.shape
    r = z.shape[1]
    nrm = np.linalg.norm(z, axis=1)
    w = 1.0 / (nrm + floor * nrm.max()) ** power
    w = w / w.mean()
    sw = np.sqrt(w)
    Tw, Yw = Theta * sw[:, None], dz * sw[:, None]

    if not energy:
        G = Tw.T @ Tw + ridge * np.eye(ncol)
        Xi = np.linalg.solve(G, Tw.T @ Yw).T
    else:
        C = energy_constraint(r, ncol)
        # nullspace of C
        _, s, Vt = np.linalg.svd(C, full_matrices=True)
        rank = int((s > 1e-10 * s[0]).sum())
        N = Vt[rank:].T                       # (r*ncol, dof)
        # blockdiag(Tw) @ N, assembled without forming the block matrix
        MN = np.zeros((n * r, N.shape[1]))
        for i in range(r):
            MN[i * n:(i + 1) * n] = Tw @ N[i * ncol:(i + 1) * ncol]
        Y = Yw.T.reshape(-1)                  # rows stacked to match
        u = np.linalg.solve(MN.T @ MN + ridge * np.eye(N.shape[1]), MN.T @ Y)
        Xi = (N @ u).reshape(r, ncol)

    return Xi, Theta, pairs, trips


def energy_residual(Xi, r):
    """RMS of q_ijk + q_jki + q_kij over all triples; zero if energy preserving."""
    pairs = list(itertools.combinations_with_replacement(range(r), 2))
    pidx = {p: n for n, p in enumerate(pairs)}

    def q(i, j, k):
        c = Xi[i, r + pidx[(min(j, k), max(j, k))]]
        return c if j == k else 0.5 * c

    v = [q(i, j, k) + q(j, k, i) + q(k, i, j)
         for i, j, k in itertools.combinations_with_replacement(range(r), 3)]
    return float(np.sqrt(np.mean(np.square(v))))


def rk4(f, z0, n, dt=0.399, sub=2, cap=1e4):
    h = dt / sub
    z = np.asarray(z0, float)
    out = np.full((n + 1, len(z)), np.nan)
    out[0] = z
    for i in range(1, n + 1):
        for _ in range(sub):
            k1 = f(z); k2 = f(z + .5 * h * k1)
            k3 = f(z + .5 * h * k2); k4 = f(z + h * k3)
            z = z + (h / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(z)) or np.linalg.norm(z) > cap:
            return out
        out[i] = z
    return out


def make_field(Xi, pairs, trips, r):
    A = Xi[:, :r]
    Q = Xi[:, r:r + len(pairs)]
    Cb = Xi[:, r + len(pairs):]

    def f(zz):
        v = A @ zz + Q @ np.array([zz[j] * zz[k] for j, k in pairs])
        if len(trips):
            v = v + Cb @ np.array([zz[i] * zz[j] * zz[k] for i, j, k in trips])
        return v

    return f


def main():
    d = np.load(ROOT / 'data' / 'latent_r8.npz')
    a, ad = d['a'].astype(float), d['adot'].astype(float)
    Re, traj = d['Re'].astype(float), d['traj'].astype(int)
    r = 8

    VARIANTS = [('quadratic (baseline)', dict(cubic=False, energy=False)),
                ('+ cubic terms',        dict(cubic=True,  energy=False)),
                ('+ energy constraint',  dict(cubic=False, energy=True)),
                ('cubic + energy',       dict(cubic=True,  energy=True))]

    for R in (36.0, 73.0):
        m = Re == R
        z, dz, tj = a[m], ad[m], traj[m]
        print(f'\n{"="*96}\nRe {R:g}   measured sigma {MEASURED[R]:+.5f}   '
              f'{m.sum()} snapshots')
        print(f'{"variant":22} {"p":>2} {"cols":>5} {"sigma":>10} {"error":>8} '
              f'{"St":>7} {"dMSE":>10} {"E-resid":>9}  steps completed')
        for label, kw in VARIANTS:
            for p in (0, 2):
                Xi, Theta, pairs, trips = fit(z, dz, p, **kw)
                s, st = osc_eig(Xi[:, :r])
                dmse = float(np.mean((Theta @ Xi.T - dz) ** 2))
                eres = energy_residual(Xi, r)
                f = make_field(Xi, pairs, trips, r)
                surv = []
                for t_ in np.unique(tj):
                    tru = z[tj == t_]
                    pr = rk4(f, tru[0], len(tru) - 1)
                    surv.append(f'{int(np.all(np.isfinite(pr), axis=1).sum())}/{len(tru)}')
                print(f'{label:22} {p:>2} {Theta.shape[1]:>5} {s:>+10.5f} '
                      f'{100*abs(s-MEASURED[R])/abs(MEASURED[R]):>7.1f}% {st:>7.4f} '
                      f'{dmse:>10.2e} {eres:>9.2e}  ' + '  '.join(surv), flush=True)


if __name__ == '__main__':
    main()
