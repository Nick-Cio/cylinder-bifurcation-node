"""SINDy vs NODE at two Reynolds numbers: one stable (36), one on the limit cycle (73).

What is being compared, and what is fair
----------------------------------------
The NODE is trained on ONE Reynolds number and its sigma is eig J(0) of the
learned field.  So the fair SINDy counterpart is also fitted on one Reynolds
number: `local` below.  The parametric fit that psindy.py performs sees the
whole sweep at once, which is a different and easier problem for sigma but the
only one that can produce a critical Reynolds number, so it is reported too.

The one knob SINDy has that the NODE does not
---------------------------------------------
SINDy is linear in its coefficients, so the sample weighting is an explicit
dial rather than something buried in a training loop.  Weighting by
1/(|z|+eps)^p and sweeping p moves the fit continuously from "reproduce the
trajectory" (p=0) to "get the Jacobian at the origin right" (large p).  That
sweep is the clearest statement of the trade-off this whole project ran into:
the best-fitting model is not the one with the right growth rate.

Both models are closed ODEs in the same latent coordinates and are integrated
with the same RK4 stepper from the same initial conditions, so the rollout
comparison is straight.  The NODE trains on a/S for one global scale S; its
field is unscaled back to physical units here.

Writes notebooks/sindy_fits.pkl.
"""
from __future__ import annotations

import itertools
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

import psindy

DT, SUBSTEPS = 0.399, 2
TARGETS = [36.0, 73.0]
MEASURED = {36.0: -0.04589, 73.0: 0.09030}
POWERS = [0, 1, 2, 3, 4]
R_HEAD = 8                      # local SINDy at r=14 is dominated by the noise modes


# --------------------------------------------------------------------- data

def load(r):
    """latent_r16 is nested, so any r <= 16 is a column slice."""
    src = r if r in (3, 8, 16) else 16
    d = np.load(ROOT / 'data' / f'latent_r{src}.npz')
    return (d['a'].astype(float)[:, :r], d['adot'].astype(float)[:, :r],
            d['Re'].astype(float), d['traj'].astype(int), d['t'].astype(float))


# -------------------------------------------------------------------- SINDy

def library(z):
    """[z_j] then [z_j z_k, j<=k].  No constant column, so z=0 stays an equilibrium."""
    r = z.shape[1]
    pairs = list(itertools.combinations_with_replacement(range(r), 2))
    quad = np.stack([z[:, j] * z[:, k] for j, k in pairs], axis=1)
    return np.concatenate([z, quad], axis=1), pairs


def fit_local(z, dz, power, thresh=0.0, floor=0.02, ridge=1e-10):
    """Fit zdot = A z + Q(z,z) on a single Reynolds number."""
    Theta, pairs = library(z)
    nrm = np.linalg.norm(z, axis=1)
    w = 1.0 / (nrm + floor * nrm.max()) ** power
    w = w / w.mean()
    if thresh > 0:
        Xi = np.stack([psindy.stlsq(Theta, dz[:, i], w, thresh)
                       for i in range(z.shape[1])])
    else:
        G = (Theta * w[:, None]).T @ Theta + ridge * np.eye(Theta.shape[1])
        Xi = np.linalg.solve(G, (Theta * w[:, None]).T @ dz).T
    return Xi, pairs


def make_field(Xi, pairs, r):
    A, Q = Xi[:, :r], Xi[:, r:]

    def f(z):
        return A @ z + Q @ np.array([z[j] * z[k] for j, k in pairs])

    return f, A


def osc_eig(A):
    """Leading OSCILLATORY eigenvalue: a Hopf pair always has nonzero imaginary part."""
    ev = np.linalg.eigvals(A)
    o = ev[np.abs(ev.imag) > 1e-6]
    if len(o) == 0:
        return np.nan, np.nan
    k = o[np.argmax(o.real)]
    return float(k.real), float(abs(k.imag) / (2 * np.pi))


# --------------------------------------------------------------------- NODE

def node_field(params, S, modes):
    """Trained on x = z/S, so dz/dt = S*(g(z/S) - g(0))."""
    (W0, b0), (W1, b1), (W2, b2) = params

    def g(x):
        h = np.tanh(x @ W0 + b0)
        h = np.tanh(h @ W1 + b1)
        return h @ W2 + b2

    g0 = g(np.zeros(modes))
    return lambda z: S * (g(z / S) - g0)


def jac0(f, r, eps=1e-6):
    J = np.zeros((r, r))
    for j in range(r):
        e = np.zeros(r)
        e[j] = eps
        J[:, j] = (f(e) - f(-e)) / (2 * eps)
    return J


# ------------------------------------------------------------------ rollout

def rk4(f, z0, n, dt=DT, sub=SUBSTEPS, cap=1e4):
    h = dt / sub
    z = np.asarray(z0, float)
    out = np.full((n + 1, len(z)), np.nan)
    out[0] = z
    for i in range(1, n + 1):
        for _ in range(sub):
            k1 = f(z)
            k2 = f(z + .5 * h * k1)
            k3 = f(z + .5 * h * k2)
            k4 = f(z + h * k3)
            z = z + (h / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(z)) or np.linalg.norm(z) > cap:
            return out
        out[i] = z
    return out


def roll_all(f, a, traj, r):
    out = {}
    for tj in np.unique(traj):
        truth = a[traj == tj]
        out[int(tj)] = rk4(f, truth[0], len(truth) - 1)
    return out


def rms(pred, truth):
    m = np.all(np.isfinite(pred), axis=1)
    if m.sum() < 2:
        return np.inf
    return float(np.sqrt(np.mean((pred[m] - truth[m]) ** 2)))


# --------------------------------------------------------------------- main

def main():
    out = {'meta': {'dt': DT, 'substeps': SUBSTEPS, 'measured': MEASURED,
                    'powers': POWERS, 'r_head': R_HEAD}}

    for r in (8, 14):
        z, dz, Re, traj, t = load(r)

        # ---- local SINDy: one Re at a time, weighting swept from fit to Jacobian
        for R in TARGETS:
            m = Re == R
            zi, dzi, tji = z[m], dz[m], traj[m]
            for p in POWERS:
                Xi, pairs = fit_local(zi, dzi, p)
                f, A = make_field(Xi, pairs, r)
                s, st = osc_eig(A)
                roll = roll_all(f, zi, tji, r)
                rec = {'Xi': Xi, 'A': A, 'sigma': (s, st), 'power': p, 'roll': roll,
                       'r': r,
                       'dmse': float(np.mean((library(zi)[0] @ Xi.T - dzi) ** 2)),
                       'rms': {tj: rms(roll[tj], zi[tji == tj]) for tj in roll}}
                out[('sindy_local', r, R, p)] = rec
                print(f'[sindy local r={r} Re{R:g} p={p}] sigma={s:+.5f} '
                      f'({100*abs(s-MEASURED[R])/abs(MEASURED[R]):6.1f}%) St={st:.4f} '
                      f'dmse={rec["dmse"]:.3e} rms=' +
                      ' '.join(f'{v:.3f}' for v in rec['rms'].values()), flush=True)

        # ---- amplitude-restricted LINEAR fit: a diagnostic, not a usable model
        for R in TARGETS:
            m = Re == R
            zi, dzi = z[m], dz[m]
            nrm = np.linalg.norm(zi, axis=1)
            band = {}
            for cut in (0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 1e9):
                mm = nrm < cut
                if mm.sum() < 3 * r:
                    continue
                A = np.linalg.lstsq(zi[mm], dzi[mm], rcond=None)[0].T
                band[cut] = (*osc_eig(A), int(mm.sum()))
            out[('linband', r, R)] = band
            print(f'[linear band r={r} Re{R:g}] ' +
                  '  '.join(f'|z|<{c:g}:{v[0]:+.4f}(n={v[2]})' for c, v in band.items()),
                  flush=True)

    # ---- parametric SINDy across the whole sweep: the only route to Re_c
    for r in (8, 14):
        zp, dzp, Rep, trajp, _ = load(r)
        for order in (1, 2):
            for tag, excl in (('insample', ()), ('heldout', TARGETS)):
                keep = ~np.isin(Rep, list(excl))
                Xi, names, pairs, T = psindy.fit(zp[keep], dzp[keep], Rep[keep],
                                                 order, 0.02)
                rec = {'Xi': Xi, 'T': T, 'order': order, 'r': r,
                       'excluded': list(excl), 'n_train': int(keep.sum()),
                       'nnz': int((np.abs(Xi) > 0).sum()), 'size': Xi.size,
                       'Re_c': psindy.critical_Re(Xi, r, order, T),
                       'sigma': {float(R): psindy.leading_eig(Xi, R, r, order, T)
                                 for R in np.unique(Rep)}, 'roll': {}}
                for R in TARGETS:
                    A = psindy.linear_operator(Xi, R, r, order, T)
                    Q = Xi[:, (order + 1) * r:]
                    prs = list(itertools.combinations_with_replacement(range(r), 2))
                    f = (lambda A=A, Q=Q, prs=prs:
                         lambda zz: A @ zz + Q @ np.array(
                             [zz[j] * zz[k] for j, k in prs]))()
                    mm = Rep == R
                    rec['roll'][R] = roll_all(f, zp[mm], trajp[mm], r)
                    rec.setdefault('rms', {})[R] = {
                        tj: rms(rec['roll'][R][tj], zp[mm][trajp[mm] == tj])
                        for tj in rec['roll'][R]}
                out[('sindy_param', r, order, tag)] = rec
                print(f'[sindy param r={r} order={order} {tag}] '
                      f'Re_c={rec["Re_c"][0]:.2f}  ' +
                      '  '.join(f'Re{R:g}={rec["sigma"][R][0]:+.5f}'
                                for R in TARGETS), flush=True)

    # ---- same library, same solver, but with the quadratic tensor forced to be
    # energy preserving.  z . Q(z,z) = 0 holds identically for the convective
    # term of Navier-Stokes, so this adds physics rather than parameters: the
    # coefficient count is unchanged and 120 degrees of freedom are removed.
    import sindy_structure as ss
    z8, dz8, Re8, traj8, _ = load(8)
    for R in TARGETS:
        m = Re8 == R
        zi, dzi, tji = z8[m], dz8[m], traj8[m]
        for p in POWERS:
            Xi, Theta, pairs, trips = ss.fit(zi, dzi, p, cubic=False, energy=True)
            s, st = ss.osc_eig(Xi[:, :8])
            f = ss.make_field(Xi, pairs, trips, 8)
            roll = roll_all(f, zi, tji, 8)
            out[('sindy_energy', 8, R, p)] = {
                'Xi': Xi, 'sigma': (s, st), 'power': p, 'roll': roll, 'r': 8,
                'dmse': float(np.mean((Theta @ Xi.T - dzi) ** 2)),
                'eres': ss.energy_residual(Xi, 8),
                'rms': {tj: rms(roll[tj], zi[tji == tj]) for tj in roll}}
            # energy violation of the UNCONSTRAINED fit, for the diagnostic
            out[('sindy_local', 8, R, p)]['eres'] = ss.energy_residual(
                out[('sindy_local', 8, R, p)]['Xi'], 8)
            print(f'[sindy energy Re{R:g} p={p}] sigma={s:+.5f} '
                  f'({100*abs(s-MEASURED[R])/abs(MEASURED[R]):6.1f}%) '
                  f'E-resid={out[("sindy_energy", 8, R, p)]["eres"]:.1e} '
                  f'(unconstrained {out[("sindy_local", 8, R, p)]["eres"]:.1e})',
                  flush=True)

    # ---- where does the parametric fit get Re 36 from?
    # Re 36 is held out throughout.  Drop groups of OTHER Reynolds numbers from
    # training and watch what happens to its prediction: whatever hurts is what
    # it was borrowing from.
    zp, dzp, Rep, _, _ = load(14)
    ABL = [('hold out 36, 73 only (baseline)', [36, 73]),
           ('drop near-onset 45, 48', [36, 73, 45, 48]),
           ('drop near-onset 45, 48, 51, 54', [36, 73, 45, 48, 51, 54]),
           ('drop all near-onset 45 to 62', [36, 73, 45, 48, 51, 54, 58, 62]),
           ('drop the 4 highest Re instead', [36, 73, 102, 118, 138, 160]),
           ('drop the other subcritical 30, 41', [36, 73, 30, 41])]
    abl = []
    for label, excl in ABL:
        keep = ~np.isin(Rep, excl)
        Xi, _, _, T = psindy.fit(zp[keep], dzp[keep], Rep[keep], 1, 0.02)
        s, _ = psindy.leading_eig(Xi, 36.0, 14, 1, T)
        rc, _ = psindy.critical_Re(Xi, 14, 1, T)
        abl.append({'label': label, 'excluded': excl, 'n_Re': int(len(np.unique(Rep[keep]))),
                    'n': int(keep.sum()), 'sigma36': float(s), 'Re_c': float(rc)})
        print(f'[ablation] {label:36s} n_Re={abl[-1]["n_Re"]:2d} '
              f'sigma(36)={s:+.5f} Re_c={rc:.2f}', flush=True)
    out['ablation'] = abl

    # is sigma identifiable from each Reynolds number's OWN data?  best linear
    # fit over amplitude cuts, which is the most favourable local estimate there is
    MEAS_ALL = {30: np.nan, 36: -0.04589, 41: -0.01579, 45: +0.00398, 48: +0.01698,
                51: +0.02859, 54: +0.03932, 58: +0.05230, 62: +0.06392,
                67: +0.07686, 73: +0.09030, 80: +0.10309, 90: +0.11770}
    z8, dz8, Re8, _, _ = load(8)
    ident = {}
    for R, meas in MEAS_ALL.items():
        if not np.isfinite(meas):
            continue
        m = Re8 == R
        nrm = np.linalg.norm(z8[m], axis=1)
        cand = []
        for cut in (0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
            mm = nrm < cut
            if mm.sum() < 30:
                continue
            A = np.linalg.lstsq(z8[m][mm], dz8[m][mm], rcond=None)[0].T
            s, _ = osc_eig(A)
            cand.append((100 * abs(s - meas) / abs(meas), float(cut), float(s), int(mm.sum())))
        if cand:
            e, cut, s, n = min(cand)
            ident[R] = {'err': e, 'cut': cut, 'sigma': s, 'n': n,
                        'snaps': int(m.sum()), 'meas': meas}
            print(f'[identifiable] Re {R:3d}: {m.sum():5d} snaps, best local '
                  f'sigma={s:+.5f} vs {meas:+.5f} -> {e:5.1f}%', flush=True)
    out['identifiable'] = ident

    # ---- NODE counterparts on the same footing
    fits = pickle.load(open(ROOT / 'notebooks' / 'node_fits.pkl', 'rb'))
    d16 = np.load(ROOT / 'data' / 'latent_r16.npz')
    for R in TARGETS:
        for modes in (8, 14):
            m = d16['Re'] == R
            a_all = d16['a'][m][:, :modes].astype(float)
            traj_all = d16['traj'][m]
            S = float(a_all.std())
            for steps in (15000, 100000):
                for seed in (0, 1):
                    keys = ([f'Re{R:g}_mb128_L9_seed{seed}_{steps}',
                             f'Re{R:g}_r8_seed{seed}_{steps}'] if modes == 8
                            else [f'Re{R:g}_r{modes}_seed{seed}_{steps}'])
                    key = next((k for k in keys if k in fits), None)
                    if key is None:
                        continue
                    f = node_field(fits[key][0], S, modes)
                    s, st = osc_eig(jac0(f, modes))
                    roll = roll_all(f, a_all, traj_all, modes)
                    out[('node', R, modes, steps, seed)] = {
                        'key': key, 'sigma': (s, st), 'S': S, 'roll': roll,
                        'rms': {tj: rms(roll[tj], a_all[traj_all == tj]) for tj in roll}}
                    print(f'[node {key}] sigma={s:+.5f} '
                          f'({100*abs(s-MEASURED[R])/abs(MEASURED[R]):5.1f}%)', flush=True)

    with open(ROOT / 'notebooks' / 'sindy_fits.pkl', 'wb') as fh:
        pickle.dump(out, fh)
    print('\nwrote notebooks/sindy_fits.pkl')


if __name__ == '__main__':
    main()
