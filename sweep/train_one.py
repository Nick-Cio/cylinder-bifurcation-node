"""Train one NODE and write it to its own file.

One process, one core, one (Re, modes, seed, steps) combination.  Writing to a
per-run file rather than the shared pickle is what makes it safe to run many of
these at once; `merge.py` folds them into node_fits.pkl afterwards.

Thread limits are set before jax is imported.  Without them each process grabs
the whole machine and eight of them fight over it, which is slower than one.
"""
import argparse, os, sys, time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false '
                                   '--xla_force_host_platform_device_count=1')

import pickle
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

DT, L, SUBSTEPS = 0.399, 8, 2
ROOT = Path(__file__).resolve().parent.parent


def build(Re, modes):
    # latent_r16 is a nested basis: its first k columns are the r=k truncation.
    # Verified against latent_r8 to 1e-6 with correlation 1.000000 on every mode.
    d = np.load(ROOT / 'data' / 'latent_r16.npz')
    m = d['Re'] == float(Re)
    a, traj = d['a'][m][:, :modes], d['traj'][m]
    S = float(a.std())
    runs = [a[traj == j] / S for j in np.unique(traj)]
    W = jnp.asarray(np.stack([r[k:k + L + 1] for r in runs
                              for k in range(len(r) - L)]), dtype=jnp.float64)
    return runs, S, W


def make(modes):
    layers = [modes, 64, 64, modes]

    def init(key):
        ps = []
        for n_in, n_out in zip(layers[:-1], layers[1:]):
            key, k = jax.random.split(key)
            ps.append((jax.random.normal(k, (n_in, n_out)) * jnp.sqrt(1.0 / n_in),
                       jnp.zeros(n_out)))
        return ps

    def g(p, z):
        h = z
        for Wm, b in p[:-1]:
            h = jnp.tanh(h @ Wm + b)
        Wm, b = p[-1]
        return h @ Wm + b

    def f(p, z):
        return g(p, z) - g(p, jnp.zeros(modes, z.dtype))

    def step(p, z):
        for _ in range(SUBSTEPS):
            dt = DT / SUBSTEPS
            k1 = f(p, z); k2 = f(p, z + .5*dt*k1)
            k3 = f(p, z + .5*dt*k2); k4 = f(p, z + dt*k3)
            z = z + (dt/6.)*(k1 + 2*k2 + 2*k3 + k4)
        return z

    def rollout(p, z0, n):
        _, tr = jax.lax.scan(lambda z, _: (step(p, z),)*2, z0, None, length=n)
        return jnp.concatenate([z0[None], tr])

    def loss(p, wins):
        pred = jax.vmap(lambda w: rollout(p, w[0], L))(wins)
        return jnp.mean((pred - wins) ** 2)

    def adam(p, gr, m_, v_, i, lr, b1=0.9, b2=0.999, eps=1e-8):
        m_ = jax.tree.map(lambda a_, b_: b1*a_ + (1-b1)*b_, m_, gr)
        v_ = jax.tree.map(lambda a_, b_: b2*a_ + (1-b2)*b_*b_, v_, gr)
        mh = jax.tree.map(lambda a_: a_/(1-b1**i), m_)
        vh = jax.tree.map(lambda a_: a_/(1-b2**i), v_)
        return jax.tree.map(lambda p_, m2, v2: p_ - lr*m2/(jnp.sqrt(v2)+eps), p, mh, vh), m_, v_

    @jax.jit
    def train_step(p, wins, m_, v_, i, lr):
        l, gr = jax.value_and_grad(loss)(p, wins)
        p, m_, v_ = adam(p, gr, m_, v_, i, lr)
        return p, m_, v_, l

    jac = jax.jit(lambda p: jax.jacfwd(lambda z: f(p, z))(jnp.zeros(modes)))

    def eig(p):
        ev = np.linalg.eigvals(np.asarray(jac(p)))
        o = ev[np.abs(ev.imag) > 1e-6]
        if len(o) == 0:
            return np.nan, np.nan
        k = o[np.argmax(o.real)]
        return k.real, abs(k.imag)

    return init, train_step, eig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--re', type=float, required=True)
    ap.add_argument('--modes', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--every', type=int, default=50)
    ap.add_argument('--out', type=Path, default=ROOT / 'sweep' / 'runs')
    a = ap.parse_args()

    tag = f'Re{a.re:g}_r{a.modes}_seed{a.seed}_{a.steps}'
    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / f'{tag}.pkl'
    if dest.exists():
        print(f'[{tag}] already done, skipping', flush=True)
        return

    _, _, W = build(a.re, a.modes)
    init, train_step, eig = make(a.modes)
    t0 = time.time()
    p = init(jax.random.PRNGKey(a.seed))
    key = jax.random.PRNGKey(a.seed + 10_000)
    m_ = jax.tree.map(jnp.zeros_like, p)
    v_ = jax.tree.map(jnp.zeros_like, p)
    n, track = W.shape[0], {}
    for i in range(1, a.steps + 1):
        key, sub = jax.random.split(key)
        idx = jax.random.permutation(sub, n)[:a.batch]
        lr = 3e-3 if i <= 0.5*a.steps else (1e-3 if i <= 0.8*a.steps else 2e-4)
        p, m_, v_, l = train_step(p, W[idx], m_, v_, float(i), lr)
        if i % a.every == 0:
            sr, si = eig(p)
            track[i] = (float(sr), float(si/(2*np.pi)), float(l))

    tmp = dest.with_suffix('.tmp')
    with open(tmp, 'wb') as fh:
        pickle.dump((jax.tree.map(np.asarray, p), track), fh)
    tmp.replace(dest)                      # atomic, so a kill never leaves a partial file
    sr = track[a.steps][0]
    print(f'[{tag}] sigma {sr:+.5f}  loss {track[a.steps][2]:.3e}  '
          f'{(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
