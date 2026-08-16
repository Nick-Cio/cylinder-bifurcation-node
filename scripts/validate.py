"""
Sanity checks before committing to the sweep.

  1. throughput -- steps/s vs batch size, to cost the sweep
  2. Re = 30, 45  full domain: must decay to a steady wake
  3. Re = 80, 140 full domain: must shed;  St should land near 0.16-0.18
  4. Re = 140     half domain: must stay steady (that is the base flow)

Run:  python -u validate.py
"""

from __future__ import annotations

import time

import numpy as np

from lbm import CylinderBatch, Geometry

GEOM = Geometry(D=20, up=8.0, down=22.0, half_width=10.0)


def throughput(B, n=100):
    c = CylinderBatch([60.0] * B, GEOM)
    c.run(20)
    t0 = time.time()
    c.run(n)
    dt = time.time() - t0
    return n / dt, B * n / dt


def kick(c, amp=1e-2):
    """Break the symmetry once with a transverse blob in the near wake."""
    _, ux, uy = c.macroscopic()
    ys = np.arange(c.ny)[:, None] - c.cy
    xs = np.arange(c.nx)[None, :] - c.cx
    blob = np.exp(-((xs - 1.5 * GEOM.D) ** 2 + ys ** 2) / (0.8 * GEOM.D) ** 2)
    import torch
    uy = uy + torch.as_tensor(amp * c.u_in * blob, dtype=torch.float32,
                              device=c.dev).view(1, 1, c.ny, c.nx)
    c.reset(ux=ux.expand(c.B, 1, c.ny, c.nx).reshape(c.B, c.ny, c.nx),
            uy=uy.expand(c.B, 1, c.ny, c.nx).reshape(c.B, c.ny, c.nx))


def probe_run(Re, half=False, n_ct=250, sample_ct=0.25, do_kick=True):
    c = CylinderBatch(Re, GEOM, half=half)
    print(f"  {c}", flush=True)
    if do_kick:
        kick(c)
    every = max(1, int(sample_ct * c.steps_per_ct))
    hist = []
    t0 = time.time()
    c.run(int(n_ct * c.steps_per_ct), probe_every=every,
          probe=lambda k, s: hist.append(s.wake_probe()))
    hist = np.asarray(hist)                       # (n_samples, B)
    dt_ct = every / c.steps_per_ct

    print(f"  {n_ct:.0f} convective times in {time.time()-t0:.0f} s", flush=True)
    for b, Reb in enumerate(c.Re):
        y = hist[:, b]
        if not np.isfinite(y).all():
            print(f"    Re={Reb:6g}  DIVERGED")
            continue
        tail = y[len(y) // 2:]
        amp = 0.5 * np.ptp(tail) / c.u_in
        st = ""
        if amp > 1e-4:
            s = tail - tail.mean()
            Y = np.abs(np.fft.rfft(s * np.hanning(len(s))))
            f = np.fft.rfftfreq(len(s), dt_ct)     # cycles per ct == Strouhal
            st = f"  St={f[np.argmax(Y[1:]) + 1]:.4f}"
        # growth/decay rate over the first half, in units of U/D
        g = np.abs(y[: len(y) // 2]) + 1e-30
        n0, n1 = len(g) // 4, len(g) // 2
        sigma = np.polyfit(np.arange(n0, n1) * dt_ct, np.log(g[n0:n1]), 1)[0]
        print(f"    Re={Reb:6g}  amp/U={amp:.3e}  sigma={sigma:+.4f}{st}",
              flush=True)
    return c, hist


if __name__ == "__main__":
    print("throughput (full domain, 461x401):")
    for B in (1, 4, 17):
        sps, cps = throughput(B)
        print(f"  B={B:3d}   {sps:6.0f} steps/s   {cps:7.0f} case-steps/s   "
              f"{sps / (GEOM.D / 0.06):5.2f} ct/s", flush=True)

    print("\nfull domain, expect decay below onset and shedding above:")
    probe_run([30.0, 45.0, 80.0, 140.0], n_ct=250)

    print("\nhalf domain at Re=140 (expect steady -- this is the base flow):")
    probe_run([140.0], half=True, n_ct=250, do_kick=False)
