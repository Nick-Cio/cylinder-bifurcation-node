"""
Generate the parametric dataset: a Reynolds sweep of the 2D cylinder wake.

For each Re this writes one .npz containing

    base_ux, base_uy   the *steady* solution, from the half domain with a
                       symmetry plane, so it exists at any Re -- above onset it
                       is the unstable fixed point
    out_ux, out_uy     trajectory started just off the base flow: the entire
                       spiral-OUT onto the limit cycle (or the decay back to
                       the fixed point, below onset)
    in_ux,  in_uy      trajectory started from a large perturbation: the decay
                       INWARD onto the same limit cycle
    t_out, t_in        sample times in convective times D/U
    probe_out/in       transverse velocity at a fixed wake point

Fields are stored as the *perturbation* about that Re's base flow.  That puts
the fixed point at the origin of state space for every Re, so a single
parametric model f(z, Re) only has to learn how the linearisation at the origin
moves with Re -- not how the origin itself wanders.  Storing raw fields instead
buries the bifurcation under a large Re-dependent mean shift.

Approaching the limit cycle from both sides matters: a model fed only the
spiral-out sees the vector field inside the cycle, never outside, and has
nothing pinning down the cycle's stability from the far side.

Time budget per case is set from the expected linear growth rate, which goes to
zero at onset (critical slowing down).  Near-onset cases genuinely need several
hundred convective times; the high-Re end needs a few tens.

Usage
-----
    python -u sweep.py --out data
    python -u sweep.py --re 40 46 48 52 60 80 --out data2
    python -u sweep.py --smoke                    # 3 cases, tiny, ~3 min
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lbm import CylinderBatch, Geometry

GEOM = Geometry(D=20, up=8.0, down=15.0, half_width=6.0)
U_IN = 0.07

# Clustered near the expected onset, sparse in the tail.  Entirely arbitrary --
# change it freely, nothing downstream assumes a uniform step.
DEFAULT_RE = [30, 36, 41, 45, 48, 51, 54, 58, 62, 67, 73, 80, 90, 102, 118, 138, 160]

CROP = dict(x0=-2.0, x1=12.0, y0=-4.0, y1=4.0, stride=2)   # in diameters

RE_C_GUESS = 50.0          # only used to budget run time, not assumed anywhere
SIGMA_SLOPE = 0.0044       # d(growth rate)/dRe in U/D, from linear theory


def budget_ct(Re, floor=90.0, cap=450.0, efolds=5.0):
    """Convective times needed: ~`efolds` e-foldings for the perturbation to
    cross from the kick amplitude to saturation, plus ~15 shedding periods of
    settled data.

    The kick projects only partly onto the unstable global mode, so the mode
    starts roughly a decade below the kick amplitude -- hence 5 e-foldings
    rather than the ~2 a naive amplitude ratio would suggest.  Growth-rate
    fitting itself needs far less; the extra is to reach the limit cycle.
    """
    sigma = max(abs(Re - RE_C_GUESS) * SIGMA_SLOPE, 0.012)
    return float(np.clip(efolds / sigma + floor, floor, cap))


def outer_scale(bux, buy, fx, fy, want, headroom=1.15):
    """Largest scale <= `want` that keeps the peak speed within `headroom` of
    the saturated state's own peak.

    Scaling a limit-cycle state out is the cheapest way to start outside the
    cycle, but it inflates the velocity, and at the top of the sweep tau_p is
    already ~0.53 -- close enough to the LBM stability floor that a careless
    factor blows the case up.  The saturated state has demonstrably run stable,
    so its own peak speed is the right yardstick.
    """
    u_sat = float(np.max(np.hypot(bux + fx, buy + fy)))
    cap = headroom * u_sat
    for s in np.linspace(want, 1.02, 20):
        if float(np.max(np.hypot(bux + s * fx, buy + s * fy))) <= cap:
            return float(s)
    return 1.02


def crop_slices(c):
    D, s = c.geom.D, CROP["stride"]
    return (slice(max(0, c.cy + int(CROP["y0"] * D)),
                  min(c.ny, c.cy + int(CROP["y1"] * D)), s),
            slice(max(0, c.cx + int(CROP["x0"] * D)),
                  min(c.nx, c.cx + int(CROP["x1"] * D)), s))


def steady_solve(Re, tol=3e-5, max_ct=300.0, check_ct=10.0, verbose=True,
                 warm=None):
    """Base flow: half domain + symmetry plane excludes the von Karman mode.

    Convergence is judged on the near wake only.  With free-slip walls 6D out,
    the *domain-scale* viscous time is (12D)^2/nu ~ 5000 convective times, so a
    global residual keeps creeping on a far-field adjustment of no consequence
    while the wake itself has been steady for hundreds of convective times.
    Measuring inside the crop window -- the only region that ends up in the
    dataset -- converges on the D^2/nu scale, as it should.

    `warm` is a previously converged (ux, uy) half-domain solution used as the
    initial condition, which cuts settling to a fraction of a cold start.

    The default tolerance is set by single precision, not by physics: this
    residual bottoms out around 1.5e-5 because float32 rounding in the
    populations random-walks the velocities by ~1e-6 of U over a check block.
    Asking for less just burns the whole time budget at the cap.
    """
    c = CylinderBatch([Re], GEOM, u_in=U_IN, half=True)
    if warm is not None:
        c.reset(ux=warm[0], uy=warm[1])
    D = GEOM.D
    jw = slice(0, min(c.ny, int(CROP["y1"] * D)))
    iw = slice(max(0, c.cx + int(CROP["x0"] * D)),
               min(c.nx, c.cx + int(CROP["x1"] * D)))

    block = max(1, int(check_ct * c.steps_per_ct))
    prev, res = None, np.inf
    for it in range(int(max_ct / check_ct)):
        c.run(block)
        _, ux, uy = c.macroscopic()
        cur = torch.stack([ux[0, 0, jw, iw], uy[0, 0, jw, iw]])
        if prev is not None:
            res = float((cur - prev).abs().max()) / c.u_in / check_ct
            if verbose and it % 5 == 0:
                print(f"      base {(it + 1) * check_ct:5.0f} ct  res={res:.2e}",
                      flush=True)
            if res < tol:
                break
        prev = cur
    _, ux, uy = c.macroscopic()
    ux = ux[0, 0].cpu().numpy()
    uy = uy[0, 0].cpu().numpy()
    if verbose:
        print(f"      base flow settled after {(it + 1) * check_ct:.0f} ct, "
              f"near-wake residual {res:.2e}", flush=True)
    # half -> full domain: ux even about the symmetry plane, uy odd
    return (CylinderBatch.mirror(ux, odd=False),
            CylinderBatch.mirror(uy, odd=True), res, (ux, uy))


def trajectory(Re, bux, buy, amp, n_ct, sample_ct, js, is_, init_pert=None):
    """Full-domain run from base flow plus an initial perturbation.

    With `init_pert=None` the perturbation is a localised antisymmetric blob of
    size amp*U.  Pass an explicit (px, py) field to start somewhere specific --
    in particular a scaled-up saturated state, which is the only way to begin
    *outside* the limit cycle.  A blob cannot do that however large it is: it
    is spatially localised, so it projects weakly onto the global POD modes,
    advects away within a few convective times, and the run then grows out from
    near the fixed point like any other.  Scaling a converged periodic state
    keeps the field divergence-free, so it is a legitimate initial condition.
    """
    c = CylinderBatch([Re], GEOM, u_in=U_IN, half=False)
    if init_pert is None:
        ys = np.arange(c.ny)[:, None] - c.cy
        xs = np.arange(c.nx)[None, :] - c.cx
        blob = np.exp(-((xs - 1.5 * GEOM.D) ** 2 + ys ** 2) / (0.8 * GEOM.D) ** 2)
        px, py = np.zeros_like(blob), amp * c.u_in * blob
    else:
        px, py = init_pert
    c.reset(ux=bux + px, uy=buy + py)

    bx, by = bux[js, is_], buy[js, is_]
    every = max(1, int(sample_ct * c.steps_per_ct))
    sx, sy, ts, pr = [], [], [], []

    # The probe point sits off the centreline, where the *base flow* already
    # has uy != 0 (the wake spreads).  Recording raw uy therefore leaves a
    # Re-dependent constant offset that never decays, which would corrupt every
    # growth-rate fit and inflate the apparent limit-cycle amplitude near
    # onset.  Record the perturbation instead, so a decaying case goes to zero.
    p_base = float(buy[c.cy + GEOM.D // 4, c.cx + 2 * GEOM.D])

    def take(k, s):
        _, ux, uy = s.macroscopic()
        sx.append(ux[0, 0].cpu().numpy()[js, is_] - bx)
        sy.append(uy[0, 0].cpu().numpy()[js, is_] - by)
        ts.append(k / s.steps_per_ct)
        pr.append(float(s.wake_probe()[0]) - p_base)

    c.run(int(n_ct * c.steps_per_ct), probe_every=every, probe=take)
    ok = bool(c.is_finite()[0])
    _, fux, fuy = c.macroscopic()
    final = (fux[0, 0].cpu().numpy() - bux, fuy[0, 0].cpu().numpy() - buy)
    return (np.asarray(sx, np.float32), np.asarray(sy, np.float32),
            np.asarray(ts, np.float32), np.asarray(pr, np.float32), ok, c,
            final)


def run_case(Re, outdir, sample_ct=0.4, n_ct=None, in_ct=110.0,
             amp_out=0.02, amp_in=0.60, outer=1.6, warm=None):
    t0 = time.time()
    n_ct = budget_ct(Re) if n_ct is None else n_ct
    print(f"  [Re={Re:g}] base flow ...", flush=True)
    bux, buy, res, warm_out = steady_solve(Re, warm=warm)

    probe_c = CylinderBatch([Re], GEOM, u_in=U_IN, half=False)
    js, is_ = crop_slices(probe_c)
    ny_c = len(range(*js.indices(probe_c.ny)))
    nx_c = len(range(*is_.indices(probe_c.nx)))
    print(f"  [Re={Re:g}] {probe_c}", flush=True)
    print(f"  [Re={Re:g}] budget {n_ct:.0f} + {in_ct:.0f} ct, "
          f"crop {ny_c}x{nx_c}", flush=True)
    del probe_c

    ox, oy, ot, op, ok1, c, fin = trajectory(Re, bux, buy, amp_out, n_ct,
                                             sample_ct, js, is_)

    # Second trajectory, approaching the attractor from the OTHER side.  Above
    # onset the first run ends on the limit cycle, so scaling that state out by
    # `outer` starts genuinely outside it and the run decays inward -- the only
    # data that constrains the cycle's stability from the far side.  Below onset
    # there is no cycle to sit outside of, so fall back to a large blob whose
    # decay to the fixed point measures the linear rate over a wide range.
    saturated = 0.5 * np.ptp(op[len(op) // 2:]) / U_IN > 3e-3
    init_in, s_out = None, 0.0
    if saturated:
        s_out = outer_scale(bux, buy, fin[0], fin[1], outer)
        init_in = (s_out * fin[0], s_out * fin[1])
        print(f"  [Re={Re:g}] inward run starts at {s_out:.2f}x the limit "
              f"cycle", flush=True)
    ix, iy, it_, ip, ok2, _, _ = trajectory(Re, bux, buy, amp_in, in_ct,
                                            sample_ct, js, is_,
                                            init_pert=init_in)
    if not (ok1 and ok2):
        print(f"  [Re={Re:g}] DIVERGED -- skipped", flush=True)
        return None, warm_out

    out = outdir / f"Re{Re:g}.npz"
    np.savez_compressed(
        out, Re=Re, u_in=U_IN, D=GEOM.D, nu=c.nu[0], tau_p=c.tau_p_np[0],
        base_ux=bux[js, is_].astype(np.float32),
        base_uy=buy[js, is_].astype(np.float32),
        out_ux=ox, out_uy=oy, t_out=ot, probe_out=op,
        in_ux=ix, in_uy=iy, t_in=it_, probe_in=ip,
        base_residual=res, crop=json.dumps(CROP),
        geom=json.dumps(GEOM.__dict__),
        # >1 means the inward run began outside the limit cycle; 0 means the
        # case never saturated and the inward run was a large blob instead
        inward_start_scale=s_out,
    )
    a_out = 0.5 * np.ptp(op[len(op) // 2:]) / U_IN
    a_in = 0.5 * np.ptp(ip[len(ip) // 2:]) / U_IN
    print(f"  [Re={Re:g}] amp(out)={a_out:.3e}  amp(in)={a_in:.3e}  "
          f"{len(ot)}+{len(it_)} snaps  {out.stat().st_size / 1e6:.0f} MB  "
          f"{time.time() - t0:.0f} s", flush=True)
    return out, warm_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--re", type=float, nargs="+", default=None)
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--sample-ct", type=float, default=0.4)
    p.add_argument("--n-ct", type=float, default=None,
                   help="override the per-Re time budget")
    p.add_argument("--smoke", action="store_true",
                   help="3 cases, short runs -- checks the pipeline only")
    a = p.parse_args()

    res = a.re
    if a.smoke:
        res, a.n_ct, a.out = [35.0, 60.0, 120.0], 90.0, Path("data_smoke")
    res = DEFAULT_RE if res is None else res

    a.out.mkdir(parents=True, exist_ok=True)
    total = sum(budget_ct(r) if a.n_ct is None else a.n_ct for r in res)
    print(f"sweep over Re = {[float(r) for r in res]}")
    print(f"geometry {GEOM}, blockage {GEOM.blockage:.1%}, u_in={U_IN}")
    print(f"~{total:.0f} convective times of full-domain integration\n", flush=True)

    # Cheapest cases first, so the sweep brackets the bifurcation early and
    # every case is written as it finishes -- a partial run is already a usable
    # dataset.  The expensive near-onset points, where critical slowing down
    # bites, come last.  Each base flow still warm-starts, from the nearest Re
    # already solved rather than from the previous one in the list.
    bank = {}
    for Re in sorted(res, key=budget_ct):
        warm = bank[min(bank, key=lambda r: abs(r - Re))] if bank else None
        _, w = run_case(Re, a.out, sample_ct=a.sample_ct, n_ct=a.n_ct,
                        in_ct=50.0 if a.smoke else 70.0, warm=warm)
        if w is not None:
            bank[Re] = w


if __name__ == "__main__":
    main()
