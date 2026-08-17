# Neural ODEs for the cylinder-wake Hopf bifurcation

Nicholas Ciordas, SURF 2026.

2D flow past a cylinder undergoes a supercritical Hopf bifurcation: below a
critical Reynolds number the wake is a steady fixed point, above it a limit
cycle. 17 Reynolds numbers were simulated across that transition and projected
onto a shared POD basis. The goal is a neural ODE `ż = f_θ(z)` whose Jacobian at
the origin reproduces the measured growth rate σ, and ultimately a parametric
model that recovers the bifurcation at Reynolds numbers it never trained on.

**Measured ground truth: `Re_c = 44.15`, `dσ/dRe = 0.00474`.**

Dataset and solver documentation is in [`DATASET.md`](DATASET.md); the physics
write-up is `report/cylinder_report.pdf`. This file documents the modelling work.

---

## Headline result

**The trajectory-rollout loss does not determine σ, and training harder makes σ
worse.** Every experiment below is a different attempt to find out why and to fix
it. None of the fixes worked, but the diagnosis is now specific and testable.

The single most compact statement, four architecturally different models at
Re = 73:

| model | σ error | free-rollout RMS |
|---|---:|---:|
| plain NODE, 15k steps | **+1%** | **1.2986** |
| explicit linear part `A` | +11% | 0.6916 |
| explicit `A` + near-origin penalty | +24% | 0.6574 |
| plain NODE, 100k steps | **+73%** | **0.3036** |

Rollout accuracy and σ accuracy run in **opposite** directions, without exception.
The 100k model reproduces all 8 modes over the full record almost exactly while
the 15k model saturates 15 convective times late — and the 100k model's growth
rate is wrong by a factor of 1.7. No quantity computable from the trajectories
would tell you to prefer the model with the correct σ.

## Why, as far as we can tell

The wake operator is strongly **non-normal**, so the amplitude can grow much
faster than the leading eigenvalue during a transient. The bound on that
transient rate is the numerical abscissa, `max eig((J+Jᵀ)/2)`.

| | σ | numerical abscissa |
|---|---:|---:|
| data's own operator at Re = 73 | +0.094 | **+0.30** |
| NODE, 15k | +0.091 | +1.92 |
| NODE, 100k | +0.156 | **+0.27** |

The 15k model matches the **eigenvalue**; the 100k model matches the **transient
operator**. The loss grades an 8-step rollout — about 3.2 convective times, deep
in the transient regime — so it selects the second.

And the transient rate is **positive at every Reynolds number in the sweep,
including 14 units below onset** (+1.52 at Re = 30, +1.18 at Re = 36). A quantity
that never crosses zero cannot locate a bifurcation. That is why the models
report positive σ below onset, where the truth is negative.

---

## Notebooks

Run in order; each is self-contained apart from the shared fit cache.

### `1_cylinder_node_problem.ipynb`
Problem statement and data inventory. What the POD coordinates are, how many
Reynolds numbers and trajectories, the bifurcation visible in the raw data, and
the σ(Re) target.

### `2_cylinder_node_single_Re.ipynb` — the main investigation
One neural ODE at Re = 73, no parameterisation. 8 POD coordinates in, 8
derivatives out, 9-state multiple-shooting windows, RK4 with 2 substeps.

Validated settings (each tested against dense σ paths across 3+ seeds):
minibatch 128 **adopted** (3.4× faster, indistinguishable); float32 **rejected**
(moves σ by more than the seed spread); 1 RK4 substep **rejected** (violates the
stability limit); 20-state windows **rejected**; **15,000 steps adopted**.

| section | finding |
|---|---|
| σ vs training length | 15k gives σ = +0.0905 ± 0.0022, a 0% bias, all 5 seeds within 4%. At 100k the loss improves 13× and σ goes to +0.1526 |
| amplitude decomposition | 72% of windows are on the attractor and produce 83% of the loss — a headcount fact, not a weighting artifact. The inward run is 100% limit cycle and carries no σ information |
| local growth rate vs `\|z\|` | the field is only flat near the origin when underfit; the converged model curves sharply where the data is densest |
| overfitting test | a contiguous hole cut in the transient, 5 seeds. Held-out loss carries **no information** about σ (Pearson −0.31/+0.22, Spearman +0.10/+0.50). Validation-based early stopping lands on a σ error of ~+50% |
| region ablation | six equal-size cuts. No region dominates; the attractor is redundant; different regions bias σ in opposite directions |
| loss reweighting | up-weighting the regions that mattered most changed nothing (bias +0% → −1%, spread 2× worse) |
| growth rate vs amplitude | the data never grows at σ; **no** amplitude measure is flat at 0.0903 |
| non-normality | the table above |

### `3_cylinder_node_linear_part.ipynb` — the structural fix
`ż = Az + g_θ(z)` with `g_θ(z) = h_θ(z) − h_θ(0) − J_h(0)z`, so `g` vanishes to
second order and `∂f/∂z|₀ = A` exactly. σ becomes a fitted parameter rather than
a derivative of a black box.

Cold start (`A = 0`) fails. Warm start from a least-squares fit gives σ = +15%
versus the plain model's +0%, but **σ wander after step 10k drops 7×** and the
frequency error improves from −11% to −6%. Adding a penalty keeping `‖g‖` small
inside `‖z‖ < 0.3` suppresses `g` by 300× and makes the field genuinely linear
near the origin — and σ moves to +22%. **The curvature was never the problem.**

### `4_cylinder_node_other_Re.ipynb` — does it transfer?
The same recipe at Re 30, 41, 80, 90, at 15k and 100k.

It does not. Re = 73's 0% bias is luck: Re 80 and 90 both give +29% with 8×
the seed spread. At Re 30 and 41, **the two seeds disagree on the sign** — the
model cannot say whether the flow is stable. σ drifted upward in **8 of 8** runs
from 15k to 100k, and at 100k both Re = 41 seeds report a strongly unstable flow
where the truth is −0.0158.

Implied critical Reynolds number, from a line through the two Re nearest onset:

| | predicted Re_c |
|---|---:|
| truth | **44.15** |
| 15k models | 31.7 |
| 100k models | 25.3 |

### `5_cylinder_node_drop_a1.ipynb` — does the state representation matter?
Mode-count and mode-choice ablations below onset. Six state dimensions at
Re = 30 (7, 8, 10, 12, 14, 16 modes) and five at Re = 41 (1, 4, 5, 7, 8, 16).

Accuracy is **flat** across the mode-count axis — every configuration at Re = 30
recovers 0–25% of the expected decay rate — while precision swings 11× with no
pattern. Truncation is not the binding constraint. A 1-mode model is included to
make the floor explicit: a 1D autonomous ODE cannot oscillate at all, since
`ż = f(z)` gives one velocity per state and an oscillation needs the same state
to be traversed in both directions.

### `7_sindy_vs_node.ipynb` — SINDy as a baseline
Head-to-head against SINDy at one stable Reynolds number (36) and one on the
limit cycle (73), measuring only the growth rate at the origin and the
trajectory. SINDy fits `ż = A z + Q(z, z)`, the linear-plus-quadratic library
that Galerkin projection of Navier-Stokes gives exactly, with no constant term so
`z = 0` stays a fixed point. Both models are fitted on one Reynolds number at a
time, so neither sees data the other does not.

At Re = 73 the two are **level** on the growth rate, +0.1296 against +0.1294,
both about 43% high. At Re = 36 SINDy reports a growing mode for a flow that
visibly decays, at every sample weighting, while the neural ODE gets the stable
case to 4%. On trajectories it is not close: the neural ODE completes all four
runs, SINDy diverges on all four, typically within 60 of 428 steps. Adding the
cubic library makes both worse, 219% and 32 of 349 steps at Re = 73.

Fits are cached in `notebooks/sindy_fits.pkl`, produced by
`scripts/sindy_vs_node.py`, with `scripts/sindy_structure.py` for the cubic
library and `scripts/psindy.py` for the shared fitting routines.

---

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install jax numpy scipy matplotlib jupyter nbconvert
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/2_cylinder_node_single_Re.ipynb
```

All fits are cached in `notebooks/node_fits.pkl`, so the notebooks re-execute in
seconds without retraining. Delete it to force a genuine retrain (hours).

`scripts/growth.py` regenerates the ground truth σ(Re) and Re_c from the probe
signals, and runs out of the box.

### Parallel sweeps

`sweep/` runs many independent fits at once. One training run uses ~1.2 cores, so
a serial sweep leaves most of a machine idle; eight single-threaded processes give
a measured **6.15×** throughput.

```bash
.venv/bin/python sweep/launch.py --modes 14 --steps 15000 --seeds 0 1 2 3 4 --workers 8
.venv/bin/python sweep/merge.py        # folds results into notebooks/node_fits.pkl
```

Each job writes its own file and skips if it already exists, so the sweep is
resumable. `--dry-run` previews the job list and a wall-clock estimate.

---

## Traps worth knowing

Costly mistakes made and fixed during this work.

1. **Scale the latent state by one global factor, never per-mode.** Per-mode
   normalisation inflates the `a₆` noise mode 45× and puts σ off by +111%.
2. **Select the leading *oscillatory* eigenvalue, not `argmax(Re λ)`.** A Hopf
   bifurcation is a complex pair crossing the axis. `scripts/psindy.py:146` has
   this bug, so the SINDy baseline's `critical_Re` may report a non-Hopf crossing.
3. **Under `@jax.jit`, values read from enclosing scope are frozen at trace
   time.** Pass training data as an argument, or a cross-Reynolds check will
   silently retrain on the wrong data.
4. `jnp.zeros(R)` is float64 under x64; use `jnp.zeros(R, z.dtype)`.
5. **Models sometimes invent a second attractor**, or diverge, while their origin
   Jacobian looks perfectly healthy. Always check a free rollout.
6. **Re = 30 has no measured σ.** `growth.py` returns NaN there — the decay dies
   before three periods. Anything quoted at Re = 30 is scored against an
   extrapolation, and the linear and quadratic forms disagree by 51%.
