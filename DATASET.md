# Parametric neural ODE for the cylinder-wake Hopf bifurcation

Everything needed to start. Read `report/cylinder_report.pdf` first — it covers
the problem, the system and the proposed modelling approach, and everything
below assumes it.

## The one-line version

2D flow past a cylinder undergoes a supercritical Hopf bifurcation: below a
critical Reynolds number the wake is a steady fixed point, above it a limit
cycle. 17 Reynolds numbers were simulated across that transition. The task is to
fit a single model `ż = f_θ(z, Re)` to all of them and see whether it recovers
the bifurcation at Reynolds numbers it never trained on.

**Measured ground truth: Re_c = 44.2 ± 0.3, dσ/dRe = 0.0050 ± 0.0004.**

## What's in here

```
report/   cylinder_report.pdf   READ THIS FIRST
          REPORT.md             same, as markdown
          fig_*.png             figures

data/     latent_r{3,8,16}.npz  the training data: POD coordinates, ragged,
                                with trajectory ids  -> use for multiple shooting
          tensor_r{3,8,16}.npz  same in (p, m, r) tensor form, uniform dt
          pod_basis.npz         the 64-mode shared POD basis
          Re*.npz               the full dataset: steady base flow, velocity
                                perturbation fields, times, probe signals

plots/    17 directories, one per Reynolds number, 8 coordinate plots each
          _bifurcation.png      sweep summary
          README.md             index table

scripts/  lbm.py          the solver (D2Q9 lattice Boltzmann, TRT)
          sweep.py        regenerates the whole dataset (~2.5 h on a GPU)
          growth.py       growth rates and Re_c        <- runs out of the box
          psindy.py       parametric SINDy baseline    <- runs out of the box
          project.py      builds the POD basis and latent tables *
          analyse.py      amplitudes, Strouhal, POD rank *
          plot_modes.py   the per-Reynolds plots *
          show_fields.py  vorticity snapshots *
          validate.py     solver sanity checks
          build_pdf.py    rebuilds the report PDF
```

`*` needs the raw velocity fields, which **are** included. Nothing here needs
regenerating — `sweep.py` is provided so you can see how the data was made, and
so you can add extra Reynolds numbers if you want them, not because you need to
run it. A full sweep takes about 2.5 hours on an M-series GPU.

The latent files are those fields already projected onto the shared POD basis;
use them for the modelling. Go back to the fields if you want a different rank,
a different basis, or flow-field figures.

## Start here

```bash
cd scripts
python -u growth.py --data ../data      # sigma(Re), Re_c -- reproduces 44.2
python -u psindy.py --data ../data      # the SINDy baseline you have to beat
```

Then open `plots/Re041p0/a2.png` and `plots/Re045p0/a2.png` side by side. The
lower panel of each is the log envelope: it slopes down in one and up in the
other. That sign change **is** the bifurcation. Onset sits between them.

## What the data actually is

Each Reynolds number has one trajectory, started at the steady base flow plus a
small kick, recorded until it reaches the attractor. Snapshots are stored as the
**perturbation about that Reynolds number's own steady base flow**, so `z = 0`
is an exact fixed point at every Re. That matters: it means a parametric model
only has to learn how the linearisation at the origin moves with Re, and you can
enforce `f_θ(0, Re) = 0` architecturally rather than hoping it is learned.

Coordinates are POD coefficients in a basis shared across the whole sweep.
`a1` is the shift mode (mean-flow deformation, slow, not an oscillation);
`a2` onward are oscillatory at the shedding frequency. `r = 8` is the
recommended latent width — `r = 3` is fine near onset but under-represents the
high-Re end.

## Four things that will bite you

**1. Do not validate on long-horizon rollout MSE.** A limit cycle is a circle of
states. Any tiny frequency error accumulates as phase drift, so rollout error
saturates at the diameter of the attractor even for a perfect model. Train on
short multiple-shooting windows; validate on attractor geometry — amplitude,
frequency, Poincaré section — and on the Jacobian eigenvalues.

**2. Trajectory-fit quality is blind to the bifurcation.** Near onset
σ/ω ≈ 0.005, so the growth rate is ~2×10⁻⁵ of the derivative variance while the
best achievable 1−R² is ~1.8×10⁻³ — the stability signal sits ~75× *below* the
residual floor. Selecting hyperparameters on held-out R² doesn't merely fail to
find the bifurcation, it actively anti-selects (Spearman −0.83). This is the
central obstacle of the project and the most interesting thing in it. The fix
that works: supervise the stability *sign* against the growth or decay directly
visible in the training trajectories.

**3. The subcritical cases drift.** The stored base flows converged to ~1.5×10⁻⁵,
so the recorded perturbation at Re = 30, 36, 41 contains a slow ramp along the
non-oscillatory directions that dominates once the oscillation has decayed. Only
the first ~20% of those runs is clean (Re = 45: first ~50%). Re ≥ 48 is
unaffected — drift is 0.1–1.8% of signal. Truncate or drop the low-Re cases.

**4. Never quote a bifurcation number without a spread.** Bootstrap it,
leave-one-Reynolds-out, and inject noise. The SINDy baseline looks accurate at
one hyperparameter setting and loses the bifurcation entirely under 72% of
physics-preserving perturbations.

## The SINDy baseline

`psindy.py` fits `ż = A(Re) z + Q(z,z)`, the structure Galerkin projection of
Navier–Stokes guarantees. It is included because you have to beat it, and
because its failure modes are instructive. Honest summary:

- It genuinely detects a bifurcation — 0/300 shuffled-Reynolds-label refits
  produce any crossing, and it does not track a moved held-out gap.
- It does **not** locate one reliably. A quadratic in 1/Re through the measured
  growth rates gives 44.16, better than any SINDy variant.
- No fit-quality-based selection criterion finds the right model (see point 2).

## What would count as beating it

Fix these before fitting anything, and do not change them afterwards:

- `|Re_c − 44.2| < 0.5`, with drop-one-trajectory + noise spread `< 1.0`
- correct stability sign at **all** training and held-out Reynolds numbers, with
  a single zero crossing
- **all four** held-out growth rates within 25%, no best-of-N quoting
- beat 44.16 (the three-parameter curve fit through measured growth rates)
- pass the null battery: shuffled Reynolds labels → no crossing; degenerate
  library → no crossing; supercritical-only training → no crossing
- training on all 17 Reynolds numbers must *improve* the estimate
