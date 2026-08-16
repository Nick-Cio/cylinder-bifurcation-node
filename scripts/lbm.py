"""
D2Q9 lattice-Boltzmann solver for 2D flow past a circular cylinder.

The whole Reynolds sweep is advanced as ONE batched tensor
------------------------------------------------------------
Every case in the sweep shares a geometry and differs only in the relaxation
times, so the state is held as (B, 9, ny, nx) and a step is the same handful of
kernels regardless of B.  On a GPU a single 265k-cell case is entirely
launch-bound -- roughly 200 steps/s, which is ~1% of the hardware -- so
batching 17 Reynolds numbers costs almost nothing and turns a four-hour sweep
into a few minutes.  Streaming is a single gather rather than nine rolls for
the same reason.

Why TRT rather than BGK
-----------------------
Two-relaxation-time collision with the "magic" parameter

    Lambda = (tau_p - 1/2)(tau_m - 1/2) = 3/16

places the bounce-back wall exactly halfway between fluid and solid nodes
*independently of viscosity*.  Under plain BGK the effective wall position
drifts with tau, so the effective diameter -- and hence the effective Reynolds
number -- changes across the sweep.  That is fatal for a study whose entire
point is locating a bifurcation in Re.  TRT is also stable much closer to
tau = 1/2, which the high-Re end needs.

Geometry and the base flow
--------------------------
Lateral boundaries are free-slip (specular reflection).  A free-slip wall *is*
a symmetry plane, so one solver gives two things:

  * `half=False` -- cylinder centred in the channel, vortex street free to grow;
  * `half=True`  -- cylinder centre placed *on* the bottom boundary, so only the
    upper half plane is simulated.  The von Karman mode is antisymmetric in the
    transverse velocity, so it cannot live in this subspace and the run relaxes
    to the *steady* solution -- the unstable base flow -- at any Reynolds
    number, however far above onset.

That base flow is what makes the bifurcation identifiable from data: starting a
full-domain run just off it records the whole spiral-out onto the limit cycle
instead of only the attractor.

Units
-----
All internal quantities are lattice units.  D is the diameter in cells, u_in
the inflow speed in cells/step, one convective time D/u_in is `steps_per_ct`
steps, and nu = u_in * D / Re.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# ---------------------------------------------------------------- D2Q9 lattice
CX = [0, 1, 0, -1, 0, 1, -1, -1, 1]
CY = [0, 0, 1, 0, -1, 1, 1, -1, -1]
W = [4 / 9] + [1 / 9] * 4 + [1 / 36] * 4
OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6]

# directions with cy > 0, and the partners they specularly reflect from
UP, UP_REF = [2, 5, 6], [4, 8, 7]
DN, DN_REF = [4, 7, 8], [2, 6, 5]


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class Geometry:
    """Domain in units of the cylinder diameter D."""

    D: int = 20
    up: float = 8.0            # inlet to cylinder centre
    down: float = 22.0         # cylinder centre to outlet
    half_width: float = 10.0   # centre to lateral boundary

    def grid(self, half: bool):
        nx = int(round((self.up + self.down) * self.D)) + 1
        ny = int(round(self.half_width * self.D)) + 1
        if not half:
            ny = 2 * ny - 1
        return nx, ny, int(round(self.up * self.D)), (0 if half else ny // 2)

    @property
    def blockage(self) -> float:
        return 1.0 / (2 * self.half_width)


class CylinderBatch:
    """A batch of cylinder cases sharing a geometry, differing only in Re."""

    def __init__(self, Re, geom: Geometry = Geometry(), u_in: float = 0.06,
                 half: bool = False, device="auto", magic: float = 3.0 / 16.0):
        self.Re = np.atleast_1d(np.asarray(Re, dtype=np.float64))
        self.geom, self.u_in, self.half = geom, u_in, half
        self.dev = d = pick_device(device)
        self.B = len(self.Re)
        self.nx, self.ny, self.cx, self.cy = geom.grid(half)

        self.nu = u_in * geom.D / self.Re
        tau_p = 3.0 * self.nu + 0.5
        tau_m = 0.5 + magic / (tau_p - 0.5)
        self.tau_p_np, self.tau_m_np = tau_p, tau_m
        v = lambda a: torch.as_tensor(a, dtype=torch.float32,
                                      device=d).view(self.B, 1, 1, 1)
        self.tau_p, self.tau_m = v(tau_p), v(tau_m)
        self.steps_per_ct = geom.D / u_in

        k = lambda a: torch.tensor(a, dtype=torch.float32, device=d).view(1, 9, 1, 1)
        self.w, self.cxf, self.cyf = k(W), k(CX), k(CY)
        self.opp = torch.tensor(OPP, dtype=torch.int64, device=d)

        # ---- solid mask: simple bounce-back nodes inside the disc
        ys, xs = torch.meshgrid(torch.arange(self.ny, device=d),
                                torch.arange(self.nx, device=d), indexing="ij")
        r2 = (xs - self.cx).float() ** 2 + (ys - self.cy).float() ** 2
        self.solid = (r2 <= (geom.D / 2.0) ** 2).view(1, 1, self.ny, self.nx)

        # ---- streaming as one gather: out[i, y, x] = post[i, y-cy_i, x-cx_i]
        yy, xx = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing="ij")
        idx = np.stack([((yy - CY[i]) % self.ny) * self.nx + ((xx - CX[i]) % self.nx)
                        for i in range(9)])
        self.idx = torch.as_tensor(idx, dtype=torch.int64, device=d).view(1, 9, -1)

        col = torch.ones((1, 1, self.ny, 1), device=d)
        self._f_inlet = self.equilibrium(col, u_in * col,
                                         torch.zeros_like(col)).contiguous()
        self.f = None
        self.reset()

    # -------------------------------------------------------------- kernels
    def equilibrium(self, rho, ux, uy):
        cu = 3.0 * (self.cxf * ux + self.cyf * uy)
        return self.w * rho * (1.0 + cu + 0.5 * cu * cu
                               - 1.5 * (ux * ux + uy * uy))

    def macroscopic(self):
        f = self.f
        rho = f.sum(1, keepdim=True)
        ux = (self.cxf * f).sum(1, keepdim=True) / rho
        uy = (self.cyf * f).sum(1, keepdim=True) / rho
        z = torch.zeros_like(ux)
        return rho, torch.where(self.solid, z, ux), torch.where(self.solid, z, uy)

    def reset(self, ux=None, uy=None, rho=None):
        """Initialise at equilibrium; defaults to uniform inflow everywhere."""
        d, sh = self.dev, (self.B, 1, self.ny, self.nx)
        t = lambda a, fill: (torch.full(sh, fill, device=d) if a is None else
                             torch.as_tensor(a, dtype=torch.float32,
                                             device=d).reshape(-1, 1, self.ny,
                                                               self.nx).expand(sh))
        rho, ux, uy = t(rho, 1.0), t(ux, self.u_in), t(uy, 0.0)
        z = torch.zeros_like(ux)
        ux, uy = torch.where(self.solid, z, ux), torch.where(self.solid, z, uy)
        self.f = self.equilibrium(rho, ux, uy).contiguous()

    def step(self):
        f = self.f
        rho = f.sum(1, keepdim=True)
        ux = (self.cxf * f).sum(1, keepdim=True) / rho
        uy = (self.cyf * f).sum(1, keepdim=True) / rho
        feq = self.equilibrium(rho, ux, uy)

        # TRT collision: relax symmetric and antisymmetric parts separately
        fo, eo = f[:, self.opp], feq[:, self.opp]
        post = (f - (0.5 * (f + fo) - 0.5 * (feq + eo)) / self.tau_p
                  - (0.5 * (f - fo) - 0.5 * (feq - eo)) / self.tau_m)

        # bounce-back on solid nodes (pre-collision populations, reversed)
        post = torch.where(self.solid, fo, post)

        # streaming (single gather)
        out = post.reshape(self.B, 9, -1) \
                  .gather(2, self.idx.expand(self.B, 9, -1)) \
                  .view(self.B, 9, self.ny, self.nx)

        # lateral free-slip == symmetry plane
        out[:, UP, 0, :] = out[:, UP_REF, 0, :]
        out[:, DN, -1, :] = out[:, DN_REF, -1, :]

        out[:, :, :, 0:1] = self._f_inlet       # inlet: uniform equilibrium
        out[:, :, :, -1] = out[:, :, :, -2]     # outlet: zero gradient
        self.f = out

    def run(self, n_steps: int, probe_every: int = 0, probe=None):
        for k in range(n_steps):
            self.step()
            if probe_every and probe is not None and (k + 1) % probe_every == 0:
                probe(k + 1, self)
        return self

    # ---------------------------------------------------------- diagnostics
    def wake_probe(self):
        """Transverse velocity at a fixed off-axis point 2D behind the body."""
        _, _, uy = self.macroscopic()
        return uy[:, 0, self.cy + self.geom.D // 4,
                  self.cx + 2 * self.geom.D].cpu().numpy().copy()

    def is_finite(self):
        return torch.isfinite(self.f).all(dim=(1, 2, 3)).cpu().numpy()

    @staticmethod
    def mirror(a: np.ndarray, odd: bool = False) -> np.ndarray:
        """Half-domain field (row 0 is the symmetry plane) -> full domain.

        `odd=True` negates the reflected half -- the correct parity for the
        transverse velocity uy.  ux and rho are even.
        """
        top = a[..., 1:, :][..., ::-1, :]
        return np.concatenate([-top if odd else top, a], axis=-2)

    def __repr__(self):
        return (f"CylinderBatch(B={self.B}, Re={self.Re.min():g}..{self.Re.max():g},"
                f" {self.nx}x{self.ny}, D={self.geom.D}, half={self.half}, "
                f"tau_p={self.tau_p_np.min():.4f}..{self.tau_p_np.max():.4f}, "
                f"{self.steps_per_ct:.0f} steps/ct, {self.dev.type})")
