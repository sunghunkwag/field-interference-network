"""
fin/memory/ssm_memory.py
StableSSMMemory — continuous-time state-space working memory.

Replaces ad-hoc stepwise propagation with a mathematically grounded
linear time-invariant SSM that provides persistent working memory.

Mathematical Formulation
------------------------
Hidden state h(t) ∈ R^m, input u(t) ∈ R^p:

  dh/dt = A h(t) + B u(t)        (1)  — state equation
  y(t)  = C h(t) + D u(t)        (2)  — output equation

Euler discretisation over timestep Δt:
  h_{t+1} = (I + Δt A) h_t + Δt B u_t   (3)

Stability parameterisation of A
--------------------------------
A must have strictly negative real eigenvalues for bounded memory.
We parameterise:

  A = diag(−softplus(λ)) + U V^T             (4)

where λ ∈ R^m is learnable (softplus ensures positivity of the
negative diagonal), and U, V ∈ R^{m×r} is a learnable low-rank
perturbation (r=4) to allow off-diagonal dynamics without
risking instability from a full unconstrained A.

Input summary u_t
-----------------
At each step we summarise N-element field state as:
  u_t = [mean(A), mean(cos φ), mean(sin φ), mean(ω), std(φ), E_total] ∈ R^6

The sin/cos decomposition of phase avoids the 2π discontinuity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StableSSMMemory(nn.Module):

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, rank: int = 4):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.lambda_diag = nn.Parameter(torch.zeros(hidden_dim))
        self.U = nn.Parameter(torch.randn(hidden_dim, rank) * 0.05)
        self.V = nn.Parameter(torch.randn(hidden_dim, rank) * 0.05)

        self.B = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.05)
        self.C = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.05)
        self.D = nn.Parameter(torch.eye(input_dim) * 0.1)

    def A_matrix(self) -> torch.Tensor:
        return torch.diag(-F.softplus(self.lambda_diag)) + self.U @ self.V.T

    def summarise(self, omega: torch.Tensor, phi: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
        """Build 6-d input vector from field state tensors."""
        return torch.stack([
            amplitude.mean(),
            torch.cos(phi).mean(),
            torch.sin(phi).mean(),
            omega.mean(),
            phi.std(unbiased=False),
            (0.5 * amplitude ** 2).sum(),
        ])

    def step(
        self,
        h: torch.Tensor,
        omega: torch.Tensor,
        phi: torch.Tensor,
        amplitude: torch.Tensor,
        dt: float,
    ):
        """Single Euler-integrated SSM step.
        Returns (h_next [H], y [input_dim], u [input_dim])."""
        A = self.A_matrix()
        u = self.summarise(omega, phi, amplitude)
        h_next = h + dt * (A @ h + self.B @ u)
        y = self.C @ h_next + self.D @ u
        return h_next, y, u

    def rollout(
        self,
        omega_seq: torch.Tensor,
        phi_seq:   torch.Tensor,
        amp_seq:   torch.Tensor,
        dt: float,
        h0: torch.Tensor = None,
    ):
        """
        Roll out T steps.
        Inputs : [T, N]  →  h_hist [T,H], y_hist [T,p], u_hist [T,p]
        """
        T      = omega_seq.shape[0]
        device = omega_seq.device
        h = torch.zeros(self.hidden_dim, device=device) if h0 is None else h0
        hs, ys, us = [], [], []
        for t in range(T):
            h, y, u = self.step(h, omega_seq[t], phi_seq[t], amp_seq[t], dt)
            hs.append(h); ys.append(y); us.append(u)
        return torch.stack(hs), torch.stack(ys), torch.stack(us)
