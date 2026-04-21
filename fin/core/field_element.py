"""
fin/core/field_element.py
FieldElement — differentiable oscillating field state.

Each element i carries:
  ω_i  : angular frequency  (learnable)
  φ_i  : phase              (learnable)
  A_i  : amplitude          (learnable)
  x_i  : spatial position   (learnable)

Instantaneous field value:
  u_i(t) = A_i · cos(ω_i · t + φ_i)

Instantaneous energy:
  E_i   = 0.5 · A_i²

All state tensors are nn.Parameter so that gradient-based
meta-learning and topology morphing can flow gradients through them.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class FieldConfig:
    n_elements : int  = 128
    spatial_dim: int  = 3
    hdc_dim    : int  = 1024
    dtype      : torch.dtype = torch.float32
    device     : str  = "cpu"


class FieldElement(nn.Module):
    """
    Batch of N oscillating field elements.
    State tensors (all shape [N] except position which is [N, d]):
      omega, phi, amplitude, position
    """

    def __init__(self, cfg: FieldConfig):
        super().__init__()
        self.cfg = cfg
        N, d, dev, dt = cfg.n_elements, cfg.spatial_dim, cfg.device, cfg.dtype

        self.omega     = nn.Parameter(torch.rand(N, dtype=dt, device=dev) * 2.0 * torch.pi + 0.1)
        self.phi       = nn.Parameter(torch.rand(N, dtype=dt, device=dev) * 2.0 * torch.pi)
        self.amplitude = nn.Parameter(torch.ones(N, dtype=dt, device=dev))
        self.position  = nn.Parameter(torch.rand(N, d, dtype=dt, device=dev) * 2.0 - 1.0)

    def field_value(self, t: torch.Tensor) -> torch.Tensor:
        """u_i(t) = A_i cos(ω_i t + φ_i).  Returns [N] for scalar t or [N,T] for [T]."""
        t_vec  = t.view(-1)
        phase  = torch.outer(self.omega, t_vec) + self.phi.unsqueeze(1)   # [N, T]
        vals   = self.amplitude.unsqueeze(1) * torch.cos(phase)
        return vals.squeeze(1) if t.dim() == 0 else vals

    def energy(self) -> torch.Tensor:
        """E_i = 0.5 A_i².  Shape [N]."""
        return 0.5 * self.amplitude ** 2

    def total_energy(self) -> torch.Tensor:
        return self.energy().sum()
