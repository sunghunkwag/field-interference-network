"""
fin/hdc/hypervector.py
HyperdimensionalEncoder — lifts continuous field state into
high-dimensional bipolar hypervectors and supports HDC algebra.

Mathematical Foundations
------------------------
Random-Fourier-Feature encoding (bipolar version):

  hv(x) = sign( cos( Ω x + b ) )           (1)

  Ω ~ N(0, σ²) ∈ R^D,   b ~ U(0, 2π) ∈ R^D

Expected inner product property:
  E[ hv(x) · hv(y) ] / D ≈ cos(σ |x−y|)   (2)

This gives a distance-preserving probabilistic embedding.

HDC Algebra
-----------
Binding    (×): bind(a, b) = a ⊙ b    element-wise product
               Binding is invertible: a ⊙ b ⊙ b = a  (bipolar ±1)

Bundling   (∑): bundle(hvs) = sign( Σ_i hvs_i )
               Bundled vector is similar to all members.

Similarity:    sim(a, b) = ⟨a, b⟩ / D  ∈ [−1, +1]

Field application
-----------------
For element i:
  hv_i = bind( encode_ω(ω_i), encode_φ(φ_i) )   [N, D]

Global state:
  HV_field = bundle( {hv_i} )                    [D]

This constructs a neuro-symbolic global memory of the entire field.
"""

import math
import torch
import torch.nn as nn


class HypervectorCodebook(nn.Module):
    """Random-Fourier-Feature bipolar codebook for one scalar domain."""

    def __init__(self, hdc_dim: int = 1024, sigma: float = 1.0, learnable: bool = False):
        super().__init__()
        self.D = hdc_dim
        omega = torch.randn(hdc_dim) * sigma
        bias  = torch.rand(hdc_dim)  * 2.0 * math.pi
        if learnable:
            self.omega = nn.Parameter(omega)
            self.bias  = nn.Parameter(bias)
        else:
            self.register_buffer("omega", omega)
            self.register_buffer("bias",  bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x : [N] → [N, D] ∈ {−1, +1}"""
        proj = x.view(-1, 1) * self.omega + self.bias   # [N, D]
        return torch.sign(torch.cos(proj))


def bind(hv_a: torch.Tensor, hv_b: torch.Tensor) -> torch.Tensor:
    """Element-wise bipolar binding: bind(a,b) ⊙ b = a (self-inverse)."""
    return hv_a * hv_b


class HDCFieldEncoder(nn.Module):
    """
    Encodes per-element (ω, φ) pairs into composite hypervectors.
    hv_field_i = bind( encode_ω(ω_i), encode_φ(φ_i) )
    """

    def __init__(self, hdc_dim: int = 1024, sigma: float = 1.0):
        super().__init__()
        self.D         = hdc_dim
        self.omega_cb  = HypervectorCodebook(hdc_dim, sigma)
        self.phi_cb    = HypervectorCodebook(hdc_dim, sigma)

    @staticmethod
    def bind(hv_a: torch.Tensor, hv_b: torch.Tensor) -> torch.Tensor:
        """Element-wise bipolar binding: bind(a,b) ⊙ b = a (self-inverse)."""
        return hv_a * hv_b

    def encode_elements(self, omega: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """omega, phi : [N] → [N, D]"""
        return self.omega_cb.encode(omega) * self.phi_cb.encode(phi)

    @staticmethod
    def bundle(hvs: torch.Tensor) -> torch.Tensor:
        """hvs : [N, D] → [D]  majority-vote superposition"""
        return torch.sign(hvs.sum(0))

    @staticmethod
    def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a, b : [D] → scalar ∈ [−1, +1]"""
        return (a * b).sum() / a.shape[0]

    def field_state_vector(self, omega: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Produce a single global-field hypervector. → [D]"""
        return self.bundle(self.encode_elements(omega, phi))
