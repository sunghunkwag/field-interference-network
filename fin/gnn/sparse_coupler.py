"""
fin/gnn/sparse_coupler.py
SparseResonanceCoupler — O(n·k) GNN-style phase coupling.

Replaces the original O(n²) dense ResonanceCoupler.

Complexity Analysis
-------------------
  Dense coupling: all-pairs sin(φ_j − φ_i)      →  O(n²)
  Sparse k-NN:    restricted to k neighbours/node →  O(n·k)
  For k ~ log n this gives O(n log n).
  For fixed small k (e.g. k=8) this is O(n).

Phase Update Rule (Kuramoto restricted to graph G)
--------------------------------------------------
  Δφ_i = Σ_{j∈N(i)}  w_{ij} · mlp([φ_j − φ_i, w_{ij}])
                               · sin(φ_j − φ_i)

The learnable message MLP allows the model to modulate each
edge's contribution beyond a plain sine, while preserving the
antisymmetric structure needed for coherent phase synchronisation.

Graph construction
------------------
k-NN graph is built once from spatial positions via pairwise L2
distance, then cached.  It is rebuilt lazily whenever positions
change significantly (call build_graph explicitly after morphing).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def _build_knn_graph(positions: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    positions: [N, d]  →  edge_index [2, N*k], edge_dist [N*k]
    """
    N  = positions.shape[0]
    k  = min(k, N - 1)
    diff   = positions.unsqueeze(1) - positions.unsqueeze(0)      # [N, N, d]
    dist2  = (diff ** 2).sum(-1)                                   # [N, N]
    dist2.fill_diagonal_(float("inf"))
    topk_d2, topk_idx = torch.topk(dist2, k, dim=1, largest=False)
    src = torch.arange(N, device=positions.device).unsqueeze(1).expand(N, k).reshape(-1)
    dst = topk_idx.reshape(-1)
    return torch.stack([src, dst], dim=0), topk_d2.sqrt().reshape(-1)


class SparseResonanceCoupler(nn.Module):
    def __init__(self, n_elements: int, k_neighbors: int = 8, n_layers: int = 2):
        super().__init__()
        self.n_elements  = n_elements
        self.k_neighbors = k_neighbors
        self.n_layers    = n_layers

        E = n_elements * k_neighbors
        self.edge_weights = nn.Parameter(torch.full((E,), 0.1))

        self.msg_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

        self.register_buffer("edge_index", torch.zeros(2, E, dtype=torch.long))
        self._graph_built = False

    def build_graph(self, positions: torch.Tensor) -> None:
        ei, _ = _build_knn_graph(positions.detach(), self.k_neighbors)
        E_new = ei.shape[1]
        if E_new != self.edge_index.shape[1]:
            self.edge_weights = nn.Parameter(
                torch.full((E_new,), 0.1, device=positions.device)
            )
            self.register_buffer(
                "edge_index", torch.zeros(2, E_new, dtype=torch.long, device=positions.device)
            )
        self.edge_index.copy_(ei)
        self._graph_built = True

    def forward(self, phi: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        phi       : [N]    current phases
        positions : [N, d] spatial coords
        Returns delta_phi : [N]
        """
        if not self._graph_built:
            self.build_graph(positions)

        src, dst = self.edge_index[0], self.edge_index[1]
        delta_phi_edge = phi[dst] - phi[src]
        w = F.softplus(self.edge_weights)
        msg_in = torch.stack([delta_phi_edge, w], dim=1)
        coeff  = self.msg_mlp(msg_in).squeeze(1)
        messages = coeff * torch.sin(delta_phi_edge)

        delta_phi = torch.zeros_like(phi)
        delta_phi.scatter_add_(0, dst, messages)
        return delta_phi
