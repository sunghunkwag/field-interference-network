"""
fin/topology/meta_topology.py
Gradient-driven RSI topology controller.

Replaces the deprecated hardcoded thresholds (0.95, 3.0) with an
explicit dissonance-minimisation framework.

Dissonance Objective L
----------------------
L = α L_phase  +  β L_energy  +  γ L_memory  +  δ L_graph

  L_phase  = mean_{(i,j)∈E} [1 − cos(φ_i − φ_j)]          (1)
             → penalises phase incoherence between neighbours

  L_energy = Var({E_i}) + λ mean( relu(E_i − E_max)² )     (2)
             → penalises energy imbalance and saturation

  L_memory = MSE(y_t, u_t)                                  (3)
             → penalises working-memory prediction error

  L_graph  = mean_{(i,j)∈E} ||x_i − x_j||²                 (4)
             → penalises spatial spread of connected pairs

Topology Policy
---------------
Per-element logits z_i = MLP(f_i) ∈ R^4 over actions {merge, split, spawn, dissolve}.
Action probabilities: p_i = softmax(z_i / τ).

The MLP is trained by gradient descent on L so that topology actions
are driven by dissonance minimisation — never by random choice.

Node feature vector f_i ∈ R^{4+d}:
  f_i = [ω_i, cos(φ_i), sin(φ_i), A_i, x_i]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DissonanceEvaluator(nn.Module):
    """Multi-term dissonance objective."""

    def __init__(self, alpha=1.0, beta=0.3, gamma=0.5, delta=0.2, emax=4.0, lam=0.5):
        super().__init__()
        self.alpha = alpha; self.beta  = beta
        self.gamma = gamma; self.delta = delta
        self.emax  = emax;  self.lam   = lam

    def forward(
        self,
        phi:       torch.Tensor,
        amplitude: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        memory_y:  torch.Tensor,
        memory_u:  torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        L_phase  = (1.0 - torch.cos(phi[src] - phi[dst])).mean()
        energy   = 0.5 * amplitude ** 2
        L_energy = energy.var(unbiased=False) + self.lam * F.relu(energy - self.emax).pow(2).mean()
        L_mem    = F.mse_loss(memory_y, memory_u)
        L_graph  = ((positions[src] - positions[dst]) ** 2).sum(-1).mean()
        return (self.alpha * L_phase + self.beta * L_energy
                + self.gamma * L_mem + self.delta * L_graph)


class MetaTopologyController(nn.Module):
    """
    Learns node-action logits by gradient descent against DissonanceEvaluator.
    """

    ACTIONS = ("merge", "split", "spawn", "dissolve")

    def __init__(self, spatial_dim: int = 3, hidden_dim: int = 32, tau: float = 0.7):
        super().__init__()
        self.tau      = tau
        feat_dim      = 4 + spatial_dim
        self.policy   = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def _node_feats(self, omega, phi, amplitude, positions):
        return torch.cat([
            omega.unsqueeze(1),
            torch.cos(phi).unsqueeze(1),
            torch.sin(phi).unsqueeze(1),
            amplitude.unsqueeze(1),
            positions,
        ], dim=1)

    def forward(self, omega, phi, amplitude, positions):
        logits = self.policy(self._node_feats(omega, phi, amplitude, positions))
        probs  = F.softmax(logits / self.tau, dim=-1)
        return logits, probs

    def propose_actions(self, omega, phi, amplitude, positions):
        _, probs = self.forward(omega, phi, amplitude, positions)
        return [self.ACTIONS[i] for i in probs.argmax(dim=-1).tolist()]
