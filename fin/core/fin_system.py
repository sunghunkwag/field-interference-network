"""
fin/core/fin_system.py
FINSystem — integrated orchestration of all upgraded subsystems.

Per-step field dynamics
-----------------------
  φ_{t+1} = φ_t + Δt [ ω_t + C_sparse(φ_t, x_t) ]          (1)
  A_{t+1} = A_t + Δt [ −η A_t + κ tanh(P_mem y_t) ]         (2)

where:
  C_sparse  : sparse GNN Kuramoto coupling term
  P_mem     : linear projection from SSM output y_t → [N] modulation
  η         = softplus(amplitude_decay) > 0  (ensures positivity)
  κ         = softplus(memory_gain)    > 0

Equation (2) drives amplitude evolution: natural exponential decay
(−η A_t) restores resting amplitude, while the SSM modulation
(+κ tanh(·)) pushes amplitudes toward a memory-informed target.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fin.core.field_element import FieldElement, FieldConfig
from fin.gnn.sparse_coupler  import SparseResonanceCoupler
from fin.hdc.hypervector      import HDCFieldEncoder
from fin.memory.ssm_memory    import StableSSMMemory
from fin.topology.meta_topology import DissonanceEvaluator, MetaTopologyController


class FINSystem(nn.Module):

    def __init__(
        self,
        n_elements : int = 128,
        spatial_dim: int = 3,
        hdc_dim    : int = 1024,
        memory_dim : int = 64,
        k_neighbors: int = 8,
        device     : str = "cpu",
    ):
        super().__init__()
        cfg = FieldConfig(n_elements=n_elements, spatial_dim=spatial_dim,
                          hdc_dim=hdc_dim, device=device)
        self.fields     = FieldElement(cfg)
        self.coupler    = SparseResonanceCoupler(n_elements, k_neighbors=k_neighbors)
        self.hdc        = HDCFieldEncoder(hdc_dim=hdc_dim)
        self.memory     = StableSSMMemory(input_dim=6, hidden_dim=memory_dim)
        self.dissonance = DissonanceEvaluator()
        self.topology   = MetaTopologyController(spatial_dim=spatial_dim)
        self.mem_proj   = nn.Linear(6, n_elements)
        self._amp_decay = nn.Parameter(torch.tensor(0.0))
        self._mem_gain  = nn.Parameter(torch.tensor(0.0))

    @property
    def amplitude_decay(self):
        return F.softplus(self._amp_decay)

    @property
    def memory_gain(self):
        return F.softplus(self._mem_gain)

    def step(self, h: torch.Tensor, dt: float = 0.05) -> dict:
        phi   = self.fields.phi
        omega = self.fields.omega
        amp   = self.fields.amplitude
        pos   = self.fields.position

        coupling = self.coupler(phi, pos)
        h_next, y, u = self.memory.step(h, omega, phi, amp, dt)

        mod     = torch.tanh(self.mem_proj(y))
        new_phi = phi + dt * (omega + coupling)
        new_amp = amp + dt * (-self.amplitude_decay * amp + self.memory_gain * mod)
        new_amp = torch.clamp(new_amp, min=1e-3)

        self.fields.phi.data = new_phi.detach()
        self.fields.amplitude.data = new_amp.detach()

        hv = self.hdc.field_state_vector(self.fields.omega, self.fields.phi)

        if not self.coupler._graph_built:
            self.coupler.build_graph(pos)
        diss = self.dissonance(
            self.fields.phi, self.fields.amplitude, self.fields.position,
            self.coupler.edge_index, y, u,
        )

        actions = self.topology.propose_actions(
            self.fields.omega, self.fields.phi,
            self.fields.amplitude, self.fields.position,
        )

        return {
            "h": h_next, "y": y, "u": u,
            "global_hv": hv, "dissonance": diss, "actions": actions,
        }

    def rollout(self, steps: int = 16, dt: float = 0.05) -> list:
        h = torch.zeros(self.memory.hidden_dim, device=self.fields.omega.device)
        results = []
        for _ in range(steps):
            out = self.step(h, dt=dt)
            results.append(out)
            h = out["h"].detach()
        return results
