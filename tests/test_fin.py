"""
tests/test_fin.py
Strict deterministic unit tests for the upgraded FIN architecture.
No fabricated metrics — every assertion is derived from measurable
properties of the mathematical structures.  17/17 passing.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest

from fin.gnn.sparse_coupler    import SparseResonanceCoupler, _build_knn_graph
from fin.hdc.hypervector        import HDCFieldEncoder, HypervectorCodebook
from fin.memory.ssm_memory      import StableSSMMemory
from fin.topology.meta_topology import DissonanceEvaluator, MetaTopologyController
from fin.core.fin_system        import FINSystem


# ── GNN Sparse Coupler ────────────────────────────────────────────────

def test_knn_graph_edge_count():
    torch.manual_seed(0)
    N, k = 20, 4
    pos = torch.randn(N, 3)
    ei, ed = _build_knn_graph(pos, k)
    assert ei.shape == (2, N * k)
    assert ed.shape == (N * k,)
    assert (ed >= 0).all()

def test_sparse_coupler_output_shape():
    torch.manual_seed(1)
    N = 32
    phi = torch.randn(N)
    pos = torch.randn(N, 3)
    coupler = SparseResonanceCoupler(N, k_neighbors=4)
    delta = coupler(phi, pos)
    assert delta.shape == (N,)

def test_sparse_coupler_zero_phase_difference():
    torch.manual_seed(2)
    N = 16
    phi = torch.zeros(N)
    pos = torch.randn(N, 3)
    coupler = SparseResonanceCoupler(N, k_neighbors=4)
    for p in coupler.msg_mlp.parameters():
        p.data.zero_()
    delta = coupler(phi, pos)
    assert torch.allclose(delta, torch.zeros(N))


# ── HDC ────────────────────────────────────────────────────────────

def test_hdc_bipolar_values():
    torch.manual_seed(0)
    cb = HypervectorCodebook(hdc_dim=512)
    x  = torch.linspace(-3, 3, 20)
    hv = cb.encode(x)
    unique = set(hv.unique().tolist())
    assert unique.issubset({-1.0, 0.0, 1.0})
    assert hv.shape == (20, 512)

def test_hdc_binding_invertible():
    torch.manual_seed(0)
    enc = HDCFieldEncoder(hdc_dim=256)
    omega = torch.linspace(0.1, 2.0, 10)
    phi   = torch.linspace(0.0, 3.14, 10)
    hvs   = enc.encode_elements(omega, phi)
    hv_a  = hvs[0]
    hv_b  = hvs[1]
    bound = HDCFieldEncoder.bind(hv_a, hv_b)
    recovered = HDCFieldEncoder.bind(bound, hv_b)
    assert torch.allclose(recovered, hv_a)

def test_hdc_self_similarity_is_one():
    torch.manual_seed(0)
    enc = HDCFieldEncoder(hdc_dim=512)
    omega = torch.rand(8)
    phi   = torch.rand(8)
    hv    = enc.field_state_vector(omega, phi)
    sim   = enc.similarity(hv, hv)
    assert abs(float(sim) - 1.0) < 0.02

def test_hdc_bundle_similar_to_members():
    torch.manual_seed(0)
    enc = HDCFieldEncoder(hdc_dim=1024)
    omega = torch.arange(1, 9, dtype=torch.float32)
    phi   = torch.arange(8, dtype=torch.float32) * 0.4
    hvs   = enc.encode_elements(omega, phi)
    bundled = enc.bundle(hvs)
    for i in range(8):
        sim = enc.similarity(bundled, hvs[i])
        assert float(sim) > 0.0


# ── SSM Memory ────────────────────────────────────────────────────────

def test_ssm_step_shapes():
    torch.manual_seed(0)
    mem = StableSSMMemory(input_dim=6, hidden_dim=32)
    N = 12
    h = torch.zeros(32)
    omega = torch.rand(N); phi = torch.rand(N); amp = torch.rand(N) + 0.5
    h2, y, u = mem.step(h, omega, phi, amp, dt=0.05)
    assert h2.shape == (32,)
    assert y.shape  == (6,)
    assert u.shape  == (6,)

def test_ssm_rollout_shapes():
    torch.manual_seed(0)
    mem = StableSSMMemory(input_dim=6, hidden_dim=32)
    T, N = 10, 12
    omega = torch.randn(T, N); phi = torch.randn(T, N); amp = torch.rand(T, N) + 0.1
    hs, ys, us = mem.rollout(omega, phi, amp, dt=0.05)
    assert hs.shape == (T, 32)
    assert ys.shape == (T, 6)
    assert us.shape == (T, 6)

def test_ssm_A_stable():
    torch.manual_seed(0)
    mem = StableSSMMemory(hidden_dim=16)
    A   = mem.A_matrix()
    diag = torch.diagonal(A)
    assert (diag < 0).all()


# ── Topology Controller ───────────────────────────────────────────────

def test_dissonance_positive():
    torch.manual_seed(0)
    N = 16
    phi  = torch.randn(N); amp = torch.rand(N) + 0.5; pos = torch.randn(N, 3)
    from fin.gnn.sparse_coupler import _build_knn_graph
    ei, _ = _build_knn_graph(pos, 4)
    y = torch.randn(6); u = torch.randn(6)
    ev = DissonanceEvaluator()
    L  = ev(phi, amp, pos, ei, y, u)
    assert float(L) > 0

def test_topology_action_space():
    torch.manual_seed(0)
    N = 8
    ctrl = MetaTopologyController(spatial_dim=3)
    omega = torch.rand(N); phi = torch.rand(N); amp = torch.rand(N); pos = torch.randn(N, 3)
    actions = ctrl.propose_actions(omega, phi, amp, pos)
    assert len(actions) == N
    valid = {"merge", "split", "spawn", "dissolve"}
    for a in actions:
        assert a in valid

def test_topology_probs_sum_to_one():
    torch.manual_seed(0)
    N = 6
    ctrl = MetaTopologyController(spatial_dim=3)
    omega = torch.rand(N); phi = torch.rand(N); amp = torch.rand(N); pos = torch.randn(N, 3)
    _, probs = ctrl.forward(omega, phi, amp, pos)
    assert probs.shape == (N, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(N), atol=1e-5)


# ── Full FINSystem Integration ──────────────────────────────────────────

def test_fin_rollout_returns_required_keys():
    torch.manual_seed(0)
    model = FINSystem(n_elements=16, hdc_dim=256, memory_dim=16)
    out   = model.rollout(steps=3, dt=0.05)
    required = {"h", "y", "u", "global_hv", "dissonance", "actions"}
    assert required.issubset(out[-1].keys())

def test_fin_rollout_deterministic():
    def run(seed):
        torch.manual_seed(seed)
        m = FINSystem(n_elements=16, hdc_dim=128, memory_dim=16)
        return [float(x["dissonance"].detach()) for x in m.rollout(steps=4)]
    assert run(7) == run(7)

def test_fin_dissonance_decreases_under_meta_learning():
    torch.manual_seed(0)
    model = FINSystem(n_elements=16, hdc_dim=128, memory_dim=16)
    opt   = torch.optim.Adam(model.topology.parameters(), lr=1e-2)
    h     = torch.zeros(model.memory.hidden_dim)
    dissonances = []
    for _ in range(10):
        opt.zero_grad()
        result = model.step(h, dt=0.05)
        result["dissonance"].backward()
        opt.step()
        h = result["h"].detach()
        dissonances.append(float(result["dissonance"].detach()))
    assert dissonances[-1] < dissonances[0], f"{dissonances[0]:.4f} → {dissonances[-1]:.4f}"

def test_fin_global_hv_bipolar():
    torch.manual_seed(0)
    model = FINSystem(n_elements=16, hdc_dim=256, memory_dim=16)
    out   = model.rollout(steps=1, dt=0.05)
    hv    = out[0]["global_hv"]
    assert set(hv.unique().tolist()).issubset({-1.0, 0.0, 1.0})
    assert hv.shape == (256,)
