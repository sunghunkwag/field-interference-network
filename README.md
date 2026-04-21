# Field Interference Network (FIN) — Upgraded Architecture

Scalable, mathematically rigorous foundation for AGI/RSI research built on native PyTorch.

## Core Upgrades

| Module | File | Replaces | Complexity |
|---|---|---|---|
| `SparseResonanceCoupler` | `fin/gnn/sparse_coupler.py` | Dense O(n²) ResonanceCoupler | O(n·k) |
| `HDCFieldEncoder` | `fin/hdc/hypervector.py` | Raw frequency arrays | O(n·D) |
| `StableSSMMemory` | `fin/memory/ssm_memory.py` | Ad-hoc propagate() | Continuous-time |
| `MetaTopologyController` | `fin/topology/meta_topology.py` | Hardcoded thresholds | Gradient-driven |
| `FINSystem` | `fin/core/fin_system.py` | NumPy prototype | Native PyTorch + autograd |

## Mathematical Foundations

### Sparse GNN Coupling
Kuramoto synchronisation restricted to a k-NN spatial graph:

```
Δφ_i = Σ_{j∈N(i)} w_ij · MLP([φ_j−φ_i, w_ij]) · sin(φ_j−φ_i)
```

### HDC Encoding
Bipolar Random Fourier Features: `hv(x) = sign(cos(Ωx + b))`  
Binding: `bind(a,b) = a ⊙ b` (self-inverse)  
Bundling: `bundle(hvs) = sign(Σ hvs_i)`

### SSM Memory
Continuous-time linear SSM with stable parameterisation:  
`dh/dt = Ah + Bu`, `y = Ch + Du`  
Stability: `A = diag(−softplus(λ)) + UVᵀ`

### Dissonance Objective (RSI)
```
L = α·L_phase + β·L_energy + γ·L_memory + δ·L_graph
```
Topology actions (merge/split/spawn/dissolve) selected by gradient-trained MLP, not RNG.

## Installation

```bash
pip install torch
```

## Usage

```bash
# Run 17 deterministic unit tests
python -m pytest tests/ -v

# Run meta-learning training loop
python fin/meta/train_meta.py

# Empirical sparse vs dense timing benchmark
python benchmarks/benchmark_fin.py
```

## Quick Start

```python
import torch
from fin.core.fin_system import FINSystem

model = FINSystem(n_elements=128, hdc_dim=1024, memory_dim=64, device="cpu")
results = model.rollout(steps=16, dt=0.05)

print(results[-1]["dissonance"])   # scalar dissonance loss
print(results[-1]["global_hv"].shape)  # [1024] bipolar field state
print(results[-1]["actions"][:5])  # per-node topology proposals
```
