# Field Interference Network

**Experimental.** An exploratory prototype investigating field-based resonance dynamics as an alternative computational substrate. Replaces neurons, backpropagation, and fixed topologies with oscillatory field elements, phase-modulated interference, and dynamic structure morphing. No claims of practical utility are made.

```bash
pip install numpy && python examples/demo.py
```

## How it works

Information is encoded as standing wave patterns in a continuous manifold. Computation emerges from interference between overlapping fields — not from weighted sums in layers.

The core operation is **phase-modulated field interference**:

```
feature_i = energy_i × Σ_h  amplitude_h × sin(ω_h · p_i + φ_h + Δφ_i(x))
```

where `Δφ_i(x) = x @ W[:, i]` is the input-dependent phase shift. Since `sin(A + B) = sin(A)cos(B) + cos(A)sin(B)`, each feature is a nonlinear combination of sine/cosine basis functions — a learnable Fourier-like representation.

| Concept | Neural network | This system |
|---|---|---|
| Computational unit | Neurons (weighted sums) | Oscillatory field elements |
| Topology | Fixed layers | Dynamic (merge / split / spawn / dissolve) |
| Training signal | Backpropagation | Vectorized Kuramoto phase sync + stochastic hill climbing |
| Nonlinearity | Activation functions | Sine interference (inherent) |

### What is NOT neural about this

- No hidden layers — fields exist in a continuous manifold
- No backpropagation through the field dynamics
- No activation functions — `sin(phase + input_shift)` IS the nonlinearity
- Network topology changes dynamically during training

### What IS conventional (honestly)

- Input/output projections are linear maps updated via error signal (a gradient step)
- This is a hybrid system: field-based representation + linear readout with gradient updates
- The "gradient-free" claim applies to field parameter adaptation only

## Benchmark results

| Task | FIN | MLP (backprop) | Linear (analytical) | FIN Speed |
|---|---|---|---|---|
| XOR (4 samples, 300 epochs) | **MSE 0.010** | MSE 0.019 | MSE 0.250 | 6.4s |
| Sine regression (30 samples, 200 epochs) | MSE 0.041 | **MSE 0.023** | MSE 0.208 | 71.1s |
| Two Spirals (100 samples, 300 epochs) | MSE 0.326 | **MSE 0.235** | MSE 0.255 | 61.0s |
| 3-class classification (90 samples, 300 epochs) | ~79% accuracy | — | — | ~151s |

Achieves MLP-comparable accuracy on XOR, but is slower due to O(n²) coupling computation. Classification works on well-separated clusters with high seed variance. The scientific value is in the alternative computation model, not practical performance.

```bash
python examples/benchmark.py
```

## Project structure

```
field-interference-network/
├── genesis_field_network/
│   ├── __init__.py              # Package exports
│   └── core.py                  # All components:
│       ├── FieldElement               # Oscillatory entity with harmonic spectrum
│       ├── ResonanceCoupler           # Dynamic coupling via resonance correlation
│       ├── PhaseAdapter               # Vectorized Kuramoto sync + hill climbing
│       ├── TopologicalMorpher         # Merge / split / spawn / dissolve
│       └── GenesisFieldNetwork        # Complete system (with save/load state)
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_field_element.py    # FieldElement unit tests
│   ├── test_resonance_coupler.py
│   ├── test_phase_adapter.py
│   ├── test_topological_morpher.py
│   ├── test_genesis_field_network.py
│   ├── test_integration.py      # End-to-end, determinism, package import
│   ├── test_classification.py   # 3-class Gaussian cluster convergence
│   └── test_serialization.py    # save_state / load_state roundtrip
├── examples/
│   ├── demo.py                  # XOR + sine + classification demos
│   ├── benchmark.py             # Honest comparison vs MLP and linear baselines
│   └── reproduce_xor.py         # Train → save → load → verify identical predictions
├── .github/workflows/tests.yml  # CI: Python 3.9, 3.10, 3.12
├── pyproject.toml
├── LICENSE                      # MIT
└── README.md
```

## Components

**FieldElement** — A continuous oscillatory entity defined over a region of the computational manifold. Each element has a position, frequency spectrum, phase configuration, amplitude envelope, and curvature tensor (with cached inverse). Information is encoded in the interference pattern between multiple elements, not in any single one. Supports `get_params()` / `set_params()` for serialization.

**ResonanceCoupler** — Manages field interactions through harmonic resonance. The coupling matrix is NOT stored weights — it is dynamically computed from field configurations and changes as fields evolve. High resonance (constructive interference) amplifies energy transfer; anti-resonance suppresses it.

**PhaseAdapter** — Genuinely gradient-free learning for field parameters. Uses vectorized Kuramoto-inspired phase synchronization (replaced per-field Python loops with `numpy` matrix ops on coupling matrix and phase arrays) plus stochastic hill climbing for frequency and position adaptation. The error signal drives adaptation intensity but no gradient chain is computed through the field dynamics.

**TopologicalMorpher** — Evolves network structure during training:
- **Merge**: highly resonant fields collapse into one (reduces redundancy)
- **Split**: complex high-energy fields divide (increases capacity)
- **Spawn**: new fields appear in high-dissonance regions (targeted exploration)
- **Dissolve**: near-zero energy fields are removed (pruning)

**GenesisFieldNetwork** — The complete system. Coupling matrix is computed once per `learn()` call and shared across `propagate()`, `adapt_fields()`, and `morph()` (eliminates 3× redundant O(n²) computation). Supports `save_state()` / `load_state()` for full reproducibility via JSON-serializable dicts.

## Testing

```bash
pip install pytest pytest-cov
python -m pytest tests/ -v --tb=short --cov=genesis_field_network
```

78 tests, 97% coverage. Includes serialization roundtrip tests and 3-class classification convergence tests.

## Reproducibility

```bash
python examples/reproduce_xor.py
```

Trains XOR, saves the full network state to JSON, loads into a fresh instance, and verifies identical predictions (max diff < 1e-10).

## Known limitations

- **O(n² coupling)** limits scalability beyond ~50 fields (cached per learn step for ~3.8× speedup over naive)
- **Classification** works on well-separated clusters but has high variance across seeds
- **Linear projection updates are the primary learning driver**; field adaptation contributes exploration but not precise optimization
- **No GPU acceleration** (numpy only, CPU-bound)
- **Not a practical alternative** to neural networks for real tasks

## Why this exists

An exploration of whether computation can emerge from harmonic interference in continuous fields rather than from weighted sums in discrete layers. The answer: yes, for toy problems, but with significant practical limitations. The value is conceptual, not competitive.

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.21

## License

MIT
