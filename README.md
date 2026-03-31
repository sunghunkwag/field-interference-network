# Genesis Field Network (GFN)

**Experimental.** An exploratory prototype investigating field-based resonance dynamics as an alternative computational substrate. Replaces neurons, backpropagation, and fixed topologies with oscillatory field elements, phase-modulated interference, and dynamic structure morphing. No claims of practical utility are made.

```bash
pip install numpy && python examples/demo.py
```

## How it works

GFN encodes information as standing wave patterns in a continuous manifold. Computation emerges from interference between overlapping fields — not from weighted sums in layers.

The core operation is **phase-modulated field interference**:

```
feature_i = energy_i × Σ_h  amplitude_h × sin(ω_h · p_i + φ_h + Δφ_i(x))
```

where `Δφ_i(x) = x @ W[:, i]` is the input-dependent phase shift. Since `sin(A + B) = sin(A)cos(B) + cos(A)sin(B)`, each feature is a nonlinear combination of sine/cosine basis functions — a learnable Fourier-like representation.

| Concept | Neural network | GFN |
|---|---|---|
| Computational unit | Neurons (weighted sums) | Oscillatory field elements |
| Topology | Fixed layers | Dynamic (merge / split / spawn / dissolve) |
| Training signal | Backpropagation | Kuramoto phase sync + stochastic hill climbing |
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

| Task | GFN | MLP (backprop) | Linear (analytical) |
|---|---|---|---|
| XOR (4 samples, 300 epochs) | **MSE 0.002** | MSE 0.017 | MSE 0.250 |
| Sine regression (30 samples, 200 epochs) | MSE 0.073 | ~0.001 | MSE 0.320 |

GFN achieves MLP-comparable accuracy on XOR (actually better on this toy problem), but is ~900× slower due to O(n²) coupling computation. The scientific value is in the alternative computation model, not practical performance.

Run the benchmark yourself:

```bash
python examples/benchmark.py
```

## Architecture

```
genesis_field_network/
├── __init__.py          # Package exports
└── core.py              # All components:
    ├── FieldElement           # Oscillatory entity with harmonic spectrum
    ├── ResonanceCoupler       # Field interaction via resonance correlation
    ├── PhaseAdapter           # Learning via Kuramoto sync + hill climbing
    ├── TopologicalMorpher     # Dynamic merge / split / spawn / dissolve
    └── GenesisFieldNetwork    # Complete system
```

### Components

**FieldElement** — A continuous oscillatory entity defined over a region of the computational manifold. Each element has a position, frequency spectrum, phase configuration, amplitude envelope, and curvature tensor. Information is encoded in the interference pattern between multiple elements, not in any single one.

**ResonanceCoupler** — Manages field interactions through harmonic resonance. The coupling matrix is NOT stored weights — it is dynamically computed from field configurations and changes as fields evolve. High resonance (constructive interference) amplifies energy transfer; anti-resonance suppresses it.

**PhaseAdapter** — Genuinely gradient-free learning for field parameters. Uses Kuramoto-inspired phase synchronization (local, pairwise) plus stochastic hill climbing for frequency and position adaptation. The error signal drives adaptation intensity but no gradient chain is computed through the field dynamics.

**TopologicalMorpher** — Evolves network structure during training:
- **Merge**: highly resonant fields collapse into one (reduces redundancy)
- **Split**: complex high-energy fields divide (increases capacity)
- **Spawn**: new fields appear in high-dissonance regions (targeted exploration)
- **Dissolve**: near-zero energy fields are removed (pruning)

## Testing

```bash
pip install pytest pytest-cov
python -m pytest tests/ -v --tb=short --cov=genesis_field_network
```

68 tests, 97% coverage.

## Known limitations

- **O(n²) coupling** limits scalability beyond ~50 fields
- **Classification** with multiple outputs requires careful hyperparameter tuning
- **The linear projection updates are the primary learning driver**; field adaptation contributes exploration but not precise optimization
- **No GPU acceleration** (numpy only, CPU-bound)
- **Not a practical alternative** to neural networks for real tasks

## Why this exists

This is an exploration of whether computation can emerge from harmonic interference in continuous fields rather than from weighted sums in discrete layers. The answer appears to be: yes, for toy problems, but with significant practical limitations. The value is conceptual, not competitive.

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.21

## License

MIT
