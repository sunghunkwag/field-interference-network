"""
Reproducibility demonstration: train XOR, save, load, verify identical predictions.
"""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from genesis_field_network.core import GenesisFieldNetwork


def main():
    print("Genesis Field Network — Reproducibility Demo")
    print("=" * 55)

    # Train XOR
    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.array([[0], [1], [1], [0]], dtype=float)

    gfn = GenesisFieldNetwork(
        input_dim=2, output_dim=1,
        manifold_dim=8, num_fields=16, num_harmonics=6,
    )
    gfn.morpher.spawn_threshold = 999.0
    gfn.adapter.adaptation_rate = 0.02

    print("\nTraining XOR (300 epochs)...")
    gfn.train(X, Y, epochs=300, verbose=False)

    preds_original = gfn.predict(X)
    print("\nOriginal predictions:")
    for i in range(len(X)):
        print(f"  {X[i]} -> {preds_original[i, 0]:.6f} (target: {Y[i, 0]:.0f})")

    # Save state
    state = gfn.save_state()
    json_str = json.dumps(state)
    print(f"\nSaved state: {len(json_str)} bytes JSON")
    print(f"  Fields: {len(state['fields'])}")
    print(f"  Morph log entries: {len(state['morph_log'])}")

    # Load into a fresh network
    gfn2 = GenesisFieldNetwork(input_dim=2, output_dim=1, manifold_dim=4, num_fields=4)
    restored_state = json.loads(json_str)
    gfn2.load_state(restored_state)

    preds_loaded = gfn2.predict(X)
    print("\nLoaded predictions:")
    for i in range(len(X)):
        print(f"  {X[i]} -> {preds_loaded[i, 0]:.6f} (target: {Y[i, 0]:.0f})")

    # Verify
    max_diff = np.max(np.abs(preds_original - preds_loaded))
    print(f"\nMax prediction difference: {max_diff:.2e}")
    if max_diff < 1e-10:
        print("PASS: Predictions are identical after save/load roundtrip")
    else:
        print("FAIL: Predictions differ after save/load")
        sys.exit(1)

    print("\n" + "=" * 55)


if __name__ == "__main__":
    main()
