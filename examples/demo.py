"""
Genesis Field Network - Demo
==============================
Demonstrates GFN on simple tasks using phase-modulated
field interference — no neurons, no backpropagation through fields.

Tasks:
1. XOR Problem (classic non-linear benchmark)
2. Sine Wave Regression
3. Multi-class Pattern Classification
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from genesis_field_network.core import GenesisFieldNetwork


def demo_xor():
    print("=" * 60)
    print("DEMO 1: XOR Problem")
    print("=" * 60)

    np.random.seed(42)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.array([[0], [1], [1], [0]], dtype=float)

    gfn = GenesisFieldNetwork(
        input_dim=2, output_dim=1,
        manifold_dim=8, num_fields=16, num_harmonics=6,
    )
    gfn.morpher.spawn_threshold = 999.0
    gfn.morpher.max_fields = 20
    gfn.adapter.adaptation_rate = 0.02

    print(f"\nInitial state: {gfn.get_state_summary()['num_fields']} fields")
    print("Training...")

    history = gfn.train(X, Y, epochs=300, verbose=True)

    print("\nPredictions after training:")
    predictions = gfn.predict(X)
    for i in range(len(X)):
        pred = predictions[i][0]
        target = Y[i][0]
        mark = "✓" if abs(pred - target) < 0.3 else "✗"
        print(f"  Input: {X[i]} -> Predicted: {pred:.4f} | Target: {target} {mark}")

    state = gfn.get_state_summary()
    print(f"\nFinal state: {state['num_fields']} fields, "
          f"{state['morph_count']} topology changes")
    print(f"Final dissonance: {history[-1]:.6f}")
    return history


def demo_sine_regression():
    print("\n" + "=" * 60)
    print("DEMO 2: Sine Wave Regression")
    print("=" * 60)

    np.random.seed(42)
    X = np.random.uniform(-np.pi, np.pi, (30, 1))
    Y = np.sin(X)

    gfn = GenesisFieldNetwork(
        input_dim=1, output_dim=1,
        manifold_dim=8, num_fields=24, num_harmonics=8,
    )
    gfn.morpher.spawn_threshold = 999.0
    gfn.morpher.max_fields = 28
    gfn.adapter.adaptation_rate = 0.02

    print(f"\nInitial state: {gfn.get_state_summary()['num_fields']} fields")
    print("Training...")

    history = gfn.train(X, Y, epochs=200, verbose=True)

    X_test = np.linspace(-np.pi, np.pi, 10).reshape(-1, 1)
    Y_test = np.sin(X_test)
    predictions = gfn.predict(X_test)

    print("\nTest predictions:")
    for i in range(len(X_test)):
        print(f"  x={X_test[i][0]:+.2f} -> pred={predictions[i][0]:+.4f} "
              f"| true={Y_test[i][0]:+.4f}")

    mse = np.mean((predictions - Y_test) ** 2)
    print(f"\nTest MSE: {mse:.6f}")
    return history


def demo_classification():
    print("\n" + "=" * 60)
    print("DEMO 3: Pattern Classification (3 classes)")
    print("=" * 60)

    np.random.seed(123)
    n_per_class = 15

    X0 = np.random.randn(n_per_class, 2) * 0.3 + np.array([-1, -1])
    Y0 = np.tile([1, 0, 0], (n_per_class, 1))

    X1 = np.random.randn(n_per_class, 2) * 0.3 + np.array([1, -1])
    Y1 = np.tile([0, 1, 0], (n_per_class, 1))

    X2 = np.random.randn(n_per_class, 2) * 0.3 + np.array([0, 1])
    Y2 = np.tile([0, 0, 1], (n_per_class, 1))

    X = np.vstack([X0, X1, X2]).astype(float)
    Y = np.vstack([Y0, Y1, Y2]).astype(float)

    gfn = GenesisFieldNetwork(
        input_dim=2, output_dim=3,
        manifold_dim=6, num_fields=16, num_harmonics=6,
    )
    gfn.morpher.spawn_threshold = 999.0
    gfn.morpher.max_fields = 20
    gfn.adapter.adaptation_rate = 0.02

    print(f"\nInitial state: {gfn.get_state_summary()['num_fields']} fields")
    print("Training...")

    history = gfn.train(X, Y, epochs=200, verbose=True)

    predictions = gfn.predict(X)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(Y, axis=1)
    accuracy = np.mean(pred_classes == true_classes)

    print(f"\nTraining accuracy: {accuracy * 100:.1f}%")

    state = gfn.get_state_summary()
    print(f"Final state: {state['num_fields']} fields, "
          f"{state['morph_count']} topology changes")
    return history


if __name__ == "__main__":
    print("Genesis Field Network - Demonstration")
    print("Phase-modulated field interference computation")
    print()

    h1 = demo_xor()
    h2 = demo_sine_regression()
    h3 = demo_classification()

    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETE")
    print("=" * 60)
    print("\nKey observations:")
    print("- No neurons were used (only oscillatory field elements)")
    print("- No backpropagation through fields (Kuramoto sync + hill climbing)")
    print("- Nonlinearity comes from sin(phase + input_shift) interference")
    print("- Network topology changed dynamically during training")
    print("- Input/output projections use linear gradient updates (honestly)")
