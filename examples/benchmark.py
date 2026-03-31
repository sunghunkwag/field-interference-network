"""
Genesis Field Network - Honest Benchmark Comparison
=====================================================
Compares GFN against simple baselines to establish where it stands.

Baselines:
1. Linear Regression (no hidden layer)
2. Random Forest (sklearn, if available)
3. 2-Layer MLP (numpy-only, for fair comparison)

Methodology:
- Same train/test split, same seeds
- Reports mean ± std over 5 runs
- No cherry-picking
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from typing import List, Tuple, Dict


# ═══════════════════════════════════════════════════════════════
# BASELINES (numpy only for fair comparison)
# ═══════════════════════════════════════════════════════════════

class LinearBaseline:
    """Simple linear regression via pseudoinverse."""
    def __init__(self):
        self.W = None
        self.b = None

    def train(self, X, Y, **kwargs):
        X_aug = np.column_stack([X, np.ones(len(X))])
        self.W = np.linalg.lstsq(X_aug, Y, rcond=None)[0]

    def predict(self, X):
        X_aug = np.column_stack([X, np.ones(len(X))])
        return X_aug @ self.W


class MLPBaseline:
    """2-layer MLP with tanh activation, trained via gradient descent."""
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.05):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.5
        self.b2 = np.zeros(output_dim)
        self.lr = lr

    def forward(self, X):
        self.h = np.tanh(X @ self.W1 + self.b1)
        return self.h @ self.W2 + self.b2

    def train(self, X, Y, epochs=300, **kwargs):
        for _ in range(epochs):
            pred = self.forward(X)
            error = pred - Y
            dW2 = self.h.T @ error / len(X)
            db2 = np.mean(error, axis=0)
            dh = error @ self.W2.T * (1 - self.h ** 2)
            dW1 = X.T @ dh / len(X)
            db1 = np.mean(dh, axis=0)
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

    def predict(self, X):
        h = np.tanh(X @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


# ═══════════════════════════════════════════════════════════════
# BENCHMARK TASKS
# ═══════════════════════════════════════════════════════════════

def make_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.array([[0], [1], [1], [0]], dtype=float)
    return X, Y, X, Y, "XOR"


def make_sine(n_train=30, n_test=20):
    X_train = np.random.uniform(-np.pi, np.pi, (n_train, 1))
    Y_train = np.sin(X_train)
    X_test = np.random.uniform(-np.pi, np.pi, (n_test, 1))
    Y_test = np.sin(X_test)
    return X_train, Y_train, X_test, Y_test, "Sine Regression"


def make_two_spirals(n_per_class=50, noise=0.3):
    theta = np.linspace(0, 2 * np.pi, n_per_class)
    r = np.linspace(0.5, 2.0, n_per_class)
    x0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    x1 = np.column_stack([-r * np.cos(theta), -r * np.sin(theta)])
    x0 += np.random.randn(*x0.shape) * noise
    x1 += np.random.randn(*x1.shape) * noise
    X = np.vstack([x0, x1])
    Y = np.vstack([np.zeros((n_per_class, 1)), np.ones((n_per_class, 1))])
    idx = np.random.permutation(len(X))
    X, Y = X[idx], Y[idx]
    split = int(0.7 * len(X))
    return X[:split], Y[:split], X[split:], Y[split:], "Two Spirals"


# ═══════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════

def run_benchmark(task_fn, models: Dict, n_runs: int = 5, epochs: int = 300):
    """Run benchmark and report mean ± std MSE."""
    results = {name: {'mse': [], 'time': []} for name in models}

    for run in range(n_runs):
        np.random.seed(run * 1000 + 42)
        X_train, Y_train, X_test, Y_test, task_name = task_fn()

        if run == 0:
            print(f"\n{'=' * 65}")
            print(f"  {task_name}")
            print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")
            print(f"  {n_runs} runs, {epochs} epochs each")
            print(f"{'=' * 65}")

        for name, model_factory in models.items():
            np.random.seed(run * 1000 + 42)
            model = model_factory()

            t0 = time.time()
            model.train(X_train, Y_train, epochs=epochs, verbose=False)
            elapsed = time.time() - t0

            preds = model.predict(X_test)
            mse = float(np.mean((preds - Y_test) ** 2))

            results[name]['mse'].append(mse)
            results[name]['time'].append(elapsed)

    # Print results table
    print(f"\n{'Model':<25s} {'MSE (mean±std)':<22s} {'Time (s)':<12s}")
    print("-" * 60)

    # Sort by MSE
    sorted_models = sorted(results.keys(),
                           key=lambda k: np.mean(results[k]['mse']))

    for name in sorted_models:
        mse_mean = np.mean(results[name]['mse'])
        mse_std = np.std(results[name]['mse'])
        t_mean = np.mean(results[name]['time'])
        print(f"{name:<25s} {mse_mean:.6f} ± {mse_std:.6f}  {t_mean:.2f}s")

    return results


def main():
    from genesis_field_network.core import GenesisFieldNetwork

    print("Genesis Field Network — Honest Benchmark Comparison")
    print("=" * 65)
    print("Baselines: Linear Regression, 2-Layer MLP (numpy)")
    print("All models use numpy only (no GPU, no PyTorch/TF)")
    print()

    def make_gfn():
        gfn = GenesisFieldNetwork(
            input_dim=2, output_dim=1, manifold_dim=8,
            num_fields=16, num_harmonics=6,
        )
        gfn.morpher.spawn_threshold = 999.0
        gfn.morpher.max_fields = 20
        gfn.adapter.adaptation_rate = 0.02
        return gfn

    def make_gfn_1d():
        gfn = GenesisFieldNetwork(
            input_dim=1, output_dim=1, manifold_dim=8,
            num_fields=20, num_harmonics=8,
        )
        gfn.morpher.spawn_threshold = 999.0
        gfn.morpher.max_fields = 24
        gfn.adapter.adaptation_rate = 0.02
        return gfn

    # === XOR ===
    models_xor = {
        'Linear': lambda: LinearBaseline(),
        'MLP (hidden=8)': lambda: MLPBaseline(2, 8, 1, lr=0.1),
        'GFN (16 fields)': make_gfn,
    }
    run_benchmark(make_xor, models_xor, n_runs=5, epochs=300)

    # === Sine ===
    models_sine = {
        'Linear': lambda: LinearBaseline(),
        'MLP (hidden=16)': lambda: MLPBaseline(1, 16, 1, lr=0.05),
        'GFN (20 fields)': make_gfn_1d,
    }
    run_benchmark(make_sine, models_sine, n_runs=5, epochs=200)

    # === Two Spirals ===
    models_spiral = {
        'Linear': lambda: LinearBaseline(),
        'MLP (hidden=16)': lambda: MLPBaseline(2, 16, 1, lr=0.05),
        'GFN (16 fields)': make_gfn,
    }
    run_benchmark(make_two_spirals, models_spiral, n_runs=5, epochs=300)

    print("\n" + "=" * 65)
    print("NOTES:")
    print("- GFN uses phase-modulated field interference (no backprop thru fields)")
    print("- MLP uses standard backpropagation (the conventional approach)")
    print("- Linear uses pseudoinverse (analytical solution)")
    print("- GFN's linear projections DO use gradient updates (honestly)")
    print("- GFN is expected to be slower due to O(n²) coupling computation")
    print("=" * 65)


if __name__ == "__main__":
    main()
