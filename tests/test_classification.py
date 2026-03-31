"""
Test classification capability with 3-class Gaussian clusters.
"""

import numpy as np
import pytest
from genesis_field_network.core import GenesisFieldNetwork


def make_gaussian_clusters(n_per_class=30, seed=42):
    """Generate 3-class Gaussian cluster dataset with well-separated centers."""
    np.random.seed(seed)
    centers = np.array([[-1, -1], [1, -1], [0, 1]], dtype=float)
    X = []
    Y = []
    for cls_idx, center in enumerate(centers):
        points = center + np.random.randn(n_per_class, 2) * 0.3
        X.append(points)
        target = np.zeros(3)
        target[cls_idx] = 1.0
        Y.append(np.tile(target, (n_per_class, 1)))
    X = np.vstack(X)
    Y = np.vstack(Y)
    idx = np.random.permutation(len(X))
    return X[idx], Y[idx]


class TestClassification:
    def test_3class_convergence(self):
        """GFN should reach >80% accuracy on well-separated Gaussian clusters."""
        X, Y = make_gaussian_clusters(n_per_class=30, seed=42)

        np.random.seed(42)
        gfn = GenesisFieldNetwork(
            input_dim=2, output_dim=3,
            manifold_dim=8, num_fields=16, num_harmonics=6,
        )
        gfn.morpher.spawn_threshold = 999.0
        gfn.adapter.adaptation_rate = 0.02

        gfn.train(X, Y, epochs=500, verbose=False)

        preds = gfn.predict(X)
        pred_classes = np.argmax(preds, axis=1)
        true_classes = np.argmax(Y, axis=1)
        accuracy = np.mean(pred_classes == true_classes)

        assert accuracy > 0.80, (
            f"3-class accuracy {accuracy:.2%} is below 80% threshold. "
            f"Classification may need output feature normalization."
        )

    def test_output_shape(self):
        """Multi-output prediction should have correct shape."""
        np.random.seed(42)
        gfn = GenesisFieldNetwork(
            input_dim=2, output_dim=3,
            manifold_dim=6, num_fields=8, num_harmonics=4,
        )
        X = np.random.randn(10, 2)
        preds = gfn.predict(X)
        assert preds.shape == (10, 3)

    def test_different_inputs_different_outputs(self):
        """Different cluster centers should produce different predictions."""
        np.random.seed(42)
        gfn = GenesisFieldNetwork(
            input_dim=2, output_dim=3,
            manifold_dim=6, num_fields=8, num_harmonics=4,
        )
        gfn.morpher.spawn_threshold = 999.0
        X, Y = make_gaussian_clusters(n_per_class=15, seed=42)
        gfn.train(X, Y, epochs=100, verbose=False)

        centers = np.array([[-1, -1], [1, -1], [0, 1]], dtype=float)
        preds = gfn.predict(centers)
        # Each center should produce a distinct argmax
        pred_classes = np.argmax(preds, axis=1)
        assert len(set(pred_classes)) >= 2, (
            "Model should distinguish at least 2 of 3 clusters after training"
        )
