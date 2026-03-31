"""Integration tests"""
import numpy as np
import pytest
from genesis_field_network.core import GenesisFieldNetwork


def make_gfn(input_dim, output_dim, num_fields=8):
    gfn = GenesisFieldNetwork(input_dim=input_dim, output_dim=output_dim,
                               manifold_dim=4, num_fields=num_fields, num_harmonics=3)
    gfn.morpher.spawn_threshold = 999.0
    gfn.morpher.max_fields = num_fields + 4
    return gfn


class TestEndToEnd:
    def test_training_completes(self):
        gfn = make_gfn(2, 1)
        X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        Y = np.array([[0],[1],[1],[0]], dtype=float)
        assert len(gfn.train(X, Y, epochs=5, verbose=False)) == 5

    def test_xor_learning(self):
        """XOR should converge with enough epochs."""
        np.random.seed(42)
        gfn = GenesisFieldNetwork(input_dim=2, output_dim=1, manifold_dim=8,
                                   num_fields=16, num_harmonics=6)
        gfn.morpher.spawn_threshold = 999.0
        gfn.morpher.max_fields = 20
        gfn.adapter.adaptation_rate = 0.02
        X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        Y = np.array([[0],[1],[1],[0]], dtype=float)
        history = gfn.train(X, Y, epochs=300, verbose=False)
        assert history[-1] < history[0] * 0.1  # 90%+ reduction

    def test_predict_shape_after_training(self):
        gfn = make_gfn(2, 3)
        X, Y = np.random.randn(6, 2), np.random.randn(6, 3)
        gfn.train(X, Y, epochs=3, verbose=False)
        assert gfn.predict(X).shape == (6, 3)


class TestDeterminism:
    def test_same_seed_same_output(self):
        np.random.seed(42)
        g1 = make_gfn(2, 1, 6)
        o1 = g1.forward(np.array([0.3, -0.7]))
        np.random.seed(42)
        g2 = make_gfn(2, 1, 6)
        o2 = g2.forward(np.array([0.3, -0.7]))
        np.testing.assert_allclose(o1, o2)


class TestPackageImport:
    def test_all_classes(self):
        from genesis_field_network import (FieldElement, GenesisFieldNetwork,
            PhaseAdapter, ResonanceCoupler, TopologicalMorpher)
        assert all(c is not None for c in [FieldElement, GenesisFieldNetwork,
            PhaseAdapter, ResonanceCoupler, TopologicalMorpher])

    def test_version(self):
        import genesis_field_network
        assert isinstance(genesis_field_network.__version__, str)
