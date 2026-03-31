"""Tests for GenesisFieldNetwork"""
import numpy as np
import pytest
from genesis_field_network.core import FieldElement, GenesisFieldNetwork


class TestInit:
    def test_stored_dims(self, small_network):
        assert small_network.input_dim == 2
        assert small_network.output_dim == 1

    def test_field_count(self, small_network):
        assert len(small_network.fields) == 6

    def test_fields_are_field_elements(self, small_network):
        assert all(isinstance(f, FieldElement) for f in small_network.fields)

    def test_projection_shapes(self, small_network):
        assert small_network.input_projection.shape == (2, 6)
        assert small_network.output_projection.shape == (6, 1)


class TestForward:
    def test_output_shape(self, small_network):
        out = small_network.forward(np.array([0.5, -0.3]))
        assert out.shape == (1,)

    def test_output_finite(self, small_network):
        out = small_network.forward(np.random.randn(2))
        assert np.all(np.isfinite(out))

    def test_different_inputs_different_outputs(self, small_network):
        o1 = small_network.forward(np.array([1.0, 0.0]))
        o2 = small_network.forward(np.array([0.0, 1.0]))
        # Phase modulation guarantees different outputs
        assert not np.allclose(o1, o2)

    def test_projection_resizes(self):
        gfn = GenesisFieldNetwork(input_dim=2, output_dim=1, manifold_dim=4,
                                   num_fields=6, num_harmonics=3)
        gfn.fields = gfn.fields[:4]
        out = gfn.forward(np.array([0.5, -0.5]))
        assert out.shape == (1,) and np.all(np.isfinite(out))


class TestLearn:
    def test_returns_float(self, small_network):
        d = small_network.learn(np.array([1.0, 0.0]), np.array([1.0]))
        assert isinstance(d, (float, np.floating))

    def test_dissonance_non_negative(self, small_network):
        d = small_network.learn(np.array([1.0, 0.0]), np.array([1.0]))
        assert d >= 0.0

    def test_no_crash_repeated(self, small_network):
        for _ in range(10):
            small_network.learn(np.array([0.5, 0.5]), np.array([0.5]))


class TestTrain:
    def test_history_length(self, small_network, xor_dataset):
        X, Y = xor_dataset
        history = small_network.train(X, Y, epochs=3, verbose=False)
        assert len(history) == 3

    def test_history_non_negative(self, small_network, xor_dataset):
        X, Y = xor_dataset
        history = small_network.train(X, Y, epochs=3, verbose=False)
        assert all(d >= 0 for d in history)

    def test_verbose_false_silent(self, small_network, xor_dataset, capsys):
        X, Y = xor_dataset
        small_network.train(X, Y, epochs=2, verbose=False)
        assert capsys.readouterr().out == ""


class TestPredict:
    def test_output_shape(self, small_network):
        preds = small_network.predict(np.random.randn(8, 2))
        assert preds.shape == (8, 1)

    def test_output_finite(self, small_network):
        preds = small_network.predict(np.random.randn(5, 2))
        assert np.all(np.isfinite(preds))


class TestGetStateSummary:
    def test_returns_dict(self, small_network):
        assert isinstance(small_network.get_state_summary(), dict)

    def test_required_keys(self, small_network):
        keys = {"num_fields", "total_energy", "mean_energy",
                "max_energy", "morph_count", "dissonance_history"}
        assert keys.issubset(set(small_network.get_state_summary().keys()))

    def test_num_fields_correct(self, small_network):
        assert small_network.get_state_summary()["num_fields"] == len(small_network.fields)
