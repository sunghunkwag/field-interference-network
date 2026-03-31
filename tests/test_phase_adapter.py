"""Tests for PhaseAdapter"""
import numpy as np
import pytest
from genesis_field_network.core import FieldElement, PhaseAdapter, ResonanceCoupler


def make_fields(n, manifold_dim=4, num_harmonics=3):
    return [FieldElement(manifold_dim, num_harmonics) for _ in range(n)]

def make_coupler(manifold_dim=4):
    return ResonanceCoupler(manifold_dim=manifold_dim, coupling_resolution=16)


class TestComputeDissonance:
    def test_identical_patterns_zero(self, adapter):
        p = np.array([0.1, 0.5, -0.3, 0.7])
        assert adapter.compute_dissonance(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_returns_float(self, adapter):
        d = adapter.compute_dissonance(np.array([1.0, 2.0]), np.zeros(2))
        assert isinstance(d, float)

    def test_non_negative(self, adapter):
        d = adapter.compute_dissonance(np.random.randn(10), np.random.randn(10))
        assert d >= 0.0

    def test_appended_to_history(self, adapter):
        adapter.compute_dissonance(np.array([1.0]), np.array([0.0]))
        assert len(adapter.dissonance_history) == 1

    def test_length_mismatch_handled(self, adapter):
        d = adapter.compute_dissonance(np.array([1.0, 2.0, 3.0]), np.array([1.0]))
        assert isinstance(d, float) and d >= 0.0


class TestAdaptFields:
    def test_returns_float(self, adapter):
        fields = make_fields(4)
        coupler = make_coupler()
        result = adapter.adapt_fields(fields, np.array([0.5, 0.3]),
                                       np.zeros(2), coupler)
        assert isinstance(result, float)

    def test_phases_in_range(self, adapter):
        fields = make_fields(4)
        coupler = make_coupler()
        adapter.adapt_fields(fields, np.random.randn(2), np.random.randn(2), coupler)
        for f in fields:
            assert np.all(f.phases >= 0)
            assert np.all(f.phases < 2 * np.pi + 1e-9)

    def test_frequencies_in_bounds(self, adapter):
        fields = make_fields(4)
        coupler = make_coupler()
        adapter.adapt_fields(fields, np.random.randn(2), np.random.randn(2), coupler)
        for f in fields:
            assert np.all(f.frequencies >= 0.01)
            assert np.all(f.frequencies <= 10.0)

    def test_amplitudes_normalized(self, adapter):
        fields = make_fields(4)
        coupler = make_coupler()
        adapter.adapt_fields(fields, np.random.randn(2), np.random.randn(2), coupler)
        for f in fields:
            assert abs(np.sum(f.amplitudes) - 1.0) < 1e-9
