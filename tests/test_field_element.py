"""Tests for FieldElement"""
import numpy as np
import pytest
from genesis_field_network.core import FieldElement


class TestFieldElementInit:
    def test_position_shape(self, field, manifold_dim):
        assert field.position.shape == (manifold_dim,)

    def test_frequencies_shape(self, field, manifold_dim, num_harmonics):
        assert field.frequencies.shape == (num_harmonics, manifold_dim)

    def test_phases_shape(self, field, num_harmonics):
        assert field.phases.shape == (num_harmonics,)

    def test_amplitudes_normalized(self, field):
        assert abs(np.sum(field.amplitudes) - 1.0) < 1e-9

    def test_amplitudes_non_negative(self, field):
        assert np.all(field.amplitudes >= 0)

    def test_curvature_symmetric(self, field):
        diff = np.max(np.abs(field.curvature - field.curvature.T))
        assert diff < 1e-12

    def test_curvature_positive_definite(self, field):
        eigenvalues = np.linalg.eigvalsh(field.curvature)
        assert np.all(eigenvalues > 0)

    def test_initial_energy(self, field):
        assert field.energy == 1.0

    def test_default_harmonics(self, manifold_dim):
        f = FieldElement(manifold_dim=manifold_dim)
        assert f.num_harmonics == 8

    def test_get_params_roundtrip(self, field):
        params = field.get_params()
        field2 = FieldElement(field.manifold_dim, field.num_harmonics)
        field2.set_params(params)
        np.testing.assert_allclose(field2.position, field.position)
        np.testing.assert_allclose(field2.phases, field.phases)


class TestFieldElementEvaluate:
    def test_output_shape(self, field, sample_points):
        result = field.evaluate(sample_points)
        assert result.shape == (len(sample_points),)

    def test_output_finite(self, field, sample_points):
        result = field.evaluate(sample_points)
        assert np.all(np.isfinite(result))

    def test_zero_energy_gives_zero(self, manifold_dim, num_harmonics, sample_points):
        f = FieldElement(manifold_dim, num_harmonics)
        f.energy = 0.0
        result = f.evaluate(sample_points)
        assert np.allclose(result, 0.0)

    def test_batch_equals_sequential(self, field, manifold_dim):
        pts = np.random.randn(5, manifold_dim)
        batch = field.evaluate(pts)
        singles = np.array([field.evaluate(pts[i:i+1])[0] for i in range(5)])
        np.testing.assert_allclose(batch, singles, rtol=1e-10)


class TestFieldElementComputeResonance:
    def test_self_resonance_is_one(self, field, sample_points):
        r = field.compute_resonance(field, sample_points)
        assert abs(r - 1.0) < 1e-9

    def test_resonance_range(self, field_pair, sample_points):
        f1, f2 = field_pair
        r = f1.compute_resonance(f2, sample_points)
        assert -1.0 <= r <= 1.0

    def test_resonance_symmetric(self, field_pair, sample_points):
        f1, f2 = field_pair
        assert abs(f1.compute_resonance(f2, sample_points) -
                   f2.compute_resonance(f1, sample_points)) < 1e-12

    def test_constant_field_returns_zero(self, field, manifold_dim, num_harmonics):
        f_const = FieldElement(manifold_dim, num_harmonics)
        f_const.energy = 0.0
        pts = np.random.randn(20, manifold_dim)
        assert field.compute_resonance(f_const, pts) == 0.0
