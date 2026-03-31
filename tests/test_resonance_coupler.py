"""Tests for ResonanceCoupler"""
import numpy as np
import pytest
from genesis_field_network.core import FieldElement, ResonanceCoupler


class TestComputeCouplingMatrix:
    def test_output_shape(self, coupler, small_field_list):
        n = len(small_field_list)
        matrix = coupler.compute_coupling_matrix(small_field_list)
        assert matrix.shape == (n, n)

    def test_symmetric(self, coupler, small_field_list):
        matrix = coupler.compute_coupling_matrix(small_field_list)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)

    def test_diagonal_is_zero(self, coupler, small_field_list):
        matrix = coupler.compute_coupling_matrix(small_field_list)
        np.testing.assert_allclose(np.diag(matrix), 0.0)

    def test_values_in_range(self, coupler, small_field_list):
        matrix = coupler.compute_coupling_matrix(small_field_list)
        assert np.all(matrix >= -1.0 - 1e-9)
        assert np.all(matrix <= 1.0 + 1e-9)


class TestPropagate:
    def test_output_shape(self, coupler, small_field_list):
        excitation = np.ones(len(small_field_list))
        energies = coupler.propagate(small_field_list, excitation)
        assert energies.shape == (len(small_field_list),)

    def test_output_finite(self, coupler, small_field_list):
        excitation = np.random.randn(len(small_field_list))
        energies = coupler.propagate(small_field_list, excitation)
        assert np.all(np.isfinite(energies))

    def test_energies_bounded(self, coupler, small_field_list):
        excitation = np.ones(len(small_field_list)) * 10
        energies = coupler.propagate(small_field_list, excitation)
        assert np.all(np.abs(energies) <= 3.0 + 1e-9)
