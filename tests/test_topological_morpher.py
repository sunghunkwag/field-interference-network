"""Tests for TopologicalMorpher"""
import numpy as np
import pytest
from genesis_field_network.core import FieldElement, ResonanceCoupler, TopologicalMorpher

D, H = 4, 3

def make_fields(n):
    return [FieldElement(D, H) for _ in range(n)]

def make_coupler():
    return ResonanceCoupler(manifold_dim=D, coupling_resolution=16)


class TestMorphGeneral:
    def test_returns_list(self, morpher):
        result = morpher.morph(make_fields(6), make_coupler(), 0.5)
        assert isinstance(result, list)
        assert all(isinstance(f, FieldElement) for f in result)

    def test_respects_min_fields(self, morpher):
        result = morpher.morph(make_fields(morpher.min_fields), make_coupler(), 0.0)
        assert len(result) >= morpher.min_fields

    def test_respects_max_fields(self, morpher):
        result = morpher.morph(make_fields(morpher.max_fields), make_coupler(), 100.0)
        assert len(result) <= morpher.max_fields


class TestMergeFields:
    def test_merged_position_average(self, morpher):
        a, b = FieldElement(D, H), FieldElement(D, H)
        merged = morpher._merge_fields(a, b)
        np.testing.assert_allclose(merged.position, (a.position + b.position) / 2)

    def test_merged_energy_sum(self, morpher):
        a, b = FieldElement(D, H), FieldElement(D, H)
        a.energy, b.energy = 1.5, 2.5
        assert morpher._merge_fields(a, b).energy == pytest.approx(4.0)

    def test_merged_amplitudes_normalized(self, morpher):
        merged = morpher._merge_fields(FieldElement(D, H), FieldElement(D, H))
        assert abs(np.sum(merged.amplitudes) - 1.0) < 1e-9


class TestSplitField:
    def test_returns_two(self, morpher):
        assert len(morpher._split_field(FieldElement(D, H))) == 2

    def test_children_energy_halved(self, morpher):
        parent = FieldElement(D, H)
        parent.energy = 4.0
        ca, cb = morpher._split_field(parent)
        assert ca.energy == pytest.approx(2.0)
        assert cb.energy == pytest.approx(2.0)

    def test_children_symmetric(self, morpher):
        parent = FieldElement(D, H)
        ca, cb = morpher._split_field(parent)
        mid = (ca.position + cb.position) / 2
        np.testing.assert_allclose(mid, parent.position, atol=1e-12)
