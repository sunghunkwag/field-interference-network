"""
Tests for save_state / load_state reproducibility.
"""

import json
import numpy as np
import pytest
from genesis_field_network.core import GenesisFieldNetwork


class TestSaveLoadState:
    def test_save_returns_dict(self):
        """save_state() returns a dict with required keys."""
        np.random.seed(42)
        gfn = GenesisFieldNetwork(input_dim=2, output_dim=1, manifold_dim=4, num_fields=6)
        state = gfn.save_state()
        assert isinstance(state, dict)
        for key in ['input_dim', 'output_dim', 'manifold_dim', 'fields',
                     'input_projection', 'output_projection']:
            assert key in state

    def test_save_is_json_serializable(self):
        """State dict should be JSON-serializable."""
        np.random.seed(42)
        gfn = GenesisFieldNetwork(input_dim=2, output_dim=1, manifold_dim=4, num_fields=6)
        state = gfn.save_state()
        json_str = json.dumps(state)
        assert len(json_str) > 0
        restored = json.loads(json_str)
        assert restored['input_dim'] == 2

    def test_roundtrip_predictions(self):
        """Predictions should be identical after save/load roundtrip."""
        np.random.seed(42)
        gfn = GenesisFieldNetwork(
            input_dim=2, output_dim=1, manifold_dim=4,
            num_fields=6, num_harmonics=4,
        )
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        preds_before = gfn.predict(X)

        state = gfn.save_state()

        gfn2 = GenesisFieldNetwork(input_dim=2, output_dim=1, manifold_dim=4, num_fields=6)
        gfn2.load_state(state)
        preds_after = gfn2.predict(X)

        np.testing.assert_allclose(preds_before, preds_after, atol=1e-12)

    def test_roundtrip_after_training(self):
        """Predictions should match after training + save/load."""
        np.random.seed(42)
        gfn = GenesisFieldNetwork(
            input_dim=2, output_dim=1, manifold_dim=8,
            num_fields=8, num_harmonics=6,
        )
        gfn.morpher.spawn_threshold = 999.0
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        Y = np.array([[0], [1], [1], [0]], dtype=float)
        gfn.train(X, Y, epochs=50, verbose=False)

        preds_before = gfn.predict(X)
        state = gfn.save_state()

        gfn2 = GenesisFieldNetwork(input_dim=2, output_dim=1, manifold_dim=4, num_fields=4)
        gfn2.load_state(state)
        preds_after = gfn2.predict(X)

        np.testing.assert_allclose(preds_before, preds_after, atol=1e-12)

    def test_json_roundtrip_after_training(self):
        """Save to JSON, load from JSON, predictions still match."""
        np.random.seed(42)
        gfn = GenesisFieldNetwork(
            input_dim=2, output_dim=1, manifold_dim=4,
            num_fields=6, num_harmonics=4,
        )
        gfn.morpher.spawn_threshold = 999.0
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        Y = np.array([[0], [1], [1], [0]], dtype=float)
        gfn.train(X, Y, epochs=20, verbose=False)

        state = gfn.save_state()
        json_str = json.dumps(state)
        restored_state = json.loads(json_str)

        gfn2 = GenesisFieldNetwork(input_dim=2, output_dim=1, manifold_dim=4, num_fields=4)
        gfn2.load_state(restored_state)

        preds_before = gfn.predict(X)
        preds_after = gfn2.predict(X)
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-12)

    def test_field_count_restored(self):
        """Number of fields should be preserved through save/load."""
        np.random.seed(42)
        gfn = GenesisFieldNetwork(input_dim=2, output_dim=1, manifold_dim=4, num_fields=12)
        state = gfn.save_state()
        assert len(state['fields']) == 12

        gfn2 = GenesisFieldNetwork(input_dim=2, output_dim=1, manifold_dim=4, num_fields=4)
        gfn2.load_state(state)
        assert len(gfn2.fields) == 12

    def test_morph_log_preserved(self):
        """Morph log should be preserved through save/load."""
        np.random.seed(42)
        gfn = GenesisFieldNetwork(
            input_dim=2, output_dim=1, manifold_dim=4,
            num_fields=6, num_harmonics=4,
        )
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        Y = np.array([[0], [1], [1], [0]], dtype=float)
        gfn.train(X, Y, epochs=20, verbose=False)

        morph_count_before = len(gfn.morpher.morph_log)
        state = gfn.save_state()

        gfn2 = GenesisFieldNetwork(input_dim=2, output_dim=1, manifold_dim=4, num_fields=4)
        gfn2.load_state(state)
        assert len(gfn2.morpher.morph_log) == morph_count_before
