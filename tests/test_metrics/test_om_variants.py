"""Tests for yasqat's OM heuristics (formerly omloc, omspell, omstran)."""

import numpy as np
import pytest

from yasqat.metrics.om_variants import (
    om_boundary_weighted_distance,
    om_spell_scaled_distance,
    om_transition_penalty_distance,
)


class TestOmBoundaryWeighted:
    """Tests for the boundary-weighted OM heuristic."""

    def test_identical_sequences(self) -> None:
        seq = np.array([0, 1, 2, 3])
        assert om_boundary_weighted_distance(seq, seq) == pytest.approx(0.0)

    def test_symmetric(self) -> None:
        a = np.array([0, 1, 2])
        b = np.array([0, 2, 1])
        assert om_boundary_weighted_distance(a, b) == pytest.approx(
            om_boundary_weighted_distance(b, a)
        )

    def test_positive_distance(self) -> None:
        a = np.array([0, 0, 1])
        b = np.array([1, 1, 0])
        assert om_boundary_weighted_distance(a, b) > 0.0

    def test_context_factor_zero_matches_om(self) -> None:
        """With context_factor=0 the heuristic reduces to standard OM."""
        from yasqat.metrics.optimal_matching import optimal_matching_distance

        a = np.array([0, 1, 2])
        b = np.array([0, 2, 1])
        d_omloc = om_boundary_weighted_distance(a, b, context_factor=0.0)
        d_om = optimal_matching_distance(a, b)
        assert d_omloc == pytest.approx(d_om)

    def test_empty_sequences(self) -> None:
        empty = np.array([], dtype=np.int64)
        assert om_boundary_weighted_distance(empty, empty) == 0.0

    def test_normalize(self) -> None:
        a = np.array([0, 1, 2])
        b = np.array([1, 0, 2])
        d_raw = om_boundary_weighted_distance(a, b)
        d_norm = om_boundary_weighted_distance(a, b, normalize=True)
        assert d_norm == pytest.approx(d_raw / 3)


class TestOmSpellScaled:
    """Tests for the spell-scaled OM heuristic."""

    def test_identical_sequences(self) -> None:
        seq = np.array([0, 0, 1, 1])
        assert om_spell_scaled_distance(seq, seq) == pytest.approx(0.0)

    def test_symmetric(self) -> None:
        a = np.array([0, 0, 1])
        b = np.array([0, 1, 1])
        assert om_spell_scaled_distance(a, b) == pytest.approx(
            om_spell_scaled_distance(b, a)
        )

    def test_long_spell_lower_cost(self) -> None:
        """Substitutions within long spells should cost less."""
        # Both differ at position 2, but a_long has a longer spell there
        a_long = np.array([0, 0, 0, 0, 1])  # long spell of 0
        a_short = np.array([0, 1, 0, 1, 1])  # short spells

        b = np.array([0, 0, 0, 0, 0])  # all same

        d_long = om_spell_scaled_distance(a_long, b)
        d_short = om_spell_scaled_distance(a_short, b)
        # d_long should be smaller because the difference is within a long spell context
        assert d_long < d_short

    def test_empty_sequences(self) -> None:
        empty = np.array([], dtype=np.int64)
        assert om_spell_scaled_distance(empty, empty) == 0.0

    def test_normalize(self) -> None:
        a = np.array([0, 0, 1])
        b = np.array([1, 1, 0])
        d_raw = om_spell_scaled_distance(a, b)
        d_norm = om_spell_scaled_distance(a, b, normalize=True)
        assert d_norm == pytest.approx(d_raw / 3)


class TestOmTransitionPenalty:
    """Tests for the transition-penalty OM heuristic."""

    def test_identical_sequences(self) -> None:
        seq = np.array([0, 1, 0, 1])
        assert om_transition_penalty_distance(seq, seq) == pytest.approx(0.0)

    def test_symmetric(self) -> None:
        a = np.array([0, 1, 2])
        b = np.array([0, 2, 1])
        assert om_transition_penalty_distance(a, b) == pytest.approx(
            om_transition_penalty_distance(b, a)
        )

    def test_transition_weight_zero_matches_om(self) -> None:
        """With otto=0 the heuristic reduces to standard OM."""
        from yasqat.metrics.optimal_matching import optimal_matching_distance

        a = np.array([0, 1, 2])
        b = np.array([0, 2, 1])
        d_omstran = om_transition_penalty_distance(a, b, otto=0.0)
        d_om = optimal_matching_distance(a, b)
        assert d_omstran == pytest.approx(d_om)

    def test_custom_transition_weights(self) -> None:
        a = np.array([0, 1, 0])
        b = np.array([0, 2, 0])
        tw = np.array(
            [[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]],
            dtype=np.float64,
        )
        d = om_transition_penalty_distance(a, b, transition_weights=tw)
        assert d > 0.0

    def test_empty_sequences(self) -> None:
        empty = np.array([], dtype=np.int64)
        assert om_transition_penalty_distance(empty, empty) == 0.0

    def test_normalize(self) -> None:
        a = np.array([0, 1, 2])
        b = np.array([2, 1, 0])
        d_raw = om_transition_penalty_distance(a, b)
        d_norm = om_transition_penalty_distance(a, b, normalize=True)
        assert d_norm == pytest.approx(d_raw / 3)
