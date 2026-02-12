"""Tests for OM variants: OMloc, OMspell, OMstran."""

import numpy as np
import pytest

from yasqat.metrics.om_variants import (
    omloc_distance,
    omspell_distance,
    omstran_distance,
)


class TestOMloc:
    """Tests for OMloc (Localized OM)."""

    def test_identical_sequences(self) -> None:
        seq = np.array([0, 1, 2, 3])
        assert omloc_distance(seq, seq) == pytest.approx(0.0)

    def test_symmetric(self) -> None:
        a = np.array([0, 1, 2])
        b = np.array([0, 2, 1])
        assert omloc_distance(a, b) == pytest.approx(omloc_distance(b, a))

    def test_positive_distance(self) -> None:
        a = np.array([0, 0, 1])
        b = np.array([1, 1, 0])
        assert omloc_distance(a, b) > 0.0

    def test_context_factor_zero_matches_om(self) -> None:
        """With context_factor=0, OMloc should behave like standard OM."""
        from yasqat.metrics.optimal_matching import optimal_matching

        a = np.array([0, 1, 2])
        b = np.array([0, 2, 1])
        d_omloc = omloc_distance(a, b, context_factor=0.0)
        d_om = optimal_matching(a, b)
        assert d_omloc == pytest.approx(d_om)

    def test_empty_sequences(self) -> None:
        empty = np.array([], dtype=np.int64)
        assert omloc_distance(empty, empty) == 0.0

    def test_normalize(self) -> None:
        a = np.array([0, 1, 2])
        b = np.array([1, 0, 2])
        d_raw = omloc_distance(a, b)
        d_norm = omloc_distance(a, b, normalize=True)
        assert d_norm == pytest.approx(d_raw / 3)


class TestOMspell:
    """Tests for OMspell (Spell-length sensitive OM)."""

    def test_identical_sequences(self) -> None:
        seq = np.array([0, 0, 1, 1])
        assert omspell_distance(seq, seq) == pytest.approx(0.0)

    def test_symmetric(self) -> None:
        a = np.array([0, 0, 1])
        b = np.array([0, 1, 1])
        assert omspell_distance(a, b) == pytest.approx(omspell_distance(b, a))

    def test_long_spell_lower_cost(self) -> None:
        """Substitutions within long spells should cost less."""
        # Both differ at position 2, but a_long has a longer spell there
        a_long = np.array([0, 0, 0, 0, 1])  # long spell of 0
        a_short = np.array([0, 1, 0, 1, 1])  # short spells

        b = np.array([0, 0, 0, 0, 0])  # all same

        d_long = omspell_distance(a_long, b)
        d_short = omspell_distance(a_short, b)
        # d_long should be smaller because the difference is within a long spell context
        assert d_long < d_short

    def test_empty_sequences(self) -> None:
        empty = np.array([], dtype=np.int64)
        assert omspell_distance(empty, empty) == 0.0

    def test_normalize(self) -> None:
        a = np.array([0, 0, 1])
        b = np.array([1, 1, 0])
        d_raw = omspell_distance(a, b)
        d_norm = omspell_distance(a, b, normalize=True)
        assert d_norm == pytest.approx(d_raw / 3)


class TestOMstran:
    """Tests for OMstran (Transition-sensitive OM)."""

    def test_identical_sequences(self) -> None:
        seq = np.array([0, 1, 0, 1])
        assert omstran_distance(seq, seq) == pytest.approx(0.0)

    def test_symmetric(self) -> None:
        a = np.array([0, 1, 2])
        b = np.array([0, 2, 1])
        assert omstran_distance(a, b) == pytest.approx(omstran_distance(b, a))

    def test_transition_weight_zero_matches_om(self) -> None:
        """With otto=0, OMstran should behave like standard OM."""
        from yasqat.metrics.optimal_matching import optimal_matching

        a = np.array([0, 1, 2])
        b = np.array([0, 2, 1])
        d_omstran = omstran_distance(a, b, otto=0.0)
        d_om = optimal_matching(a, b)
        assert d_omstran == pytest.approx(d_om)

    def test_custom_transition_weights(self) -> None:
        a = np.array([0, 1, 0])
        b = np.array([0, 2, 0])
        tw = np.array(
            [[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]],
            dtype=np.float64,
        )
        d = omstran_distance(a, b, transition_weights=tw)
        assert d > 0.0

    def test_empty_sequences(self) -> None:
        empty = np.array([], dtype=np.int64)
        assert omstran_distance(empty, empty) == 0.0

    def test_normalize(self) -> None:
        a = np.array([0, 1, 2])
        b = np.array([2, 1, 0])
        d_raw = omstran_distance(a, b)
        d_norm = omstran_distance(a, b, normalize=True)
        assert d_norm == pytest.approx(d_raw / 3)
