"""Tests for transition statistics."""

import numpy as np
import polars as pl
import pytest

from yasqat.core.pool import SequencePool
from yasqat.statistics.transition import (
    first_occurrence_time,
    state_duration_stats,
    state_spell_stats,
    substitution_cost_matrix,
    transition_rate_matrix,
    transition_rates,
)


class TestTransitionRateMatrix:
    """Tests for transition rate matrix."""

    def test_transition_matrix_shape(self, sequence_pool: SequencePool) -> None:
        """Test that transition matrix has correct shape."""
        trate = transition_rate_matrix(sequence_pool)

        n_states = len(sequence_pool.alphabet)
        assert trate.shape == (n_states, n_states)

    def test_transition_matrix_rows_sum_to_one(
        self, sequence_pool: SequencePool
    ) -> None:
        """Test that rows sum to 1 (or 0 for states with no outgoing transitions)."""
        trate = transition_rate_matrix(sequence_pool)

        row_sums = trate.sum(axis=1)
        for s in row_sums:
            assert s == pytest.approx(1.0) or s == pytest.approx(0.0)

    def test_transition_matrix_as_counts(self, sequence_pool: SequencePool) -> None:
        """Test getting raw transition counts."""
        counts = transition_rate_matrix(sequence_pool, as_counts=True)

        # Should have integer-like values
        assert np.all(counts >= 0)
        assert np.all(counts == counts.astype(int))

    def test_transition_dataframe(self, sequence_pool: SequencePool) -> None:
        """Test transition rate DataFrame."""
        df = transition_rates(sequence_pool)

        assert "from_state" in df.columns
        assert "to_state" in df.columns
        assert "count" in df.columns
        assert "rate" in df.columns

        n_states = len(sequence_pool.alphabet)
        assert len(df) == n_states * n_states


class TestFirstOccurrence:
    """Tests for first occurrence time."""

    def test_first_occurrence(self) -> None:
        """Test finding first occurrence of a state."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2],
                "time": [0, 1, 2, 0, 1, 2],
                "state": ["A", "B", "C", "B", "C", "A"],
            }
        )
        pool = SequencePool(data)

        result = first_occurrence_time(pool, "C")

        assert len(result) == 2
        # Sequence 1 reaches C at time 2
        # Sequence 2 reaches C at time 1
        seq1_time = result.filter(pl.col("id") == 1)["first_occurrence"].item()
        seq2_time = result.filter(pl.col("id") == 2)["first_occurrence"].item()
        assert seq1_time == 2
        assert seq2_time == 1


class TestStateDurationStats:
    """Tests for state duration statistics (deprecated alias)."""

    def test_duration_stats(self, sequence_pool: SequencePool) -> None:
        """Test state duration statistics."""
        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            stats = state_duration_stats(sequence_pool)

        assert "state" in stats.columns
        assert "mean_spell_length" in stats.columns
        assert "median_spell_length" in stats.columns
        assert "min_spell_length" in stats.columns
        assert "max_spell_length" in stats.columns
        assert "n_spells" in stats.columns


class TestStateSpellStats:
    """Tests for state_spell_stats."""

    def test_basic_spell_stats(self, sequence_pool: SequencePool) -> None:
        """state_spell_stats returns spell-length stats per state."""
        result = state_spell_stats(sequence_pool)
        assert "state" in result.columns
        assert "mean_spell_length" in result.columns
        assert "median_spell_length" in result.columns
        assert "min_spell_length" in result.columns
        assert "max_spell_length" in result.columns
        assert "std_spell_length" in result.columns
        assert "n_spells" in result.columns

    def test_spell_stats_values(self, sequence_pool: SequencePool) -> None:
        """spell-length stats have positive counts and non-negative lengths."""
        result = state_spell_stats(sequence_pool)
        assert (result["n_spells"] > 0).all()
        assert (result["min_spell_length"] >= 1).all()
        assert (result["mean_spell_length"] >= 1).all()

    def test_backward_compat_alias(self, sequence_pool: SequencePool) -> None:
        """state_duration_stats still works as a deprecated alias."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = state_duration_stats(sequence_pool)
            assert any(
                issubclass(warning.category, DeprecationWarning) for warning in w
            )
        assert "mean_spell_length" in result.columns


class TestExcludeSelfTransitions:
    """Tests for exclude_self parameter."""

    def test_exclude_self_transitions_removes_diagonal(self) -> None:
        """Test that self-transitions are excluded from DataFrame."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [0, 1, 2, 3],
                "state": ["A", "A", "B", "A"],
            }
        )
        pool = SequencePool(data)
        df = transition_rates(pool, exclude_self=True)

        # No rows where from_state == to_state
        self_rows = df.filter(pl.col("from_state") == pl.col("to_state"))
        assert len(self_rows) == 0

    def test_exclude_self_default_false(self) -> None:
        """Test that exclude_self defaults to False (backward compat)."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1],
                "time": [0, 1, 2],
                "state": ["A", "A", "B"],
            }
        )
        pool = SequencePool(data)
        df = transition_rates(pool)  # default exclude_self=False

        # Should include A→A
        self_rows = df.filter(
            (pl.col("from_state") == "A") & (pl.col("to_state") == "A")
        )
        assert len(self_rows) == 1
        assert self_rows["count"].item() == 1

    def test_exclude_self_matrix(self) -> None:
        """Test exclude_self on transition_rate_matrix."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [0, 1, 2, 3],
                "state": ["A", "A", "B", "A"],
            }
        )
        pool = SequencePool(data)
        counts = transition_rate_matrix(pool, as_counts=True, exclude_self=True)

        # Diagonal should be all zeros
        assert np.allclose(np.diag(counts), 0)


class TestSubstitutionCostMatrix:
    """Tests for substitution cost matrix generation."""

    def test_constant_costs(self, sequence_pool: SequencePool) -> None:
        """Test constant substitution costs."""
        sm = substitution_cost_matrix(sequence_pool, method="constant")

        n_states = len(sequence_pool.alphabet)
        assert sm.shape == (n_states, n_states)

        # Diagonal should be zero
        assert np.allclose(np.diag(sm), 0)

        # Off-diagonal should be 2.0
        for i in range(n_states):
            for j in range(n_states):
                if i != j:
                    assert sm[i, j] == 2.0

    def test_trate_costs(self, sequence_pool: SequencePool) -> None:
        """Test transition-rate based costs."""
        sm = substitution_cost_matrix(sequence_pool, method="trate")

        n_states = len(sequence_pool.alphabet)
        assert sm.shape == (n_states, n_states)

        # Diagonal should be zero
        assert np.allclose(np.diag(sm), 0)

        # Should be symmetric
        assert np.allclose(sm, sm.T)

    def test_invalid_method(self, sequence_pool: SequencePool) -> None:
        """Test error on invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            substitution_cost_matrix(sequence_pool, method="invalid")
