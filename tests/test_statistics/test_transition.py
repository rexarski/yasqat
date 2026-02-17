"""Tests for transition statistics."""

import numpy as np
import polars as pl
import pytest

from yasqat.core.pool import SequencePool
from yasqat.statistics.transition import (
    first_occurrence_time,
    state_duration_stats,
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
    """Tests for state duration statistics."""

    def test_duration_stats(self, sequence_pool: SequencePool) -> None:
        """Test state duration statistics."""
        stats = state_duration_stats(sequence_pool)

        assert "state" in stats.columns
        assert "mean_duration" in stats.columns
        assert "median_duration" in stats.columns
        assert "min_duration" in stats.columns
        assert "max_duration" in stats.columns
        assert "n_spells" in stats.columns


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
