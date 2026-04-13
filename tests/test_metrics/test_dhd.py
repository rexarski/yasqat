"""Tests for Dynamic Hamming Distance (DHD) metric."""

import numpy as np
import polars as pl
import pytest

from yasqat.core.pool import SequencePool
from yasqat.metrics.dhd import DHDMetric, build_position_costs, dhd_distance


class TestBuildPositionCosts:
    """Tests for building position-dependent costs."""

    def test_shape(self, sequence_pool: SequencePool) -> None:
        """Test that position costs have correct shape."""
        costs = build_position_costs(sequence_pool)

        n_states = len(sequence_pool.alphabet)
        # 4 time points, 4 states (A,B,C,D)
        assert costs.shape == (4, n_states, n_states)

    def test_diagonal_zero(self, sequence_pool: SequencePool) -> None:
        """Test that diagonal costs are zero (same state = no cost)."""
        costs = build_position_costs(sequence_pool)

        for t in range(costs.shape[0]):
            for s in range(costs.shape[1]):
                assert costs[t, s, s] == 0.0

    def test_symmetric(self, sequence_pool: SequencePool) -> None:
        """Test that cost matrices are symmetric at each position."""
        costs = build_position_costs(sequence_pool)

        for t in range(costs.shape[0]):
            np.testing.assert_array_almost_equal(costs[t], costs[t].T)

    def test_nonnegative(self, sequence_pool: SequencePool) -> None:
        """Test that all costs are non-negative."""
        costs = build_position_costs(sequence_pool)
        assert np.all(costs >= 0)

    def test_unequal_lengths_raises(self) -> None:
        """Test that unequal-length sequences raise ValueError."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2],
                "time": [0, 1, 2, 0, 1],
                "state": ["A", "B", "C", "A", "B"],
            }
        )
        pool = SequencePool(data)

        with pytest.raises(ValueError, match="same length"):
            build_position_costs(pool)


class TestDHDDistance:
    """Tests for DHD distance."""

    def test_identical_sequences(self, sequence_pool: SequencePool) -> None:
        """Test distance between identical sequences."""
        costs = build_position_costs(sequence_pool)
        seq = np.array([0, 1, 2, 3], dtype=np.int32)

        assert dhd_distance(seq, seq, costs) == 0.0

    def test_different_sequences(self, sequence_pool: SequencePool) -> None:
        """Test distance between different sequences."""
        costs = build_position_costs(sequence_pool)
        seq_a = np.array([0, 0, 1, 2], dtype=np.int32)
        seq_b = np.array([1, 1, 2, 3], dtype=np.int32)

        dist = dhd_distance(seq_a, seq_b, costs)
        assert dist > 0.0

    def test_symmetric(self, sequence_pool: SequencePool) -> None:
        """Test symmetry of distance."""
        costs = build_position_costs(sequence_pool)
        seq_a = np.array([0, 0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 1, 1, 2], dtype=np.int32)

        assert dhd_distance(seq_a, seq_b, costs) == pytest.approx(
            dhd_distance(seq_b, seq_a, costs)
        )

    def test_unequal_length_raises(self, sequence_pool: SequencePool) -> None:
        """Test that unequal lengths raise ValueError."""
        costs = build_position_costs(sequence_pool)
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 1], dtype=np.int32)

        with pytest.raises(ValueError, match="same length"):
            dhd_distance(seq_a, seq_b, costs)

    def test_empty_sequences(self) -> None:
        """Test with empty sequences."""
        costs = np.zeros((0, 4, 4), dtype=np.float64)
        empty = np.array([], dtype=np.int32)

        assert dhd_distance(empty, empty, costs) == 0.0

    def test_normalize(self, sequence_pool: SequencePool) -> None:
        """Test normalized distance."""
        costs = build_position_costs(sequence_pool)
        seq_a = np.array([0, 0, 1, 2], dtype=np.int32)
        seq_b = np.array([1, 1, 2, 3], dtype=np.int32)

        dist_raw = dhd_distance(seq_a, seq_b, costs)
        dist_norm = dhd_distance(seq_a, seq_b, costs, normalize=True)

        assert dist_norm == pytest.approx(dist_raw / 4.0)

    def test_single_mismatch_uses_position_cost(
        self, sequence_pool: SequencePool
    ) -> None:
        """Test that distance for single mismatch equals the position cost."""
        costs = build_position_costs(sequence_pool)
        seq_a = np.array([0, 0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 1, 1, 2], dtype=np.int32)

        dist = dhd_distance(seq_a, seq_b, costs)

        # Only position 1 differs (state 0 vs state 1), so distance equals
        # the position-dependent cost at position 1 for states (0, 1)
        assert dist == costs[1, 0, 1]


class TestDHDMetric:
    """Tests for DHDMetric class."""

    def test_basic_distance(self, sequence_pool: SequencePool) -> None:
        """Test metric compute gives same result as bare function."""
        costs = build_position_costs(sequence_pool)
        metric = DHDMetric(position_costs=costs)
        seq_a = np.array([0, 0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 1, 1, 2], dtype=np.int32)

        dist = metric.compute(seq_a, seq_b)

        # Only position 1 differs, so distance = cost at position 1 for (0,1)
        assert dist == costs[1, 0, 1]
