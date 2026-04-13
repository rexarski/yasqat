"""Tests for SoftDTW distance."""

import numpy as np
import pytest

from yasqat.metrics.softdtw import SoftDTWMetric, softdtw_distance, softdtw_divergence


class TestSoftDTWDistance:
    """Tests for SoftDTW distance."""

    def test_identical_sequences(self) -> None:
        """Test SoftDTW of identical sequences."""
        seq = np.array([0, 1, 2, 3], dtype=np.int32)

        dist = softdtw_distance(seq, seq, gamma=1.0)

        # Raw SoftDTW can be negative due to soft-min smoothing (log-sum-exp
        # correction makes it less than 0 for identical sequences).
        # For identical sequences, the optimal path cost is 0 along the
        # diagonal, but soft-min adds a negative correction at each step.
        assert dist < 0.0

        # The distance for identical sequences should be less than for
        # different sequences
        seq_diff = np.array([3, 2, 1, 0], dtype=np.int32)
        dist_diff = softdtw_distance(seq, seq_diff, gamma=1.0)
        assert dist < dist_diff

    def test_different_sequences(self) -> None:
        """Test SoftDTW of different sequences."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([3, 4, 5], dtype=np.int32)

        dist = softdtw_distance(seq_a, seq_b, gamma=1.0)

        # Different sequences should have larger distance than identical
        dist_same = softdtw_distance(seq_a, seq_a, gamma=1.0)
        assert dist > dist_same

    def test_gamma_effect(self) -> None:
        """Test that gamma affects smoothness."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 2, 3], dtype=np.int32)

        # Small gamma - closer to hard DTW
        dist_small = softdtw_distance(seq_a, seq_b, gamma=0.01)

        # Large gamma - softer, considers more alignment paths which
        # adds more negative correction from the log-sum-exp soft-min,
        # resulting in a smaller (more negative) value
        dist_large = softdtw_distance(seq_a, seq_b, gamma=10.0)

        # Larger gamma produces a smaller value because the soft-min
        # correction term (-gamma * log(sum(exp(...)))) grows larger
        assert dist_large < dist_small

    def test_small_gamma_approaches_dtw(self) -> None:
        """Test that small gamma approximates hard DTW."""
        seq_a = np.array([0, 0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 1, 2], dtype=np.int32)

        # Very small gamma
        dist_soft = softdtw_distance(seq_a, seq_b, gamma=0.001)

        # Should be close to DTW (which would be 0 for this pair)
        assert abs(dist_soft) < 0.1

    def test_window_constraint(self) -> None:
        """Test SoftDTW with window constraint."""
        seq_a = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        seq_b = np.array([0, 1, 2, 3, 4], dtype=np.int32)

        dist = softdtw_distance(seq_a, seq_b, gamma=1.0, window=2)

        # For identical sequences, should be small in magnitude
        assert isinstance(dist, float)

    def test_normalized(self) -> None:
        """Test normalized SoftDTW."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([3, 4, 5], dtype=np.int32)

        dist_raw = softdtw_distance(seq_a, seq_b, gamma=1.0, normalize=False)
        dist_norm = softdtw_distance(seq_a, seq_b, gamma=1.0, normalize=True)

        # Normalized should be smaller
        assert dist_norm < dist_raw

    def test_empty_sequences(self) -> None:
        """Test with empty sequences."""
        seq_a = np.array([], dtype=np.int32)
        seq_b = np.array([], dtype=np.int32)

        dist = softdtw_distance(seq_a, seq_b, gamma=1.0)

        assert dist == 0.0

    def test_one_empty_sequence(self) -> None:
        """Test with one empty sequence."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([], dtype=np.int32)

        dist = softdtw_distance(seq_a, seq_b, gamma=1.0)

        assert dist == float("inf")

    def test_invalid_gamma(self) -> None:
        """Test that non-positive gamma raises error."""
        seq = np.array([0, 1, 2], dtype=np.int32)

        with pytest.raises(ValueError, match="gamma must be positive"):
            softdtw_distance(seq, seq, gamma=0)

        with pytest.raises(ValueError, match="gamma must be positive"):
            softdtw_distance(seq, seq, gamma=-1.0)


class TestSoftDTWDivergence:
    """Tests for SoftDTW divergence."""

    def test_identical_sequences_zero_divergence(self) -> None:
        """Test that identical sequences have near-zero divergence."""
        seq = np.array([0, 1, 2, 3], dtype=np.int32)

        div = softdtw_divergence(seq, seq, gamma=1.0)

        # Divergence should be close to 0 for identical sequences
        assert abs(div) < 0.01

    def test_divergence_symmetric(self) -> None:
        """Test that divergence is symmetric."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 2, 3], dtype=np.int32)

        div_ab = softdtw_divergence(seq_a, seq_b, gamma=1.0)
        div_ba = softdtw_divergence(seq_b, seq_a, gamma=1.0)

        assert abs(div_ab - div_ba) < 1e-6

    def test_divergence_positive_for_different(self) -> None:
        """Test that divergence is positive for different sequences."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([3, 4, 5], dtype=np.int32)

        div = softdtw_divergence(seq_a, seq_b, gamma=1.0)

        # Divergence should be positive
        assert div > 0


class TestSoftDTWMetric:
    """Tests for SoftDTWMetric class."""

    def test_metric_class(self) -> None:
        """Test SoftDTWMetric class."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([0, 2, 3], dtype=np.int32)

        metric = SoftDTWMetric(gamma=1.0)
        dist = metric.compute(seq_a, seq_b)

        # Raw SoftDTW can be negative; just check it returns a float
        assert isinstance(dist, float)

    def test_metric_with_divergence(self) -> None:
        """Test SoftDTWMetric with divergence mode."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 1, 2], dtype=np.int32)

        metric = SoftDTWMetric(gamma=1.0, use_divergence=True)
        div = metric.compute(seq_a, seq_b)

        # Divergence for identical should be near 0
        assert abs(div) < 0.01

    def test_metric_with_all_options(self) -> None:
        """Test SoftDTWMetric with all options."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 2, 3], dtype=np.int32)

        metric = SoftDTWMetric(
            gamma=0.5,
            window=2,
            sm="binary",
            normalize=True,
        )
        dist = metric.compute(seq_a, seq_b)

        assert isinstance(dist, float)
