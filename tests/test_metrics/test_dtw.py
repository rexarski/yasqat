"""Tests for DTW distance."""

import numpy as np

from yasqat.metrics.dtw import DTWMetric, dtw_distance


class TestDTWDistance:
    """Tests for DTW distance."""

    def test_identical_sequences(self) -> None:
        """Test DTW of identical sequences."""
        seq = np.array([0, 1, 2, 3], dtype=np.int32)

        dist = dtw_distance(seq, seq)

        assert dist == 0.0

    def test_different_lengths_same_pattern(self) -> None:
        """Test DTW with time warping (repeated elements)."""
        seq_a = np.array([0, 0, 1, 2, 2], dtype=np.int32)
        seq_b = np.array([0, 1, 2], dtype=np.int32)

        dist = dtw_distance(seq_a, seq_b)

        # Should align perfectly with warping
        assert dist == 0.0

    def test_completely_different(self) -> None:
        """Test DTW of completely different sequences."""
        seq_a = np.array([0, 0, 0], dtype=np.int32)
        seq_b = np.array([1, 1, 1], dtype=np.int32)

        dist = dtw_distance(seq_a, seq_b)

        # Each element differs, cost = 1 per element pair
        assert dist == 3.0

    def test_with_window_constraint(self) -> None:
        """Test DTW with Sakoe-Chiba window."""
        seq_a = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        seq_b = np.array([0, 1, 2, 3, 4], dtype=np.int32)

        # Should still find optimal alignment
        dist = dtw_distance(seq_a, seq_b, window=2)

        assert dist == 0.0

    def test_window_affects_distance(self) -> None:
        """Test that small window can increase distance."""
        seq_a = np.array([0, 0, 0, 0, 1], dtype=np.int32)
        seq_b = np.array([1, 0, 0, 0, 0], dtype=np.int32)

        # No window - can warp freely
        dist_no_window = dtw_distance(seq_a, seq_b, window=0)

        # Small window - limited warping
        dist_window = dtw_distance(seq_a, seq_b, window=1)

        # Window constraint should increase or maintain distance
        assert dist_window >= dist_no_window

    def test_normalized(self) -> None:
        """Test normalized DTW distance."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([3, 4, 5], dtype=np.int32)

        dist_raw = dtw_distance(seq_a, seq_b, normalize=False)
        dist_norm = dtw_distance(seq_a, seq_b, normalize=True)

        # Normalized should be smaller
        assert dist_norm < dist_raw
        assert dist_norm > 0

    def test_symmetric(self) -> None:
        """Test that DTW is symmetric."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([0, 2, 4], dtype=np.int32)

        dist_ab = dtw_distance(seq_a, seq_b)
        dist_ba = dtw_distance(seq_b, seq_a)

        assert dist_ab == dist_ba

    def test_empty_sequences(self) -> None:
        """Test with empty sequences."""
        seq_a = np.array([], dtype=np.int32)
        seq_b = np.array([], dtype=np.int32)

        dist = dtw_distance(seq_a, seq_b)

        assert dist == 0.0

    def test_one_empty_sequence(self) -> None:
        """Test with one empty sequence."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([], dtype=np.int32)

        dist = dtw_distance(seq_a, seq_b)

        assert dist == float("inf")

    def test_custom_substitution_matrix(self) -> None:
        """Test with custom substitution costs."""
        seq_a = np.array([0, 1], dtype=np.int32)
        seq_b = np.array([0, 2], dtype=np.int32)

        # Custom matrix where 1->2 costs 5
        sm = np.array(
            [
                [0, 1, 5],
                [1, 0, 5],
                [5, 5, 0],
            ],
            dtype=np.float64,
        )

        dist = dtw_distance(seq_a, seq_b, sm=sm)

        assert dist == 5.0  # Cost of 1->2

    def test_binary_substitution(self) -> None:
        """Test with binary substitution (0/1)."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 1, 3], dtype=np.int32)

        dist = dtw_distance(seq_a, seq_b, sm="binary")

        # Only last element differs
        assert dist == 1.0

    def test_metric_class(self) -> None:
        """Test DTWMetric class."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([0, 2, 3], dtype=np.int32)

        metric = DTWMetric(window=0, normalize=False)
        dist = metric.compute(seq_a, seq_b)

        assert isinstance(dist, float)
        assert dist >= 0

    def test_metric_class_with_config(self) -> None:
        """Test DTWMetric with configuration."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 2, 3], dtype=np.int32)

        metric = DTWMetric(window=2, sm="binary", normalize=True)
        dist = metric.compute(seq_a, seq_b)

        assert isinstance(dist, float)
        assert 0 <= dist <= 1  # Normalized

    def test_triangle_inequality_may_not_hold(self) -> None:
        """DTW does not always satisfy triangle inequality."""
        # This is a property test, not a failure test
        seq_a = np.array([0, 1], dtype=np.int32)
        seq_b = np.array([1, 0], dtype=np.int32)
        seq_c = np.array([0, 0], dtype=np.int32)

        dist_ab = dtw_distance(seq_a, seq_b)
        dist_bc = dtw_distance(seq_b, seq_c)
        dist_ac = dtw_distance(seq_a, seq_c)

        # Just verify these are valid distances
        assert all(d >= 0 for d in [dist_ab, dist_bc, dist_ac])
