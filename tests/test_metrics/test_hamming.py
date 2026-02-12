"""Tests for Hamming distance."""

import numpy as np
import pytest

from yasqat.metrics.hamming import HammingMetric, hamming_distance


class TestHammingDistance:
    """Tests for Hamming distance."""

    def test_identical_sequences(self) -> None:
        """Test distance between identical sequences is zero."""
        seq = np.array([0, 1, 2, 3], dtype=np.int32)

        dist = hamming_distance(seq, seq)

        assert dist == 0.0

    def test_single_mismatch(self) -> None:
        """Test distance with single mismatch."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([0, 1, 2, 4], dtype=np.int32)

        dist = hamming_distance(seq_a, seq_b)

        assert dist == 1.0

    def test_all_different(self) -> None:
        """Test distance when all positions differ."""
        seq_a = np.array([0, 0, 0, 0], dtype=np.int32)
        seq_b = np.array([1, 1, 1, 1], dtype=np.int32)

        dist = hamming_distance(seq_a, seq_b)

        assert dist == 4.0

    def test_normalized(self) -> None:
        """Test normalized distance."""
        seq_a = np.array([0, 0, 0, 0], dtype=np.int32)
        seq_b = np.array([1, 1, 0, 0], dtype=np.int32)

        dist = hamming_distance(seq_a, seq_b, normalize=True)

        # 2 mismatches out of 4 = 0.5
        assert dist == 0.5

    def test_unequal_length_error(self) -> None:
        """Test error when sequences have different lengths."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 1, 2, 3], dtype=np.int32)

        with pytest.raises(ValueError, match="same length"):
            hamming_distance(seq_a, seq_b)

    def test_empty_sequences(self) -> None:
        """Test distance between empty sequences."""
        seq_a = np.array([], dtype=np.int32)
        seq_b = np.array([], dtype=np.int32)

        dist = hamming_distance(seq_a, seq_b)

        assert dist == 0.0

    def test_symmetric(
        self, equal_length_sequences: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that distance is symmetric."""
        seq_a, seq_b = equal_length_sequences

        dist_ab = hamming_distance(seq_a, seq_b)
        dist_ba = hamming_distance(seq_b, seq_a)

        assert dist_ab == dist_ba

    def test_metric_class(
        self, equal_length_sequences: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test HammingMetric class."""
        seq_a, seq_b = equal_length_sequences

        metric = HammingMetric(normalize=False)
        dist = metric.compute(seq_a, seq_b)

        assert isinstance(dist, float)
        assert dist >= 0
