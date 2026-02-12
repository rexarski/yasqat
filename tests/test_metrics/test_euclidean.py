"""Tests for Euclidean distance metric."""

import numpy as np
import pytest

from yasqat.metrics.euclidean import EuclideanMetric, euclidean_distance


class TestEuclideanDistance:
    """Tests for Euclidean distance."""

    def test_identical_sequences(self) -> None:
        """Test distance between identical sequences."""
        seq = np.array([0, 0, 1, 1, 2], dtype=np.int32)
        assert euclidean_distance(seq, seq) == 0.0

    def test_same_distribution(self) -> None:
        """Test sequences with same state distribution (reordered)."""
        seq_a = np.array([0, 0, 1, 1, 2], dtype=np.int32)
        seq_b = np.array([1, 0, 2, 1, 0], dtype=np.int32)
        assert euclidean_distance(seq_a, seq_b) == pytest.approx(0.0)

    def test_completely_different(self) -> None:
        """Test completely different sequences."""
        seq_a = np.array([0, 0, 0], dtype=np.int32)
        seq_b = np.array([1, 1, 1], dtype=np.int32)
        # props_a = [1.0, 0.0], props_b = [0.0, 1.0]
        # d = sqrt(1 + 1) = sqrt(2)
        assert euclidean_distance(seq_a, seq_b) == pytest.approx(2.0**0.5)

    def test_symmetric(self, encoded_sequences: tuple[np.ndarray, np.ndarray]) -> None:
        """Test symmetry of distance."""
        seq_a, seq_b = encoded_sequences
        assert euclidean_distance(seq_a, seq_b) == pytest.approx(
            euclidean_distance(seq_b, seq_a)
        )

    def test_empty_sequences(self) -> None:
        """Test with empty sequences."""
        empty = np.array([], dtype=np.int32)
        seq = np.array([0, 1], dtype=np.int32)
        assert euclidean_distance(empty, empty) == 0.0
        assert euclidean_distance(empty, seq) == float("inf")

    def test_normalize(self) -> None:
        """Test normalized distance is in [0, 1]."""
        seq_a = np.array([0, 0, 0], dtype=np.int32)
        seq_b = np.array([1, 1, 1], dtype=np.int32)
        dist = euclidean_distance(seq_a, seq_b, normalize=True)
        # Max distance between probability vectors normalized by sqrt(2) -> 1.0
        assert dist == pytest.approx(1.0)

    def test_with_n_states(self) -> None:
        """Test explicit n_states parameter."""
        seq_a = np.array([0, 0, 1], dtype=np.int32)
        seq_b = np.array([0, 1, 1], dtype=np.int32)
        # With n_states=2 vs inferred should give same result
        d1 = euclidean_distance(seq_a, seq_b, n_states=2)
        d2 = euclidean_distance(seq_a, seq_b)
        assert d1 == pytest.approx(d2)

    def test_different_lengths(self) -> None:
        """Test sequences of different lengths."""
        seq_a = np.array([0, 0, 0, 0], dtype=np.int32)
        seq_b = np.array([0, 0], dtype=np.int32)
        # Both are 100% state 0 -> distance is 0
        assert euclidean_distance(seq_a, seq_b) == pytest.approx(0.0)


class TestEuclideanMetric:
    """Tests for EuclideanMetric class."""

    def test_basic_distance(self) -> None:
        """Test metric compute method."""
        metric = EuclideanMetric()
        seq_a = np.array([0, 0, 1, 1], dtype=np.int32)
        seq_b = np.array([0, 0, 1, 1], dtype=np.int32)

        assert metric.compute(seq_a, seq_b) == 0.0

    def test_with_normalize(self) -> None:
        """Test metric with normalization."""
        metric = EuclideanMetric(normalize=True)
        seq_a = np.array([0, 0, 0], dtype=np.int32)
        seq_b = np.array([1, 1, 1], dtype=np.int32)

        dist = metric.compute(seq_a, seq_b)
        assert 0.0 <= dist <= 1.0

    def test_name(self) -> None:
        """Test metric name."""
        metric = EuclideanMetric()
        assert metric.name == "euclidean"
