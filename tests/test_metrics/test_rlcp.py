"""Tests for Reverse Longest Common Prefix (RLCP) distance metric."""

import numpy as np
import pytest

from yasqat.metrics.rlcp import RLCPMetric, rlcp_distance, rlcp_length, rlcp_similarity


class TestRLCPLength:
    """Tests for RLCP length computation."""

    def test_identical_sequences(self) -> None:
        """Test RLCP of identical sequences."""
        seq = np.array([0, 1, 2, 3], dtype=np.int32)
        assert rlcp_length(seq, seq) == 4

    def test_no_common_suffix(self) -> None:
        """Test when sequences have no common suffix."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([4, 5, 6, 7], dtype=np.int32)
        assert rlcp_length(seq_a, seq_b) == 0

    def test_partial_suffix(self) -> None:
        """Test with partial common suffix."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([4, 5, 2, 3], dtype=np.int32)
        assert rlcp_length(seq_a, seq_b) == 2

    def test_one_is_suffix(self) -> None:
        """Test when one sequence is a suffix of the other."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([2, 3], dtype=np.int32)
        assert rlcp_length(seq_a, seq_b) == 2

    def test_empty_sequences(self) -> None:
        """Test with empty sequences."""
        empty = np.array([], dtype=np.int32)
        seq = np.array([0, 1], dtype=np.int32)
        assert rlcp_length(empty, empty) == 0
        assert rlcp_length(empty, seq) == 0


class TestRLCPDistance:
    """Tests for RLCP distance."""

    def test_identical_sequences(self) -> None:
        """Test distance between identical sequences."""
        seq = np.array([0, 1, 2, 3], dtype=np.int32)
        assert rlcp_distance(seq, seq) == 0.0

    def test_no_common_suffix(self) -> None:
        """Test distance with no common suffix."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([3, 4, 5], dtype=np.int32)
        # d = 3 + 3 - 2*0 = 6
        assert rlcp_distance(seq_a, seq_b) == 6.0

    def test_partial_suffix(self) -> None:
        """Test distance with partial common suffix."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([4, 5, 2, 3], dtype=np.int32)
        # d = 4 + 4 - 2*2 = 4
        assert rlcp_distance(seq_a, seq_b) == 4.0

    def test_normalize(self) -> None:
        """Test normalized distance."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([4, 5, 2, 3], dtype=np.int32)
        # d_norm = 4 / (4+4) = 0.5
        assert rlcp_distance(seq_a, seq_b, normalize=True) == pytest.approx(0.5)

    def test_empty_sequences(self) -> None:
        """Test with empty sequences."""
        empty = np.array([], dtype=np.int32)
        assert rlcp_distance(empty, empty) == 0.0

    def test_symmetric(self, encoded_sequences: tuple[np.ndarray, np.ndarray]) -> None:
        """Test symmetry of distance."""
        seq_a, seq_b = encoded_sequences
        assert rlcp_distance(seq_a, seq_b) == rlcp_distance(seq_b, seq_a)


class TestRLCPSimilarity:
    """Tests for RLCP similarity."""

    def test_identical_sequences(self) -> None:
        """Test similarity of identical sequences."""
        seq = np.array([0, 1, 2, 3], dtype=np.int32)
        assert rlcp_similarity(seq, seq) == 1.0

    def test_no_common_suffix(self) -> None:
        """Test similarity with no common suffix."""
        seq_a = np.array([0, 1], dtype=np.int32)
        seq_b = np.array([2, 3], dtype=np.int32)
        assert rlcp_similarity(seq_a, seq_b) == 0.0

    def test_partial_suffix(self) -> None:
        """Test partial suffix similarity."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([4, 5, 2, 3], dtype=np.int32)
        # sim = 2 / min(4,4) = 0.5
        assert rlcp_similarity(seq_a, seq_b) == 0.5

    def test_empty_sequences(self) -> None:
        """Test with empty sequences."""
        empty = np.array([], dtype=np.int32)
        seq = np.array([0, 1], dtype=np.int32)
        assert rlcp_similarity(empty, empty) == 1.0
        assert rlcp_similarity(empty, seq) == 0.0


class TestRLCPMetric:
    """Tests for RLCPMetric class."""

    def test_basic_distance(self) -> None:
        """Test metric compute method."""
        metric = RLCPMetric()
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([4, 5, 2, 3], dtype=np.int32)

        dist = metric.compute(seq_a, seq_b)

        assert dist == 4.0

    def test_with_normalize(self) -> None:
        """Test metric with normalization."""
        metric = RLCPMetric(normalize=True)
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([4, 5, 2, 3], dtype=np.int32)

        dist = metric.compute(seq_a, seq_b)

        assert 0.0 <= dist <= 1.0

    def test_name(self) -> None:
        """Test metric name."""
        metric = RLCPMetric()
        assert metric.name == "rlcp"
