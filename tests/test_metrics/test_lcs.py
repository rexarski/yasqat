"""Tests for LCS distance."""

import numpy as np

from yasqat.metrics.lcs import LCSMetric, lcs_distance, lcs_length, lcs_similarity


class TestLCSDistance:
    """Tests for LCS distance."""

    def test_identical_sequences(self) -> None:
        """Test LCS of identical sequences."""
        seq = np.array([0, 1, 2, 3], dtype=np.int32)

        length = lcs_length(seq, seq)
        dist = lcs_distance(seq, seq)

        assert length == 4
        assert dist == 0.0

    def test_no_common_elements(self) -> None:
        """Test when sequences have no common elements."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([3, 4, 5], dtype=np.int32)

        length = lcs_length(seq_a, seq_b)
        dist = lcs_distance(seq_a, seq_b)

        assert length == 0
        assert dist == 6.0  # len(a) + len(b) - 2*0

    def test_subsequence(self) -> None:
        """Test when one is subsequence of the other."""
        seq_a = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        seq_b = np.array([0, 2, 4], dtype=np.int32)

        length = lcs_length(seq_a, seq_b)
        dist = lcs_distance(seq_a, seq_b)

        assert length == 3  # 0, 2, 4
        assert dist == 2.0  # 5 + 3 - 2*3 = 2

    def test_reordered(self) -> None:
        """Test with reordered elements."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([0, 2, 1, 3], dtype=np.int32)

        length = lcs_length(seq_a, seq_b)

        # LCS could be [0, 1, 3] or [0, 2, 3]
        assert length == 3

    def test_normalized_distance(self) -> None:
        """Test normalized LCS distance."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([0, 2, 4], dtype=np.int32)

        dist = lcs_distance(seq_a, seq_b, normalize=True)

        # Unnormalized: 4 + 3 - 2*2 = 3
        # Normalized: 3 / 7 ≈ 0.43
        assert 0 <= dist <= 1

    def test_similarity(self) -> None:
        """Test LCS similarity."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([0, 1, 2, 3], dtype=np.int32)

        sim = lcs_similarity(seq_a, seq_b)

        assert sim == 1.0

    def test_similarity_partial(self) -> None:
        """Test partial similarity."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([0, 2, 4, 5], dtype=np.int32)

        sim = lcs_similarity(seq_a, seq_b)

        # LCS is [0, 2], length 2
        # Similarity = 2 / 4 = 0.5
        assert sim == 0.5

    def test_empty_sequences(self) -> None:
        """Test with empty sequences."""
        seq_a = np.array([], dtype=np.int32)
        seq_b = np.array([], dtype=np.int32)

        length = lcs_length(seq_a, seq_b)
        dist = lcs_distance(seq_a, seq_b)
        sim = lcs_similarity(seq_a, seq_b)

        assert length == 0
        assert dist == 0.0
        assert sim == 1.0

    def test_symmetric(
        self, unequal_length_sequences: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test that distance is symmetric."""
        seq_a, seq_b = unequal_length_sequences

        dist_ab = lcs_distance(seq_a, seq_b)
        dist_ba = lcs_distance(seq_b, seq_a)

        assert dist_ab == dist_ba

    def test_metric_class(
        self, unequal_length_sequences: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test LCSMetric class."""
        seq_a, seq_b = unequal_length_sequences

        metric = LCSMetric(normalize=False)
        dist = metric.compute(seq_a, seq_b)

        assert isinstance(dist, float)
        assert dist >= 0
