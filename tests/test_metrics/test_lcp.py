"""Tests for LCP (Longest Common Prefix) distance."""

import numpy as np

from yasqat.metrics.lcp import LCPMetric, lcp_distance, lcp_length, lcp_similarity


class TestLCPDistance:
    """Tests for LCP distance."""

    def test_identical_sequences(self) -> None:
        """Test LCP of identical sequences."""
        seq = np.array([0, 1, 2, 3], dtype=np.int32)

        length = lcp_length(seq, seq)
        dist = lcp_distance(seq, seq)

        assert length == 4
        assert dist == 0.0

    def test_no_common_prefix(self) -> None:
        """Test when sequences have no common prefix."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([3, 4, 5], dtype=np.int32)

        length = lcp_length(seq_a, seq_b)
        dist = lcp_distance(seq_a, seq_b)

        assert length == 0
        assert dist == 6.0  # len(a) + len(b) - 2*0

    def test_partial_common_prefix(self) -> None:
        """Test sequences with partial common prefix."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([0, 1, 4, 5], dtype=np.int32)

        length = lcp_length(seq_a, seq_b)
        dist = lcp_distance(seq_a, seq_b)

        assert length == 2  # 0, 1
        assert dist == 4.0  # 4 + 4 - 2*2

    def test_one_is_prefix_of_other(self) -> None:
        """Test when one sequence is prefix of the other."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 1, 2, 3, 4], dtype=np.int32)

        length = lcp_length(seq_a, seq_b)
        dist = lcp_distance(seq_a, seq_b)

        assert length == 3  # All of seq_a
        assert dist == 2.0  # 3 + 5 - 2*3 = 2

    def test_normalized_distance(self) -> None:
        """Test normalized LCP distance."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([0, 1, 4, 5], dtype=np.int32)

        dist = lcp_distance(seq_a, seq_b, normalize=True)

        # Unnormalized: 4 + 4 - 2*2 = 4
        # Normalized: 4 / 8 = 0.5
        assert dist == 0.5

    def test_similarity_identical(self) -> None:
        """Test LCP similarity for identical sequences."""
        seq = np.array([0, 1, 2, 3], dtype=np.int32)

        sim = lcp_similarity(seq, seq)

        assert sim == 1.0

    def test_similarity_no_common(self) -> None:
        """Test LCP similarity with no common prefix."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([3, 4, 5], dtype=np.int32)

        sim = lcp_similarity(seq_a, seq_b)

        assert sim == 0.0

    def test_similarity_partial(self) -> None:
        """Test partial LCP similarity."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([0, 1, 4, 5], dtype=np.int32)

        sim = lcp_similarity(seq_a, seq_b)

        # LCP is 2, min length is 4
        assert sim == 0.5

    def test_empty_sequences(self) -> None:
        """Test with empty sequences."""
        seq_a = np.array([], dtype=np.int32)
        seq_b = np.array([], dtype=np.int32)

        length = lcp_length(seq_a, seq_b)
        dist = lcp_distance(seq_a, seq_b)
        sim = lcp_similarity(seq_a, seq_b)

        assert length == 0
        assert dist == 0.0
        assert sim == 1.0  # Both empty = identical

    def test_one_empty_sequence(self) -> None:
        """Test with one empty sequence."""
        seq_a = np.array([0, 1, 2], dtype=np.int32)
        seq_b = np.array([], dtype=np.int32)

        length = lcp_length(seq_a, seq_b)
        sim = lcp_similarity(seq_a, seq_b)

        assert length == 0
        assert sim == 0.0

    def test_symmetric(self) -> None:
        """Test that distance is symmetric."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([0, 1, 4], dtype=np.int32)

        dist_ab = lcp_distance(seq_a, seq_b)
        dist_ba = lcp_distance(seq_b, seq_a)

        assert dist_ab == dist_ba

    def test_metric_class(self) -> None:
        """Test LCPMetric class."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([0, 1, 4, 5], dtype=np.int32)

        metric = LCPMetric(normalize=False)
        dist = metric.compute(seq_a, seq_b)

        assert isinstance(dist, float)
        assert dist >= 0

    def test_metric_class_normalized(self) -> None:
        """Test LCPMetric with normalization."""
        seq_a = np.array([0, 1, 2, 3], dtype=np.int32)
        seq_b = np.array([0, 1, 4, 5], dtype=np.int32)

        metric = LCPMetric(normalize=True)
        dist = metric.compute(seq_a, seq_b)

        assert 0 <= dist <= 1

    def test_different_from_lcs(self) -> None:
        """Test that LCP differs from LCS."""
        # LCP only considers prefix, LCS considers any subsequence
        seq_a = np.array([1, 0, 1, 2], dtype=np.int32)
        seq_b = np.array([0, 1, 2, 3], dtype=np.int32)

        # LCP = 0 (first elements differ)
        lcp = lcp_length(seq_a, seq_b)

        # But LCS would be 3 ([0, 1, 2])
        assert lcp == 0
