"""Tests for metric function exports."""

from __future__ import annotations

import numpy as np

from yasqat.metrics import (
    lcp_length,
    lcp_similarity,
    lcs_length,
    lcs_similarity,
    rlcp_length,
    rlcp_similarity,
)


class TestMetricExports:
    """Tests for newly exported length/similarity functions."""

    def test_lcs_length(self) -> None:
        a = np.array([0, 1, 2, 3], dtype=np.int32)
        b = np.array([0, 2, 3, 4], dtype=np.int32)
        # LCS is [0, 2, 3]: these elements appear in order in both sequences
        assert lcs_length(a, b) == 3

    def test_lcs_similarity(self) -> None:
        a = np.array([0, 1, 2, 3], dtype=np.int32)
        b = np.array([0, 2, 3, 4], dtype=np.int32)
        sim = lcs_similarity(a, b)
        # LCS length is 3, max sequence length is 4, similarity = 3/4 = 0.75
        assert sim == 0.75

    def test_lcp_length(self) -> None:
        a = np.array([0, 1, 2], dtype=np.int32)
        b = np.array([0, 1, 3], dtype=np.int32)
        # Common prefix is [0, 1]; third elements differ (2 vs 3)
        assert lcp_length(a, b) == 2

    def test_lcp_similarity(self) -> None:
        a = np.array([0, 1, 2], dtype=np.int32)
        b = np.array([0, 1, 3], dtype=np.int32)
        sim = lcp_similarity(a, b)
        # LCP length is 2, min sequence length is 3, similarity = 2/3
        assert sim == 2.0 / 3.0

    def test_rlcp_length(self) -> None:
        a = np.array([0, 1, 2], dtype=np.int32)
        b = np.array([3, 1, 2], dtype=np.int32)
        # Common suffix is [1, 2]; first elements differ (0 vs 3)
        assert rlcp_length(a, b) == 2

    def test_rlcp_similarity(self) -> None:
        a = np.array([0, 1, 2], dtype=np.int32)
        b = np.array([3, 1, 2], dtype=np.int32)
        sim = rlcp_similarity(a, b)
        # RLCP length is 2, min sequence length is 3, similarity = 2/3
        assert sim == 2.0 / 3.0
