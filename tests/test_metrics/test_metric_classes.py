"""Tests for pairwise distance computation via SequencePool.compute_distances().

Exercises the full pipeline: SequencePool → compute_distances(method=...) →
DistanceMatrix for every supported metric. This covers dispatch logic,
encoding, and the class-based metric interfaces that individual metric
tests don't reach.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from yasqat.core.pool import SequencePool
from yasqat.metrics.base import DistanceMatrix


@pytest.fixture
def small_pool() -> SequencePool:
    """Pool with 3 short equal-length sequences over a 3-state alphabet."""
    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "state": ["A", "B", "C", "A", "A", "C", "B", "C", "C"],
        }
    )
    return SequencePool(df)


def _check_dm(dm: DistanceMatrix, n: int = 3) -> None:
    """Validate basic distance matrix invariants."""
    assert dm.shape == (n, n)
    # Diagonal is zero
    for i in range(n):
        assert dm.values[i, i] == pytest.approx(0.0, abs=1e-9)
    # Symmetric
    np.testing.assert_allclose(dm.values, dm.values.T, atol=1e-9)
    # Finite
    assert np.all(np.isfinite(dm.values))


@pytest.mark.parametrize(
    "method",
    [
        "hamming",
        "lcs",
        "lcp",
        "rlcp",
        "dtw",
        "euclidean",
        "chi2",
        "om",
        "nms",
        "nmsmst",
        "svrspell",
        "omloc",
        "omspell",
        "omstran",
    ],
)
class TestPairwiseDistanceByMethod:
    """Parametrized tests for each distance method via compute_distances()."""

    def test_returns_distance_matrix(
        self, small_pool: SequencePool, method: str
    ) -> None:
        dm = small_pool.compute_distances(method=method)
        assert isinstance(dm, DistanceMatrix)
        _check_dm(dm)

    def test_labels_match_sequence_ids(
        self, small_pool: SequencePool, method: str
    ) -> None:
        dm = small_pool.compute_distances(method=method)
        assert dm.labels is not None
        assert dm.labels == [1, 2, 3]


class TestTWEDPairwise:
    """TWED tested separately — needs explicit params and may differ in sign."""

    def test_twed_pairwise(self, small_pool: SequencePool) -> None:
        dm = small_pool.compute_distances(method="twed")
        assert isinstance(dm, DistanceMatrix)
        _check_dm(dm)


class TestUnsupportedMethod:
    """Test that unsupported methods raise clear errors."""

    def test_unknown_method_raises(self, small_pool: SequencePool) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            small_pool.compute_distances(method="nonexistent")


class TestParallelComputation:
    """Test n_jobs parameter for parallel pairwise computation."""

    def test_parallel_matches_sequential(self, small_pool: SequencePool) -> None:
        dm_seq = small_pool.compute_distances(method="hamming", n_jobs=1)
        dm_par = small_pool.compute_distances(method="hamming", n_jobs=2)
        np.testing.assert_allclose(dm_seq.values, dm_par.values, atol=1e-12)
