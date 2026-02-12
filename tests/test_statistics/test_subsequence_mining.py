"""Tests for frequent subsequence mining."""

import polars as pl
import pytest

from yasqat.core.pool import SequencePool
from yasqat.statistics.subsequence_mining import (
    FrequentSubsequence,
    frequent_subsequences,
)


@pytest.fixture
def mining_pool() -> SequencePool:
    """Pool with clear frequent patterns."""
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [0, 1, 2, 0, 1, 2, 0, 1, 2],
            "state": [
                "A",
                "B",
                "C",  # seq 1
                "A",
                "B",
                "C",  # seq 2
                "A",
                "C",
                "B",  # seq 3
            ],
        }
    )
    return SequencePool(data)


class TestFrequentSubsequences:
    def test_returns_list(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=0.5)
        assert isinstance(results, list)
        assert all(isinstance(r, FrequentSubsequence) for r in results)

    def test_all_states_frequent(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=0.5)
        patterns = {r.pattern for r in results}
        # All single states appear in all sequences
        assert ("A",) in patterns
        assert ("B",) in patterns
        assert ("C",) in patterns

    def test_ab_frequent(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=0.5)
        patterns = {r.pattern for r in results}
        # "A", "B" subsequence appears in all 3 sequences
        assert ("A", "B") in patterns

    def test_support_values(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=0.5)
        for r in results:
            assert r.proportion >= 0.3  # at least min_support threshold
            assert 0.0 < r.proportion <= 1.0

    def test_max_length(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=0.5, max_length=1)
        for r in results:
            assert len(r.pattern) == 1

    def test_high_support_filters(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=1.0)
        # Only patterns in ALL sequences
        for r in results:
            assert r.support == 3

    def test_sorted_by_support(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=0.3)
        supports = [r.support for r in results]
        assert supports == sorted(supports, reverse=True)
