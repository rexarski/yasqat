"""Tests for frequent subsequence mining."""

import polars as pl
import pytest

from yasqat.core.pool import SequencePool
from yasqat.statistics.subsequence_mining import frequent_subsequences


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
    def test_returns_dataframe(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=0.5)
        assert isinstance(results, pl.DataFrame)
        assert "subsequence" in results.columns
        assert "support" in results.columns
        assert "proportion" in results.columns

    def test_all_states_frequent(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=0.5)
        # Extract single-state patterns
        single_state = results.filter(
            pl.col("subsequence").list.len() == 1
        )
        patterns = single_state["subsequence"].to_list()
        assert ["A"] in patterns
        assert ["B"] in patterns
        assert ["C"] in patterns

    def test_ab_frequent(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=0.5)
        patterns = results["subsequence"].to_list()
        # "A", "B" subsequence appears in all 3 sequences
        assert ["A", "B"] in patterns

    def test_support_values(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=0.5)
        for prop in results["proportion"].to_list():
            assert prop >= 0.3  # at least min_support threshold
            assert 0.0 < prop <= 1.0

    def test_max_length(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=0.5, max_length=1)
        for subseq in results["subsequence"].to_list():
            assert len(subseq) == 1

    def test_high_support_filters(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=1.0)
        # Only patterns in ALL sequences
        for support in results["support"].to_list():
            assert support == 3

    def test_sorted_by_support(self, mining_pool: SequencePool) -> None:
        results = frequent_subsequences(mining_pool, min_support=0.3)
        supports = results["support"].to_list()
        assert supports == sorted(supports, reverse=True)

    def test_filter_by_length(self, mining_pool: SequencePool) -> None:
        """Test that results can be filtered using polars expressions."""
        results = frequent_subsequences(mining_pool, min_support=0.3)
        two_step = results.filter(pl.col("subsequence").list.len() == 2)
        assert isinstance(two_step, pl.DataFrame)
        for subseq in two_step["subsequence"].to_list():
            assert len(subseq) == 2
