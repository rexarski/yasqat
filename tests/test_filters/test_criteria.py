"""Tests for filtering criteria."""

from __future__ import annotations

import polars as pl
import pytest

from yasqat.core.pool import SequencePool
from yasqat.core.sequence import StateSequence
from yasqat.filters.criteria import (
    ContainsStateCriterion,
    LengthCriterion,
    QueryCriterion,
    StartsWithCriterion,
    TimeCriterion,
    filter_sequences,
)


@pytest.fixture
def test_sequence() -> StateSequence:
    """Create a test sequence for filtering."""
    data = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
            "time": [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4],
            "state": [
                "A",
                "B",
                "B",
                "C",  # Seq 1: length 4
                "A",
                "A",
                "C",  # Seq 2: length 3
                "B",
                "C",
                "C",
                "D",
                "D",  # Seq 3: length 5
            ],
        }
    )
    return StateSequence(data)


class TestLengthCriterion:
    """Tests for LengthCriterion."""

    def test_min_length(self, test_sequence: StateSequence) -> None:
        """Test filtering by minimum length."""
        criterion = LengthCriterion(min_length=4)
        matching = criterion.get_matching_ids(test_sequence)

        assert 1 in matching  # length 4
        assert 2 not in matching  # length 3
        assert 3 in matching  # length 5

    def test_max_length(self, test_sequence: StateSequence) -> None:
        """Test filtering by maximum length."""
        criterion = LengthCriterion(max_length=4)
        matching = criterion.get_matching_ids(test_sequence)

        assert 1 in matching  # length 4
        assert 2 in matching  # length 3
        assert 3 not in matching  # length 5

    def test_exact_length(self, test_sequence: StateSequence) -> None:
        """Test filtering by exact length."""
        criterion = LengthCriterion(exact_length=4)
        matching = criterion.get_matching_ids(test_sequence)

        assert 1 in matching  # length 4
        assert 2 not in matching  # length 3
        assert 3 not in matching  # length 5

    def test_length_range(self, test_sequence: StateSequence) -> None:
        """Test filtering by length range."""
        criterion = LengthCriterion(min_length=3, max_length=4)
        matching = criterion.get_matching_ids(test_sequence)

        assert 1 in matching  # length 4
        assert 2 in matching  # length 3
        assert 3 not in matching  # length 5

    def test_length_criterion_min_equals_max(
        self, test_sequence: StateSequence
    ) -> None:
        """min_length == max_length should work like exact_length."""
        criterion_range = LengthCriterion(min_length=4, max_length=4)
        criterion_exact = LengthCriterion(exact_length=4)

        matching_range = sorted(criterion_range.get_matching_ids(test_sequence))
        matching_exact = sorted(criterion_exact.get_matching_ids(test_sequence))

        assert matching_range == matching_exact
        assert 1 in matching_range  # length 4
        assert 2 not in matching_range  # length 3
        assert 3 not in matching_range  # length 5


class TestTimeCriterion:
    """Tests for TimeCriterion."""

    def test_end_after(self, test_sequence: StateSequence) -> None:
        """Test filtering by end time."""
        criterion = TimeCriterion(end_after=2)
        matching = criterion.get_matching_ids(test_sequence)

        # end_after=2 means end > 2
        assert 1 in matching  # ends at 3, 3 > 2 = True
        assert 2 not in matching  # ends at 2, 2 > 2 = False
        assert 3 in matching  # ends at 4, 4 > 2 = True

    def test_contains_time(self, test_sequence: StateSequence) -> None:
        """Test filtering by containing a time point."""
        criterion = TimeCriterion(contains_time=3)
        matching = criterion.get_matching_ids(test_sequence)

        assert 1 in matching  # has t=3
        assert 2 not in matching  # ends at t=2
        assert 3 in matching  # has t=3


class TestContainsStateCriterion:
    """Tests for ContainsStateCriterion."""

    def test_contains_any_state(self, test_sequence: StateSequence) -> None:
        """Test filtering by containing any of multiple query states."""
        # Single state
        criterion_single = ContainsStateCriterion(states=["D"])
        matching_single = criterion_single.get_matching_ids(test_sequence)
        assert 1 not in matching_single
        assert 2 not in matching_single
        assert 3 in matching_single

        # Multiple query states: should match sequences containing A OR D
        criterion_multi = ContainsStateCriterion(states=["A", "D"])
        matching_multi = criterion_multi.get_matching_ids(test_sequence)
        assert 1 in matching_multi  # has A
        assert 2 in matching_multi  # has A
        assert 3 in matching_multi  # has D

    def test_contains_state_not_in_alphabet(self, test_sequence: StateSequence) -> None:
        """Filtering by a state not in the data should return empty."""
        criterion = ContainsStateCriterion(states=["Z"])
        matching = criterion.get_matching_ids(test_sequence)
        assert matching == []

    def test_contains_all_states(self, test_sequence: StateSequence) -> None:
        """Test filtering by containing all states."""
        criterion = ContainsStateCriterion(states=["A", "C"], require_all=True)
        matching = criterion.get_matching_ids(test_sequence)

        assert 1 in matching  # has A and C
        assert 2 in matching  # has A and C
        assert 3 not in matching  # no A

    def test_exclude_state(self, test_sequence: StateSequence) -> None:
        """Test filtering by excluding states."""
        criterion = ContainsStateCriterion(states=["D"], exclude=True)
        matching = criterion.get_matching_ids(test_sequence)

        assert 1 in matching  # no D
        assert 2 in matching  # no D
        assert 3 not in matching  # has D


class TestStartsWithCriterion:
    """Tests for StartsWithCriterion."""

    def test_starts_with_single(self, test_sequence: StateSequence) -> None:
        """Test filtering by starting state."""
        criterion = StartsWithCriterion(states=["A"])
        matching = criterion.get_matching_ids(test_sequence)

        assert 1 in matching  # starts with A
        assert 2 in matching  # starts with A
        assert 3 not in matching  # starts with B

    def test_starts_with_multiple(self, test_sequence: StateSequence) -> None:
        """Test filtering by starting sequence."""
        criterion = StartsWithCriterion(states=["A", "B"])
        matching = criterion.get_matching_ids(test_sequence)

        assert 1 in matching  # starts with A, B
        assert 2 not in matching  # starts with A, A
        assert 3 not in matching  # starts with B, C


class TestQueryCriterion:
    """Tests for QueryCriterion."""

    def test_query_criterion(self, test_sequence: StateSequence) -> None:
        """Test query-based filtering."""
        # Filter sequences that have state 'D'
        expr = pl.col("state") == "D"
        criterion = QueryCriterion(expression=expr)
        matching = criterion.get_matching_ids(test_sequence)

        assert 1 not in matching
        assert 2 not in matching
        assert 3 in matching

    def test_complex_query(self, test_sequence: StateSequence) -> None:
        """Test complex query."""
        # Sequences with state B at time > 0
        expr = (pl.col("state") == "B") & (pl.col("time") > 0)
        criterion = QueryCriterion(expression=expr)
        matching = criterion.get_matching_ids(test_sequence)

        assert 1 in matching  # B at times 1, 2
        assert 2 not in matching  # no B
        assert 3 not in matching  # B only at time 0


class TestFilterSequences:
    """Tests for filter_sequences function."""

    def test_single_criterion(self, test_sequence: StateSequence) -> None:
        """Test filtering with single criterion."""
        criterion = LengthCriterion(min_length=4)
        filtered = filter_sequences(test_sequence, criterion)

        # Should only have sequences 1 and 3
        unique_ids = filtered["id"].unique().sort().to_list()
        assert unique_ids == [1, 3]

    def test_multiple_criteria_and(self, test_sequence: StateSequence) -> None:
        """Test filtering with multiple criteria (AND)."""
        criteria = [
            LengthCriterion(min_length=4),
            ContainsStateCriterion(states=["A"]),
        ]
        filtered = filter_sequences(test_sequence, criteria, combine="and")

        # Only sequence 1 has length >= 4 AND contains A
        unique_ids = filtered["id"].unique().to_list()
        assert unique_ids == [1]

    def test_multiple_criteria_or(self, test_sequence: StateSequence) -> None:
        """Test filtering with multiple criteria (OR)."""
        criteria = [
            LengthCriterion(exact_length=3),  # Seq 2
            ContainsStateCriterion(states=["D"]),  # Seq 3
        ]
        filtered = filter_sequences(test_sequence, criteria, combine="or")

        unique_ids = filtered["id"].unique().sort().to_list()
        assert unique_ids == [2, 3]

    def test_invalid_combine(self, test_sequence: StateSequence) -> None:
        """Test error with invalid combine method."""
        criterion = LengthCriterion(min_length=1)

        with pytest.raises(ValueError, match="Unknown combine method"):
            filter_sequences(test_sequence, criterion, combine="invalid")


class TestCriterionFilter:
    """Tests for criterion.filter() method."""

    def test_filter_method(self, test_sequence: StateSequence) -> None:
        """Test the filter() method on criterion."""
        criterion = LengthCriterion(min_length=4)
        filtered = criterion.filter(test_sequence)

        unique_ids = filtered["id"].unique().sort().to_list()
        assert unique_ids == [1, 3]


class TestStartsWithCriterionVectorized:
    def test_single_state_prefix(self) -> None:
        """StartsWithCriterion should find sequences starting with given state."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "time": [0, 1, 2, 0, 1, 2, 0, 1, 2],
                "state": ["A", "B", "C", "A", "A", "B", "B", "A", "C"],
            }
        )
        seq = StateSequence(data)
        criterion = StartsWithCriterion(states=["A"])
        ids = criterion.get_matching_ids(seq)
        assert sorted(ids) == [1, 2]

    def test_multi_state_prefix(self) -> None:
        """StartsWithCriterion should match multi-state prefixes."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2],
                "time": [0, 1, 2, 0, 1, 2],
                "state": ["A", "B", "C", "A", "B", "B"],
            }
        )
        seq = StateSequence(data)
        criterion = StartsWithCriterion(states=["A", "B"])
        ids = criterion.get_matching_ids(seq)
        assert sorted(ids) == [1, 2]

    def test_empty_states_returns_all(self) -> None:
        """Empty states list should return all sequence IDs."""
        data = pl.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "time": [0, 1, 0, 1],
                "state": ["A", "B", "C", "D"],
            }
        )
        seq = StateSequence(data)
        criterion = StartsWithCriterion(states=[])
        ids = criterion.get_matching_ids(seq)
        assert sorted(ids) == [1, 2]


class TestFiltersAcceptPool:
    """Filters accept either container via the SequenceData surface."""

    def test_criterion_with_pool(self, test_sequence: StateSequence) -> None:
        """get_matching_ids gives identical results for pool and sequence."""
        pool = SequencePool.coerce(test_sequence)
        criterion = LengthCriterion(min_length=4)

        assert (
            sorted(criterion.get_matching_ids(pool))
            == sorted(criterion.get_matching_ids(test_sequence))
            == [1, 3]
        )

    def test_filter_sequences_with_pool(self, test_sequence: StateSequence) -> None:
        """filter_sequences accepts a SequencePool directly."""
        pool = SequencePool.coerce(test_sequence)
        df = filter_sequences(pool, ContainsStateCriterion(states=["D"]))

        assert df["id"].unique().to_list() == [3]

    def test_criterion_filter_with_pool(self, test_sequence: StateSequence) -> None:
        """SequenceCriterion.filter accepts a SequencePool directly."""
        pool = SequencePool.coerce(test_sequence)
        df = LengthCriterion(exact_length=3).filter(pool)

        assert df["id"].unique().to_list() == [2]
