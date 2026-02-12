"""Tests for filtering criteria."""

import polars as pl
import pytest

from yasqat.core.sequence import StateSequence
from yasqat.filters.criteria import (
    ContainsStateCriterion,
    LengthCriterion,
    PatternCriterion,
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
        """Test filtering by containing any of the states."""
        criterion = ContainsStateCriterion(states=["D"])
        matching = criterion.get_matching_ids(test_sequence)

        assert 1 not in matching  # no D
        assert 2 not in matching  # no D
        assert 3 in matching  # has D

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


class TestPatternCriterion:
    """Tests for PatternCriterion."""

    def test_exact_pattern(self, test_sequence: StateSequence) -> None:
        """Test exact pattern matching."""
        criterion = PatternCriterion(pattern="A-B-B-C")
        matching = criterion.get_matching_ids(test_sequence)

        assert 1 in matching  # A-B-B-C
        assert 2 not in matching
        assert 3 not in matching

    def test_wildcard_pattern(self, test_sequence: StateSequence) -> None:
        """Test pattern with single wildcard."""
        criterion = PatternCriterion(pattern="A-*-C")
        matching = criterion.get_matching_ids(test_sequence)

        assert 1 not in matching  # A-B-B-C (4 elements)
        assert 2 in matching  # A-A-C (3 elements)
        assert 3 not in matching

    def test_pattern_match_anywhere(self, test_sequence: StateSequence) -> None:
        """Test pattern matching anywhere in sequence."""
        criterion = PatternCriterion(pattern="C-D", match_anywhere=True)
        matching = criterion.get_matching_ids(test_sequence)

        assert 1 not in matching
        assert 2 not in matching
        assert 3 in matching  # has C-D

    def test_regex_pattern(self, test_sequence: StateSequence) -> None:
        """Test regex pattern matching."""
        criterion = PatternCriterion(
            pattern="A-.*-C", pattern_type="regex", match_anywhere=True
        )
        matching = criterion.get_matching_ids(test_sequence)

        assert 1 in matching  # A-B-B-C
        assert 2 in matching  # A-A-C
        assert 3 not in matching


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
