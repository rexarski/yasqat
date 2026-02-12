"""Filtering criteria for sequence selection."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from yasqat.core.sequence import BaseSequence


class SequenceCriterion(ABC):
    """Abstract base class for sequence filtering criteria."""

    @abstractmethod
    def get_matching_ids(self, sequence: BaseSequence) -> list[int | str]:
        """
        Return IDs of sequences that match this criterion.

        Args:
            sequence: Sequence object to filter.

        Returns:
            List of matching sequence IDs.
        """

    def filter(self, sequence: BaseSequence) -> pl.DataFrame:
        """
        Filter sequence data to only matching sequences.

        Args:
            sequence: Sequence object to filter.

        Returns:
            Filtered DataFrame containing only matching sequences.
        """
        matching_ids = self.get_matching_ids(sequence)
        id_col = sequence.config.id_column
        return sequence.data.filter(pl.col(id_col).is_in(matching_ids))


@dataclass
class LengthCriterion(SequenceCriterion):
    """
    Filter sequences by length.

    Examples:
        >>> criterion = LengthCriterion(min_length=5)  # At least 5 elements
        >>> criterion = LengthCriterion(max_length=10)  # At most 10 elements
        >>> criterion = LengthCriterion(min_length=5, max_length=10)  # Between 5 and 10
        >>> criterion = LengthCriterion(exact_length=7)  # Exactly 7 elements
    """

    min_length: int | None = None
    max_length: int | None = None
    exact_length: int | None = None

    def get_matching_ids(self, sequence: BaseSequence) -> list[int | str]:
        """Return IDs of sequences with matching length."""
        id_col = sequence.config.id_column

        # Get sequence lengths
        lengths = sequence.data.group_by(id_col).agg(pl.len().alias("_length"))

        # Apply length filters
        if self.exact_length is not None:
            lengths = lengths.filter(pl.col("_length") == self.exact_length)
        else:
            if self.min_length is not None:
                lengths = lengths.filter(pl.col("_length") >= self.min_length)
            if self.max_length is not None:
                lengths = lengths.filter(pl.col("_length") <= self.max_length)

        return lengths[id_col].to_list()


@dataclass
class TimeCriterion(SequenceCriterion):
    """
    Filter sequences by time range.

    Examples:
        >>> criterion = TimeCriterion(start_after=10)  # Sequences starting after t=10
        >>> criterion = TimeCriterion(end_before=100)  # Sequences ending before t=100
        >>> criterion = TimeCriterion(contains_time=50)  # Sequences active at t=50
    """

    start_after: int | float | None = None
    start_before: int | float | None = None
    end_after: int | float | None = None
    end_before: int | float | None = None
    contains_time: int | float | None = None

    def get_matching_ids(self, sequence: BaseSequence) -> list[int | str]:
        """Return IDs of sequences matching time criteria."""
        id_col = sequence.config.id_column
        config = sequence.config

        # Determine time columns based on sequence type
        if (
            hasattr(config, "start_column")
            and config.start_column in sequence.data.columns
        ):
            # Interval sequence
            time_stats = sequence.data.group_by(id_col).agg(
                [
                    pl.col(config.start_column).min().alias("_start"),
                    pl.col(config.end_column).max().alias("_end"),
                ]
            )
        else:
            # State or event sequence
            time_stats = sequence.data.group_by(id_col).agg(
                [
                    pl.col(config.time_column).min().alias("_start"),
                    pl.col(config.time_column).max().alias("_end"),
                ]
            )

        # Apply time filters
        if self.start_after is not None:
            time_stats = time_stats.filter(pl.col("_start") > self.start_after)
        if self.start_before is not None:
            time_stats = time_stats.filter(pl.col("_start") < self.start_before)
        if self.end_after is not None:
            time_stats = time_stats.filter(pl.col("_end") > self.end_after)
        if self.end_before is not None:
            time_stats = time_stats.filter(pl.col("_end") < self.end_before)
        if self.contains_time is not None:
            time_stats = time_stats.filter(
                (pl.col("_start") <= self.contains_time)
                & (pl.col("_end") >= self.contains_time)
            )

        return time_stats[id_col].to_list()


@dataclass
class ContainsStateCriterion(SequenceCriterion):
    """
    Filter sequences that contain specific states.

    Examples:
        >>> criterion = ContainsStateCriterion(states=["A", "B"])  # Contains A or B
        >>> criterion = ContainsStateCriterion(states=["A", "B"], require_all=True)  # Contains A and B
        >>> criterion = ContainsStateCriterion(states=["A"], exclude=True)  # Does NOT contain A
    """

    states: list[str]
    require_all: bool = False
    exclude: bool = False

    def get_matching_ids(self, sequence: BaseSequence) -> list[int | str]:
        """Return IDs of sequences containing (or not containing) specified states."""
        id_col = sequence.config.id_column
        state_col = sequence.config.state_column

        # Get states per sequence
        seq_states = sequence.data.group_by(id_col).agg(
            pl.col(state_col).unique().alias("_states")
        )

        if self.require_all:
            # All specified states must be present
            for state in self.states:
                seq_states = seq_states.filter(pl.col("_states").list.contains(state))
        else:
            # Any specified state must be present
            condition = pl.lit(False)
            for state in self.states:
                condition = condition | pl.col("_states").list.contains(state)
            seq_states = seq_states.filter(condition)

        matching = seq_states[id_col].to_list()

        if self.exclude:
            # Return sequences NOT in matching
            all_ids = sequence.data[id_col].unique().to_list()
            return [id_ for id_ in all_ids if id_ not in matching]

        return matching


@dataclass
class StartsWithCriterion(SequenceCriterion):
    """
    Filter sequences that start with specific states.

    Examples:
        >>> criterion = StartsWithCriterion(states=["A"])  # Starts with A
        >>> criterion = StartsWithCriterion(states=["A", "B"])  # Starts with A then B
    """

    states: list[str]

    def get_matching_ids(self, sequence: BaseSequence) -> list[int | str]:
        """Return IDs of sequences starting with the specified states."""
        id_col = sequence.config.id_column
        state_col = sequence.config.state_column
        time_col = sequence.config.time_column

        matching_ids = []

        for seq_id in sequence.sequence_ids():
            seq_data = sequence.data.filter(pl.col(id_col) == seq_id).sort(time_col)
            seq_states = seq_data[state_col].to_list()

            # Check if sequence starts with the required states
            if len(seq_states) >= len(self.states):
                if seq_states[: len(self.states)] == self.states:
                    matching_ids.append(seq_id)

        return matching_ids


@dataclass
class PatternCriterion(SequenceCriterion):
    """
    Filter sequences matching a pattern.

    Patterns use a simple grammar:
        - "A-B-C": Exact sequence A, B, C
        - "A-*-C": A, any single state, C
        - "A-+-C": A, one or more states, C
        - "A-?-C": A, zero or one state, C

    For more complex patterns, use regex with pattern_type="regex".

    Examples:
        >>> criterion = PatternCriterion(pattern="A-B-C")  # Exact match
        >>> criterion = PatternCriterion(pattern="A-*-C")  # A, any, C
        >>> criterion = PatternCriterion(pattern="A.*C", pattern_type="regex")
    """

    pattern: str
    pattern_type: str = "simple"  # "simple" or "regex"
    match_anywhere: bool = False  # If True, pattern can match anywhere in sequence

    def get_matching_ids(self, sequence: BaseSequence) -> list[int | str]:
        """Return IDs of sequences matching the pattern."""
        id_col = sequence.config.id_column
        state_col = sequence.config.state_column
        time_col = sequence.config.time_column

        matching_ids = []

        # Convert simple pattern to regex if needed
        if self.pattern_type == "simple":
            regex_pattern = self._simple_to_regex(self.pattern)
        else:
            regex_pattern = self.pattern

        compiled = re.compile(regex_pattern)

        for seq_id in sequence.sequence_ids():
            seq_data = sequence.data.filter(pl.col(id_col) == seq_id).sort(time_col)
            seq_states = seq_data[state_col].to_list()

            # Convert to string for pattern matching
            seq_string = "-".join(str(s) for s in seq_states)

            if self.match_anywhere:
                if compiled.search(seq_string):
                    matching_ids.append(seq_id)
            else:
                if compiled.fullmatch(seq_string):
                    matching_ids.append(seq_id)

        return matching_ids

    def _simple_to_regex(self, pattern: str) -> str:
        """Convert simple pattern to regex."""
        # Escape special regex characters except our wildcards
        parts = pattern.split("-")
        regex_parts = []

        for part in parts:
            if part == "*":
                regex_parts.append("[^-]+")  # Match any single state
            elif part == "+":
                regex_parts.append("(?:[^-]+-)*[^-]+")  # Match one or more states
            elif part == "?":
                regex_parts.append("(?:[^-]+)?")  # Match zero or one state
            else:
                regex_parts.append(re.escape(part))

        return "-".join(regex_parts) if regex_parts else ""


@dataclass
class QueryCriterion(SequenceCriterion):
    """
    Filter sequences using a polars expression.

    This provides full flexibility for complex filtering.

    Examples:
        >>> import polars as pl
        >>> # Filter sequences where any time point has state "A" at time > 5
        >>> expr = (pl.col("state") == "A") & (pl.col("time") > 5)
        >>> criterion = QueryCriterion(expression=expr)
    """

    expression: pl.Expr

    def get_matching_ids(self, sequence: BaseSequence) -> list[int | str]:
        """Return IDs of sequences where any row matches the expression."""
        id_col = sequence.config.id_column

        # Find rows matching the expression
        matching_rows = sequence.data.filter(self.expression)

        # Get unique IDs
        return matching_rows[id_col].unique().to_list()


def filter_sequences(
    sequence: BaseSequence,
    criteria: SequenceCriterion | list[SequenceCriterion],
    combine: str = "and",
) -> pl.DataFrame:
    """
    Filter sequences based on one or more criteria.

    Args:
        sequence: Sequence object to filter.
        criteria: Single criterion or list of criteria.
        combine: How to combine multiple criteria:
            - "and": All criteria must match (intersection)
            - "or": Any criterion must match (union)

    Returns:
        Filtered DataFrame containing only matching sequences.

    Example:
        >>> from yasqat.filters import filter_sequences, LengthCriterion, ContainsStateCriterion
        >>> criteria = [
        ...     LengthCriterion(min_length=5),
        ...     ContainsStateCriterion(states=["A"]),
        ... ]
        >>> filtered_df = filter_sequences(sequence, criteria)
    """
    if isinstance(criteria, SequenceCriterion):
        criteria = [criteria]

    id_col = sequence.config.id_column
    all_ids = set(sequence.sequence_ids())

    matching_ids: set[int | str]
    if combine == "and":
        # Intersection: start with all, reduce
        matching_ids = all_ids
        for criterion in criteria:
            criterion_ids = set(criterion.get_matching_ids(sequence))
            matching_ids = matching_ids & criterion_ids
    elif combine == "or":
        # Union: start with none, expand
        matching_ids = set()
        for criterion in criteria:
            criterion_ids = set(criterion.get_matching_ids(sequence))
            matching_ids = matching_ids | criterion_ids
    else:
        raise ValueError(f"Unknown combine method: {combine}. Use 'and' or 'or'.")

    return sequence.data.filter(pl.col(id_col).is_in(list(matching_ids)))
