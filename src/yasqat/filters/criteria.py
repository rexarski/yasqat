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
            Filtered DataFrame containing only matching sequences. To continue
            working with a typed sequence object, wrap the result::

                df = criterion.filter(seq)
                filtered = StateSequence(df, config=seq.config, alphabet=seq.alphabet)
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

    Time units are determined by the values in the sequence's time column.
    Comparisons are performed directly without unit conversion, so the
    threshold values you pass (e.g. ``start_after=10``) must be expressed
    in the same scale as the data (integer indices, timestamps, years, etc.).

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

        n_prefix = len(self.states)
        if n_prefix == 0:
            return sequence.sequence_ids

        # Add row number within each sequence (sorted by time)
        data = sequence.data.sort([id_col, time_col])
        data = data.with_columns(
            pl.col(state_col).cum_count().over(id_col).alias("_pos")
        )

        # Filter to only the first n_prefix positions per sequence
        prefix_data = data.filter(pl.col("_pos") <= n_prefix)

        # Group by id, collect the prefix states as a list
        prefixes = prefix_data.group_by(id_col, maintain_order=True).agg(
            pl.col(state_col).alias("_prefix")
        )

        # Filter: prefix list must equal the target states
        target = self.states
        matching = prefixes.filter(pl.col("_prefix") == target)

        return matching[id_col].to_list()


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
        """Return IDs of sequences matching the pattern.

        Vectorized via polars: sequences are joined into a single string per
        id using ``group_by + str.join``, then filtered with a single
        ``str.contains`` call. The regex is compiled once, outside the loop.
        """
        id_col = sequence.config.id_column
        state_col = sequence.config.state_column
        time_col = sequence.config.time_column

        # Decide which separator the joined sequence string should use, and
        # therefore which regex dialect to emit. Wildcards require the
        # null-byte separator so state names containing '-' don't collide
        # with the grammar; literal patterns keep '-' for backwards compat.
        _simple_has_wildcards = self.pattern_type == "simple" and any(
            f"-{w}" in self.pattern or f"{w}-" in self.pattern or self.pattern == w
            for w in ("*", "+", "?")
        )
        sep = "\x00" if _simple_has_wildcards else "-"

        if self.pattern_type == "simple":
            regex_pattern = self._simple_to_regex(self.pattern)
        else:
            regex_pattern = self.pattern

        # Anchor the pattern when a full-sequence match was requested. polars'
        # str.contains defaults to "search anywhere", matching Python re.search.
        if not self.match_anywhere:
            regex_pattern = f"^(?:{regex_pattern})$"

        # Build one joined string per sequence, preserving temporal order.
        joined = (
            sequence.data.sort([id_col, time_col])
            .group_by(id_col, maintain_order=True)
            .agg(pl.col(state_col).cast(pl.String).str.join(sep).alias("_joined"))
        )

        # Single vectorized regex pass over all sequences.
        matched = joined.filter(pl.col("_joined").str.contains(regex_pattern))
        return matched[id_col].to_list()

    def _simple_to_regex(self, pattern: str) -> str:
        """Convert simple pattern to regex.

        When the pattern contains wildcards (``*``, ``+``, ``?``), the
        internal sequence string uses ``\\x00`` as the state separator and
        ``-`` adjacent to a wildcard is treated as the grammar separator.
        Non-wildcard segments between wildcards are treated as literal state
        names (which may themselves contain ``-``).

        When the pattern contains *no* wildcards, the sequence string uses
        ``-`` as the separator (legacy behaviour) and the pattern is split
        on ``-`` into individual state literals, exactly as before.  This
        keeps ``"A-B-B-C"`` meaning four states while still allowing
        ``"A-1"`` to match the literal state name ``"A-1"`` via substring
        search (``match_anywhere=True``).

        Examples::

            "A-B-C"    → no wildcards → regex A-B-C  (matched against A-B-C)
            "A-*-C"    → wildcards    → regex A\\x00[^\\x00]+\\x00C
            "A-1-*-C"  → wildcards    → regex A-1\\x00[^\\x00]+\\x00C
            "A-1"      → no wildcards → regex A-1    (matched against A-1-B-2)
        """
        wildcards = {"*", "+", "?"}
        has_wildcards = any(
            f"-{w}" in pattern or f"{w}-" in pattern or pattern == w for w in wildcards
        )

        if not has_wildcards:
            # Legacy path: '-'-joined seq_string, split pattern on '-'.
            parts = pattern.split("-")
            regex_parts = [re.escape(p) for p in parts if p]
            return "-".join(regex_parts)

        # Wildcard path: '\x00'-joined seq_string.
        sep = "\x00"
        escaped_sep = re.escape(sep)
        not_sep = f"[^{re.escape(sep)}]"

        # Split only on '-' immediately adjacent to a wildcard token.
        parts = re.split(r"(?<=[*+?])-|-(?=[*+?])", pattern)

        # Emit each non-first segment with its leading separator *absorbed*
        # into the segment.  For wildcard segments we fold the separator
        # inside the quantifier group so that "?"/"*" can truly match zero
        # occurrences — otherwise the surrounding separators would still be
        # mandatory and patterns like "A-?-C" against "A\x00C" could never
        # match.  See v0.3.2 hot-fix F2.
        wildcard_regex_parts: list[str] = []
        is_first = True
        for part in parts:
            if not part:
                continue

            if part == "*":
                # Exactly one middle state (any single state). Absorb the
                # leading separator so the emit chain stays flat.
                wildcard_regex_parts.append(f"{escaped_sep}{not_sep}+")
                is_first = False
            elif part == "+":
                # One or more middle states, each with a leading separator.
                wildcard_regex_parts.append(f"(?:{escaped_sep}{not_sep}+)+")
                is_first = False
            elif part == "?":
                # Zero or one middle state, with the separator folded inside
                # so zero-middle sequences like ``A\x00C`` still match.
                wildcard_regex_parts.append(f"(?:{escaped_sep}{not_sep}+)?")
                is_first = False
            else:
                # Literal state name. Prepend a separator only if this is not
                # the first emitted piece, so the chain reads
                # ``A\x00(…)\x00C`` while still collapsing cleanly when the
                # middle piece matches zero times.
                literal = re.escape(part)
                if is_first:
                    wildcard_regex_parts.append(literal)
                else:
                    wildcard_regex_parts.append(f"{escaped_sep}{literal}")
                is_first = False

        return "".join(wildcard_regex_parts)


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
        Filtered DataFrame containing only matching sequences. To continue
        working with a typed sequence object, wrap the result::

            df = filter_sequences(seq, criteria)
            filtered = StateSequence(df, config=seq.config, alphabet=seq.alphabet)

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
    all_ids = set(sequence.sequence_ids)

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
