"""Sequence classes for state and event sequences."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

from yasqat.core.alphabet import Alphabet

if TYPE_CHECKING:
    import numpy as np


@dataclass
class SequenceConfig:
    """Configuration for sequence data structure."""

    id_column: str = "id"
    time_column: str = "time"
    state_column: str = "state"
    start_column: str = "start"
    end_column: str = "end"
    granularity: str | None = None
    """Time-unit truncation applied to the time column(s) at construction.

    Accepts any polars :py:meth:`Expr.dt.truncate` unit string — e.g.
    ``"1d"``, ``"1w"``, ``"1mo"``, ``"1h"``, ``"15m"``, ``"1y"``. When set,
    the time column (state/event sequences) or both the start and end
    columns (interval sequences) are rounded **down** to the unit boundary,
    so observations within the same bucket collapse into a single time
    point for all downstream operations.

    Requires the relevant column(s) to have a polars datetime/date dtype —
    a clear ``ValueError`` is raised otherwise. Numeric granularities were
    removed in v0.3.2 (hot-fix A6); if you have integer-indexed time,
    pre-bucket it with ``(pl.col("t") // k * k)`` before constructing the
    sequence."""


class BaseSequence(ABC):
    """Abstract base class for sequences."""

    def __init__(
        self,
        data: pl.DataFrame,
        config: SequenceConfig | None = None,
        alphabet: Alphabet | None = None,
    ) -> None:
        """
        Initialize a sequence.

        Args:
            data: Polars DataFrame containing sequence data.
            config: Column configuration.
            alphabet: State alphabet (inferred if not provided).

        Raises:
            ValueError: If the supplied ``alphabet`` does not cover every
                state observed in ``data[config.state_column]``. (The reverse
                — an alphabet declaring more states than appear in the data
                — is fine, since users often declare a domain alphabet
                larger than the observed sample.)
        """
        self._config = config or SequenceConfig()
        self._data = self._validate_and_prepare(data)

        # Only validate user-supplied alphabets — an inferred alphabet is a
        # function of the data and therefore cannot mismatch it by construction
        # (see v0.3.2 hot-fix A2). The `is not None` narrowing is kept on the
        # direct branch so mypy can see ``self._alphabet`` is always ``Alphabet``.
        if alphabet is not None:
            self._alphabet = alphabet
            self._validate_alphabet_covers_data()
        else:
            self._alphabet = self._infer_alphabet()

    def _validate_alphabet_covers_data(self) -> None:
        """Raise if observed states aren't a subset of the declared alphabet.

        Only called when the user explicitly passed an ``alphabet`` argument.
        The check is silent about alphabet states that don't appear in the
        data — that's a legitimate use of a wider domain alphabet.
        """
        state_col = self._config.state_column
        if state_col not in self._data.columns:
            # _validate_and_prepare will have already raised for missing
            # columns in subclasses that require them; interval sequences
            # that don't carry a state column bypass this check.
            return

        observed = set(self._data[state_col].drop_nulls().unique().to_list())
        declared = set(self._alphabet.states)
        missing = observed - declared
        if missing:
            raise ValueError(
                f"Alphabet does not cover all observed states. "
                f"Missing from alphabet: {sorted(map(str, missing))}. "
                f"Alphabet has {len(declared)} states; data contains "
                f"{len(observed)} distinct states."
            )

    @abstractmethod
    def _validate_and_prepare(self, data: pl.DataFrame) -> pl.DataFrame:
        """Validate and prepare the input data."""

    @abstractmethod
    def _infer_alphabet(self) -> Alphabet:
        """Infer the alphabet from the data."""

    def _apply_granularity(
        self, data: pl.DataFrame, time_cols: list[str]
    ) -> pl.DataFrame:
        """Truncate listed datetime columns to ``config.granularity``.

        No-op when ``config.granularity`` is ``None``. Raises ``ValueError``
        if granularity is set but any listed column is not a datetime/date
        dtype — see the ``SequenceConfig.granularity`` docstring for the
        full rationale (v0.3.2 hot-fix A6).
        """
        granularity = self._config.granularity
        if granularity is None:
            return data

        exprs = []
        for col in time_cols:
            if col not in data.columns:
                continue
            dtype = data.schema[col]
            if not dtype.is_temporal():
                raise ValueError(
                    f"SequenceConfig.granularity={granularity!r} requires "
                    f"column {col!r} to be a polars datetime/date dtype; "
                    f"got {dtype}. Either drop granularity or cast the "
                    f"column to Datetime before constructing the sequence."
                )
            exprs.append(pl.col(col).dt.truncate(granularity).alias(col))

        if exprs:
            data = data.with_columns(exprs)
        return data

    @property
    def data(self) -> pl.DataFrame:
        """Return the underlying DataFrame."""
        return self._data

    @property
    def alphabet(self) -> Alphabet:
        """Return the state alphabet."""
        return self._alphabet

    @property
    def config(self) -> SequenceConfig:
        """Return the column configuration."""
        return self._config

    def __len__(self) -> int:
        """Return the number of rows in the sequence data."""
        return len(self._data)

    @abstractmethod
    def n_sequences(self) -> int:
        """Return the number of unique sequences."""

    @property
    @abstractmethod
    def sequence_ids(self) -> list[int | str]:
        """Return the unique sequence IDs."""


@dataclass
class StateSequence(BaseSequence):
    """
    State sequence representation.

    A state sequence represents a series of categorical states over time,
    where each time point has exactly one state.

    The data should be in long format:
    - id_column: Identifier for each sequence
    - time_column: Time point (integer or datetime)
    - state_column: State value at that time point
    """

    _data: pl.DataFrame = field(default_factory=pl.DataFrame)
    _config: SequenceConfig = field(default_factory=SequenceConfig)
    _alphabet: Alphabet = field(default_factory=lambda: Alphabet(states=()))

    def __init__(
        self,
        data: pl.DataFrame,
        config: SequenceConfig | None = None,
        alphabet: Alphabet | None = None,
    ) -> None:
        """Initialize a state sequence."""
        super().__init__(data, config, alphabet)

    def _validate_and_prepare(self, data: pl.DataFrame) -> pl.DataFrame:
        """Validate and prepare state sequence data."""
        required_cols = [
            self._config.id_column,
            self._config.time_column,
            self._config.state_column,
        ]

        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Apply granularity truncation before sorting so downstream ops see
        # the bucketed time values (v0.3.2 hot-fix A6).
        data = self._apply_granularity(data, [self._config.time_column])

        # Sort by id and time
        return data.sort([self._config.id_column, self._config.time_column])

    def _infer_alphabet(self) -> Alphabet:
        """Infer alphabet from state column."""
        return Alphabet.from_series(self._data[self._config.state_column])

    def n_sequences(self) -> int:
        """Return the number of unique sequences."""
        return self._data[self._config.id_column].n_unique()

    @property
    def sequence_ids(self) -> list[int | str]:
        """Return the unique sequence IDs."""
        return self._data[self._config.id_column].unique().sort().to_list()

    def get_sequence(self, seq_id: int | str) -> pl.DataFrame:
        """Get data for a specific sequence ID."""
        return self._data.filter(pl.col(self._config.id_column) == seq_id)

    def to_sts(self) -> pl.DataFrame:
        """Return State-Time-Sequence format (the default format)."""
        return self._data

    def to_sps(self) -> pl.DataFrame:
        """
        Convert to State-Permanence-Sequence format (run-length encoded).

        Returns a DataFrame with columns:
        - id_column: Sequence identifier
        - spell_id: Spell number within sequence
        - state_column: State value
        - start: Start time of spell
        - end: End time of spell
        - duration: Spell duration (number of observations in the spell)

        Note:
            Under v0.3.2 hot-fix A6, ``SequenceConfig.granularity`` truncates
            datetime time columns at construction rather than rewriting
            duration arithmetic here — so ``to_sps`` now always counts rows
            per spell and there is no per-call granularity branch.
        """
        id_col = self._config.id_column
        time_col = self._config.time_column
        state_col = self._config.state_column

        agg_exprs = [
            pl.col(state_col).first().alias(state_col),
            pl.col(time_col).min().alias("start"),
            pl.col(time_col).max().alias("end"),
            pl.len().alias("duration"),
        ]

        return (
            self._data.with_columns(
                [
                    # Mark state changes
                    (
                        (pl.col(state_col) != pl.col(state_col).shift(1))
                        | (pl.col(id_col) != pl.col(id_col).shift(1))
                    )
                    .fill_null(True)
                    .cum_sum()
                    .alias("spell_id")
                ]
            )
            .group_by([id_col, "spell_id"])
            .agg(agg_exprs)
            .sort([id_col, "spell_id"])
        )

    def to_dss(self) -> pl.DataFrame:
        """
        Convert to Distinct-Successive-States format.

        Removes consecutive duplicate states, keeping only transitions.
        """
        id_col = self._config.id_column
        time_col = self._config.time_column
        state_col = self._config.state_column

        return (
            self._data.with_columns(
                [
                    (
                        (pl.col(state_col) != pl.col(state_col).shift(1))
                        | (pl.col(id_col) != pl.col(id_col).shift(1))
                    )
                    .fill_null(True)
                    .alias("is_transition")
                ]
            )
            .filter(pl.col("is_transition"))
            .drop("is_transition")
            .sort([id_col, time_col])
        )

    def encode_states(self) -> np.ndarray:
        """Encode states as integer indices using the alphabet."""
        states = self._data[self._config.state_column].to_list()
        return self._alphabet.encode(states)

    def get_states_for_sequence(self, seq_id: int | str) -> list[str]:
        """Get the list of states for a specific sequence."""
        seq_data = self.get_sequence(seq_id).sort(self._config.time_column)
        return seq_data[self._config.state_column].to_list()


@dataclass
class IntervalSequence(BaseSequence):
    """
    Interval sequence representation.

    An interval sequence represents a series of duration-based events,
    where each event has a start time, end time, and a state/category.
    Intervals may overlap.

    The data should be in long format:
    - id_column: Identifier for each sequence
    - start_column: Start time of the interval
    - end_column: End time of the interval
    - state_column: State/category during the interval

    Use cases: treatments, hospital stays, projects, employment periods.
    """

    _data: pl.DataFrame = field(default_factory=pl.DataFrame)
    _config: SequenceConfig = field(default_factory=SequenceConfig)
    _alphabet: Alphabet = field(default_factory=lambda: Alphabet(states=()))

    def __init__(
        self,
        data: pl.DataFrame,
        config: SequenceConfig | None = None,
        alphabet: Alphabet | None = None,
    ) -> None:
        """Initialize an interval sequence."""
        super().__init__(data, config, alphabet)

    def _validate_and_prepare(self, data: pl.DataFrame) -> pl.DataFrame:
        """Validate and prepare interval sequence data."""
        required_cols = [
            self._config.id_column,
            self._config.start_column,
            self._config.end_column,
            self._config.state_column,
        ]

        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Validate that end >= start
        invalid = data.filter(
            pl.col(self._config.end_column) < pl.col(self._config.start_column)
        )
        if len(invalid) > 0:
            raise ValueError(f"Found {len(invalid)} intervals where end < start")

        # Truncate both endpoint columns to config.granularity so intervals
        # collapse to the same bucket boundaries (v0.3.2 hot-fix A6).
        data = self._apply_granularity(
            data, [self._config.start_column, self._config.end_column]
        )

        # Sort by id and start time
        return data.sort([self._config.id_column, self._config.start_column])

    def _infer_alphabet(self) -> Alphabet:
        """Infer alphabet from state column."""
        return Alphabet.from_series(self._data[self._config.state_column])

    def n_sequences(self) -> int:
        """Return the number of unique sequences."""
        return self._data[self._config.id_column].n_unique()

    @property
    def sequence_ids(self) -> list[int | str]:
        """Return the unique sequence IDs."""
        return self._data[self._config.id_column].unique().sort().to_list()

    def get_sequence(self, seq_id: int | str) -> pl.DataFrame:
        """Get intervals for a specific sequence ID."""
        return self._data.filter(pl.col(self._config.id_column) == seq_id)

    def duration(self) -> pl.DataFrame:
        """
        Compute duration for each interval.

        Returns DataFrame with an additional 'duration' column.
        """
        return self._data.with_columns(
            (pl.col(self._config.end_column) - pl.col(self._config.start_column)).alias(
                "duration"
            )
        )

    def total_duration_by_state(self) -> pl.DataFrame:
        """
        Compute total duration spent in each state across all sequences.

        Returns DataFrame with state and total_duration columns.
        """
        return (
            self.duration()
            .group_by(self._config.state_column)
            .agg(pl.col("duration").sum().alias("total_duration"))
            .sort("total_duration", descending=True)
        )

    def intervals_per_sequence(self) -> pl.DataFrame:
        """Count intervals per sequence."""
        return (
            self._data.group_by(self._config.id_column)
            .agg(pl.len().alias("n_intervals"))
            .sort(self._config.id_column)
        )

    def overlapping_intervals(self, seq_id: int | str) -> pl.DataFrame:
        """
        Find overlapping intervals within a sequence.

        Returns pairs of overlapping interval indices.
        """
        seq_data = self.get_sequence(seq_id).with_row_index("_idx")
        start_col = self._config.start_column
        end_col = self._config.end_column

        # Cross join to find all pairs
        overlaps = (
            seq_data.join(seq_data, how="cross", suffix="_other")
            .filter(
                (pl.col("_idx") < pl.col("_idx_other"))
                & (pl.col(start_col) < pl.col(f"{end_col}_other"))
                & (pl.col(end_col) > pl.col(f"{start_col}_other"))
            )
            .select(["_idx", "_idx_other"])
        )
        return overlaps

    def has_overlaps(self) -> bool:
        """Check if any sequence has overlapping intervals."""
        id_col = self._config.id_column
        start_col = self._config.start_column
        end_col = self._config.end_column

        # Sort by sequence ID and start time, then check if any interval
        # starts before the previous one (within the same sequence) ends
        sorted_data = self._data.sort([id_col, start_col])
        has_any = (
            sorted_data.with_columns(
                pl.col(end_col).shift(1).over(id_col).alias("_prev_end")
            )
            .filter(pl.col("_prev_end").is_not_null())
            .filter(pl.col(start_col) < pl.col("_prev_end"))
        )
        return len(has_any) > 0

    def to_state_sequence(self, time_points: list[int] | None = None) -> StateSequence:
        """
        Convert interval sequence to state sequence by sampling at time points.

        If multiple intervals contain a time point, the interval with the
        **latest start time** is chosen (the most recently started active
        interval). This matches the common "last write wins" semantic for
        time-indexed state inference.

        Args:
            time_points: List of time points to sample. If None, uses integer
                         range from min start to max end.

        Returns:
            StateSequence sampled at the given time points.

        Note:
            Vectorized via ``polars.join_asof`` (v0.3.2). Replaces the previous
            per-sequence Python loop with a single O((S·T) + I) pass, where
            S is the number of sequences, T the number of time points, and I
            the number of intervals. The pre-v0.3.2 tiebreaker was "earliest
            start among intervals containing t", which is a side-effect of
            row order rather than a deliberate rule — the new "latest start"
            tiebreaker is intentional and consistent with join_asof backward
            search.
        """
        id_col = self._config.id_column
        start_col = self._config.start_column
        end_col = self._config.end_column
        state_col = self._config.state_column

        if time_points is None:
            min_start_val = self._data[start_col].min()
            max_end_val = self._data[end_col].max()
            if min_start_val is None or max_end_val is None:
                return StateSequence(
                    data=pl.DataFrame(
                        schema={id_col: pl.Int64, "time": pl.Int64, state_col: pl.Utf8}
                    ),
                    config=self._config,
                    alphabet=self._alphabet,
                )
            # Cast to int for range()
            min_start_int = int(float(min_start_val))  # type: ignore[arg-type]
            max_end_int = int(float(max_end_val))  # type: ignore[arg-type]
            time_points = list(range(min_start_int, max_end_int + 1))

        seq_ids = self._data[id_col].unique(maintain_order=True).to_list()
        if not seq_ids or not time_points:
            return StateSequence(
                data=pl.DataFrame(
                    schema={id_col: pl.Int64, "time": pl.Int64, state_col: pl.Utf8}
                ),
                config=self._config,
                alphabet=self._alphabet,
            )

        # Build the (id, time) cartesian grid. join_asof requires the left
        # frame sorted ascending on the asof key ("time") within each `by`
        # group (id_col) — sort once here and tag the column as sorted so
        # polars skips the "sortedness cannot be checked" warning.
        grid = (
            pl.DataFrame({id_col: seq_ids})
            .join(pl.DataFrame({"time": time_points}), how="cross")
            .sort([id_col, "time"])
            .with_columns(pl.col("time").set_sorted())
        )

        # Right side: intervals already sorted by (id_col, start_col) via
        # _validate_and_prepare. Tag start_col as sorted for the same reason.
        intervals = self._data.select(
            [id_col, start_col, end_col, state_col]
        ).with_columns(pl.col(start_col).set_sorted())

        # Polars always emits "Sortedness of columns cannot be checked when
        # 'by' groups provided" for join_asof+by; the invariant is preserved
        # by the explicit .sort() above, so suppress the informational noise.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Sortedness of columns cannot be checked",
                category=UserWarning,
            )
            joined = grid.join_asof(
                intervals,
                left_on="time",
                right_on=start_col,
                by=id_col,
                strategy="backward",
            )

        result = (
            joined
            # Backward asof gives the interval whose start is the largest value
            # ≤ time within the sequence. It may still be a *past* interval
            # (already ended), so filter to intervals actually covering time.
            # The null check handles time points with no matching interval
            # (e.g. before any interval starts).
            .filter(pl.col(end_col).is_not_null() & (pl.col("time") < pl.col(end_col)))
            .select([id_col, "time", state_col])
            .sort([id_col, "time"])
        )

        return StateSequence(data=result, config=self._config, alphabet=self._alphabet)

    def span(self) -> pl.DataFrame:
        """
        Get the temporal span (first start to last end) for each sequence.

        Returns DataFrame with id, first_start, last_end, and span columns.
        """
        id_col = self._config.id_column
        start_col = self._config.start_column
        end_col = self._config.end_column

        return (
            self._data.group_by(id_col)
            .agg(
                [
                    pl.col(start_col).min().alias("first_start"),
                    pl.col(end_col).max().alias("last_end"),
                ]
            )
            .with_columns((pl.col("last_end") - pl.col("first_start")).alias("span"))
            .sort(id_col)
        )
