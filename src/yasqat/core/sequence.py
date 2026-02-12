"""Sequence classes for state and event sequences."""

from __future__ import annotations

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
        """
        self._config = config or SequenceConfig()
        self._data = self._validate_and_prepare(data)
        self._alphabet = alphabet or self._infer_alphabet()

    @abstractmethod
    def _validate_and_prepare(self, data: pl.DataFrame) -> pl.DataFrame:
        """Validate and prepare the input data."""

    @abstractmethod
    def _infer_alphabet(self) -> Alphabet:
        """Infer the alphabet from the data."""

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

        # Sort by id and time
        return data.sort([self._config.id_column, self._config.time_column])

    def _infer_alphabet(self) -> Alphabet:
        """Infer alphabet from state column."""
        return Alphabet.from_series(self._data[self._config.state_column])

    def n_sequences(self) -> int:
        """Return the number of unique sequences."""
        return self._data[self._config.id_column].n_unique()

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
        - duration: Spell duration
        """
        id_col = self._config.id_column
        time_col = self._config.time_column
        state_col = self._config.state_column

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
            .agg(
                [
                    pl.col(state_col).first().alias(state_col),
                    pl.col(time_col).min().alias("start"),
                    pl.col(time_col).max().alias("end"),
                    pl.len().alias("duration"),
                ]
            )
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

    def to_event_sequence(self) -> EventSequence:
        """
        Convert to EventSequence (treat each state observation as an event).

        Returns:
            EventSequence with the same data.
        """
        return EventSequence(
            data=self._data.select(
                [
                    self._config.id_column,
                    self._config.time_column,
                    self._config.state_column,
                ]
            ),
            config=self._config,
            alphabet=self._alphabet,
        )

    def to_interval_sequence(self) -> IntervalSequence:
        """
        Convert to IntervalSequence using spell (run-length) encoding.

        Each consecutive run of the same state becomes an interval
        with start and end times.

        Returns:
            IntervalSequence derived from spells.
        """
        sps = self.to_sps()

        interval_data = sps.select(
            [
                pl.col(self._config.id_column),
                pl.col("start").alias(self._config.start_column),
                pl.col("end").alias(self._config.end_column),
                pl.col(self._config.state_column),
            ]
        )

        return IntervalSequence(
            data=interval_data,
            config=self._config,
            alphabet=self._alphabet,
        )


@dataclass
class EventSequence(BaseSequence):
    """
    Event sequence representation.

    An event sequence represents a series of point-in-time events,
    where each event has a timestamp and an event type.

    The data should be in long format:
    - id_column: Identifier for each sequence
    - time_column: Timestamp of the event
    - state_column: Event type
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
        """Initialize an event sequence."""
        super().__init__(data, config, alphabet)

    def _validate_and_prepare(self, data: pl.DataFrame) -> pl.DataFrame:
        """Validate and prepare event sequence data."""
        required_cols = [
            self._config.id_column,
            self._config.time_column,
            self._config.state_column,
        ]

        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Sort by id and time
        return data.sort([self._config.id_column, self._config.time_column])

    def _infer_alphabet(self) -> Alphabet:
        """Infer alphabet from event type column."""
        return Alphabet.from_series(self._data[self._config.state_column])

    def n_sequences(self) -> int:
        """Return the number of unique sequences."""
        return self._data[self._config.id_column].n_unique()

    def sequence_ids(self) -> list[int | str]:
        """Return the unique sequence IDs."""
        return self._data[self._config.id_column].unique().sort().to_list()

    def get_sequence(self, seq_id: int | str) -> pl.DataFrame:
        """Get events for a specific sequence ID."""
        return self._data.filter(pl.col(self._config.id_column) == seq_id)

    def event_counts(self) -> pl.DataFrame:
        """Count events by type."""
        return (
            self._data.group_by(self._config.state_column)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )

    def events_per_sequence(self) -> pl.DataFrame:
        """Count events per sequence."""
        return (
            self._data.group_by(self._config.id_column)
            .agg(pl.len().alias("n_events"))
            .sort(self._config.id_column)
        )

    def to_state_sequence(self) -> StateSequence:
        """
        Convert to StateSequence (treat each event as a state observation).

        Returns:
            StateSequence with the same data.
        """
        return StateSequence(
            data=self._data.select(
                [
                    self._config.id_column,
                    self._config.time_column,
                    self._config.state_column,
                ]
            ),
            config=self._config,
            alphabet=self._alphabet,
        )


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

        # Sort by id and start time
        return data.sort([self._config.id_column, self._config.start_column])

    def _infer_alphabet(self) -> Alphabet:
        """Infer alphabet from state column."""
        return Alphabet.from_series(self._data[self._config.state_column])

    def n_sequences(self) -> int:
        """Return the number of unique sequences."""
        return self._data[self._config.id_column].n_unique()

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
        for seq_id in self.sequence_ids():
            if len(self.overlapping_intervals(seq_id)) > 0:
                return True
        return False

    def to_state_sequence(self, time_points: list[int] | None = None) -> pl.DataFrame:
        """
        Convert interval sequence to state sequence by sampling at time points.

        If multiple intervals overlap at a time point, the first one (by start time)
        is used.

        Args:
            time_points: List of time points to sample. If None, uses integer
                         range from min start to max end.

        Returns:
            DataFrame in state sequence format (id, time, state).
        """
        id_col = self._config.id_column
        start_col = self._config.start_column
        end_col = self._config.end_column
        state_col = self._config.state_column

        if time_points is None:
            min_start_val = self._data[start_col].min()
            max_end_val = self._data[end_col].max()
            if min_start_val is None or max_end_val is None:
                return pl.DataFrame(
                    schema={id_col: pl.Int64, "time": pl.Int64, state_col: pl.Utf8}
                )
            # Cast to int for range()
            min_start_int = int(float(min_start_val))  # type: ignore[arg-type]
            max_end_int = int(float(max_end_val))  # type: ignore[arg-type]
            time_points = list(range(min_start_int, max_end_int + 1))

        # Create time points DataFrame
        time_df = pl.DataFrame({"time": time_points})

        # For each sequence, find the state at each time point
        results = []
        for seq_id in self.sequence_ids():
            seq_data = self.get_sequence(seq_id)

            # Cross join with time points and filter
            seq_states = (
                time_df.join(seq_data, how="cross")
                .filter(
                    (pl.col("time") >= pl.col(start_col))
                    & (pl.col("time") < pl.col(end_col))
                )
                .group_by("time")
                .agg(pl.col(state_col).first())  # Take first if overlap
                .with_columns(pl.lit(seq_id).alias(id_col))
                .select([id_col, "time", state_col])
            )
            results.append(seq_states)

        if not results:
            return pl.DataFrame(
                schema={id_col: pl.Int64, "time": pl.Int64, state_col: pl.Utf8}
            )

        return pl.concat(results).sort([id_col, "time"])

    def to_event_sequence(self) -> EventSequence:
        """
        Convert to EventSequence (one event per interval start time).

        Returns:
            EventSequence with one event per interval.
        """
        event_data = self._data.select(
            [
                pl.col(self._config.id_column),
                pl.col(self._config.start_column).alias(self._config.time_column),
                pl.col(self._config.state_column),
            ]
        ).sort([self._config.id_column, self._config.time_column])

        return EventSequence(
            data=event_data,
            config=self._config,
            alphabet=self._alphabet,
        )

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
