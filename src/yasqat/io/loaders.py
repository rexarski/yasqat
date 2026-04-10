"""Data loaders for sequence data using polars backend."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from yasqat.core.sequence import (
    EventSequence,
    IntervalSequence,
    SequenceConfig,
    StateSequence,
)

if TYPE_CHECKING:
    from yasqat.core.alphabet import Alphabet
    from yasqat.core.pool import SequencePool


def load_csv(
    path: str | Path,
    sequence_type: str = "state",
    config: SequenceConfig | None = None,
    **kwargs: Any,
) -> StateSequence | EventSequence | IntervalSequence:
    """
    Load sequence data from a CSV file.

    Uses polars for fast CSV parsing.

    Args:
        path: Path to the CSV file.
        sequence_type: Type of sequence to create. One of:
            - "state": StateSequence (default)
            - "event": EventSequence
            - "interval": IntervalSequence
        config: Column configuration. If None, uses defaults
            (id, time, state for state/event; id, start, end, state for interval).
        **kwargs: Additional arguments passed to polars.read_csv().
            Common options include:
            - separator: Column delimiter (default: ",")
            - has_header: Whether file has header row (default: True)
            - null_values: Values to interpret as null
            - dtypes: Column data types

    Returns:
        A StateSequence, EventSequence, or IntervalSequence object.

    Example:
        >>> from yasqat.io import load_csv
        >>> seq = load_csv("data.csv", sequence_type="state")
        >>> seq.n_sequences()
        100
    """
    config = config or SequenceConfig()
    data = pl.read_csv(path, **kwargs)

    return _create_sequence(data, sequence_type, config)


def load_json(
    path: str | Path,
    sequence_type: str = "state",
    config: SequenceConfig | None = None,
) -> StateSequence | EventSequence | IntervalSequence:
    """
    Load sequence data from a JSON file.

    Expects JSON in one of these formats:
    1. Array of records: [{"id": 1, "time": 0, "state": "A"}, ...]
    2. Column-oriented: {"id": [1, 1, 2], "time": [0, 1, 0], "state": ["A", "B", "A"]}

    Args:
        path: Path to the JSON file.
        sequence_type: Type of sequence ("state", "event", "interval").
        config: Column configuration.

    Returns:
        A StateSequence, EventSequence, or IntervalSequence object.

    Example:
        >>> from yasqat.io import load_json
        >>> seq = load_json("data.json", sequence_type="event")
    """
    config = config or SequenceConfig()
    data = pl.read_json(path)

    return _create_sequence(data, sequence_type, config)


def load_parquet(
    path: str | Path,
    sequence_type: str = "state",
    config: SequenceConfig | None = None,
    **kwargs: Any,
) -> StateSequence | EventSequence | IntervalSequence:
    """
    Load sequence data from a Parquet file.

    Parquet is recommended for large datasets due to:
    - Columnar storage (efficient for sequence analysis)
    - Compression
    - Type preservation

    Args:
        path: Path to the Parquet file.
        sequence_type: Type of sequence ("state", "event", "interval").
        config: Column configuration.
        **kwargs: Additional arguments passed to polars.read_parquet().

    Returns:
        A StateSequence, EventSequence, or IntervalSequence object.

    Example:
        >>> from yasqat.io import load_parquet
        >>> seq = load_parquet("data.parquet")
    """
    config = config or SequenceConfig()
    data = pl.read_parquet(path, **kwargs)

    return _create_sequence(data, sequence_type, config)


def save_csv(
    sequence: StateSequence | EventSequence | IntervalSequence,
    path: str | Path,
    **kwargs: Any,
) -> None:
    """
    Save sequence data to a CSV file.

    Args:
        sequence: Sequence object to save.
        path: Output file path.
        **kwargs: Additional arguments passed to polars DataFrame.write_csv().
            Common options include:
            - separator: Column delimiter (default: ",")
            - include_header: Whether to write header row (default: True)
            - null_value: String to write for null values

    Example:
        >>> from yasqat.io import save_csv
        >>> save_csv(sequence, "output.csv")
    """
    sequence.data.write_csv(path, **kwargs)


def save_json(
    sequence: StateSequence | EventSequence | IntervalSequence,
    path: str | Path,
) -> None:
    """
    Save sequence data to a JSON file.

    Args:
        sequence: Sequence object to save.
        path: Output file path.

    Example:
        >>> from yasqat.io import save_json
        >>> save_json(sequence, "output.json")
    """
    sequence.data.write_json(path)


CompressionCodec = Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"]


def save_parquet(
    sequence: StateSequence | EventSequence | IntervalSequence,
    path: str | Path,
    compression: CompressionCodec = "zstd",
    **kwargs: Any,
) -> None:
    """
    Save sequence data to a Parquet file.

    Args:
        sequence: Sequence object to save.
        path: Output file path.
        compression: Compression codec ("zstd", "snappy", "gzip", "lz4", "brotli", "uncompressed").
        **kwargs: Additional arguments passed to polars DataFrame.write_parquet().

    Example:
        >>> from yasqat.io import save_parquet
        >>> save_parquet(sequence, "output.parquet")
    """
    sequence.data.write_parquet(path, compression=compression, **kwargs)


def load_dataframe(
    df: pl.DataFrame,
    config: SequenceConfig | None = None,
    alphabet: Alphabet | None = None,
    drop_nulls: bool = False,
) -> SequencePool:
    """
    Build a SequencePool directly from a polars DataFrame.

    This is the recommended entry point when loading data from Hive tables,
    Spark (via parquet export or Arrow bridge), or any other source that
    already produces a polars DataFrame.

    Args:
        df: Polars DataFrame in long format with id, time, and state columns.
        config: Column configuration. Defaults to id/time/state.
        alphabet: State alphabet. Inferred from data if not provided.
        drop_nulls: If True, drop rows where the state column is null.

    Returns:
        A SequencePool ready for analysis.

    Example:
        >>> df = pl.read_parquet("s3://bucket/events_export/")
        >>> pool = load_dataframe(df, config=SequenceConfig(
        ...     id_column="user_id", time_column="ts", state_column="event"
        ... ))
    """
    from yasqat.core.pool import SequencePool

    config = config or SequenceConfig()

    if drop_nulls:
        df = df.drop_nulls(subset=[config.state_column])

    # Validate alphabet covers all data states
    if alphabet is not None:
        data_states = set(df[config.state_column].unique().to_list())
        alphabet_states = set(alphabet.states)
        missing = data_states - alphabet_states
        if missing:
            raise ValueError(
                f"Data contains states not in the provided alphabet: {sorted(missing)}. "
                f"Alphabet states: {sorted(alphabet_states)}"
            )

    return SequencePool(data=df, config=config, alphabet=alphabet)


def _create_sequence(
    data: pl.DataFrame,
    sequence_type: str,
    config: SequenceConfig,
) -> StateSequence | EventSequence | IntervalSequence:
    """Create the appropriate sequence type from a DataFrame."""
    if sequence_type == "state":
        return StateSequence(data, config)
    elif sequence_type == "event":
        return EventSequence(data, config)
    elif sequence_type == "interval":
        return IntervalSequence(data, config)
    else:
        raise ValueError(
            f"Unknown sequence_type: {sequence_type}. "
            f"Expected one of: 'state', 'event', 'interval'"
        )


# Additional utility functions


def infer_sequence_type(
    data: pl.DataFrame,
    config: SequenceConfig | None = None,
) -> str:
    """
    Infer the sequence type from DataFrame columns.

    Args:
        data: Input DataFrame.
        config: Column configuration.

    Returns:
        Inferred sequence type: "state", "event", or "interval".
    """
    config = config or SequenceConfig()

    has_start = config.start_column in data.columns
    has_end = config.end_column in data.columns
    has_time = config.time_column in data.columns

    if has_start and has_end:
        return "interval"
    elif has_time:
        # Check if data looks like continuous states or discrete events
        # by examining the time column density
        n_rows = len(data)
        n_sequences = data[config.id_column].n_unique()

        if n_sequences > 0:
            avg_length = n_rows / n_sequences
            # If sequences are relatively long and dense, likely state sequence
            if avg_length >= 3:
                return "state"
            else:
                return "event"
        return "state"
    else:
        # Default to state sequence
        return "state"


