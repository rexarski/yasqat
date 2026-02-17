"""Data loaders for sequence data using polars backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import polars as pl

from yasqat.core.sequence import (
    EventSequence,
    IntervalSequence,
    SequenceConfig,
    StateSequence,
)


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


def load_wide_format(
    path: str | Path,
    id_column: str = "id",
    file_format: str = "csv",
    **kwargs: Any,
) -> pl.DataFrame:
    """
    Read wide-format sequence data and convert to long format.

    Wide format has one row per sequence with time points as columns:
        id, t0, t1, t2, t3, ...

    This function converts it to long format:
        id, time, state

    Args:
        path: Path to the data file.
        id_column: Name of the ID column.
        file_format: File format ("csv", "parquet", "json").
        **kwargs: Additional arguments for the reader.

    Returns:
        DataFrame in long format.

    Example:
        >>> from yasqat.io.loaders import load_wide_format
        >>> df = load_wide_format("wide_data.csv")
        >>> df.head()
        shape: (5, 3)
        ┌─────┬──────┬───────┐
        │ id  ┆ time ┆ state │
        └─────┴──────┴───────┘
    """
    # Read the file
    if file_format == "csv":
        data = pl.read_csv(path, **kwargs)
    elif file_format == "parquet":
        data = pl.read_parquet(path, **kwargs)
    elif file_format == "json":
        data = pl.read_json(path)
    else:
        raise ValueError(f"Unknown file format: {file_format}")

    # Identify time columns (all columns except id)
    time_columns = [col for col in data.columns if col != id_column]

    # Convert to long format
    long_data = data.unpivot(
        index=id_column,
        on=time_columns,
        variable_name="time",
        value_name="state",
    )

    # Try to convert time column to integer
    try:
        long_data = long_data.with_columns(
            pl.col("time").str.extract(r"(\d+)").cast(pl.Int64).alias("time")
        )
    except Exception:
        # If conversion fails, keep as string
        pass

    return long_data.sort([id_column, "time"])


def to_wide_format(
    sequence: StateSequence | EventSequence,
) -> pl.DataFrame:
    """
    Convert a sequence to wide format.

    Long format:
        id, time, state

    Wide format:
        id, t0, t1, t2, ...

    Args:
        sequence: Sequence to convert.

    Returns:
        DataFrame in wide format.

    Example:
        >>> from yasqat.io.loaders import to_wide_format
        >>> wide_df = to_wide_format(state_sequence)
    """
    config = sequence.config
    data = sequence.data

    # Pivot to wide format
    return data.pivot(
        values=config.state_column,
        index=config.id_column,
        on=config.time_column,
    ).sort(config.id_column)
