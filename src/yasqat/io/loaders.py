"""Data loaders for sequence data using polars backend."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from yasqat.core.sequence import SequenceConfig

if TYPE_CHECKING:
    from yasqat.core.alphabet import Alphabet
    from yasqat.core.pool import SequencePool
    from yasqat.core.protocols import SequenceData


def load_csv(
    path: str | Path,
    config: SequenceConfig | None = None,
    **kwargs: Any,
) -> SequencePool:
    """Load state-shaped sequence data from a CSV file.

    For interval-shaped input, read with polars then call
    ``StateSequence.from_intervals(df, time_points=...)``.

    Args:
        path: Path to the CSV file.
        config: Column configuration. If None, uses defaults
            (id, time, state).
        **kwargs: Additional arguments passed to polars.read_csv().
            Common options include:
            - separator: Column delimiter (default: ",")
            - has_header: Whether file has header row (default: True)
            - null_values: Values to interpret as null
            - dtypes: Column data types

    Returns:
        A SequencePool ready for analysis.

    Example:
        >>> from yasqat.io import load_csv
        >>> pool = load_csv("data.csv")
        >>> len(pool)
        100
    """
    return load_dataframe(pl.read_csv(path, **kwargs), config)


def load_json(
    path: str | Path,
    config: SequenceConfig | None = None,
) -> SequencePool:
    """Load state-shaped sequence data from a JSON file.

    Expects JSON in one of these formats:
    1. Array of records: [{"id": 1, "time": 0, "state": "A"}, ...]
    2. Column-oriented: {"id": [1, 1, 2], "time": [0, 1, 0], "state": ["A", "B", "A"]}

    Args:
        path: Path to the JSON file.
        config: Column configuration.

    Returns:
        A SequencePool ready for analysis.

    Example:
        >>> from yasqat.io import load_json
        >>> pool = load_json("data.json")
    """
    return load_dataframe(pl.read_json(path), config)


def load_parquet(
    path: str | Path,
    config: SequenceConfig | None = None,
    **kwargs: Any,
) -> SequencePool:
    """Load state-shaped sequence data from a Parquet file.

    Parquet is recommended for large datasets due to:
    - Columnar storage (efficient for sequence analysis)
    - Compression
    - Type preservation

    Args:
        path: Path to the Parquet file.
        config: Column configuration.
        **kwargs: Additional arguments passed to polars.read_parquet().

    Returns:
        A SequencePool ready for analysis.

    Example:
        >>> from yasqat.io import load_parquet
        >>> pool = load_parquet("data.parquet")
    """
    return load_dataframe(pl.read_parquet(path, **kwargs), config)


def save_csv(
    sequence: SequenceData,
    path: str | Path,
    **kwargs: Any,
) -> None:
    """Save sequence data to a CSV file.

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
    sequence: SequenceData,
    path: str | Path,
) -> None:
    """Save sequence data to a JSON file.

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
    sequence: SequenceData,
    path: str | Path,
    compression: CompressionCodec = "zstd",
    **kwargs: Any,
) -> None:
    """Save sequence data to a Parquet file.

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
    """Build a SequencePool directly from a polars DataFrame.

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
