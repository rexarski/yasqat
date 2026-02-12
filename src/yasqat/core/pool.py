"""SequencePool class for managing collections of sequences."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from yasqat.core.alphabet import Alphabet
from yasqat.core.sequence import SequenceConfig, StateSequence

if TYPE_CHECKING:
    pass


@dataclass
class SequencePool:
    """
    A pool (collection) of sequences for batch operations.

    SequencePool provides efficient operations on multiple sequences,
    including pairwise distance computation and batch statistics.
    """

    _data: pl.DataFrame = field(default_factory=pl.DataFrame)
    _config: SequenceConfig = field(default_factory=SequenceConfig)
    _alphabet: Alphabet = field(default_factory=lambda: Alphabet(states=()))
    _sequences: dict[int | str, list[str]] = field(default_factory=dict)

    def __init__(
        self,
        data: pl.DataFrame,
        config: SequenceConfig | None = None,
        alphabet: Alphabet | None = None,
    ) -> None:
        """
        Initialize a sequence pool.

        Args:
            data: Polars DataFrame with sequence data in long format.
            config: Column configuration.
            alphabet: State alphabet (inferred if not provided).
        """
        self._config = config or SequenceConfig()
        self._data = self._validate_and_prepare(data)
        self._alphabet = alphabet or self._infer_alphabet()
        self._sequences = self._extract_sequences()

    def _validate_and_prepare(self, data: pl.DataFrame) -> pl.DataFrame:
        """Validate and prepare the data."""
        required_cols = [
            self._config.id_column,
            self._config.time_column,
            self._config.state_column,
        ]

        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        return data.sort([self._config.id_column, self._config.time_column])

    def _infer_alphabet(self) -> Alphabet:
        """Infer alphabet from data."""
        return Alphabet.from_series(self._data[self._config.state_column])

    def _extract_sequences(self) -> dict[int | str, list[str]]:
        """Extract sequences as lists of states."""
        sequences: dict[int | str, list[str]] = {}
        id_col = self._config.id_column
        state_col = self._config.state_column

        for seq_id in self.sequence_ids:
            states = self._data.filter(pl.col(id_col) == seq_id)[state_col].to_list()
            sequences[seq_id] = states

        return sequences

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

    @property
    def sequence_ids(self) -> list[int | str]:
        """Return sorted list of unique sequence IDs."""
        return self._data[self._config.id_column].unique().sort().to_list()

    def __len__(self) -> int:
        """Return the number of sequences in the pool."""
        return len(self._sequences)

    def __getitem__(self, seq_id: int | str) -> list[str]:
        """Get a sequence by ID."""
        return self._sequences[seq_id]

    def __iter__(self):  # type: ignore[no-untyped-def]
        """Iterate over sequence IDs."""
        return iter(self._sequences)

    def get_sequence(self, seq_id: int | str) -> list[str]:
        """Get states for a specific sequence."""
        return self._sequences[seq_id]

    def get_encoded_sequence(self, seq_id: int | str) -> np.ndarray:
        """Get an encoded sequence as numpy array."""
        return self._alphabet.encode(self._sequences[seq_id])

    def to_state_sequence(self) -> StateSequence:
        """Convert to a StateSequence object."""
        return StateSequence(
            data=self._data,
            config=self._config,
            alphabet=self._alphabet,
        )

    def sequence_lengths(self) -> pl.DataFrame:
        """Get the length of each sequence."""
        return (
            self._data.group_by(self._config.id_column)
            .agg(pl.len().alias("length"))
            .sort(self._config.id_column)
        )

    def compute_distances(
        self,
        method: str = "om",
        **kwargs: float,
    ) -> np.ndarray:
        """
        Compute pairwise distance matrix.

        Args:
            method: Distance method ("om", "hamming", "lcs").
            **kwargs: Method-specific parameters.

        Returns:
            Symmetric distance matrix as numpy array.
        """
        from yasqat.metrics import (
            chi2_distance,
            euclidean_distance,
            hamming_distance,
            lcp_distance,
            lcs_distance,
            nms_distance,
            nmsmst_distance,
            omloc_distance,
            omspell_distance,
            omstran_distance,
            optimal_matching,
            rlcp_distance,
            svrspell_distance,
            twed_distance,
        )
        from yasqat.metrics.dtw import dtw_distance

        methods: dict[str, Callable[..., float]] = {
            "om": optimal_matching,
            "hamming": hamming_distance,
            "lcs": lcs_distance,
            "lcp": lcp_distance,
            "rlcp": rlcp_distance,
            "euclidean": euclidean_distance,
            "chi2": chi2_distance,
            "dtw": dtw_distance,
            "twed": twed_distance,
            "omloc": omloc_distance,
            "omspell": omspell_distance,
            "omstran": omstran_distance,
            "nms": nms_distance,
            "nmsmst": nmsmst_distance,
            "svrspell": svrspell_distance,
        }

        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(methods)}")

        metric_fn = methods[method]
        n = len(self)
        ids = self.sequence_ids
        distances = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                seq_a = self.get_encoded_sequence(ids[i])
                seq_b = self.get_encoded_sequence(ids[j])
                dist = metric_fn(seq_a, seq_b, **kwargs)
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def to_wide_format(self) -> pl.DataFrame:
        """
        Convert to wide format (one row per sequence, columns for time points).

        Returns:
            DataFrame with sequence IDs as rows and time points as columns.
        """
        return self._data.pivot(
            on=self._config.time_column,
            index=self._config.id_column,
            values=self._config.state_column,
        )

    def to_long_format(self) -> pl.DataFrame:
        """Return long format (the default format)."""
        return self._data

    def filter_by_length(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
    ) -> SequencePool:
        """Filter sequences by length."""
        lengths = self.sequence_lengths()
        id_col = self._config.id_column

        if min_length is not None:
            lengths = lengths.filter(pl.col("length") >= min_length)
        if max_length is not None:
            lengths = lengths.filter(pl.col("length") <= max_length)

        valid_ids = lengths[id_col].to_list()
        filtered_data = self._data.filter(pl.col(id_col).is_in(valid_ids))

        return SequencePool(
            data=filtered_data,
            config=self._config,
            alphabet=self._alphabet,
        )

    def sample(self, n: int, seed: int | None = None) -> SequencePool:
        """Sample n sequences from the pool."""
        rng = np.random.default_rng(seed)
        ids = self.sequence_ids
        sampled_ids = rng.choice(ids, size=min(n, len(ids)), replace=False).tolist()

        sampled_data = self._data.filter(
            pl.col(self._config.id_column).is_in(sampled_ids)
        )

        return SequencePool(
            data=sampled_data,
            config=self._config,
            alphabet=self._alphabet,
        )

    def recode_states(self, mapping: dict[str, str]) -> SequencePool:
        """
        Recode (merge or rename) states using a mapping.

        Creates a new SequencePool with states transformed according to
        the mapping. States not in the mapping are kept as-is.
        The alphabet is rebuilt from the recoded data.

        Args:
            mapping: Dictionary mapping old state names to new state names.
                Multiple old states can map to the same new state (merging).

        Returns:
            New SequencePool with recoded states and updated alphabet.

        Example:
            >>> pool.recode_states({"A": "X", "B": "X"})  # Merge A and B into X
        """
        state_col = self._config.state_column

        recoded_data = self._data.with_columns(
            pl.col(state_col).replace(mapping).alias(state_col)
        )

        return SequencePool(
            data=recoded_data,
            config=self._config,
        )

    def describe(self) -> dict[str, int | float | list[str]]:
        """Get summary statistics about the pool."""
        lengths = self.sequence_lengths()["length"]

        return {
            "n_sequences": len(self),
            "n_states": len(self._alphabet),
            "states": list(self._alphabet.states),
            "total_observations": len(self._data),
            "min_length": int(lengths.min()),  # type: ignore[arg-type]
            "max_length": int(lengths.max()),  # type: ignore[arg-type]
            "mean_length": float(lengths.mean()),  # type: ignore[arg-type]
            "median_length": float(lengths.median()),  # type: ignore[arg-type]
        }
