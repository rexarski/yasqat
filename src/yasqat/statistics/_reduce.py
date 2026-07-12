"""Shared per-sequence reduce backing the loop-shaped statistics functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from yasqat.core.pool import SequencePool

if TYPE_CHECKING:
    from collections.abc import Callable

    from yasqat.core.protocols import SequenceData


def reduce_per_sequence(
    sequence: SequenceData,
    fn: Callable[[list[str]], float],
    name: str,
    per_sequence: bool = False,
    aggregate: Literal["mean", "sum"] = "mean",
) -> float | pl.DataFrame:
    """Map a per-sequence scalar over a pool, returning the house shape.

    Owns the contract shared by the per-sequence statistics: coerce to the
    canonical container, evaluate ``fn`` on each sequence's state list in
    ``sequence_ids`` order, and either return the per-sequence DataFrame
    (``[id_column, name]``) or collapse to one aggregate scalar.

    ``fn`` receives one sequence's states and returns its scalar; edge cases
    (empty sequence, single state) are ``fn``'s own responsibility — the
    reduce makes no assumptions about them.

    Args:
        sequence: StateSequence or SequencePool.
        fn: Scalar function of one sequence's state list.
        name: Column name for the per-sequence value.
        per_sequence: If True, return the per-sequence DataFrame.
        aggregate: Collapse used when ``per_sequence=False`` — ``"mean"``
            (default) or ``"sum"``.

    Returns:
        If per_sequence=False: the aggregated scalar.
        If per_sequence=True: DataFrame with columns ``[id_column, name]``.
    """
    pool = SequencePool.coerce(sequence)
    seq_ids = pool.sequence_ids
    values = [fn(pool.get_sequence(seq_id)) for seq_id in seq_ids]

    if per_sequence:
        return pl.DataFrame({pool.config.id_column: seq_ids, name: values})
    if aggregate == "sum":
        return sum(values)
    return float(np.mean(values))
