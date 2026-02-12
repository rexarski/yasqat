"""Transition rate matrix and related statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence


def transition_rate_matrix(
    sequence: StateSequence | SequencePool,
    as_counts: bool = False,
) -> np.ndarray:
    """
    Compute state-to-state transition rate matrix.

    The transition rate matrix shows the probability (or count) of
    transitioning from one state to another.

    Entry [i, j] represents P(next_state = j | current_state = i).

    Args:
        sequence: StateSequence or SequencePool.
        as_counts: If True, return raw counts instead of probabilities.

    Returns:
        Square numpy array of transition rates/counts.
        Rows and columns are ordered by the alphabet.

    Example:
        >>> trate = transition_rate_matrix(pool)
        >>> # trate[i, j] = P(transition from state i to state j)
    """
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        pool = SequencePool(
            data=sequence.data,
            config=sequence.config,
            alphabet=sequence.alphabet,
        )
    else:
        pool = sequence

    alphabet = pool.alphabet
    n_states = len(alphabet)
    state_to_idx = {s: i for i, s in enumerate(alphabet.states)}

    counts = np.zeros((n_states, n_states), dtype=np.float64)

    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        for t in range(len(states) - 1):
            i = state_to_idx[states[t]]
            j = state_to_idx[states[t + 1]]
            counts[i, j] += 1

    if as_counts:
        return np.asarray(counts)

    # Normalize rows to get probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    rates: np.ndarray = counts / row_sums

    return np.asarray(rates)


def transition_rate_dataframe(
    sequence: StateSequence | SequencePool,
) -> pl.DataFrame:
    """
    Get transition rates as a DataFrame.

    Args:
        sequence: StateSequence or SequencePool.

    Returns:
        DataFrame with columns: from_state, to_state, count, rate.
    """
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        pool = SequencePool(
            data=sequence.data,
            config=sequence.config,
            alphabet=sequence.alphabet,
        )
    else:
        pool = sequence

    alphabet = pool.alphabet
    counts = transition_rate_matrix(pool, as_counts=True)
    rates = transition_rate_matrix(pool, as_counts=False)

    rows = []
    for i, from_state in enumerate(alphabet.states):
        for j, to_state in enumerate(alphabet.states):
            rows.append(
                {
                    "from_state": from_state,
                    "to_state": to_state,
                    "count": int(counts[i, j]),
                    "rate": float(rates[i, j]),
                }
            )

    return pl.DataFrame(rows)


def first_occurrence_time(
    sequence: StateSequence | SequencePool,
    state: str,
) -> pl.DataFrame:
    """
    Get the first occurrence time of a specific state for each sequence.

    Args:
        sequence: StateSequence or SequencePool.
        state: The state to find.

    Returns:
        DataFrame with sequence IDs and first occurrence times.
        Sequences that never reach the state have null values.
    """
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        data = sequence.data
        config = sequence.config
    else:
        data = sequence.data
        config = sequence.config

    return (
        data.filter(pl.col(config.state_column) == state)
        .group_by(config.id_column)
        .agg(pl.col(config.time_column).min().alias("first_occurrence"))
        .sort(config.id_column)
    )


def state_duration_stats(
    sequence: StateSequence | SequencePool,
) -> pl.DataFrame:
    """
    Calculate duration statistics for each state.

    Uses the spell (run-length encoded) representation to compute
    statistics about how long sequences stay in each state.

    Args:
        sequence: StateSequence or SequencePool.

    Returns:
        DataFrame with columns: state, mean_duration, median_duration,
        min_duration, max_duration, std_duration, n_spells.
    """
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        state_seq = sequence
    else:
        state_seq = sequence.to_state_sequence()

    sps = state_seq.to_sps()
    config = state_seq.config

    return (
        sps.group_by(config.state_column)
        .agg(
            [
                pl.col("duration").mean().alias("mean_duration"),
                pl.col("duration").median().alias("median_duration"),
                pl.col("duration").min().alias("min_duration"),
                pl.col("duration").max().alias("max_duration"),
                pl.col("duration").std().alias("std_duration"),
                pl.len().alias("n_spells"),
            ]
        )
        .sort(config.state_column)
    )


def substitution_cost_matrix(
    sequence: StateSequence | SequencePool,
    method: str = "trate",
) -> np.ndarray:
    """
    Generate substitution cost matrix for distance metrics.

    Args:
        sequence: StateSequence or SequencePool.
        method: Method for computing costs.
            - "trate": Based on transition rates (c = 2 - p_ij - p_ji).
            - "constant": Constant cost of 2.0.

    Returns:
        Square numpy array of substitution costs.
    """
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        pool = SequencePool(
            data=sequence.data,
            config=sequence.config,
            alphabet=sequence.alphabet,
        )
    else:
        pool = sequence

    n_states = len(pool.alphabet)

    if method == "constant":
        result = np.full((n_states, n_states), 2.0)
        np.fill_diagonal(result, 0.0)
        return result

    if method == "trate":
        rates = transition_rate_matrix(pool, as_counts=False)
        # Cost inversely proportional to transition probability
        result = 2.0 - rates - rates.T
        np.fill_diagonal(result, 0.0)
        return np.asarray(result)

    raise ValueError(f"Unknown method: {method}")
