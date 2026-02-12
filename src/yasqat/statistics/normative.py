"""Normative sequence indicators.

Social science indicators that require labeling states as positive or
negative. Analogous to TraMineR's seqivolatility, seqprecarity,
seqinsecurity, seqidegrad, seqibad, seqintegr, seqipos.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence


def _get_pool(sequence: StateSequence | SequencePool) -> SequencePool:
    """Convert StateSequence to SequencePool if needed."""
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence

    if isinstance(sequence, StateSequence):
        return SequencePool(
            data=sequence.data,
            config=sequence.config,
            alphabet=sequence.alphabet,
        )
    return sequence


def _state_signs(
    states: list[str],
    positive_states: set[str],
    negative_states: set[str],
) -> list[int]:
    """Map states to +1 (positive), -1 (negative), or 0 (neutral)."""
    signs = []
    for s in states:
        if s in positive_states:
            signs.append(1)
        elif s in negative_states:
            signs.append(-1)
        else:
            signs.append(0)
    return signs


def proportion_positive(
    sequence: StateSequence | SequencePool,
    positive_states: set[str],
    per_sequence: bool = False,
) -> float | pl.DataFrame:
    """
    Proportion of time spent in positive states.

    Args:
        sequence: StateSequence or SequencePool.
        positive_states: Set of state names considered positive.
        per_sequence: If True, return for each sequence.

    Returns:
        If per_sequence=False: Mean proportion positive.
        If per_sequence=True: DataFrame with sequence IDs and proportions.
    """
    pool = _get_pool(sequence)
    config = pool.config
    proportions = []
    seq_ids = []

    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        n = len(states)
        if n == 0:
            proportions.append(0.0)
        else:
            n_pos = sum(1 for s in states if s in positive_states)
            proportions.append(n_pos / n)
        seq_ids.append(seq_id)

    if per_sequence:
        return pl.DataFrame(
            {config.id_column: seq_ids, "proportion_positive": proportions}
        )
    return float(np.mean(proportions))


def volatility(
    sequence: StateSequence | SequencePool,
    positive_states: set[str],
    negative_states: set[str],
    per_sequence: bool = False,
) -> float | pl.DataFrame:
    """
    Sequence volatility: frequency of sign changes (positive/negative transitions).

    Measures how often the sequence switches between positive and negative
    states, normalized by (length - 1).

    Args:
        sequence: StateSequence or SequencePool.
        positive_states: Set of positive state names.
        negative_states: Set of negative state names.
        per_sequence: If True, return for each sequence.

    Returns:
        If per_sequence=False: Mean volatility.
        If per_sequence=True: DataFrame with sequence IDs and values.
    """
    pool = _get_pool(sequence)
    config = pool.config
    values = []
    seq_ids = []

    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        signs = _state_signs(states, positive_states, negative_states)
        n = len(signs)
        if n <= 1:
            values.append(0.0)
        else:
            changes = sum(
                1
                for i in range(n - 1)
                if signs[i] != 0 and signs[i + 1] != 0 and signs[i] != signs[i + 1]
            )
            values.append(changes / (n - 1))
        seq_ids.append(seq_id)

    if per_sequence:
        return pl.DataFrame({config.id_column: seq_ids, "volatility": values})
    return float(np.mean(values))


def precarity(
    sequence: StateSequence | SequencePool,
    negative_states: set[str],
    per_sequence: bool = False,
) -> float | pl.DataFrame:
    """
    Sequence precarity: proportion of time spent in negative states,
    weighted by their recency (later positions count more).

    Args:
        sequence: StateSequence or SequencePool.
        negative_states: Set of negative state names.
        per_sequence: If True, return for each sequence.

    Returns:
        If per_sequence=False: Mean precarity.
        If per_sequence=True: DataFrame with sequence IDs and values.
    """
    pool = _get_pool(sequence)
    config = pool.config
    values = []
    seq_ids = []

    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        n = len(states)
        if n == 0:
            values.append(0.0)
        else:
            # Linearly increasing weights: 1, 2, ..., n
            total_weight = n * (n + 1) / 2
            weighted_neg = sum(
                (i + 1) for i, s in enumerate(states) if s in negative_states
            )
            values.append(weighted_neg / total_weight)
        seq_ids.append(seq_id)

    if per_sequence:
        return pl.DataFrame({config.id_column: seq_ids, "precarity": values})
    return float(np.mean(values))


def insecurity(
    sequence: StateSequence | SequencePool,
    negative_states: set[str],
    per_sequence: bool = False,
) -> float | pl.DataFrame:
    """
    Sequence insecurity: expected proportion of remaining time in negative states.

    At each position, computes the proportion of future positions in negative
    states, then averages across positions.

    Args:
        sequence: StateSequence or SequencePool.
        negative_states: Set of negative state names.
        per_sequence: If True, return for each sequence.

    Returns:
        If per_sequence=False: Mean insecurity.
        If per_sequence=True: DataFrame with sequence IDs and values.
    """
    pool = _get_pool(sequence)
    config = pool.config
    values = []
    seq_ids = []

    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        n = len(states)
        if n <= 1:
            values.append(1.0 if n == 1 and states[0] in negative_states else 0.0)
        else:
            # Suffix sum of negative states
            neg_suffix = [0] * (n + 1)
            for i in range(n - 1, -1, -1):
                neg_suffix[i] = neg_suffix[i + 1] + (
                    1 if states[i] in negative_states else 0
                )

            # Average insecurity at each position
            total = 0.0
            for i in range(n):
                remaining = n - i
                total += neg_suffix[i] / remaining
            values.append(total / n)
        seq_ids.append(seq_id)

    if per_sequence:
        return pl.DataFrame({config.id_column: seq_ids, "insecurity": values})
    return float(np.mean(values))


def degradation(
    sequence: StateSequence | SequencePool,
    positive_states: set[str],
    negative_states: set[str],
    per_sequence: bool = False,
) -> float | pl.DataFrame:
    """
    Sequence degradation: tendency to move from positive to negative states.

    Measures the longest uninterrupted sequence of transitions from positive
    to negative states, normalized by sequence length.

    Args:
        sequence: StateSequence or SequencePool.
        positive_states: Set of positive state names.
        negative_states: Set of negative state names.
        per_sequence: If True, return for each sequence.

    Returns:
        If per_sequence=False: Mean degradation.
        If per_sequence=True: DataFrame with sequence IDs and values.
    """
    pool = _get_pool(sequence)
    config = pool.config
    values = []
    seq_ids = []

    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        signs = _state_signs(states, positive_states, negative_states)
        n = len(signs)

        if n <= 1:
            values.append(0.0)
        else:
            # Count transitions from positive to negative
            neg_transitions = sum(
                1 for i in range(n - 1) if signs[i] == 1 and signs[i + 1] == -1
            )
            values.append(neg_transitions / (n - 1))
        seq_ids.append(seq_id)

    if per_sequence:
        return pl.DataFrame({config.id_column: seq_ids, "degradation": values})
    return float(np.mean(values))


def badness(
    sequence: StateSequence | SequencePool,
    negative_states: set[str],
    per_sequence: bool = False,
) -> float | pl.DataFrame:
    """
    Sequence badness: proportion of time in negative states.

    Simple ratio of negative state occurrences to total length.

    Args:
        sequence: StateSequence or SequencePool.
        negative_states: Set of negative state names.
        per_sequence: If True, return for each sequence.

    Returns:
        If per_sequence=False: Mean badness.
        If per_sequence=True: DataFrame with sequence IDs and values.
    """
    pool = _get_pool(sequence)
    config = pool.config
    values = []
    seq_ids = []

    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        n = len(states)
        if n == 0:
            values.append(0.0)
        else:
            n_neg = sum(1 for s in states if s in negative_states)
            values.append(n_neg / n)
        seq_ids.append(seq_id)

    if per_sequence:
        return pl.DataFrame({config.id_column: seq_ids, "badness": values})
    return float(np.mean(values))


def integration(
    sequence: StateSequence | SequencePool,
    positive_states: set[str],
    per_sequence: bool = False,
) -> float | pl.DataFrame:
    """
    Sequence integration: cumulative proportion of time in positive states.

    Measures how quickly and permanently the sequence enters positive states.
    Higher values indicate earlier and more sustained time in positive states.

    Args:
        sequence: StateSequence or SequencePool.
        positive_states: Set of positive state names.
        per_sequence: If True, return for each sequence.

    Returns:
        If per_sequence=False: Mean integration.
        If per_sequence=True: DataFrame with sequence IDs and values.
    """
    pool = _get_pool(sequence)
    config = pool.config
    values = []
    seq_ids = []

    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        n = len(states)
        if n == 0:
            values.append(0.0)
        else:
            # Cumulative proportion positive at each step
            cum_pos = 0
            total = 0.0
            for i, s in enumerate(states):
                if s in positive_states:
                    cum_pos += 1
                total += cum_pos / (i + 1)
            values.append(total / n)
        seq_ids.append(seq_id)

    if per_sequence:
        return pl.DataFrame({config.id_column: seq_ids, "integration": values})
    return float(np.mean(values))
