"""Normative sequence indicators.

Social science indicators that require labeling states as positive or
negative. Analogous to TraMineR's seqivolatility, seqprecarity,
seqinsecurity, seqidegrad, seqibad, seqintegr, seqipos.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from yasqat.core.pool import SequencePool

if TYPE_CHECKING:
    from yasqat.core.protocols import SequenceData


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


def individual_state_distribution(sequence: SequenceData) -> pl.DataFrame:
    """
    Per-sequence time spent in each state (TraMineR ``seqistatd``).

    For every sequence and every state of the alphabet, reports the number of
    time units spent in that state and its proportion of the sequence length.
    Unvisited states are included with a count of zero so the result is
    rectangular. Many normative indicators (e.g. :func:`proportion_positive`,
    :func:`badness`) are aggregates of this distribution.

    Args:
        sequence: StateSequence or SequencePool.

    Returns:
        Long-format DataFrame with columns ``[id_column, state, count,
        proportion]`` — one row per (sequence, alphabet state).
    """
    pool = SequencePool.coerce(sequence)
    config = pool.config
    alphabet_states = sorted(pool.alphabet.states)

    rows = []
    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        n = len(states)
        counts = Counter(states)
        for state in alphabet_states:
            count = counts.get(state, 0)
            rows.append(
                {
                    config.id_column: seq_id,
                    "state": state,
                    "count": count,
                    "proportion": (count / n) if n else 0.0,
                }
            )

    return pl.DataFrame(rows)


def proportion_positive(
    sequence: SequenceData,
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
    pool = SequencePool.coerce(sequence)
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
    sequence: SequenceData,
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
    pool = SequencePool.coerce(sequence)
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


def objective_volatility(
    sequence: SequenceData,
    w: float = 0.5,
    per_sequence: bool = False,
) -> float | pl.DataFrame:
    """
    Objective sequence volatility (TraMineR ``seqivolatility``).

    A label-free measure of how much a sequence moves around, blending two
    normalized components with weight ``w``:

    - ``pvisited`` — proportion of the alphabet visited, adjusted as
      ``(distinct states visited - 1) / (alphabet size - 1)``.
    - ``ptrans`` — proportion of possible transitions taken,
      ``(number of state changes) / (length - 1)``.

    The volatility is ``w * pvisited + (1 - w) * ptrans``. Unlike
    :func:`volatility`, this does not depend on positive/negative labels.

    Args:
        sequence: StateSequence or SequencePool.
        w: Weight on the state-coverage component, in ``[0, 1]``. Higher ``w``
            emphasizes how many distinct states appear; lower ``w`` emphasizes
            how often the state changes. Defaults to ``0.5``.
        per_sequence: If True, return for each sequence.

    Returns:
        If per_sequence=False: Mean objective volatility.
        If per_sequence=True: DataFrame with sequence IDs and values.

    Raises:
        ValueError: If ``w`` is outside ``[0, 1]``.
    """
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"w must be in [0, 1], got {w}")

    pool = SequencePool.coerce(sequence)
    config = pool.config
    n_alphabet = len(pool.alphabet.states)
    values = []
    seq_ids = []

    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        n = len(states)
        if n <= 1 or n_alphabet <= 1:
            # No transitions possible, or a degenerate alphabet.
            values.append(0.0)
        else:
            pvisited = (len(set(states)) - 1) / (n_alphabet - 1)
            transitions = sum(1 for i in range(n - 1) if states[i] != states[i + 1])
            ptrans = transitions / (n - 1)
            values.append(w * pvisited + (1.0 - w) * ptrans)
        seq_ids.append(seq_id)

    if per_sequence:
        return pl.DataFrame({config.id_column: seq_ids, "objective_volatility": values})
    return float(np.mean(values))


def precarity(
    sequence: SequenceData,
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
    pool = SequencePool.coerce(sequence)
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
    sequence: SequenceData,
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
    pool = SequencePool.coerce(sequence)
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
    sequence: SequenceData,
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
    pool = SequencePool.coerce(sequence)
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
    sequence: SequenceData,
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
    pool = SequencePool.coerce(sequence)
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
    sequence: SequenceData,
    positive_states: set[str] | None = None,
    per_sequence: bool = False,
) -> float | pl.DataFrame:
    """
    Sequence integration: cumulative proportion of time in positive states.

    Measures how quickly and permanently the sequence enters positive states.
    Higher values indicate earlier and more sustained time in positive states.

    Args:
        sequence: StateSequence or SequencePool.
        positive_states: Set of positive state names. If None, computes
            integration for each state independently and returns a DataFrame
            with columns [state, integration].
        per_sequence: If True, return for each sequence.

    Returns:
        If positive_states is None: DataFrame with per-state integration values.
        If per_sequence=False: Mean integration.
        If per_sequence=True: DataFrame with sequence IDs and values.
    """
    pool = SequencePool.coerce(sequence)

    if positive_states is None:
        all_states = sorted(pool.alphabet.states)
        rows = []
        for state in all_states:
            val = integration(sequence, positive_states={state}, per_sequence=False)
            rows.append({"state": state, "integration": val})
        return pl.DataFrame(rows)
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
