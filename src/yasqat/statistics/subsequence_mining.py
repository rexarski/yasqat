"""Frequent subsequence mining.

Discovers frequent subsequences (patterns) in a collection of sequences
based on minimum support threshold, and derives sequential association rules
(confidence, lift, leverage, conviction) from those patterns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from yasqat.core.pool import SequencePool

if TYPE_CHECKING:
    from polars.datatypes import DataType, DataTypeClass

    from yasqat.core.protocols import SequenceData


@dataclass
class Rule:
    """A sequential association rule antecedent => consequent with its measures."""

    antecedent: tuple[str, ...]
    """The ordered prefix pattern that triggers the rule."""

    consequent: tuple[str, ...]
    """The ordered suffix pattern predicted to follow the antecedent."""

    support: float
    """Proportion of sequences containing the combined ordered pattern."""

    confidence: float
    """Proportion of antecedent-matching sequences that also match the rule."""

    lift: float
    """Confidence relative to the consequent's marginal support (1 = independent)."""

    leverage: float
    """Observed minus expected joint support under independence."""

    conviction: float
    """Directed dependence; ``inf`` when the rule is exact (confidence 1)."""


def _is_subsequence(pattern: tuple[str, ...], sequence: list[str]) -> bool:
    """Check if pattern is a subsequence of sequence."""
    p_idx = 0
    for s in sequence:
        if p_idx < len(pattern) and s == pattern[p_idx]:
            p_idx += 1
    return p_idx == len(pattern)


def _mine_frequent(
    all_sequences: list[list[str]],
    alphabet: list[str],
    min_count: int,
    max_length: int,
) -> dict[tuple[str, ...], int]:
    """Level-wise (Apriori-like) frequent subsequence mining.

    Returns every frequent subsequence (support >= ``min_count``) mapped to its
    support count. No ``min_length`` filtering happens here — callers apply that
    themselves — so the map is a complete support lookup: any subsequence of a
    frequent pattern (e.g. a rule's antecedent or consequent) is guaranteed to
    be present, by the Apriori anti-monotonicity property.
    """
    frequent: dict[tuple[str, ...], int] = {}
    current_level: list[tuple[str, ...]] = [(s,) for s in alphabet]

    for level in range(1, max_length + 1):
        level_frequent: list[tuple[str, ...]] = []
        for pattern in current_level:
            support = sum(1 for seq in all_sequences if _is_subsequence(pattern, seq))
            if support >= min_count:
                frequent[pattern] = support
                level_frequent.append(pattern)

        if not level_frequent or level >= max_length:
            break

        # Extend each frequent pattern by every alphabet symbol (right-growth).
        current_level = [(*pattern, s) for pattern in level_frequent for s in alphabet]

    return frequent


def frequent_subsequences(
    sequence: SequenceData,
    min_support: float = 0.1,
    max_length: int = 5,
    min_length: int = 1,
) -> pl.DataFrame:
    """
    Find frequent subsequences using a level-wise approach.

    Discovers all subsequences that appear in at least min_support proportion
    of sequences. Uses an Apriori-like approach: if a subsequence is not
    frequent, no extension of it can be frequent.

    Args:
        sequence: StateSequence or SequencePool.
        min_support: Minimum support proportion (0 to 1).
        max_length: Maximum subsequence length to search.
        min_length: Minimum subsequence length to include in results.

    Returns:
        DataFrame with columns:
            - subsequence: List of states forming the pattern.
            - support: Number of sequences containing this pattern.
            - proportion: Proportion of sequences containing this pattern.
        Sorted by support descending.

    Example:
        >>> from yasqat.statistics.subsequence_mining import frequent_subsequences
        >>> results = frequent_subsequences(pool, min_support=0.5)
        >>> results.filter(pl.col("subsequence").list.len() == 2)
    """
    pool = SequencePool.coerce(sequence)

    n_sequences = len(pool)
    min_count = max(1, int(min_support * n_sequences))

    all_sequences = [pool.get_sequence(sid) for sid in pool.sequence_ids]
    alphabet = list(pool.alphabet.states)

    frequent = _mine_frequent(all_sequences, alphabet, min_count, max_length)

    rows = [
        (pattern, support)
        for pattern, support in frequent.items()
        if len(pattern) >= min_length
    ]
    # Sort by support descending, then pattern for a stable order.
    rows.sort(key=lambda row: (-row[1], row[0]))

    return pl.DataFrame(
        {
            "subsequence": [list(pattern) for pattern, _ in rows],
            "support": [support for _, support in rows],
            "proportion": [support / n_sequences for _, support in rows],
        }
    )


def association_rules(
    sequence: SequenceData,
    min_support: float = 0.1,
    min_confidence: float = 0.0,
    max_length: int = 5,
) -> pl.DataFrame:
    """
    Derive sequential association rules from frequent subsequences.

    Each frequent subsequence of length >= 2 is split at every internal
    position into an ordered ``antecedent`` (prefix) and ``consequent``
    (suffix); the rule ``antecedent => consequent`` fires when a sequence
    contains the antecedent followed later by the consequent. Standard
    association-rule measures are reported for each rule.

    For a rule ``A => B`` whose combined ordered pattern is ``A`` followed by
    ``B`` (support ``s_ab``), with marginal supports ``s_a`` and ``s_b``:

    - ``confidence`` = ``s_ab / s_a`` — how often ``B`` follows given ``A``.
    - ``lift`` = ``confidence / s_b`` — > 1 means the pair co-occurs more than
      chance, < 1 less; 1 is independence.
    - ``leverage`` = ``s_ab - s_a * s_b`` — the same idea on an additive scale.
    - ``conviction`` = ``(1 - s_b) / (1 - confidence)`` — ``inf`` when the rule
      is exact (``confidence == 1``).

    Args:
        sequence: StateSequence or SequencePool.
        min_support: Minimum support proportion (0 to 1) for the combined
            pattern to be mined.
        min_confidence: Minimum rule confidence to include in results.
        max_length: Maximum combined-pattern length to search.

    Returns:
        DataFrame with columns ``antecedent`` (list of states), ``consequent``
        (list of states), ``support`` (proportion of the combined pattern),
        ``confidence``, ``lift``, ``leverage`` and ``conviction``. Sorted by
        confidence descending, then lift descending.

    Example:
        >>> from yasqat.statistics.subsequence_mining import association_rules
        >>> rules = association_rules(pool, min_support=0.5, min_confidence=0.6)
        >>> rules.filter(pl.col("lift") > 1.0)
    """
    pool = SequencePool.coerce(sequence)

    n_sequences = len(pool)
    min_count = max(1, int(min_support * n_sequences))

    all_sequences = [pool.get_sequence(sid) for sid in pool.sequence_ids]
    alphabet = list(pool.alphabet.states)

    frequent = _mine_frequent(all_sequences, alphabet, min_count, max_length)
    # Support as a proportion; every prefix/suffix of a frequent pattern is
    # itself frequent, so these lookups never miss (Apriori anti-monotonicity).
    proportion = {pattern: count / n_sequences for pattern, count in frequent.items()}

    rules: list[Rule] = []
    for pattern, count in frequent.items():
        if len(pattern) < 2:
            continue
        support_ab = count / n_sequences
        for split in range(1, len(pattern)):
            antecedent = pattern[:split]
            consequent = pattern[split:]
            support_a = proportion[antecedent]
            support_b = proportion[consequent]

            confidence = support_ab / support_a
            if confidence < min_confidence:
                continue

            lift = confidence / support_b
            leverage = support_ab - support_a * support_b
            conviction = (
                math.inf
                if confidence >= 1.0
                else (1.0 - support_b) / (1.0 - confidence)
            )
            rules.append(
                Rule(
                    antecedent=antecedent,
                    consequent=consequent,
                    support=support_ab,
                    confidence=confidence,
                    lift=lift,
                    leverage=leverage,
                    conviction=conviction,
                )
            )

    # Sort by confidence descending, then lift descending, then the patterns.
    rules.sort(key=lambda r: (-r.confidence, -r.lift, r.antecedent, r.consequent))

    schema: dict[str, DataType | DataTypeClass] = {
        "antecedent": pl.List(pl.Utf8),
        "consequent": pl.List(pl.Utf8),
        "support": pl.Float64,
        "confidence": pl.Float64,
        "lift": pl.Float64,
        "leverage": pl.Float64,
        "conviction": pl.Float64,
    }
    return pl.DataFrame(
        {
            "antecedent": [list(r.antecedent) for r in rules],
            "consequent": [list(r.consequent) for r in rules],
            "support": [r.support for r in rules],
            "confidence": [r.confidence for r in rules],
            "lift": [r.lift for r in rules],
            "leverage": [r.leverage for r in rules],
            "conviction": [r.conviction for r in rules],
        },
        schema=schema,
    )
