"""Frequent subsequence mining.

Discovers frequent subsequences (patterns) in a collection of sequences
based on minimum support threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence


@dataclass
class FrequentSubsequence:
    """A frequent subsequence with its support."""

    pattern: tuple[str, ...]
    """The subsequence pattern."""

    support: int
    """Number of sequences containing this pattern."""

    proportion: float
    """Proportion of sequences containing this pattern."""


def _is_subsequence(pattern: tuple[str, ...], sequence: list[str]) -> bool:
    """Check if pattern is a subsequence of sequence."""
    p_idx = 0
    for s in sequence:
        if p_idx < len(pattern) and s == pattern[p_idx]:
            p_idx += 1
    return p_idx == len(pattern)


def frequent_subsequences(
    sequence: StateSequence | SequencePool,
    min_support: float = 0.1,
    max_length: int = 5,
) -> list[FrequentSubsequence]:
    """
    Find frequent subsequences using a level-wise approach.

    Discovers all subsequences that appear in at least min_support proportion
    of sequences. Uses an Apriori-like approach: if a subsequence is not
    frequent, no extension of it can be frequent.

    Args:
        sequence: StateSequence or SequencePool.
        min_support: Minimum support proportion (0 to 1).
        max_length: Maximum subsequence length to search.

    Returns:
        List of FrequentSubsequence objects sorted by support (descending).

    Example:
        >>> from yasqat.statistics.subsequence_mining import frequent_subsequences
        >>> results = frequent_subsequences(pool, min_support=0.5)
        >>> results[0].pattern
        ('A',)
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

    n_sequences = len(pool)
    min_count = max(1, int(min_support * n_sequences))

    # Get all sequences as lists
    all_sequences = [pool.get_sequence(sid) for sid in pool.sequence_ids]
    alphabet = list(pool.alphabet.states)

    frequent: list[FrequentSubsequence] = []

    # Level 1: single states
    current_level: list[tuple[str, ...]] = [(s,) for s in alphabet]

    for level in range(1, max_length + 1):
        level_frequent: list[tuple[str, ...]] = []

        for pattern in current_level:
            support = sum(1 for seq in all_sequences if _is_subsequence(pattern, seq))
            if support >= min_count:
                frequent.append(
                    FrequentSubsequence(
                        pattern=pattern,
                        support=support,
                        proportion=support / n_sequences,
                    )
                )
                level_frequent.append(pattern)

        if not level_frequent or level >= max_length:
            break

        # Generate next level candidates by extending frequent patterns
        next_level: list[tuple[str, ...]] = []
        for pattern in level_frequent:
            for s in alphabet:
                candidate = (*pattern, s)
                next_level.append(candidate)

        current_level = next_level

    # Sort by support descending
    frequent.sort(key=lambda x: (-x.support, x.pattern))
    return frequent
