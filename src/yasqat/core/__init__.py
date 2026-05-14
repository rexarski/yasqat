"""Core data structures for sequence analysis."""

from yasqat.core.alphabet import Alphabet
from yasqat.core.pool import SequencePool
from yasqat.core.sequence import (
    SequenceConfig,
    StateSequence,
)

__all__ = [
    "Alphabet",
    "SequenceConfig",
    "SequencePool",
    "StateSequence",
]
