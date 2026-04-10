"""
yasqat - Yet Another Sequence Analytics Toolkit

A modern Python library for sequence analysis with polars and plotnine.
"""

from importlib.metadata import PackageNotFoundError, version

from yasqat.core.alphabet import Alphabet
from yasqat.core.pool import SequencePool
from yasqat.core.sequence import (
    EventSequence,
    IntervalSequence,
    SequenceConfig,
    StateSequence,
)
try:
    __version__ = version("yasqat")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "Alphabet",
    "EventSequence",
    "IntervalSequence",
    "SequenceConfig",
    "SequencePool",
    "StateSequence",
    "__version__",
]
