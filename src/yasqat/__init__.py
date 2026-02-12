"""
yasqat - Yet Another Sequence Analytics Toolkit

A modern Python library for sequence analysis with polars and plotnine.
"""

from yasqat.core.alphabet import Alphabet
from yasqat.core.pool import SequencePool
from yasqat.core.sequence import EventSequence, IntervalSequence, StateSequence
from yasqat.core.trajectory import Trajectory, TrajectoryPool

__version__ = "0.1.0"

__all__ = [
    "Alphabet",
    "EventSequence",
    "IntervalSequence",
    "SequencePool",
    "StateSequence",
    "Trajectory",
    "TrajectoryPool",
    "__version__",
]
