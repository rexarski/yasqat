"""Structural typing for sequence containers.

``StateSequence`` and ``SequencePool`` both wrap a long-format ``pl.DataFrame``
over a shared ``Alphabet`` and ``SequenceConfig``. The ``SequenceData`` protocol
names that shared read surface so callers (notably ``statistics.*``) can accept
either type without an explicit ``StateSequence | SequencePool`` union.

The protocol intentionally declares only
``data``/``config``/``alphabet``/``sequence_ids`` — the attributes both types
expose with identical semantics. It omits ``get_sequence``, whose return type
differs between the two classes, so duck-typing through this protocol can
never hit that mismatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import polars as pl

    from yasqat.core.alphabet import Alphabet
    from yasqat.core.sequence import SequenceConfig


@runtime_checkable
class SequenceData(Protocol):
    """Shared read surface of ``StateSequence`` and ``SequencePool``."""

    @property
    def data(self) -> pl.DataFrame:
        """The underlying long-format DataFrame."""
        ...

    @property
    def config(self) -> SequenceConfig:
        """The column configuration."""
        ...

    @property
    def alphabet(self) -> Alphabet:
        """The state alphabet."""
        ...

    @property
    def sequence_ids(self) -> list[int | str]:
        """Sorted unique sequence identifiers."""
        ...
