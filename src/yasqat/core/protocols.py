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

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    import polars as pl

    from yasqat.core.alphabet import Alphabet
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import SequenceConfig, StateSequence


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


_Container = TypeVar("_Container", "StateSequence", "SequencePool")


def coerce_container(cls: type[_Container], sequence: SequenceData) -> _Container:
    """Normalize a sequence container to ``cls`` — the one rebuild rule.

    Returns ``sequence`` unchanged if it is already a ``cls``; otherwise
    rebuilds one from the ``SequenceData`` surface. Backs both
    :meth:`StateSequence.coerce` and :meth:`SequencePool.coerce` so the
    protocol's field set is spelled out exactly once.
    """
    if isinstance(sequence, cls):
        return sequence
    return cls(
        data=sequence.data,
        config=sequence.config,
        alphabet=sequence.alphabet,
    )
