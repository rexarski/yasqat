"""Parallel coordinate plot for sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from plotnine import (
    aes,
    element_blank,
    geom_line,
    ggplot,
    labs,
    theme,
    theme_minimal,
)

if TYPE_CHECKING:
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence


def parallel_coordinate_plot(
    sequence: StateSequence | SequencePool,
    max_sequences: int | None = 100,
    alpha: float = 0.3,
    title: str | None = None,
    x_label: str = "Time",
    y_label: str = "State",
    figsize: tuple[float, float] = (12, 6),
    seed: int | None = None,
) -> ggplot:
    """
    Create a parallel coordinate plot of sequences.

    Each sequence is drawn as a line connecting its states across time
    positions. This gives an overview of individual trajectories and
    common patterns.

    Args:
        sequence: StateSequence or SequencePool to visualize.
        max_sequences: Maximum number of sequences to plot (for readability).
            If None, plot all. Default: 100.
        alpha: Line transparency (0-1). Lower = more transparent.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        figsize: Figure size as (width, height).
        seed: Random seed for subsampling.

    Returns:
        A plotnine ggplot object.
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

    # Subsample if needed
    if max_sequences is not None and len(pool) > max_sequences:
        pool = pool.sample(max_sequences, seed=seed)

    config = pool.config
    alphabet = pool.alphabet
    state_to_num = {s: i for i, s in enumerate(alphabet.states)}

    # Build plotting data: id, time, state_num
    data = pool.data.with_columns(
        pl.col(config.state_column)
        .replace(state_to_num)
        .cast(pl.Float64)
        .alias("state_num")
    )

    pdf = data.select([config.id_column, config.time_column, "state_num"]).to_pandas()

    p = (
        ggplot(
            pdf,
            aes(
                x=config.time_column,
                y="state_num",
                group=config.id_column,
            ),
        )
        + geom_line(alpha=alpha, color="#377EB8", size=0.5)
        + theme_minimal()
        + theme(
            figure_size=figsize,
            panel_grid_minor=element_blank(),
        )
        + labs(x=x_label, y=y_label)
    )

    if title:
        p = p + labs(title=title)

    return p
