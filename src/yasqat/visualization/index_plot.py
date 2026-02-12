"""Index plot visualization for sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl
from plotnine import (
    aes,
    element_blank,
    geom_tile,
    ggplot,
    labs,
    scale_fill_manual,
    scale_y_reverse,
    theme,
    theme_minimal,
)

if TYPE_CHECKING:
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence


def index_plot(
    sequence: StateSequence | SequencePool,
    sort_by: Literal["from.start", "from.end", "length", None] = None,
    title: str | None = None,
    x_label: str = "Time",
    y_label: str = "Sequence",
    show_legend: bool = True,
    figsize: tuple[float, float] = (10, 8),
) -> ggplot:
    """
    Create a sequence index plot.

    An index plot shows each sequence as a horizontal row, with
    states represented as colored tiles. This visualization is
    useful for seeing patterns across many sequences.

    Args:
        sequence: StateSequence or SequencePool to visualize.
        sort_by: How to sort sequences.
            - "from.start": Sort by first state.
            - "from.end": Sort by last state.
            - "length": Sort by sequence length.
            - None: Keep original order.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        show_legend: Whether to show the legend.
        figsize: Figure size as (width, height).

    Returns:
        A plotnine ggplot object.

    Example:
        >>> from yasqat.visualization import index_plot
        >>> plot = index_plot(pool, sort_by="from.start")
        >>> plot.save("sequences.png")
    """
    from yasqat.core.sequence import StateSequence

    # Get data and config
    if isinstance(sequence, StateSequence):
        data = sequence.data
        config = sequence.config
        alphabet = sequence.alphabet
    else:
        data = sequence.data
        config = sequence.config
        alphabet = sequence.alphabet

    id_col = config.id_column
    time_col = config.time_column
    state_col = config.state_column

    # Prepare plot data
    plot_data = data.clone()

    # Apply sorting if specified
    if sort_by is not None:
        plot_data = _sort_sequences(plot_data, id_col, time_col, state_col, sort_by)

    # Create y-axis position based on sequence order
    seq_ids = plot_data[id_col].unique(maintain_order=True).to_list()
    id_to_pos = {seq_id: i for i, seq_id in enumerate(seq_ids)}

    plot_data = plot_data.with_columns(pl.col(id_col).replace(id_to_pos).alias("y_pos"))

    # Convert to pandas for plotnine
    pdf = plot_data.to_pandas()

    # Build color mapping
    colors = {state: alphabet.get_color(state) for state in alphabet.states}

    # Create plot
    p = (
        ggplot(pdf, aes(x=time_col, y="y_pos", fill=state_col))
        + geom_tile(width=1, height=0.9)
        + scale_fill_manual(values=colors)
        + scale_y_reverse()  # First sequence at top
        + theme_minimal()
        + theme(
            figure_size=figsize,
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
        )
        + labs(x=x_label, y=y_label, fill="State")
    )

    if title:
        p = p + labs(title=title)

    if not show_legend:
        p = p + theme(legend_position="none")

    return p


def _sort_sequences(
    data: pl.DataFrame,
    id_col: str,
    time_col: str,
    state_col: str,
    sort_by: str,
) -> pl.DataFrame:
    """Sort sequences according to specified method."""
    if sort_by == "from.start":
        # Sort by first state
        first_states = (
            data.sort([id_col, time_col])
            .group_by(id_col)
            .first()
            .select([id_col, state_col])
            .rename({state_col: "_sort_key"})
        )
        data = data.join(first_states, on=id_col)
        data = data.sort(["_sort_key", id_col, time_col]).drop("_sort_key")

    elif sort_by == "from.end":
        # Sort by last state
        last_states = (
            data.sort([id_col, time_col])
            .group_by(id_col)
            .last()
            .select([id_col, state_col])
            .rename({state_col: "_sort_key"})
        )
        data = data.join(last_states, on=id_col)
        data = data.sort(["_sort_key", id_col, time_col]).drop("_sort_key")

    elif sort_by == "length":
        # Sort by sequence length
        lengths = data.group_by(id_col).agg(pl.len().alias("_sort_key"))
        data = data.join(lengths, on=id_col)
        data = data.sort(["_sort_key", id_col, time_col]).drop("_sort_key")

    return data
