"""Timeline visualization for sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from plotnine import (
    aes,
    element_blank,
    geom_point,
    geom_segment,
    ggplot,
    labs,
    scale_color_manual,
    scale_y_reverse,
    theme,
    theme_minimal,
)

if TYPE_CHECKING:
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import EventSequence, StateSequence


def timeline_plot(
    sequence: StateSequence | EventSequence | SequencePool,
    max_sequences: int | None = 50,
    title: str | None = None,
    x_label: str = "Time",
    y_label: str = "Sequence",
    show_legend: bool = True,
    figsize: tuple[float, float] = (12, 8),
) -> ggplot:
    """
    Create a timeline visualization.

    Shows sequences as horizontal timelines with events or state
    intervals marked. Useful for examining individual sequences
    in detail.

    Args:
        sequence: StateSequence, EventSequence, or SequencePool.
        max_sequences: Maximum number of sequences to display.
            If None, show all sequences.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        show_legend: Whether to show the legend.
        figsize: Figure size as (width, height).

    Returns:
        A plotnine ggplot object.

    Example:
        >>> from yasqat.visualization import timeline_plot
        >>> plot = timeline_plot(pool, max_sequences=20)
        >>> plot.save("timeline.png")
    """
    from yasqat.core.sequence import EventSequence, StateSequence

    # Get data and config based on input type
    if isinstance(sequence, EventSequence):
        return _event_timeline_plot(
            sequence,
            max_sequences=max_sequences,
            title=title,
            x_label=x_label,
            y_label=y_label,
            show_legend=show_legend,
            figsize=figsize,
        )

    if isinstance(sequence, StateSequence):
        data = sequence.data
        config = sequence.config
        alphabet = sequence.alphabet
    else:  # SequencePool
        data = sequence.data
        config = sequence.config
        alphabet = sequence.alphabet

    id_col = config.id_column
    state_col = config.state_column

    # Convert to spells (intervals)
    state_seq = StateSequence(data, config, alphabet)
    spells = state_seq.to_sps()

    # Limit sequences if needed
    seq_ids = spells[id_col].unique(maintain_order=True).to_list()
    if max_sequences is not None and len(seq_ids) > max_sequences:
        seq_ids = seq_ids[:max_sequences]
        spells = spells.filter(pl.col(id_col).is_in(seq_ids))

    # Create y-axis positions
    id_to_pos = {seq_id: i for i, seq_id in enumerate(seq_ids)}
    spells = spells.with_columns(pl.col(id_col).replace(id_to_pos).alias("y_pos"))

    # Convert to pandas for plotnine
    pdf = spells.to_pandas()

    # Build color mapping
    colors = {state: alphabet.get_color(state) for state in alphabet.states}

    # Create plot with segments
    p = (
        ggplot(
            pdf,
            aes(
                x="start",
                xend="end",
                y="y_pos",
                yend="y_pos",
                color=state_col,
            ),
        )
        + geom_segment(size=6, alpha=0.8)
        + scale_color_manual(values=colors)
        + scale_y_reverse()
        + theme_minimal()
        + theme(
            figure_size=figsize,
            panel_grid_major_y=element_blank(),
            panel_grid_minor=element_blank(),
        )
        + labs(x=x_label, y=y_label, color="State")
    )

    if title:
        p = p + labs(title=title)

    if not show_legend:
        p = p + theme(legend_position="none")

    return p


def _event_timeline_plot(
    sequence: EventSequence,
    max_sequences: int | None = 50,
    title: str | None = None,
    x_label: str = "Time",
    y_label: str = "Sequence",
    show_legend: bool = True,
    figsize: tuple[float, float] = (12, 8),
) -> ggplot:
    """Create timeline plot for event sequences."""
    data = sequence.data
    config = sequence.config
    alphabet = sequence.alphabet

    id_col = config.id_column
    time_col = config.time_column
    state_col = config.state_column

    # Limit sequences if needed
    seq_ids = data[id_col].unique(maintain_order=True).to_list()
    if max_sequences is not None and len(seq_ids) > max_sequences:
        seq_ids = seq_ids[:max_sequences]
        data = data.filter(pl.col(id_col).is_in(seq_ids))

    # Create y-axis positions
    id_to_pos = {seq_id: i for i, seq_id in enumerate(seq_ids)}
    plot_data = data.with_columns(pl.col(id_col).replace(id_to_pos).alias("y_pos"))

    # Convert to pandas for plotnine
    pdf = plot_data.to_pandas()

    # Build color mapping
    colors = {state: alphabet.get_color(state) for state in alphabet.states}

    # Create plot with points for events
    p = (
        ggplot(pdf, aes(x=time_col, y="y_pos", color=state_col))
        + geom_point(size=3, alpha=0.8)
        + scale_color_manual(values=colors)
        + scale_y_reverse()
        + theme_minimal()
        + theme(
            figure_size=figsize,
            panel_grid_major_y=element_blank(),
            panel_grid_minor=element_blank(),
        )
        + labs(x=x_label, y=y_label, color="Event")
    )

    if title:
        p = p + labs(title=title)

    if not show_legend:
        p = p + theme(legend_position="none")

    return p


def spell_duration_plot(
    sequence: StateSequence | SequencePool,
    title: str | None = None,
    x_label: str = "State",
    y_label: str = "Duration",
    figsize: tuple[float, float] = (10, 6),
) -> ggplot:
    """
    Create a boxplot of spell durations by state.

    Shows the distribution of how long sequences stay in each state.

    Args:
        sequence: StateSequence or SequencePool.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        figsize: Figure size as (width, height).

    Returns:
        A plotnine ggplot object.
    """
    from plotnine import geom_boxplot, scale_fill_manual

    from yasqat.core.sequence import StateSequence

    # Get data
    if isinstance(sequence, StateSequence):
        state_seq = sequence
        alphabet = sequence.alphabet
        config = sequence.config
    else:
        state_seq = sequence.to_state_sequence()
        alphabet = sequence.alphabet
        config = sequence.config

    # Get spells
    spells = state_seq.to_sps()
    state_col = config.state_column

    # Convert to pandas
    pdf = spells.to_pandas()

    # Build color mapping
    colors = {state: alphabet.get_color(state) for state in alphabet.states}

    # Create boxplot
    p = (
        ggplot(pdf, aes(x=state_col, y="duration", fill=state_col))
        + geom_boxplot(alpha=0.8)
        + scale_fill_manual(values=colors)
        + theme_minimal()
        + theme(
            figure_size=figsize,
            legend_position="none",
        )
        + labs(x=x_label, y=y_label)
    )

    if title:
        p = p + labs(title=title)

    return p
