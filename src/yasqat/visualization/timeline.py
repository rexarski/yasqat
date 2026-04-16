"""Timeline visualization for sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from plotnine import (
    aes,
    element_blank,
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
    from yasqat.core.sequence import StateSequence


def timeline_plot(
    sequence: StateSequence | SequencePool,
    max_sequences: int | None = 50,
    title: str | None = None,
    x_label: str = "Time",
    y_label: str = "Sequence",
    show_legend: bool | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> ggplot:
    """
    Create a timeline visualization.

    Shows sequences as horizontal timelines with state intervals marked.
    Useful for examining individual sequences in detail.

    Args:
        sequence: StateSequence or SequencePool.
        max_sequences: Maximum number of sequences to display.
            If None, show all sequences.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        show_legend: Whether to show the legend. None (default) auto-hides
            when alphabet has more than 15 states.
        figsize: Figure size as (width, height).

    Returns:
        A plotnine ggplot object.

    Example:
        >>> from yasqat.visualization import timeline_plot
        >>> plot = timeline_plot(pool, max_sequences=20)
        >>> plot.save("timeline.png")
    """
    from yasqat.core.sequence import StateSequence

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

    # Create y-axis positions.
    # replace() preserves the source column dtype (Utf8 for string IDs), so
    # integer positions land as strings. Cast to Int64 so plotnine treats
    # y_pos as a continuous scale — required by scale_y_reverse().
    id_to_pos = {seq_id: i for i, seq_id in enumerate(seq_ids)}
    spells = spells.with_columns(
        pl.col(id_col).replace(id_to_pos).cast(pl.Int64).alias("y_pos")
    )

    # Build color mapping
    colors = {state: alphabet.get_color(state) for state in alphabet.states}

    # Create plot with segments
    p = (
        ggplot(
            spells,
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

    # Auto-suppress legend when alphabet has >15 states (unreadable).
    # show_legend=None means "auto", True/False are explicit overrides.
    if show_legend is None:
        show_legend = len(alphabet) <= 15

    if not show_legend:
        p = p + theme(legend_position="none")

    return p


def spell_duration_plot(
    sequence: StateSequence | SequencePool,
    title: str | None = None,
    x_label: str = "State",
    y_label: str = "Duration",
    show_legend: bool | None = None,
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
        show_legend: Whether to show the legend. None (default) auto-hides
            when alphabet has more than 15 states.
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

    # Build color mapping
    colors = {state: alphabet.get_color(state) for state in alphabet.states}

    # Create boxplot
    p = (
        ggplot(spells, aes(x=state_col, y="duration", fill=state_col))
        + geom_boxplot(alpha=0.8)
        + scale_fill_manual(values=colors)
        + theme_minimal()
        + theme(figure_size=figsize)
        + labs(x=x_label, y=y_label)
    )

    if title:
        p = p + labs(title=title)

    # Auto-suppress legend when alphabet has >15 states (unreadable).
    # show_legend=None means "auto", True/False are explicit overrides.
    if show_legend is None:
        show_legend = len(alphabet) <= 15

    if not show_legend:
        p = p + theme(legend_position="none")

    return p
