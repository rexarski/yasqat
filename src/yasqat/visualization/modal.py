"""Modal state and mean time plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from plotnine import (
    aes,
    element_blank,
    geom_bar,
    geom_col,
    ggplot,
    labs,
    scale_fill_manual,
    theme,
    theme_minimal,
)

if TYPE_CHECKING:
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence


def modal_state_plot(
    sequence: StateSequence | SequencePool,
    title: str | None = None,
    x_label: str = "Time",
    y_label: str = "Proportion",
    figsize: tuple[float, float] = (10, 6),
) -> ggplot:
    """
    Create a bar chart of the modal (most frequent) state at each time position.

    Shows the most common state at each time point and its proportion.

    Args:
        sequence: StateSequence or SequencePool to visualize.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        figsize: Figure size as (width, height).

    Returns:
        A plotnine ggplot object.
    """
    from yasqat.core.sequence import StateSequence
    from yasqat.statistics.descriptive import modal_states

    if isinstance(sequence, StateSequence):
        alphabet = sequence.alphabet
    else:
        alphabet = sequence.alphabet

    modal_df = modal_states(sequence)
    pdf = modal_df.to_pandas()

    colors = {state: alphabet.get_color(state) for state in alphabet.states}

    p = (
        ggplot(pdf, aes(x="time", y="proportion", fill="modal_state"))
        + geom_col(alpha=0.8)
        + scale_fill_manual(values=colors)
        + theme_minimal()
        + theme(
            figure_size=figsize,
            panel_grid_minor=element_blank(),
        )
        + labs(x=x_label, y=y_label, fill="State")
    )

    if title:
        p = p + labs(title=title)

    return p


def mean_time_plot(
    sequence: StateSequence | SequencePool,
    title: str | None = None,
    x_label: str = "State",
    y_label: str = "Mean Time",
    figsize: tuple[float, float] = (10, 6),
) -> ggplot:
    """
    Create a bar chart of mean time spent in each state.

    Args:
        sequence: StateSequence or SequencePool to visualize.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        figsize: Figure size as (width, height).

    Returns:
        A plotnine ggplot object.
    """
    from yasqat.core.sequence import StateSequence
    from yasqat.statistics.descriptive import mean_time_in_state

    if isinstance(sequence, StateSequence):
        config = sequence.config
        alphabet = sequence.alphabet
    else:
        config = sequence.config
        alphabet = sequence.alphabet

    mt_df = mean_time_in_state(sequence)
    state_col = config.state_column
    pdf = mt_df.to_pandas()

    colors = {state: alphabet.get_color(state) for state in alphabet.states}

    p = (
        ggplot(pdf, aes(x=state_col, y="mean_time", fill=state_col))
        + geom_bar(stat="identity", alpha=0.8)
        + scale_fill_manual(values=colors)
        + theme_minimal()
        + theme(
            figure_size=figsize,
            panel_grid_minor=element_blank(),
        )
        + labs(x=x_label, y=y_label, fill="State")
    )

    if title:
        p = p + labs(title=title)

    return p
