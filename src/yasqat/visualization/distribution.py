"""Distribution and entropy plots for sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from plotnine import (
    aes,
    element_blank,
    geom_area,
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_fill_manual,
    scale_y_continuous,
    theme,
    theme_minimal,
)

if TYPE_CHECKING:
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence


def distribution_plot(
    sequence: StateSequence | SequencePool,
    stacked: bool = True,
    title: str | None = None,
    x_label: str = "Time",
    y_label: str = "Proportion",
    show_legend: bool = True,
    figsize: tuple[float, float] = (10, 6),
) -> ggplot:
    """
    Create a state distribution plot over time.

    Shows the proportion of sequences in each state at each time point.
    This is useful for understanding aggregate patterns and trends.

    Args:
        sequence: StateSequence or SequencePool to visualize.
        stacked: If True, create stacked area chart. If False, line chart.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        show_legend: Whether to show the legend.
        figsize: Figure size as (width, height).

    Returns:
        A plotnine ggplot object.

    Example:
        >>> from yasqat.visualization import distribution_plot
        >>> plot = distribution_plot(pool, stacked=True)
        >>> plot.save("distribution.png")
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

    time_col = config.time_column
    state_col = config.state_column

    # Calculate proportions at each time point
    counts = data.group_by([time_col, state_col]).agg(pl.len().alias("count"))

    totals = data.group_by(time_col).agg(pl.len().alias("total"))

    proportions = counts.join(totals, on=time_col).with_columns(
        (pl.col("count") / pl.col("total")).alias("proportion")
    )

    # Convert to pandas for plotnine
    pdf = proportions.to_pandas()

    # Build color mapping
    colors = {state: alphabet.get_color(state) for state in alphabet.states}

    # Create plot
    if stacked:
        p = (
            ggplot(pdf, aes(x=time_col, y="proportion", fill=state_col))
            + geom_area(position="stack", alpha=0.8)
            + scale_fill_manual(values=colors)
        )
    else:
        p = ggplot(pdf, aes(x=time_col, y="proportion", color=state_col)) + geom_line(
            size=1
        )

    p = (
        p
        + theme_minimal()
        + theme(
            figure_size=figsize,
            panel_grid_minor=element_blank(),
        )
        + scale_y_continuous(limits=(0, 1))
        + labs(x=x_label, y=y_label, fill="State", color="State")
    )

    if title:
        p = p + labs(title=title)

    if not show_legend:
        p = p + theme(legend_position="none")

    return p


def entropy_plot(
    sequence: StateSequence | SequencePool,
    normalize: bool = True,
    title: str | None = None,
    x_label: str = "Time",
    y_label: str = "Entropy",
    show_points: bool = True,
    figsize: tuple[float, float] = (10, 4),
) -> ggplot:
    """
    Create a transversal entropy plot over time.

    Shows the entropy (diversity) of states at each time point.
    Higher entropy indicates more diverse state usage across sequences.

    Args:
        sequence: StateSequence or SequencePool to visualize.
        normalize: If True, normalize entropy by maximum possible.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        show_points: Whether to show points on the line.
        figsize: Figure size as (width, height).

    Returns:
        A plotnine ggplot object.

    Example:
        >>> from yasqat.visualization import entropy_plot
        >>> plot = entropy_plot(pool)
        >>> plot.save("entropy.png")
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

    time_col = config.time_column
    state_col = config.state_column
    n_states = len(alphabet)

    # Calculate entropy at each time point
    time_points = data[time_col].unique().sort().to_list()
    entropies = []

    for t in time_points:
        t_data = data.filter(pl.col(time_col) == t)
        counts = t_data[state_col].value_counts()
        total = counts["count"].sum()

        entropy = 0.0
        for count in counts["count"].to_list():
            p = count / total
            if p > 0:
                entropy -= p * np.log(p)

        if normalize and n_states > 1:
            max_entropy = np.log(n_states)
            if max_entropy > 0:
                entropy /= max_entropy

        entropies.append(entropy)

    # Create DataFrame
    entropy_df = pl.DataFrame(
        {
            time_col: time_points,
            "entropy": entropies,
        }
    ).to_pandas()

    # Create plot
    p = (
        ggplot(entropy_df, aes(x=time_col, y="entropy"))
        + geom_line(color="#377EB8", size=1)
        + theme_minimal()
        + theme(
            figure_size=figsize,
            panel_grid_minor=element_blank(),
        )
        + labs(x=x_label, y=y_label)
    )

    if normalize:
        p = p + scale_y_continuous(limits=(0, 1))

    if show_points:
        p = p + geom_point(color="#377EB8", size=2)

    if title:
        p = p + labs(title=title)

    return p


def frequency_plot(
    sequence: StateSequence | SequencePool,
    n_most_frequent: int = 10,
    title: str | None = None,
    x_label: str = "Sequence Pattern",
    y_label: str = "Frequency",
    figsize: tuple[float, float] = (10, 6),
) -> ggplot:
    """
    Create a plot of most frequent sequence patterns.

    Shows the most common complete sequence patterns and their frequencies.

    Args:
        sequence: StateSequence or SequencePool to visualize.
        n_most_frequent: Number of most frequent patterns to show.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        figsize: Figure size as (width, height).

    Returns:
        A plotnine ggplot object.
    """
    from plotnine import coord_flip, geom_bar

    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence

    # Get data and config
    if isinstance(sequence, StateSequence):
        pool = SequencePool(
            data=sequence.data,
            config=sequence.config,
            alphabet=sequence.alphabet,
        )
    else:
        pool = sequence

    # Get sequence patterns
    patterns = []
    for seq_id in pool.sequence_ids:
        states = pool.get_sequence(seq_id)
        pattern = "-".join(states)
        patterns.append(pattern)

    # Count patterns
    pattern_counts: dict[str, int] = {}
    for pat in patterns:
        pattern_counts[pat] = pattern_counts.get(pat, 0) + 1

    # Get top N
    sorted_patterns = sorted(
        pattern_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:n_most_frequent]

    freq_df = pl.DataFrame(
        {
            "pattern": [pat for pat, _ in sorted_patterns],
            "frequency": [cnt for _, cnt in sorted_patterns],
        }
    ).to_pandas()

    # Create plot
    plot = (
        ggplot(freq_df, aes(x="pattern", y="frequency"))
        + geom_bar(stat="identity", fill="#377EB8", alpha=0.8)
        + coord_flip()
        + theme_minimal()
        + theme(figure_size=figsize)
        + labs(x=x_label, y=y_label)
    )

    if title:
        plot = plot + labs(title=title)

    return plot
