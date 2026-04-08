"""Sunburst plot for hierarchical sequence visualization."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from plotnine import (
    aes,
    coord_equal,
    geom_polygon,
    ggplot,
    labs,
    scale_fill_manual,
    theme,
    theme_void,
)

if TYPE_CHECKING:
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence


def sunburst_plot(
    sequence: StateSequence | SequencePool,
    max_depth: int = 5,
    title: str | None = None,
    show_legend: bool | None = None,
    figsize: tuple[float, float] = (8, 8),
) -> ggplot:
    """
    Create a sunburst plot showing hierarchical state transitions.

    The sunburst displays concentric rings where each ring represents
    a time position. The innermost ring shows state proportions at
    time 0, and each outer ring shows state proportions conditioned
    on the path from the center.

    Args:
        sequence: StateSequence or SequencePool to visualize.
        max_depth: Maximum number of rings (time positions) to show.
        title: Plot title.
        show_legend: Whether to show the legend. None (default) auto-hides
            when alphabet has more than 15 states.
        figsize: Figure size as (width, height).

    Returns:
        A plotnine ggplot object.

    Example:
        >>> from yasqat.visualization import sunburst_plot
        >>> plot = sunburst_plot(pool, max_depth=5)
        >>> plot.save("sunburst.png")
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

    alphabet = pool.alphabet

    # Get all sequences and truncate to max_depth
    all_seqs = [pool.get_sequence(sid) for sid in pool.sequence_ids]
    time_points = sorted(pool.data[pool.config.time_column].unique().to_list())
    depth = min(max_depth, len(time_points))

    truncated = [seq[:depth] for seq in all_seqs]
    n_seqs = len(truncated)

    # Build arc polygon data for each depth level
    arc_records: list[dict[str, object]] = []
    arc_id = 0

    for d in range(depth):
        # Count prefixes of length d+1
        prefix_counts: Counter[tuple[str, ...]] = Counter()
        for seq in truncated:
            if len(seq) > d:
                prefix_counts[tuple(seq[: d + 1])] += 1

        # Sort prefixes for consistent arc ordering
        sorted_prefixes = sorted(prefix_counts.keys())

        # Compute angular positions and create arc polygons
        angle_pos = 0.0
        for prefix in sorted_prefixes:
            count = prefix_counts[prefix]
            angle_span = count / n_seqs * 2 * np.pi

            r_inner = d + 0.5
            r_outer = d + 1.4
            n_pts = max(4, int(angle_span * 20))

            theta = np.linspace(angle_pos, angle_pos + angle_span, n_pts)
            x_inner = r_inner * np.cos(theta)
            y_inner = r_inner * np.sin(theta)
            x_outer = r_outer * np.cos(theta[::-1])
            y_outer = r_outer * np.sin(theta[::-1])

            xs = np.concatenate([x_inner, x_outer])
            ys = np.concatenate([y_inner, y_outer])

            state = prefix[-1]
            for x_val, y_val in zip(xs, ys, strict=True):
                arc_records.append(
                    {
                        "x": float(x_val),
                        "y": float(y_val),
                        "state": state,
                        "group": arc_id,
                    }
                )
            arc_id += 1
            angle_pos += angle_span

    plot_df = pl.DataFrame(arc_records).to_pandas()

    # Build color mapping
    colors = {state: alphabet.get_color(state) for state in alphabet.states}

    # Create plot
    p = (
        ggplot(plot_df, aes(x="x", y="y", fill="state", group="group"))
        + geom_polygon(color="white", size=0.3)
        + scale_fill_manual(values=colors)
        + coord_equal()
        + theme_void()
        + theme(figure_size=figsize)
        + labs(fill="State")
    )

    if title:
        p = p + labs(title=title)

    # Auto-suppress legend when alphabet has >15 states (unreadable).
    if show_legend is None:
        show_legend = len(alphabet) <= 15

    if not show_legend:
        p = p + theme(legend_position="none")

    return p
