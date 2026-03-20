"""Tree plot for sequence branching visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    scale_size_continuous,
    theme,
    theme_minimal,
)

if TYPE_CHECKING:
    from yasqat.core.pool import SequencePool
    from yasqat.core.sequence import StateSequence


@dataclass
class _TreeNode:
    """Internal tree node for prefix tree."""

    state: str
    depth: int
    count: int = 0
    children: dict[str, _TreeNode] = field(default_factory=dict)
    y: float = 0.0


def _build_trie(
    sequences: list[list[str]], max_depth: int, min_count: int
) -> _TreeNode:
    """Build a prefix tree from sequences, pruning rare branches."""
    root = _TreeNode(state="root", depth=0, count=len(sequences))
    for seq in sequences:
        node = root
        for d, state in enumerate(seq[:max_depth]):
            if state not in node.children:
                node.children[state] = _TreeNode(state=state, depth=d + 1)
            node.children[state].count += 1
            node = node.children[state]

    _prune(root, min_count)
    return root


def _prune(node: _TreeNode, min_count: int) -> None:
    """Remove children below minimum count."""
    to_remove = [s for s, child in node.children.items() if child.count < min_count]
    for s in to_remove:
        del node.children[s]
    for child in node.children.values():
        _prune(child, min_count)


def _layout(node: _TreeNode, y_start: float = 0.0) -> float:
    """Compute y positions for tree nodes. Returns total span used."""
    if not node.children:
        node.y = y_start + node.count / 2.0
        return float(node.count)

    total = 0.0
    for child_state in sorted(node.children.keys()):
        child = node.children[child_state]
        span = _layout(child, y_start + total)
        total += span

    node.y = y_start + total / 2.0
    return total


def _collect(
    node: _TreeNode,
    segments: list[dict[str, object]],
    points: list[dict[str, object]],
) -> None:
    """Collect segments and points from tree for plotting."""
    for child in node.children.values():
        segments.append(
            {
                "x": float(node.depth),
                "y": node.y,
                "xend": float(child.depth),
                "yend": child.y,
                "state": child.state,
                "count": child.count,
            }
        )
        points.append(
            {
                "x": float(child.depth),
                "y": child.y,
                "state": child.state,
                "count": child.count,
            }
        )
        _collect(child, segments, points)


def tree_plot(
    sequence: StateSequence | SequencePool,
    max_depth: int = 5,
    min_support: float = 0.01,
    title: str | None = None,
    x_label: str = "Time",
    show_legend: bool | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> ggplot:
    """
    Create a sequence tree plot showing branching patterns.

    Displays how sequences branch over time as a prefix tree.
    Each node represents a state at a given time position, with
    branches connecting states across consecutive time positions.
    Branch thickness indicates the number of sequences following
    that path.

    Args:
        sequence: StateSequence or SequencePool to visualize.
        max_depth: Maximum tree depth (time positions) to show.
        min_support: Minimum proportion of sequences for a branch
            to be shown (pruning threshold).
        title: Plot title.
        x_label: X-axis label.
        show_legend: Whether to show the legend. None (default) auto-hides
            when alphabet has more than 15 states.
        figsize: Figure size as (width, height).

    Returns:
        A plotnine ggplot object.

    Example:
        >>> from yasqat.visualization import tree_plot
        >>> plot = tree_plot(pool, max_depth=5, min_support=0.05)
        >>> plot.save("tree.png")
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

    # Get all sequences
    all_seqs = [pool.get_sequence(sid) for sid in pool.sequence_ids]
    n_seqs = len(all_seqs)
    min_count = max(1, int(min_support * n_seqs))

    # Build and layout tree
    root = _build_trie(all_seqs, max_depth, min_count)
    _layout(root)

    # Collect plot data
    segments: list[dict[str, object]] = []
    points: list[dict[str, object]] = []
    _collect(root, segments, points)

    if not segments:
        # Fallback: show root only
        segments = [
            {"x": 0, "y": 0, "xend": 1, "yend": 0, "state": "root", "count": 1}
        ]
        points = [{"x": 0, "y": 0, "state": "root", "count": 1}]

    seg_df = pl.DataFrame(segments).to_pandas()
    pt_df = pl.DataFrame(points).to_pandas()

    # Build color mapping
    colors = {state: alphabet.get_color(state) for state in alphabet.states}
    if "root" not in colors:
        colors["root"] = "#999999"

    # Create plot
    p = (
        ggplot()
        + geom_segment(
            seg_df,
            aes(x="x", y="y", xend="xend", yend="yend", color="state", size="count"),
            alpha=0.7,
        )
        + geom_point(pt_df, aes(x="x", y="y", color="state", size="count"))
        + scale_color_manual(values=colors)
        + scale_size_continuous(range=(0.5, 4))
        + theme_minimal()
        + theme(
            figure_size=figsize,
            panel_grid_minor=element_blank(),
            axis_text_y=element_blank(),
            axis_ticks_major_y=element_blank(),
        )
        + labs(x=x_label, y="", color="State", size="Count")
    )

    if title:
        p = p + labs(title=title)

    # Auto-suppress legend when alphabet has >15 states (unreadable).
    if show_legend is None:
        show_legend = len(alphabet) <= 15

    if not show_legend:
        p = p + theme(legend_position="none")

    return p
