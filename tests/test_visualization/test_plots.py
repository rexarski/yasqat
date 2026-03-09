"""Tests for visualization functions."""

import polars as pl
from plotnine import ggplot

from yasqat.core.pool import SequencePool
from yasqat.visualization import (
    distribution_plot,
    entropy_plot,
    frequency_plot,
    index_plot,
    mean_time_plot,
    modal_state_plot,
    parallel_coordinate_plot,
    spell_duration_plot,
    timeline_plot,
)


class TestIndexPlot:
    """Tests for index plot."""

    def test_creates_ggplot(self, sequence_pool: SequencePool) -> None:
        """Test that index_plot returns a ggplot object."""
        plot = index_plot(sequence_pool)

        assert isinstance(plot, ggplot)

    def test_with_sorting(self, sequence_pool: SequencePool) -> None:
        """Test index plot with sorting."""
        plot = index_plot(sequence_pool, sort_by="from.start")

        assert isinstance(plot, ggplot)

    def test_with_title(self, sequence_pool: SequencePool) -> None:
        """Test index plot with title."""
        plot = index_plot(sequence_pool, title="Test Plot")

        assert isinstance(plot, ggplot)

    def test_without_legend(self, sequence_pool: SequencePool) -> None:
        """Test index plot without legend."""
        plot = index_plot(sequence_pool, show_legend=False)

        assert isinstance(plot, ggplot)


class TestDistributionPlot:
    """Tests for distribution plot."""

    def test_creates_ggplot(self, sequence_pool: SequencePool) -> None:
        """Test that distribution_plot returns a ggplot object."""
        plot = distribution_plot(sequence_pool)

        assert isinstance(plot, ggplot)

    def test_stacked_false(self, sequence_pool: SequencePool) -> None:
        """Test distribution plot with stacked=False."""
        plot = distribution_plot(sequence_pool, stacked=False)

        assert isinstance(plot, ggplot)


class TestEntropyPlot:
    """Tests for entropy plot."""

    def test_creates_ggplot(self, sequence_pool: SequencePool) -> None:
        """Test that entropy_plot returns a ggplot object."""
        plot = entropy_plot(sequence_pool)

        assert isinstance(plot, ggplot)

    def test_without_normalization(self, sequence_pool: SequencePool) -> None:
        """Test entropy plot without normalization."""
        plot = entropy_plot(sequence_pool, normalize=False)

        assert isinstance(plot, ggplot)


class TestTimelinePlot:
    """Tests for timeline plot."""

    def test_creates_ggplot(self, sequence_pool: SequencePool) -> None:
        """Test that timeline_plot returns a ggplot object."""
        plot = timeline_plot(sequence_pool)

        assert isinstance(plot, ggplot)

    def test_with_max_sequences(self, sequence_pool: SequencePool) -> None:
        """Test timeline plot with max_sequences limit."""
        plot = timeline_plot(sequence_pool, max_sequences=2)

        assert isinstance(plot, ggplot)


class TestFrequencyPlot:
    """Tests for frequency plot."""

    def test_creates_ggplot(self, sequence_pool: SequencePool) -> None:
        """Test that frequency_plot returns a ggplot object."""
        plot = frequency_plot(sequence_pool)

        assert isinstance(plot, ggplot)

    def test_with_n_most_frequent(self, sequence_pool: SequencePool) -> None:
        """Test frequency plot with n_most_frequent."""
        plot = frequency_plot(sequence_pool, n_most_frequent=2)

        assert isinstance(plot, ggplot)


class TestSpellDurationPlot:
    """Tests for spell duration plot."""

    def test_creates_ggplot(self, sequence_pool: SequencePool) -> None:
        """Test that spell_duration_plot returns a ggplot object."""
        plot = spell_duration_plot(sequence_pool)

        assert isinstance(plot, ggplot)

    def test_with_title(self, sequence_pool: SequencePool) -> None:
        """Test spell duration plot with title."""
        plot = spell_duration_plot(sequence_pool, title="Test")

        assert isinstance(plot, ggplot)


class TestModalStatePlot:
    """Tests for modal state plot."""

    def test_creates_ggplot(self, sequence_pool: SequencePool) -> None:
        plot = modal_state_plot(sequence_pool)
        assert isinstance(plot, ggplot)

    def test_with_title(self, sequence_pool: SequencePool) -> None:
        plot = modal_state_plot(sequence_pool, title="Modal States")
        assert isinstance(plot, ggplot)


class TestMeanTimePlot:
    """Tests for mean time plot."""

    def test_creates_ggplot(self, sequence_pool: SequencePool) -> None:
        plot = mean_time_plot(sequence_pool)
        assert isinstance(plot, ggplot)

    def test_with_title(self, sequence_pool: SequencePool) -> None:
        plot = mean_time_plot(sequence_pool, title="Mean Time")
        assert isinstance(plot, ggplot)


class TestParallelCoordinatePlot:
    """Tests for parallel coordinate plot."""

    def test_creates_ggplot(self, sequence_pool: SequencePool) -> None:
        plot = parallel_coordinate_plot(sequence_pool)
        assert isinstance(plot, ggplot)

    def test_with_max_sequences(self, sequence_pool: SequencePool) -> None:
        plot = parallel_coordinate_plot(sequence_pool, max_sequences=2, seed=42)
        assert isinstance(plot, ggplot)

    def test_with_title(self, sequence_pool: SequencePool) -> None:
        plot = parallel_coordinate_plot(sequence_pool, title="PC Plot")
        assert isinstance(plot, ggplot)


def _make_many_state_pool(n_states: int = 20) -> SequencePool:
    """Helper: build a pool with n_states distinct states."""
    states = [f"S{i}" for i in range(n_states)]
    rows = []
    for seq_id in range(3):
        for t, s in enumerate(states):
            rows.append({"id": seq_id, "time": t, "state": s})
    return SequencePool(pl.DataFrame(rows))


def _has_legend_none(plot: ggplot) -> bool:
    """Check whether the plot's theme sets legend_position to 'none'."""
    items = plot.theme.themeables
    if "legend_position" not in items:
        return False
    return items["legend_position"].properties["value"] == "none"


class TestAutoLegendSuppression:
    """Tests for automatic legend suppression with >15 categories."""

    def test_index_plot_auto_hidden(self) -> None:
        """Test that index_plot auto-hides legend for >15 states."""
        pool = _make_many_state_pool(20)
        plot = index_plot(pool)
        assert _has_legend_none(plot)

    def test_index_plot_explicit_true_overrides(self) -> None:
        """Test that show_legend=True forces legend even with >15 states."""
        pool = _make_many_state_pool(20)
        plot = index_plot(pool, show_legend=True)
        assert not _has_legend_none(plot)

    def test_index_plot_explicit_false(self, sequence_pool: SequencePool) -> None:
        """Test that show_legend=False hides legend even with few states."""
        plot = index_plot(sequence_pool, show_legend=False)
        assert _has_legend_none(plot)

    def test_index_plot_shown_for_few_states(
        self, sequence_pool: SequencePool
    ) -> None:
        """Test that legend is shown for <=15 states by default."""
        plot = index_plot(sequence_pool)
        assert not _has_legend_none(plot)

    def test_distribution_plot_auto_hidden(self) -> None:
        """Test that distribution_plot auto-hides legend for >15 states."""
        pool = _make_many_state_pool(20)
        plot = distribution_plot(pool)
        assert _has_legend_none(plot)

    def test_timeline_plot_auto_hidden(self) -> None:
        """Test that timeline_plot auto-hides legend for >15 states."""
        pool = _make_many_state_pool(20)
        plot = timeline_plot(pool)
        assert _has_legend_none(plot)

    def test_spell_duration_plot_auto_hidden(self) -> None:
        """Test that spell_duration_plot auto-hides legend for >15 states."""
        pool = _make_many_state_pool(20)
        plot = spell_duration_plot(pool)
        assert _has_legend_none(plot)

    def test_spell_duration_plot_explicit_true(self) -> None:
        """Test that spell_duration_plot respects show_legend=True."""
        pool = _make_many_state_pool(20)
        plot = spell_duration_plot(pool, show_legend=True)
        assert not _has_legend_none(plot)
