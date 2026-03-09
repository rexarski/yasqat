# yasqat Documentation Site — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Quarto documentation website for yasqat, hosted on GitHub Pages at `https://rexarski.github.io/yasqat`, with auto-generated API reference via quartodoc and taste-informed design.

**Architecture:** Quarto website format (not book) in `/docs`, Bootstrap 5 + custom SCSS, quartodoc auto-generates API pages from docstrings on every build. GitHub Actions deploys the rendered `_site/` to the `gh-pages` branch on push to `main`.

**Tech Stack:** Quarto CLI, quartodoc (pip), Bootstrap 5 SCSS, GitHub Actions, GitHub Pages

**Design reference:** `docs/plans/2026-03-04-documentation-design.md`

---

## Prerequisites (manual, one-time)

Before starting tasks, install Quarto CLI if not already present:

```bash
# macOS with Homebrew
brew install quarto

# Verify
quarto --version   # should print 1.4.x or later
```

Install quartodoc into the project's dev environment:

```bash
# With uv (recommended for this project)
uv add --dev quartodoc

# Verify
uv run quartodoc --version
```

Confirm yasqat is importable in development mode (needed for quartodoc):

```bash
uv run python -c "import yasqat; print(yasqat.__version__)"
# Expected: 0.3.0
```

---

## Task 1: Bootstrap site structure

**Files:**
- Create: `docs/_quarto.yml`
- Create: `docs/styles.scss`

**Step 1: Create `docs/_quarto.yml`**

```yaml
project:
  type: website
  output-dir: _site

website:
  title: "yasqat"
  site-url: "https://rexarski.github.io/yasqat"
  description: "Sequence analysis for social science — built on polars and plotnine"
  repo-url: "https://github.com/rexarski/yasqat"
  repo-actions: [issue]

  navbar:
    background: "#18181b"
    foreground: "#ffffff"
    logo-alt: "yasqat"
    left:
      - text: "**yasqat**"
        href: index.qmd
    right:
      - text: "Get Started"
        href: getting-started/installation.qmd
      - text: "Tutorials"
        href: tutorials/index.qmd
      - text: "API"
        href: api/core.qmd
      - text: "Demo"
        href: demo.qmd
      - text: "Changelog"
        href: changelog.qmd
      - icon: github
        href: https://github.com/rexarski/yasqat

  sidebar:
    - id: getting-started
      title: "Get Started"
      style: "docked"
      contents:
        - getting-started/installation.qmd
        - getting-started/quickstart.qmd

    - id: tutorials
      title: "Tutorials"
      style: "docked"
      contents:
        - tutorials/index.qmd
        - tutorials/data-structures.qmd
        - tutorials/statistics.qmd
        - tutorials/distance.qmd
        - tutorials/clustering.qmd
        - tutorials/visualization.qmd

    - id: api
      title: "API Reference"
      style: "docked"
      contents:
        - api/core.qmd
        - api/metrics.qmd
        - api/statistics.qmd
        - api/clustering.qmd
        - api/visualization.qmd

  page-footer:
    center: "yasqat · MIT License · Built with [Quarto](https://quarto.org)"

format:
  html:
    theme: [cosmo, styles.scss]
    highlight-style: github
    code-copy: true
    code-overflow: wrap
    toc: true
    toc-depth: 3
    number-sections: false
    smooth-scroll: true
    link-external-newwindow: true

execute:
  freeze: auto
  echo: true

quartodoc:
  package: yasqat
  source_dir: src
  dir: api
  title: "API Reference"
  sidebar: "api"
  sections:
    - title: Core
      desc: "Core data structures — Alphabet, SequencePool, sequence types, trajectories"
      contents:
        - yasqat.core.alphabet
        - yasqat.core.pool
        - yasqat.core.sequence
        - yasqat.core.trajectory
    - title: Metrics
      desc: "Pairwise distance metrics — OM, Hamming, LCS, DTW, and more"
      contents:
        - yasqat.metrics.hamming
        - yasqat.metrics.optimal_matching
        - yasqat.metrics.lcs
        - yasqat.metrics.dtw
        - yasqat.metrics.base
    - title: Statistics
      desc: "Descriptive statistics, normative indicators, transition analysis, subsequence mining"
      contents:
        - yasqat.statistics.descriptive
        - yasqat.statistics.normative
        - yasqat.statistics.transition
        - yasqat.statistics.discrepancy
        - yasqat.statistics.subsequence_mining
    - title: Clustering
      desc: "PAM, CLARA, hierarchical clustering, and quality metrics"
      contents:
        - yasqat.clustering.pam
        - yasqat.clustering.clara
        - yasqat.clustering.hierarchical
        - yasqat.clustering.quality
        - yasqat.clustering.representatives
    - title: Visualization
      desc: "plotnine-based sequence plots"
      contents:
        - yasqat.visualization.index_plot
        - yasqat.visualization.timeline
        - yasqat.visualization.distribution
        - yasqat.visualization.modal
        - yasqat.visualization.parallel
```

**Step 2: Create `docs/styles.scss`**

```scss
/*-- scss:defaults --*/

// Typography — Outfit + JetBrains Mono (taste: no Inter)
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=JetBrains+Mono:ital,wght@0,400;0,500;1,400&display=swap');

$font-family-sans-serif: 'Outfit', system-ui, -apple-system, sans-serif;
$font-family-monospace: 'JetBrains Mono', ui-monospace, 'Cascadia Code', monospace;
$font-size-base: 1rem;
$line-height-base: 1.7;
$headings-font-weight: 700;

// Color palette — zinc neutral base + deep teal accent
$body-bg: #fafafa;
$body-color: #18181b;
$link-color: #0f766e;
$link-hover-color: #115e59;
$border-color: #e4e4e7;
$code-bg: #f4f4f5;
$pre-bg: #f4f4f5;

// Navbar
$navbar-bg: #18181b;
$navbar-fg: #ffffff;
$navbar-hl: #0f766e;

/*-- scss:rules --*/

// Headings — tight tracking for display weight
h1, h2, h3 { letter-spacing: -0.02em; }
h1 { font-size: 2.25rem; }
h2 { font-size: 1.5rem; border-bottom: 1px solid #e4e4e7; padding-bottom: 0.4rem; }

// Body width — 65ch for comfortable reading
.content-container, article {
  max-width: 72ch;
}

// Sidebar active state transition (fluidity)
.sidebar-item a {
  transition: color 0.15s cubic-bezier(0.16, 1, 0.3, 1),
              border-left-color 0.15s cubic-bezier(0.16, 1, 0.3, 1);
}
.sidebar-item.active > a {
  color: #0f766e;
  font-weight: 600;
  border-left: 2px solid #0f766e;
  padding-left: 0.5rem;
}

// Code blocks — clean zinc background
pre.sourceCode {
  background: #f4f4f5;
  border: 1px solid #e4e4e7;
  border-radius: 6px;
  font-size: 0.875rem;
}
code.sourceCode { background: transparent; }
p code {
  background: #f4f4f5;
  border: 1px solid #e4e4e7;
  border-radius: 3px;
  padding: 0.1em 0.35em;
  font-size: 0.875em;
}

// Callouts — teal left border, no loud background (taste: selective delight)
.callout {
  border-left-width: 3px;
  background: transparent !important;
}
.callout-tip    { border-left-color: #0f766e !important; }
.callout-note   { border-left-color: #71717a !important; }
.callout-warning { border-left-color: #d97706 !important; }
.callout-caution { border-left-color: #dc2626 !important; }
.callout .callout-title { font-weight: 600; }

// Copy button — visible only on hover (taste: hide until needed)
div.sourceCode:hover .code-copy-button { opacity: 1; }
.code-copy-button {
  opacity: 0;
  transition: opacity 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}

// Footer — muted
.nav-footer { color: #71717a; font-size: 0.875rem; border-top: 1px solid #e4e4e7; }

// Landing page hero (applied via .hero class on index.qmd)
.hero {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 3rem;
  padding: 5rem 0 4rem;
  align-items: center;
}
.hero-text h1 {
  font-size: clamp(2.5rem, 5vw, 3.5rem);
  letter-spacing: -0.04em;
  line-height: 1.05;
  margin-bottom: 1rem;
}
.hero-text .lead {
  color: #71717a;
  font-size: 1.125rem;
  line-height: 1.6;
  max-width: 42ch;
  margin-bottom: 2rem;
}
.hero-text .cta-group { display: flex; gap: 0.75rem; flex-wrap: wrap; }
.btn-primary-custom {
  background: #0f766e;
  color: #fff;
  border: none;
  padding: 0.65rem 1.5rem;
  border-radius: 6px;
  font-weight: 600;
  text-decoration: none;
  transition: background 0.15s cubic-bezier(0.16, 1, 0.3, 1),
              transform 0.1s cubic-bezier(0.16, 1, 0.3, 1);
}
.btn-primary-custom:hover { background: #115e59; transform: translateY(-1px); color: #fff; }
.btn-primary-custom:active { transform: scale(0.98); }
.btn-secondary-custom {
  background: transparent;
  color: #18181b;
  border: 1px solid #e4e4e7;
  padding: 0.65rem 1.5rem;
  border-radius: 6px;
  font-weight: 600;
  text-decoration: none;
  transition: border-color 0.15s cubic-bezier(0.16, 1, 0.3, 1);
}
.btn-secondary-custom:hover { border-color: #0f766e; color: #0f766e; }

.hero-code pre {
  background: #18181b;
  color: #e4e4e7;
  border-radius: 10px;
  padding: 1.5rem;
  font-size: 0.875rem;
  line-height: 1.6;
  border: none;
  box-shadow: 0 4px 24px rgba(0,0,0,0.08);
}

// Feature zig-zag section
.feature-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 5rem 3rem;
  padding: 4rem 0;
}
.feature-grid .feature:nth-child(even) { direction: rtl; }
.feature-grid .feature:nth-child(even) > * { direction: ltr; }
.feature-label {
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #0f766e;
  margin-bottom: 0.5rem;
}
.feature h3 { font-size: 1.25rem; margin-bottom: 0.5rem; }
.feature p { color: #71717a; font-size: 0.95rem; }

@media (max-width: 768px) {
  .hero { grid-template-columns: 1fr; }
  .feature-grid { grid-template-columns: 1fr; }
  .feature-grid .feature:nth-child(even) { direction: ltr; }
}
```

**Step 3: Verify Quarto can see the config**

```bash
cd /Users/rexarski/Developer/yasqat
quarto check docs/
# Expected: "No issues found" (or warnings about missing .qmd files — that's OK at this stage)
```

**Step 4: Commit**

```bash
cd /Users/rexarski/Developer/yasqat
git add docs/_quarto.yml docs/styles.scss
git commit -m "docs: bootstrap Quarto site config and taste-informed SCSS design system"
```

---

## Task 2: Landing page (index.qmd)

**Files:**
- Create: `docs/index.qmd`

**Step 1: Create `docs/index.qmd`**

```markdown
---
title: ""
page-layout: full
toc: false
---

::: {.hero}

::: {.hero-text}
# Sequence analysis\nfor social science.

::: {.lead}
A modern Python library for categorical sequence analysis.
Built on **polars** and **plotnine** — no pandas, no matplotlib API.
Designed for life-course researchers.
:::

::: {.cta-group}
<a href="getting-started/installation.html" class="btn-primary-custom">Get Started →</a>
<a href="api/core.html" class="btn-secondary-custom">API Reference</a>
:::
:::

::: {.hero-code}
```python
from yasqat import Alphabet, SequencePool
from yasqat.metrics.optimal_matching import OptimalMatchingMetric
from yasqat.clustering import pam_clustering

# Define labour-market states
alpha = Alphabet(
    states=["Employed", "Unemployed", "Education", "Inactive"],
    colors=["#0f766e", "#dc2626", "#d97706", "#71717a"],
)

# Build a sequence pool from your data
pool = SequencePool.from_dataframe(df, alphabet=alpha)

# Compute pairwise distances and cluster
dm = OptimalMatchingMetric().pairwise(pool)
result = pam_clustering(dm, k=4)
```
:::

:::

---

::: {.feature-grid}

::: {.feature}
::: {.feature-label}
Polars-native
:::
### Fast by design
No pandas dependency. All data operations run through polars — columnar, lazy, and fast even on large panel datasets.

```python
# Returns a polars DataFrame
pool.state_distribution()
```
:::

::: {.feature}
::: {.feature-label}
Metrics
:::
### Seven distance metrics
Optimal Matching, Hamming, LCS, DTW, and more. Every metric outputs a `DistanceMatrix` — compatible with all clustering functions.

```python
dm = HammingMetric().pairwise(pool)
```
:::

::: {.feature}
::: {.feature-label}
TraMineR-compatible
:::
### Built on proven algorithms
Algorithm correctness cross-referenced against TraMineR (R) and TanaT (Python). Familiar to social-science analysts.

```python
turbulence(pool)       # Brzinsky-Fay
discrepancy_analysis() # pseudo-R²
```
:::

::: {.feature}
::: {.feature-label}
Visualization
:::
### Nine plot types, one grammar
Index plots, timelines, modal states, parallel coordinates — all return composable plotnine `ggplot` objects.

```python
p = index_plot(pool)
p + theme_bw()  # fully composable
```
:::

:::
```

> Note: Replace `\n` in the hero h1 with an actual newline in the file.

**Step 2: Preview the landing page**

```bash
cd /Users/rexarski/Developer/yasqat
quarto preview docs/index.qmd
# Open browser — verify hero layout: left text, right code block, 2-col feature zig-zag
# Check on mobile width: should stack to single column
```

**Step 3: Commit**

```bash
git add docs/index.qmd
git commit -m "docs: add taste-informed landing page with left-aligned hero"
```

---

## Task 3: Getting Started pages

**Files:**
- Create: `docs/getting-started/installation.qmd`
- Create: `docs/getting-started/quickstart.qmd`

**Step 1: Create installation page**

```markdown
---
title: "Installation"
sidebar: getting-started
---

## Requirements

- Python ≥ 3.11
- No pandas dependency — yasqat uses polars throughout

## Install from PyPI

```bash
pip install yasqat
```

With [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv add yasqat
```

## Verify installation

```python
import yasqat
print(yasqat.__version__)  # 0.3.0
```

## Development install

```bash
git clone https://github.com/rexarski/yasqat.git
cd yasqat
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/
```

## Dependencies

yasqat pulls in:

| Package | Role |
|---|---|
| polars | DataFrames (never pandas) |
| numpy | Numeric arrays, distance matrices |
| plotnine | ggplot2-grammar visualization |
| numba | JIT-compiled distance inner loops |
| scipy | Hierarchical clustering linkage |
```

**Step 2: Create quickstart page**

```markdown
---
title: "Quickstart"
sidebar: getting-started
---

This guide walks you from raw data to your first sequence plot in under 5 minutes.

## 1. Define an Alphabet

An `Alphabet` defines the valid states in your sequences, plus colours and labels.

```python
from yasqat import Alphabet

alpha = Alphabet(
    states=["Employed", "Unemployed", "Education", "Inactive"],
    colors=["#0f766e", "#dc2626", "#d97706", "#71717a"],
)
```

## 2. Load sequences

```python
import polars as pl
from yasqat import SequencePool

# Your data: one row per person-period, wide format
df = pl.read_csv("sequences.csv")

pool = SequencePool.from_dataframe(
    df,
    alphabet=alpha,
    id_col="person_id",
    state_cols=["t1", "t2", "t3", "t4", "t5"],
)

print(f"{len(pool)} sequences, {pool.length} time points")
```

## 3. Descriptive statistics

```python
from yasqat.statistics import state_distribution, turbulence

# State distribution at each time point
dist = state_distribution(pool)

# Sequence turbulence (Brzinsky-Fay)
turb = turbulence(pool)
print(turb.head())
```

## 4. Compute distances

```python
from yasqat.metrics.optimal_matching import OptimalMatchingMetric

dm = OptimalMatchingMetric().pairwise(pool)
print(dm)  # DistanceMatrix: 100 × 100
```

## 5. Cluster and visualise

```python
from yasqat.clustering import pam_clustering
from yasqat.visualization import index_plot

labels = pam_clustering(dm, k=4).labels

# Colour sequences by cluster
p = index_plot(pool, group=labels)
p.draw()
```

::: {.callout-tip}
All plot functions return a `plotnine.ggplot` object — compose with `+` just like in R's ggplot2.
:::
```

**Step 3: Preview**

```bash
quarto preview docs/getting-started/installation.qmd
# Verify sidebar shows "Get Started" section with both pages
```

**Step 4: Commit**

```bash
git add docs/getting-started/
git commit -m "docs: add installation and quickstart pages"
```

---

## Task 4: Tutorial overview page

**Files:**
- Create: `docs/tutorials/index.qmd`

**Step 1: Create `docs/tutorials/index.qmd`**

```markdown
---
title: "Tutorials"
sidebar: tutorials
toc: false
---

These tutorials follow a single synthetic dataset — 1,000 financial customers
tracked daily across 15 lifecycle states — from raw data through clustering and
visualization. Each tutorial is self-contained; jump to any section.

| Tutorial | What you'll learn |
|---|---|
| [Data Structures](data-structures.qmd) | `Alphabet`, `SequencePool`, `StateSequence`, `Trajectory` |
| [Descriptive Statistics](statistics.qmd) | Entropy, turbulence, normative indicators, transition rates |
| [Distance Computation](distance.qmd) | OM, Hamming, LCS, DTW — when to use each |
| [Clustering](clustering.qmd) | PAM, CLARA, hierarchical; silhouette and pseudo-R² quality |
| [Visualization](visualization.qmd) | Index plots, timelines, modal states, parallel coordinates |

::: {.callout-note}
Tutorials assume yasqat is installed. See [Installation](../getting-started/installation.qmd) if needed.
:::
```

**Step 2: Commit**

```bash
git add docs/tutorials/index.qmd
git commit -m "docs: add tutorials landing page"
```

---

## Task 5: Tutorial — Data Structures

**Files:**
- Create: `docs/tutorials/data-structures.qmd`
- Source reference: `yasqat_demo.py` Section 0, 1, 2

**Step 1: Create the tutorial**

Write `docs/tutorials/data-structures.qmd`. Content should cover:

1. **Alphabet** — constructing one, state validation, color assignment
2. **SequenceConfig** — configuring gaps, missing values
3. **StateSequence** — creating from a list, accessing states
4. **SequencePool** — from DataFrame, key properties (`length`, `n_sequences`, `sequence_ids`)
5. **Trajectory / TrajectoryPool** — time-stamped variant, when to use

Use `eval: false` on all code chunks (no execution at build time). Pull example code
directly from `yasqat_demo.py` Sections 1 and 2.

Starter frontmatter:

```yaml
---
title: "Data Structures"
sidebar: tutorials
execute:
  eval: false
---
```

**Step 2: Preview**

```bash
quarto preview docs/tutorials/data-structures.qmd
```

**Step 3: Commit**

```bash
git add docs/tutorials/data-structures.qmd
git commit -m "docs: add data structures tutorial"
```

---

## Task 6: Tutorial — Descriptive Statistics

**Files:**
- Create: `docs/tutorials/statistics.qmd`
- Source reference: `yasqat_demo.py` Sections 3, 4, 5

**Step 1: Create the tutorial**

Write `docs/tutorials/statistics.qmd`. Content should cover:

1. **Descriptive stats** — `state_distribution`, `turbulence`, `longitudinal_entropy`, `complexity_index`, `spell_count`
2. **Transition analysis** — `transition_rate_matrix`, `first_occurrence_time`, `state_duration_stats`
3. **Normative indicators** — `badness`, `insecurity`, `degradation`, `precarity`, `volatility` — explain what each measures (useful for social-science audience)
4. **Subsequence mining** — `frequent_subsequences`

Include a callout explaining the TraMineR heritage for social-science readers:

```markdown
::: {.callout-note}
These indicators follow TraMineR's naming conventions. If you know `seqST()` or
`seqindic()` in R, these are the equivalent Python functions.
:::
```

Use `eval: false` throughout.

**Step 2: Preview and commit**

```bash
quarto preview docs/tutorials/statistics.qmd
git add docs/tutorials/statistics.qmd
git commit -m "docs: add descriptive statistics tutorial"
```

---

## Task 7: Tutorial — Distance Computation

**Files:**
- Create: `docs/tutorials/distance.qmd`
- Source reference: `yasqat_demo.py` Section 7

**Step 1: Create the tutorial**

Write `docs/tutorials/distance.qmd`. Content should cover:

1. **What is a distance matrix** — `DistanceMatrix` wrapper, `.values` property
2. **Optimal Matching** — substitution cost matrix, indel cost, when to use
3. **Hamming** — equal-length sequences only, fastest
4. **LCS** — longest common subsequence distance
5. **DTW** — dynamic time warping, elastic matching
6. **Choosing a metric** — a short decision table

Decision table:

| Metric | When to use |
|---|---|
| Hamming | Fixed-length, equal time points, fast comparison |
| Optimal Matching | Variable-length, theoretically grounded, social-science standard |
| LCS | You care about common subsequences, not timing |
| DTW | Elastic alignment needed, sequences of different lengths/speeds |

Use `eval: false`.

**Step 2: Preview and commit**

```bash
quarto preview docs/tutorials/distance.qmd
git add docs/tutorials/distance.qmd
git commit -m "docs: add distance computation tutorial"
```

---

## Task 8: Tutorial — Clustering

**Files:**
- Create: `docs/tutorials/clustering.qmd`
- Source reference: `yasqat_demo.py` Sections 8, 9

**Step 1: Create the tutorial**

Write `docs/tutorials/clustering.qmd`. Content should cover:

1. **PAM** (`pam_clustering`) — takes a `DistanceMatrix`, returns `PAMClustering` with `.labels` and `.medoids`
2. **CLARA** (`clara_clustering`) — sampling-based PAM for large datasets (N > 500)
3. **Hierarchical** (`hierarchical_clustering`) — Ward linkage, dendrogram-based
4. **Choosing k** — `pam_range` function, plotting silhouette by k
5. **Quality metrics** — `silhouette_score`, `cluster_quality` (pseudo-R²)
6. **Discrepancy analysis** — `discrepancy_analysis`, `multi_factor_discrepancy`, how to interpret pseudo-R²
7. **Representatives** — `extract_representatives` for finding medoid sequences

Use `eval: false`.

**Step 2: Preview and commit**

```bash
quarto preview docs/tutorials/clustering.qmd
git add docs/tutorials/clustering.qmd
git commit -m "docs: add clustering tutorial"
```

---

## Task 9: Tutorial — Visualization

**Files:**
- Create: `docs/tutorials/visualization.qmd`
- Source reference: `yasqat_demo.py` Sections 10, 11

**Step 1: Create the tutorial**

Write `docs/tutorials/visualization.qmd`. Content covers all 9 built-in plot functions:

For each function, show: function signature, a code example, and a one-sentence description
of what the plot shows and when to use it.

| Function | What it shows |
|---|---|
| `index_plot` | One row per sequence, colored by state at each time point |
| `timeline_plot` | Horizontal Gantt-style bars per spell |
| `spell_duration_plot` | Distribution of spell lengths per state |
| `distribution_plot` | State proportions at each time point |
| `entropy_plot` | Cross-sectional Shannon entropy over time |
| `frequency_plot` | Most frequent sequences as bars |
| `modal_state_plot` | Modal (most common) state at each time point |
| `mean_time_plot` | Mean time spent in each state |
| `parallel_coordinate_plot` | Parallel coordinates for sequence trajectories |

Also cover: composing with plotnine (`p + theme_bw()`), `Alphabet.colors` for consistent palettes.

Use `eval: false`.

**Step 2: Preview and commit**

```bash
quarto preview docs/tutorials/visualization.qmd
git add docs/tutorials/visualization.qmd
git commit -m "docs: add visualization tutorial"
```

---

## Task 10: Demo page and Changelog page

**Files:**
- Create: `docs/demo.qmd`
- Create: `docs/changelog.qmd`

**Step 1: Create `docs/demo.qmd`**

The demo script (`yasqat_demo.py`) is Databricks-formatted — convert it to a readable
Quarto page by copying it as static code display (no execution):

```markdown
---
title: "Full Demo — Customer Lifecycle Analytics"
toc: true
toc-depth: 2
execute:
  eval: false
---

This demo walks through the complete yasqat workflow using a synthetic dataset of
1,000 financial customers tracked across 15 lifecycle states.

Source: [`yasqat_demo.py`](https://github.com/rexarski/yasqat/blob/dev/yasqat_demo.py)
on GitHub (Databricks notebook format — importable directly).

## Section 0: Imports

[paste Section 0 code from yasqat_demo.py]

## Section 1: Synthetic Data Generation

[paste Section 1 code]

...and so on for all 12 sections.
```

**Step 2: Create `docs/changelog.qmd`**

```markdown
---
title: "Changelog"
toc: false
---

```{=html}
<!-- Source: CHANGELOG.md — keep in sync manually or via CI script -->
```

[Copy the full content of CHANGELOG.md here, converting markdown headers]
```

**Step 3: Preview and commit**

```bash
quarto preview docs/demo.qmd
git add docs/demo.qmd docs/changelog.qmd
git commit -m "docs: add demo walkthrough and changelog pages"
```

---

## Task 11: Configure and generate API reference

**Files:**
- Run: `quartodoc build` to auto-generate `docs/api/*.qmd`
- Modify: `docs/_quarto.yml` (if quartodoc config needs adjustment)

**Step 1: Verify quartodoc can find your modules**

```bash
cd /Users/rexarski/Developer/yasqat
uv run quartodoc build --config docs/_quarto.yml --verbose
# Expected: generates docs/api/core.qmd, docs/api/metrics.qmd, etc.
# If a module isn't found, check the module path in the quartodoc sections config
```

**Step 2: If quartodoc errors on a module path**

Check which modules actually exist:

```bash
uv run python -c "import yasqat.metrics; print(dir(yasqat.metrics))"
uv run python -c "from yasqat.metrics import hamming; print(hamming.__file__)"
```

Adjust the `contents:` entries in `_quarto.yml` to match exact importable paths.

**Step 3: Preview the generated API pages**

```bash
quarto preview docs/api/core.qmd
# Verify: function signatures render, docstrings appear, parameter tables show
```

**Step 4: Add generated files to git**

Generated `docs/api/*.qmd` files should be committed so the site renders on GitHub Pages
without needing to run quartodoc in CI (simpler setup):

```bash
git add docs/api/
git commit -m "docs: generate API reference pages via quartodoc"
```

> Alternative: Re-generate in CI on every build. For now, commit generated files for simplicity.
> Revisit if API changes frequently enough to warrant full CI regeneration.

---

## Task 12: GitHub Actions — docs deployment workflow

**Files:**
- Create: `.github/workflows/docs.yml`

**Step 1: Create `.github/workflows/docs.yml`**

```yaml
name: Build and deploy docs

on:
  push:
    branches: [main]
  workflow_dispatch:   # allow manual trigger from GitHub UI

permissions:
  contents: write      # needed to push to gh-pages branch

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Quarto
        uses: quarto-dev/quarto-actions/setup@v2

      - name: Install yasqat (needed for quartodoc imports)
        run: pip install -e . quartodoc

      - name: Render docs
        run: quarto render docs/
        # Note: quartodoc-generated api/ files are committed — no need to re-run quartodoc in CI

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: docs/_site
          publish_branch: gh-pages
          user_name: "github-actions[bot]"
          user_email: "github-actions[bot]@users.noreply.github.com"
```

**Step 2: Validate YAML syntax**

```bash
# Quick YAML syntax check (requires PyYAML)
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/docs.yml'))"
# Expected: no output (no errors)
```

**Step 3: Commit**

```bash
git add .github/workflows/docs.yml
git commit -m "ci: add GitHub Actions workflow to deploy docs to GitHub Pages"
```

---

## Task 13: Add quartodoc as dev dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add quartodoc to dev dependencies**

In `pyproject.toml`, add to the `[project.optional-dependencies]` `dev` group (or wherever
dev deps are listed). If there's no dev group yet, add one:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov",
    "ruff",
    "mypy",
    "quartodoc",   # ← add this line
]
```

**Step 2: Sync**

```bash
uv sync
# Verify quartodoc is installed
uv run quartodoc --version
```

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add quartodoc to dev dependencies"
```

---

## Task 14: Full local smoke test

**Step 1: Full render**

```bash
cd /Users/rexarski/Developer/yasqat
quarto render docs/
# Expected: renders to docs/_site/ with no errors
# Warnings about "freeze" or "unexecuted code chunks" are OK
```

**Step 2: Browse locally**

```bash
quarto preview docs/
# Open http://localhost:4444 (or whatever port Quarto prints)
# Walk through: landing page, quickstart, one tutorial, one API page, changelog
```

**Smoke test checklist:**

- [ ] Landing page: hero is 2-column, code block on right, buttons visible
- [ ] Navbar: dark zinc-900 background, all links work
- [ ] Sidebar: "Get Started", "Tutorials", "API Reference" sections present
- [ ] Fonts: Outfit for headers/body (not Inter or system-ui fallback in dev)
- [ ] Teal accent: links and active sidebar item are `#0f766e`
- [ ] Code blocks: zinc-100 background, JetBrains Mono font
- [ ] Callouts: teal left border, no background fill
- [ ] Copy button: hidden, appears on hover
- [ ] API pages: function signatures render, docstrings show as prose
- [ ] Mobile (resize browser to 375px): hero stacks to single column

**Step 3: Commit final state**

```bash
git add -A
git commit -m "docs: complete initial documentation site implementation"
```

---

## Task 15: Merge and deploy

**Step 1: Merge `doc` branch into `main`**

```bash
git checkout main
git merge doc --no-ff -m "docs: add Quarto documentation site with quartodoc API reference"
git push origin main
```

**Step 2: Enable GitHub Pages (one-time, in GitHub UI)**

1. Go to `https://github.com/rexarski/yasqat/settings/pages`
2. Source: **Deploy from a branch**
3. Branch: **`gh-pages`** / `/ (root)`
4. Save

**Step 3: Watch the Actions run**

1. Go to `https://github.com/rexarski/yasqat/actions`
2. "Build and deploy docs" workflow should start automatically
3. Wait for green check (~2-3 minutes)

**Step 4: Verify live site**

```
https://rexarski.github.io/yasqat/
```

Check the same smoke test checklist from Task 14, but now in production.

**Step 5: Update README with docs link**

In `README.md`, add a Docs badge or link near the top:

```markdown
**[Documentation →](https://rexarski.github.io/yasqat/)**
```

Commit and push to `main`.

---

## Summary

| Task | Deliverable | Commit message pattern |
|---|---|---|
| 1 | `_quarto.yml` + `styles.scss` | `docs: bootstrap Quarto site config...` |
| 2 | `index.qmd` — landing page | `docs: add taste-informed landing page...` |
| 3 | Installation + Quickstart | `docs: add installation and quickstart...` |
| 4 | Tutorials overview | `docs: add tutorials landing page` |
| 5–9 | 5 tutorial pages | `docs: add [topic] tutorial` |
| 10 | Demo + Changelog | `docs: add demo walkthrough and changelog` |
| 11 | `docs/api/*.qmd` (quartodoc) | `docs: generate API reference via quartodoc` |
| 12 | `.github/workflows/docs.yml` | `ci: add GitHub Actions docs workflow` |
| 13 | `pyproject.toml` + `uv.lock` | `chore: add quartodoc to dev dependencies` |
| 14 | Local smoke test | `docs: complete initial documentation site` |
| 15 | Merge → deploy → live | — |

**Execution approach:** This plan is designed for manual execution — run tasks in order.
Each commit leaves the `doc` branch in a deployable state. You can stop after any task
and resume later without losing progress.
