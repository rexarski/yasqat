# Gap Report: yasqat vs TanaT + TraMineR
### Date: 2026-02-12

## Distance Metrics

| Feature | TraMineR | TanaT | yasqat | Status | Description |
|---------|----------|-------|--------|--------|-------------|
| Optimal Matching (OM) | `OM` | `edit` | `optimal_matching` | Implemented | Classic edit distance with configurable substitution cost matrix and indel costs. The standard metric for categorical sequence comparison (see TraMineR [`seqdist`](http://traminer.unige.ch/doc/seqdist.html)). |
| Hamming | `HAM` | `hamming` | `hamming_distance` | Implemented | Position-wise mismatch count for equal-length sequences. Only counts state differences at each position — no insertions or deletions allowed (see TraMineR [`HAM`](http://traminer.unige.ch/doc/seqdist.html)). |
| LCS | `LCS` | `lcs` | `lcs_distance` | Implemented | Distance based on the Longest Common Subsequence, measuring how much two sequences share in common order. Suitable for variable-length sequences where ordering matters more than timing (see TraMineR [`LCS`](http://traminer.unige.ch/doc/seqdist.html)). |
| LCP | `LCP` | `lcp` | `lcp_distance` | Implemented | Longest Common Prefix distance — measures shared initial segment between two sequences. Useful when early states are most important (see TraMineR [`LCP`](http://traminer.unige.ch/doc/seqdist.html)). |
| RLCP | `RLCP` | — | `rlcp_distance` | Implemented | Reversed Longest Common Prefix — measures shared suffix (ending segment) between sequences. Complement of LCP for cases where final states matter most (see TraMineR [`RLCP`](http://traminer.unige.ch/doc/seqdist.html)). |
| Chi-squared | `CHI2` | `chi2` | `chi2_distance` | Implemented | Chi-squared distance on within-sequence state proportion vectors. Compares the distribution of time spent in each state across two sequences (see TraMineR [`CHI2`](http://traminer.unige.ch/doc/seqdist.html)). |
| Euclidean | `EUCLID` | — | `euclidean_distance` | Implemented | L2 (Euclidean) distance on state proportion vectors. Similar to Chi-squared but without frequency weighting (see TraMineR [`EUCLID`](http://traminer.unige.ch/doc/seqdist.html)). |
| DTW | — | `dtw` | `dtw_distance` | Implemented | Dynamic Time Warping finds optimal alignment between sequences with different timing. Allows elastic matching by warping the time axis (see [Sakoe & Chiba 1978](https://doi.org/10.1109/TASSP.1978.1163055)). |
| Soft DTW | — | `softdtw` | `softdtw_distance` | Implemented | Differentiable approximation of DTW using soft-minimum operator. Enables gradient-based optimization and ML-compatible distance (see [Cuturi & Blondel 2017](https://arxiv.org/abs/1703.01541)). |
| DHD | `DHD` | — | `dhd_distance` | Implemented | Dynamic Hamming Distance with position-dependent substitution costs derived from cross-sectional state frequencies. Accounts for how common state differences are at each time point (see TraMineR [`DHD`](http://traminer.unige.ch/doc/seqdist.html)). |
| TWED | `TWED` | — | `twed_distance` | Implemented | Time Warp Edit Distance combines edit distance with temporal elasticity via stiffness (nu) and deletion penalty (lambda) parameters. Metric distance that penalizes time-shifting (see [Marteau 2009](https://doi.org/10.1109/TPAMI.2008.76)). |
| OMloc | `OMloc` | — | `omloc_distance` | Implemented | Localized Optimal Matching with position-dependent substitution costs. States that are rare at a given position incur higher substitution costs (see TraMineR [`OMloc`](http://traminer.unige.ch/doc/seqdist.html)). |
| OMspell | `OMspell` | — | `omspell_distance` | Implemented | Spell-length sensitive Optimal Matching that weights substitution costs by the length of the spell (run) being modified. Longer spells of the same state cost more to change (see [Studer & Ritschard 2016](https://doi.org/10.1177/0049124115577211)). |
| OMstran | `OMstran` | — | `omstran_distance` | Implemented | Transition-sensitive Optimal Matching that incorporates transition pattern costs alongside standard OM. Weighs both state substitutions and transition structure differences (see [Studer & Ritschard 2016](https://doi.org/10.1177/0049124115577211)). |
| NMS | `NMS` | — | `nms_distance` | Implemented | Number of Matching Subsequences — distance based on counting common subsequences between two sequences. More fine-grained than LCS as it considers all shared subsequences, not just the longest (see TraMineR [`NMS`](http://traminer.unige.ch/doc/seqdist.html)). |
| NMSMST | `NMSMST` | — | `nmsmst_distance` | Implemented | NMS normalized by minimum shared time using geometric mean. Accounts for sequence length differences when comparing matching subsequence counts (see TraMineR [`NMSMST`](http://traminer.unige.ch/doc/seqdist.html)). |
| SVRspell | `SVRspell` | — | `svrspell_distance` | Implemented | Subsequence Vectorial Representation based on spell structure. Converts sequences to spell vectors (state + duration) and computes distance in this representation (see TraMineR [`SVRspell`](http://traminer.unige.ch/doc/seqdist.html)). |
| OMslen | `OMslen` | — | — | Missing | Spell-length-sensitive OM variant distinct from OMspell. Weights transitions by total spell length rather than per-transition weighting. |
| Linear Pairwise | — | `linearpairwise` | — | Missing | TanaT-specific position-by-position comparison with aggregation (mean/sum) and padding penalty for unequal lengths. |

## Substitution Cost Methods

| Method | TraMineR | yasqat | Status | Description |
|--------|----------|--------|--------|-------------|
| CONSTANT | `CONSTANT` | `"constant"` | Implemented | Uniform substitution cost (default: 2) for all state pairs. The simplest cost structure, treating all state differences equally (see TraMineR [`seqcost`](http://traminer.unige.ch/doc/seqcost.html)). |
| TRATE | `TRATE` | `"trate"` | Implemented | Substitution costs derived from transition rates: `cost(i,j) = 2 - p(i→j) - p(j→i)`. States that frequently transition between each other are treated as more similar (see TraMineR [`seqcost`](http://traminer.unige.ch/doc/seqcost.html)). |
| FUTURE | `FUTURE` | `"future"` | Implemented | Chi-squared distance between conditional future state distributions. States with similar likely futures receive lower substitution costs (see TraMineR [`seqcost`](http://traminer.unige.ch/doc/seqcost.html)). |
| INDELS | `INDELS` | `"indels"` | Implemented | Indel-based costs inversely proportional to state frequency: `cost = 1/freq(state)`. Rare states are more expensive to insert or delete (see TraMineR [`seqcost`](http://traminer.unige.ch/doc/seqcost.html)). |
| INDELSLOG | `INDELSLOG` | `"indelslog"` | Implemented | Logarithmic indel-based costs: `cost = log(1/freq(state))`. Smoother version of INDELS that reduces the impact of very rare states (see TraMineR [`seqcost`](http://traminer.unige.ch/doc/seqcost.html)). |
| FEATURES | `FEATURES` | `"features"` | Implemented | Gower distance on user-defined state feature vectors. Users supply a numeric feature matrix describing each state, and costs are computed as mean absolute differences normalized by range (see TraMineR [`seqcost`](http://traminer.unige.ch/doc/seqcost.html)). |

## Statistical Indicators

| Indicator | TraMineR | yasqat | Status | Description |
|-----------|----------|--------|--------|-------------|
| Sequence length | `seqlength` | `sequence_length` | Implemented | Number of time positions in each sequence. Basic descriptor for variable-length sequence data (see TraMineR [`seqlength`](http://traminer.unige.ch/doc/seqlength.html)). |
| Transition count | `seqtransn` | `transition_count` | Implemented | Number of state changes (transitions) within a sequence. Higher values indicate more volatile trajectories (see TraMineR [`seqtransn`](http://traminer.unige.ch/doc/seqtransn.html)). |
| Transition proportion | `transp` | `transition_proportion` | Implemented | Ratio of transitions to maximum possible transitions: `n_transitions / (length - 1)`. Normalized version of transition count for comparing sequences of different lengths. |
| Spell count | via DSS | `spell_count` | Implemented | Number of spells (runs of consecutive identical states) in a sequence. Equivalent to the length of the DSS (Distinct Successive States) representation (see TraMineR via `seqdss`). |
| Visited states count | `visited` | `visited_states` | Implemented | Number of distinct states appearing in a sequence. Measures the diversity of states experienced (see TraMineR [`seqindic`](http://traminer.unige.ch/doc/seqindic.html)). |
| Visited proportion | `visitp` | `visited_proportion` | Implemented | Ratio of visited states to total alphabet size: `visited / |alphabet|`. Normalizes visited states count to a [0,1] range (see TraMineR [`seqindic`](http://traminer.unige.ch/doc/seqindic.html)). |
| Longitudinal entropy | `seqient` | `longitudinal_entropy` | Implemented | Normalized Shannon entropy of the within-sequence state distribution. Ranges from 0 (all time in one state) to 1 (equal time in all states) (see TraMineR [`seqient`](http://traminer.unige.ch/doc/seqient.html)). |
| Complexity index | `seqici` | `complexity_index` | Implemented | Composite index combining number of transitions and state diversity, normalized by sequence length: `sqrt(n_transitions × n_distinct) / length`. Captures both diversity and instability (see TraMineR [`seqici`](http://traminer.unige.ch/doc/seqici.html)). |
| Turbulence | `seqST` | `turbulence` | Implemented | Based on the number of distinct subsequences and the variance of spell durations. Higher turbulence indicates more complex, unpredictable sequences (see [Elzinga & Liefbroer 2007](https://doi.org/10.1111/j.1467-985X.2007.00478.x)). |
| Normalized turbulence | `turbn` | `normalized_turbulence` | Implemented | Turbulence divided by its theoretical maximum `log2(length)`. Rescales turbulence to [0,1] for cross-length comparison (see TraMineR [`seqindic`](http://traminer.unige.ch/doc/seqindic.html)). |
| Subsequence count | `seqsubsn` | `subsequence_count` | Implemented | Number of distinct subsequences derivable from the DSS representation. Core component of the turbulence calculation (see TraMineR [`seqsubsn`](http://traminer.unige.ch/doc/seqsubsn.html)). |
| Mean time in state | `seqmeant` | `mean_time_in_state` | Implemented | Average duration spent in each state across all sequences. Returns a cross-tabulation of states and their mean durations (see TraMineR [`seqmeant`](http://traminer.unige.ch/doc/seqmeant.html)). |
| Cross-sectional distribution | `seqstatd` | `state_distribution` | Implemented | Proportion of sequences in each state at each time position. Produces the data behind distribution (chronogram) plots (see TraMineR [`seqstatd`](http://traminer.unige.ch/doc/seqstatd.html)). |
| Modal states | `seqmodst` | `modal_states` | Implemented | Most frequent state at each time position, with frequency and proportion. Shows the "typical" state trajectory across time (see TraMineR [`seqmodst`](http://traminer.unige.ch/doc/seqmodst.html)). |
| Sequence frequency table | `seqtab` | `sequence_frequency_table` | Implemented | Frequency table of complete sequence patterns showing count and proportion. Identifies the most common full trajectories in the data (see TraMineR [`seqtab`](http://traminer.unige.ch/doc/seqtab.html)). |
| Transition rates | `seqtrate` | `transition_rate_matrix` | Implemented | Transition probability matrix estimated from observed state-to-state transitions. Foundation for Markov chain analysis and substitution cost calculation (see TraMineR [`seqtrate`](http://traminer.unige.ch/doc/seqtrate.html)). |
| Sequence log-probabilities | `seqlogp` | `sequence_log_probability` | Implemented | Log-probability of each sequence given the empirical transition model. Measures how "typical" a sequence is relative to the observed transition structure (see TraMineR [`seqlogp`](http://traminer.unige.ch/doc/seqlogp.html)). |
| State duration stats | — | `state_duration_stats` | Implemented | Summary statistics (mean, median, std, min, max) of spell durations per state. Extends mean time to provide full distributional information. |
| First occurrence time | `seqfpos` | `first_occurrence_time` | Implemented | Time position of the first occurrence of each state in each sequence. Useful for studying when states are first entered (see TraMineR [`seqfpos`](http://traminer.unige.ch/doc/seqfpos.html)). |
| Substitution cost matrix | `seqcost` | `substitution_cost_matrix` | Implemented | Constructs substitution cost matrices using various methods (CONSTANT, TRATE, FUTURE, INDELS, INDELSLOG, FEATURES). Wrapper around `build_substitution_matrix` for end-user access. |

### Normative Indicators

| Indicator | TraMineR | yasqat | Status | Description |
|-----------|----------|--------|--------|-------------|
| Volatility | `seqivolatility` | `volatility` | Implemented | Frequency of sign changes between positive and negative states, normalized by `length - 1`. Measures how often trajectories switch between favorable and unfavorable conditions (see TraMineR [`seqivolatility`](http://traminer.unige.ch/doc/seqivolatility.html)). |
| Precarity | `seqprecarity` | `precarity` | Implemented | Recency-weighted proportion of time in negative states — later positions count more. Captures worsening conditions toward the end of observation (see TraMineR [`seqprecarity`](http://traminer.unige.ch/doc/seqprecarity.html)). |
| Insecurity | `seqinsecurity` | `insecurity` | Implemented | Expected proportion of future time in negative states, averaged across all positions. Measures forward-looking exposure to unfavorable states (see TraMineR [`seqinsecurity`](http://traminer.unige.ch/doc/seqinsecurity.html)). |
| Degradation | `seqidegrad` | `degradation` | Implemented | Rate of transitions from positive to negative states, normalized by `length - 1`. Captures the tendency of trajectories to deteriorate (see TraMineR [`seqidegrad`](http://traminer.unige.ch/doc/seqidegrad.html)). |
| Badness | `seqibad` | `badness` | Implemented | Simple proportion of time spent in negative states. The most basic quality-of-trajectory indicator (see TraMineR [`seqibad`](http://traminer.unige.ch/doc/seqibad.html)). |
| Integration | `seqintegr` | `integration` | Implemented | Cumulative proportion of time in positive states, measuring speed and permanence of entering favorable states. Higher values indicate earlier and more sustained positive outcomes (see TraMineR [`seqintegr`](http://traminer.unige.ch/doc/seqintegr.html)). |
| Proportion positive | `seqipos` | `proportion_positive` | Implemented | Overall proportion of time spent in positive states. Complement of badness for user-defined positive state sets (see TraMineR [`seqipos`](http://traminer.unige.ch/doc/seqipos.html)). |
| Normative volatility | `nvolat` | — | Missing | Volatility computed on positive/negative classification rather than raw states. A special case of volatility using a binary recoding. |

## Visualization

| Visualization | TraMineR | TanaT | yasqat | Status | Description |
|---------------|----------|-------|--------|--------|-------------|
| Distribution plot | `seqdplot` | `distribution` | `distribution_plot` | Implemented | Stacked area chart showing cross-sectional state proportions at each time position (chronogram). The most common sequence visualization (see TraMineR [`seqdplot`](http://traminer.unige.ch/doc/seqplot.html)). |
| Index plot | `seqIplot` | — | `index_plot` | Implemented | Horizontal colored bars representing each individual sequence, stacked vertically. Shows all sequences in the dataset at once (see TraMineR [`seqIplot`](http://traminer.unige.ch/doc/seqplot.html)). |
| Entropy plot | `seqHtplot` | — | `entropy_plot` | Implemented | Transversal entropy (diversity) plotted over time positions. Shows when sequences are most diverse or converge to common states (see TraMineR [`seqHtplot`](http://traminer.unige.ch/doc/seqplot.html)). |
| Frequency plot | `seqfplot` | — | `frequency_plot` | Implemented | Displays the N most frequent complete sequence patterns as colored bars with counts. Highlights dominant trajectory types in the data (see TraMineR [`seqfplot`](http://traminer.unige.ch/doc/seqplot.html)). |
| Timeline | — | `timeline` | `timeline_plot` | Implemented | Horizontal timeline showing spell durations per sequence. TanaT-style visualization with relative/absolute time support. |
| Spell duration plot | — | — | `spell_duration_plot` | Implemented | Bar chart of mean spell durations per state. Shows how long sequences typically remain in each state before transitioning. |
| Modal state plot | `seqmsplot` | — | `modal_state_plot` | Implemented | Bar chart showing the most frequent (modal) state at each time position. Visualizes the "typical" trajectory across the observation window (see TraMineR [`seqmsplot`](http://traminer.unige.ch/doc/seqplot.html)). |
| Mean time plot | `seqmtplot` | — | `mean_time_plot` | Implemented | Bar chart of mean time spent in each state across all sequences. Shows the average allocation of time across states (see TraMineR [`seqmtplot`](http://traminer.unige.ch/doc/seqplot.html)). |
| Parallel coordinate plot | `seqpcplot` | — | `parallel_coordinate_plot` | Implemented | Lines connecting states across time positions, with optional subsampling. Reveals common transition patterns and trajectory flows (see TraMineR [`seqpcplot`](http://traminer.unige.ch/doc/seqpcplot.html)). |
| Distribution + entropy | `seqdHplot` | — | — | Missing | Combined distribution plot with entropy curve overlay. Could be composed from existing `distribution_plot` + `entropy_plot`. |
| Representative plot | `seqrplot` | — | — | Missing | Visualization of representative sequences extracted by centrality/frequency/density. Would use `extract_representatives` data with index plot styling. |
| Relative frequency plot | `seqrfplot` | — | — | Missing | Equal-sized groups with medoid sequences shown. Requires relative frequency grouping (not yet implemented). |
| Histogram | — | `histogram` | — | Missing | TanaT-style bar chart of occurrence counts/frequencies with sorting options. Partially covered by `frequency_plot`. |
| Multidomain plot | `seqplotMD` | — | — | Missing | Stacked multi-sequence-domain visualization. Requires multidomain sequence support. |
| Discrepancy position plot | `plot.seqdiff` | — | — | Missing | Position-wise discrepancy between groups over time. Shows where group differences are largest. |
| Subsequence frequency plot | `plot.subseqelist` | — | — | Missing | Visualization of frequent subsequence discovery results from `seqefsub`. |

## Clustering & Advanced Analysis

| Feature | TraMineR/WC | TanaT | yasqat | Status | Description |
|---------|-------------|-------|--------|--------|-------------|
| Hierarchical clustering | `hclust` | `hierarchical` | `HierarchicalClustering` | Implemented | Agglomerative hierarchical clustering from a distance matrix with configurable linkage methods (ward, complete, average, single). Standard approach for sequence typology (see TraMineR companion [WeightedCluster](http://mephisto.unige.ch/weightedcluster/)). |
| PAM (k-medoids) | `wcKMedoids` | `pam` | `PAMClustering` | Implemented | Partitioning Around Medoids — finds k representative sequences minimizing within-cluster distances. More robust to outliers than k-means (see [WeightedCluster `wcKMedoids`](http://mephisto.unige.ch/weightedcluster/)). |
| CLARA | — | `clara` | `clara_clustering` | Implemented | Clustering Large Applications — sampling-based PAM for large datasets. Runs PAM on random subsamples and selects the best partition (see TanaT `clara` clusterer). |
| Silhouette scores | `wcSilhouetteObs` | — | `silhouette_scores` | Implemented | Per-observation silhouette width measuring how well each sequence fits its assigned cluster vs. the next-best cluster. Ranges from -1 (misclassified) to +1 (well-clustered) (see [WeightedCluster](http://mephisto.unige.ch/weightedcluster/)). |
| Cluster quality | `as.clustrange` | — | `cluster_quality` | Implemented | Suite of quality metrics: Average Silhouette Width (ASW), Point Biserial Correlation (PBC), Hubert's Gamma (HG), and R-squared (R2). Enables systematic comparison of clustering solutions (see [WeightedCluster](http://mephisto.unige.ch/weightedcluster/)). |
| PAM range | `wcKMedRange` | — | `pam_range` | Implemented | Runs PAM over a range of k values and reports quality metrics for each. Helps determine the optimal number of clusters (see [WeightedCluster `wcKMedRange`](http://mephisto.unige.ch/weightedcluster/)). |
| Representative sequences | `dissrep` | — | `extract_representatives` | Implemented | Extracts representative sequences by centrality (closest to medoid), frequency (most common pattern), or density (in densest neighborhood). Summarizes cluster content (see TraMineR [`seqrep`](http://traminer.unige.ch/doc/seqrep.html)). |
| Distance to center | `disscenter` | — | `distance_to_center` | Implemented | Mean distance from each observation to its cluster center (medoid). Measures within-cluster dispersion and identifies outliers (see TraMineR [`disscenter`](http://traminer.unige.ch/doc/disscenter.html)). |
| Discrepancy analysis | `dissassoc` | — | `discrepancy_analysis` | Implemented | Pseudo-ANOVA testing whether a grouping variable explains distance matrix structure. Reports pseudo-F statistic, pseudo-R2, and permutation-based p-values (see TraMineR [`dissassoc`](http://traminer.unige.ch/doc/dissassoc.html)). |
| Multi-factor ANOVA | `dissmfacw` | — | `multi_factor_discrepancy` | Implemented | Runs discrepancy analysis independently for each of multiple grouping factors. Enables comparison of multiple covariates' explanatory power over sequence dissimilarities (see TraMineR [`dissmfacw`](http://traminer.unige.ch/doc/dissmfacw.html)). |
| Dissimilarity trees | `disstree` | — | `dissimilarity_tree` | Implemented | Recursive binary partitioning of a distance matrix by covariates, maximizing pseudo-R2 at each split. Produces an interpretable tree of sequence types (see TraMineR [`disstree`](http://traminer.unige.ch/doc/disstree.html)). |
| Frequent subsequence mining | `seqefsub` | — | `frequent_subsequences` | Implemented | Apriori-like level-wise discovery of frequent subsequence patterns with minimum support threshold. Identifies recurring trajectory motifs (see TraMineR [`seqefsub`](http://traminer.unige.ch/doc/seqefsub.html)). |
| Relative frequency groups | `dissrf` | — | — | Missing | Equal-sized partitioning of sequences with medoid extraction per group. Used for relative frequency plots. |
| Discriminant subsequences | `seqecmpgroup` | — | — | Missing | Chi-squared tests to identify subsequences that discriminate between groups. Extension of frequent subsequence mining. |
| Position-wise comparison | `seqdiff` | — | — | Missing | Sliding-window discrepancy analysis between groups across time positions. Identifies when group differences emerge. |
| Domain association | `seqdomassoc` | — | — | Missing | Association analysis between multiple life domains (e.g., work + family). Requires multidomain support. |
| Gower centering | `gower_matrix` | — | — | Missing | Transform a dissimilarity matrix to doubly-centered (Gower) form. Used internally by some advanced analysis methods. |
| Group merging | `dissmergegroups` | — | — | Missing | Iteratively merge cluster groups to minimize quality loss. For simplifying over-partitioned solutions. |
| Multidomain distance | `seqdistmc` | — | — | Missing | Cost Additive Trick for combining distances from multiple sequence domains into a single matrix. |
| Method comparison | `wcCmpCluster` | — | — | Missing | Compare multiple clustering algorithms/parameters on the same distance matrix with standardized quality metrics. |

## Sequence Manipulation

| Feature | TraMineR | TanaT | yasqat | Status | Description |
|---------|----------|-------|--------|--------|-------------|
| STS/SPS/DSS formats | `seqformat` | — | `to_sts/to_sps/to_dss` | Implemented | Convert between State-Time (one state per position), State-Permanence (run-length encoded), and Distinct Successive States (deduplicated) representations. Core sequence transformations (see TraMineR [`seqformat`](http://traminer.unige.ch/doc/seqformat.html)). |
| Type conversions | — | `as_event/state/interval` | `.to_event/state/interval_sequence()` | Implemented | Bidirectional conversion between StateSequence, EventSequence, and IntervalSequence types. Enables working with the same data in different temporal representations (see TanaT type conversion API). |
| State recoding | `seqrecode` | — | `SequencePool.recode_states` | Implemented | Merge or rename states with automatic alphabet rebuild. For simplifying complex state spaces by grouping related states (see TraMineR [`seqrecode`](http://traminer.unige.ch/doc/seqrecode.html)). |
| Pattern matching | `seqpm` | `pattern` | `PatternCriterion` | Implemented | Find sequences containing specified state patterns. Supports exact subsequence matching as a filter criterion (see TraMineR [`seqpm`](http://traminer.unige.ch/doc/seqpm.html)). |
| First occurrence | `seqfpos` | — | `first_occurrence_time` | Implemented | Time position of the first occurrence of each state in each sequence. Useful for studying timing of first entry into states (see TraMineR [`seqfpos`](http://traminer.unige.ch/doc/seqfpos.html)). |
| Random generation | `seqgen` | — | `generate_financial_journeys` | Implemented | Generate synthetic sequence data from transition models. Currently generates realistic financial journey data; TraMineR generates from arbitrary transition matrices (see TraMineR [`seqgen`](http://traminer.unige.ch/doc/seqgen.html)). |
| Length filtering | — | `length` | `LengthCriterion` | Implemented | Filter sequences by minimum/maximum length. Part of yasqat's criteria-based filtering system. |
| Time filtering | — | `time` | `TimeCriterion` | Implemented | Filter sequences by time range. Restricts sequences to a specified temporal window. |
| State filtering | — | — | `ContainsStateCriterion` | Implemented | Filter sequences that contain (or don't contain) specified states. For selecting trajectories passing through certain states. |
| Query filtering | — | `query` | `QueryCriterion` | Implemented | Arbitrary polars expression-based filtering on sequence data. The most flexible filtering mechanism. |
| Missing values detection | `seqhasmiss` | — | — | Missing | Detect and handle missing or void values in sequences. Not yet needed as polars handles nulls natively. |
| Sequence alignment | `seqalign` | — | — | Missing | Detailed pairwise alignment information showing the optimal edit path. Useful for visualization and interpretation of OM distances. |
| SRS format | `seqformat` | — | — | Missing | Shifted Replicated Sequences format conversion. A niche representation rarely used outside TraMineR. |
| Case weights | throughout | — | — | Missing | Native support for observation weights throughout all analyses. Would affect statistics, distances, and clustering. |

## Data I/O

| Feature | TanaT | yasqat | Status | Description |
|---------|-------|--------|--------|-------------|
| CSV | `csv` loader | `load_csv` / `save_csv` | Implemented | Read and write sequence data in CSV format using polars for fast I/O. Supports configurable column names. |
| JSON | `json` loader | `load_json` / `save_json` | Implemented | Read and write sequence data in JSON format. Useful for web integration and API-based workflows. |
| Parquet | — | `load_parquet` / `save_parquet` | Implemented | Read and write sequence data in Apache Parquet columnar format. Best for large datasets and polars-native workflows. |
| Wide format | — | `read_wide_format` / `to_wide_format` | Implemented | Convert between long format (one row per state-time) and wide format (one column per time position). Bridges different data conventions. |
| SQL | `sql` loader | — | Out of scope | Database connectivity for loading sequence data. Excluded as users can use polars' own SQL connectors. |

## Infrastructure (Out of Scope)

| Feature | Source | Status | Reason |
|---------|--------|--------|--------|
| YAML pipeline orchestration | TanaT | Out of scope | Workflow orchestration is outside a pure analytics library. |
| SQL data loaders | TanaT | Out of scope | Users can use polars' own SQL connectors (`pl.read_database`). |
| On-disk distance matrix storage | TanaT | Out of scope | Memory-mapped matrices add infrastructure complexity; numpy/polars handle most sizes. |
| Working environment / registry | TanaT | Out of scope | TanaT's extensibility pattern; yasqat uses direct imports. |
| Survival analysis (Cox, trees) | TanaT | Out of scope | Requires scikit-survival dependency; separate domain from sequence analytics. |
| Built-in datasets (MIMIC, MOOC) | TanaT | Out of scope | Example data is provided via synthetic generation instead. |

---

## Summary

- **Total features compared: ~90**
- **Implemented: 64**
- **Partial: 0**
- **Missing: ~20**
- **Out of scope: 6**

### Progress Since Last Report

13 features moved from Missing to Implemented since the previous report:

| # | Feature | Category |
|---|---------|----------|
| 1 | `twed_distance` (TWED) | Distance Metrics |
| 2 | `omloc_distance` (Localized OM) | Distance Metrics |
| 3 | `omspell_distance` (Spell-based OM) | Distance Metrics |
| 4 | `omstran_distance` (Transition-sensitive OM) | Distance Metrics |
| 5 | `nms_distance` (Number of Matching Subsequences) | Distance Metrics |
| 6 | `nmsmst_distance` (NMS with MST normalization) | Distance Metrics |
| 7 | `svrspell_distance` (Spell-based vectorial) | Distance Metrics |
| 8 | `modal_state_plot` / `mean_time_plot` | Visualization |
| 9 | `parallel_coordinate_plot` | Visualization |
| 10 | `subsequence_count` / `normalized_turbulence` | Statistics |
| 11 | Normative indicators (7 functions) | Statistics |
| 12 | `multi_factor_discrepancy` / `"features"` cost | Advanced Analysis |
| 13 | `dissimilarity_tree` / `frequent_subsequences` | Advanced Analysis |
| 14 | Type conversions (`to_event/state/interval_sequence`) | Sequence Manipulation |

### Module Export Counts

| Module | Exports |
|--------|---------|
| `yasqat.metrics` | 19 |
| `yasqat.statistics` | 33 |
| `yasqat.visualization` | 9 |
| `yasqat.clustering` | 11 |
| `yasqat.core` | 7 |
| `yasqat.filters` | 8 |
| `yasqat.io` | 9 |
| `yasqat.synthetic` | 1 |
| **Total public API** | **97** |

### Test Coverage

- **526 tests** collected across all modules
- All tests passing, ruff lint and format clean

---

## Remaining Missing Features (Prioritized)

### Medium Priority

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 1 | OMslen | TraMineR | Spell-length-sensitive OM variant (different weighting from OMspell) |
| 2 | Linear Pairwise | TanaT | Position-by-position comparison with aggregation |
| 3 | Representative sequence plot | TraMineR `seqrplot` | Visualization using `extract_representatives` data |
| 4 | Distribution + entropy overlay | TraMineR `seqdHplot` | Combined chronogram with entropy curve |
| 5 | Position-wise group comparison | TraMineR `seqdiff` | Sliding discrepancy between groups |
| 6 | Discriminant subsequences | TraMineR `seqecmpgroup` | Chi-squared test for group-specific patterns |
| 7 | Normative volatility | TraMineR `nvolat` | Binary recoding variant of volatility |
| 8 | Method comparison | WeightedCluster `wcCmpCluster` | Compare clustering algorithms on same data |

### Low Priority

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 9 | Relative frequency groups | TraMineR `dissrf` | Equal-size partitioning with medoid extraction |
| 10 | Relative frequency plot | TraMineR `seqrfplot` | Visualization of relative frequency groups |
| 11 | Gower centering | TraMineR `gower_matrix` | Doubly-centered dissimilarity transformation |
| 12 | Group merging | TraMineR `dissmergegroups` | Iterative group simplification |
| 13 | Multidomain distance | TraMineR `seqdistmc` | Cost Additive Trick for multi-domain |
| 14 | Multidomain plot | TraMineR `seqplotMD` | Stacked multi-domain visualization |
| 15 | Missing values detection | TraMineR `seqhasmiss` | Void/missing value handling |
| 16 | Sequence alignment display | TraMineR `seqalign` | Pairwise alignment visualization |
| 17 | Case weights | TraMineR (throughout) | Weighted observations in all analyses |
| 18 | Histogram plot | TanaT `histogram` | Occurrence/frequency bar chart |
| 19 | Subsequence frequency plot | TraMineR `plot.subseqelist` | Visualize mining results |
| 20 | General sequence generator | TraMineR `seqgen` | Generate from arbitrary transition matrices |
