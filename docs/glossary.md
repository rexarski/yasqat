# Glossary

## Plain-English definitions

The eight terms you need to read the rest of this documentation.

Sequence
: An ordered run of categorical states for one unit (a person, an account)
  over time. One row of a sequence index plot.

Alphabet
: The finite set of possible states, with a fixed integer encoding. Every
  sequence in a population shares one alphabet.

Spell
: A maximal run of one repeated state within a sequence, for example four
  consecutive months `employed`. Spells carry durations.

STS, SPS, DSS
: The three standard sequence representations: state per time point (what
  you load), state-permanence (one row per spell with its duration), and
  distinct successive states (repeats collapsed).

Optimal matching (OM)
: An edit distance between sequences using insertion and deletion (indel) and
  substitution costs. The field-standard dissimilarity.

Substitution cost
: The price of swapping one state for another when aligning two sequences.
  Constant by default, or derived from transition rates or state features.

Medoid
: The most central actual sequence in a cluster. PAM clusters around medoids,
  not synthetic centroids, so every cluster has a real exemplar.

Discrepancy analysis
: An ANOVA-like test of how much a grouping or covariate explains the
  variation in a distance matrix, assessed by permutation.

## Full vocabulary

The complete domain vocabulary, including the container and seam names used
throughout the codebase, is maintained in the repository's `CONTEXT.md` and
rendered here directly.

```{include} ../CONTEXT.md
:start-line: 6
```
