# Why yasqat

The methods of categorical sequence analysis are settled. Their availability
is not. This page states the problem yasqat solves, the positions it takes,
and the things it deliberately does not do, so that you can decide whether
it belongs in your stack.

## The problem

The canonical toolkit, TraMineR, is mature, trusted, and R-bound. A growing
share of applied data work happens in Python on columnar, Arrow-era stacks.
Teams that want sequence analysis inside such a pipeline have had two options:
bridge across the language boundary, which puts an R runtime in every
environment and copies data on every call, or reimplement the pieces they
need, which is slow, unverified, and invisible as a maintenance cost.

There has been no native, fast, DataFrame-first sequence-analysis library for
Python that a practitioner can drop into an existing pipeline without an R
round-trip or a private port. That is the gap.

## The approach

yasqat takes a deliberately narrow set of technical positions.

**polars-native.** polars is the only DataFrame library in core code; pandas
never appears there, and pyarrow serves only as an interop bridge. Input is a
long-format frame, output is a frame, and nothing is pivoted to wide.

**Compiled inner loops.** The pairwise distance kernels are numba-compiled
over integer-encoded arrays and release the GIL. scipy supplies hierarchical
linkage. Everything else is plain Python over polars.

**DataFrame in, DataFrame out.** Every public method returns a polars
`DataFrame` or a small result object that converts to one. There is no
plotting layer, so yasqat never competes with, or locks you out of, the
plotting library you already use.

**Fidelity over novelty.** TraMineR is the oracle for correctness, naming,
and scope. The test suite pins expected values against it rather than
asserting that results are positive. Where a method has a TraMineR name, the
docstring says which.

## Deep modules, narrow seams

The codebase favours a lot of capability reached through a small, obvious
interface. Four seams carry most of the surface: one dispatch for every
distance, one place where states become integers, one coercion that lets
statistics accept either container, and one reducer that every per-sequence
indicator runs through. The [pipeline page](concepts/pipeline.md) describes
them. The practical consequence is that adding a metric is a function and a
dictionary entry, and adding a per-sequence statistic is a scalar function.

The v0.5.0 release was a subtraction pass: a dead metric-class layer, an
unused encoding method, thirty duplicated type-coercion blocks, and seventeen
hand-written per-sequence loops were removed, each replaced by one of the
seams above. Less surface to learn, and less to break.

## What it is not

**Not a distributed system.** yasqat runs on one machine. Distance matrices
are O(n²) in memory, and the pattern miner is a level-wise search in pure
Python. It is built for cohort scale: thousands to a few hundred thousand
sequences, alphabets of a couple of dozen states, patterns up to length five.
CLARA and `pool.sample` are the tools for pushing that boundary; a cluster is
not.

**Not a replacement for Spark MLlib's pattern mining.** FP-Growth mines
unordered itemsets, which discards the order that is the whole signal here.
PrefixSpan mines ordered patterns at cluster scale but returns patterns and
counts only, with no confidence, lift, or rules, and nothing around the miner:
no situations, durations, distances between whole sequences, or clustering.
If your data are transactions at warehouse scale, use MLlib. If your data are
trajectories and the question is "given this situation, how much more
likely", the miner is five percent of the answer and yasqat provides the
other ninety-five. Should a cohort ever outgrow one machine, PrefixSpan can
generate candidate patterns and yasqat can score and condition them.

**Not a plotting library.** By design (recorded as ADR-0001). The alphabet
carries a colour map so that your own figures stay consistent; that is the
extent of it.

## Engineering discipline

ruff (lint and format, 88 columns, Python 3.11 target) and mypy in strict mode
run on every change. Tests pin exact values. Releases publish to PyPI from CI
on a version tag. Architectural decisions are recorded as ADRs and the issue
tracker lives in the repository as markdown, so the project's reasoning
travels with the code.

## Status and roadmap

yasqat is pre-1.0 and follows semantic versioning accordingly: breaking
changes are possible, land in minor bumps, and are documented in the
[changelog](changelog.md) with migration notes.

The container question is settled: `SequencePool` is the canonical analysis
container and `StateSequence` the representation view (ADR-0002). Remaining
pre-1.0 work is narrower: routing discrepancy analysis through
`DistanceMatrix` everywhere, a preparation seam inside the distance engine,
settling where cluster-quality indices live, and growing the miner with gap
and window constraints, closed and maximal patterns, and discriminant
subsequences.

If you read one thing: yasqat brings TraMineR-class sequence analysis to
fast, DataFrame-native Python. It is usable today, on the way to a stable 1.0.
