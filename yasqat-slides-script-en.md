# yasqat v0.5.0 — Speaker Script

Target length: about 8 minutes at a relaxed pace.
Cues: **[next]** = advance one slide. *(pause)* = take a short breath, let people read.
Words in `code font` — just read them as normal words.

---

## Slide 1 · Cover — about 40 seconds

Hi everyone. Today I want to introduce a small open-source library I maintain.
It is called yasqat. The name stands for "Yet Another Sequence Analytics
Toolkit". Version 0.5.0 went out on PyPI this month.

In the next eight minutes I will cover three things. First, what the library
does. Second, how it compares to two existing tools, TraMineR and TanaT.
Third, how it fits into a Spark data platform. Let's start.

**[next]**

## Slide 2 · The Data Shape — about 60 seconds

Everything in this library works on one simple data shape: an ID, a time, and
a state. One row per observation. *(pause)*

A sequence is just one unit's states, in time order. For example: a member's
account status, month by month. A patient's care stages. An employee's roles
over the years.

With that data, the library answers three questions. One — how similar are two
sequences? Two — what types of sequences exist in a population? Three — how do
states change over time?

And that is the whole scope. It does not do event processing, and it does not
do plotting. Those stay in your existing tools.

**[next]**

## Slide 3 · The Landscape — about 75 seconds

So why build this? Let me place it next to the two closest tools. *(pause)*

On the left: TraMineR. It is an R package, around since 2008, and it is the
reference implementation in this field. It has the full set of methods, and it
comes with plotting. We treat it as our source of truth — the expected values
in our tests are pinned to TraMineR's output.

In the middle: TanaT, a Python library from Inria in France. It focuses on
patient care pathways, and it has a wider data model — events, intervals, and
states — plus built-in charts. We use it as a reference for Python API design.

And on the right: yasqat. It is deliberately narrower. Categorical state
sequences only. It is built on polars, it takes long-format data directly, and
it has no plotting layer. The goal is to sit inside an existing data platform,
not next to it.

**[next]**

## Slide 4 · Four Differences — about 70 seconds

Here are the four practical differences. *(pause)*

First, "long". The input is one row per ID, time, and state — the shape your
event tables already have. In TraMineR you first build a wide matrix. Here,
the wide format is something you can derive, not something you must prepare.

Second, "compiled". The core runs on polars from start to finish, and the
heavy distance calculations are compiled with numba. There is no pandas in the
core.

Third, "pinned". Our test values are TraMineR's numbers. When we are not sure,
we match R.

Fourth, "tables". Every result comes back as a polars DataFrame — cluster
labels, indicators, rules. You can join them straight back onto your data.

And honestly, there is a fifth difference: scope. We only do state sequences.
Events and intervals stay in TanaT's territory.

**[next]**

## Slide 5 · Spark Interop — about 60 seconds

Now the part I want to dig into: how this connects to a Spark platform.
*(pause)*

The path has five steps. You start with an event table in Spark — user ID,
timestamp, state. It is already in long format, so there is nothing to
reshape.

You export it as parquet, or hand it over in memory through Arrow. Polars
reads it — object storage and Hive partitions are supported.

Then one function call, `load_dataframe`, turns it into the analysis
container. A small config maps your column names, and the alphabet — the list
of valid states — is checked at load time.

At the end, the results are written back as parquet, so labels and indicators
return to the lake as normal tables.

The key point: there is no JVM in this loop. The exchange formats are parquet
and Arrow, which both sides already speak.

**[next]**

## Slide 6 · In Code — about 55 seconds

Here is the same round trip as real code. *(pause — let people scan it)*

On the Spark side, one line: write the event table to parquet.

On the Python side: read the parquet, load it with a config that maps the
column names, compute the distances, run the clustering, and convert the
result to a DataFrame. Then write that DataFrame back to the lake.

One detail I like: if you pass an explicit alphabet, any unexpected state
fails at load time, with a clear error. It does not become a silent extra
category in your results.

**[next]**

## Slide 7 · After the Clusters — about 70 seconds

Clustering is not the end of the story. A typology is only useful if you can
defend it and reuse it. Four things help here. *(pause)*

Choose: `pam_range` tries a range of cluster counts and reports quality scores
for each. So the number of clusters is chosen with evidence, not assumed.

Test: `discrepancy_analysis` runs a permutation test. It answers a question
your stakeholders will ask: does tenure, or region, or cohort, actually
explain the differences between trajectories?

Operate: the PAM model has a `predict` method. You fit the typology once, and
later you assign new sequences to the nearest cluster. So the segmentation
becomes a scoring step, not a one-time study.

And rehearse: the library ships a synthetic data generator for financial user
journeys — onboarding, product adoption, risk events. You can build and demo
the whole pipeline before you touch any production data.

**[next]**

## Slide 8 · Release — about 40 seconds

A quick word on version 0.5.0 itself. Most of the work was subtraction, not
addition. *(pause)*

We deleted an unused class layer, unified everything on one protocol, and
collapsed seventeen similar statistics onto one shared skeleton. Less surface
to learn, less surface to break.

On top of that, three new instruments landed: sequential association rules,
per-sequence state distributions, and an objective volatility measure. Each
one is checked against its TraMineR reference. Docs are live on GitHub Pages.

**[next]**

## Slide 9 · Scope — about 45 seconds

Let me be clear about what this library does not do. *(pause)*

No plotting — that is a design choice. Results are tables; you bring
matplotlib, or your BI layer. No event or interval calculus — categorical
states only. And it is pre-1.0, so breaking changes can still happen in minor
versions. They are documented in the changelog.

The backlog is public, in the repository. Version 1.0 will mean a committed,
stable interface.

**[next]**

## Slide 10 · Closing — about 30 seconds

That is yasqat. To try it: pip install yasqat. Python 3.11 or newer, MIT
license, docs at rexarski dot github dot io slash yasqat.

Three things to remember. The methods are checked against TraMineR. Data goes
in and out as polars DataFrames. And it speaks parquet, so it plugs into a
Spark platform without any bridge code.

Thank you. Happy to take questions.

---

## Backup answers — if someone asks

**"How fast is it?"** — The distance kernels are numba-compiled, and the cost
grows with the square of the number of sequences. I have not published
benchmark numbers yet, and I would rather show a measured result than a claim.
(Note to self: do not mention `n_jobs` — the parallel path is currently slower
than sequential; fix is tracked as issue 17.)

**"Why not just use TraMineR through rpy2?"** — You can. But then you carry an
R runtime in your image, convert data at the boundary, and debug across two
languages. This library removes that layer for the common cases.

**"Can it handle our data volume?"** — Distances are pairwise, so very large
populations need sampling or a pre-aggregation step. That is the honest
current answer; scaling work is on the public backlog.

**"Is it production-ready?"** — It is pre-1.0. The API can still move between
minor versions, and changes are documented. Correctness is tested against the
R reference on every release.
