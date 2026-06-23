# Triage Labels

The skills speak in terms of five canonical triage roles. Because yasqat tracks
issues as **local markdown** (see `issue-tracker.md`), these roles are recorded
as the `Status:` line string near the top of each `.scratch/<feature>/issues/NN-*.md`
file — not as GitHub labels.

| Canonical role     | `Status:` string in our files | Meaning                                  |
| ------------------ | ----------------------------- | ---------------------------------------- |
| `needs-triage`     | `needs-triage`                | Maintainer needs to evaluate this issue  |
| `needs-info`       | `needs-info`                  | Waiting on reporter for more information |
| `ready-for-agent`  | `ready-for-agent`             | Fully specified, ready for an AFK agent  |
| `ready-for-human`  | `ready-for-human`             | Requires human implementation            |
| `wontfix`          | `wontfix`                     | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), write the
corresponding string into the issue file's `Status:` line.

Edit the right-hand column to match whatever vocabulary you actually use.
