# Sequential implementation tasks

[Specification index](../README.md) · [Open review items](../review-status.md)

The tasks are performed in this order. A task is closed from code inspection
and its focused current-contract tests, never from legacy tests alone.

- [Task 0: freeze the baseline and pass the backend feasibility gate](00-baseline.md)
- [Task 1: freeze and prove the executable ABI and state contracts](01-abi.md)
- [Task 2: build the bounded publication and lifetime foundation](02-lifetime-foundation.md)
- [Task 3: cut synchronous LCQ over as one vertical slice](03-lcq-cutover.md)
- [Task 4: complete native chaining and safe cutover](04-native-linking.md)
- [Task 5: build deterministic multi-entry HCQ with exclusive ownership](05-hcq-regions.md)
- [Task 6: enable functional sampling and bounded background admission](06-background-admission.md)
- [Task 7: add family leases, boundary sampling and versioned reshape](07-reshape.md)
- [Task 8: close lifecycle, fault and pressure races](08-lifecycle-stress.md)
- [Task 9: remove the superseded architecture](09-legacy-removal.md)
- [Task 10: prove cross-target conformance](10-cross-target-conformance.md)

## Task execution contract

Each task depends on acceptance of its predecessor; Task 0 first establishes
the baseline and feasibility gates. A task file is an entry point into the
shared specification, not a self-contained replacement for its required
contracts. The review remains open: check [review status](../review-status.md)
before beginning or closing a task. Pending review work is distinct from a
failed implementation test or an unavailable external host.

Before editing, inspect the relevant current implementation, list the named
contracts and acceptance cases affected, and identify the obsolete path to
remove in that vertical slice. Implement the task through its current-contract
checks, including the required cleanup and failure paths. Keep test-only
oracles and diagnostics out of the production hot path as CONTRIBUTING.md
requires. Do not add an alternative decoder, ABI or runtime ownership layer
solely to make an intermediate task easier to test.

A task may be divided into small implementation changes inside its file, but
its predecessor, final behavior and acceptance criteria stay unchanged. A
missing proof is not an implicit waiver; report the exact unmet gate. Record
new design decisions in the owning contract, not only in a session transcript.

## Completion and handoff

Append one concise evidence record to the task file when implementation work
starts or is handed off. Update that record rather than accumulating repeated
session logs. Use these fields:

- Status: pending, in progress, blocked or accepted.
- Code revision: exact tested commit, or commit plus an explicit dirty-worktree
  description when work is not committed; never invent a commit.
- Implemented: paths/symbols and the acceptance requirements they satisfy.
- Verification: exact commands, host ISA/features, outcomes and artifact paths;
  distinguish checks run from checks still required.
- Removal: superseded routes/adapters/tests removed, or the exact later task
  that already authorizes their temporary lifetime.
- Open items: stable review IDs, failed gates, external prerequisites and the
  next concrete action. Use “none” only when none remain.

Accept only after every exit criterion has evidence. Cross-compilation does
not substitute for native x86-64/AArch64 execution. External publication still
requires the authority specified by Task 0; document that dependency without
claiming a local unpublished fork has passed clean-CI acceptance.
