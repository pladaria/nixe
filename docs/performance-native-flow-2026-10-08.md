# Interprocedural native execution and incremental draw bindings

This follow-up implements shared allocation across captured hot calls and their
continuations, separates stable draw preparation from resource rebinding, and
validates the result in the supplied gameplay room. It builds on the register
contracts described in `performance-hcq-2026-10-08.md`.

## CPU execution

Captured BL and observed BLR destinations become native CFG successors in the
same SSA function as their callers. LR is defined architecturally; BLR reads its
destination before writing LR, including BLR X30. Callee return continuations
are inferred from captured call sites and guarded against the actual return
register. Public callee entries and unexpected LR values retain valid external
dispatch. This does not assume that guest code follows a host calling convention.

These edges participate in native liveness, dirty-state propagation, lazy NZCV,
FP ownership, collision trimming and cycle analysis. Ordinary internal edges
perform no architectural writeback or native link adaptation. DFS cycle edges
retain precise budget/control checkpoints, while faults retain their local PRE
prefix and the already charged earlier blocks.

Discovery grows through observed hot calls rather than transitively importing
every demanded callee. Connectivity trimming removes continuations that cannot
be reached through a captured callee. External returns do not grow a shared
callee into arbitrary callers: their existing PIC/link dispatch remains active.
Captured internal returns still execute within the allocation domain. Reshape
observations belong to the actual source instruction, independently of the
region's public entry. Trimming uses these native successors directly; the old
separate dynamic-reachability side channel has been removed.

The warmup policy identity is versioned again. The opt-in compiler trace also
records LCQ input counts; default builds erase this instrumentation.

## Graphics preparation

A bounded `DrawPlan` retains fixed layout/shader validation and its latest
immutable resource-binding snapshot. Prepared draw reuse compares consumed
resource components instead of requiring identity of an entire resolved-resource
composition. The old composition-wide identity and its comparison helpers have
been removed.

Changed buffer views, sampled images and samplers are prepared independently.
Unchanged bindings protect their dependencies while changed views are resolved.
Shader descriptors and access/dependency arrays are rebuilt only when a changed
component is consumed by a shader. Each emitted draw owns its bindings, so later
rebinding cannot change queued operations. Attachments that change still consume
render-pass format and materialization state through normal attachment lowering.
Sampled images retain normal opaque-storage, copy-revision and alias checks.

Retiring a view, sampler or dependent descriptor invalidates every consuming
snapshot while retaining reusable fixed validation. A single retirement path
handles those dependencies. Fixed plans cannot restore retired dynamic bindings.

## Experiments and validation

Artifacts are under `/tmp/nixe-continuity-20261008/` and the corresponding
`/tmp/nixe-a136/continuity-*` capture directories. `source-before/` preserves the
two affected crates before this task, independently of earlier uncommitted work.
Benchmarks use disposable save copies and automated B presses. A screenshot
must match the supplied gameplay room before settling for 20 seconds and taking
the measured interval. Menu FPS and startup compilation are excluded from the
gameplay FPS comparisons.

The first experimental implementation imported callees transitively and admitted
external-return reshapes. It made startup substantially slower; one attempt hit
the 240-second window deadline. Another reached gameplay and measured 30.65 FPS,
19.20 main-vCPU CPU ms/frame and 19.02 frontend CPU ms/frame. Stack sampling found
active LCQ compilation, not a GPU deadlock. Separate 60-second startup traces
showed approximately unchanged LCQ compilation cost per unit (0.155 versus
0.158 ms) but more expensive HCQ jobs (2.77 versus 8.95 ms on average).

Restricting transitive callee expansion recovered much of startup time, but
still measured only 29.89 FPS and increased optimizer CPU work. The resulting
iteration removes external-return growth, fixes source attribution in reshape
discovery, and extends graphics binding refresh to sampled images. The discarded
policies are not retained as runtime alternatives.

Final results and validation are recorded below after the matched captures.
