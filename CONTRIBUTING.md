# Contributing

## Project goal

Nixe aims to run commercial Nintendo Switch software and, eventually, Switch 2 software. The production path
must therefore be correct and competitive with high-performance emulators.

Correctness and execution performance take priority over implementation simplicity. Do not choose a simpler
implementation if it adds overhead to a hot path, limits JIT optimization, or prevents the architecture needed
to reach the performance target. Complex mechanisms are appropriate when they provide a real performance benefit
or are required by emulation semantics.

Testing, diagnostics, and abstractions are means to that goal, not goals in themselves. Do not add layers,
validation, state duplication, compatibility paths, or other runtime work merely to make testing easier. Keep the
guest-visible path direct and efficient; put exhaustive checks in tests, debug builds, or dedicated tools when
they are not required by emulation semantics. Avoid quick wins and simplifications that create a less efficient
design or defer the required architecture; simplicity means avoiding accidental complexity, not reducing the
necessary implementation.

## Architecture

The current architecture, module structure, and crate boundaries are not constraints. Large architectural
changes, including substantial rewrites or migrations, are acceptable when they move the project towards its
goal: a highly efficient Switch 1 emulator, eventually extended to Switch 2, with a very fast JIT. Change or
remove boundaries whenever the target architecture or performance requires it. Do not preserve temporary
boundaries or legacy implementations merely because they work.

An architectural change, migration, refactor, or replacement is complete only when the superseded implementation
and its obsolete adapters, branches, abstractions, and tests have been removed. Keep multiple paths only when
they represent genuinely distinct guest-visible behavior or supported host backends.

Keep platform-independent code separate from console-specific behavior. Share code across platforms only when
the abstraction is supported by verified technical knowledge.

The code is the source of truth for the current behavior and architecture. The implementation plan for the
current task defines the intended work in progress. Other documents and notes are historical context only and
may not reflect the current design, implementation, or project intent; do not treat them as requirements without
confirming them against the code or the active plan.

## Correctness and failure handling

Unsupported guest-visible behavior must stop execution with a precise, actionable error. Do not ignore it,
fabricate success, substitute defaults, or hide it as a warning.

Validate state when the operation consumes it. Avoid speculative preflight checks, transactions, rollback, and
recovery when direct execution is sufficient and guest-visible semantics do not require them.

Do not classify removal of an emulator-critical bottleneck as premature optimization.

## Testing

Add focused tests for behavior that benefits from regression coverage. Use Rust's conventional unit tests for
internal logic and integration tests for public interfaces. Tests must verify the current contract; do not retain
production compatibility paths solely to keep obsolete tests passing.

## References and implementation notes

When implementation relies on external technical references, link them in a nearby comment. For CPU instructions
implemented by the interpreter or JIT, consult the official Arm documentation and include a nearby link to the
relevant page.

## Language and contents

Use English for source code, comments, documentation, and commits. Do not include copyrighted games, firmware,
cryptographic keys, leaked material, or other content that cannot legally be redistributed.
