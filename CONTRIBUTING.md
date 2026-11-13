# Contributing

## Project goal

Nixe aims to run commercial Nintendo Switch software and, eventually, Switch 2 software. The production path
must therefore be correct and competitive with high-performance emulators.

## Architecture

The current architecture, module structure, and crate boundaries are not constraints. Large architectural
changes, including substantial rewrites or migrations, are acceptable when they move the project towards its
goal: a highly efficient Switch 1 emulator, eventually extended to Switch 2. Change or
remove boundaries whenever the target architecture or performance requires it. Do not preserve temporary
boundaries or legacy implementations merely because they work.

An architectural change, migration, refactor, or replacement is complete only when the superseded implementation
and its obsolete adapters, branches, abstractions, and tests have been removed. Keep multiple paths only when
they represent genuinely distinct guest-visible behavior or supported host backends.

Keep platform-independent code separate from console-specific behavior. Share code across platforms only when
the abstraction is supported by verified technical knowledge.

The code is the source of truth for the current behavior and architecture. Other documents and notes are
historical context only and may not reflect the current design, implementation, or project intent; do not
treat them as requirements.

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

When implementation relies on external technical references, link them in a nearby comment.

## Language and contents

Use English for source code, comments, documentation, and commits. Do not include copyrighted games, firmware,
cryptographic keys, leaked material, or other content that cannot legally be redistributed.
