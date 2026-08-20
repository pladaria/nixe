# Contributing

Thank you for your interest in contributing to this project.

## Project Structure

The current directory and crate structure is an initial proposal. It is temporary and may change as the
project evolves, new technical information becomes available, and implementation needs become clearer.

Contributors should avoid treating the existing module boundaries as permanent. Architectural changes are
welcome when they improve correctness, maintainability, testing, or meaningful code reuse between the
supported platforms.

## Language

English is the required language for all project content, including: source code, source code comments,
documentation, commits, etc.

## Testing Guidelines

We follow standard Rust conventions to separate unit and integration tests. Please adhere to the following rules:

- **Unit Tests:** Place them at the bottom of the source file they test (e.g., `src/foo.rs`) inside a `#[cfg(test)] mod tests` block. Use them to verify internal logic and private functions.
- **Integration Tests:** Place them in separate files inside the root `tests/` directory (e.g., `tests/api_tests.rs`). Use them to test the public API as an external consumer.
- **Prohibited:** Do **NOT** put integration tests, end-to-end flows, or public API testing inside `src/lib.rs` or `mod.rs`. Keep these files focused exclusively on module definitions and internal unit tests.

## Fail-fast Policy

Nixe prioritizes correctness over resilience.

- Unsupported guest-visible behavior must stop execution immediately with a precise, actionable error. Never ignore
  it, fabricate success, substitute defaults, or downgrade it to a warning or debug message.
- Validate state when an operation consumes it. Unrelated partial state must not block the operation, but required
  invalid or incomplete state must fail.
- Prefer direct, single-pass execution. Add preflight validation, transactions, rollback, or recovery only when
  required by guest-visible semantics or a documented emulator invariant, not speculatively.
- Tests must preserve these failure boundaries and reject fabricated success paths.

## Contribution Principles

- Prefer correctness and clear behavior over premature optimization.
- Add tests for new behavior whenever practical.
- When an implementation relies on external references, record those references in nearby code comments and link
  to the relevant resources. Prefer stable, versioned, or commit-pinned links when available.
- When implementing CPU instructions for the interpreter or JIT, consult the official Arm documentation and add
  a nearby comment linking to the relevant page.
- Keep platform-independent code separate from console-specific behavior.
- Share code between platforms only when the abstraction is supported by verified technical knowledge.
- Do not include copyrighted games, firmware, cryptographic keys, leaked material, or other content that
  cannot be legally redistributed.
