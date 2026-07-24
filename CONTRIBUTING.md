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

## Contribution Principles

- Prefer correctness and clear behavior over premature optimization.
- If an emulated service or component lacks required semantics, stop the emulator with a typed host-side error.
  Never hide missing implementation with a guest-visible error or fabricated data; guest-visible failures are valid
  only when they faithfully model the emulated system.
- Add tests for new behavior whenever practical.
- When an implementation relies on external references, record those references in nearby code comments and link
  to the relevant resources. Prefer stable, versioned, or commit-pinned links when available.
- When implementing CPU instructions for the interpreter or JIT, consult the official Arm documentation and add
  a nearby comment linking to the relevant page.
- Keep platform-independent code separate from console-specific behavior.
- Share code between platforms only when the abstraction is supported by verified technical knowledge.
- Do not include copyrighted games, firmware, cryptographic keys, leaked material, or other content that
  cannot be legally redistributed.
