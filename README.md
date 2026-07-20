# Swiitx

This is an educational and experimental project written in Rust. Its long-term goal is to research and build
functional emulators for Nintendo Switch and Nintendo Switch 2 while sharing well-defined components between
both platforms whenever technically appropriate.

## Goals

- Study modern console emulation and low-level systems programming.
- Build a functional Nintendo Switch emulator incrementally.
- Prepare an extensible foundation for Nintendo Switch 2 research and emulation.
- Reuse CPU, memory, graphics, tooling, and other infrastructure when the underlying behavior is genuinely
  shared.
- Favor testable, documented, and maintainable Rust code.

## Project Status

The project is currently in an early development and research phase.

Milestones:

- **Content formats**
  - [x] Random-access storage abstraction with bounded sub-storage views.
  - [x] User-supplied key management; cryptographic keys are never bundled or downloaded.
  - [x] NSP/PFS0 and XCI/HFS0 package loading.
  - [x] NSZ, XCZ, and NCZ compressed package support.
  - [x] Block-compressed random access and progressive caching for solid NCZ streams.
  - [x] NCA header parsing, section discovery, decryption, and integrity validation.
  - [x] ExeFS and RomFS filesystem loading.
  - [x] BKTR patch RomFS composition using base and update content.
  - [x] CNMT and NACP metadata parsing.
  - [x] Title names, publishers, languages, versions, properties, and icons.
  - [x] Recursive package discovery and title cataloguing.
  - [x] Base application, update, DLC, duplicate, and conflict resolution.
- **Executable formats**
  - [x] NRO executable loading with segments, BSS, metadata, and optional ASET support.
  - [x] Classic NSO executable loading with LZ4-compressed segments, BSS, module metadata, and `MOD0` parsing.
  - [x] Execute-only and ZBIC-compressed NSO variants.
  - [x] NPDM process metadata, filesystem permissions, service access control, and kernel capabilities.
  - [x] AArch64 executable relocations and immutable, page-safe NRO/NSO mapping plans.
- **Runtime**
  - [x] Atomic installation of prepared executable mappings into permission-aware process memory.
  - [x] Deterministic launch plans and loader coordination for packaged titles, DLC, and NRO homebrew.
- **CPU**
  - [x] CPU contracts: A64/A32/T32 profiles and state, plus permission-aware synthetic process memory and instruction fetching.
  - [x] IR foundation: typed block construction, verification, deterministic printing, and shared canonical semantic primitives.
  - [x] Declarative decoder framework: indexed A64/A32/T32 classification, typed operands, profile gating, and stable coverage IDs.

## Legal Notice

This project is intended for lawful education, research, interoperability, and preservation work. It does not
provide or distribute games, firmware, cryptographic keys, copyrighted console files, or leaked confidential
material.

Users and contributors are responsible for complying with the laws applicable in their jurisdictions and for
using only software and data they are legally entitled to use.

Nintendo Switch and Nintendo Switch 2 are trademarks of Nintendo. This project is independent and is not
affiliated with, sponsored by, or endorsed by Nintendo or NVIDIA.

## Testing

To run integration tests against caller-owned titles, copy `.env.integration.example` to
`.env.integration`, configure the paths, and run `./scripts/test-integration.sh`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
