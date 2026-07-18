# Swiix

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

The project is currently in the planning and research phase.

## Shared Configuration

The CLI and future desktop application use the same versioned TOML configuration schema, implemented by the
`swiitx-config` crate. By default, applications look for `swiitx.toml` in the current directory and then in the
platform user configuration directory. Set `SWIITX_CONFIG` or pass `--config` to the CLI to select another
file explicitly.

Relative library and key paths are resolved from the directory containing the configuration file. The
configuration stores only the location of caller-owned key files, never cryptographic key material itself.

Run the CLI without a title path to recursively scan the configured library directories and print one summary
per resolved application, including its selected patch and compatible add-on content. Pass one title package
or directory path to request the detailed package inspection instead:

```text
swiitx-cli [--config <file>]
swiitx-cli [--config <file>] <title-path>
```

## Legal Notice

This project is intended for lawful education, research, interoperability, and preservation work. It does not
provide or distribute games, firmware, cryptographic keys, copyrighted console files, or leaked confidential
material.

Users and contributors are responsible for complying with the laws applicable in their jurisdictions and for
using only software and data they are legally entitled to use.

Nintendo Switch and Nintendo Switch 2 are trademarks of Nintendo. This project is independent and is not
affiliated with, sponsored by, or endorsed by Nintendo or NVIDIA.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
