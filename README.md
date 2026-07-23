# Nixe

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

## Testing

Run

```
cargo test-all
```

### Integration tests against real titles

To run integration tests against caller-owned titles, copy `.env.integration.example` to
`.env.integration`, configure the paths, and run `./scripts/test-integration.sh`.

### Differential tests

Optional CPU differential tests require QEMU user-mode (`qemu-aarch64` and `qemu-arm`), the Rust
`aarch64-unknown-linux-gnu` and `armv7-unknown-linux-gnueabihf` targets, and their cross-linkers.

```bash
sudo apt update && sudo apt install qemu-user gcc-aarch64-linux-gnu gcc-arm-linux-gnueabihf
rustup target add aarch64-unknown-linux-gnu armv7-unknown-linux-gnueabihf
```

Verify with

```bash
qemu-aarch64 --version
qemu-arm --version
aarch64-linux-gnu-gcc --version
arm-linux-gnueabihf-gcc --version
```

Then run

```bash
cargo test-diff
```

### Fuzz tests

CPU decoder, translation, IR verifier, and diagnostic fuzz targets require a nightly Rust toolchain
and `cargo-fuzz`:

```bash
rustup toolchain install nightly
cargo install cargo-fuzz
cargo fuzz-all
```

The aggregate alias runs 10,000 iterations for each target. See [fuzz/README.md](fuzz/README.md) for
target-specific commands and configuration.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Legal Notice

This project is intended for lawful education, research, interoperability, and preservation work. It does not
provide or distribute games, firmware, cryptographic keys, copyrighted console files, or leaked confidential
material.

Users and contributors are responsible for complying with the laws applicable in their jurisdictions and for
using only software and data they are legally entitled to use.

Nintendo Switch and Nintendo Switch 2 are trademarks of Nintendo. This project is independent and is not
affiliated with, sponsored by, or endorsed by Nintendo or NVIDIA.

## License

Nixe is licensed under the GNU General Public License version 3 or later. See [LICENSE.txt](LICENSE.txt).
