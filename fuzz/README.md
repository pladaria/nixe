# CPU frontend fuzzing

The fuzz package uses `cargo-fuzz` and libFuzzer. Its inputs are synthetic raw
bytes only; no game, firmware, key, or other copyrighted fixture is required.

Install a nightly toolchain and the runner, then execute all four bounded targets with:

```bash
rustup toolchain install nightly
cargo install cargo-fuzz
cargo fuzzer
```

The alias runs 10,000 iterations per target. Override that finite budget with
`SWIITX_FUZZ_RUNS`, or use the ordinary target-specific interface:

```bash
SWIITX_FUZZ_RUNS=100000 cargo fuzzer
cargo +nightly fuzz run decoder
```

`decoder` covers arbitrary A64/A32/T32 encodings, normalization, operand
extraction, immediate expansion, and shifts. `translation` varies mapped and
unmapped executable layouts and asserts frontend resource bounds. `ir_verifier`
constructs malformed typed blocks. `diagnostics` exercises context, record, and
text-export limits.

Seed corpora and minimized crashes under `fuzz/corpus` and `fuzz/artifacts` are
local generated data. Commit only small, redistributable regression inputs whose
purpose is documented by a normal test.
