# CPU frontend fuzzing

The fuzz package uses `cargo-fuzz` and libFuzzer. Its inputs are synthetic raw
bytes only; no game, firmware, key, or other copyrighted fixture is required.

Install a nightly toolchain and the runner, then execute the bounded decoder target with:

```bash
rustup toolchain install nightly
cargo install cargo-fuzz
cargo +nightly fuzz run decoder
```

`decoder` covers arbitrary A64 encodings on every supported platform table,
normalization, operand extraction, immediate expansion, and shifts.

Seed corpora and minimized crashes under `fuzz/corpus` and `fuzz/artifacts` are
local generated data. Commit only small, redistributable regression inputs whose
purpose is documented by a normal test.
