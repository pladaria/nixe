# ruzstd-zbic

This directory is based on `ruzstd` 0.8.3 by Moritz Borcherding, licensed under MIT. The upstream
source was retrieved from crates.io.

Local changes are intentionally limited to Nintendo's ZBIC decoding variant:

- use the little-endian frame magic `0x4349425A` (`ZBIC`);
- decode normalized FSE probability tables with binary interpolative coding; and
- remove upstream development dependencies and tests from this vendored build.

The BIC representation was independently implemented in safe Rust after comparing the behavior of
the Horizon-compatible Atmosphere loader. No Atmosphere source code is included here.
