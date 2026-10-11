# zstd-sys with ZBIC frame support

Based on `zstd-sys` 2.0.16+zstd.1.5.7 from crates.io, including Zstandard 1.5.7.
The Rust bindings retain the upstream MIT/Apache-2.0 licensing. Zstandard's
BSD license and alternative GPL license are preserved in `zstd/LICENSE` and
`zstd/COPYING`. Upstream: https://github.com/gyscos/zstd-rs and
https://github.com/facebook/zstd/tree/v1.5.7.

The Cargo crates.io patch preserves the `zstd` and `zstd-safe` public APIs. Both
standard Zstandard frames (NCZ/NSZ/XCZ) and Nintendo ZBIC frames (NSO segments)
use this library. Compression still produces standard Zstandard frames.

Local changes:

- recognize little-endian magic `0x4349425A`, alongside standard frame magic,
  including partial streaming headers;
- retain the entropy variant in the opaque decompression context, selected
  independently for each frame;
- decode ZBIC normalized FSE counts using the independently written binary
  interpolative reader, derived from Nixe's previous MIT Rust decoder;
- propagate the variant through sequence tables and FSE-compressed Huffman
  weights, while preserving runtime BMI2 dispatch and Huffman assembly;
- require the bundled library instead of accepting an unpatched system library.

Internal entropy dispatch uses bit 0 for BMI2 and bit 1 for ZBIC. The Huffman
flags carry a separate `HUF_flags_zbic` flag. These flags are private to the C
implementation; exported signatures and Rust bindings are unchanged. There is
no global/thread-local variant selector and no fallback decoder.

The prior MIT decoder notice is retained in `LICENSE-ruzstd` for the
translated BIC-count reader.

Format reference: https://github.com/Atmosphere-NX/Atmosphere/tree/master/libraries/libstratosphere/source/ro.
No Atmosphere source code is included.
