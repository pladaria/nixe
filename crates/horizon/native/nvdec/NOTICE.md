# VP9 header reconstruction

`vp9.cpp`, `vp9.h` and `types.h` are adapted from Eden's public NVDEC VP9 reconstructor at revision d85005b2b27ebdbe785316886cdf04c2894d550e, available at https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/host1x.

Copyright 2020 yuzu Emulator Project; copyright 2026 Eden Emulator Project.
GPL-3.0-or-later (original yuzu portions GPL-2.0-or-later). The GPL license is included in `LICENSE.txt`.

Nixe retains the reference decoder's one-frame lookahead, last/golden/alt reference selection, segmentation state and separate visible/hidden probability contexts. Explicit inputs replace emulator-specific memory dependencies. FFmpeg receives every reconstructed frame; only visible frames acquire presentation metadata for VIC. Quantization deltas remain signed. Only the VP9 reconstructor and the consumed wire types are retained.
