# VP9 header reconstruction

`vp9.cpp`, `vp9.h` and `types.h` are adapted from Eden's public NVDEC VP9 reconstructor at revision 67bada77f8a43a90da2e94e89b8e7da73c256989, available at https://github.com/eden-emulator/mirror/tree/67bada77f8a43a90da2e94e89b8e7da73c256989/src/video_core/host1x.

Copyright 2020 yuzu Emulator Project; copyright 2026 Eden Emulator Project.
GPL-3.0-or-later (original yuzu portions GPL-2.0-or-later). The GPL license is included in `LICENSE.txt`.

Nixe replaces emulator-specific memory/decoder dependencies with explicit inputs. It maps NVDEC's physical reference surfaces directly to the software decoder's eight DPB slots, decodes each submitted visible frame without lookahead, and supplies the complete hardware entropy table through a non-adapting software frame context. Quantization deltas are signed. Hidden/intra-only output is an explicit unsupported boundary, rather than silently dropping surface writes. Only the VP9 reconstructor and the consumed wire types are retained.
