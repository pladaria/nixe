# Tessellation support through wgpu native interoperability

Status: **Phases A–E are complete for the initial supported subset on NVIDIA.**
The unmodified `deko_examples` Simple Tessellation demo renders through the
production backend. Controlled Switch/Nixe captures of smooth wireframe widths
1/4 pass the agreed visual/JPEG-tolerant comparison. This is not general Maxwell
compatibility, bit-exact coverage or cross-vendor acceptance.

This document records the current architecture, supported boundaries and evidence.
New modes are implemented when needed, with their own tests; they are not
unfinished work within this scope.

## 1. Architecture and ownership

**wgpu remains the primary backend.** Native Vulkan execution through `wgpu-hal`
supplies tessellation inside `nixe-gpu-wgpu`, sharing the existing device,
resident resources, queue, completion tracking and presentation. There is no
separate Vulkan backend, resource table, CPU tessellator or interop framebuffer
copy. Ordinary draws and presentation use wgpu.

| Location under `crates/` | Responsibility |
| --- | --- |
| `gpu-maxwell/src/engines/threed/tessellation.rs` | Maxwell mode decoding and consumed patch state |
| `gpu-maxwell/src/engines/threed/{bindings,state,draw}.rs` and `draw/` | Register provenance, MME replay, invalidation, raster/color state and lowering |
| `gpu-maxwell/src/shader.rs` and `shader/` | SPH/SASS translation, patch addressing, ordering and linkage; no Vulkan types |
| `gpu/src/{command,color,tessellation,shader}.rs` and `shader/` | Neutral contracts, shared verified IR and stage linkage |
| `gpu/src/shader/spirv/` | Device-independent IR-to-SPIR-V emission using rspirv |
| `gpu-wgpu/src/native/vulkan/` | Device setup, pipelines, descriptors, limits, persistence and raw recording |
| `gpu-wgpu/src/{lib,driver}.rs` | Shared device/resources, ordered submission and write/completion accounting |
| `gpu-wgpu/tests/native_interop/` | Interop fixtures, production-runtime oracles and opt-in measurements |
| `video-winit/src/screenshot.rs` | On-demand PNG dumps of resident presentation images |

The native draw implementation is a private child of the existing driver, not
another resource-owning layer. Unsupported consumed semantics use the existing
error path. Optional native features do not disable ordinary wgpu rendering.
Metal keeps its ordinary functionality; native Metal/DX12 tessellation is not
implemented by this bridge.

The baseline is wgpu/core/hal/types and Naga 30.0.0, with ash 0.38.0+1.3.281.
Unsafe interoperability assumptions must be rechecked on dependency upgrades.

### Device and submission

The selected Vulkan adapter supplies HAL-required features, extensions and queue
requirements. Setup preserves that chain, augments supported native features,
creates one logical device and transfers it through HAL `device_from_raw` and
wgpu `create_device_from_hal`, with explicit ownership/failure cleanup. HAL 30's
feature-chain builder installs its own core-feature pointer; the augmented core
record is installed afterward. Capabilities/physical limits are queried once and
checked at consumption. See [HAL adapter][wgpu-adapter] and [wrapping API][wgpu-wrap].

wgpu 30 forbids mixing Wgpu and Raw recording APIs on one encoder;
`transition_resources` uses the former. Submissions therefore contain:

1. Normal encoding for preceding commands/uploads and native-entry usage declarations.
2. A fresh raw encoder for native dependencies, render pass and draws.
3. Another normal encoder for following wgpu commands with matching tracked state.

The ordered buffers use one wgpu queue submission per existing guest boundary;
there is no native `vkQueueSubmit` owner. Compatible native draws share a pass
and boundary dependency pair. See [encoding separation][wgpu-encoding] and
[resource transitions][wgpu-transitions].

HAL's generic shader-access mapping omits TCS/TES. Explicit entry/exit dependencies
include the emitted descriptor ABI's actual stages, vertex/index reads and
attachment accesses, including read-before-write execution hazards. Native passes
preserve tracked layouts. Read-only buffer uses register once per segment;
there is no per-draw `ALL_COMMANDS` barrier or duplicate persistent tracker.
Queue order alone is not a memory dependency. See [HAL mapping][wgpu-conv].

Transitions do not establish wgpu initialization. A legitimate load/clear pass
does so before first raw attachment use, preserving contents; explicit clears
are honored, store-discard invalidates that fact, and warm load/store needs no
initializer. Buffer reads require initialized canonical backing through ordinary
uploads.

Submitted uses retain native objects, descriptor pool pages and backing resources
through completion. Cache eviction drops cache ownership only; cold resource
destruction/residency eviction invalidates dependent entries. Terminal teardown
and direct drop wait for raw uses; submission/cache eviction do not wait idle.
Native writes use normal coherence and resident-presentation accounting.

## 2. Maxwell and shader semantics

`MAXWELL_B` byte method `0x0320` carries domain in bits 0..1, spacing in 4..5 and
connectedness/winding in 8..9. Typed state retains raw values/provenance and reuses
patch-size/default-level registers. Dispatch, MME replay, channel lifetime and
prepared-draw invalidation are tested. State is validated at draw consumption,
not by constructing pipelines on writes. See [registers][deko-registers],
[binding producer][deko-bind] and [DKSH metadata][deko-dksh].

[Compiler metadata][uam-tess-mode] identifies `0x201` as triangles/equal-spacing/CCW
in domain space, despite conflicting dump labels. Neutral winding uses the GLSL
lower-left domain; native pipelines explicitly select Vulkan `LOWER_LEFT`.
Viewport/front-face state is not inverted to compensate. Negative-height
viewports preserve Y without modifying intermediate VS/TCS positions.

Prepared draws distinguish input patch size from TCS output count. Default
levels preserve raw bits and definition masks, including domain-unused levels
read by TES; absent consumed state is not fabricated as zero. Raster checks
classify patches by generated primitive, not input topology.

### IR, emission and communication

`ShaderBackendModule` shares verified IR without mandatory WGSL. Ordinary
vertex/fragment draws lazily generate/cache WGSL/Naga; native pipeline misses
compile a coherent VS/TCS/TES/FS SPIR-V chain. Both are active execution paths.

IR represents indexed control-point arrays, patch attributes, TCS output reads,
invocation/patch/primitive IDs, coordinates/levels, output cardinality and patch
rendezvous. Component-wise linking checks adjacent stages; fragment interpolation
applies only to the final pre-raster producer.

The rspirv emitter builds SPIR-V 1.3 for Vulkan 1.1 plus declared extensions.
Registers/predicates use SSA and predication uses structured selections/phis.
Undefined values stay undefined. There is no register-bank interpreter,
instruction dispatch loop or external production compiler. Compilation prunes
fragment-unused TES generic components before arithmetic/resource liveness,
without altering guest IR; whole-chain backward interface pruning is not
implemented. See [SPIR-V][spirv] and [Vulkan stages][vk-shaders].

Without a guest TCS, a generated control shader forwards required points unchanged
and invocation zero supplies levels. Cardinality is preserved; six dynamic level
words use 24-byte push constants, not recompilation or descriptors. Missing
producers and unsupported patch inputs fail precisely.

[UAM removes non-compute BAR.SYNC][uam-barriers], relying on Maxwell warp lockstep.
Direct unconditional TCS `ALD.O.P` output reads are translated; scalar-slot bitsets
detect RAW/WAR/WAW conflicts and insert unconditional `PatchBarrier` operations.
Independent slots/repeated reads add none. Predicated stores retain unconditional
rendezvous. The pass is O(instructions), with O(1) slot checks and no auxiliary
allocation when no barriers are needed; the demo TCS needs none.

The compiled `patch_address/barriers.tesc` fixture has three source barriers but
none in SASS. Translation restores barriers at 0x30, 0x58 and 0x68, and execution
checks that every invocation preserves a shared value across an overwrite.
Unproven control-flow uniformity remains unsupported.

### Numerical contract

Float32 requires RTE and signed-zero/Inf/NaN guarantees. With float32 denormal
preservation, per-operation DAZ/FTZ uses integer classification and signed zero
selection. Add/multiply carry `NoContraction`. Correctly rounded FMA requires
`OpFmaKHR` and enabled `VK_KHR_shader_fma`, not GLSL Fma or separate multiply/add.

Without float32 denormal preservation, RNE operations with both DAZ and FTZ use
float32/integer cancellation repair for addition and guarded float64 repair for
tiny multiply/FMA results. An exact product and TwoSum residual classify the
minimum-normal midpoint without double rounding. Repair requires float64 RTE and
signed-zero/Inf/NaN, not denormal preservation or FMA64. Normal results retain
native float32; absent consumed guarantees fail at the source instruction.

Maxwell FTZ applies to inputs/results; distinct DNZ semantics remain unsupported.
Signed/unsigned integer-to-float conversion uses native RTE without underflow
repair. The registry-derived `repr(C)` binding in `native/vulkan/fma.rs` fills
the current ash API gap without another ash version/fork; replace it when HAL's
ash provides it. Capabilities come from queries, not vendor assumptions.
See [FMA][vk-fma], [float controls][vk-float-controls] and the arithmetic arguments
beside `gpu/src/shader/spirv/float.rs`.

## 3. Execution, resources and caches

Production native execution supports triangle-domain/equal-spacing tessellation,
triangle output, explicit/generated TCS, Float32 vertex formats, instancing,
fill/supported smooth wireframe, one color target and optional depth. Attachments
use one mip/layer/sample, preserved stores and representable viewport depth
transforms. Other combinations fail explicitly.

The neutral backend supports nonindexed and uint16/uint32 indexed patch lists,
independent binding/first-index offsets, signed base vertex, first instance and
incomplete trailing patches. Checks are O(1), without CPU index scans/conversion.
Indexed vertex-rate streams need robustness and canonical backing through the
host buffer end; uint32 needs `fullDrawIndexUint32`. Uint8, restart, narrowly
bounded indexed vertex subranges and zero-element/instance neutral commands
are unsupported. Maxwell still lowers nonindexed draws: indexed tests do not
establish new guest command decoding.

Native descriptors expose live read-only constant buffers as raw-word storage
blocks: initialized canonical backing, word-sized, offset zero, covering the
resident allocation within physical limits. Images/samplers/writable descriptors
are not supported by this production ABI. Robustness ensures host safety, not
Maxwell out-of-bounds numerical equivalence. Raw fixtures can exercise broader
HAL capabilities without advertising guest support.

### Warm-path ownership and lookup

- Pipeline/layout/render-pass keys include shader handles/generations, formats,
  patch structure, vertex layout and static depth/raster/color state. Dynamic
  levels, viewport, width, bindings and arguments do not specialize pipelines.
  Prepared draws use a pointer fast path; other hits use hashing. Bounded
  eviction scans occur on misses.
- Framebuffer keys include logical generation and host view identity, preventing
  stale reuse after physical residency recreation.
- Native tables lazily build a 256-entry binding index: O(1) resolution per binding
  and O(live bindings) key construction using scratch storage. Ordinary-only
  tables allocate no native index.
- Descriptor fingerprints select buckets; full pipeline/buffer identities prevent
  collision aliasing. Contents update through normal uploads without changing
  descriptor identity. Immutable sets allocate in 32-set pages on misses; warm
  hits neither update/allocate Vulkan sets nor acquire the pool lock.
- Consecutive draws suppress unchanged bindings/dynamic state using the preceding
  retained draw, not a second persistent cache. Buffer usages register once per
  segment; shared XXH3 uses a static secret without allocation.
- Maxwell resource-cache rejection checks cheap retained-state identity before
  role lists. Non-current misses retain the bounded existing search, without an
  unmeasured duplicate index.

### Persistence

A lazy bridge-owned `VkPipelineCache` is separate from wgpu's opaque cache.
Only native misses access it. Explicit teardown saves
`native-vulkan-<vendor>-<device>.bin` beside wgpu's cache; direct drop releases
objects safely but is not a persistence checkpoint.

The little-endian envelope includes format/translation ABI versions, host word
size/endianness, vendor/device/driver, UUID, payload length and SHA-256.
Incompatible/truncated/corrupt/oversized data is discarded before import; real
I/O/API errors remain errors. Sibling temporary files and atomic replacement
prevent partial publication. Concurrent writers use last-complete-writer-wins.

The byte budget bounds each payload, not combined files or opaque driver memory.
Extraction bounds allocation and accepts Vulkan's valid `VK_INCOMPLETE` subset,
not arbitrary truncation. Below the 32-byte header budget, native data cannot be
saved. Persistence restores no handles/SPIR-V and does not prewarm shaders.

## 4. Raster and shared fixed-function state

Both paths use neutral RGB/alpha equations and masks. Maxwell common/per-target
state and physical routing are consumed at draw; disabled blend ignores equations,
min/max ignore factors. Add/subtract/reverse-subtract/min/max and ordinary
source/destination factors use hardware blending. Constant/coupled/dual-source
factors remain unsupported. Maxwell blending supports RGBA8/BGRA8 UNORM/sRGB;
native RGBA32F blending needs further format-capability negotiation.

Separate-alpha selectors have verified initial value one in typed state, raw
reads and MME replay; other missing equations are not defaulted. References and
initialization-table digest are beside `engines/threed/output.rs`, without any
firmware asset in the repository.

Front-face/culling share framebuffer-space conventions. Only visible face modes
are consumed, or modes must match when both faces produce fragments. Window
origin is upper-left without flip. Native front-and-back culling retains
pre-raster execution; ordinary wgpu rejects it and culling before emulated
fill-rectangle expansion.

Native wireframe explicitly uses `RECTANGULAR`/`RECTANGULAR_SMOOTH`, with optional
non-solid fill/wide-line/line-rasterization features. Width is dynamic; mode and
smoothing are static. Missing features/invalid widths fail without clamps or
CPU expansion. Maxwell accepts smooth polygon lines through patches, consuming
line smoothing/width, clipping-edge policy, flags and stipple rather than
unrelated filled-polygon smooth state. Ordinary wireframe, aliased polygon lines,
mixed visible face modes, point polygon mode, stipple, suppressed edges, consumed
depth bias and rasterizer discard remain unsupported. See [line rules][vk-lines].

Inactive alpha-to-coverage footprints (`0x12e0`) retain source/MME state without
changing image/pipeline identity. Enabled alpha-to-coverage/dithering/alpha-to-one
fails at consumption; this is not MSAA support. Explicit absence of a Z target
suppresses draw depth/stencil dependencies even with default testing enabled.
Unknown selection remains conservative; explicit clears have separate needs.

## 5. Acceptance and reproduction

| Phase | Completed scope |
| --- | --- |
| A | One-device/queue interop, initialization, synchronization and retained lifetimes on NVIDIA |
| B | Maxwell state, verified patch IR, stage linkage and unsupported boundaries |
| C | SPIR-V/numerical output, default TCS, cached execution, descriptors, indexed/instanced neutral draws, persistence and compiled communication |
| D | Demo raster behavior, manual demo/re-entry acceptance and paired Switch cases 14/15 |
| E | CPU/allocation, cache, command-stream and live GPU measurements; no temporary production probes |

Hardware acceptance uses NVIDIA RTX 4070 Ti SUPER / 595.58.03 and Khronos 1.4.341
core/synchronization validation. SPIRV-Tools validates emission, including
`SPV_KHR_fma`. Another physical vendor is unverified.

The [native interop guide](../../crates/gpu-wgpu/tests/native_interop/README.md)
contains commands, tooling, oracles and measurement scripts:

- Raw fixtures isolate import, dependencies, initialization and lifetime, including
  sampling/subresources beyond the production ABI.
- Public-runtime tests use real resource/visibility/presentation ownership for
  explicit/default TCS, offsets, partial patches, built-ins, dynamic levels,
  ordinary/native transitions, discard/reuse, generations, aliases, real residency
  pressure, descriptor-pool rollover and persistence.
- Captured VS/TCS/TES/FS runs use controlled vertices, queued colors and both
  viewport signs. Fill, compiled-patch-communication and combined smooth/blended/
  culling oracles are not full guest-stream replays.
- The combined 64x64 mesh checks edges, holes, partial coverage, gradients,
  background and coverage growth with width. On transparent black, one-hot
  vertex colors preserve `sum(RGB) == alpha` through blending within six UNORM8
  steps; this is not a hardware coverage golden image.
- Arithmetic tests compare 65,536 results per operation bitwise against the IR
  evaluator, including rounding, cancellation, underflow and FMA residuals.

The independent communication fixture's unused VS outputs produce
`Shader-OutputNotConsumed` performance warnings, not suppressed or counted as
errors. GPU tests require physical adapters and print `SKIP:` if absent; native
entry points also check capabilities. Missing tools/layers on compatible hardware
and initialization/execution/validation errors fail. Rust labels early returns
`ok`: use `--nocapture` and do not count skips as acceptance. This does not change
production adapter selection. Third-party tools belong in ignored `dev-tools/`.

Ordinary verification, with hardware-specific tests additional:

~~~sh
cargo fmt --all -- --check
cargo test -p nixe-gpu -p nixe-gpu-maxwell -p nixe-gpu-wgpu -p nixe-video-winit
cargo clippy -p nixe-gpu -p nixe-gpu-maxwell -p nixe-gpu-wgpu -p nixe-video-winit --all-targets -- -D warnings
cargo build --release -p nixe-cli
~~~

### Switch comparison and screenshots

The sibling `switch-examples/graphics/deko3d/deko_examples2` is based on revision
`669786898205b7beb25ff1731e72982e6d0397d3`. It preserves the original example;
cases 10–18 use fixed geometry and a 512x512 viewport in a 1280x720 framebuffer.
Cases 19/20 are exploratory Switch-only aliased-line references, not acceptance.
Its `RASTER-TESTS.txt` records state, expected black culling cases and Docker
instructions. Build: `devkitpro/devkita64:20260219` (UAM 1.1.0, GCC 15.2.0).
NRO: `roms/homebrew/custom/deko_examples2.nro`.

Paired captures: `docs/screenshots/reference/deko_examples2/{14,15}.jpg` and
`docs/screenshots/nixe/{14,15}.png`, all 1280x720. Cases 14/15 use smooth widths
1/4. Placement, connectivity, gradients, thickness, intersections and interiors
agree visually, with black RGB outside the viewport. Originals were not resized,
registered or overwritten.

Switch JPEGs use vertical chroma subsampling. An in-memory diagnostic encoding
uses each reference's quantization/sampling tables; transposed pixels/matrices
adapt Pillow's horizontal 4:2:2 encoder, with transpose undone after decoding.
This neither assumes identical encoders nor replaces originals. Measurements
use the union of pixels above 16/255 in any channel, expanded by four pixels:

| Case | Raw RGB MAE near mesh (0–255) | JPEG-matched diagnostic MAE | Diagnostic p95 channel error |
| --- | ---: | ---: | ---: |
| 14, width 1 | 3.547 | 0.540 | 3 |
| 15, width 4 | 3.474 | 0.730 | 3 |

Maximum diagnostic errors are 15/16; integrated active 16x16 block mean errors
are 0.078/0.122. Offset search ±2 pixels selects `(0,0)`: one pixel raises viewport
MSE from 0.299/0.478 to at least 80.958/106.027. This rejects whole-image
displacement, not subpixel differences. Local read-only
`dev-tools/compare-tessellation-captures.py` reproduces metrics/input hashes with
Pillow/NumPy. These cases pass JPEG tolerance, not bit-exact alpha/coverage or
other-variant/vendor acceptance.

S dumps the next leased resident image to
`docs/screenshots/nixe/<title>-<UTC timestamp>.png`: native cropped pixels with
display orientation, no scaling/render pass/RGB conversion. PNG omits internal
alpha as the opaque presenter does. Readback is on demand, one capture in flight,
with background writing. Tests cover padding, cropping, RGBA/BGRA and eight transforms.

## 6. Performance evidence

Evidence distinguishes CPU submit cost, serialized replay timing and live backend
GPU spans on the tested NVIDIA adapter, not general speedup/uncapped FPS/full
frame latency. Validation, allocation profiling and RenderDoc controls are
test-only. Temporary timestamp/present-mode probes are not in production.

### CPU and allocations

The 32x32 release benchmark uses 64 warm-ups, 512 timing samples without allocation
counting and 64 separate calling-thread allocation samples. Construction/readback/
batch retirement are outside timing. Counts include reallocations and wgpu/runtime,
not driver C allocations/other threads; bytes are requests, not peak memory.

Attributed static-secret XXH3 and segment-wide usage registration reductions:

| Case | Allocation requests before / after | Requested bytes before / after |
| --- | --- | --- |
| Ordinary | 129.03 / 125.03 | 15,816 / 15,072 |
| Native, 2 draws | 164.92 / 161.03 | 25,431 / 22,873 |
| Native, 32 draws | 229.03 / 164.03 | 110,029 / 35,381 |
| Mixed | 228.89 / 220.14 | 35,069 / 29,485 |

These establish call-site allocation reductions, not FPS gains; bookkeeping,
retention and wgpu command storage still allocate. Four fresh-process runs with
no validation/RenderDoc/backtraces compare fill and smooth-width-4/blend/culling:

| Submission | CPU p50 range (µs) | CPU p95 range (µs) | Mean allocation calls |
| --- | ---: | ---: | ---: |
| Native fill | 76.31–104.69 | 97.82–159.65 | 160.05–160.94 |
| Native smooth/blend | 78.14–84.82 | 99.49–176.59 | 160.70–161.03 |
| Mixed fill | 108.05–133.02 | 137.56–208.37 | 220.28–221.03 |
| Mixed smooth/blend | 110.36–120.16 | 137.75–162.86 | 220.09–221.03 |

No systematic allocation increase appears for raster state. Variable timing and
fixed case order preclude zero-overhead claims. Shaders are synthetic, not the
captured chain; raster preparation is outside timing.

### Persistence and commands

Six balanced fresh-process pairs with the same seeded wgpu cache measured
first-submit medians 8.860 ms without and 7.551 ms with the native file (ranges
8.291–9.893 and 7.291–8.531 ms). Each checks pixels. Translation, uploads and
uncontrolled OS/driver caches remain included; these are not isolated compilation
times or driver hit ratios.

RenderDoc 1.46 warm captures:

| Workload | Native / ordinary draws | Render passes | Pipeline binds | Descriptor binds | Vulkan barriers | Queue submits |
| --- | --- | --- | --- | --- | --- | --- |
| Ordinary | 0 / 3 | 3 | 2 | 1 | 5 | 1 |
| Native | 2 / 1 | 4 | 2 | 2 | 8 | 1 |
| Native, 32 draws | 32 / 1 | 4 | 2 | 2 | 8 | 1 |
| Mixed | 2 / 4 | 7 | 4 | 3 | 14 | 1 |

Ordinary counts include partial-clear draws. Native 2/32 bind/barrier counts are
equal. Entry/exit masks are `0x1fbc -> 0x7bc` and `0x7bc -> 0x1fbc`; other
barriers are wgpu transitions. No `ALL_COMMANDS` appears. Warm copies update
partial-clear uniforms, not interop geometry/images; no warm image copies appear.

Twenty replay samples, minus three warm-ups, give median summed native draw
durations 4.096 µs (two), 22.528 µs (32) and 3.072 µs (mixed's two). Replay
serializes work, shows 1.024-µs quantization and excludes non-draw costs. Cold
creation records are 1,377 µs native and 5,564 µs ordinary partial-clear pipeline
under interposition. Warm-capture initial-state creation records are not misses.
Scripts and reproduction caveats are in the test guide.

### Live workloads

User-space `perf` at 49 Hz sampled release `deko_examples2` without tracing.
Of 208 tessellation samples, 49.0% were on `nixe-gpu-owner`, 38.5% on `nixe-guest`
and 3.4% on JIT vCPU: CPU shares, not GPU timing. Whole-process RSS settled near
1.22 GiB, including the driver.

A 1920x1080 Cube profile motivated cheap-state-first resource-cache rejection.
Single 30-second before/after runs had 379/334 samples, 46/0 separately attributed
to role equality and about 7.73/6.82 CPU seconds. Rotation/load/sampling variance
preclude repeatable speedup claims. The user verified unchanged tessellation.

A same-process handheld run measured both demos at 1280x720. Temporary timestamps
bracketed backend submissions and their ordered segments; readback was batched
until teardown, without validation/RenderDoc. Stable windows exclude transitions:

| Workload | Window | Submissions/s | GPU p50/p95/p99 (µs) | Summed backend GPU time/s |
| --- | ---: | ---: | ---: | ---: |
| Simple Tessellation | 25 s | 60 | 239.616 / 420.864 / 454.656 | 14.24 ms |
| Cube | 30 s | 120 | 12.288 / 50.176 / 57.344 | 2.65 ms |

No span exceeded 0.472 ms for tessellation or 0.089 ms for Cube. Cube used about
21% of one CPU across 26 one-second `pidstat` samples, RSS near 1.23 GiB.
Spans exclude queue waits/presentation/commands outside the pair: GPU headroom,
not full-frame latency. Probe overhead prevents a CPU/FPS speedup claim.

Both displayed 60 FPS. A separate Immediate-presenter diagnostic did not remove
guest `DisplayClock::new(60)` pacing; uncapped testing would alter the correctness
workload. Both probes were removed; tests, core/sync hardware validation, Clippy,
formatting and release build passed after cleanup.

## 7. Explicit limits and future extensions

- Only the documented NVIDIA subset is accepted. Other physical vendors need
  validation; native Metal/DX12 tessellation is not implemented.
- Quad/isoline domains, fractional spacing and other generated primitives lack
  production acceptance from triangle/equal-spacing tests. Vulkan allows
  observable tessellator/smooth-line differences; new modes need hardware oracles,
  not just pipeline creation. See [tessellation][vk-tessellation].
- General guest control flow, explicit BAR modes, conditional/indirect patch
  reads, TCS per-vertex output-address decoding, geometry and transform feedback
  remain unsupported. Neutral tests do not imply corresponding SASS support.
  Images/writable descriptors, MRT, broader raster modes and guest out-of-bounds
  behavior are outside the documented boundary.
- Cases 14/15 validate geometry/color/smooth widths within JPEG tolerance, not
  bit-exact alpha/coverage. Other variants/vendors are unverified; captured-chain
  oracles are not full guest-stream replays.
- Full frame latency, uncapped throughput, opaque cache-miss ratio and bridge-only
  peak memory are not measured. Evidence is bounded CPU/GPU spans, command counts,
  cache comparisons and whole-process RSS.

## References

Implementation comments link sources beside code. Pin ABI sources on changes;
repository code is authoritative. Dependency upgrades need renewed interop review.

[deko-registers]: https://github.com/devkitPro/deko3d/blob/master/source/maxwell/engine_3d.def
[deko-bind]: https://github.com/devkitPro/deko3d/blob/master/source/cmd_bind_common.cpp
[deko-dksh]: https://github.com/devkitPro/deko3d/blob/master/source/dksh.h
[uam-tess-mode]: https://github.com/devkitPro/uam/blob/master/source/compiler_iface.cpp#L491-L528
[uam-barriers]: https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_lowering_nvc0.cpp#L831-L835
[wgpu-encoding]: https://github.com/gfx-rs/wgpu/blob/v30.0.0/wgpu-core/src/command/mod.rs
[wgpu-transitions]: https://docs.rs/wgpu/30.0.0/wgpu/struct.CommandEncoder.html#method.transition_resources
[wgpu-conv]: https://github.com/gfx-rs/wgpu/blob/v30.0.0/wgpu-hal/src/vulkan/conv.rs
[wgpu-adapter]: https://github.com/gfx-rs/wgpu/blob/v30.0.0/wgpu-hal/src/vulkan/adapter.rs
[wgpu-wrap]: https://docs.rs/wgpu/30.0.0/wgpu/struct.Adapter.html#method.create_device_from_hal
[spirv]: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
[vk-shaders]: https://docs.vulkan.org/spec/latest/chapters/shaders.html#shaders-tessellation-control
[vk-tessellation]: https://docs.vulkan.org/spec/latest/chapters/tessellation.html
[vk-lines]: https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineRasterizationLineStateCreateInfo.html
[vk-fma]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_shader_fma.html
[vk-float-controls]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_shader_float_controls.html
