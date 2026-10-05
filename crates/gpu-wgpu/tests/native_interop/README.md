# Native tessellation interoperability and production execution tests

This opt-in suite covers public wgpu 30 / HAL Vulkan interop and patch execution
through the production neutral runtime. Device creation/import uses production
`initialize_backend`. Low-level fixtures isolate handoff behavior; production
cases exercise the real resource table, visibility, submission and resident
presentation. Neither substitutes for a complete guest command-stream replay.

The GLSL files are original host fixtures compiled only by the test, not
replacements for Maxwell shaders. Emitted-IR fixtures execute Nixe's own
VS/TCS/TES/FS SPIR-V; a separate captured-SASS oracle is described below.
Demo acceptance, Switch comparisons, performance evidence and unsupported modes
are recorded in the [tessellation spec](../../../../docs/specs/tesselation-support.md).
`ash` is a production dependency matching HAL; `wgpu-types` is test-only.

## Running

Requirements:

- A physical Vulkan GPU exposing `tessellationShader`. Software rasterizers,
  virtual GPUs and unknown adapter types are excluded.
- Recent Khronos validation layers. Use **1.4.341 or later** for devices enabling
  `VK_KHR_shader_fma`; 1.4.313 predates that extension and rejects its feature
  chain. The 1.3.204
  layer distributed with Ubuntu 22.04 rejects newer structures and WGSL source
  metadata emitted by wgpu 30; do not filter those errors to obtain a pass.
- `glslangValidator` and `spirv-val` in `PATH`, or absolute executable paths in
  `GLSLANG_VALIDATOR` and `SPIRV_VAL`. The four fixture modules are compiled and
  validated for Vulkan 1.1 in a temporary directory.

```sh
cargo test -p nixe-gpu-wgpu --test native_interop -- --ignored --nocapture
```

Use the Vulkan loader's `VK_DRIVER_FILES` (or older `VK_ICD_FILENAMES`) to select
a driver when testing multiple implementations. For unpacked validation-layer
packages, set `VK_LAYER_PATH` to their manifest directory and `LD_LIBRARY_PATH`
to their library directory. No system-wide installation is required. Do not use
layer settings that disable core or synchronization checks.

The test requires the validation layer and its `VK_EXT_validation_features`
extension. wgpu-hal 30 explicitly enables synchronization validation through that
extension when creating the instance. The test captures backend error logs and
fails on any error, including destruction-time diagnostics. Normal `cargo test`
reports this test as **ignored**, not as a passing hardware test. An explicit run
prints `SKIP:` and returns before validation/tool setup when no physical GPU or
required native raster/tessellation capability is available. Rust's standard
test harness reports these early returns as `ok`; use `--nocapture` and do not
count a `SKIP:` result as hardware validation. With a compatible GPU, missing
validation/tools, initialization errors, validation errors and pixel mismatches
fail the test; they are not converted into skips.
Run this test in a debug build: it asserts that production's default instance
flags enable validation instead of silently running an unchecked release test.

Local tools can be retained in the ignored `dev-tools/` directory. With the
Vulkan-ValidationLayers v1.4.341 source build and unpacked glslang package there,
run from the repository root using:

```sh
export VK_LAYER_PATH="$PWD/dev-tools/Vulkan-ValidationLayers/build/layers"
export LD_LIBRARY_PATH="$VK_LAYER_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SPIRV_VAL="$PWD/dev-tools/Vulkan-ValidationLayers/external/Release/64/SPIRV-Tools/build/tools/spirv-val"
export NIXE_SPIRV_VAL="$SPIRV_VAL"
export GLSLANG_VALIDATOR="$PWD/dev-tools/glslang/usr/bin/glslangValidator"
```

These paths do not install system packages or change normal emulator startup.
Only the test shell needs these variables; the tools are not runtime dependencies.

Additional physical vendors remain unverified.

## Low-level fixture coverage

| Case | Oracle |
| --- | --- |
| HAL-created device with native tessellation enabled, wrapped by wgpu | Actual TCS/TES pipeline creation and execution; no second device or queue |
| wgpu clear/draw → native tessellation → wgpu draw/sample | Every output pixel's RGB and depth (encoded in alpha) |
| Upload/copy and compute writes → TCS buffer / TES sampled image | Green vs blue output; either a stale buffer or a stale image changes the expected result |
| Native reads → later normal copy/compute writes → native reads | Inputs rewritten within the ordered command sequence, without a host wait |
| Consecutive native segments with identical tracked usage | Core/sync validation and pixels; explicit native boundary dependencies |
| Consecutive native draws in one segment/pass | Two draws share one binding setup, pass, and boundary barrier pair |
| Prior normal depth, native depth write, later normal depth test | Near red strip survives native drawing; a normal draw behind native output fails depth; nearer blue strip succeeds |
| Fresh native render destination | One legitimate allocation-time initialization, native first draw, subsequent normal sampling without lazy zero erasing the result |
| Independent subresources | Layer 0 retains magenta and depth 0.75 while native work uses only layer 1 |
| Submission boundaries | Both one ordered `Queue::submit` and a split at a native boundary |
| Frames/resources replaced without waiting | Six generations, queued before the readback checks; full neutral resource handles key the fixture cache |
| Cache eviction and lifetime | Bindings/attachments/pipeline retained through completion callbacks; weak references expire afterward |
| Optional native feature disabled | Separate ordinary wgpu device still clears, draws, and samples correctly |
| Nixe-emitted VS → TCS → TES → FS | Arrayed component linkage, invocation indexing, indexed constant-buffer reads and interpolated colors reach checked GPU pixels |
| One live binding shared by VS, TCS, TES and FS | Layout visibility comes from the emitted chain; each stage's read contributes to checked color/depth changes |
| Dead descriptor declarations | The IR fixture's native layout contains only emitted bindings; default control needs no descriptors |
| Emitted patch barrier and cross-invocation output read | All TCS invocations read invocation two's output after `PatchBarrier`; changing its buffer word changes the rendered color |
| Generated default TCS, patch sizes 1, 4, 5 and 32 | TES reads the last incoming point, verifying cardinality and unchanged per-point forwarding |
| Default levels supplied as 24-byte push constants | One pipeline per patch size handles changing levels, zero-level culling and recovery; TES reads both fixed-function and additional levels |

The raw fixture's generation-keyed cache is test-owned, not an alternate production
resource table. Its buffer/image ABI and subresource cases isolate HAL behavior
beyond the production descriptor subset. The production-runtime cases below use
Nixe's actual canonical-memory visibility, residency and presentation accounting;
`accelerated/resource_lifetime` also covers production image slots.

## Tested handoff contract

### Polygon facing acceptance (Phase D)

Run `wgpu_native_polygon_facing` with the same Vulkan/core/sync validation setup.
It checks front/back/both/no culling for both TES winding modes and both viewport
Y signs, in-pass culling changes, cache eviction/retention and subsequent ordinary
partial clears. It passes on NVIDIA. Winding is derived from the
lower-left neutral tessellation domain and the actual viewport determinant, not
assumed to equal the final framebuffer orientation. The ordinary
`accelerated_polygon_facing_preserves_winding_for_triangles_and_quads` pixel test
checks the corresponding wgpu path. These host-facing tests do
not establish full Maxwell/Switch raster equivalence.

### Native wireframe acceptance (Phase D)

Run `wgpu_native_wireframe_rasterization` explicitly with the same Vulkan/core/sync
validation setup. It consumes the optional native rectangular/smooth line features
and checks width 1/4, coverage alpha, unchanged interiors/background, width changes
within a pass, zero color masks, native fill/wireframe transitions and later normal
partial clears. It passes on NVIDIA with core/synchronization validation.
This test alone does not establish Maxwell/Switch smooth-wireframe equivalence.

### Existing interop and color-output acceptance

The production-runtime color-output scenario additionally tests RGB/alpha blending,
component masks (including no writes), reverse subtraction and min/max over a
nonzero destination. It changes masks inside one pass without changing the logical
pipeline or shaders, switches equations between submissions, and verifies that a
later ordinary partial clear ignores the native draw's mask. Independent pixel
expectations pass on NVIDIA with core/synchronization validation.
The ordinary `accelerated_blending_and_write_masks_preserve_destination_components`
test exercises the same neutral semantics through wgpu rather than the native path.

1. Inputs and attachments are initialized by legitimate wgpu uploads/clears.
   A native first draw on a fresh allocation is supported **after** this explicit
   allocation-time initialization. There is no public API used here to mark an
   arbitrary raw write initialized after the fact. Uninitialized native reads and
   native writes followed by unaware wgpu lazy initialization are not supported.
   Preserved guest contents must be uploaded/materialized, not overwritten with
   this fixture's clears. No recurring hidden clear is part of the protocol.
2. A normal encoder declares `STORAGE_READ_ONLY` for the TCS buffer, `RESOURCE`
   for the TES image, `COLOR_TARGET` for the color subresource, and
   `DEPTH_STENCIL_WRITE` for depth. These establish the layouts known to wgpu.
3. An otherwise unused encoder records only native commands via `as_hal_mut`.
   Entry/exit memory dependencies cover TCS/TES explicitly, including read→write
   execution hazards. Native render passes preserve the declared attachment
   layouts. No native queue submission, manual command-buffer end, or imported
   duplicate resource allocation is used.
4. Normal encoding resumes on another encoder. wgpu sees precisely the usages
   and layouts declared before the raw segment, not an invented final state.
5. All segments are submitted in order through the same wgpu queue. The fixture
   also splits a sequence to exercise cross-submission dependencies. Queue order
   alone is never treated as a memory dependency.
6. Native framebuffers/descriptors own clones of their backing textures, views,
   and buffers; submitted uses retain these owners and their pipeline until the
   completion callback. Logical eviction removes only the cache reference.
   Explicit `Buffer::destroy`/`Texture::destroy` must not bypass this retention.
   Imported-device ownership includes failure cleanup. The separate opt-in unit
   test `imported_device_drops_after_queue_and_resources_without_an_ownership_cycle`
   uses a test-only destruction counter to detect leaked ownership cycles.

The fixture uses a conservative **segment-level** union of its known normal
producer/consumer stages and accesses; no `ALL_COMMANDS` barrier or per-draw
handoff is used. The production bridge derives descriptor stages from the emitted
ABI, registers read-only buffer usages once per segment and batches compatible
native draws. Fixture masks are not a claim of optimal production synchronization;
the command-capture workflow below measures actual emitted barriers.

All CPU waits and readbacks occur in the final oracle/cleanup phase. Copies used
to upload fixture input and read back the oracle are not interop framebuffer
copies. No external shader compiler belongs in the production draw path.

References reviewed alongside the executable proof:

- [wgpu encoding API separation](https://github.com/gfx-rs/wgpu/blob/v30.0.0/wgpu-core/src/command/mod.rs)
- [HAL device setup](https://github.com/gfx-rs/wgpu/blob/v30.0.0/wgpu-hal/src/vulkan/adapter.rs)
- [HAL validation setup](https://github.com/gfx-rs/wgpu/blob/v30.0.0/wgpu-hal/src/vulkan/instance.rs)
- [HAL stage/access conversion](https://github.com/gfx-rs/wgpu/blob/v30.0.0/wgpu-hal/src/vulkan/conv.rs)
- [Vulkan execution and memory dependencies](https://docs.vulkan.org/spec/latest/chapters/synchronization.html#synchronization-dependencies)

## Results and limits

Validated with Khronos layers 1.4.341:

- **NVIDIA GeForce RTX 4070 Ti SUPER**, driver **595.58.03**: pixel/depth,
  lifecycle, core-validation and synchronization-validation checks pass.

AMD/Intel hardware validation remains required before claiming general interoperability.
The ordinary-device case disables the optional feature on a capable adapter; it
does not replace testing an adapter that lacks the capability physically.

The emitted-IR fixture deliberately contains no float arithmetic. It validates
patch execution and synchronization without inventing unsupported numerical
capabilities. It is not execution of the complete guest TES or evidence of its
float32 equivalence. The captured-chain and arithmetic oracles below cover those
separate contracts; the spec documents their scope and numerical requirements.

The default-control fixture additionally executes the generated adapter with
dynamic level words, without descriptor allocation or shader recompilation for
level updates. It passes on NVIDIA 595.58.03 with Khronos
1.4.341 validation. Its four modules are checked with SPIRV-Tools before execution.
The low-level fixtures remain distinct from the production-runtime checks below.

Both emitted-IR fixtures use the combined chain compiler, including interface
linking, TCS output cardinality and live descriptor ABI. The raw fixture derives
its descriptor layouts/writes from that ABI. The separate GLSL interop fixture
retains its own buffer/image ABI for upload, sampling and lifetime tests.

`production.rs` additionally submits neutral commands through the real runtime:
ordinary clear -> emitted patches -> ordinary partial clear -> resident export.
It checks cached explicit/default control shaders, distinct input/output patch
sizes, staged vertex streams, nonzero vertex/instance/binding offsets, instancing,
trailing incomplete patches, depth, changing default levels, patch suppression,
ordinary store-discard followed by raw reuse, queued work, cache eviction with
submitted users retained, and direct runtime drop.
These pass on NVIDIA with core/synchronization validation enabled.
The production resource case reads a shared constant buffer in VS/TCS/TES/FS,
checking changed RGB/depth after canonical-memory updates across submissions.
It then replaces resources with a one-entry descriptor cache and queues 40
submissions before waiting, exercising pool-page rollover and retained evicted
sets. Descriptors are immutable; buffer-content updates do not enter their cache
identity. These tests also pass on NVIDIA with Khronos 1.4.341.
Two additional production cases cover resource lifetime and CPU-backed aliasing:

- Eight submissions retire and recreate the same buffer, descriptor-table and
  color-image IDs. The explicit completion waits permit logical-ID reuse; they
  are test lifecycle steps, not waits added to production native execution.
- Eight queued submissions alternate two buffers/descriptors over the same
  canonical page. Each alias consumes both payloads on successive uses, so an
  initial upload alone cannot pass. Four descriptor-cache entries permit both
  aliases to stay cached; checked color/depth distinguish stale contents.
  This case preserves the explicit color clear each frame, independently of the
  existing store-discard case.

These cases pass on NVIDIA with Khronos 1.4.341 core/sync validation.
Images/writable shader descriptors remain
explicitly unsupported; the supported boundary is documented in the main
spec. Buffer oracles use valid in-bounds accesses, not guest out-of-range behavior.

The separate ignored unit test
`native_binding_keys_distinguish_residency_recreation_and_slot_reuse` uses the real
residency-budget eviction/recreation path with a retained old WGPU buffer. It
checks that physical recreation changes the native binding key without changing
the logical handle, and that slot reuse changes its generation. It passes on
NVIDIA. A test-only driver observer delegates real submission; there
is no production test hook, configurable residency policy or extra runtime check.

```sh
cargo test -p nixe-gpu-wgpu --lib \
  native_binding_keys_distinguish_residency_recreation_and_slot_reuse -- --ignored --nocapture
```

The identity test is distinct from rendering. A combined production case
fills/touches the current 4096-object residency budget with zero-byte
allocations between native draws. Subsequent draws reuse the logical buffer with
changed canonical contents after LRU eviction/recreation. Pixel copies before
and after pressure are checked without an explicit intervening wait, exercising
cached-descriptor invalidation and retention of prior submitted resources. No
production eviction hook or alternate budget was added.

The built-in case checks nonzero vertex/instance bases, TCS invocation indexing,
four TCS input versus three TES input points, patch IDs in TCS/TES and all three
tessellation coordinates. It also changes pipeline/viewport within a pass; the
default-control cases change levels within a pass. These guard suppression of
redundant raw bindings while preserving real state changes.

Two backend instances with a shared temporary cache directory render the same
checked sequence before and after native pipeline-cache persistence. The separate
ignored unit test `native_pipeline_cache_roundtrip_and_small_bound` exercises
bounded driver extraction and reload. File compatibility/integrity cases are
ordinary unit tests. These GPU cases pass on NVIDIA with the same
validation setup. Persistence saves driver data, not native object handles or
translated SPIR-V. The cold/persisted measurements below report whole-submit CPU
time, not isolated driver compilation or cache-hit feedback.

The production indexed cases exercise uint16/uint32 patch lists with both
explicit/default TCS, reordered points, binding and first-index offsets, positive
and negative base vertices, instancing, canonical index updates, partial trailing
patches and entirely incomplete patches followed by resumed rendering. Uint16's
`0xffff` remains a regular index (no restart); uint32 indices exceed 24 bits to
exercise the enabled full-range capability. These pass on NVIDIA
with Khronos 1.4.341 validation. No CPU index scan/conversion or readback is part
of production execution. Uint8 indices and narrowly bounded indexed vertex
subranges remain unsupported. The Maxwell frontend still emits nonindexed draws;
these are public neutral-runtime tests, not new indexed Maxwell command decoding.

No end-to-end Maxwell draw, wireframe/smoothing equivalence,
Metal/DX12 native tessellation, or original-Switch comparison is claimed by these
interop fixtures. Complete captured SASS execution has a separate oracle below.

## Warm-path CPU and allocation measurements

Run this test **by name in release**, with validation layers disabled:

```sh
cargo test -p nixe-gpu-wgpu --release --test native_interop \
  native_warm_path_measurements -- --ignored --nocapture
```

It measures ordinary, native, native-32-draw and mixed public-runtime submissions,
plus native/mixed variants with smooth wireframe, width 4, back-face culling and
source-alpha RGB / separate-alpha blending:
64 warm-up samples, 512 CPU timing samples, then 64 calling-thread Rust allocation
samples. Allocation tracking lives only in this test executable and is disabled
for the timing samples. Construction/readback/explicit batch retirement are
excluded. Allocation calls include reallocations; bytes are total requested bytes,
not live/peak memory. GPU/driver C allocations and other threads are not counted.
The 32x32 workloads have different shaders and geometry; the results describe
CPU submission overhead, not relative GPU throughput or FPS. The
smooth/blend variants reuse the native fill baseline's synthetic shaders and
resources; they do not execute the captured guest shader chain. Raster-state
preparation is outside the measured interval. The separate
RenderDoc workflow below measures replay draw timestamps and captures commands;
it does not measure live full-frame GPU latency.
The main spec records representative hardware results and their scope limits.

Set `NIXE_TEST_ALLOC_STACKS=1` on the warm-path command to collect and group
allocation backtraces in one additional, untimed submission per case. Unwinder
and sample-storage allocations are excluded by a recursion guard. This is a
test-executable option, not an emulator option; no tracing code is linked into
`nixe-cli`. The ordinary `gpu_cache_fingerprints_do_not_allocate` test enforces
allocation-free hashing for short and long streaming inputs.

All read-only native buffer transitions are registered as one usage scope
per segment, not per draw/category. The shared-vertex production scenario reads
one buffer through vertex fetch and TCS/TES/FS descriptors, changes the vertex
binding offset within the pass, and checks pixels after canonical updates. It
passes core/synchronization validation on NVIDIA. The native entry
and exit barriers remain in place; fewer registration calls are not themselves
a measurement of emitted driver barriers.

## Cold/persisted pipeline submission measurements

Use a dedicated disposable cache directory, never the emulator's normal cache:

```sh
export NIXE_TEST_NATIVE_CACHE_DIR="$(mktemp -d /tmp/nixe-native-cache.XXXXXX)"
cargo test -p nixe-gpu-wgpu --release --test native_interop \
  native_pipeline_cache_measurements -- --ignored --nocapture
cargo test -p nixe-gpu-wgpu --release --test native_interop \
  native_pipeline_cache_measurements -- --ignored --nocapture
```

Each invocation creates a new device in a new process, checks all fixture pixels
and explicitly persists caches at teardown. `CACHE_MEASURE` reports existing
native files and CPU submit time for each of eight submissions, including the
first cold in-memory pipeline creation. The interval includes resources/uploads,
IR/SPIR-V translation, both host pipeline paths and native cache import; it does
not isolate `vkCreateGraphicsPipelines` or report driver cache-hit feedback.
To isolate the native-file variable, retain the wgpu cache and temporarily move
only `native-vulkan-*.bin` out of its expected name between fresh-process runs.
Other driver/OS caches are not reset, so repetitions and ordering controls are
needed for a causal performance claim. See the spec for the initial comparison.

`measure_cache.py /absolute/path/to/native_interop-test --pairs 6` automates
balanced-order fresh-process comparisons without capture instrumentation. It
creates its own seed and sample caches under a temporary directory, copies the
same wgpu cache into both conditions, and only adds the native file in one.
Each sample checks all fixture pixels. Driver/OS caches remain uncontrolled;
the output is whole-submit CPU time, not a driver cache-hit counter. The tool
does not touch the emulator's normal cache directory.

## Linux command capture and GPU replay timestamps

`NIXE_TEST_CAPTURE_DIR` switches the warm benchmark into capture mode: 64 warm-up
submissions and one captured submission for each of ordinary/native/native32/mixed
and the native/mixed smooth-blend variants.
CPU/allocator results are not reported while the capture tool is attached.
On `native_pipeline_cache_measurements`, the same option captures the first
submission as `cold_capture.rdc` instead. Use a dedicated existing absolute
directory; captures may overwrite earlier outputs with the same name.

Preload RenderDoc **before** the test process starts and make its Vulkan implicit
layer discoverable. Tested with the official RenderDoc 1.46 Linux distribution:

```sh
NIXE_TEST_CAPTURE_DIR=/tmp/nixe-captures \
LD_PRELOAD=/path/to/renderdoc/lib/librenderdoc.so \
XDG_CONFIG_DIRS=/path/to/renderdoc/etc \
ENABLE_VULKAN_RENDERDOC_CAPTURE=1 \
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /absolute/path/to/release/native_interop-test \
  native_warm_path_measurements --ignored --nocapture --exact
```

The distribution's `etc/vulkan/implicit_layer.d/renderdoc_capture.json` must
reference its actual `lib/librenderdoc.so` path, not the package builder's path.
Use a private copy if adjusting it; do not change global Vulkan configuration.
Forcing the layer through `VK_INSTANCE_LAYERS` instead of discovering it as an
implicit layer can bypass its pre-instance extension filter and fail device
creation. Khronos validation should be disabled for measurements, and enabled
separately for correctness tests.

Run the analyzer inside RenderDoc's embedded Python interpreter:

```sh
NIXE_TEST_CAPTURE_DIR=/tmp/nixe-captures \
  qrenderdoc --python crates/gpu-wgpu/tests/native_interop/analyze_capture.py
```

For headless automation, wrap the command in `xvfb-run -a`. RenderDoc's first-run
analytics dialog precedes scripts. A private `XDG_DATA_HOME` containing
`qrenderdoc/UI.config` with the following contents disables telemetry and update
checks without changing user preferences:

```json
{"rdocConfigData": 1, "Analytics_TotalOptOut": true, "CheckUpdate_AllowChecks": false}
```

The analyzer emits JSON command counts, barrier stage masks, copied resource
names, fragment uniform buffers and GPU timestamp sums for native/ordinary
draws. It takes 20 replay counter samples, discards the first three, and verifies
every draw has a finite nonnegative duration. Replay counters serialize work
and omit non-draw costs; sums are not live frame latency, submission overhead or
an isolated barrier measurement. At this tiny size the measured durations are
visibly quantized. Keep CPU timing, replay timing and validation runs separate.
When both native captures are present, it also asserts 2/32 native draws, equal
barrier/pipeline/descriptor-bind counts and one queue submit in each capture.

Creation records include pipeline initial state from before a warm capture;
their presence does not prove a warm cache miss. The cold capture identifies
native versus ordinary `vkCreateGraphicsPipelines` durations, under RenderDoc's
interposition. It cannot establish opaque driver cache hit rates. API references:
[capture ABI](https://github.com/baldurk/renderdoc/blob/v1.18/renderdoc/api/app/renderdoc_app.h),
[in-application capture](https://github.com/baldurk/renderdoc/blob/v1.46/docs/in_application_api.rst),
[structured metadata](https://github.com/baldurk/renderdoc/blob/v1.46/renderdoc/api/replay/structured_data.h).

All capture code and its loader dependency belong to the test target. Nothing
is added to `nixe-cli` or to the production driver's hot path.

## Captured guest shader-chain oracle

The test beside `gpu-maxwell`'s existing patch instruction fixtures translates the
complete captured VS/TCS/TES/FS chain and submits it through the production neutral
backend, with canonical-backed vertex inputs and resident presentation. Six queued
frames vary colors and viewport orientation; an independent barycentric pixel
oracle checks interiors/background with a two-step UNORM8 tolerance. Readback only
occurs for the final oracle. The linked emitter removes unconsumed TES generic
components before arithmetic/resource liveness, without altering the guest IR.

The same test also runs an independently compiled TCS from
`gpu-maxwell/src/shader/patch_address/barriers.tesc`, with the captured VS/TES/FS.
UAM 1.1.0 removes all three source `barrier()` calls; the actual Maxwell fixture
contains predicated `AST.P` and unconditional `ALD.O.P`. The frontend reconstructs
RAW/WAR output ordering, and every TCS invocation must preserve the first shared
value (0.25) across its overwrite with 0.75. All interior pixels must be
`[64, 191, 64, 255]` within the same tolerance, in both viewport orientations.
The adjacent Rust fixture documents reproduction with UAM/nvdisasm; neither tool
is a build/test/runtime dependency. The ordinary SPIRV-Tools oracle also validates
both shader chains with and without native float32 denormal preservation.

A third run combines the original captured shaders with smooth wireframe,
source-alpha RGB / separate-alpha blending and back-face culling. Six queued
frames rotate the vertex colors, alternate line widths 1/4, and exercise both
viewport orientations with the corresponding front-face setting. Its pixel oracle
checks interior mesh edges and holes, partial edge coverage, spatial color
gradients, untouched background, and increasing integrated coverage with width.
On transparent black, the one-hot vertex colors preserve `sum(RGB) == alpha`
through blending, including overlapping edges; the oracle checks that invariant
within six UNORM8 steps to catch missing blending or squared coverage alpha.
Gradient tolerances account for line width and UNORM rounding. These are semantic
invariants, not a pixel-exact Maxwell edge-coverage golden image.

```sh
cargo test -p nixe-gpu-maxwell --lib \
  captured_guest_chain_executes_through_production_native_backend -- --ignored --nocapture
```

Select ICD/layer paths as above. Requires native tessellation, float32 FMA and the
actual numerical profile consumed by the captured TES (including float64 underflow
repair on the tested device). Missing physical GPU or required capabilities
produce `SKIP:`; missing validation/tools and execution errors fail the test.
Passes on NVIDIA 595.58.03 with Khronos 1.4.341 core/sync validation
without errors. The independent patch-communication fixture leaves the captured
VS color output unused and receives benign `Shader-OutputNotConsumed` performance
warnings; the original full captured chain does not. No glslang executable is
used in this test.
This is the original shader chain with controlled test vertices in fill and
smooth/blended wireframe modes,
not the whole guest demo or a Switch raster-equivalence test.

## Native arithmetic oracle

The separate ignored unit test `native_float_conversions_and_daz_ftz_match_ir_bits` executes
the neutral emitter's addition, multiplication, FMA and signed/unsigned integer
conversion to float32 against an `R32Uint` target,
comparing 65,536 results per operation bit for bit with the IR evaluator. It uses
the production native device constructor, a test-only passthrough shader request,
and readback only for its oracle. Run in a debug build with Khronos validation
**1.4.341 or newer** and SPIRV-Tools supporting `SPV_KHR_fma`:

```sh
NIXE_SPIRV_VAL=/path/to/spirv-val \
cargo test -p nixe-gpu-wgpu native_float_conversions_and_daz_ftz_match_ir_bits -- --ignored --nocapture
```

Select the ICD/layer paths as for the interop test. The numerical test requires
float64 RTE and signed-zero/Inf/NaN guarantees for underflow repair. Addition and
multiplication and FMA pass on NVIDIA 595.58.03. FMA is skipped on devices
without `shaderFmaFloat32`.
These checks do not benchmark the generated repair branches or exercise a complete
guest tessellation workload. No numerical feature is assumed from the vendor ID.
Both integer conversion variants pass on NVIDIA, including signed
limits and halfway rounding cases. Conversion cannot produce subnormals and uses
native conversion instructions without the arithmetic underflow-repair path.

The numerical oracle also checks binary16 packing and unpacking in native SPIR-V
and WGSL. Unpacking covers all 65,536 half encodings in both lanes; packing
covers signed zeros, infinities, NaNs, subnormal and overflow boundaries,
nearest-even ties, and randomized float32 significands/exponents.
