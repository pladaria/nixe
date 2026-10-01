# Deferred rendering and tile-local optimization

Status: proposal for future analysis and implementation. This document does not
declare these optimizations implemented or measured. Recheck the code before
starting; it remains the source of truth.

## Objective and recommendation

Execute guest multipass rendering efficiently without changing its observable
results. Keep wgpu as the primary backend. Use the existing isolated native
interop path only for capabilities that ordinary wgpu cannot express and only
when correctness and measurements justify the additional implementation.

Start with the ordinary path: measure existing pass batching, eliminate redundant
attachment loads/stores where provably safe, and preserve resident images across
producer/consumer passes. Consider native framebuffer-local reads afterward.
Do not implement a software simulation of Maxwell's physical tile cache.

An optimal implementation is host-dependent. Fewer external-memory transfers may
help a tile-based GPU considerably, while a desktop GPU may benefit more from
lower CPU submission overhead. Neither native Vulkan nor fewer logical passes
guarantees an improvement.

## Workload and terminology

Deferred shading separates geometry from lighting:

1. Geometry draws write several color attachments (the G-buffer), plus depth.
2. A later pass reads those attachments and computes lighting into the final
   framebuffer.
3. Intermediate attachments may be discarded once their contents are no longer
   needed.

The local reference workload is
`dev-tools/deko_examples2/source/Example08_DeferredShading.cpp`. It writes three
RGBA16F attachments containing albedo, normal and view direction, then uses
integer-coordinate texture fetches at the current fragment position for
composition. It issues tile barriers and explicit attachment discards.

This is not the same as tiled or clustered *lighting*, where lights are assigned
to screen regions, often using compute shaders. Compute is not required for the
described two-pass workload. Nor does implementing this workload establish
general compute support in Nixe.

Maxwell tiled-cache commands describe guest ordering and caching behavior.
Matching their observable semantics does not require reproducing the guest tile
size, binning implementation or physical cache. However, tile-local ordering is
not permission to replace arbitrary image reads with same-pixel reads.

## Current implementation baseline

Inspect these locations before designing changes:

| Area | Existing implementation | Implication |
| --- | --- | --- |
| Draw lowering | `crates/gpu-maxwell/src/engines/lowering.rs` emits begin/draw/end operations; draw attachments normally use Load/Store | A per-draw lowered pass is not necessarily a separate host pass |
| Pass batching | `append_batchable_operations`, `batchable_render_pass`, `render_passes_can_merge` in `crates/gpu-maxwell/src/execution.rs` | Adjacent compatible passes are already merged; extend or replace this mechanism, do not add a second batcher |
| Ordering | `three_d_synchronization_operation` in the same file | Pixel/tile barriers currently become conservative ordered-write boundaries |
| Discard | `MethodAction::DiscardRenderTarget` in `crates/gpu-maxwell/src/engines/threed/mod.rs` | Currently retains contents, which discard permits; does not propagate a store-elimination opportunity |
| Ordinary encoding | `encode_submission`, `encode_render_pass`, `encode_clear` in `crates/gpu-wgpu/src/driver.rs` | Encodes MRT passes and separate clears; wgpu owns ordinary resource transitions |
| Image reuse | Resource preparation and materialized-image tracking in `draw.rs` | Rendered images can be reused for sampling without canonical-memory round trips |
| Shader reads | `LoadTexture2D` in `crates/gpu/src/shader.rs` and WGSL lowering | General integer-coordinate fetch, not a framebuffer-local input declaration |
| Native interop | `crates/gpu-wgpu/src/native/` | Existing Vulkan integration to reuse, not a reason to create another device or backend |

The current batcher requires matching render-pass identity and matching ordered
attachment descriptions, including resources, subresources, formats and sample
counts. The previous store and next load/store must preserve contents. Backend
operations interrupt batching; canonical-memory work can split execution into
segments. Measure actual emitted passes before attributing overhead to per-draw
lowering.

In ordinary encoding, neutral barrier/cache-maintenance markers do not trigger a
CPU wait or a manual physical cache flush. Their sequence preserves boundaries;
wgpu tracks resource usages and inserts host synchronization. This does not mean
all barriers can be removed or moved inside a pass: changing pass structure also
changes the usage scopes available to wgpu.

Existing MRT pipeline identity includes ordered attachment formats and active
per-target output state. Preserve the prepared-draw fast path and avoid rebuilding
pipelines, bind groups or textures on every frame.

## Correctness requirements

All transformations must preserve:

- Command order where observable, including semaphore/query results, CPU-visible
  writes, presentation, resolves and native/ordinary execution boundaries.
- Canonical backing identity, mapping generations, overlapping aliases, image
  subresources, formats, sample counts and materialization validity.
- Blending, write masks, depth/stencil behavior, color-space conversion, clear
  precision, sample selection and shader-visible values.
- Resource lifetime through asynchronous execution, eviction and slot reuse.

Do not change guest G-buffer precision, pack normals differently, simplify
lighting, skip valid barriers or approximate cross-pixel reads for performance.
Do not identify optimization candidates by ROM name, shader hash allowlists or a
particular command sequence. Recognize semantics.

An unproven optimization uses the existing correct ordinary path. An unsupported
guest semantic still requires a precise failure; optimization fallback must not
hide missing emulation behavior.

## Ordinary wgpu improvements

### 1. Audit and extend existing batching

Instrument actual batch boundaries in a dedicated benchmark or capture tool.
Classify why passes split: attachment changes, identity changes, clears,
synchronization, resolves, canonical writes or native execution.

Extend `render_passes_can_merge` only when evidence identifies a material missed
opportunity. Equal framebuffer descriptions alone are insufficient: check the
entire proposed host usage scope, especially images written as attachments and
read as textures, overlapping resource views, and storage read/write hazards.
Two individually legal passes need not form a legal single wgpu pass.

Keep the initial optimization within the existing submission/segment boundaries.
Cross-submission grouping is separate work because it can delay completion,
change resource ownership and increase latency. Do not add it merely to produce
larger batches.

Use bounded comparisons for attachment slots and existing indexed resource
identities. A linear walk over commands with constant-bounded attachment work is
preferable to rescanning prior draws. Do not require a general frame graph or
duplicate scheduler for a local transformation.

### 2. Fold equivalent full clears into attachment loads

Replace a separate image clear followed by a compatible pass with an attachment
load clear when the operations are equivalent. This can remove a host pass and
avoid loading discarded prior contents.

Eligibility must cover the exact image view, mip/layers, aspects, render extent,
sample count and clear value conversion. No intervening observer or overlapping
alias may need the pre-clear contents or observe a different ordering. A masked,
scissored or partial clear cannot become an unconditional whole-view load clear.
Depth and stencil preservation must be considered independently.

Transfer dependencies, access records and initialization/materialization effects
with the clear. Removing its command must not remove the evidence that a
compressed resident image has valid contents. Standalone and ineligible clears
remain supported by the existing encoder.

### 3. Propagate discard opportunities safely

A guest discard relinquishes preservation; it is not a clear, deallocation,
unbind or synchronization operation. Retaining contents is valid and remains the
fallback when an earlier store cannot safely be eliminated.

Preserve enough discard intent to eliminate a store only when no consumer between
that store and the discard needs the contents. In deferred rendering the lighting
pass reads the G-buffer *before* its final discard. On the ordinary two-pass path,
that producer store generally remains necessary. The depth attachment may have a
different lifetime and permit an earlier store to be discarded.

Track the relevant image subresources and aliases, not only a currently bound
render-target slot. Never infer that an image is dead because it is unbound, absent
from the next draw, or named as a transient buffer by the application. Do not mark
discarded compressed storage as initialized or turn undefined contents into
invented zero data.

Avoid unbounded lookahead: begin with an already available bounded command
segment. Maintain a last-use record keyed by existing canonical resource identity
where useful; resolve overlapping aliases through the existing memory tracking,
not a new scan of every resident image per draw. If a segment has already been
submitted, do not wait or resubmit merely to exploit a later discard.

### 4. Preserve the existing fast resource path

Check that repeated geometry/composition cycles retain the same resident images
and prepared pipelines. Eliminate any measured redundant lookups at their owner,
using generation-aware indexed access where possible. Cache keys must distinguish
all relevant formats, subresources and shader variants; a fast stale hit is not an
optimization.

There should be no G-buffer CPU readback, re-upload, forced queue-idle wait or
per-frame descriptor/pipeline compilation in steady-state rendering. Readback for
explicit screenshots and test assertions is a separate operation.

## Optional native framebuffer-local path

### Capability and tradeoff

The public wgpu 30 render-pass descriptor has attachments but no subpass/input
attachment interface. Ordinary MRT and subsequent texture reads are supported;
framebuffer-local pass grouping needs a separate capability assessment. Recheck
the pinned API before implementation. [wgpu RenderPassDescriptor](https://docs.rs/wgpu/30.0.1/wgpu/struct.RenderPassDescriptor.html)

Vulkan subpasses/input attachments are one candidate. Dynamic rendering with
local reads is another, subject to the actual device features and backend
integration. Neither guarantees physical tile-memory retention; attachment count,
formats and driver decisions affect the outcome. Khronos demonstrates why
subpasses can reduce bandwidth on tile-based GPUs, not a universal speedup for
Nixe. [Khronos subpass sample](https://docs.vulkan.org/samples/latest/samples/performance/subpasses/README.html)

Dynamic local reads also require explicit attachment/input mappings, layouts and
dependencies. Do not treat the extension as an automatic rewrite of existing
sampled-image descriptors. [VK_KHR_dynamic_rendering_local_read](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_dynamic_rendering_local_read.html)

### Proof before specialization

Candidate consumers must read the matching producer pixel and supported
layer/sample/mip, with equivalent coordinate orientation and format conversion.
The current `LoadTexture2D` operation alone does not prove this: its coordinates
can be arbitrary. Prove eligibility from shader dataflow and attachment bindings
at translation/preparation time, not by examining every fragment or recognizing a
specific shader binary.

Reject specialization for unproven coordinates, neighbor reads, filtering,
incompatible LOD/layer/sample selection, unsupported feedback, and externally
observable side effects that cannot be preserved. Start with single-sample
same-pixel reads; prove MSAA behavior separately. Retain ordinary sampled-image
execution for general reads.

If needed, extend the neutral contract with the minimum information for
framebuffer-local dependencies and attachment lifetime. Maxwell-specific register
decoding stays in `gpu-maxwell`; Vulkan layouts, handles and subpass indices stay
in `gpu-wgpu`. A tile barrier is useful evidence, not by itself a complete proof.

Use a specialized shader variant with input-attachment/local-read operations and
the appropriate SPIR-V interface. Include specialization and attachment mappings
in shader/pipeline cache identity. Audit the current native emitter's supported
operations before assuming the composition shader can run there unchanged.

### Interop ownership

Reuse existing native resource ownership, command segmentation, cache and
completion/lifetime machinery. Native grouping must own the full producer and
consumer group; an ordinary wgpu pass cannot be nested inside a raw Vulkan render
pass. This may require native encoding of ordinary graphics draws in that group,
not merely inserting one Vulkan barrier.

Explicitly establish incoming/outgoing layouts, access dependencies and wgpu's
subsequent resource state. Keep resource usage flags compatible with both paths.
Do not create a second device, copy the G-buffer into a parallel resource registry,
or add CPU waits at every native boundary. Validate mixed ordinary/native chains
with core and synchronization validation.

Transient/lazily allocated attachment storage is only a candidate when all uses
and host capabilities permit it. An image later sampled outside the group, copied,
presented or read through canonical memory cannot lose its required contents.
Ordinary wgpu remains a supported path on other backends and on ineligible Vulkan
workloads. Backend-specific optimization paths are legitimate; obsolete prototype
layers are not.

## Measurement and regression plan

Establish a release baseline before editing production behavior. Separate cold
shader/pipeline compilation from warm execution. Record hardware, driver, backend,
resolution, sample count, power mode and presentation pacing. Do not compare
handheld and docked resolutions as equivalent workloads.

A 60 FPS presentation cap conceals rendering headroom. Use a controlled replay or
benchmark without that limit, or measure CPU work and GPU execution inside paced
frames. Do not change guest timing to manufacture a performance result.

Measure:

- CPU time in lowering, preparation, pass encoding and submission; allocations and
  cache misses in warm execution.
- Actual host pass count, draws per pass, standalone clears, loads/stores,
  resource creations and native/ordinary boundaries.
- GPU time for geometry, composition and the whole group; aggregate bandwidth or
  cache counters only when available. Estimated bytes are not measured traffic.
- Frame-time distributions and repeated-run variance, not just average FPS.

Use test-only timestamp queries/readbacks, external profiling, and RenderDoc
captures where useful. Keep capture overhead out of benchmark comparisons.
Tools belong in `dev-tools/`; temporary production traces should be removed after
validation, with no unconditional per-frame measurement overhead left behind.

Focused tests must include:

- Existing batching and its rejected merge cases, including resource hazards.
- Full versus partial/masked clears, multiple mip/layers, depth/stencil aspects,
  MSAA and compressed-image materialization.
- Discard after sampling, alias reads before discard, remapping, CPU writes,
  eviction, resource generation reuse and repeated submission cycles.
- MRT formats, per-target blending/write masks and attachment order; extend the
  production-path `tests/accelerated/multiple_targets.rs` coverage as appropriate.
- Same-pixel eligible shaders and deliberately ineligible offset/LOD/layer reads;
  compare native and ordinary paths with an independent pixel oracle.
- Mixed native/ordinary chains, queued frames, presentation and explicit readback.

GPU tests use a physical adapter and report a skip when none is available. A skip
is not correctness or performance evidence. Require zero validation errors on
tested configurations and preservation of existing graphics regressions.

## Implementation sequence and acceptance gates

1. **Baseline and audit.** Capture the actual pass graph and profile representative
   single-target, MRT and multipass workloads. Identify redundant boundaries and
   verify the current reuse path. Select a measurable bottleneck.
2. **Ordinary-path changes.** Extend the existing batcher and add clear/discard
   transformations incrementally where justified. Each change requires focused
   positive/negative tests and equivalent guest-visible output.
3. **Performance decision.** Compare warm and cold results against baseline.
   Retain changes with demonstrated benefit and no material regressions. If the
   ordinary path meets the target, stop here.
4. **Native feasibility prototype, if justified.** Prove local-read semantics,
   shader translation, device capabilities and safe interop for a bounded group.
   Measure benefit on the intended hardware before expanding support.
5. **Production integration and cleanup.** Integrate the proven path into existing
   ownership/caching, retain legitimate backend fallbacks, remove superseded
   helpers and temporary instrumentation, and document supported semantics.

Before calling the work complete, record the measured improvement, tested devices,
remaining capability limits and why each retained path exists. Do not claim
universal optimality, Maxwell physical-cache equivalence, or native tile retention
from a visually correct frame alone.
