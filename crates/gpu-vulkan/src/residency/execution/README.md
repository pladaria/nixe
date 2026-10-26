# Native execution

The recorder consumes prepared Vulkan objects, not `OperationSubmission`,
`ResolvedBackendResources`, a generic driver, WGSL, or a HAL encoder. Game batch
integration and steady-state grouping use this same API, independently executable
and tested offscreen.

Prepare a pipeline on the compilation pool, then prepare its program, graphics
parameters, attachments and vertex/index streams when those inputs change.
`prepare_program` resolves resident IDs and access metadata. Consumption resolves
native descriptor payloads through the growing layout banks.
SPIR-V reflection determines buffer/image read and write scopes, including actual
shader stages. Simultaneous overlapping bindings are unioned before recording.
A caller cannot omit shader writes from native hazards or canonical publication.
Storage windows are conservatively preserved and published in full, including
bytes a conditional store leaves unchanged. The frontend can supply proven bounded
windows; recording never scans shader code to discover accesses.

`draw` and `dispatch` consume those references directly. They use the same ordered
buffer/image access tracker, canonical producer index, queue and completion leases
as transfers. Dynamic state aliases resolve once at device initialization. Changed
graphics parameters emit setters; repeated parameters compare an Arc identity.
Pipelines and immutable descriptor records remain leased through GPU completion.
Linked executable pipelines also retain their library components, independently
of compiler/cache lifetime. Vertex/index and attachment preparation pins resident
objects; eviction cannot invalidate a live prepared reference.

Read-only draws with compatible LOAD/STORE targets retain a dynamic-rendering
instance across frontend deliveries, including changed pipeline parameters,
descriptors and physical vertex bindings. Vertex/index/shader reads cannot alias
attachment writes. Small CPU-authored shader/vertex inputs can consume immutable
mapped stream slices without a transfer; GPU-authored inputs retain native
imports. Transfers, real hazards, shader writes, changed targets and explicit
ordering terminate rendering. Push, vertex/index, pipeline, descriptor and
graphics-state caches reset on every new command buffer. Compatible parameters
also avoid setters when only bindings changed.

The first draw in a compatible scope publishes its attachment coverage. Continued
draws advance each image's content version and producer ticket while retaining
that already published coverage. This avoids repeated page walks without delaying
ownership until the end of rendering. CPU demands and native imports terminate
the scope before consuming or changing that ownership.

Accepted non-final segments can share a command buffer. Final segments, actual
ordering/visibility consumption, waits and staging pressure submit it. Distinct
logical completion tokens may refer to one host timeline tick; acceptance never
publishes a guest fence. Accepted recorded writes immediately publish pending
canonical ownership. A CPU/frontend demand submits that producer and waits for
the actual timeline before reading its bytes; failed submission invalidates its
ownership. Tests modify CPU vertex inputs between accepted segments and compare
the two resulting color regions exactly, then demand an unsubmitted producer and
verify that its new pixels replace the preceding image.

Images retain physical/layout identity, compact immutable footprint plans and
native views. The shared physical interval index finds image and buffer aliases.
Exact producer domains carry modification sequences separately from completion
tickets; repeated domains reuse their record. CPU writes invalidate intersecting
resources through dirty-page tracking, while native writes invalidate their
actual consumers. Depth, stencil, mips, layers and padding keep separate coverage.

Alias synchronization occurs when a resource is consumed, following the retained
image and modification-stamp model of Eden's texture cache. Compatible images
copy only newer intersections. Layout changes and partial transfer blocks use
transient native storage without creating a second physical resource. Unknown
opaque bytes remain undefined; only proven contents may be read or conservatively
published as modified. CPU readbacks publish only still-current producer versions.

Buffers retain linear physical mappings, join overlapping storage and import only
consumed dirty ranges. Images and buffers share the allocation budget and LRU.
Views and recorded commands retain native objects until their actual uses finish;
recycled cache slots receive a new generation. No physical allocation is kept
alive merely because a cached interpretation refers to its weak mapping.

CPU page demand locates image rows arithmetically and groups authoritative
fragments by source image. Native transfer blocks are merged into bounded row
regions and copied directly into readback staging. The CPU scatters only actual
producer bytes into its preserved baseline, including when D24 padding, stencil
and foreign partial writes share a page. There is no per-texel temporary GPU
allocation. Buffer alias reconstruction searches only overlapping sorted producers
and batches consecutive baseline holes without reordering GPU overlays.

Packed depth/stencil address legalization retains compact byte permutations.
Exact physical intervals are expanded only for demanded pages or conflicting
partial producers, rather than stored as one segment per texel.
Repeated immutable write coverage is queued once per recording ticket. CPU demand
flushes that ticket before materialization, so later writes publish a new point.
Every draw advances its native content version and producer ticket; compatible
continued rendering reuses the existing exact coverage and rasterization ordering.

## Raster and dispatch

| Behavior | Contract |
| --- | --- |
| Draw/indexed/instanced | Native counts, first vertex/index, signed base vertex, first instance and vertex/instance fetch. |
| Indirect draw/dispatch | Resident command buffers with actual `DRAW_INDIRECT` visibility; multi-draw/first-instance and host bounds checked when consumed. Compute-to-indirect execution stays on GPU. |
| Vertex formats | Native integer, normalized, half/float and 8/16-bit scaled fetch. `vertex::VertexFetchFormat` identifies 32-bit scaled integer inputs; the shared SSA conversion preserves their floating-point interpretation and has a distinct portable shader variant identity. Unsupported native format support is an error, not an approximate conversion. |
| Triangle fan | Native `TRIANGLE_FAN`, with exact selected provoking vertex; portability restrictions are errors requiring a frontend index lowering. No WebGPU fan index rotation. |
| Quads | `ssa::spirv::emit_quad_geometry` consumes four original vertices through `LINE_LIST_WITH_ADJACENCY`, emits `(0,1,2)` and `(0,2,3)`, and forwards flat outputs from the selected first/last corner. Guest VS effects execute on the original corners, without shader re-evaluation. Requires host geometry support; this does not enable guest geometry instructions. Point size is irrelevant to the emitted triangles. |
| Fill rectangle | Verified `VK_NV_fill_rectangle`, selected as a host capability at initialization. Other hosts get an explicit request for an exact lowering. No vendor/device/title checks. |
| Blend/masks/logic | Vulkan per-target blend state, write masks, constants and supported logic operations; attachment/blend format and core feature checks. |
| Depth/stencil | Native compares, writes, bounds, bias, clamp, separate front/back operations, masks and references. Combined depth/stencil attachments use one actual view. D24 publication excludes the unused high transfer byte. |
| Viewport/scissor/winding | Native viewport/scissor arrays, negative-height viewport, front face and front/back/both culling; bounded finite coordinates, no silent clamping. |
| Lines/wireframe | Native polygon and line modes, dynamic width, enabled fill/wide-line/mode/range checks. No replacement of guest width. |
| Tessellation | Native patch lists and compiled TCS/TES, including dynamic default levels and zero-level patch culling. Unconsumed trailing parameter words are not pushed outside the reflected ABI. |
| Multisample | Canonical native sample ownership with a frontend-defined sample grid, sample mask/alpha-to-coverage and native color resolve. CPU visibility extracts individual samples on demand; sRGB uses a bit-preserving UNORM view. Host-only intermediate attachments share the same recorder. |
| Empty draw | Attachment loads/clears/stores still execute; there are no shader accesses or writes. Empty direct dispatch has no shader effects. |

Host conversion attachments cannot be reconstructed from guest RAM. The conversion
recorder releases them after recording; pending commands retain their image/view
on the same native timeline. Canonical
MSAA producers serialize RGBA8/BGRA8, D24/D32 and stencil through bounded readback
windows. Importing arbitrary CPU-written samples, partial undefined MSAA readback,
multisample shader bindings and 3D canonical imports remain explicitly unsupported.
Native allocations themselves are not proof of guest sample layout semantics.

Full-plane producers also cache a physical full-page coverage proof. A dense,
injective projection covering every byte of a physical page supersedes older
image interpretations directly, including different texel sizes and sample grids.
This check precedes whole-view equality, avoiding one full image-map comparison
per page during reinterpretation. Partial regions, pitch padding, packed depth
and repeated projections retain exact producer coverage.

Canonical uncompressed image allocation defers import until a reader or LOAD
consumes bytes. A complete clear or device overwrite validates its own native
plane without importing overwritten CPU contents. Pitch padding never becomes
GPU-authored through that validation.

GPU buffer reconstruction merges contiguous source/destination runs across page
and GOB boundaries before recording transfers. In-place buffer moves preserve
their original ordered snapshots. Materializing a page consumes its lazy image
epoch once, rather than revisiting all packed intervals for every fragment.
Attachment feedback checks filter the current target identities before expanding
physical spans; historical images sharing a page do not participate in that test.

## Formats and coherence

Native images consume aspect-native linear copy planes. The concrete frontend is
responsible for guest tiling and encoded plane mappings. The native Maxwell
frontend now supplies those mappings;
it must not label guest packed/tiled bytes as a native plane without legalization.
Depth24 native elements contain only three depth bytes, so their unused high byte
cannot overwrite canonical padding or an independently owned stencil byte.
Producer publication preserves row order when canonical mappings repeat.

Native BC uses real compressed storage even for non-block-aligned base dimensions,
eliminating the WebGPU size workaround. Unsupported host ASTC/BC formats use the
shared exact `texture_decode` implementation. Decoded images observe encoded
canonical bytes and are read-only representations; they never claim to write back
encoded data. Modified planes are decoded and uploaded through shared staging
without splitting them to fit a command-context budget; unchanged reads do no work. Linear ASTC retains FP16 output; sRGB retains the specified UNORM8 decode.
GPU-written encoded bytes require real CPU visibility for this CPU decoder.
A future GPU decoder is a measured optimization, not a reason to approximate ASTC.

Native views implement component swizzles. Immutable sampler keys include custom
border components/format and reduction mode, with enabled feature checks. No WGSL
border-color patch or arbitrary unkeyed sampler `pNext` state enters the recorder.

Clear rectangles use dynamic rendering and `vkCmdClearAttachments`. Aligned buffer
fills use `vkCmdFillBuffer`; arbitrary boundaries need at most six upload bytes.
Native blits perform supported format conversion/scaling, and copies snapshot an
overlapping source on GPU. Unsupported filter/format/depth combinations fail.
VIC RGB conversion consumes immutable host NV12 planes through ordinary cached
raster execution. These inputs have no guest-memory alias or authority entry;
only the output uses canonical guest image tracking. CPU VIC handles linear
NV12 output directly. Presentation imports use typed leases and versioned snapshots.

Buffer storage writes and sampled image aliases share the canonical producer
index. The next image use imports the latest GPU bytes without CPU readback for
native formats. Same-image attachment feedback requires an explicit exact
snapshot/feedback-loop lowering and is rejected when supplied as ordinary bindings.

## Explicit gaps and reference behavior

- Guest queries/counter reports/resets are not advertised as emulated support.
  Binding-probe timestamps are diagnostics.
- Maxwell counter enable/configuration is implemented state programming, not a
  counter result (`gpu-maxwell/src/engines/threed/counters.rs`, `zcull.rs`).
- Supported report-semaphore payload/PTIMER behavior belongs to Horizon/frontend
  ordering (`threed/operations.rs`); other counter/reduction/acquire forms remain
  explicit unsupported operations.
- Stream output is rejected at draw consumption by
  `MaxwellLoweringError::UnsupportedStreamOutputSemantics`. A queried host
  transform-feedback feature does not remove that guest error.
- Native barriers implement ordered resource dependencies and shader barriers;
  they do not invent completion for unsupported guest synchronization methods.
- Unsupported ISA forms and resource combinations retain precise frontend/compiler
  errors. Shader execution uses the standalone recompiler and native Vulkan.

The fixed-color/compute helper renderer and its separate solid pipeline factory
have been removed. Tests use fixtures through this recorder. The alternative
binding ABI implementation is compiled only with `binding-probe`; its independent
pools/byte/address candidates are deliberately a diagnostic experiment.
