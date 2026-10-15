# Native shader, descriptor and pipeline caches

This is the Task 09 cache path. It uses semantic SSA/SPIR-V and the independent
Vulkan device/queue; it imports no wgpu cache configuration, WGSL/Naga identity,
per-resource variant quota or frame-count retirement rule.

## Identity and prepared references

The concrete frontend selects a `ShaderNamespace` once. Maxwell capture produces
an immutable, versioned content identity from explicitly encoded stage/header,
code, bounded indirect targets and guest specialization. GPU virtual addresses,
CPU epochs and mappings prove freshness separately; identical code relocated to
another address reuses its translation. Current captures resolve diagnostic PCs
against current mappings, rather than cached source addresses. Both graphics and
headerless QMD compute have this boundary.

`MaxwellShaderCompiler` caches semantic programs using the shared bounded
`CompilationCache`. Its actual decode/SSA/link elapsed time is recorded only on
misses. Fresh captures remain with prepared state; cached code contains no old
mapping freshness proofs. CPU/staged writes and indirect table bindings invalidate
only their relevant dependencies. Event/revision-driven preparation is wired in
task 11; no cache API hashes guest registers or polls every dependency per draw.

`ShaderCompiler` independently keys artifacts by guest/ISA namespace, compiler
schema, canonical source identity and exact emission options. Its callback can
borrow an already-ready `Arc<MaxwellShaderCode>` through the function-view callback:
no SSA clone or native-to-old-IR bridge is required. Compose completed frontend
values before requesting host work; callbacks must not wait for newly submitted
jobs on the same bounded pool. Disk hits bypass that callback and SPIR-V emission.

A returned shader's **native identity** follows its final SPIR-V code. Source
variants or namespaces which produce identical code and the same ABI therefore
share native pipelines. Stage code IDs also let pipeline libraries reuse unchanged
shaders when another stage changes. Source cache identities and physical descriptor
binding identities remain separate responsibilities.

A ready `Arc<NativePipeline>` and `Arc<BindingRecord>` are the resolved prepared
references. Repeated draws bypass full key hashing, normalization, lookup,
descriptor allocation/update and reference-count increments. The recorder retains
each used reference once per command-buffer batch and emits a bind only when it
changes. Bound-state knowledge resets at command-buffer boundaries. Content
synchronization/hazards still belong to the residency recorder on every consumed
access; descriptor reuse is not permission to omit those effects.

## Bounded asynchronous work

One shared `CompilePool` runs cold frontend/artifact/driver jobs. A key has one
pending publication, so concurrent duplicate misses compile once. Entry and byte
budgets include pending reservations and are updated to actual artifact payload
sizes. LRU eviction drops only derived cache ownership; prepared/in-flight
references retain their values. Queue/budget pressure is explicit and retryable.
Failed or panicking compilation publishes an error, never a successful substitute.
Structured shader diagnostics retain stage and guest PC.

Pending results expose polling and bounded waits. The GPU owner can service
visibility/control while waiting, retaining the required draw/dispatch in order.
No worker owns a Vulkan queue. Each worker has a separate `VkPipelineCache`; its
mutex excludes explicit checkpoint merge/serialization as well as creation, without
serializing every worker on a global driver cache lock. The final scheduler
composition and required-miss servicing are task 11.

## Vulkan pipeline construction

Initialization selects graphics pipeline libraries when the enabled feature and
`graphicsPipelineLibraryFastLinking` are available. Hosts without that combination
use the same construction implementation monolithically. Compute pipelines remain
monolithic. The four library subsets have projected keys containing only their
relevant shader/layout/static state, and concurrent library misses are deduplicated
without submitting nested jobs. Libraries and layouts use bounded LRU retention.

The [focused comparison](../../../../dump/vulkan-migration/task-09/README.md)
records first-use creation, valid creation feedback, linking, warm CPU/GPU time,
image equality and memory observations. It supports the selected fast-linking
policy on the tested device, not a universal speedup claim. Pipeline binaries and
background LTO replacement have no demonstrated need here and are not implemented
as extra persistence/renderer strategies. Reevaluate if complex guest shaders show
warm GPU regression or opaque driver-cache persistence/budget problems.

Useful dynamic state is selected from enabled capabilities. Viewport/scissor,
line width, depth bias, blend constants and stencil masks/references are dynamic;
EDS1 also removes culling/winding and depth/stencil controls from static keys.
Topology remains static: unrestricted dynamic topology has not been assumed from
an extension/version bit. Task 10 records raster state/setters through the same normalized key and
construction path. Additional dynamic states should be selected with real
steady-state evidence, rather than enabling an extension list indiscriminately.

When GPL is enabled, stages chain `VkShaderModuleCreateInfo` directly into pipeline
creation. Other hosts create temporary shader modules and destroy them after the
synchronous call. No redundant shader-module object cache is retained. Creation
feedback is consumed only with `VALID`; cache-control misses are compiled on the
worker and never cause a draw to be skipped. Actual stage stores/atomics, workgroup,
shared-memory, tessellation and graphics limits are checked at consumption.

References: [graphics pipeline libraries](https://docs.vulkan.org/samples/latest/samples/extensions/graphics_pipeline_library/README.html),
[creation feedback](https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineCreationFeedback.html),
[pipeline cache control](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_pipeline_creation_cache_control.html).

## Layouts and descriptors

Layout keys reflect live dense set-zero bindings, stage visibility and push ranges.
Buffer descriptors match the current emitter's std430 word-addressed SSBO storage
class, including guest read-only constant buffers; the guest semantic role is not
a Vulkan uniform-buffer declaration. Image/sampler classes remain separate.

Descriptor records key resource/view/sampler identities, offsets/ranges and layout
signatures. Leases keep the actual Vulkan allocations/views/samplers alive, so raw
handle reuse cannot alias a live record. Residency IDs still check their original
generations when resolving a changed binding. Image views now independently retain
their parent allocation. A cached record survives content uploads unchanged.

Pools allocate immutable sets in geometrically growing pages (4–64 sets per
page/layout, constrained by the total configured capacity). Sparse interfaces
therefore do not reserve 64 sets before another layout can bind. No individual
set free/update is needed. A page resets only after every cache/prepared/batch
reference is released, which includes real GPU completion. Under pressure, idle
LRU records can be dropped; if live references prevent reuse, explicit pressure
lets recording flush and resume. Resource leases remain charged until released.
Sampler keys normalize inactive compare/anisotropy fields and include explicit
custom border components/format and reduction state with enabled feature checks.

The fixed-color/compute helper renderer has been retired. Guest execution and
NV12 use the same pipeline/layout/descriptor preparation. Alternative binding ABI
pools exist only in the optional `binding-probe` experiment. Linked pipelines
retain their actual library components so cache/compiler eviction cannot destroy
their compiled dependency graph while a prepared/batch reference is live.

## Persistence and verification

Optional storage keeps only SPIR-V words, stable reflection and explicit stage
metadata, never raw host pointers or unversioned Rust IR. Files have length-framed
keys and payload checksums, bounded reads/writes, atomic publication and LRU disk
pruning. Invalid/corrupt entries are misses and are rebuilt. Driver blobs have a
separate device/driver namespace and verified Vulkan version-one header/UUID.
Each store is bounded; this is not a claim to bound the driver's private heap.

Tests cover duplicate requests, queue pressure, actual byte accounting, publication
of failures, shader code/staged-write/relocation invalidation, QMD compute metadata,
corrupt persistence rebuild, namespace isolation, native-code pipeline sharing,
dynamic key normalization, library subset reuse and shader/pool/resource lifetimes.
Gated real GPU submissions prove descriptors cannot recycle before completion.
Vulkan 1.2/KHR and 1.3 paths are exercised on hardware. The hardware gate skips only
when no real GPU is available; available-device initialization/validation failures
fail. The focused strategy probe additionally verifies actual raster output bytes.

Task 10 implements [native execution](../residency/execution/README.md); task 11 integrates
these caches with guest batches and dirty revisions. Full-game cold/warm behavior,
latency and working-set tuning remain those integration/performance gates.
