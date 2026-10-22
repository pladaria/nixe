# Native shader, descriptor and pipeline caches

The renderer uses captured Maxwell inputs and owned SPIR-V artifacts from
`nixe-shader-recompiler`. Host conversion and presentation shaders use the same
native module, pipeline and descriptor caches.

## Identity and prepared references

The concrete frontend selects a `ShaderNamespace` once. Captured instruction,
constant-buffer and texture observations determine immutable source identity.
GPU virtual addresses, CPU epochs and mapping generations prove freshness
separately. Bit-identical inputs retain compilation identity after relocation or
an equivalent write.

`ShaderCompiler::request_guest` consumes owned `ShaderInputs` and the enabled
`Configuration`. Workers retain reusable compiler scratch; published artifacts
contain no IR IDs or arena references. Stage keys include linked runtime state,
incoming binding counters and renderer legalization. Unchanged stages can be
reused before SPIR-V emission. Disk hits restore complete stage metadata and
bypass translation/emission. No callback waits for nested work on the same pool.

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

Separate bounded `CompilePool` instances run source/artifact and driver jobs.
Concurrent duplicate requests share one artifact or pipeline. Artifact entry and
byte budgets include pending reservations and account for actual published payloads.
Pipeline metadata has its own bounded LRU; prepared/in-flight references retain
the executable and its linked libraries. Queue/budget pressure is explicit and retryable.
Failed or panicking compilation publishes an error, never a successful substitute.
Structured shader diagnostics retain stage and guest PC.

Pipeline requests publish immutable resource/layout metadata before driver
construction. The command owner consumes resources and records native parameters
immediately; the Vulkan recording worker waits for the executable at its ordered
bind. Published results use a lock-free read, and build errors fail the native
submission with their original cause. Driver compiler workers do not own a Vulkan
queue. Each has a separate `VkPipelineCache`; its mutex excludes explicit
checkpoint merge/serialization and creation without serializing all workers.

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
Topology remains static: unrestricted dynamic topology is not assumed from
an extension/version bit. Raster state/setters use the same normalized keys. Additional dynamic states should be selected with real
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
Constant buffers use uniform descriptors and storage buffers use storage descriptors.
Reflection retains combined image/sampler arrays and uniform/storage texel buffers.
Descriptor resource origins remain independent of shared Vulkan layout identity.

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
