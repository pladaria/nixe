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

Ready pipeline references and immutable binding leases are consumed directly.
Unchanged native descriptor payloads reuse their last set without hashing a
historical cache key or updating descriptors. The recorder retains each used
reference once per command-buffer batch and emits a bind only when it changes.
Bound-state knowledge resets at command-buffer boundaries. Content synchronization
and hazards still belong to the residency recorder on every consumed access;
descriptor reuse is not permission to omit those effects.

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

The command owner captures resolved resources and compact template entries. Its
weak lookup reuses live CPU captures without owning their storage. The Vulkan
recording worker selects the physical set, updates it with the layout-owned
update template, and binds it in command order. CPU captures carry complete
reconstructible data and never retain a physical descriptor set.

Layout allocators share native pools by per-type capacity and grow sets in groups
of sixteen. Sets recycle by their actual last GPU tick, independently of retained
CPU preparations. Before growing, the worker refreshes actual GPU completion and
searches again. Pool exhaustion allocates another pool without submitting or
restarting a draw. Rebinding an old capture after recycling restores its data.
The worker compares its last payload using weak resource identities, so unchanged
bindings reuse their set after GPU completion without pinning staging storage or
matching a destroyed object's recycled handle. Payload metadata reuses its array.

Each native command context retains the descriptor pools it actually uses until
GPU completion, including if the worker unwinds. Retention is deduplicated once
per pool/context; there is no per-command producer lock. Cold layout retirement
can release metadata without destroying a pool still used by the GPU.

Binding captures retain actual buffers, views, samplers and layouts through their
CPU and GPU use. Texel views belong to their buffer and are reused by offset,
range and format. Immutable format capabilities are cached rather than queried
again after an unrelated binding changes. There is no descriptor admission budget,
page-reset loop, historical payload LRU or resource-free invalidation scan.
Resource contents are not a binding identity.
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
