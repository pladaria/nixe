# Vulkan shader, descriptor and pipeline ownership

The cache architecture follows Eden's `PipelineCache`, `ShaderCache` and
`DescriptorLayoutBuilder` at revision
`d85005b2b27ebdbe785316886cdf04c2894d550e`.

## Lookup and construction

The raw shader cache indexes immutable bytecode hashes and sizes by device address.
Tracked writes invalidate overlapping entries. Graphics pipelines use raw bytecode
hashes and feature-dependent fixed state; compute pipelines additionally key shared
memory and workgroup dimensions. A graphics lookup checks the owner's current
pipeline, its packed transition keys, and then the shared table. Feature-dependent
key prefixes use CityHash and byte comparisons. Pipeline entries remain
owned for the renderer's lifetime.

Current shader facts and the fixed pipeline key belong to the renderer. Channel
changes retain those selections and invalidate the new channel's canonical dirty
bank, native topology and stencil state. A new command buffer invalidates emitted
native state independently of the channel's persistent resource selections.

On a guest miss, the graphics owner records the environment while translating the
shader once into reusable compiler arenas. Those programs directly supply linkage
and SPIR-V emission. The new pipeline owns its Vulkan shader modules, layout,
resource metadata and executable. Construction workers build one complete Vulkan
pipeline; native recording waits for that executable at its ordered bind. Published
results use a lock-free read, and shader errors retain the stage and guest PC.

The tables perform hit detection before translation or construction. There are no
additional SPIR-V, stage, native-code, layout or pipeline-library lookup caches, and
no pipeline LRU or compilation admission budget. Host utility pipelines belong to
their presentation, conversion or readback owners.

All construction workers use one internally synchronized `VkPipelineCache` for the
title. They do not submit work to a Vulkan queue. A single serialization worker
writes both persistent files and drains before the final driver-cache checkpoint.

## Layouts and descriptors

Layout keys reflect live dense set-zero bindings, stage visibility and push ranges.
Constant buffers use uniform descriptors and storage buffers use storage descriptors.
Reflection retains combined image/sampler arrays and uniform/storage texel buffers.
Descriptor resource origins remain independent of shared Vulkan layout identity.
Guest host parameters use the fixed SPIR-V ABI shared with the renderer; draw and
dispatch fill stack constants directly, without interpreting reflected layouts.

Pipeline binding, changed push constants and descriptor updates enter the native
recorder as one ordered configuration. The owner selects changes and retains
resources; the recording worker waits for the executable and emits the grouped
calls. Graphics, compute and host utilities consume the same configuration path.

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

Native pipelines, samplers and descriptor pools have stable renderer ownership.
Recording jobs borrow pipeline references under that ownership instead of cloning
pipeline handles or maintaining per-command identity sets. Object insertion locks
only cold construction. Submission drains before renderer storage is released,
including when recording fails.

Binding captures retain resolved inputs until recording consumes them. Actual
buffer/image retirement protects their physical GPU versions through completion;
stable renderer ownership protects samplers, layouts and descriptor pools. Texel views belong to their buffer and are reused by offset,
range and format. Immutable format capabilities are cached rather than queried
again after an unrelated binding changes. There is no descriptor admission budget,
page-reset loop, historical payload LRU or resource-free invalidation scan.
Resource contents are not a binding identity.
Sampler keys normalize inactive compare/anisotropy fields and include explicit
custom border components/format and reduction state with enabled feature checks.

## Persistence

Each title uses its sixteen-digit lowercase ID beneath `gpu.cache_directory`:

- `vulkan.bin` stores the pipeline keys and recorded guest environments, including
  the code and constant/descriptor observations required for translation.
- `vulkan_pipelines.bin` stores the opaque Vulkan driver cache. Its version-one
  Vulkan header, vendor/device IDs and pipeline-cache UUID must match the device.

Both files have an explicit renderer cache version. Environment records also carry
the shader recompiler ABI version. Incompatible or truncated derived data is
recreated. The format is Nixe-owned; it is not interchangeable with Eden's files.

Startup streams records directly into reconstruction work and supports cancellation.
It reconstructs every compatible saved pipeline in parallel, using fresh
compiler arenas per job, and waits for completion before guest execution. It does
not restore serialized SPIR-V or impose an entry quota. Runtime misses append their
environments when every instruction belongs to the cached code span, as Eden's
`CanBeSerialized` rule requires. Disk jobs consume owned environments without
cloning instruction or constant-buffer maps. Driver checkpoints use Eden's threshold of 128 new pipelines and an
interval of at least 30 seconds, increasing with the previous driver-cache size.
Disk I/O failures are logged and do not change guest execution semantics.

Native resources and descriptor payloads remain retained through CPU recording and
GPU completion. A cache hit never bypasses resource coherence or hazard tracking.

The device policy chooses native scaled formats or integer fetch with shader
conversion, as Eden's `MustEmulateScaledFormats`. The proprietary Qualcomm driver
uses the conversion path; static vertex keys preserve the original numerical
class and emitted shaders consume the same device policy. Native Vulkan format
mapping covers the reference's aliases and packed formats. A combination absent
from the selected reference path fails precisely.

Info logs include the driver-cache byte count and load duration, environment
read/queue duration, and completed pipeline count with total startup duration.
