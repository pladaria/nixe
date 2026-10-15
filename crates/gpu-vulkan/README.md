# Independent Vulkan backend

The backend owns an independent ash instance/device, selects a graphics+compute queue
and actual surface presentation support, freezes its capability profile, and
allocates real Vulkan buffers/images through `gpu-allocator`. It has no wgpu,
Maxwell decoder or Horizon dependency. Real guest batches arrive through the
statically selected frontend/owner composition described below.

`VulkanDevice` owns graphics/compute submission and moves to the GPU worker.
Windowed execution selects a distinct presentation queue at startup; one WSI
worker owns its swapchain acquire/recreate/present/release operations. Resources
retain the device/allocator/instance lifetime with `Arc`, without retaining queue
access. Allocation/free operations use an allocator mutex because resource drops
can occur on another thread; command recording/submission does not acquire it.
Buffer/image handles stay internal. Destruction releases the Vulkan object before
its allocation, and the allocator before its device. Surface ownership retains
the native window until after surface destruction.

## Initialization policy

- Vulkan 1.3 uses core synchronization2/dynamic rendering entry points; Vulkan
  1.2 requires their KHR extensions. Both require actual timeline semaphore,
  synchronization2 and dynamic rendering feature bits. API aliases resolve once.
- Native window creation precedes device selection. Actual queue-family surface
  support, formats, present modes and color-attachment usage are queried; a
  distinct presentation queue is requested, preferably in the same family.
  Headless initialization creates neither a window nor a surface.
- The read-only profile records enabled dynamic state, pipeline libraries,
  descriptors/addressing, subgroup and numerical facts, native rasterization
  features, unified layouts and swapchain maintenance. These are capabilities,
  not benchmark results or guarantees of particular external-video resource support.
- Portability enumeration and the mandatory portability subset extension are
  enabled when exposed, with subset restrictions recorded in the profile.
- Unified layouts select `GENERAL` for the diagnostic's ordinary images while
  preserving memory dependencies. Without the feature, ordinary transfer layouts
  apply. Swapchain maintenance selects KHR, EXT or disabled using actual instance
  and device dependencies; aliased commands resolve at initialization.
- Narrow registry-verified declarations cover extensions missing from ash 0.38:
  unified layouts, shader FMA and KHR swapchain maintenance aliases. They are
  native Vulkan ABI declarations, not a driver workaround or a fork. Replace them
  when a published ash release includes these bindings. See the pinned registry
  reference in [extensions.rs](src/extensions.rs).

Ordinary allocations are suballocated; driver-required or explicitly requested
dedicated allocations use the corresponding Vulkan dedicated object. Mapped
suballocations isolate non-coherent atoms; dedicated allocations preserve the
driver's exact size. Host writes flush and completed reads invalidate when memory
is not coherent. No production validation callback is enabled by default.

## Queue progress and transient memory

One device-owned timeline controls command-pool reuse, staging reuse and resource
retirement. `work_ticket()` reserves a session-scoped, monotonically increasing
host tick; `flush()` accepts recorded work without waiting. `wait_for_work()`
submits a recording target before a bounded, targeted timeline wait. A timeout
does not mark work complete. `poll_progress()` advances retirement from the real
semaphore counter. Neither ordinary submission nor shutdown uses device-wide idle.

`WorkTicket` identifies host work; the native owner associates distinct logical
tokens with it when batching several accepted segments. Host completion is only
one prerequisite: the residency layer
must also finish required canonical ownership effects before returning
`BackendExecutionCompletion`. Horizon alone publishes its guest reservation.
The owner uses bounded waits and services control messages even at full data
admission; deferred data retains its original submission/byte permits. Frontend
preparation, including compilation, cannot hold the owner's control inbox.

Command slots own a reusable primary command buffer, upload/readback bump arenas
and typed retirement lists. Reset requires the slot's real last tick to complete
and every retained readback to be consumed/discarded. Resource destruction can
advance at GPU completion while an unread staging slice remains pinned. Arena
slices isolate non-coherent atoms, and flush/invalidate touches only consumed
ranges. Outstanding readback counts are cached to avoid scanning every slice.

`QueueConfiguration` selects fixed bounds at initialization: eight command slots,
4 MiB for each arena/direction/slot, and 16 MiB total live overflow capacity by
default. Arenas are allocated lazily with dedicated memory, avoiding the
allocator's large managed block granularity for this bounded transient storage.
Overflow is charged across all in-flight slots and freed only after its real uses
and retained reads end. `QueuePressure` returns a pending tick when waiting can
help, or reports unread ranges that the caller must consume/discard. It never
grows the pool or staging budget silently. Larger transfer requests must be split
into chunks by the copy/residency recorder; these bounds are not guest image limits.

The diagnostic buffer/image recorder uses this same queue, arenas and retirement.
The former per-transfer command pool/semaphore and its separate cleanup path have
been removed. `transfer_round_trip`/`image_round_trip` remain synchronous diagnostic
wrappers; `enqueue_*_readback` batches recording without waiting. Task 06 adds
ordered native access/barrier generation; task 05 adds the caches below and task 08 adds compilation.

Shutdown cancels unsubmitted commands, drains the last accepted tick and releases
resources without publishing canceled guest completion. Device loss is terminal.
If the driver cannot establish completion or loss during destruction, the process
terminates rather than freeing possibly in-flight memory or retaining a leak path.

## Specialized residency (task 05)

`VulkanResidency` owns the device and specialized buffer/image entries with typed
session/slot/generation IDs. [buffer.rs](src/residency/buffer.rs) handles canonical
buffer identity and CPU epochs; [image.rs](src/residency/image.rs) handles native
images, compatible views and texel-row representations. Cold binding resolution
uses canonical identity, independent of CPU mapping permissions/generations.
Prepared state keeps cache IDs; `ensure_buffer`/`ensure_image` use the CPU summary
and representation invalidation on the warmed path, without an overlap/page walk.
Native draw/dispatch preparation allocates buffers and images without eager byte
imports; the ordered consumer refreshes them once. Buffer lookup hashes immutable
byte mappings and proves equality on collisions, with separate usage variants.
Per-page bindings are indexed by resource, and vacant buffer slots use a free list.

One [canonical index](src/residency/index.rs) serves both overlap lookup and exact
GPU authority. Later producers replace only intersecting byte ranges. Image
bindings cover texels, excluding pitch padding and D24's unused transfer byte.
Fragmented mappings and repeated
canonical offsets keep their logical order, including repeated rows of one image.
A new alias copies from the native producer or shares an existing representation;
it does not publish GPU contents in CPU memory. Compatible UNORM/sRGB views share
the same mutable native image and survive content changes. Views die before their
image, under the image's existing timeline retirement.

Partial image writes retain cached full-plane footprints and clip only candidate
pages. A proven full physical-page overwrite retires obsolete exact producers.
Buffer reconstruction merges physically ordered bindings and producer intervals,
then coalesces native copies in logical byte order. A dense, current equivalent
image imports directly. Other complete mappings from one current image can retain
a coalesced copy plan (at most 16 KiB per buffer), independent of pixel epochs;
replay revalidates the source generation, CPU dependencies and complete native
contents. Mixed producers, padding holes and stale sources use exact demand
reconstruction. Undefined opaque bytes are never replaced with a fabricated clear.

[CPU visibility callbacks](src/residency/visibility.rs) preserve local page bytes
under canonical locks; they never submit, wait or reenter canonical memory. CPU
requests go to the single owner through `BackendVisibilityRequester`, while owner
snapshots resolve inline. Demanded readbacks download only authoritative intervals
and overlay them on preserved bytes, retaining the result until canonical memory
acknowledges publication. A stale acknowledgement cannot release a newer producer.
Disjoint producer rows share bounded readbacks rather than one allocation/wait per
row. Aligned image/buffer aliases use native copies directly; split texel/block
requests realign on GPU before transferring exactly the demanded bytes to CPU.

CPU uploads cover changed page intersections. Snapshot epoch acknowledgement and
ordered repeated offsets are handled together; cached invalid intervals retain an
unfinished upload across `QueuePressure`. Upload/readback arenas are independent.
Sparse CPU snapshots retain only their changed/aligned windows, without expanding
the unchanged tiled topology. Fully CPU-visible inputs retain adaptive tracking;
only mixed ownership needs an additional rearm of excluded GPU pages.
`write_buffer` returns a `BufferUpload` with the consumed prefix and native ticket;
the execution owner must retain and continue the remaining bytes. It never claims
a large upload was fully consumed or turns arena size into a guest resource limit.
Standalone residency writes publish canonical GPU ownership through `flush` after
actual queue acceptance, separately from completion. The native execution owner
can also accept a non-final recorded batch without submitting its command buffer:
it publishes pending ownership immediately, and CPU demands force submission and
wait for the real timeline before publishing bytes. Failed submission invalidates
all affected ranges, including already accepted pending producers.

The residency budget charges driver-reported native memory requirements, including
GPU temporaries and objects already evicted but still in flight. Charges follow
native objects into timeline retirement. Reconstructible entries are evicted by
age; authoritative contents and presentation leases are pinned. Pressure returns
`ResidencyPressure`, without a hidden whole-memory download. `preserve_buffer` and
`preserve_image` explicitly publish only the selected resource's canonical pages,
using exact producer intervals, when preservation is needed for eviction. Leases
retain an exact content version/producer and block overwrite until the last release;
ordered presentation reads or GPU snapshots are implemented with the presenter.

`allocated_bytes` measures charged live/in-flight resources. Staging has its own
queue budget. `allocator_capacity_bytes` separately reports actual allocator
capacity, including slabs, staging and fragmentation; the resource budget is not
a claim that physical committed VRAM equals useful allocation sizes. Managed
slabs start at 8 MiB and grow up to 64 MiB for device storage, instead of reserving
256 MiB for the first small resource.

The transfer slice consumes native single-sample 2D planes, with native packed
formats, compressed blocks, mips, array layers and separate aspects. Geometry and
usage are checked at consumption. RGB565 keeps packed Vulkan storage, without the
wgpu RGBA expansion. Task 10 implements general native execution and resident-only
multisample attachments. Guest tiling/plane legalization and real Maxwell batch
integration remain the concrete frontend's task 11 work. Unsupported canonical
imports fail explicitly; allocating a native image does not fabricate its contents.
The old renderer remains the game reference until migration cutover.

Native image address descriptions stay compact through resource creation and
producer publication. Maxwell supplies shared byte-tile permutations and packed
aspect strides; the canonical memory layer retains physical pages independently
of the ordered byte mapping. The Vulkan index stores ordered image producer
layers and disjoint byte-lane proofs, resolving exact intersections only for a
byte consumer, a partial buffer producer or bounded dependency fan-in. Temporary
page interval maps are discarded after consumption. Full clears acquire storage
without importing overwritten content; CPU changes to untouched aspects remain
invalid until read.

CPU page faults reuse versioned native row-window downloads through a bounded
hash directory (32 windows, at most two staging arenas of CPU bytes). Small
multi-plane images retain single-batch downloads. Undefined regions, rows larger than the arena and projected working sets that
exceed the cache use exact bounded regions. Both paths scatter only current
authoritative bytes into the preserved canonical baseline, excluding D24 padding.
See [mapping and performance evidence](../../dump/vulkan-migration/compact-image-mapping/README.md).


## Ordered recording (task 06)

[sync.rs](src/residency/sync.rs) tracks native buffer byte ranges and image
aspect/mip/layer state separately from the canonical producer index. RAW/WAW
consume writer visibility; WAR needs execution ordering; independent ranges and
already-visible reads emit nothing. The last writer survives earlier reads so a
second reader stage/access kind cannot inherit visibility it never received.
Depth/stencil state is coupled when independent layouts are not enabled. Layout
transitions order previous readers too, rather than losing them after a read.

A reusable barrier batch emits immediately before its dependent command. The
normal policy coalesces dependencies into memory barriers with concrete stage and
access masks, including unchanged-layout images. Image barriers handle layout
changes and queue ownership. `PerResource` and conventional layouts are explicit
startup/probe policies, verified with the same semantic tests. Neither policy adds
an `ALL_COMMANDS` memory barrier after every command. `barrier_statistics` exposes
calls and memory/buffer/image counts. Feedback-loop combinations require an exact
supported lowering; ordinary sampled/attachment layouts cannot stand in for one.

[transfers.rs](src/residency/transfers.rs) records scoped fills, color/depth/stencil
clears and image-region copies. Same-subresource overlapping copies snapshot their
source on GPU before writing. Writes publish exact texel intervals, preserving
other aspects, mips, layers and pitch padding. The former fixed-color and compute
helper renderer has been retired. [The task 10 recorder](src/residency/execution/README.md)
uses cached pipelines/descriptors for draw, indexed/indirect execution, compute,
blend/depth/stencil/line/tessellation state, exact quad expansion, native partial
clears/blits and NV12 conversion. [attachment.rs](src/residency/attachment.rs)
keeps host-only intermediate attachments until explicit release. Canonical MSAA
images retain individual samples using a frontend-supplied storage grid. Native
resolves produce single-sample texels; bounded sample readback runs only for CPU
visibility and preserves RGBA8/BGRA8, D24/D32 and stencil bits, including sRGB
through compatible UNORM views.

`image_for_device_write` allocates undefined storage without uploading or clearing
it. LOAD and image reads consume actual initialized contents. Discard invalidates
the rendered region; later producers restore their own coverage. All commands
share the same native ownership and hazard model. The execution documentation
separates implemented semantics, host-dependent capabilities and explicit gaps.


[ownership.rs](src/residency/ownership.rs) records matching queue release/acquire
halves and pins released subresources against eviction/local use. Its unsafe
contracts require eligible allocations, real foreign completion/semaphore ordering
and retained foreign lifetimes. These primitives do not implement external memory
imports or claim external-video parity; those adapters remain task 13. Host-write
visibility is covered in access-state tests; staging/descriptor host writes flush
isolated atoms and become available through actual queue submission.

Helper modules live in [shaders](shaders). The offline build pins SDK 1.4.341.1,
compiler/validator binary hashes, Vulkan 1.2 and SPIR-V 1.5. Checked-in modules are
embedded by the runtime; normal builds need no shader compiler. Rebuild/check:

```sh
python scripts/build-vulkan-shaders.py --sdk-bin /path/to/sdk/x86_64/bin
python scripts/build-vulkan-shaders.py --sdk-bin /path/to/sdk/x86_64/bin --check
```

The initial shader binding ABI uses Vulkan set 0 with dense reflected live bindings,
typed descriptors and small push constants for scalar/window arguments; it does
not encode Maxwell register limits. The real slice exercises set 0/binding 0 as a
storage buffer. Descriptors are cached in bounded command slots; pools and mapped
descriptor arenas recycle only after completion. Repeated unchanged bindings avoid
hashing and descriptor updates; device addresses and extension entry points are
queried once. Pipeline/helper objects remain alive until the owner's final drain.
Tasks 08/09 expand the same ABI for linked stages, live resources and descriptors.

The optional `binding-probe` feature compares descriptor sets, physical device
addresses and descriptor buffers using identical word conversion and matched
hazards. It warms pipelines/residency, checks output, reports median CPU recording
and actual GPU timestamps, and is excluded from the normal build:

```sh
NIXE_VULKAN_TEST_ADAPTER=NVIDIA cargo run --release -p nixe-gpu-vulkan --features binding-probe --example binding-probe
NIXE_VULKAN_TEST_ADAPTER=NVIDIA cargo run --release -p nixe-gpu-vulkan --features binding-probe --example binding-probe -- --per-resource
```

Sequential NVIDIA runs (1024 words, 256 dispatches, 7 trials, validation disabled)
reported scoped-memory CPU/GPU ns per dispatch: descriptor sets 1660/935, BDA
1638/880 and descriptor buffers 1530/927. Per-resource results were 1672/936,
1655/881 and 1600/927 respectively. These are one synthetic helper workload, not
game performance or proof of a universal winner. The initial ABI keeps reusable
baseline descriptor sets; optional transport/address optimizations need broader
binding churn, images/samplers and guest-path evidence before promotion. Raw results
and interpretation are in `dump/vulkan-migration/task06/`.

Task 06 validation passed 37 Vulkan tests on real NVIDIA hardware with current
Khronos/synchronization validation, including core/KHR paths, unified/conventional
layouts and scoped-memory/per-resource barriers. The existing native Horizon owner
test also passes. External-import/video and Windows/MoltenVK hardware validation
remain open; no production guest renderer or game-performance parity is claimed.

## Diagnostic and focused checks

### Local Linux loader

Run `./scripts/build-vulkan-loader.sh` once to build the official Vulkan Loader
and Headers 1.4.341 at pinned commits. It requires Git, CMake >= 3.22.1, Ninja,
Python 3, a C compiler, pkg-config and X11/XCB/Xrandr development packages
(`libx11-dev`, `libxcb1-dev`, `libxrandr-dev` on Debian/Ubuntu). The initial
source fetch needs network access; subsequent builds reuse clean pinned sources.

Sources, builds and the install prefix live under the ignored
`dev-tools/vulkan-loader/` directory. The installed library is
`dev-tools/vulkan-loader/install/lib/libvulkan.so.1`; licenses and
`build-info.txt` retain source provenance and the resulting library checksum.
The recipe pins source and build options; the host compiler and system libraries
still determine the resulting binary. Nothing is installed system-wide.

Cargo's Linux runner prepends this local library directory to `LD_LIBRARY_PATH`
when the library exists. This applies to `cargo cli`, `cargo cli-dev`,
`cargo run`, `cargo test` and `cargo bench`, preserving Cargo's dynamic-library
paths and the existing environment. Without a local build, normal system loader
discovery applies. Running an executable directly does not use the Cargo runner;
use the wrapper and include the binary's other runtime libraries (such as SDL):

```sh
LD_LIBRARY_PATH="target/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    ./scripts/with-vulkan-loader.sh target/release/nixe-cli ...
```

This does not install a driver or validation layers. Linux development tooling
is separate from the application's runtime loader selection and packaging.

### Checks

```sh
cargo run -p nixe-cli -- vulkan-check --validation
cargo run -p nixe-cli -- vulkan-check --headless --validation
cargo run -p nixe-cli -- vulkan-check --validation --vulkan-api 1.2
cargo run -p nixe-cli -- vulkan-check --validation --adapter NVIDIA --log-level debug
cargo test -p nixe-gpu-vulkan
cargo test -p nixe-gpu-vulkan -- --nocapture --test-threads=1
cargo test -p nixe-horizon --lib nvdrv::gpu_executor::vulkan_queue_tests -- --nocapture
```

`vulkan-check` requires no title/configuration. It checks exact buffer readback,
managed and dedicated optimal-tiled RGBA8 image readback, and validation after
device destruction. `--log-level debug` prints the selected immutable profile.
Requested validation fails clearly if the layer is unavailable. Use a current
Khronos validation layer that understands the enabled extensions; configure
`VK_LAYER_PATH` for a separately installed SDK. For synchronization validation,
current layers accept `VK_LAYER_VALIDATE_SYNC=1`. Device tests automatically use
a hardware GPU and report `SKIP` when no hardware GPU or Vulkan runtime is
available. Software devices are excluded; no software-driver fallback is used.
Once hardware is available, missing validation, unmet baseline features, setup
errors and validation/transfer failures fail the test. `NIXE_VULKAN_TEST_ADAPTER`
optionally selects a hardware adapter by name substring; an unmatched filter is
an error. Rust's test harness counts early returns as successful tests; use
`--nocapture` to see the skip reasons. Skipped runs provide no GPU validation evidence.

## Verified host matrix (2026-10-13)

| Host | Coverage | Result |
| --- | --- | --- |
| Linux / NVIDIA GeForce RTX 4070 Ti SUPER, driver 595.58.03 | Core 1.3 and forced 1.2+KHR; headless transfers/lifetimes and native X11 surface initialization | Passed with Khronos validation 1.4.341 and synchronization validation; no errors |
| Windows / native Vulkan | Initialization uses platform surface extensions; physical verification unavailable here | Unverified; supported-host requirement remains open before cutover |
| macOS / MoltenVK | Portability enumeration/subset policy implemented; physical verification unavailable here | Unverified; required profile/guest semantics must be verified before cutover |

NVIDIA exposed unified image layouts, explicit shader FMA, descriptor buffers,
graphics pipeline libraries and shader objects on this host. Its X11 surface
initialization selected family 0 for both graphics/compute and presentation;
swapchain maintenance was unavailable with the installed instance extensions and
was disabled. These initialization results are not a performance baseline.
No games or pre-port performance captures were run for task 03.

Task 04 checks hold real GPU work behind a test-only external semaphore gate,
exercise more reuse cycles than the configured pool size, verify retained reads
and bounded overflow, and submit a final fence without later work. A native
Horizon owner fixture verifies real Vulkan completion tokens and visibility while
preparation is pending. This resource-free fixture is test code; it does not claim
full guest rendering/canonical residency. Loss bookkeeping is tested without
claiming actual driver fault injection. GPU checks use the real hardware selection
policy in [tests/support/device.rs](tests/support/device.rs); no software fallback is used.

Task 05 adds twelve real-GPU residency tests in
[src/residency/tests.rs](src/residency/tests.rs): partial CPU uploads, exact demanded
readback, device-only aliases, disjoint page writers, repeated/fragmented offsets,
image/buffer aliases, compatible views, compressed mips/layers, packed RGB565,
explicit preservation, allocation pressure, generation reuse, in-flight accounting,
presentation leases and cross-thread CPU requests. The full Vulkan suite passed
25 tests on the NVIDIA host above, with synchronization validation and no errors.
This verifies cache/transfer semantics; it does not establish game performance.

## Native compilation and caches

Task 09 adds [native shader, descriptor and pipeline caches](src/cache/README.md).
Prepared pipeline/binding references avoid hashing and repeated binds on unchanged
draws. Bounded compilation workers own separate driver caches; descriptor/resource
leases retire with actual GPU completion. The focused GPL/monolithic comparison and
its limitations are recorded in `dump/vulkan-migration/task-09/`. Guest integration
and full raster/compute semantics remain tasks 10–11.

Task 10 execution evidence is recorded in
[the execution contract](src/residency/execution/README.md). The renderer is callable
and attached to real Maxwell/Horizon batches through the temporary explicit
`--headless --vulkan-dev run <title>` development selection. The CLI selects one
concrete frontend and GPU owner at startup. Native execution has no wgpu/HAL
fallback and does not construct old driver submissions/resource definitions.

The native boundary carries dense SSA code, immutable binding/draw references,
arguments, swizzles and aspect-native canonical planes. Logical image/allocation
registries and Tegra tiling parameters do not cross that boundary. Maxwell owns
its code/descriptor/mapping proofs; Vulkan owns native formats, caches, recording,
hazards and completion. The old execution remains the reference until task 15;
the generic composition seam and its old default types retire in task 16.

Draw arguments, shader bindings and graphics parameters have independent retained
identities. Physical vertex-window changes preserve vertex formats and raster
preparation. If intermediate register writes restore the same consumed state,
bit-exact semantic comparison retains the previous parameters instead of issuing
a fresh native identity. Shader bindings likewise compare physical buffer
mappings and immutable image views, independently of content epochs. A changed
binding set reuses the compiled pipeline and validated dynamic-state parameters;
unchanged sets retain their descriptor preparation. Current render targets also
retain native view preparation separately from vertex/shader bindings.

Compatible LOAD/STORE attachments keep one dynamic-rendering instance across
frontend deliveries, pipeline/dynamic-state changes and descriptor changes.
Ordinary framebuffer operations retain Vulkan's rasterization ordering; shader
writes, feedback, transfers, real hazards, clears, presentation and submission
terminate the instance. Canonical write ownership still publishes at accepted
command boundaries, and command-buffer retirement retains every native use.
Within an unchanged compatible scope, the first draw publishes the attachment's
physical coverage. Later draws retain that coverage and advance the image's
content version and producer ticket without walking its pages again. Imports,
CPU visibility and other ownership changes end the scope before it can be reused.

Small CPU-authored read-only shader/vertex windows can bind immutable snapshots
directly from mapped upload arenas. Fresh versions use distinct aligned regions;
unchanged versions reuse their region only within the same reserved submission
and with a current CPU dependency. An independent observer preserves the native
buffer's dirty proof. GPU-authored/mixed pages use ordinary native reconstruction
and barriers. The 64 KiB policy bounds per-read snapshots; fixed-arena exhaustion
uses ordinary transfers rather than retaining cached overflow allocations outside
their budget. Non-coherent flush/atom isolation and timeline retirement apply on
both discrete and unified-memory hosts.

Maxwell derives constant-buffer windows from live optimized SSA once during
compilation. Proven static loads retain their maximum consumed byte end across
stages; unproved addresses retain the declared binding. Texture-handle reads also
extend the resolved window. This bounds snapshots to consumed bytes without
per-draw shader inspection or assumptions about guest buffer contents.

Task 11 verifies real offscreen homebrew execution and prepared-state reuse.
Task 12 adds native windowed presentation, egui mesh/texture recording and PNG
readback. Video producers/conversion remain task 13. Complete guest coverage, host verification and final workload/performance
acceptance remain tasks 14/17. Development selection reports missing native guest
forms precisely, including multisample shader bindings, fragment alpha
specialization, attachment feedback snapshots and unsupported vertex defaults.


## Native window presentation

`--vulkan-dev run <title>` uses the same native device for guest rendering,
scanout, egui and screenshots. The main thread handles window/input and builds
bounded host packets; graphics recording stays on the Horizon GPU owner. Blocking
WSI calls use a separate queue/worker and never acquire the graphics queue mutex.
A single outstanding acquire/present handoff bounds the worker channel. Different
queue families share only WSI images concurrently; ordinary resources stay exclusive.

Scanout takes immutable, versioned native snapshots ordered after guest writes.
The guest's buffer-slot lease is released independently of WSI availability; the
snapshot's lease and last GPU use separately prevent reuse. Cached source identity
is revalidated against current contents; a newly authoritative resident image
replaces a stale alias. A taller attachment can supply its visible prefix only
when native geometry and physical byte projection agree exactly. Undefined
padding is neither imported nor sampled. Opaque sources require an existing,
current, initialized resident producer; canonical import cannot substitute for it.

Host shaders are checked-in, reproducibly built SPIR-V helpers. Presentation and
UI reuse native pipeline/descriptor caches and command arenas. Egui font/full and
partial texture updates, premultiplied blending, clipping/DPI and texture frees
are supported; coalescing an undisplayed UI packet still consumes its frees.
Custom paint callbacks and mipmap generation are explicitly outside this bounded
UI contract. PNG capture performs optional, incremental native-resolution readback
of the cropped/transformed frame, then encodes on a CPU worker. Routine scanout
has no CPU readback or cross-device transfer.

Windowed hosts require a distinct presentation queue and KHR/EXT
swapchain-maintenance with its actual instance dependencies. Presentation fences
prove semaphore/swapchain retirement independently of graphics timeline ticks;
queue/device idle is not used as a substitute. Missing support fails precisely,
while headless keeps the ordinary Vulkan 1.2/1.3 requirements. The system loader
must expose the maintenance instance extension; an old loader can hide an extension
supported by the GPU driver. Missing instance support reports the extension names
and Khronos Vulkan Loader 1.3.237 as the first upstream version with EXT support;
missing device support instead reports the GPU/driver feature requirement.
These checks use capabilities, not a loader version gate. Nonlinear RGBA8/BGRA8
UNORM surfaces, opaque window compositing and identity surface transforms are
supported. sRGB-only surfaces need
`VK_KHR_swapchain_mutable_format`: an UNORM attachment view preserves nonlinear UI
blending without a gamma shader or extra render pass. Other color-space/compositing
contracts fail explicitly. Successful present submission acknowledgements feed
frontend FPS counters; these are not physical display timing measurements.

For a real-window software-keyboard check, run:

```sh
cargo run -p nixe-cli --release --example native-keyboard
```

Configure the actual validation layer path when it is not installed globally.
The example uses the normal native renderer and Horizon owner. Broader host,
external-video and commercial-game acceptance remain tasks 13/14/17.
