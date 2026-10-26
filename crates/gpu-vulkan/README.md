# Native Vulkan renderer

`nixe-gpu-vulkan` owns the Vulkan instance, device, resources, recording and
presentation used by Nixe. It consumes immutable native commands and captured
shader environments through `nixe-gpu`; it has no Maxwell decoder or Horizon
dependency.

## Device and ownership

Vulkan 1.3 uses core synchronization2 and dynamic rendering. Vulkan 1.2 requires
the corresponding KHR extensions. Both require timeline semaphores and the actual
feature bits. Device initialization freezes the enabled compiler profile, including
numerical, subgroup and descriptor capabilities.

Window creation occurs in the event loop's resume callback, before surface-aware
device and queue selection. Headless startup creates neither a surface nor a
window. The graphics/compute device moves to Horizon's GPU owner, which consumes mutable
guest resources before delegating native work. The recording worker owns command
pools, upload/main recording and queue submission. Recyclable byte-sized chunks
contain owned native parameters and retained handles; queues and native pools
grow without capacity-driven submits or waits. A separate WSI worker owns
swapchain acquire, present and recreation. Presentation may share the graphics
queue, with native queue calls externally synchronized. Windowed hosts require swapchain maintenance and its instance dependencies;
missing support is an explicit startup error.

Resources retain the device, allocator and instance until their last use. One
monotonic timeline controls command-slot reuse, staging and resource retirement.
A global streaming/reusable staging pool preserves uploads and deferred CPU
readbacks independently of command-context reuse. Work-count cadence and real
dependencies drive submission; there is no elapsed-time submit deadline.
Ordinary submission and shutdown do not use device-wide idle. CPU visibility requests resolve only the required producers;
acceptance, GPU completion and guest fence publication remain separate events.

## Compilation and rendering

`nixe-shader-recompiler` is the guest shader compiler. Captured instructions and
observations are replayed on bounded workers, linked and emitted as owned stage
artifacts. SPIR-V, descriptor provenance, interfaces and execution metadata survive
scratch reuse and disk reload. Stage caches share unchanged linked stages before
emission. Vulkan window-depth legalization and synthetic quad/default tessellation
stages use the same owned artifact contract.

[Shader, descriptor and pipeline caches](src/cache/README.md) retain prepared
references and immutable descriptors. Enabled fast-linking graphics pipeline
libraries share relevant subsets; other hosts use monolithic construction.
Independent checked-in [host shaders](shaders) implement presentation, UI, transfers
and format conversion.

`VulkanResidency` tracks canonical byte ownership separately from native Vulkan
hazards. Buffer/image aliases reconstruct exact GPU-authored ranges without CPU
readback. Partial writes preserve unrelated texels, aspects and padding. Native
resource caches obey the device budget; transient pools grow and recycle by
actual completion. Undefined
opaque contents cannot be replaced with a clear or a canonical import.

[Native execution](src/residency/execution/README.md) covers draw/dispatch,
indexed and indirect arguments, raster state, depth/stencil, multisample resolves,
transfers and conversion. Compatible rendering scopes and prepared state are reused
across frontend deliveries. Unsupported guest/resource combinations fail at
consumption with diagnostic context; host feature availability alone does not
establish guest instruction support.

Scanout uses immutable versioned snapshots ordered after guest rendering. Each
offscreen producer is submitted independently of swapchain image acquisition;
completed unleased snapshots recycle under the ordinary allocation budget.
Actual pressure reselects released storage after its GPU use finishes, preserving
live leases. WSI availability and idle results wake the owner after publication;
presentation progress does not require another guest or host packet.
There is no fixed snapshot-count failure threshold. Egui,
screenshots and video conversion use the same native device and owner. Routine
presentation performs no CPU readback. PNG screenshots use incremental readback
followed by encoding on a CPU worker. Presentation leases and WSI fences control
reuse independently of guest buffer-slot release.

## Running and checking

```sh
cargo cli run <title>
cargo cli --headless run <title>
cargo cli --validation vulkan-check
cargo test -p nixe-gpu-vulkan -- --test-threads=1 --nocapture
cargo test -p nixe-gpu-vulkan captured_probe_modules_validate_without_a_gpu
```

GPU tests require physical hardware and Khronos validation. A host without hardware
prints a skip diagnostic; the harness reports that return as `ok`, which is not
hardware validation. Available-device initialization or validation failures fail
the test. CPU shader checks use `spirv-val` and do not require a GPU. Run hardware
tests serially: concurrent device tests have produced a driver/loader crash whose
cause has not been established.

The Linux Cargo runner uses `dev-tools/vulkan-loader/install/lib` when present.
`./scripts/build-vulkan-loader.sh` builds that local loader. Configure the validation
layer path when the layer is not installed globally.

Rebuild/check independent host modules with the pinned SDK toolchain:

```sh
python scripts/build-vulkan-shaders.py --sdk-bin /path/to/sdk/x86_64/bin
python scripts/build-vulkan-shaders.py --sdk-bin /path/to/sdk/x86_64/bin --check
```

The optional `binding-probe` example compares descriptor transport strategies. The
ignored pipeline-construction probe checks exact pixels and reports cold creation
and warm CPU/GPU timings. Neither synthetic probe establishes commercial workload
parity or verification on untested host platforms.
