# Triangle-to-Commercial Graphics Implementation Plan

Status: planned; no task in this document is implemented merely because the
current software-framebuffer path exists

Primary milestone: execute and present the first correct frame from the
`graphics/opengl/simple_triangle` homebrew example

Long-term objective: grow the same architecture into a correct and efficient
graphics implementation for commercial software without replacing the memory,
submission, synchronization, or backend boundaries established for the
triangle

`CONTRIBUTING.md` is the primary reference for how to contribute.

Related documents:

- [`../docs/switch1/Graphics and Display Architecture.md`](<../docs/switch1/Graphics and Display Architecture.md>)
- [`../docs/switch1/Memory Architecture.md`](<../docs/switch1/Memory Architecture.md>)
- [`../docs/CPU Dynamic Recompiler Technical Specification.md`](<../docs/CPU Dynamic Recompiler Technical Specification.md>)
- [`interpreter-roadmap.md`](interpreter-roadmap.md)

Pinned implementation references:

- [`simple_triangle/main.cpp`](https://github.com/switchbrew/switch-examples/blob/669786898205b7beb25ff1731e72982e6d0397d3/graphics/opengl/simple_triangle/source/main.cpp)
- [`simple_triangle/Makefile`](https://github.com/switchbrew/switch-examples/blob/669786898205b7beb25ff1731e72982e6d0397d3/graphics/opengl/simple_triangle/Makefile)
- [libnx NVIDIA service wrapper](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/nv.c)
- [libnx NVIDIA ioctl definitions](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/nvidia/ioctl.h)
- [libnx `nvhost-ctrl-gpu` wrappers](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/ioctl/nvhost-ctrl-gpu.c)
- [versioned Switchbrew NV services reference](https://switchbrew.org/w/index.php?title=NV_services&oldid=14790)
- [Switchbrew GPU classes reference](https://switchbrew.org/wiki/GPU_Classes)

Any implementation that relies on additional external information must add a
nearby commit-pinned or versioned source. A passing external application is not
evidence that guessed data or ignored operations are correct.

## 1. Objective

The first target is intentionally small at the API level:

1. Initialize EGL and the guest Mesa/Nouveau graphics stack.
2. Create an OpenGL 4.3 context and window surface.
3. Compile the example vertex and fragment shaders.
4. Create and fill one vertex buffer.
5. Clear one color render target.
6. Draw one three-vertex triangle.
7. Submit and complete the resulting Maxwell GPU work.
8. Queue the rendered image through the existing Binder/BufferQueue path.
9. Publish a correct host-visible frame while input and application lifetime
   continue to work.

The example is not a high-level OpenGL emulation target. EGL, OpenGL, Mesa, and
Nouveau execute as guest code and generate NVIDIA driver operations, GPU virtual
addresses, pushbuffers, shaders, and synchronization. Nixe implements the
guest-visible driver and GPU semantics below that software.

The milestone is reached only when the rendered pixels originate from decoded
and executed guest GPU work. Replacing the draw with a host-generated triangle,
recognizing the executable, intercepting guest OpenGL calls, returning success
for unimplemented ioctls, or publishing a fabricated frame does not satisfy the
milestone.

## 2. Long-term definition of success

The architecture established here must be able to grow toward commercial
software with the following properties:

- CPU and GPU virtual aliases converge on canonical guest backing identities.
- Host pointers never appear in guest address, mapping, descriptor, or command
  state.
- GPU mappings have explicit lifetime and generation tracking.
- CPU/GPU visibility is correct on discrete and unified-memory hosts.
- Guest synchronization is represented independently from host API completion.
- NVIDIA driver ABI parsing is separate from Maxwell command execution.
- Switch 1 hardware semantics are separate from host rendering APIs.
- Future console frontends can reuse verified common infrastructure without
  inheriting Maxwell assumptions.
- The accelerated backend is replaceable by a deterministic headless validator
  or another host graphics API.
- Unsupported service commands, ioctls, GPU methods, formats, shader
  instructions, and synchronization states stop with typed host-side errors.
- Guest-visible failures are returned only when they model a verified guest
  failure condition.
- Correctness tests do not require proprietary games, firmware, keys, or shader
  binaries.
- Performance optimizations can change residency, caching, batching, and
  translation strategy without changing guest-visible behavior.

This plan covers the graphics-specific path. A commercial title may also expose
unrelated CPU, kernel, service, audio, storage, or applet gaps; those remain in
their respective plans.

## 3. Non-negotiable invariants

### 3.1 Canonical memory identity

The same guest physical bytes have one stable identity even when reachable
through several CPU mappings, GPU mappings, `nvmap` handles, buffer views, image
views, or host resource mirrors.

```text
CPU virtual address ----+
                        |
nvmap allocation -------+--> canonical backing range
                        |       page identities
GPU virtual address ----+       mapping generations
                                visibility versions
                                      |
                         +------------+-------------+
                         |                          |
                    host RAM                 host GPU resource
```

No subsystem may infer identity from a CPU virtual address, GPU virtual address,
`nvmap` handle, host allocation address, or host API object alone.

### 3.2 Address-domain separation

Use distinct types for at least:

- guest CPU virtual addresses;
- guest physical page or allocation identities;
- guest GPU virtual addresses;
- `nvmap` handles and exported IDs;
- NVIDIA service file descriptors;
- GPU channel identifiers;
- guest syncpoint identifiers and values;
- backend resource and submission identifiers; and
- host pointers, which must remain internal implementation details.

Checked conversions must occur at explicit boundaries. Generic `u64` values
must not silently cross address domains.

### 3.3 Completion is not visibility

A guest fence or syncpoint says that modeled GPU work reached a defined point.
It does not by itself guarantee that:

- a host queue completed;
- CPU caches observe GPU writes;
- a device-local mirror has been downloaded;
- a BufferQueue consumer may acquire the image; or
- a VI refresh boundary occurred.

The synchronization layer must perform the host operations required to make the
guest-visible transition true before reporting it.

### 3.4 Mapping lifetime is explicit

GPU work retains versioned mapping identities or validated backing ranges. It
must not retain raw CPU pointers across unmap, remap, backing migration, process
teardown, or resource-cache eviction.

Unmapping or freeing an allocation must define what happens to in-flight work
according to verified guest behavior. It must never leave a dangling host
reference.

### 3.5 Resource views do not own memory

Buffers, textures, render targets, depth targets, descriptor tables, and command
buffers are interpretations of backing ranges. Their format, dimensions,
layout, kind, pitch, mip levels, and plane offsets belong to the view.

Destroying or invalidating a view does not implicitly destroy an allocation.
Writing through one aliased view invalidates or updates every conflicting cached
view.

### 3.6 Console behavior is not backend behavior

Horizon, `nvdrv`, `nvmap`, GPU virtual-memory rules, Maxwell packets, classes,
methods, and shader ISA are guest-facing behavior. `wgpu`, Vulkan, Metal,
Direct3D, command encoders, staging buffers, and host fences are backend
implementation choices.

Neither side may leak its identifiers or failure model into the other.

### 3.7 No false progress

Every newly accepted operation must have one of these outcomes:

- complete verified semantics;
- a verified guest-visible error for the supplied invalid state or arguments;
- or a typed fatal host diagnostic describing the missing emulator semantics.

Logging and continuing, returning zero-filled capability data without a
hardware basis, immediately signaling unfinished work, and silently ignoring
unknown GPU methods are prohibited.

## 4. Target ownership and dependency boundaries

The exact crate split may be introduced incrementally, but the final dependency
direction must follow this model:

```text
                         applications
                              |
                           runtime
                    +---------+---------+
                    |                   |
                 Horizon              CPU
                    |                   |
             Switch nvdrv ABI           |
                    |                   |
             Switch 1 GPU frontend       |
                    +---------+---------+
                              |
                      neutral memory core
                              |
                      neutral GPU contracts
                        /             \
             headless validator    host GPU backend
                                         |
                                wgpu/Vulkan/Metal/etc.

VI + Binder + compositor --> host-independent video presentation
```

The intended package responsibilities are:

- `nixe-memory`: canonical physical backing identities, CPU mapping contracts,
  generation tracking, visibility hooks, and pointer-free range access.
- `nixe-cpu`: CPU architecture, execution, and CPU-specific memory access
  behavior; it consumes `nixe-memory` rather than owning cross-device policy.
- `nixe-horizon`: HIPC/CMIF, VI, Binder, BufferQueue, `nvdrv` service wire ABI,
  process permissions, handles, and verified guest results.
- a Switch 1 GPU frontend crate or isolated module: NVIDIA device state, GPU
  address spaces, channels, GPFIFO, Maxwell command processing, and shader
  decoding.
- a neutral GPU contract crate or isolated module: typed resources, accesses,
  operations, submission dependencies, completion tokens, backend capability
  negotiation, and diagnostics.
- an accelerated backend crate: implementation of neutral GPU operations using
  `wgpu` initially, without guest ABI or Maxwell parsing.
- a headless backend: deterministic validation, resource-state checking, and
  small software results required by tests.
- `nixe-video`: immutable host-ready frames, display timing, and presentation
  mailboxes; it does not own guest GPU execution.
- `nixe-video-winit`: window-system integration and final host presentation
  only.

If moving all memory implementation out of `nixe-cpu` at once creates an unsafe
review surface, the transition may use a temporary adapter. The adapter must
expose canonical page identities and generations, must not expose `Process` or
host pointers to the GPU frontend, and must have a tracked removal task in the
same block. It must not become the permanent architecture.

### 4.1 Proposed file and package structure

The following structure records intended ownership and dependency direction. It
is a proposal, not a requirement to create empty crates or preserve every
filename permanently. `CONTRIBUTING.md` explicitly permits module-boundary
changes when verified behavior makes a better division clear.

```text
crates/
|-- memory/
|   |-- Cargo.toml
|   `-- src/
|       |-- lib.rs
|       |-- address.rs
|       |-- backing.rs
|       |-- range.rs
|       |-- mapping.rs
|       |-- generation.rs
|       |-- visibility.rs
|       `-- access.rs
|
|-- cpu/
|   `-- src/
|       `-- memory/
|           `-- ... CPU access, translation, and execution adapters
|
|-- gpu/
|   |-- Cargo.toml
|   `-- src/
|       |-- lib.rs
|       |-- capability.rs
|       |-- address.rs
|       |-- allocation.rs
|       |-- resource.rs
|       |-- view.rs
|       |-- access.rs
|       |-- command.rs
|       |-- submission.rs
|       |-- synchronization.rs
|       |-- backend.rs
|       |-- diagnostics.rs
|       |-- capture.rs
|       `-- shader/
|           |-- mod.rs
|           |-- ir.rs
|           `-- verify.rs
|
|-- gpu-maxwell/
|   |-- Cargo.toml
|   `-- src/
|       |-- lib.rs
|       |-- profile.rs
|       |-- address_space.rs
|       |-- channel.rs
|       |-- gpfifo.rs
|       |-- pushbuffer/
|       |   |-- mod.rs
|       |   |-- packet.rs
|       |   `-- dispatch.rs
|       |-- engines/
|       |   |-- mod.rs
|       |   |-- threed/
|       |   |   |-- mod.rs
|       |   |   |-- state.rs
|       |   |   |-- resource.rs
|       |   |   `-- draw.rs
|       |   |-- compute.rs
|       |   |-- copy.rs
|       |   `-- inline_to_memory.rs
|       |-- shader/
|       |   |-- mod.rs
|       |   |-- decode.rs
|       |   |-- semantics.rs
|       |   `-- translate.rs
|       `-- diagnostics.rs
|
|-- gpu-headless/
|   |-- Cargo.toml
|   `-- src/
|       |-- lib.rs
|       |-- validator.rs
|       |-- timeline.rs
|       `-- reference.rs
|
|-- gpu-wgpu/
|   |-- Cargo.toml
|   `-- src/
|       |-- lib.rs
|       |-- adapter.rs
|       |-- capability.rs
|       |-- residency.rs
|       |-- allocation.rs
|       |-- resource.rs
|       |-- shader.rs
|       |-- pipeline.rs
|       |-- command.rs
|       |-- synchronization.rs
|       `-- readback.rs
|
|-- horizon/
|   `-- src/
|       |-- nvdrv/
|       |   |-- mod.rs
|       |   |-- service.rs
|       |   |-- session.rs
|       |   |-- device.rs
|       |   |-- ioctl.rs
|       |   |-- nvmap.rs
|       |   |-- nvhost_ctrl.rs
|       |   |-- nvhost_ctrl_gpu.rs
|       |   |-- nvhost_as_gpu.rs
|       |   `-- nvhost_gpu.rs
|       |-- vi/
|       |   `-- ... display and layer service semantics
|       |-- binder/
|       |   `-- ... Parcel and BufferQueue service semantics
|       `-- graphics.rs
|
|-- video/
|   `-- ... host-independent frames, clocks, and mailboxes
|
`-- video-winit/
    `-- ... window-system and final presentation integration
```

Unit tests should remain beside the implementation they constrain. Cross-crate
synthetic tests should live in the `tests/` directory of the highest-level crate
whose public contract they exercise. Caller-owned binaries, GPU captures, and
commercial shader data must remain outside the repository.

The important distinction between the proposed packages is:

- `nixe-memory` identifies and exposes guest storage without CPU, Horizon,
  Maxwell, or host graphics policy.
- `nixe-gpu` defines host-independent GPU resources, operations, timelines,
  shader IR, and backend contracts without NVIDIA or Horizon ABI.
- `nixe-gpu-maxwell` implements Switch 1 GPU behavior and lowers it into
  `nixe-gpu` contracts.
- `nixe-gpu-headless` validates the neutral contract and provides deterministic
  completion or limited reference execution for tests.
- `nixe-gpu-wgpu` implements the neutral backend using host graphics APIs.
- `nixe-horizon::nvdrv` parses the guest service and ioctl ABI, owns Horizon
  permissions and handles, and invokes typed Switch 1 frontend operations.

Placing backend-facing shader IR in `nixe-gpu` does not require a future console
frontend to reuse the Maxwell decoder or its source-level semantics. Only
verified, already-lowered operations belong in the neutral IR; console-specific
instructions and translation rules remain in their frontend.

`nixe-horizon::nvdrv` may know the numeric ioctl ABI, while
`nixe-gpu-maxwell` may know the semantic operation. For example,
`nvhost_as_gpu.rs` validates and decodes a map ioctl into a typed mapping
request; `address_space.rs` applies the Switch 1 GPU mapping rules. Neither
component calls `wgpu`.

#### Allowed dependency direction

In the following graph, `A -> B` means that crate or component A may depend on
B:

```text
nixe-cpu          -> nixe-memory
nixe-runtime      -> nixe-cpu
nixe-runtime      -> nixe-memory

nixe-gpu          -> nixe-memory
nixe-gpu-maxwell  -> nixe-gpu
nixe-gpu-maxwell  -> nixe-memory
nixe-gpu-headless -> nixe-gpu
nixe-gpu-wgpu     -> nixe-gpu

nixe-horizon      -> nixe-runtime
nixe-horizon      -> nixe-memory
nixe-horizon      -> nixe-gpu
nixe-horizon      -> nixe-gpu-maxwell
nixe-horizon      -> nixe-video

application composition root -> nixe-horizon
application composition root -> nixe-gpu-wgpu or nixe-gpu-headless
application composition root -> nixe-video-winit
nixe-video-winit             -> nixe-video
```

The application composition root constructs the selected backend and injects
only neutral interfaces into the emulated system. This prevents Horizon or the
Switch 1 frontend from selecting a host API.

The inverse dependencies are prohibited:

- `nixe-memory` must not depend on CPU, runtime, Horizon, Maxwell, video, or a
  host graphics API.
- `nixe-gpu` must not depend on Horizon, Maxwell, `wgpu`, `winit`, or a concrete
  host backend.
- `nixe-gpu-maxwell` must not depend on Horizon, `wgpu`, `winit`, or VI.
- `nixe-gpu-wgpu` and `nixe-gpu-headless` must not parse ioctls, GPU packets, or
  Maxwell shaders.
- `nixe-video` and `nixe-video-winit` must not advance guest GPU syncpoints or
  own guest GPU address spaces.
- `nixe-cpu` and generated CPU code must not submit host graphics work.

#### Incremental introduction

Packages and directories should appear when they acquire a real tested
responsibility:

| Block | Structure introduced or expanded                                                                                              |
| ----- | ----------------------------------------------------------------------------------------------------------------------------- |
| T0    | Start the diagnostics portion of `nixe-gpu` with pointer-free coverage vocabulary and stable identifier formatting.           |
| T1    | Start `nixe-gpu-maxwell::profile`, or an isolated equivalent that can move without importing Horizon ABI.                     |
| T2    | Introduce `nixe-memory`; move neutral identities, ranges, generations, and visibility contracts before GPU mappings use them. |
| T3    | Split `nixe-horizon::nvdrv` into service, session, device, ioctl, and `nvmap` ownership modules.                              |
| T4    | Add `nixe-gpu-maxwell::address_space` and the `/dev/nvhost-as-gpu` ABI adapter.                                               |
| T5    | Introduce the timeline and synchronization portion of `nixe-gpu`, plus deterministic fake completion used by tests.           |
| T6    | Add Maxwell channel and GPFIFO state and the `/dev/nvhost-gpu` ABI adapter.                                                   |
| T7    | Add the pushbuffer decoder, class dispatch, diagnostics, and capture boundaries.                                              |
| T8    | Complete the initial neutral resource/backend contracts and introduce `nixe-gpu-headless`.                                    |
| T9    | Add Maxwell engine modules as real class coverage appears. Do not create empty handlers for unreached engines.                |
| T10   | Add Maxwell shader decoding and neutral shader IR/verification in their respective crates.                                    |
| T11   | Introduce `nixe-gpu-wgpu` and keep all concrete host resource and residency objects there.                                    |
| T12   | Expand existing VI, Binder, `nixe-video`, and presentation modules without moving GPU execution into them.                    |

A component may begin as an internal module when that keeps a change reviewable.
It may do so only if its public inputs and outputs already respect the intended
dependency direction. Moving the module into its target crate must then be a
mechanical ownership change rather than an architectural rewrite.

Do not pre-create empty engine, backend, or shader modules to make the tree look
complete. The structure documents ownership; implemented behavior and tests
justify files.

### 4.2 Future-console boundary

Only behavior supported by verified knowledge is shared. A future console may
reuse:

- canonical memory identity and range types;
- CPU/GPU visibility and residency machinery;
- generic submission dependencies and timelines;
- resource lifetime and backend contracts;
- presentation and diagnostics;
- testing and capture infrastructure.

It must be free to provide distinct:

- service or driver ABI;
- capability profile;
- GPU virtual-memory rules;
- channel and packet decoder;
- GPU classes and state;
- shader ISA;
- cache and ordering behavior.

Do not add product-name conditionals throughout common code. Select immutable,
typed profiles and frontend implementations at construction.

## 5. Delivery and validation policy

Each block below has three validation layers where applicable:

1. Unit tests for parsing, state transitions, overflow, malformed input, and
   typed unsupported behavior.
2. Synthetic integration tests using redistributable command streams,
   allocations, shaders, and expected images created specifically for Nixe.
3. External `simple_triangle` executions performed by the user.

The user-run checkpoint is diagnostic, not the proof of correctness. A real
application progressing further does not replace the first two layers.

Every external checkpoint should record:

- Nixe revision;
- homebrew source revision and build identity when available;
- instruction count and stop reason;
- device path, ioctl, channel, class, method, or shader location;
- bounded GPU mapping and submission context;
- whether teardown released resources;
- and a GPU trace artifact once capture support exists.

Nixe must continue to stop at the first unsupported semantic boundary. The next
observed failure selects work only after it is placed in the correct block.

## 6. Ordered implementation blocks

### Block T0: Reproducible baseline and source manifest

Purpose: make subsequent progress attributable to implemented semantics rather
than changing binaries, profiles, or undocumented assumptions.

- [x] **TRI-000** Record the pinned `simple_triangle` source and build
      dependencies used as the milestone reference without committing the
      resulting NRO.
- [x] **TRI-001** Record the pinned libnx ioctl layouts and versioned Switchbrew
      service tables used by the first initialization path.
- [x] **TRI-002** Add a graphics coverage vocabulary that distinguishes service
      command, device open, ioctl, GPU packet, class method, shader instruction,
      resource format, and backend capability gaps.
- [x] **TRI-003** Preserve a bounded, pointer-free diagnostic for the current
      stop at `/dev/nvhost-ctrl-gpu` ioctl `0xc0184706`.
- [x] **TRI-004** Define stable formatting for CPU VA, GPU VA, allocation,
      mapping generation, channel, GPFIFO entry, class, method, and syncpoint
      values before traces begin to depend on ad hoc strings.
- [x] **TRI-005** Confirm that a failing graphics process still performs
      deterministic handle, allocation, queue, layer, and backend teardown.

Exit criterion: one external run can be tied to exact public sources and its
first unsupported graphics operation is classified without host-pointer data.

External run checkpoint: none; the current failure is the baseline.

### Block T1: Immutable Switch 1 GPU capability profile

Purpose: answer hardware discovery requests from one internally consistent,
testable profile rather than scattering constants through ioctls.

- [x] **TRI-010** Define immutable GPU profile identifiers and typed topology,
      class, virtual-address, page-size, Z-cull, cache, and shader capability
      fields required by verified discovery operations.
- [x] **TRI-011** Populate a Switch 1 GM20B profile solely from pinned public
      references.
- [x] **TRI-012** Refactor existing `GetCharacteristics`, Z-cull context-size,
      and Z-cull-info replies to read the profile while preserving their exact
      wire layouts.
- [x] **TRI-013** Implement `NVGPU_GPU_IOCTL_GET_TPC_MASKS`
      (`0xc0184706`) with its 24-byte in/out ABI, inline output, argument
      validation, and profile-derived masks.
- [x] **TRI-014** Check profile consistency: GPC count, TPC count and masks,
      GPU VA width, big-page sizes, advertised classes, and shader architecture
      must not contradict one another.
- [x] **TRI-015** Test exact output bytes, preserved input fields, malformed
      sizes, invalid arguments, and unsupported profile data.
- [x] **TRI-016** Keep every other unknown `nvhost-ctrl-gpu` ioctl fatal and
      typed.

Exit criterion: all implemented discovery ioctls are generated from one
coherent profile, and `0xc0184706` has faithful semantics rather than a
call-site-specific response.

External run checkpoint T1 (2026-07-26):

- Nixe base revision:
  `3a06d6d101d504f50976535cd57c738e920a34a5`, with the uncommitted T1
  implementation listed above.
- Workload: the retained `simple_triangle` artifact identified by the T0
  manifest.
- Progress: `893774` guest instructions; the final Horizon SVC dispatch
  executed `159` instructions.
- Next typed boundary:
  `graphics-gap=device-open nvdrv device open is not implemented:
path="/dev/nvhost-as-gpu"`.
- GPU context: capability discovery completed; no GPU address-space descriptor,
  channel, mapping, or submission had been created at the stop.
- Teardown: `25` handles, `1` layer, `1` queue, `0` pending frames, `3` nvdrv
  file descriptors, and `0` nvmap allocations were released.
- Trace: disabled; no GPU trace exists because execution stopped before GPU
  address-space or submission state.

### Block T2: Device-neutral canonical memory foundation

Purpose: establish the shared-memory model before GPU mappings or backend
resources make the current CPU-owned representation difficult to replace.

#### T2 delivery batches

T2 is delivered as the following ordered review batches. Each batch must keep
the workspace passing before the next one begins. The grouping does not relax
task dependencies or permit a temporary interface to expose CPU internals,
host pointers, or CPU virtual addresses as backing identity.

1. **T2-A — Current-memory inventory**
   - **TRI-020**
   - Record the existing ownership, identity, mapping, permission, generation,
     exclusive-monitor, and teardown contracts without changing behavior.
   - Confirm or adjust the remaining batch boundaries from the observed code
     before introducing the neutral package.
2. **T2-B — Neutral identity and generation boundary**
   - **TRI-021**, **TRI-022**, and **TRI-026**
   - Introduce `nixe-memory`, move or wrap physical backing identities, and
     distinguish mapping generations from content generations.
   - Any temporary CPU adapter must be explicit, pointer-free at its public
     boundary, and removable without changing the neutral contract.
3. **T2-C — Checked ranges, translation, and lifetime**
   - **TRI-023**, **TRI-024**, and **TRI-029**
   - Represent page-spanning canonical backing ranges, translate validated CPU
     virtual ranges without exposing host pointers, and define teardown and
     retention rules.
4. **T2-D — Non-CPU access and visibility**
   - **TRI-025**, **TRI-027**, and **TRI-028**
   - Add device access declarations, conservative visibility state, and the CPU
     slow path required when a non-CPU device owns newer contents.
5. **T2-E — Acceptance and regression**
   - **TRI-030** and **TRI-031**
   - Exercise aliases, page crossings, permissions, generations, remaps,
     concurrency, overflow, and teardown, then prove the CPU interpreter and
     software-framebuffer paths remain behaviorally unchanged.

- [x] **TRI-020** Inventory the existing physical-page identity, allocation,
      mapping, permission, generation, and exclusive-monitor contracts in
      `nixe-cpu`. See the
      [canonical memory foundation audit](<Canonical Memory Foundation Audit.md>).
- [x] **TRI-021** Introduce a neutral memory package or equivalent dependency
      boundary consumed by CPU, runtime, and GPU code.
- [x] **TRI-022** Move or wrap guest physical page and backing identities
      without changing current CPU-visible behavior.
- [x] **TRI-023** Define checked backing ranges that can span physical pages and
      retain per-segment identity, offset, size, permissions, and generation.
- [x] **TRI-024** Provide pointer-free translation from a process CPU virtual
      range to canonical backing segments for validated `nvmap` allocation.
- [x] **TRI-025** Define read and write access declarations for non-CPU devices,
      including the point at which data must become visible.
- [x] **TRI-026** Define mapping and content generations separately so a byte
      write does not masquerade as an address-space remap and vice versa.
- [x] **TRI-027** Add a conservative per-range visibility state capable of
      representing `Clean`, `CpuNewer`, `GpuNewer`, and invalid/conflicting
      state.
- [x] **TRI-028** Route CPU access to a slow path when a range is `GpuNewer`,
      leaving host API work outside generated or interpreted CPU semantics.
- [x] **TRI-029** Define allocation teardown so CPU unmap, process exit, and GPU
      references cannot leave dangling backing access.
- [x] **TRI-030** Test CPU aliases, page-crossing ranges, permissions, remaps,
      generations, concurrent observations, overflow, and teardown.
- [x] **TRI-031** Prove that the CPU interpreter and existing software
      framebuffer acceptance tests remain behaviorally unchanged.

Exit criterion: GPU code can identify and access validated backing ranges
without depending on `ExceptionProcessContext`, a CPU virtual address as
identity, or a host pointer.

External run checkpoint: none unless the refactor changes existing homebrew
behavior; `hello-world` remains a regression target.

### Block T3: Semantic `nvdrv` device and `nvmap` ownership model

Purpose: separate Horizon wire transport from persistent NVIDIA device
semantics and bind `nvmap` objects to canonical memory.

#### T3 delivery batches

T3 is delivered as the following ordered review batches. The split keeps wire
ABI refactoring, shared descriptor ownership, and `nvmap` lifetime changes
independently testable. No intermediate batch may return success for an
unsupported device or ioctl, identify an allocation by CPU virtual address, or
weaken canonical-backing retention.

1. **T3-A — Semantic service and device boundary**
   - **TRI-040**, **TRI-041**, **TRI-047**, and **TRI-049**
   - Split `nvdrv` into service, session, device, ioctl, diagnostics, and
     `nvmap` ownership modules while keeping HIPC/CMIF decoding and response
     encoding in the Horizon wire layer.
   - Introduce typed device descriptors, ownership, lifecycle, permission, and
     contextual error vocabulary before changing session or allocation
     lifetime.
2. **T3-B — Shared sessions and descriptor ownership**
   - **TRI-042**
   - Separate one service connection from its shared client state and
     descriptor table, then define clone, descriptor access, close, and
     teardown behavior with focused tests.
3. **T3-C — Canonical `nvmap` object lifetime**
   - **TRI-043**, **TRI-044**, and **TRI-046**
   - Make retained canonical backing the allocation identity, separate objects
     from handles and exported IDs, and introduce future view metadata without
     assigning buffer or image interpretation to the allocation.
   - Adapt current software-framebuffer consumption to canonical ranges so CPU
     virtual address remains optional diagnostic or mapping metadata.
4. **T3-D — Validation and acceptance**
   - **TRI-045** and **TRI-048**
   - Complete operation validation and exercise cloned sessions, duplicate and
     imported references, aliases, premature free, final release, ownership,
     and teardown before verifying the block exit criterion.

- [x] **TRI-040** Split HIPC buffer decoding and response encoding from semantic
      NVIDIA service/device dispatch.
- [x] **TRI-041** Define typed device descriptors with owning session/process,
      device kind, lifecycle, and permission profile.
- [x] **TRI-042** Preserve shared state across cloned `nvdrv` sessions while
      retaining correct file-descriptor ownership and close behavior.
- [x] **TRI-043** Replace `NvMapAllocation.cpu_address` as allocation identity
      with a validated canonical backing-range reference; retain CPU VA only as
      diagnostic or mapping metadata when useful.
- [x] **TRI-044** Model `nvmap` object lifecycle separately from handles and
      exported IDs, including reference counts, import, free, and process
      teardown.
- [x] **TRI-045** Validate size, alignment, heap mask, cache flags, kind,
      permissions, page coverage, duplicate allocation, and ownership on every
      implemented `nvmap` operation.
- [x] **TRI-046** Define allocation/view metadata required later for pitch,
      block-linear kind, planes, and image layout without assigning those
      interpretations to the allocation itself.
- [x] **TRI-047** Add typed errors carrying device, request, fd, allocation, and
      validation reason without exposing a host pointer.
- [x] **TRI-048** Test cloned sessions, duplicate handles, imported IDs,
      aliased CPU mappings, premature free, final release, and invalid ownership.
- [x] **TRI-049** Keep unsupported devices and ioctls fatal rather than
      returning a generic NVIDIA error.

Exit criterion: an `nvmap` object is a lifetime-managed reference to canonical
guest bytes, and NVIDIA service ABI code no longer owns the underlying memory
model.

External run checkpoint T3: run the demo if its next required operations are in
the implemented `nvmap` subset and report the next typed boundary.

### Block T4: GPU virtual address-space manager

Purpose: resolve every GPU address through an explicit, versioned mapping to the
same backing used by CPU and `nvmap`.

#### T4 delivery batches

T4 is delivered as the following ordered review batches. The split keeps the
Horizon ioctl ABI separate from Maxwell address-space semantics and prevents
mapping operations from being accepted before their ownership, lifetime, and
canonical-backing rules are representable.

1. **T4-A — Profile-bound address-space foundation**
   - **TRI-060** and **TRI-061**
   - Add the `/dev/nvhost-as-gpu` descriptor and persistent address-space
     identity, bind each instance to one immutable GPU profile, and introduce
     checked profile-sized GPU virtual addresses.
   - Unknown address-space ioctls remain fatal until their semantic operations
     are implemented by a later batch.
2. **T4-B — VA allocation and regions**
   - **TRI-062** and **TRI-063**, plus the reserve and free portion of
     **TRI-064**
   - Implement initialization, small-page and big-page regions, allocation,
     reservation, release, and state-derived region queries.
   - **TRI-064** remains incomplete until mapping and unmapping semantics are
     delivered by T4-C.
3. **T4-C — Mapping semantics and lifetime**
   - **TRI-064**, **TRI-065**, **TRI-066**, **TRI-068**, and **TRI-069**
   - Implement fixed and allocated maps, remap, unmap, and free semantics over
     retained canonical `nvmap` backing with explicit permissions, page size,
     kind, ownership, aliasing, and mapping generations.
4. **T4-D — Resolution, diagnostics, and acceptance**
   - **TRI-067**, **TRI-070**, and **TRI-071**
   - Add maximal-range and scatter/gather resolution, deterministic bounded
     mapping diagnostics, and the complete boundary, stale-generation, alias,
     and teardown test matrix.
   - Verify the block exit criterion before requesting the external T4
     checkpoint.

- [x] **TRI-060** Implement `/dev/nvhost-as-gpu` descriptor lifecycle and bind
      it to an immutable GPU profile.
- [x] **TRI-061** Define a distinct `GpuVirtualAddress` type with profile-sized,
      checked arithmetic.
- [x] **TRI-062** Implement verified address-space allocation and initialization
      operations, including small and big page sizes.
- [x] **TRI-063** Implement VA-region queries from profile and address-space
      state rather than returning fixed unowned ranges.
- [x] **TRI-064** Implement reserve, fixed-map, allocated-map, remap, unmap, and
      free semantics in the order exposed by the demo.
- [x] **TRI-065** Represent one GPU mapping as GPU VA range, backing range,
      permissions, page size, kind, mapping generation, and allocation
      reference.
- [x] **TRI-066** Reject overlap, misalignment, overflow, invalid page size,
      invalid kind, unmapped `nvmap`, ownership mismatch, and partial invalid
      operations with verified guest results where applicable.
- [x] **TRI-067** Provide maximal-range lookup and checked scatter/gather
      resolution across GPU mappings.
- [x] **TRI-068** Define in-flight mapping retention and invalidation rules so
      unmap cannot create use-after-free.
- [x] **TRI-069** Ensure CPU VA aliases and multiple GPU VA aliases resolve to
      identical backing identities.
- [x] **TRI-070** Add deterministic mapping diagnostics and bounded mapping
      dumps.
- [x] **TRI-071** Test holes, overlaps, boundary addresses, page crossings,
      remaps, stale generations, aliases, and teardown.

Exit criterion: a synthetic GPU VA read or write resolves to the correct
canonical bytes, and stale or invalid mappings fail without host memory access.

External run checkpoint T4: run the demo after its observed address-space ioctl
sequence is covered and return the next typed stop.

### Block T5: Guest syncpoints, fences, events, and timelines

Purpose: model the completion vocabulary before a channel can submit work or a
BufferQueue can consume GPU-produced images.

#### T5 delivery batches

T5 is delivered as the following ordered review batches. The split establishes
the neutral completion vocabulary before adding Horizon ABI or scheduler
behavior, and it keeps guest timeline progress separate from backend completion
and memory visibility from the first implementation. No intermediate batch may
signal an uncompleted point, poll a guest wait, reuse a VI or BufferQueue event
as a GPU syncpoint event, or expose a runtime or host-backend object through the
neutral GPU contract.

1. **T5-A — Neutral timeline identity and completion vocabulary**
   - **TRI-080** and **TRI-083**
   - Introduce the timeline and synchronization portion of `nixe-gpu` with
     distinct typed identities for guest syncpoints, reserved timeline points,
     frontend submissions, backend submission tokens, host completion, and
     visibility completion.
   - Define ownership, checked comparison, wraparound, reservation, and
     monotonic advancement rules without importing Horizon, runtime, Maxwell,
     or a concrete backend into `nixe-gpu`.
2. **T5-B — `nvhost-ctrl` ABI and event-source isolation**
   - The nonblocking portions of **TRI-081**, plus **TRI-086**
   - Add the `/dev/nvhost-ctrl` descriptor and decode the pinned read,
     increment, wait, and event-registration ABIs in `nixe-horizon`, while the
     semantic timeline state remains in `nixe-gpu`.
   - Implement reads, increments, registration, already-satisfied waits, and
     verified invalid-argument results. An unresolved blocking wait must be
     represented explicitly and **TRI-081** remains incomplete until T5-C; it
     must not return early or fabricate completion.
   - Keep GPU syncpoint events, VI VSync, BufferQueue availability, and future
     acquire/release fences as distinct typed sources even where they all use
     runtime event primitives internally.
3. **T5-C — Scheduler-backed waits and lifecycle**
   - Complete **TRI-081**, then implement **TRI-082** and **TRI-085**
   - Bridge unresolved Horizon waits to runtime thread suspension, deadline and
     event wakeup using the existing scheduler continuation model rather than
     putting runtime dependencies in `nixe-gpu`.
   - Complete immediate success, finite and infinite timeout, cancellation,
     wraparound, event reuse, close, and process-teardown behavior, including
     removal of every waiter and reservation owned by the terminating process.
4. **T5-D — Completion, visibility, and deterministic acceptance**
   - **TRI-084**, **TRI-087**, and **TRI-088**
   - Add a manually advanced deterministic completion driver for tests. It
     exercises the neutral backend-completion boundary without introducing the
     `nixe-gpu-headless` package or a host graphics API before T8.
   - Propagate completion in the required order: backend work completes,
     declared GPU writes become visible through `nixe-memory`, and only then
     may the corresponding guest timeline point and Horizon event become
     observable.
   - Exercise delayed and out-of-order host completion, visibility failure,
     incomplete submissions, multiple waiters, wraparound, cancellation, and
     teardown before verifying the block exit criterion.

- [x] **TRI-080** Define guest syncpoint IDs, monotonically comparable values,
      wraparound rules, reservations, and ownership.
- [x] **TRI-081** Implement the required `/dev/nvhost-ctrl` syncpoint reads,
      increments, waits, and event registration from pinned ABIs.
- [x] **TRI-082** Connect blocking waits to runtime thread suspension and event
      wakeup rather than polling or returning early.
- [x] **TRI-083** Distinguish a guest timeline point, a frontend submission, a
      backend submission token, host completion, and memory visibility.
- [x] **TRI-084** Define completion propagation: host completion and required
      write visibility occur before the corresponding guest fence is signaled.
- [x] **TRI-085** Model immediate success, timeout, cancellation, process
      teardown, and wraparound according to verified behavior.
- [x] **TRI-086** Keep VI VSync, BufferQueue availability, acquire/release
      fences, and GPU syncpoints as separate event sources.
- [x] **TRI-087** Add deterministic tests with a manually advanced fake backend
      timeline.
- [x] **TRI-088** Test that an incomplete backend submission cannot be observed
      as a completed guest fence.

Exit criterion: synthetic submissions can reserve, advance, wait for, and signal
guest timeline points with correct scheduler behavior and without a real GPU.

External run checkpoint T5: run when the demo has reached a syncpoint-related
gap; verify that it advances without spinning or hanging.

### Block T6: GPU channels and GPFIFO submission

Purpose: accept real work only after memory and completion lifetimes are
representable.

#### T6 delivery batches

T6 is delivered as the following ordered review batches. The split keeps the
Horizon ioctl ABI separate from Switch 1 channel and GPFIFO semantics, and it
requires a submission to be completely decoded, resolved, and retained before
it can enter scheduling state. No intermediate batch may queue partially
validated work, read command bytes through a CPU virtual address, advance a
fence for undecoded work, or interpret Maxwell packets that belong to T7.

1. **T6-A — Channel identity, binding, and configuration**
   - **TRI-100**, **TRI-101**, and **TRI-102**
   - Add the `/dev/nvhost-gpu` descriptor and introduce real channel state in
     `nixe-gpu-maxwell::channel`, with typed channel identity, process owner,
     immutable profile, address-space binding, syncpoint association, frontend
     execution state, priority, and timeslice.
   - Implement the verified channel creation, binding, configuration, close,
     and teardown operations required by the pinned libnx path. The
     `/dev/nvhost-as-gpu` bind ioctl must connect existing semantic objects; it
     must not copy or reconstruct an address space inside Horizon ABI code.
   - Retain a deterministic scheduling policy as explicit channel state, while
     leaving GPFIFO contents and submission side effects unsupported until the
     following batches. Every other device operation remains typed and fatal.
2. **T6-B — Atomic GPFIFO descriptor decoding**
   - **TRI-103** and **TRI-109**
   - Add `nixe-gpu-maxwell::gpfifo` and decode the complete guest GPFIFO array
     into typed entry addresses, word counts, flags, and submission modes using
     checked arithmetic and profile-sized GPU addresses.
   - Keep ioctl structure and buffer decoding in `nixe-horizon::nvdrv`; place
     descriptor semantics and validation in the Maxwell frontend. Validate the
     complete request, including every entry and cross-field constraint, before
     mutating channel, fence, or scheduler state.
   - Distinguish verified invalid guest arguments from known but unsupported
     submission modes. Unsupported semantics remain fatal and no rejected
     request may execute or retain a prefix of its entries.
3. **T6-C — Versioned command resolution, retention, and diagnostics**
   - **TRI-104**, **TRI-105**, and **TRI-108**
   - Resolve every GPFIFO descriptor and complete pushbuffer byte range through
     the channel's bound `MaxwellGpuAddressSpace`, retaining the exact mapping
     identities, generations, canonical ranges, and allocation lifetimes used
     by the accepted submission.
   - Define immutable validated-submission and bounded-capture records carrying
     channel, frontend submission, GPFIFO entry, GPU VA, word offset, mapping
     generation, and retained source ranges. These records form T7's decoder
     input without importing Horizon ABI or a host backend.
   - Reject holes, page-boundary failures, stale generations, remaps, overflow,
     and teardown races before the first command word is consumed. A later
     unmap may remove the active mapping but cannot leave dangling in-flight
     backing or silently retarget retained work.
4. **T6-D — Submission ordering and deterministic scheduling**
   - **TRI-106** and **TRI-107**
   - Connect validated submissions to the T5 timeline and completion contracts,
     preserving CPU writes, verified cache-maintenance operations, device-read
     visibility, GPU writes, and fence increments as distinct ordered stages.
   - Introduce a deterministic single-queue scheduler over typed channel and
     submission identities. The policy may serialize initial work, but the
     object model must permit multiple channels and must retain dependencies,
     cancellation, teardown, and completion ownership explicitly.
   - Dispatch only complete validated submissions to the frontend-consumer
     boundary. Until T7 implements packet semantics, non-empty work stops there
     with the structured first-packet diagnostic and must not be reported as
     backend-complete or guest-visible.
5. **T6-E — Submission acceptance and regression matrix**
   - **TRI-110**, plus final acceptance of **TRI-100** through **TRI-109**
   - Exercise empty, chained, multiple-entry, page-crossing, unmapped, stale,
     remapped, malformed-mode, multiple-channel, cancellation, close, and
     process-teardown cases across the Horizon-to-Maxwell boundary.
   - Prove that rejected submissions are atomic, retained submissions remain
     pointer-free and lifetime-safe, deterministic ordering does not fabricate
     completion, and the bounded capture produces the same first unsupported
     packet location on replay.
   - Verify the block exit criterion before requesting the external T6
     checkpoint from the user.

- [x] **TRI-100** Implement `/dev/nvhost-gpu` channel creation, binding, close,
      and verified configuration operations required by Mesa/Nouveau.
- [x] **TRI-101** Associate each channel with process identity, GPU address
      space, capability profile, syncpoint state, and frontend execution state.
- [x] **TRI-102** Implement channel priorities and timeslices as explicit state,
      even if the initial scheduler serializes all channels.
- [x] **TRI-103** Decode GPFIFO descriptors with checked entry counts, sizes,
      flags, and GPU addresses.
- [x] **TRI-104** Resolve GPFIFO and pushbuffer memory through versioned GPU
      mappings, never through an `nvmap` CPU address or host pointer.
- [x] **TRI-105** Define submission lifetime and mapping retention through
      completion.
- [x] **TRI-106** Preserve guest ordering between CPU writes, cache-maintenance
      requests, submission, GPU reads/writes, and fence increments.
- [x] **TRI-107** Define deterministic single-queue scheduling as the first
      correct policy without encoding single-channel assumptions in object
      types.
- [x] **TRI-108** Produce typed diagnostics containing channel, submission,
      GPFIFO entry, pushbuffer GPU VA, word offset, and mapping generation.
- [x] **TRI-109** Reject malformed or unsupported submission modes before
      executing partial work.
- [x] **TRI-110** Test empty, chained, multiple-entry, page-crossing, unmapped,
      stale, and teardown submissions.

Exit criterion: Nixe can capture a bounded, validated real GPFIFO submission and
stop at its first unsupported packet or method without executing guessed work.

External run checkpoint T6: the user runs the demo and returns the first
structured GPFIFO/pushbuffer diagnostic plus the bounded capture.

### Block T7: Maxwell packet decoder and class dispatch

Purpose: turn validated pushbuffer words into exact Switch 1 GPU frontend state
transitions independently of the host backend.

#### T7 delivery batches

T7 is delivered as the following ordered review batches. The split establishes
packet syntax and immutable source records before any class state can change,
then introduces subchannel binding before engine-specific method coverage. No
batch may parse Horizon ioctl structures, depend on a host GPU backend, dispatch
a partially validated packet, silently ignore a method, or publish submission
completion; completion remains unavailable until later blocks can execute the
resulting work.

1. **T7-A — Atomic packet decoding and validation**
   - **TRI-120**, **TRI-121**, and **TRI-127**
   - Add `nixe-gpu-maxwell::pushbuffer::packet` and decode the verified Maxwell
     packet encodings from retained T6 command ranges into immutable packet and
     method-argument records. Keep the decoder independent of channel state,
     class handlers, Horizon ABI, and backend operations.
   - Validate the complete packet header and payload with checked arithmetic,
     including encoding, method range, subchannel, increment mode, immediate
     form, argument count, and command-buffer bounds, before returning any
     dispatchable record. A malformed or truncated packet must consume no
     frontend state and must identify its exact retained source word.
   - Add focused decoder tests while leaving all syntactically valid methods at
     a typed class-dispatch boundary for T7-B.
2. **T7-B — Subchannel binding and method provenance**
   - **TRI-122** and **TRI-126**
   - Add `nixe-gpu-maxwell::pushbuffer::dispatch` and explicit per-channel
     subchannel bindings. Implement only the verified class-binding operation,
     including replacement and reset behavior, and reject ordinary methods on
     an unbound subchannel.
   - Carry channel, frontend submission, GPFIFO entry, pushbuffer GPU VA, word
     index, subchannel, class, method, and argument through immutable dispatch
     records and every diagnostic. Binding state must remain frontend-owned and
     must not contain GPU mappings, host pointers, or backend handles.
   - Preflight every method generated by a packet before committing that
     packet's binding or class state, so a later invalid argument cannot leave
     an accepted prefix behind.
3. **T7-C — Declarative class and method dispatch**
   - **TRI-123**, **TRI-124**, and **TRI-125**
   - Introduce class handlers only for engines reached by verified workloads,
     under `nixe-gpu-maxwell::engines`; do not create empty 3D, compute, copy,
     or inline-to-memory handlers merely to complete the proposed directory
     tree.
   - Define declarative, source-linked method metadata for class identifiers,
     method numbers, names, argument validation, and handler selection where it
     removes duplicated constants without hiding semantics or provenance.
   - Keep unknown packet encoding, unsupported class, unsupported method,
     invalid method value, and missing neutral/backend capability as distinct
     typed outcomes. Only verified invalid guest state may become a guest error;
     missing emulator coverage remains a fatal host diagnostic at the first
     unsupported method.
4. **T7-D — Capture, replay, and acceptance matrix**
   - **TRI-128** and **TRI-129**, plus final acceptance of **TRI-120** through
     **TRI-127**
   - Exercise every supported packet form and increment mode, immediate and
     multi-argument packets, method and subchannel boundaries, truncation,
     overflow, unbound and rebound subchannels, unsupported classes and
     methods, invalid values, multiple GPFIFO entries, and teardown.
   - Extend the bounded T6 frontend capture with the pointer-free packet words
     and source metadata required for deterministic T7 replay. Repository
     fixtures must be synthetic and redistributable; caller-owned real captures
     remain outside the repository.
   - Prove that replay preserves packet order, names, arguments, source
     locations, class bindings, and the first fatal boundary, and that every
     rejected packet is atomic and cannot advance completion or mutate later
     frontend state.
   - Verify the block exit criterion before requesting the external T7
     checkpoint from the user.

- [x] **TRI-120** Implement a table-driven Maxwell pushbuffer packet decoder
      from pinned public sources.
- [x] **TRI-121** Validate packet length, method range, subchannel, increment
      mode, immediate data, and command-buffer bounds before dispatch.
- [x] **TRI-122** Model subchannel-to-class binding and reject methods without a
      valid bound class.
- [x] **TRI-123** Define separate class handlers for 3D, compute, copy,
      inline-to-memory, and other verified engines; implement only the classes
      actually reached.
- [x] **TRI-124** Generate method metadata from declarative tables where this
      reduces duplicated masks and names without hiding source references.
- [x] **TRI-125** Distinguish unknown packet encoding, known unsupported class,
      known unsupported method, invalid method value, and missing backend
      capability.
- [x] **TRI-126** Preserve source location for every method: channel,
      submission, GPFIFO entry, pushbuffer VA, word index, class, subchannel,
      method, and argument.
- [x] **TRI-127** Make packet decoding side-effect-free until the full packet is
      validated.
- [x] **TRI-128** Add synthetic packets for every supported encoding and
      boundary, plus malformed and truncation cases.
- [x] **TRI-129** Add bounded capture/replay at the validated frontend boundary;
      captures may contain only redistributable synthetic data unless the user
      stores caller-owned artifacts outside the repository.

Exit criterion: a real demo submission produces a deterministic sequence of
named class methods or a precise fatal diagnostic at the first unsupported
method.

### Block T8: Neutral GPU execution and resource contracts

Purpose: prevent Maxwell state and host API objects from becoming one coupled
implementation.

#### T8 delivery batches

T8 is delivered as the following ordered review batches. The split establishes
resource identity and view semantics before commands can refer to resources,
then defines access and capability requirements before a backend can accept a
submission. The headless backend is introduced only after the neutral backend
contract is complete. No batch may import Horizon ABI, Maxwell classes or
packet state, VI, `wgpu`, `winit`, or concrete host handles into `nixe-gpu`;
backend validation must not mutate state until the complete operation or
submission has been accepted.

1. **T8-A — Neutral resources, backing, and views**
   - The resource and view portions of **TRI-140**, plus **TRI-141** and
     **TRI-142**
   - Expand `nixe-gpu` with typed resource descriptions and identifiers for
     buffers, images, samplers, shaders, pipelines, descriptors, render passes,
     and queries. Keep descriptions, logical resource lifetime, and backing
     allocations distinct, and attach canonical backing ranges only through
     explicit views.
   - Define checked buffer and image view construction, including formats,
     dimensions, mip levels, samples, planes, pitch or block-linear layout,
     kind, swizzle, and subresource ranges. Invalid or overlapping ranges must
     be rejected without creating a partial resource. Do not add Maxwell
     encodings or host API format values to these neutral types.
2. **T8-B — Operations, accesses, and capability requirements**
   - Complete **TRI-140**, then implement **TRI-143** and **TRI-144**
   - Define immutable neutral copy, clear, draw, dispatch, barrier, query, and
     render-pass operations over the T8-A resource vocabulary, together with
     explicit read/write scopes, usage transitions, resource dependencies, and
     submission ordering requirements.
   - Express the capabilities required by an operation independently from the
     immutable guest GPU profile. Capability negotiation may reject work as an
     unrepresentable backend operation, but must not change discovery data,
     substitute a different format, or commit a prefix of the submission.
3. **T8-C — Backend contract and lifetime boundary**
   - **TRI-145**, plus the backend-facing acceptance and completion contracts
     required by **TRI-144**
   - Define the backend interface consumed by a composition root, with
     pointer-free typed resource and submission handles, explicit creation and
     destruction, ownership validation, completion tokens, and deterministic
     device-loss behavior. Connect it to the existing T5 completion vocabulary
     without equating backend completion with guest timeline visibility.
   - Resource and submission handles must be backend-instance scoped and
     generation-safe so stale, cross-device, use-after-destroy, and teardown
     operations fail before a concrete backend observes them.
4. **T8-D — Deterministic headless backend and acceptance**
   - **TRI-146** and **TRI-148**, with an explicit assessment of conditional
     **TRI-147**
   - Introduce `nixe-gpu-headless` as a real consumer of `nixe-gpu`. Validate
     resource creation, aliases, state transitions, access hazards, operation
     ordering, destruction, submissions, and manually controlled completion
     without parsing guest ABI or Maxwell commands.
   - Add synthetic cross-crate tests for invalid views, overlapping writes,
     missing barriers, unsupported formats, stale handles, use-after-destroy,
     capability failures, atomic rejection, completion ordering, device loss,
     and teardown. Add the tiny reference raster path from **TRI-147** only if
     these acceptance tests require deterministic pixels; otherwise record it
     as deliberately unnecessary for T8 rather than creating unused rendering
     machinery.
   - Verify the block exit criterion with `nixe-gpu-headless` and confirm the
     neutral crates remain free of Horizon, Maxwell, VI, and host graphics API
     dependencies.

- [x] **TRI-140** Define typed buffer, image, sampler, shader, pipeline,
      descriptor, render-pass, copy, clear, draw, dispatch, barrier, and query
      concepts needed by verified Maxwell methods.
- [x] **TRI-141** Keep resource descriptions separate from backing allocations
      and attach explicit backing ranges to resource views.
- [x] **TRI-142** Represent image format, dimensions, mip levels, samples,
      planes, pitch/block-linear layout, kind, swizzle, and subresource ranges.
- [x] **TRI-143** Define read/write access scopes and usage transitions emitted
      by the frontend.
- [x] **TRI-144** Define backend capability negotiation that can reject an
      unrepresentable operation without changing guest capability discovery.
- [x] **TRI-145** Define pointer-free backend resource and submission handles
      with explicit destruction and device-loss behavior.
- [x] **TRI-146** Define a deterministic headless validator that tracks resource
      creation, aliasing, state transitions, access hazards, and submission
      ordering.
- [x] **TRI-147** Assess whether a tiny reference raster path is required for
      deterministic pixel tests. It is deliberately not introduced in T8:
      contract validation and manually controlled completion require no pixel
      output. If a later block needs it, it must consume the same neutral
      operations and must not become a guest OpenGL shortcut.
- [x] **TRI-148** Test invalid views, overlapping writes, missing barriers,
      unsupported formats, use-after-destroy, and backend capability failures.

Exit criterion: synthetic frontend operations can be validated and completed
without linking `wgpu`, Horizon, VI, or the Maxwell packet decoder into the
neutral contract.

External run checkpoint: none.

### Block T9: Maxwell 3D state and resource interpretation

Purpose: implement the subset of Maxwell 3D behavior actually needed to form
the triangle while retaining state shapes that can expand to commercial use.

#### T9 delivery batches

T9 is delivered as the following ordered review batches. The split keeps raw
Maxwell method state, derived semantic state, canonical resource resolution,
and neutral operation emission as separate responsibilities. A later batch may
consume only immutable, completely validated output from the preceding one.
No batch may import Horizon ABI or a concrete host backend into
`nixe-gpu-maxwell`, silently ignore a reached method, infer an unverified reset
value, or key retained state by GPU virtual address alone.

1. **T9-A — Atomic 3D state foundation**
   - **TRI-160**
   - Replace the currently stateless `MAXWELL_B` method effect with typed 3D
     register groups and explicit validity. Preserve raw method values where
     later interpretation depends on their exact fields, but expose derived
     semantic values only through checked constructors or snapshots.
   - Record a reset value only when it is supported by a pinned public source;
     otherwise retain an explicit unknown or unset state. Preflight every
     method in a packet against a private candidate state and commit class
     bindings and 3D state together only after the whole packet is valid.
   - Introduce `engines/threed/state.rs` and related files only as they acquire
     this real responsibility; moving the existing `three_d.rs` handler must
     preserve the T7 source and diagnostic boundary.
2. **T9-B — Render targets and fixed-function output state**
   - **TRI-161** and **TRI-162**
   - Implement the render-target, depth/stencil, clear, viewport, scissor,
     clipping, rasterization, culling, blending, color-mask, multisample, and
     primitive-raster state actually reached by the demo. Keep disabled,
     unprogrammed, contradictory, and profile-unavailable states distinct.
   - Validate field encodings and cross-register combinations before commit.
     This batch records complete frontend state but does not create host
     textures, render passes, pipelines, or commands.
3. **T9-C — Vertex input and shader-visible binding state**
   - **TRI-163** and **TRI-164**
   - Implement vertex streams and attributes, index state, primitive topology,
     constant-buffer bindings, descriptor tables, textures, and samplers using
     typed per-slot state with explicit enable and validity rules.
   - Retain GPU virtual addresses only as unresolved frontend references.
     Range arithmetic, slot counts, alignment, format compatibility, and
     stage visibility must be validated without dereferencing guest memory or
     creating neutral resources during a method write.
4. **T9-D — Canonical resource resolution and layout boundary**
   - **TRI-165** and **TRI-166**
   - Resolve complete immutable state snapshots through the channel's bound
     Maxwell address space, retaining mapping identities, mapping generations,
     canonical backing, content generations, access ranges, and aliases until
     the resulting work is released.
   - Construct neutral buffer and image views only after every referenced range
     resolves. Preserve pitch-linear or block-linear layout in the neutral
     resource description and track dirty subresources; do not eagerly swizzle
     canonical memory or expose a host-layout decision from the frontend.
   - A stale mapping, truncated range, unsupported kind, contradictory alias
     interpretation, or incomplete multi-resource resolution rejects the
     complete snapshot without publishing a resource prefix.
5. **T9-E — Neutral clear/draw lowering and dependency invalidation**
   - **TRI-167**, **TRI-168**, and **TRI-169**
   - Lower a complete semantic snapshot into neutral resources, accesses,
     transitions, render-pass operations, clears, and draws. Validate all
     required state, capabilities, resource relationships, and operation
     ordering before submitting anything to a backend.
   - Keep shader translation as the explicit T10 boundary rather than
     fabricating shader semantics. Derived view and pipeline identities must
     use complete dependency keys including relevant method state, backing
     identity, layout, and generations; aliases or state changes invalidate
     only the affected derived records.
   - Preflight lowering, resource ownership, cache updates, and backend
     submission as distinct phases so a failure cannot commit a method prefix,
     cache entry, neutral resource, or backend operation.
6. **T9-F — Synthetic coverage and external acceptance**
   - **TRI-170**, followed by the block exit criterion and external checkpoint
   - Add focused tests for each supported state transition and cross-field
     contradiction, plus redistributable synthetic method streams that cover
     clear and draw formation, resource aliases, remapping/generation changes,
     cache invalidation, atomic rejection, and teardown.
   - Exercise the emitted operations through `nixe-gpu-headless` without
     weakening the dependency direction: integration belongs at a consumer or
     composition boundary, not inside the Maxwell frontend.
   - Run the real demo only after synthetic coverage passes. The expected next
     boundary is shader translation or backend execution; every earlier
     reached Maxwell method must either have semantics or produce a precise
     typed fatal diagnostic.
7. **T9-G — Fermi 2D initialization state coverage**
   - **TRI-171**, **TRI-172**, **TRI-173**, **TRI-174**, **TRI-175**,
     **TRI-176**, **TRI-177**, **TRI-178**, **TRI-179**, **TRI-179-000**, and
     **TRI-179-001**, followed by an external checkpoint
   - Add `engines/twod/` as a distinct `FERMI_TWOD_A` engine boundary; do not
     place 2D state in `engines/threed/` or treat a reached 2D method as a 3D
     no-op. Preserve the existing per-packet preflight and atomic commit model.
   - Implement `SET_NUM_PROCESSING_CLUSTERS` (`0x0260`) from NVIDIA's pinned
     public
     [`cl902d.h`](https://github.com/NVIDIA/open-gpu-doc/blob/9e6d83fe0770bc8644850a0b1bf5ddb1519905ba/classes/twod/cl902d.h#L493-L496).
     Model its verified `All` and `One` values as typed state with explicit
     validity, reject undefined bits before mutation, and keep every other
     unimplemented 2D method fatal and typed.
   - Add focused tests for both valid values, malformed values, source
     retention, packet-level atomic rejection, and separation from 3D state.
     Repeat `simple_triangle` afterwards; extend 2D coverage only for each
     subsequently reached operation with verified semantics rather than
     accepting an unknown initialization prefix.
   - Implement `SET_OPERATION` (`0x02ac`) from the same pinned header as typed
     state for every documented value `0..=6`. Reject reserved bits and
     undefined enum values atomically, retain exact source provenance, and do
     not emit a neutral copy merely because an operation mode was selected;
     execution begins only at a verified trigger.
   - Implement `SET_CLIP_ENABLE` (`0x0290`) as explicitly unset or programmed
     `Disabled`/`Enabled` state. Reject reserved bits atomically and retain
     exact source provenance. Disabling clipping must not require unobserved
     rectangle state, and selecting either value must not itself emit a neutral
     operation.
   - Implement `SET_COLOR_KEY_ENABLE` (`0x029c`) as explicitly unset or
     programmed `Disabled`/`Enabled` state distinct from clip enable despite
     their shared encoding. Reject reserved bits atomically, retain exact
     source provenance, and defer color-key format/value requirements until a
     verified 2D trigger consumes enabled color-key state. Use NVIDIA's pinned
     [`cl902d.h`](https://github.com/NVIDIA/open-gpu-doc/blob/9e6d83fe0770bc8644850a0b1bf5ddb1519905ba/classes/twod/cl902d.h#L551-L557)
     for the field and enum values.
   - Implement `SET_PIXELS_FROM_MEMORY_CORRAL_SIZE` (`0x0884`) as a bounded,
     source-preserving 10-bit value in a dedicated pixels-from-memory state
     group. Do not infer an undocumented unit, reset value, or execution
     effect. Reject reserved bits atomically and keep every other unsupported
     pixels-from-memory method fatal and typed. Use NVIDIA's pinned
     [`cl902d.h`](https://github.com/NVIDIA/open-gpu-doc/blob/9e6d83fe0770bc8644850a0b1bf5ddb1519905ba/classes/twod/cl902d.h#L931-L932)
     for the verified field width.
   - Implement `SET_PIXELS_FROM_MEMORY_SAFE_OVERLAP` (`0x0888`) as explicitly
     unset or programmed `Disabled`/`Enabled` pixels-from-memory state. Reject
     reserved bits atomically, retain exact source provenance, and defer its
     execution consequences until a verified trigger consumes the state. Use
     NVIDIA's pinned
     [`cl902d.h`](https://github.com/NVIDIA/open-gpu-doc/blob/9e6d83fe0770bc8644850a0b1bf5ddb1519905ba/classes/twod/cl902d.h#L934-L937)
     for the field and enum values.
   - Implement `SET_RENDER_ENABLE_C` (`0x026c`) as explicitly unset or one of
     the five documented render-enable modes in a dedicated render-enable
     state group. Reject undefined values and reserved bits atomically, retain
     exact source provenance, and defer conditional address requirements and
     condition evaluation until a verified 2D trigger consumes the state.
     Keep `SET_RENDER_ENABLE_A/B` fatal until their semantics are required and
     implemented. Use NVIDIA's pinned
     [`cl902d.h`](https://github.com/NVIDIA/open-gpu-doc/blob/9e6d83fe0770bc8644850a0b1bf5ddb1519905ba/classes/twod/cl902d.h#L498-L510)
     for the fields and mode values.
   - Implement `SET_NOTIFY_A` (`0x0104`) as a bounded, source-preserving 25-bit
     upper-address fragment in a dedicated notification state group. Do not
     construct a GPU virtual address, perform a memory write, publish a
     completion, or wake a waiter until `SET_NOTIFY_B` and `NOTIFY` have
     verified semantics. Reject reserved bits atomically and keep those
     unsupported companion methods fatal and typed. Use NVIDIA's pinned
     [`cl902d.h`](https://github.com/NVIDIA/open-gpu-doc/blob/9e6d83fe0770bc8644850a0b1bf5ddb1519905ba/classes/twod/cl902d.h#L234-L247)
     for the field and trigger values.
   - Implement `SET_NOTIFY_B` (`0x0108`) as the source-preserving 32-bit lower
     address fragment. Accept its complete documented bit domain, allow `A`
     and `B` to commit atomically in one incrementing packet, and keep both
     fragments separate rather than exposing an incompletely verified GPU
     virtual address. A following unsupported `NOTIFY` must reject its complete
     packet without publishing either fragment or any memory, synchronization,
     completion, or wakeup effect. Use the same pinned `cl902d.h` definition.
   - Implement `MAXWELL_B::SET_RENDER_ENABLE_C` (`0x1558`) as explicitly unset
     or one of its five documented modes in 3D-owned render-enable state.
     Preserve class separation rather than reusing the identically encoded 2D
     type, reject undefined values and reserved bits atomically, and do not
     require `SET_RENDER_ENABLE_A/B` while merely programming the selector.
     Neutral clear/draw lowering may proceed for the documented `Enabled` mode;
     explicitly programmed disabled or conditional modes must remain typed
     unsupported execution until their verified effects and address state are
     implemented. Use NVIDIA's pinned
     [`clb197.h`](https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2759-L2771)
     for the fields and values.
   - Implement `MAXWELL_B::SET_SM_TIMEOUT_INTERVAL` (`0x0de4`) as a bounded,
     source-preserving six-bit `COUNTER_BIT` value in dedicated 3D shader
     execution state. Reject reserved bits atomically and do not infer a time
     unit, duration, watchdog policy, or backend behavior from the field name.
     Programming the register must not emit a neutral operation. Any later
     shader execution that requires its temporal effect must remain a typed
     unsupported boundary until that behavior is verified rather than silently
     ignoring the programmed state. Use NVIDIA's pinned
     [`clb197.h`](https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1079-L1090)
     for the field width.
   - Implement `MAXWELL_B::SET_Z_COMPRESSION` (`0x19cc`) as a typed,
     source-preserving depth/stencil-target selector. Accept only the public
     `FALSE` and `TRUE` values, reject reserved bits atomically, and do not
     treat programming the selector as an operation or as proof that a depth
     attachment exists. Keep the selector distinct from memory kind and image
     layout. Neutral operations that do not consume depth may proceed, while
     an enabled selector on an operation that consumes depth must remain a
     typed unsupported boundary until compressed representation and coherency
     semantics are verified. Use NVIDIA's pinned
     [`clb197.h`](https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3545-L3548)
     for the field and values.
   - Implement the eight `MAXWELL_B::SET_COLOR_COMPRESSION(i)` selectors
     (`0x19e0..=0x19fc`) as typed, source-preserving state owned independently
     by each color target. Accept only the public `FALSE` and `TRUE` values,
     reject reserved bits atomically, and do not let a selector write create
     or complete an attachment. Keep color compression distinct from Z
     compression, memory kind, and image layout. Operations that do not
     consume an enabled target may proceed; clear/draw lowering that consumes
     one must remain a typed unsupported boundary until compressed color
     representation and coherency are verified. Use NVIDIA's pinned
     [`clb197.h`](https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3575-L3578)
     for the indexed field and values.
   - Implement `MAXWELL_B::SET_CT_SELECT` (`0x121c`) as typed,
     source-preserving color-target routing state. Decode the four-bit
     `TARGET_COUNT` and all eight three-bit `TARGETi` selectors, reject bits
     28..31 and counts greater than the eight exposed selectors atomically,
     and retain inactive selector fields without assigning them unverified
     reset values or rejecting their contents during register programming.
     A register write must neither require attachments nor emit work.
   - Consume `SET_CT_SELECT` only from immutable draw snapshots. Build the
     neutral render-pass color attachment sequence from the first
     `TARGET_COUNT` selectors in their declared order rather than attaching
     every configured color target. Require every selected target to resolve
     completely, keep `CLEAR_SURFACE` independent because it carries its own
     MRT selector, and reject duplicate, missing, disabled, or otherwise
     unrepresentable active routes with typed errors before cache or backend
     effects. Do not silently normalize or deduplicate routes.
   - Propagate the ordered active routes through draw resource dependencies,
     alias validation, render-pass identity, pipeline identity, capability
     negotiation, and cache invalidation. Unselected configured targets may
     remain in the immutable frontend snapshot but must not become draw
     attachments or observable draw dependencies. Add focused tests for
     counts zero through eight, ordered permutations, inactive-field
     retention, malformed encodings, incomplete and duplicate active routes,
     packet-level rollback, exact dependency selection, and render-pass and
     pipeline cache-key separation. Use NVIDIA's pinned
     [`clb197.h`](https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2003-L2012)
     for the verified field layout.

- [x] **TRI-160** Model Maxwell 3D register/method state with verified reset
      values and explicit validity.
- [x] **TRI-161** Implement render-target and depth/stencil descriptions,
      formats, dimensions, layouts, kinds, and clear state.
- [x] **TRI-162** Implement viewport, scissor, clip, rasterization, cull,
      depth/stencil, blend, color-mask, multisample, and primitive state reached
      by the demo.
- [x] **TRI-163** Implement vertex stream, vertex attribute, index-buffer, and
      primitive topology state.
- [x] **TRI-164** Implement constant-buffer, descriptor-table, texture, and
      sampler state reached during shader setup, even if the triangle uses only
      a subset in its shader logic.
- [x] **TRI-165** Resolve every referenced resource through GPU VA mappings and
      construct views over canonical backing.
- [x] **TRI-166** Convert pitch-linear or block-linear guest layouts only at a
      resource/backend boundary that records the representation and dirty
      subresources.
- [x] **TRI-167** Emit neutral clear and draw operations from complete validated
      frontend state.
- [x] **TRI-168** Reject incomplete or contradictory draw state before backend
      submission.
- [x] **TRI-169** Invalidate cached views and pipelines when aliased memory or
      relevant methods change their dependencies.
- [x] **TRI-170** Add focused state-transition tests and synthetic clear/draw
      command streams.
- [x] **TRI-171** Implement `FERMI_TWOD_A`
      `SET_NUM_PROCESSING_CLUSTERS` (`0x0260`) initialization state in a
      separate 2D engine module, with typed values and atomic validation.
- [x] **TRI-172** Implement `FERMI_TWOD_A::SET_OPERATION`
      (`0x02ac`) selector as typed, source-preserving state without executing
      or lowering an unobserved 2D operation.
- [x] **TRI-173** Implement `FERMI_TWOD_A::SET_CLIP_ENABLE`
      (`0x0290`) selector as typed, source-preserving state without inventing
      clip rectangles or execution semantics.
- [x] **TRI-174** Implement `FERMI_TWOD_A::SET_COLOR_KEY_ENABLE` (`0x029c`)
      as typed, source-preserving state without requiring unprogrammed
      color-key data while disabled or inventing execution semantics.
- [x] **TRI-175** Implement
      `FERMI_TWOD_A::SET_PIXELS_FROM_MEMORY_CORRAL_SIZE` (`0x0884`) as bounded,
      source-preserving pixels-from-memory state without inventing units or
      execution semantics.
- [x] **TRI-176** Implement
      `FERMI_TWOD_A::SET_PIXELS_FROM_MEMORY_SAFE_OVERLAP` (`0x0888`) as typed,
      source-preserving pixels-from-memory state without inventing overlap
      execution semantics.
- [x] **TRI-177** Implement `FERMI_TWOD_A::SET_RENDER_ENABLE_C` (`0x026c`) as
      typed, source-preserving render-enable state without prematurely
      requiring conditional-address state or evaluating the condition.
- [x] **TRI-178** Implement `FERMI_TWOD_A::SET_NOTIFY_A` (`0x0104`) as a
      bounded, source-preserving upper-address fragment without composing an
      incomplete address or inventing notification effects.
- [x] **TRI-179** Implement `FERMI_TWOD_A::SET_NOTIFY_B` (`0x0108`) as a
      source-preserving lower-address fragment with atomic `A+B` packet
      behavior and no notification effects.
- [x] **TRI-179-000** Implement `MAXWELL_B::SET_RENDER_ENABLE_C` (`0x1558`) as
      typed, source-preserving 3D state, keeping non-enabled execution modes
      explicitly unsupported until their complete semantics exist.
- [x] **TRI-179-001** Implement `MAXWELL_B::SET_SM_TIMEOUT_INTERVAL` (`0x0de4`) as
      bounded, source-preserving shader-execution state without inventing
      temporal or watchdog semantics.
- [x] **TRI-179-002** Implement `MAXWELL_B::SET_Z_COMPRESSION` (`0x19cc`) as typed
      depth/stencil state, preserving non-depth operation progress while
      keeping unverified compressed depth execution explicitly unsupported.
- [x] **TRI-179-003** Implement all eight
      `MAXWELL_B::SET_COLOR_COMPRESSION(i)` selectors as isolated color-target
      state without inventing compressed representation or coherency.
- [x] **TRI-179-004** Implement `MAXWELL_B::SET_CT_SELECT` (`0x121c`) as complete
      typed color-target routing consumed by draw dependencies, render-pass
      formation, pipeline identity, and cache invalidation.
- [x] **TRI-179-005** Implement the undocumented-in-`clb197.h` Maxwell method
      `0x15b4`, identified as `CSAA_ENABLE` by the pinned envytools register
      database, as source-preserving typed coverage-sample state.
- [x] **TRI-179-006** Implement `MAXWELL_B::SET_ALIASED_LINE_WIDTH_ENABLE`
      (`0x020c`) as source-preserving typed line-rasterization state.
- [x] **TRI-179-007** Implement
      `MAXWELL_B::SET_DA_PRIMITIVE_RESTART_VERTEX_ARRAY` (`0x0de8`) as an
      explicitly unset or source-preserving `Disabled`/`Enabled` selector in
      primitive-input state.
- [x] **TRI-179-008** Implement `MAXWELL_B::SET_BLEND_ENABLE_COMMON` (`0x135c`)
      as explicitly unset or source-preserving `Disabled`/`Enabled`
      fixed-function output state.
- [x] **TRI-179-009** Implement `MAXWELL_B::SET_SHADE_MODE` (`0x12d4`) as
      source-preserving `Flat` (`0x1d00`) or `Smooth` (`0x1d01`) state with
      atomic validation. Consume it only from draw snapshots, retain it in
      pipeline identity, keep clears independent, and reject execution unless
      the selected interpolation behavior is explicitly representable.
- [x] **TRI-179-010** Implement `MAXWELL_B::SET_API_VISIBLE_CALL_LIMIT` (`0x0d64`)
      as source-preserving typed state for encodings `0..=8` and `15`, with
      atomic validation. Consume it only where its verified draw or scheduling
      semantics apply, and reject execution rather than treating it as an
      ignorable performance hint.
- [x] **TRI-179-011** Implement `MAXWELL_B::SET_ZCULL_STATS` (`0x151c`) as
      source-preserving `Disabled`/`Enabled` 3D Z-cull statistics state,
      separate from channel Z-cull bindings and profile capabilities. Validate
      reserved bits atomically and reject any future guest-visible statistics
      consumer until counter accumulation, visibility, and reporting semantics
      are implemented.
- [x] **TRI-179-012** Implement `MAXWELL_B::SET_L1_CONFIGURATION` (`0x0308`) as
      source-preserving typed shader-memory state for the documented 16 KiB
      and 48 KiB directly addressable memory configurations. Validate selectors
      and reserved bits atomically, keep it separate from host caches and guest
      canonical memory, and retain it as a shader translation and execution
      dependency. Reject shaders whose required directly addressable memory
      cannot be verified or represented rather than assuming host L1 topology.
- [x] **TRI-179-013** Implement `SET_REDUCE_COLOR_THRESHOLDS_UNORM8` (`0x10cc`)
      as atomic, source-preserving typed color-reduction threshold state.
- [x] **TRI-179-014** Implement `SET_REDUCE_COLOR_THRESHOLDS_UNORM10` (`0x10e0`)
      as independent typed color-reduction threshold state.
- [x] **TRI-179-015** Implement `SET_REDUCE_COLOR_THRESHOLDS_UNORM16` (`0x10e4`)
      as independent typed color-reduction threshold state.
- [x] **TRI-179-016** Implement `SET_REDUCE_COLOR_THRESHOLDS_FP16` (`0x10ec`) as
      independent, source-preserving typed color-reduction threshold state.
- [x] **TRI-179-017** Implement `SET_REDUCE_COLOR_THRESHOLDS_SRGB8` (`0x10f0`) as
      independent, source-preserving typed color-reduction threshold state.
- [x] **TRI-179-018** Implement `SET_ALPHA_FRACTION` (`0x074c`) as validated,
      source-preserving typed raster state with an opaque eight-bit value.
- [x] **TRI-179-019** Implement `CHECK_SPH_VERSION` (`0x16a8`) as a typed, atomic
      shader-program-header compatibility check against the Maxwell profile.
- [x] **TRI-179-020** Implement `CHECK_AAM_VERSION` (`0x1794`) as a typed, atomic
      AAM-version compatibility check against the Maxwell profile.
- [x] **TRI-179-021** Implement the five ROP L2 cache-control methods identified
      together in `dump/pushbuffer.txt` as independent, typed eviction-policy state.
- [x] **TRI-179-022** Implement `SET_BLEND_PER_FORMAT_ENABLE` (`0x1140`) and the
      related `0x0fdc`/`0x19c0` blend controls from `dump/pushbuffer.txt` as independent typed state.
- [x] **TRI-179-023** Implement `SET_ATTRIBUTE_DEFAULT` (`0x1610`) and the related
      `SET_DA_OUTPUT` (`0x164c`) as typed, source-preserving vertex-assembly state.
- [x] **TRI-179-024** Implement `SET_RENDER_ENABLE_CONTROL` (`0x030c`) as typed,
      source-preserving render-enable control with explicit conditional-load semantics.
- [x] **TRI-179-025** Implement `SET_PS_OUTPUT_SAMPLE_MASK_USAGE` (`0x0300`) as
      typed fragment-output coverage state qualified by antialias enablement.
- [x] **TRI-179-026** Implement `SET_PRIM_CIRCULAR_BUFFER_THROTTLE` (`0x02d0`) as
      typed, source-preserving primitive-buffer throttle state with justified pipeline neutrality.
- [x] **TRI-179-027** Implement `SET_PROGRAM_REGION_A/B` (`0x1608`/`0x160c`) as
      a source-preserving 40-bit shader-program base and draw/translation dependency.
- [x] **TRI-179-028** Implement `LOAD_CONSTANT_BUFFER_OFFSET/DATA` (`0x238c`/`0x2390`-`0x23fc`)
      as validated, atomic inline uploads through the existing constant-buffer selector.
- [x] **TRI-179-029** Implement `SET_VERTEX_STREAM_SUBSTITUTE_A/B` (`0x0f84`/`0x0f88`)
      as a validated, source-preserving 40-bit vertex-stream substitution address.
- [x] **TRI-179-030** Implement `SET_SHADER_LOCAL_MEMORY_A-E/WINDOW` (`0x0790`-`0x07a0`/`0x077c`)
      as validated, source-preserving shader-local-memory region and allocation state.
- [x] **TRI-179-031** Implement `SET_ACTIVE_ZCULL_REGION` (`0x1590`) as a validated,
      source-preserving six-bit Z-cull region selector with explicit draw semantics.
- [x] **TRI-179-032** Implement all eight `SET_WINDOW_CLIP_HORIZONTAL/VERTICAL`
      pairs (`0x0d00`-`0x0d3c`) as validated, source-preserving window-clip rectangles.
- [x] **TRI-179-033** Implement `SET_CLIP_ID_TEST` (`0x197c`) as validated,
      source-preserving boolean clip-ID state with explicit effective draw semantics.
- [x] **TRI-179-034** Implement `SET_CLEAR_SURFACE_CONTROL` (`0x10f8`) as typed,
      source-preserving clear-selection state, including full-surface clear lowering.
- [x] **TRI-179-035** Implement `SET_VIEWPORT_SCALE_OFFSET` (`0x192c`) as a typed,
      source-preserving selector for the effective viewport transform.
- [x] **TRI-179-036** Implement the four MME instruction/start-address RAM load methods
      (`0x0114`-`0x0120`) as bounded, atomic, source-preserving macro-program storage.
- [x] **TRI-179-037** Implement `SET_CT_MRT_ENABLE` (`0x0fac`) as validated,
      source-preserving fragment-output routing state with explicit draw semantics.
- [x] **TRI-179-038** Implement the indexed `CALL_MME_MACRO/DATA` aperture (`0x3800`-`0x3fff`)
      with bounded, atomic execution of the captured Maxwell MME programs.
- [x] **TRI-179-039** Model the verified `SET_FRONT/BACK_POLYGON_MODE` reset state
      required by MME register reads, preserving raw values and reset provenance.
- [x] **TRI-179-040** Model the verified reset headers for all six
      `SET_PIPELINE_SHADER(i)` slots required by MME register reads.
- [x] **TRI-179-041** Implement `SET_RASTER_BOUNDING_BOX` (`0x02ec`) as validated,
      source-preserving raster-control state for MME-emitted selections.
- [x] **TRI-179-042** Implement `SET_RT_LAYER` (`0x15cc`) as validated,
      source-preserving render-target layer selection with explicit draw semantics.
- [x] **TRI-179-043** Implement `SET_PATCH` (`0x0dcc`) as validated,
      source-preserving patch size consumed only by patch-topology draws.
- [x] **TRI-179-044** Implement `SET_POINT_SPRITE_SELECT` (`0x1604`) and the related
      `SET_POINT_CENTER_MODE` (`0x165c`) as validated, source-preserving point-rasterization state.
- [x] **TRI-179-045** Implement `SET_EDGE_FLAG` (`0x15e4`) as validated,
      source-preserving polygon-edge state with topology-aware draw semantics.
- [x] **TRI-179-046** Introduce `MAXWELL_COMPUTE_B` state and implement its captured
      shader local/shared-memory base, allocation, and window configuration methods.
- [x] **TRI-179-047** Implement compute `SET_PROGRAM_REGION_A/B` and `SET_SPA_VERSION`
      as validated, source-preserving shader-code location and architecture state.
- [x] **TRI-179-048** Implement compute texture-header and sampler pool `A/B/C`
      methods as validated, source-preserving descriptor-pool state.
- [x] **TRI-179-049** Implement indexed compute `SET_CWD_REF_COUNTER` (`0x0248`)
      as a validated, source-preserving 64-entry reference-counter bank.
- [x] **TRI-179-050** Implement compute `WAIT_FOR_IDLE` (`0x0110`) as an explicit,
      source-preserving channel-ordering operation with validated neutral lowering.
- [x] **TRI-179-051** Implement compute `SET_BINDLESS_TEXTURE` (`0x2608`) as a validated,
      source-preserving three-bit constant-buffer slot selector for bindless descriptors.
- [x] **TRI-179-052** Implement the captured compute inline-to-memory pitch upload
      methods as validated, atomic GPU-memory transfer operations.
- [x] **TRI-179-053** Implement compute `INVALIDATE_SHADER_CACHES_NO_WFI` (`0x1698`)
      as a validated, source-preserving ordered cache-maintenance operation.
- [x] **TRI-179-054** Implement 3D `FLUSH_PENDING_WRITES` (`0x1144`) and the following
      `INCREMENT_SYNC_POINT` (`0x02c8`) as validated, ordered completion operations.
- [x] **TRI-179-055** Build the submission-level neutral execution preflight and
      completion handoff for a fully decoded Maxwell packet replay.
- [x] **TRI-179-056** Separate `nvmap` CPU allocation access from GPU mapping
      permissions so Maxwell write targets remain writable.
- [x] **TRI-179-057** Lower compute `WAIT_FOR_IDLE` as an ordered neutral barrier
      that retains whether an earlier execution prefix must be drained.
- [x] **TRI-179-058** Execute write-only Maxwell initialization submissions
      atomically and publish their reserved syncpoint only after visibility.
- [x] **TRI-179-059** Implement A64 Advanced SIMD integer-to-floating-point
      vector conversions, starting with the captured `SCVTF V28.4S, V31.4S`.
- [x] **TRI-179-060** Implement exact interpreter semantics for all allocated
      A64 Advanced SIMD `FDIV` vector arrangements, starting with `V28.4S`.
- [x] **TRI-179-061** Implement exact interpreter semantics for the A64 scalar
      `FMOV` immediate family, starting with captured `FMOV S31, #1.0`.
- [x] **TRI-179-062** Implement exact interpreter semantics for scalar A64
      `SCVTF`/`UCVTF` W/X-to-S/D conversions, starting with `UCVTF D0, X1`.
- [x] **TRI-179-063** Implement exact interpreter semantics for scalar A64
      `FCVT` conversions between single and double precision, starting with `FCVT D30, S30`.
- [x] **TRI-179-064** Implement exact interpreter semantics for scalar A64
      `FDIV` in single and double precision, starting with `FDIV D31, D31, D30`.
- [x] **TRI-179-065** Implement exact interpreter semantics for scalar A64
      `FCMP`/`FCMPE` register and zero comparisons, starting with `FCMPE D0, D31`.
- [x] **TRI-179-066** Implement exact interpreter semantics for the scalar A64
      `FRINTN/P/M/Z/A/X/I` S/D family, starting with `FRINTM D31, D31`.
- [x] **TRI-179-067** Implement exact interpreter semantics for scalar A64
      `FADD`/`FSUB` in single and double precision, starting with `FADD D29, D31, D29`.
- [x] **TRI-179-068** Implement exact interpreter semantics for scalar A64
      `FCVTZS`/`FCVTZU` S/D-to-W/X conversions, starting with `FCVTZU X2, D29`.
- [x] **TRI-179-069** Implement exact interpreter semantics for scalar A64
      `FMUL`/`FNMUL` in single and double precision, starting with `FMUL D28, D29, D28`.
- [x] **TRI-179-070** Implement exact interpreter semantics for scalar A64
      `FCVT{N,P,M,A}{S,U}` conversions, starting with `FCVTMU X1, D28`.
- [x] **TRI-179-071** Implement exact interpreter semantics for the A64 Advanced
      SIMD `EXT` byte-extract family, starting with `EXT V31.16B, V31.16B, V31.16B, #8`.
- [x] **TRI-179-072** Implement the scheduler-independent Horizon process-wide
      key signal path and zero-timeout atomic wait semantics.

Exit criterion: the demo's clear and draw methods become validated neutral
operations with complete resource dependencies, apart from shader translation
or explicitly identified later engine coverage.

### Block PRE-T10: Discovered frontend prerequisites

Purpose: track concrete gaps discovered by external `simple_triangle` runs that
must be resolved before work on T10 can begin. Add new tasks sequentially as
`PRE-T10-NNN`, without retroactively expanding the scope of completed blocks.

- [x] **PRE-T10-001** Implement the verified Maxwell host `MEM_OP_A`/`MEM_OP_B`
      subset required by the captured GPFIFO stream, beginning with
      `MEM_OP_B(L2_FLUSH_DIRTY)`, and preserve its cache-ordering and memory-
      visibility effects through the neutral GPU contract with focused atomic
      dispatch and ordering tests.
- [x] **PRE-T10-002** Implement the captured legacy
      `MEM_OP_B(L2_SYSMEM_INVALIDATE)` operation as a distinct neutral device-
      read cache invalidation, preserving packet atomicity and its ordering
      relative to L2 dirty writeback and later GPU work.
- [x] **PRE-T10-003** Implement `MAXWELL_B::INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI`
      (`0x1288`) with its documented all-lines and tagged-line selectors,
      preserving the source request while lowering it to an ordered neutral
      texture-read cache invalidation without inventing an idle wait.
- [x] **PRE-T10-004** Implement `MAXWELL_B::INVALIDATE_SHADER_CACHES_NO_WFI`
      (`0x0da4`) for instruction, global-data, and constant caches, sharing the
      verified selector representation with the compute class and preserving
      its no-wait ordering through a neutral shader-cache invalidation.
- [x] **PRE-T10-005** Implement the shared all-lines/tagged-line invalidation
      family for `MAXWELL_B` sampler (`0x1424`) and texture-header (`0x1428`)
      caches, preserving each cache target and lowering both without an
      implicit wait alongside the existing texture-data invalidation.
- [x] **PRE-T10-006** Implement the captured single-line pitch upload path for
      `MAXWELL_INLINE_TO_MEMORY_A`, including destination setup, launch
      validation, ordered inline words, and atomic GPU-memory execution.
- [x] **PRE-T10-007** Extend `MAXWELL_INLINE_TO_MEMORY_A::LAUNCH_DMA` with the
      captured one-word semaphore-structure selection (`0x1001`), preserving
      the latent field without fabricating completion or interrupt effects.
- [x] **PRE-T10-008** Implement the waiting `MAXWELL_B` sampler (`0x1330`),
      texture-header (`0x1334`), and texture-data (`0x1338`) cache invalidation
      family with shared line/tag decoding and explicit prior-work draining.
- [x] **PRE-T10-009** Implement `MAXWELL_B::SET_SURFACE_CLIP_HORIZONTAL` (`0x0ff4`)
      and `SET_SURFACE_CLIP_VERTICAL` (`0x0ff8`) as source-preserving origin and
      extent state and retain them as raster dependencies without inventing
      draw-time region composition beyond a verified full-attachment no-op.
- [x] **PRE-T10-010** Implement `MAXWELL_B::SET_ZT_SELECT` (`0x1538`) as a
      validated zero-or-one depth/stencil target selector, preventing an
      explicitly unselected configured target from becoming a resolved draw
      resource or dependency.
- [x] **PRE-T10-011** Implement `MAXWELL_B::SET_ZT_LAYER` (`0x179c`) as a
      source-preserving 16-bit depth/stencil array-layer selector, using it to
      resolve exactly one valid array subresource and rejecting reserved bits,
      out-of-range layers, and non-zero layers on three-dimensional targets.
- [x] **PRE-T10-012** Implement `MAXWELL_B::SET_HYBRID_ANTI_ALIAS_CONTROL`
      (`0x0754`) as typed coverage state, accepting the captured single-pass,
      per-fragment configuration as neutral while stopping non-neutral hybrid
      sampling before draw publication.
- [x] **PRE-T10-013** Implement the four `MAXWELL_B::SAMPLE_LOCATIONS`
      registers (`0x11e0`-`0x11ec`) as sixteen source-preserving 4-bit sample
      coordinates, accepting the captured all-center pattern and stopping
      custom sample placement before draw publication.
- [x] **PRE-T10-014** Implement the `MAXWELL_B::SET_LOGIC_OP` (`0x19c4`) and
      `SET_LOGIC_OP_FUNC` (`0x19c8`) family as typed color-output state,
      accepting the captured disabled path while validating and stopping
      enabled bitwise color operations before draw publication.
- [x] **PRE-T10-015** Implement `MAXWELL_B::SET_SINGLE_CT_WRITE_CONTROL`
      (`0x0f90`) and connect it to the existing eight `SET_CT_WRITE` masks,
      accepting complete RGBA writes while rejecting incomplete or partial
      effective color masks before draw publication.
- [x] **PRE-T10-016** Implement the `MAXWELL_B` alpha-test family
      (`SET_ALPHA_TEST`, `SET_ALPHA_REF`, and `SET_ALPHA_FUNC`) as typed state,
      accepting the captured disabled path and stopping enabled alpha tests
      before draw publication until shader or backend lowering exists.
- [x] **PRE-T10-017** Implement `SET_PROVOKING_VERTEX` (`0x1684`) and the
      adjacent `SET_TWO_SIDED_LIGHT` (`0x1688`) as typed raster state, accepting
      the captured last-vertex and disabled-lighting path while rejecting
      unsupported alternatives before draw publication.
- [x] **PRE-T10-018** Implement the color-clamping controls
      `SET_COLOR_CLAMP` (`0x2600`) and `SET_PS_SATURATE` (`0x13a8`) as typed
      output state, accepting the captured disabled path and stopping effective
      clamping before draw publication until lowering support exists.
- [x] **PRE-T10-019** Extend typed line-rasterization state with
      `SET_ANTI_ALIASED_LINE` (`0x1570`), aliased width, and line-stipple
      controls, accepting the captured disabled triangle path while rejecting
      unsupported effective line smoothing or stippling before publication.
- [x] **PRE-T10-020** Implement `SET_ATTRIBUTE_POINT_SIZE` (`0x1910`) together
      with point-sprite and point-antialias enables as typed, topology-aware
      raster state, accepting the captured triangle path while rejecting
      unsupported effective point behavior before draw publication.
- [x] **PRE-T10-021** Implement `SET_FILL_VIA_TRIANGLE` (`0x113c`) and the
      nearby `SET_CONSERVATIVE_RASTER` (`0x1148`) as typed raster state,
      accepting their captured disabled modes while rejecting effective modes
      before draw publication until neutral lowering exists.
- [x] **PRE-T10-022** Implement polygon smoothing (`0x0db4`) together with
      polygon-stipple enable and its 32-word pattern (`0x168c`,
      `0x1700..0x177c`), preserving the captured disabled state while rejecting
      unsupported effective polygon rasterization before draw publication.
- [x] **PRE-T10-023** Implement `SET_VIEWPORT_PIXEL` (`0x1924`) as a validated,
      source-preserving viewport pixel-center convention, accepting the
      captured half-integer centers while rejecting integer-center draws until
      the neutral pipeline exposes an equivalent transform.
- [x] **PRE-T10-024** Implement all 16 indexed
      `SET_VIEWPORT_COORDINATE_SWIZZLE` registers (`0x0a18` plus a `0x20`
      stride) as typed signed-component mappings, accepting identity while
      rejecting effective swizzles before draw publication.
- [x] **PRE-T10-025** Implement ordered `INVALIDATE_SHADER_CACHES` (`0x021c`)
      with typed instruction/data/constant selectors and lock/flush controls,
      draining prior GPU work for the captured no-lock, no-flush request while
      rejecting unsupported effective control flags during lowering.
- [x] **PRE-T10-026** Implement all six indexed `SET_PIPELINE_PROGRAM` and
      `SET_PIPELINE_REGISTER_COUNT` registers as source-preserving shader
      binding state, making both fields pipeline dependencies only while their
      slot is enabled in preparation for T10 shader translation.
- [x] **PRE-T10-027** Implement the six default tessellation LOD registers
      (`0x0324..0x0338`) as source-preserving full-width state, making them
      pipeline dependencies only while a tessellation shader stage is enabled.
- [x] **PRE-T10-028** Implement `SET_SUBTILING_PERF_KNOB_A/B`
      (`0x0360`/`0x0364`) as validated, source-preserving Maxwell work-distribution
      policy state that remains outside semantic host-pipeline dependencies.
- [x] **PRE-T10-029** Implement `SET_ZCULL` and `SET_ZCULL_BOUNDS`
      (`0x1968`/`0x196c`) as validated, source-preserving early depth/stencil
      rejection policy that can remain pipeline-neutral when host Z-cull is disabled.
- [x] **PRE-T10-030** Implement `SET_OFFSET_RENDER_TARGET_INDEX` (`0x11f0`)
      as validated, source-preserving viewport-derived attachment routing,
      accepting the disabled capture while rejecting effective routing at lowering.
- [x] **PRE-T10-031** Implement `END` (`0x1614`) as a validated,
      source-preserving primitive-sequence transition that closes the active
      `BEGIN` without altering draw snapshots already queued for execution.
- [x] **PRE-T10-032** Model `SET_REPORT_SEMAPHORE_A..C` (`0x1b00..0x1b08`)
      as source-preserving address and payload state, and execute the captured
      one-word `SET_REPORT_SEMAPHORE_D` release after all preceding writes as
      an ordered, atomically staged guest-memory write.
- [x] **PRE-T10-033** Interpret the captured `STENCIL8_Z24` depth-target
      encoding (`0x16`) as neutral 24-bit depth plus 8-bit stencil semantics
      while retaining its distinct Maxwell packing for future layout conversion.
- [x] **PRE-T10-034** Recognize the captured `S8Z24_2CZ` PTE kind (`0x17`) as
      compressed `STENCIL8_Z24` storage, preserve that guest representation,
      and materialize a neutral depth/stencil image only when a complete clear
      makes importing compressed guest bytes unnecessary; reject prior or
      partial consumption with a typed error.
- [x] **PRE-T10-035** Materialize an enabled compressed Maxwell color target as
      an independent neutral image when a complete RGBA clear overwrites its
      contents, reuse that representation for later operations, and reject
      draws or partial clears that would require importing compressed bytes.
- [x] **PRE-T10-036** Treat `SET_SHADE_MODE=SMOOTH` as the neutral shader-
      declared interpolation path so the captured draw reaches T10, while
      retaining `FLAT` as a typed unsupported primitive-wide interpolation
      override and keeping clears independent of shade mode.
- [x] **PRE-T10-037** Move finite `SET_API_VISIBLE_CALL_LIMIT` enforcement to
      T10 shader evidence, require each translated stage to publish a
      conservative maximum, and reject values above the captured 128-call
      limit before cache or backend effects.
- [x] **PRE-T10-038** Preserve `SET_SM_TIMEOUT_INTERVAL` as validated guest SM
      watchdog policy for diagnostics, while keeping it outside neutral draw
      semantics and never deriving an undocumented host deadline from its
      `COUNTER_BIT` encoding.
- [x] **PRE-T10-039** Preserve `SET_ZCULL_STATS` as validated guest
      instrumentation policy without making counter accumulation a neutral draw
      requirement; require verified semantics only when a future operation
      exposes those counters to the guest.
- [x] **PRE-T10-040** Validate the programmed surface-clip rectangle against
      every selected draw attachment, accept full-coverage clips as neutral,
      and retain typed rejection for partial or offset clipping until the
      neutral raster contract can represent the effective intersection.
- [x] **PRE-T10-041** Add an exact affine viewport transform to neutral draw
      commands and lower enabled Maxwell viewport-zero scale/offset state into
      it, preserving negative axes while rejecting incomplete, empty, or
      non-finite transforms before backend publication.
- [x] **PRE-T10-042** Interpret enabled vertex-array primitive restart
      conservatively at non-indexed draw boundaries, accepting complete point,
      line, and triangle lists where restarting cannot alter assembly while
      retaining typed rejection for connected or incomplete primitives.
- [x] **PRE-T10-043** Derive depth/stencil draw attachment consumption from
      explicitly programmed fragment-test enables, omitting configured
      compressed depth storage when both tests are disabled while retaining
      conservative materialization requirements for active or unknown state.

Exit criterion: every discovered pre-shader blocker is implemented and tested,
and the next external run reaches the T10 shader-decoding boundary.

### Block T10: Maxwell shader decoder, IR, and translation

Purpose: translate guest shader programs without treating guest binaries as
host shaders or tying the decoder to one host language.

- [x] **TRI-180** Identify and pin public sources for every implemented Maxwell
      shader instruction family and header field.
- [x] **TRI-181** Define a bounded shader reader over versioned GPU mappings with
      executable-range validation.
- [x] **TRI-182** Decode shader headers, stage, entry point, instruction
      encodings, control flow, predicates, registers, constant buffers,
      attributes, interpolation, and exits required by the demo.
- [x] **TRI-183** Introduce a typed, verified Nixe shader IR independent of
      WGSL, SPIR-V, MSL, and host backend handles.
- [x] **TRI-184** Preserve numeric width, signedness, rounding, NaN, and
      control-flow semantics for implemented operations, and reject predicate
      registers or numeric modes that the current neutral contract cannot
      represent.
- [x] **TRI-185** Represent resource accesses and stage interfaces explicitly
      so descriptor and pipeline validation can occur before backend code
      generation.
- [x] **TRI-186** Distinguish malformed encoding, unsupported header feature,
      unsupported instruction, unsupported semantic detail, and backend
      lowering limitation.
- [x] **TRI-187** Verify reachable control-flow targets, path-sensitive
      register definitions, reachable exits, resource declarations, and stage
      interfaces before lowering.
- [x] **TRI-188** Implement the minimal vertex and fragment instruction
      families required by the example through ordinary table-driven coverage,
      not shader-byte-pattern recognition.
- [x] **TRI-189** Add a backend-neutral shader interpreter or evaluator for
      small synthetic differential tests where practical.
- [x] **TRI-190** Add WGSL or backend lowering as a separate pass with stable
      source-to-IR-to-host diagnostic locations.
- [x] **TRI-191** Cache translations by shader backing identity, content
      generation, entry point, stage, and translation options rather than GPU
      VA alone.
- [x] **TRI-192** Replace translations after aliased CPU or staged GPU writes,
      retire published shader and dependent pipeline identities, and keep the
      per-binding translation cache bounded.
- [x] **TRI-193** Test hand-authored and captured redistributable shader
      encodings, generated valid instruction subsets, invalid encodings,
      control flow, numeric edge cases, backend lowering, and cache
      invalidation.
- [x] **TRI-194** Make perspective-reciprocal interpolation identical in the
      reference evaluator and WGSL lowering.
- [x] **TRI-195** Replace linear verification with reachable CFG data-flow
      verification and typed rejection of unsupported predicate registers.
- [x] **TRI-196** Validate semantics-affecting SPH flags and keep Maxwell VTG
      launch/output ABI adaptation inside the Maxwell translator with pinned
      public references.
- [x] **TRI-197** Require every draw shader identity to originate from a cached
      verified translation; keep synthetic constructors and cache seeding
      crate-private and test-only.
- [x] **TRI-198** Remove inert translation-generation state and retire replaced
      shader and dependent pipeline resources through the atomic lowering
      plan.
- [x] **TRI-199** Move the captured production translation flow to an
      integration test, reduce public decoder surface, remove stale duplicate
      tests, and version this roadmap checkpoint.

Exit criterion: the example vertex and fragment shaders translate from guest
Maxwell code into verified neutral IR and backend shader modules without
matching their source or binary identity.

External run checkpoint T10: the user confirms shader compilation, program
linking, and the first real draw submission.

### Block T11: Correct accelerated host backend

Purpose: execute neutral operations on a host GPU without making the host API
part of the guest contract.

- [ ] **TRI-200** Define backend initialization and immutable capability
      reporting independently from the Switch 1 GPU profile.
- [ ] **TRI-201** Implement resource creation, destruction, views, samplers,
      shader modules, pipelines, command recording, submission, and completion
      using `wgpu`.
- [ ] **TRI-202** Keep Vulkan as the first enabled backend while avoiding
      Vulkan-specific types in neutral or Switch-specific crates.
- [ ] **TRI-203** Leave an explicit backend selection path for Metal and other
      `wgpu` backends; do not claim macOS support while only Vulkan is enabled.
- [ ] **TRI-204** Implement a conservative canonical-host-memory plus staging
      or device-local-mirror policy.
- [ ] **TRI-205** Upload CPU-newer input ranges before backend consumption.
- [ ] **TRI-206** Mark GPU-written ranges and make required downloads or
      invalidations occur before guest CPU visibility.
- [ ] **TRI-207** Serialize initial submissions if needed for correctness, while
      retaining explicit submission and dependency objects.
- [ ] **TRI-208** Translate neutral access declarations into backend resource
      usages, command ordering, and barriers.
- [ ] **TRI-209** Implement backend completion tokens without equating them to
      guest syncpoints until visibility work is complete.
- [ ] **TRI-210** Handle unsupported formats or operations with typed backend
      capability errors; never substitute a semantically different format
      silently.
- [ ] **TRI-211** Handle host device loss as a typed emulator failure with
      deterministic guest resource teardown.
- [ ] **TRI-212** Compare accelerated clear/draw results against deterministic
      headless or reference results with tolerances defined by guest semantics.

Exit criterion: synthetic buffers, shaders, clears, and draws execute through
the neutral interface on the accelerated backend, and completion makes the
correct backing contents visible.

External run checkpoint T11: run the demo and report backend validation
messages, typed failures, or the produced frame hash/capture.

### Block T12: GPU render target to BufferQueue and presentation

Purpose: connect rendering, ownership, composition, and presentation without
collapsing their separate timelines.

- [ ] **TRI-220** Resolve queued `NvGraphicBuffer` metadata to an image view over
      the correct canonical `nvmap` backing.
- [ ] **TRI-221** Preserve dimensions, format, planes, pitch, kind, transform,
      crop, usage, and slot ownership from producer registration through
      consumption.
- [ ] **TRI-222** Associate queue operations with acquire and release fences
      backed by the guest GPU timeline.
- [ ] **TRI-223** Prevent the compositor from reading a render target before its
      producing fence and visibility transition complete.
- [ ] **TRI-224** Prevent the producer from reusing a slot before consumer
      release.
- [ ] **TRI-225** Convert or copy a completed guest image to the existing
      host-independent `Frame` representation as the initial correct
      presentation path.
- [ ] **TRI-226** Keep composition and VI VSync independent from GPU completion
      and host-window VSync.
- [ ] **TRI-227** Preserve the existing software-framebuffer path and ensure CPU
      and GPU producers use the same BufferQueue ownership rules.
- [ ] **TRI-228** Add headless tests for render completion before/after VSync,
      delayed fences, slot reuse, transforms, and teardown with queued work.
- [ ] **TRI-229** Add frame hash or image assertions for the synthetic triangle.

Exit criterion: a GPU-produced render target travels through the real guest
queue and composition path and becomes a host-ready frame only after correct
completion and ownership transitions.

External run checkpoint T12: the user verifies the first visible
`simple_triangle` frame and that Plus input still exits cleanly.

### Block T13: End-to-end triangle acceptance and regression gate

Purpose: turn the first successful external frame into permanent, layered
evidence without adding a copyrighted binary to the repository.

- [ ] **TRI-240** Record the complete observed service, ioctl, mapping, channel,
      class, method, shader-family, format, and synchronization coverage used by
      the pinned example.
- [ ] **TRI-241** Ensure every accepted operation in that coverage is marked
      complete or explicitly partial with unreachable unsupported variants.
- [ ] **TRI-242** Add a redistributable synthetic end-to-end workload that
      exercises the same architecture: allocation, GPU map, submission, shader,
      clear, draw, fence, queue, and frame.
- [ ] **TRI-243** Assert expected triangle geometry, interpolated colors,
      background clear color, dimensions, and orientation rather than accepting
      any non-empty frame.
- [ ] **TRI-244** Test repeated frames, slot rotation, resize-independent host
      presentation, input exit, guest shutdown, and resource teardown.
- [ ] **TRI-245** Test malformed variants at every external boundary and verify
      typed failures without partial backend effects.
- [ ] **TRI-246** Run formatting, warning-free Clippy, unit, synthetic,
      software-framebuffer, and headless graphics suites.
- [ ] **TRI-247** Document the remaining deliberately conservative policies and
      measured costs using the appendix below.

Exit criterion: the external pinned example visibly works, an equivalent
redistributable synthetic acceptance path protects the architecture, and
software framebuffer behavior remains correct.

External run checkpoint T13: final confirmation across several frames, clean
input exit, and no resource leak or backend validation error.

### Block T14: Coverage-driven expansion toward commercial software

Purpose: expand from one simple draw without abandoning the boundaries proven
by the triangle.

- [ ] **TRI-260** Maintain machine-readable coverage for device ioctls, packet
      forms, classes, methods, shader operations, formats, and backend features.
- [ ] **TRI-261** Prioritize gaps by reproducible caller-owned traces while
      implementing complete semantic families rather than individual title
      signatures.
- [ ] **TRI-262** Complete 3D state coverage for indexed and indirect draws,
      instancing, multiple render targets, depth/stencil, blending, MSAA,
      queries, conditional rendering, mipmaps, arrays, cubemaps, and compressed
      formats as workloads require them.
- [ ] **TRI-263** Expand shader coverage for control flow, integer and
      floating-point edge cases, textures, derivatives, atomics, local memory,
      interpolation, geometry/tessellation where exposed, and cross-stage
      interfaces.
- [ ] **TRI-264** Implement copy, DMA, 2D, inline-to-memory, and compute engines
      as separate verified class handlers.
- [ ] **TRI-265** Add descriptor, texture, sampler, shader, and pipeline cache
      dependency tracking based on backing generations and frontend state.
- [ ] **TRI-266** Support multiple channels, contexts, priorities, and
      inter-channel synchronization with deterministic scheduling tests.
- [ ] **TRI-267** Expand GPU memory support for sparse mappings, large pages,
      remaps, partial residency, aliases, and unusual kinds encountered in
      verified workloads.
- [ ] **TRI-268** Implement cache-maintenance and ordering operations with their
      real guest-visible consequences.
- [ ] **TRI-269** Expand BufferQueue and composition for multiple layers,
      application/applet interaction, transforms, scaling, screenshots, and
      docked/handheld changes.
- [ ] **TRI-270** Add asynchronous backend execution only after timeline,
      lifetime, and visibility stress tests cover races and teardown.
- [ ] **TRI-271** Add bounded capture/replay, resource inspection, shader dumps,
      pipeline diagnostics, and GPU timing metrics without storing proprietary
      artifacts in the repository.
- [ ] **TRI-272** Add differential tests against reference formulas, software
      evaluators, public conformance cases, and real hardware observations that
      can be legally recorded.
- [ ] **TRI-273** Define savestate or suspend/resume behavior only after all
      guest-visible GPU state has pointer-free serialization boundaries.
- [ ] **TRI-274** Introduce another console GPU frontend only from verified
      behavior and reuse neutral memory/backend contracts without weakening
      Switch 1 correctness.

Exit criterion: there is no single final commercial-graphics checkbox. Each
supported semantic family has permanent tests and coverage, and new titles
expand general hardware behavior rather than accumulating per-title hacks.

## 7. Required observability

Graphics development becomes impractical if every unsupported method appears
only as the enclosing `svcSendSyncRequest`. Before broad Maxwell coverage, Nixe
must be able to produce bounded reports at these layers:

```text
Horizon SVC
  nvdrv service command
    fd + device + ioctl
      GPU address-space operation
        channel + submission
          GPFIFO entry
            pushbuffer packet
              class + method + argument
                resource view + backing generations
                  shader PC + decoded instruction
                    neutral operation
                      backend command + completion
```

Reports must:

- use guest and emulator identifiers rather than host addresses;
- include mapping and content generations where stale state is possible;
- bound command, shader, memory, and history dumps;
- preserve the first semantic failure;
- distinguish guest-invalid input from missing emulator support;
- remain usable with the deterministic headless backend;
- and avoid requiring verbose logging for basic failure context.

Capture/replay begins after GPFIFO validation. A capture must declare the GPU
profile and every mapping/resource dependency it contains. Replaying a capture
with missing data or a different profile must fail, not infer defaults.

## 8. Review gates

The following reviews are mandatory before crossing major boundaries:

### Memory gate

Before T4:

- no GPU identity depends on CPU VA;
- canonical page/backing identity is observable;
- mapping and content generations are distinct;
- CPU aliases are tested;
- teardown cannot leave a GPU reference to freed storage.

### Submission gate

Before T7:

- GPU VA mappings are validated and versioned;
- guest timelines exist independently of host completion;
- submissions retain dependencies until completion;
- malformed submission cannot execute partially.

### Backend gate

Before T11:

- Maxwell decoding emits neutral typed operations;
- resource views are separate from allocations;
- shader IR is independent of host shader language;
- the headless validator can reject lifetime and access errors;
- backend capability failure remains a typed host failure.

### Presentation gate

Before declaring the triangle complete:

- GPU completion, memory visibility, BufferQueue ownership, VI VSync, and host
  presentation remain distinct;
- the frame comes from guest GPU output;
- repeated frames and clean teardown work;
- no unsupported operation was accepted as success.

## Appendix A: Deliberately conservative first implementations

The following policies are acceptable during triangle bring-up because they
preserve correctness and the architecture contains an explicit optimization
boundary. They are not permanent performance decisions.

| Area                       | Initial correct policy                                                                | Later optimization                                                                              | Invariant that must survive                                                                      |
| -------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| CPU execution              | Reference interpreter executes guest driver code.                                     | Baseline and optimizing JIT tiers.                                                              | GPU APIs stay outside generated CPU code; memory ordering remains visible at submission helpers. |
| Canonical storage          | Ordinary host RAM stores canonical guest bytes.                                       | Backend-compatible shared allocations or migrated backing.                                      | Stable guest backing identity and generations do not change with residency.                      |
| Discrete-GPU input         | Upload every declared CPU-newer range before use.                                     | Dirty subranges, batched staging, persistent device-local mirrors, compression-aware transfers. | GPU never consumes stale CPU data.                                                               |
| Discrete-GPU output        | Complete and download every declared GPU-written range before it becomes CPU-visible. | Deferred writeback, read-triggered download, subresource tracking.                              | CPU never observes stale data after guest synchronization.                                       |
| Unified-memory hosts       | Use the same staging path initially.                                                  | Shared/imported allocations, host-visible resources, coherent no-copy access.                   | Completion and logical visibility remain explicit even when copies become no-ops.                |
| Dirty granularity          | Whole allocation or whole resource.                                                   | Pages, byte ranges, texture subresources, tiles.                                                | Overlapping aliases invalidate all affected interpretations.                                     |
| Visibility state           | One conservative state per range.                                                     | Version vectors, subresource ownership, queue-family-aware state.                               | Conflicting unsynchronized access is never silently declared coherent.                           |
| GPU scheduling             | Execute one submission at a time.                                                     | Multiple queues, overlap, priorities, asynchronous workers.                                     | Guest ordering, dependencies, and completion values remain identical.                            |
| Host completion            | Wait synchronously where required.                                                    | Polling, callbacks, batched timeline advancement.                                               | Guest fences are not signaled before work and visibility are complete.                           |
| Command parsing            | Decode and validate each submission again.                                            | Cache decoded pushbuffers by backing identity and content generation.                           | Remap or write invalidates stale decoded work.                                                   |
| Shader translation         | Compile synchronously on first use.                                                   | Memory/disk caches and background compilation.                                                  | Cache keys include profile, code generation, stage, entry point, and options.                    |
| Pipeline creation          | Create from complete current state on demand.                                         | Canonical state keys, pipeline caches, prewarming.                                              | A cached pipeline cannot outlive or ignore a relevant state dependency.                          |
| Resource cache             | Recreate views conservatively after writes or state changes.                          | Dependency-directed view reuse and partial invalidation.                                        | Aliased views never retain stale contents or interpretation.                                     |
| Texture layout             | Convert complete resources at explicit boundaries.                                    | Incremental swizzle/unswizzle, GPU compute conversion, native-compatible layouts.               | Guest layout and visible bytes remain correct.                                                   |
| Memory lookup              | Ordered maps and checked segment walks.                                               | Interval trees, radix tables, software GPU TLBs.                                                | Lookup results include mapping generation and preserve holes/permissions.                        |
| GPU TLB                    | No cache, or a trivially invalidated cache.                                           | Per-channel software TLB keyed by address-space and mapping generation.                         | Mapping changes cannot leave usable stale translations.                                          |
| Render-target presentation | Read back or convert to a host-ready `Frame`.                                         | Direct compositor sampling of the backend image.                                                | Buffer ownership and producer fence complete before consumption.                                 |
| Composition                | One visible application layer may be composed directly.                               | Multiple layers, overlays, color management, scaling and capture.                               | Layer state remains explicit and VI timing stays separate from host VSync.                       |
| Frame mailbox              | Publish the latest completed immutable CPU image.                                     | Backend-image mailbox or zero-copy presentation.                                                | Host slowness does not falsify guest GPU or VI completion.                                       |
| Backend choice             | `wgpu` with Vulkan first.                                                             | Metal and other `wgpu` backends, or a more explicit native backend when required.               | Backend limitations do not alter advertised guest GPU behavior.                                  |
| Validation                 | Full frontend and backend validation in normal development.                           | Cached validation results and configurable diagnostics in optimized builds.                     | Unsupported or invalid semantics retain a precise failure path.                                  |
| Capture                    | Bounded in-memory capture around the first failure.                                   | Indexed trace files, selective resource snapshots, deterministic replay tooling.                | Captures contain no raw host pointers or repository-owned proprietary data.                      |

## Appendix B: Unified-memory host path

Unified physical memory does not make CPU/GPU synchronization implicit. The
residency manager must select a policy from actual backend capabilities rather
than an operating-system or product-name check.

Conceptually supported policies are:

```text
HostCanonicalWithDeviceMirror
SharedHostVisibleCoherent
SharedHostVisibleNonCoherent
DeviceLocalWithStaging
BackendSpecificImportedMemory
```

On a suitable unified-memory host, a backing may eventually be allocated in
storage that both CPU execution and the host GPU can access. This can remove
uploads and downloads, but:

- CPU and GPU virtual addresses remain guest addresses;
- resource views and texture layouts remain explicit;
- GPU completion is still required before dependent CPU reads;
- non-coherent host mappings still require flush/invalidate operations;
- API access rules may prohibit simultaneous host mapping and GPU use;
- device-local or transformed image representations may still be faster or
  necessary;
- and alias invalidation remains required.

The neutral backend contract therefore expresses visibility transitions, not
copies. A discrete backend implements a transition with upload/download and
barriers. A coherent unified-memory backend may implement the data movement as
a no-op while retaining ordering and lifetime operations.

The first `wgpu` backend must not promise zero-copy behavior. If `wgpu` mapping
rules prevent persistent shared access, the correct staging implementation is
used. Because shared backing and residency are abstracted above `wgpu`, a Metal
backend or another explicit API backend can later exploit unified memory without
redesigning `nvmap`, GPU VA, Maxwell, or guest synchronization.

## Appendix C: Performance evidence required before optimization

Optimization work should be selected from measurements rather than assumed
bottlenecks. Add counters for:

- CPU-to-GPU bytes uploaded per frame and per allocation;
- GPU-to-CPU bytes downloaded;
- bytes converted between guest and host image layouts;
- full-allocation transfers that could have been ranges;
- mapping and software-TLB hit rates;
- resource-view, shader, and pipeline cache hits and invalidations;
- pushbuffer words decoded and reused;
- host submissions and guest submissions;
- synchronous wait time;
- GPU completion latency;
- fence and BufferQueue wait duration;
- render-target readback cost;
- backend memory usage by canonical, staging, and device-local storage;
- alias-triggered invalidations;
- and frames dropped only by host presentation.

An optimization is complete only when:

1. the measured cost decreases on its target host class;
2. headless and external behavior remains correct;
3. discrete and unified-memory policies still pass the same semantic tests;
4. mapping, aliasing, teardown, and device-loss stress tests pass; and
5. the optimization can be disabled to aid differential diagnosis.
