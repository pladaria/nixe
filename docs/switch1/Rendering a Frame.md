# Rendering a Frame in a Switch 1 Graphics Emulator

This document describes how one rendered frame travels from a title's graphics
API calls to a visible image on the host display. It is a technical reference
for implementing and reviewing the graphics path.

The description follows a GPU-rendered path using OpenGL, NVN, or a similar
guest graphics library. A CPU software renderer uses the same presentation
path; it replaces only the Maxwell image-production stage.

## 1. Complete path

```text
guest title and graphics library
        |
        | Horizon IPC, memory writes, SVCs
        v
guest resources and command streams
        |
        | nvdrv device operations
        v
GPU virtual address space and channel state
        |
        | GPFIFO submission
        v
Switch 1 Maxwell frontend
        |
        | decoded commands and shader execution
        v
render-target backing memory
        |
        | completion, visibility, queueBuffer
        v
Binder / BufferQueue producer-consumer transfer
        |
        v
VI layer and compositor
        |
        v
host-ready immutable frame
        |
        v
host window or headless presentation sink
```

The GPU frontend produces or updates an image. VI and the compositor do not
execute Maxwell commands; they consume a completed image according to display
layer, queue, and synchronization rules.

## 2. Actors and responsibilities

| Actor                       | Responsibility                                                                    |
| --------------------------- | --------------------------------------------------------------------------------- |
| Title                       | Builds scene data and requests draws through a guest graphics API.                |
| Guest graphics library      | Allocates resources, prepares shaders, descriptors, and command streams.          |
| Horizon IPC and SVC layer   | Transports requests, manages handles, mappings, events, waits, and threads.       |
| `nvdrv`                     | Exposes the Switch-facing NVIDIA device ABI and session semantics.                |
| `nvmap`                     | Gives shared allocations an NVIDIA-visible identity and lifetime.                 |
| GPU address-space manager   | Maps allocations into a guest GPU virtual address space.                          |
| GPU channel                 | Owns submission state and accepts GPFIFO work.                                    |
| Maxwell frontend            | Decodes packets, updates GPU state, resolves resources, and emits GPU operations. |
| GPU backend                 | Executes validated operations using a host GPU or reference implementation.       |
| Synchronization coordinator | Relates host completion, memory visibility, guest syncpoints, and fences.         |
| Binder / BufferQueue        | Transfers buffer-slot ownership between producer and compositor.                  |
| VI                          | Owns displays, layers, scaling, visibility, stacking, and VSync events.           |
| Compositor                  | Selects completed layer images and produces host-ready frames.                    |
| Host presenter              | Uploads or copies host-ready frames to the native window.                         |

Identifiers must not cross domains implicitly. A GPU virtual address is not an
`nvmap` handle, a syncpoint is not a host fence, and a BufferQueue slot is not a
render-target resource.

## 3. Frame lifetime and ownership

One frame normally uses a rotating set of image slots:

```text
                 producer                         consumer

      FREE ──dequeue──> DEQUEUED ──queue──> QUEUED ──acquire──> ACQUIRED
       ^                                                            |
       |                                                            |
       └─────────────────────── release ────────────────────────────┘
```

The producer may write only a `DEQUEUED` slot. The compositor may read only an
`ACQUIRED` slot. A queued image cannot be reused until the consumer releases it.

Storage and interpretations have different lifetimes:

```text
canonical allocation
    ├── CPU mapping
    ├── GPU virtual mapping
    ├── vertex-buffer view
    ├── render-target view
    └── graphic-buffer view
```

Destroying a view does not destroy its allocation. Unmapping a GPU address does
not invalidate bytes already retained by an in-flight submission. Such a
submission retains a versioned backing reference, never a raw CPU or host
pointer.

## 4. Establishing the display connection

The title opens a VI root service, normally `vi:u` for an application. The
request crosses the guest Horizon boundary:

```text
guest service call
    → CMIF request
    → svcSendSyncRequest
    → emulated VI service
    → CMIF response and guest handle
```

The SVC and IPC layers carry the request; they do not interpret pixels or GPU
commands.

The title opens the default display, creates or obtains a layer, and sets size,
position, visibility, scaling, transform, alpha, and stacking order. The layer
returns native-window data containing the Binder relay and producer identity.

```text
VI display
    └── layer
          ├── composition properties
          └── Binder producer endpoint
```

The producer endpoint configures buffers, dequeues slots, and queues completed
images. VI decides where the layer is composed, not how its pixels are made.

## 5. Opening `nvdrv` and discovering the GPU

The guest graphics stack opens the NVIDIA service through Horizon. Although the
guest API resembles file descriptors and ioctls, requests are transported by
Horizon IPC:

```text
guest nvOpen / nvIoctl
        |
        v
Horizon CMIF request to nvdrv
        |
        v
typed device operation
```

Typical devices are:

```text
/dev/nvmap          allocation identity and memory objects
/dev/nvhost-as-gpu  GPU virtual address spaces and mappings
/dev/nvhost-gpu     channels and submissions
/dev/nvhost-ctrl    syncpoints, waits, and events
```

Discovery establishes one consistent hardware profile: architecture and class
identifiers, GPC/TPC topology, page sizes, GPU virtual-address width, shader
capabilities, engine capabilities, and synchronization limits. Discovery
determines how later requests are validated; it does not draw.

## 6. The memory model

Graphics memory has explicit address domains:

```text
guest CPU virtual address
        |
        | process page-table translation
        v
canonical guest backing allocation
        ^
        | nvmap identity and GPU mapping
        |
guest GPU virtual address
```

CPU-written vertices and a GPU command can refer to the same bytes:

```text
CPU VA 0x0000001072000000 ─┐
                            ├── canonical allocation 37, offset 0x0000
GPU VA 0x0000000123400000 ─┘
```

Neither guest address is a host pointer. Resolution always goes through the
owning process or GPU address-space tables.

### 6.1 `nvmap` objects

An `nvmap` object gives an allocation an NVIDIA-facing identity and lifetime.
It refers to size, alignment, canonical backing ranges, cache and heap flags,
ownership, imported references, and allocation or mapping generations.

The object is not itself a texture, buffer, or render target. Those are views
over ranges of the allocation.

### 6.2 GPU mappings

The GPU address-space manager maps an `nvmap` allocation into a GPU virtual
address range:

```text
GPU VA range [A, A + size)
        |
        +── mapping generation G
        +── allocation identity
        +── canonical backing range
        +── permissions
        +── page size
        +── surface kind and cache properties
```

Mapping operations validate alignment, bounds, overlap, permissions, page size,
object ownership, and generation. A command referencing an unmapped or stale
range fails before host memory is accessed.

### 6.3 Visibility and cache ordering

A byte write and an address-space remap are different events, so they use
different generations or visibility versions.

```text
Clean
  ├── CPU writes  ──> CpuNewer
  └── GPU writes  ──> GpuNewer

CpuNewer ──device acquire/upload──> Clean
GpuNewer ──visibility/download───> Clean
```

A host backend may use device-local resources, staging buffers, or unified
memory. The guest-visible rule is that a consumer observes writes only after
the required visibility transition completes.

## 7. Creating draw resources

Guest OpenGL/NVN calls execute guest code. A call such as `DrawArrays` does not
directly invoke a host draw call. The guest graphics stack turns it into
resources and commands.

A simple triangle needs at least:

```text
vertex allocation          → vertex-buffer view
vertex-shader allocation   → shader view
fragment-shader allocation → shader view
constants allocation       → descriptor/resource views
color allocation           → render-target view
optional depth allocation  → depth/stencil view
```

Views carry interpretation metadata: buffer offset and size, image format,
dimensions, mip levels, sample count, pitch or block-linear layout, plane
offsets, surface kind, and access permissions.

Every view resolves to canonical backing ranges when consumed. A cached host
resource must be invalidated or updated when an aliased view writes the same
backing bytes.

## 8. Shaders and pipeline state

The guest graphics stack compiles, links, uploads, or selects shaders. The
shader code eventually exists in guest memory in the Switch GPU instruction
format.

```text
guest shader bytes
        |
        v
bounded Maxwell shader reader
        |
        v
instruction decoder and verifier
        |
        v
Nixe shader IR
        |
        +── reference evaluator
        |
        └── host shader lowering
```

The decoder validates instruction boundaries, control-flow targets, register
use, stage information, and resource references before lowering. The IR
preserves numeric widths, signedness, rounding, predicates, register and local
memory access, textures, stage interfaces, barriers, and special values.

Pipeline state combines shader stages with fixed-function state:

```text
shader stages
      + vertex layout
      + resource bindings
      + viewport and scissor
      + rasterization/depth/blend state
      + render-target formats
      = validated pipeline state
```

## 9. Building and submitting GPU commands

The guest graphics stack writes Maxwell command words into guest memory. They
are referenced through GPFIFO entries:

```text
command allocation
    └── pushbuffer words

GPFIFO entry
    ├── pushbuffer GPU address
    ├── word count
    └── submission flags
```

Submitting through `/dev/nvhost-gpu` performs the following operations:

1. Validate the channel and its ownership.
2. Validate GPFIFO counts, sizes, flags, and address arithmetic.
3. Resolve each GPFIFO address through the channel's GPU address space.
4. Retain mapping generations and backing ranges needed by the work.
5. Establish CPU-to-GPU visibility for commands and resources.
6. Assign a frontend submission identity.
7. Schedule the submission according to channel ordering rules.

The frontend retains typed mapping references until completion and visibility.
It does not retain a CPU virtual address or host pointer as a substitute.

## 10. Decoding Maxwell pushbuffers

The Maxwell frontend reads validated pushbuffer words and decodes packet
encodings into method operations:

```text
GPFIFO
  → pushbuffer range
  → packet decoder
  → subchannel and class dispatch
  → class-specific GPU state
```

Packet decoding and execution are separate stages:

```text
raw words
  → decoded packets
  → validated method sequence
  → state transitions
  → neutral GPU operations
```

The decoder validates packet type and length, method range, subchannel,
increment mode, class binding, and word availability across mapping boundaries.
Class dispatch additionally checks that a method is valid for the bound class and
hardware profile.

Unknown packet encodings, unsupported classes, and unsupported methods are
distinct diagnostics. Source context should include channel, submission, GPFIFO
entry, pushbuffer GPU address, word offset, class, method, and mapping
generation.

## 11. Executing the draw

The 3D class accumulates state until a clear or draw has enough information:

```text
bind color render target
bind viewport and scissor
bind vertex buffer and attributes
bind vertex shader
bind fragment shader
bind constants and descriptors
clear color target
draw three vertices
```

The frontend resolves every resource through the GPU address space and emits
host-independent operations:

```text
Clear {
    target: image view,
    color: clear value,
}

Draw {
    pipeline: validated pipeline,
    vertex_streams: ...,
    first_vertex: 0,
    vertex_count: 3,
    target: image view,
}
```

The neutral operation layer contains neither Horizon ioctl numbers, Binder
objects, Maxwell packet words, nor host graphics objects.

For each vertex, the vertex stage reads attributes, applies the vertex shader,
produces a clip-space position, and produces values for later stages. Clipping,
perspective division, viewport mapping, rasterization, interpolation,
depth/stencil, blending, and the fragment shader determine the final color
target writes.

Those writes change the visibility state of the canonical render-target backing:

```text
render-target backing
        |
        └── GPU writes → GpuNewer
```

The result is not guest-visible merely because a host command encoder accepted
the operation.

## 12. Backend execution and completion

The backend receives neutral operations after guest semantics have been
validated:

```text
neutral operations
        |
        v
backend resource and pipeline translation
        |
        v
host command submission
        |
        v
host completion token
```

Completion has distinct stages:

```text
frontend accepted
        |
        v
backend completed
        |
        v
declared GPU writes visible to guest memory
        |
        v
guest syncpoint advanced
        |
        v
guest fence or wait becomes satisfiable
```

A backend token is not a guest syncpoint. Host queue completion is not
automatically guest memory visibility. A completed GPU submission is not a
BufferQueue acquire fence, and a satisfied GPU fence is not a VSync event.

If a GPU write must be downloaded from a host resource, that download and the
canonical-memory visibility transition occur before the guest syncpoint becomes
observable.

## 13. Preparing the display buffer

The render target becomes displayable through a graphic-buffer description:

```text
graphic buffer
  ├── allocation identity
  ├── width and height
  ├── pixel format
  ├── pitch
  ├── plane offsets and sizes
  ├── linear or block-linear layout
  ├── surface kind
  ├── transform and crop
  └── acquire/release fence metadata
```

The same allocation may have different views for GPU rendering and display
composition. The display view preserves the negotiated format and layout.

For a block-linear surface, the compositor performs a layout calculation rather
than treating the allocation as a row-major array:

```text
display pixel (x, y)
        |
        v
surface layout calculation
        |
        v
backing byte offset
```

Any conversion to a host-ready linear image occurs at this explicit boundary
and accounts for format, pitch, planes, block height, kind, crop, and transform.

## 14. Binder and BufferQueue

The title communicates with the producer endpoint through Binder transactions:

```text
Producer                                  Consumer

dequeueBuffer(slot) --------------------->
       |
       | wait for acquire dependencies
       |
       | render into slot
       |
queueBuffer(slot, metadata, fence) ------>
                                           acquire buffer
                                           wait for fence
                                           compose image
<------------------------------------------releaseBuffer(slot)
```

The queue owns state transitions and its availability event. A successful
Binder transaction is not a successful state transition unless the slot,
ownership, metadata, and fence are valid.

Invalid transitions include queuing a free slot, dequeuing an already dequeued
slot, acquiring a non-queued slot, releasing a non-acquired slot, and reusing a
slot before its release.

The queue transfers ownership. It should not silently copy a full image for
each transaction; copying or layout conversion belongs to an explicit
composition or readback step.

## 15. VI composition and VSync

When the compositor can acquire a completed buffer, it associates it with the
VI layer and applies visibility, position, scaling, transform, alpha, crop, and
z-order.

GPU completion and display timing are independent:

```text
GPU completion ───────────────┐
                              v
                         queued image
                              |
VSync boundary ───────────────┘
                              |
                              v
                         latched layer image
```

If the GPU finishes before a refresh boundary, the image may be latched at
that boundary. If it finishes afterward, it waits for a later one. A host
window refresh event must not replace the guest VI VSync event.

## 16. Host-ready frame and final presentation

The compositor converts the selected guest image into immutable presentation
data:

```text
guest graphic buffer
  → resolve canonical backing
  → wait for acquire dependencies
  → decode format and layout
  → apply crop and transform
  → compose layer
  → immutable host-ready frame
```

A host-ready frame contains dimensions, format, sequence, and immutable pixel
storage. It does not expose guest handles, GPU addresses, Binder objects,
runtime events, or host graphics objects to the window-system layer.

The host presenter owns the native window and host graphics API:

```text
host-ready frame
        |
        v
host texture or upload buffer
        |
        v
fullscreen presentation primitive
        |
        v
native window surface
```

The presenter may resize, letterbox, or scale the image to fit the host window.
It must not modify guest layer state or guest render-target memory, and it must
not advance guest GPU syncpoints.

## 17. Software-rendered images

A CPU software renderer follows the same path after image production:

```text
CPU stores pixels
  → guest image-layout conversion
  → cache visibility transition
  → queueBuffer
  → VI/compositor
  → host-ready frame
  → host window
```

No Maxwell channel, pushbuffer, shader decoder, or 3D engine is required. The
producer still needs a valid graphic-buffer description, BufferQueue ownership,
fences, and display timing.

This gives a useful debugging split:

```text
wrong pixels before queueBuffer  → memory, GPU, shader, or layout
wrong slot transition             → Binder/BufferQueue
correct queued image, wrong layer → VI/compositor
correct host-ready frame          → host presenter
```

## 18. Diagnostics and invariants

Every accepted operation has one of three outcomes:

1. verified guest-visible semantics;
2. a verified guest-visible error for invalid guest state or arguments; or
3. a typed emulator diagnostic identifying missing semantics.

Diagnostics should identify the narrowest failed boundary: service command,
device open, `nvmap` ioctl, GPU mapping, GPFIFO entry, pushbuffer packet,
Maxwell method, shader instruction, resource format, backend capability,
BufferQueue transition, or composition operation.

Useful bounded, pointer-free context includes process, file descriptor,
allocation, mapping generation, channel, submission, GPU address, packet offset,
class, method, syncpoint, slot, and frame sequence.

An unknown GPU method must not become a no-op, and an incomplete backend
submission must not become an already-signaled guest fence.

## 19. End-to-end summary

```text
1.  The title opens VI and creates a display layer.
2.  The title obtains a Binder producer endpoint.
3.  The guest graphics stack opens nvdrv devices.
4.  The GPU profile is queried.
5.  Guest allocations are registered with nvmap.
6.  Allocations are mapped into a GPU virtual address space.
7.  Vertices, descriptors, constants, shaders, and images are written.
8.  Guest graphics code builds Maxwell pushbuffers and GPFIFO entries.
9.  A GPU channel accepts a validated submission.
10. Maxwell packets update 3D state and produce neutral clear/draw operations.
11. The backend executes those operations.
12. Host completion and guest-memory visibility are established.
13. Guest syncpoints and fences become observable.
14. The completed render target is described as a graphic buffer.
15. The producer dequeues a BufferQueue slot.
16. The producer queues the slot through Binder.
17. The compositor acquires it after its fence is satisfied.
18. VI applies layer properties at a display refresh boundary.
19. The compositor creates an immutable host-ready frame.
20. The host presenter uploads and displays that frame.
```

The central invariant is that every transition is explicit: address
translation, resource interpretation, command execution, completion, memory
visibility, slot ownership, composition, and host presentation are represented
by the layer that owns their semantics.
