# Nintendo Switch 1 Graphics and Display Architecture

This document explains how a Nintendo Switch 1 application reaches the
display, where Horizon, VI, Binder, `nvdrv`, NVIDIA GPU work, framebuffers,
fences, and VSync fit into that path, and how Nixe should emulate those layers.

The public description of the Switch interfaces is largely the result of
community reverse engineering. Names such as `nvdrv`, `nvmap`, and the VI
command IDs below are supported by public implementations and interface
documentation, but they are not a complete official specification from
Nintendo. Statements about Nixe's future design describe the intended
architecture, not functionality that is necessarily implemented today.

## The short version

A title does not normally access the display controller or NVIDIA GPU
registers directly. It executes graphics libraries in its own process, uses
Horizon IPC to create display objects and access the NVIDIA driver service,
writes pixels or GPU command streams into memory, and submits buffers for
presentation.

```text
                         GUEST APPLICATION
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
         SOFTWARE RENDERING              GPU RENDERING
         CPU writes pixels               NVN prepares:
             directly                     - commands
                 │                        - shaders
                 │                        - textures
                 │                        - geometry
                 │                             │
                 │                             ▼
                 │                    Emulated Maxwell GPU
                 │                    executes the commands
                 │                             │
                 ▼                             ▼
        Software framebuffer            GPU render target
                 │                             │
                 └──────────────┬──────────────┘
                                │
                                ▼
                       Binder BufferQueue
                    transfers buffer ownership
                                │
                                ▼
                            VI layer
                  defines where/how it is displayed
                                │
                                ▼
                      Emulated compositor
                                │
                                ▼
                   Host window or headless sink
```

There is no general `svcCreateFramebuffer` call. A framebuffer is a
composition of ordinary guest memory, an NVIDIA memory object, graphic-buffer
metadata, a BufferQueue slot, and display-service state.

Similarly, an API function named `nvIoctl` is not a direct host-kernel
`ioctl`. In libnx it encodes a CMIF request to the Horizon `nvdrv` service.
The pinned libnx implementation shows `nvOpen` using command 0 and `nvIoctl`
using command 1 through `serviceDispatch*`:
[`nv.c`](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/nv.c).

## The layers

The graphics path contains several layers with different responsibilities:

```text
┌─────────────────────────────────────────────────────────────────┐
│ Application and graphics libraries                              │
│ console renderer, game engine, NVN, EGL/OpenGL, Vulkan, etc.    │
├─────────────────────────────────────────────────────────────────┤
│ Image production                                                │
│ CPU-written framebuffer or GPU-produced render target           │
├─────────────────────────────────────────────────────────────────┤
│ NVIDIA driver interface                                         │
│ nvdrv, nvmap, GPU address spaces, channels, submissions, fences │
├─────────────────────────────────────────────────────────────────┤
│ Window and buffer transport                                     │
│ NWindow, Binder relay, IGraphicBufferProducer, BufferQueue      │
├─────────────────────────────────────────────────────────────────┤
│ Horizon display services                                        │
│ VI displays, layers, composition properties, VSync events       │
├─────────────────────────────────────────────────────────────────┤
│ Physical implementation on a Switch                             │
│ Maxwell GPU, display engine, panel or HDMI output               │
└─────────────────────────────────────────────────────────────────┘
```

The top four layers execute partly inside the title and partly in Horizon
services. The last layer is real hardware on the console. Nixe must reproduce
the guest-visible contracts of all relevant upper layers, but it does not need
to reproduce electrical LCD or HDMI signals.

### Library calls are not SVCs

Functions such as `consoleInit`, `nwindowGetDefault`, `framebufferCreate`, and
`nvMapCreate` are guest library functions. They execute many ordinary AArch64
instructions. Some operations remain entirely local, while others eventually
send IPC or invoke an SVC.

```text
guest function call
        │
        ├── local computation
        │   ├── calculate dimensions and alignment
        │   ├── allocate memory
        │   ├── draw pixels
        │   └── build command or Parcel data
        │
        └── cross a Horizon boundary
            ├── svcSendSyncRequest for HIPC/CMIF
            ├── memory-management SVCs
            └── wait or event SVCs
```

The SVC layer transports requests, manages handles and memory, and suspends or
wakes threads. It does not inherently understand pixels, RGB565, framebuffers,
GPU channels, or BufferQueue slots. Those meanings belong to the services and
protocols above it. See [IPC, HIPC and CMIF](IPC,%20HIPC%20and%20CMIF.md) for
the transport model.

## Horizon, VI, and the window

VI is Horizon's display-service family. Publicly documented roots include
`vi:u`, `vi:s`, and `vi:m`, with different privilege levels. The application
display service exposes operations including opening a display, opening or
creating a layer, obtaining the Binder relay service, and retrieving a VSync
event. The command inventory is recorded in
[Switchbrew's display-service documentation](https://switchbrew.org/wiki/Display_services),
and the same command IDs can be inspected in pinned libnx
[`vi.c`](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/vi.c).

For a contemporary libnx application, the default-window initialization is
approximately:

```text
__nx_win_init()
    │
    ├── viInitialize()
    ├── viOpenDefaultDisplay()
    ├── viCreateLayer()
    │   ├── request a managed layer from applet, when applicable
    │   └── open the VI layer and receive NativeWindow data
    ├── viSetLayerScalingMode()
    └── nwindowCreateFromLayer()
        ├── extract the Binder object ID
        ├── connect to IGraphicBufferProducer
        └── obtain its availability event
```

This exact high-level sequence appears in pinned libnx
[`default_window.c`](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/display/default_window.c)
and
[`native_window.c`](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/display/native_window.c).

A VI layer is not the pixel allocation itself. It is a compositor-visible
surface with properties such as size, position, scaling, transform, alpha,
visibility, and stacking order. Its native-window data identifies the
producer side of the queue through which images reach that layer.

## Binder and BufferQueue

The Switch graphics stack reuses Android-style Binder Parcel and
`IGraphicBufferProducer` concepts. On the Switch, libnx sends Binder
transactions through the Binder relay object returned by VI rather than by
invoking a Linux Binder device directly. Pinned libnx
[`binder.c`](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/display/binder.c)
shows each transaction being carried by `serviceDispatchIn` on that relay
session.

The conceptual queue lifecycle is:

```text
Producer: title                         Consumer: compositor

dequeueBuffer()
       │
       ▼
obtain a free slot
       │
       ├── wait for its acquire fence when necessary
       │
       ▼
write pixels or render with the GPU
       │
       ▼
queueBuffer(slot, metadata, fence) ───────────────► acquire image
                                                     │
                                                     ▼
                                                  compose
                                                     │
                                                     ▼
release slot + signal availability ◄─────────────────┘
```

BufferQueue moves ownership of slots and buffer handles; it should not copy a
full image for every queue operation. This producer/consumer behavior is also
described by the
[Android Open Source Project BufferQueue documentation](https://source.android.com/docs/core/graphics/arch-bq-gralloc).
The Android documentation is useful for the inherited model, while pinned
libnx remains the closer reference for the Switch wire representation and
service path.

Nixe therefore needs real queue state:

- which slots have been configured;
- which slot is dequeued by the producer;
- which queued buffers are ready for the consumer;
- whether a fence still blocks a buffer;
- when the consumer releases a slot;
- whether a dequeue operation should succeed, return `WouldBlock`, or sleep;
- which event becomes signaled when a buffer becomes available; and
- the negotiated dimensions, format, usage, transform, crop, and swap
  interval.

Returning success without maintaining this state would allow initialization
to advance, but the title would eventually overwrite an in-use image, wait
forever, or observe impossible queue transitions.

## `nvdrv` is the Horizon gateway to the NVIDIA driver

`nvdrv` is both part of Horizon's service architecture and the title-facing
interface to NVIDIA driver functionality. These descriptions are compatible:

```text
Application
    │ HIPC/CMIF through svcSendSyncRequest
    ▼
nvdrv / nvdrv:a / nvdrv:s
    │ driver operations represented as fd + ioctl number + buffers
    ▼
NVIDIA memory, channels, engines, fences, and GPU
```

The service variant depends on the process type. Pinned libnx opens `nvdrv`
for applications, `nvdrv:a` for applets, and `nvdrv:s` for system processes in
[`nv.c`](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/nv.c).
The known service commands, device paths, permission masks, and ioctl
structures are catalogued in the versioned
[Switchbrew NV services revision](https://switchbrew.org/w/index.php?title=NV_services&oldid=14790).

The exposed paths resemble NVIDIA's Linux driver interfaces:

| Path | Main role |
| --- | --- |
| `/dev/nvmap` | Create and identify memory objects shared with NVIDIA engines. |
| `/dev/nvhost-as-gpu` | Manage a GPU virtual address space and mappings. |
| `/dev/nvhost-gpu` | Create a GPU channel and submit graphics or compute work. |
| `/dev/nvhost-ctrl` | Syncpoints, waits, events, and control operations. |
| `/dev/nvhost-vic` | Video image compositor engine operations. |
| `/dev/nvhost-nvdec` | Video decoder operations. |

An application can bypass NVN or another high-level graphics API and issue
`nvdrv` requests itself. That is “direct driver use” in an API sense. A normal
application still does not legitimately map arbitrary Tegra MMIO registers or
program the hardware outside the permissions and mappings granted by Horizon.
The privileged driver service performs the hardware-facing work on a real
console.

## Memory has more than one address space

Graphics memory must not be modeled as a native host pointer. At least three
address domains are relevant:

```text
Guest CPU virtual address
0x0000001072000000
        │
        │ process page-table translation
        ▼
Emulated backing allocation
object 37, offset 0
        ▲
        │ nvmap identity + GPU address-space mapping
        │
Guest GPU virtual address
0x0000000123400000
```

The same backing bytes may therefore be reachable from different guest CPU
and GPU addresses. Neither address is the Rust address of the host allocation.

`nvmap` provides the identity and properties of a memory allocation.
`/dev/nvhost-as-gpu` operations map such allocations into a GPU virtual
address space. Command buffers, shaders, textures, vertex buffers, and render
targets then refer to GPU virtual addresses. Nixe must translate those GPU
addresses through the emulated GPU mappings to the same backing storage used
by CPU memory accesses.

This relationship is essential for correctness:

```text
CPU stores vertex data at CPU VA
              │
              ▼
       shared backing bytes
              ▲
              │
GPU command references mapped GPU VA
```

If Nixe creates separate copies without a defined coherence mechanism, CPU
writes will not reach the GPU and GPU-produced images will not reach the
display.

### Layout and cache visibility

Image memory is not necessarily row-major linear memory. NVIDIA surfaces can
use pitch-linear or block-linear layouts and carry a `kind` describing their
interpretation. For example, pinned libnx's software framebuffer allocates
RGB565 storage, registers it with `nvmap`, describes the display plane as
block-linear, converts from a convenient linear drawing buffer, and flushes
the data cache before queueing:
[`framebuffer.c`](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/display/framebuffer.c).

Nixe must consequently track:

- allocation size and alignment;
- CPU mappings and permissions;
- `nvmap` handle and exported ID;
- GPU virtual mappings;
- color format, pitch, planes, offsets, layout, and kind;
- CPU/GPU ownership or visibility transitions; and
- dirty ranges where an optimized backend needs them.

A first correct implementation may make cache maintenance immediately
coherent while still validating every cache-management request. Later
optimization may defer copies or conversions, but it must preserve the same
observable ordering.

## The software-framebuffer path

The libnx software console is a useful first graphics target because no
Maxwell 3D execution is required to produce its pixels.

The renderer initialization in pinned
[`console_sw.c`](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/runtime/devices/console_sw.c)
does the following:

```text
consoleInit()
    │
    ├── select the software console renderer
    ├── nwindowGetDefault()
    ├── choose character-grid dimensions
    ├── framebufferCreate(..., RGB565, 2)
    └── framebufferMakeLinear()
```

`framebufferCreate` then:

1. initializes `nvdrv`, `nvmap`, and fence support;
2. allocates page-aligned CPU memory for the images;
3. creates an `nvmap` object backed by that memory;
4. constructs `NvGraphicBuffer` metadata;
5. describes block-linear RGB565 planes; and
6. registers each image in a BufferQueue slot with
   `SetPreallocatedBuffer`.

Drawing and presentation follow a different path:

```text
framebufferBegin()
    │
    ├── dequeue a free BufferQueue slot
    └── return the linear drawing allocation

console writes RGB565 glyph pixels with ordinary CPU stores

framebufferEnd()
    │
    ├── convert linear pixels to NVIDIA block-linear layout
    ├── flush the guest data cache
    └── queue the selected slot through Binder
```

This gives Nixe a deliberately narrow first milestone:

```text
emulate VI + Binder queue + minimal nvmap
                   │
                   ▼
recognize the registered RGB565 allocation
                   │
                   ▼
on QueueBuffer, resolve its guest backing bytes
                   │
                   ▼
unswizzle block-linear RGB565 to a host image
                   │
                   ▼
present it or retain it in a headless test sink
                   │
                   ▼
release the slot and signal its availability
```

No Maxwell draw commands need to be interpreted for this path. Nevertheless,
the service, memory, queue, and synchronization behavior must be real enough
that the guest produces and owns the buffers normally.

## The 3D path

Commercial games generally produce render targets with the GPU rather than
writing their final images pixel by pixel on the CPU. The Switch 1 hardware
uses a Maxwell-generation Tegra GPU; NVIDIA's
[Tegra X1 architecture whitepaper](https://images.nvidia.com/content/pdf/tegra/Tegra-X1-whitepaper-v1.0.pdf)
documents the Maxwell GPU organization and shared system memory of that SoC
family. Public Switch GPU class and shader information remains primarily
community-derived.

The guest CPU still executes the title, game engine, and graphics library. A
high-level draw call does not become one Horizon IPC request per triangle.
Instead, the guest graphics stack builds GPU state and command streams in
memory, then submits batches through a GPU channel:

```text
game engine
    │
    ▼
guest graphics library, commonly NVN
    │
    ├── allocate/map resources through nvdrv
    ├── write shaders, descriptors, constants, vertices, and textures
    ├── encode Maxwell command streams in guest memory
    └── submit GPFIFO entries through an nvdrv ioctl
                                      │
                                      ▼
                              GPU channel execution
```

The versioned
[Switchbrew NV services revision](https://switchbrew.org/w/index.php?title=NV_services&oldid=14790)
records the GPU address-space and channel ioctls, including GPFIFO submission.
Known Maxwell class methods are separately catalogued by
[Switchbrew GPU Classes](https://switchbrew.org/wiki/GPU_Classes).

### Nixe should intercept work at the driver boundary

Nixe does not need to recognize every internal NVN function by guest program
counter. The durable boundary is the emulated driver:

```text
nvdrv GPU submission
        │
        ├── resolve GPFIFO entries
        ├── translate GPU virtual addresses
        ├── decode Maxwell pushbuffer packets and class methods
        ├── update emulated GPU state
        ├── execute draws, dispatches, copies, and clears
        └── advance syncpoints and signal fences
```

This preserves compatibility with titles that use NVN, another guest graphics
library, or direct `nvdrv` calls, provided they generate supported hardware
commands.

### Proposed internal graphics split

The console-facing protocol and host rendering backend should remain
separate:

```text
┌───────────────────────────────────────────────────────────┐
│ Switch-specific frontend                                  │
│ VI, Binder, nvdrv, nvmap, GPU VA, Maxwell command decoder │
└──────────────────────────────┬────────────────────────────┘
                               │ validated internal commands
                               ▼
┌───────────────────────────────────────────────────────────┐
│ Platform-independent graphics execution model             │
│ resources, shaders, pipelines, draws, copies, barriers     │
└──────────────────────────────┬────────────────────────────┘
                               │
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
┌──────────────────────────┐         ┌────────────────────────┐
│ Accelerated host backend │         │ Headless/test backend  │
│ Vulkan or another API    │         │ validation and capture │
└──────────────────────────┘         └────────────────────────┘
```

The final host API should be selected when requirements are understood; this
document does not prescribe Vulkan as a permanent choice. The important
boundary is that Horizon services do not call SDL or a host GPU API directly,
and that the Maxwell decoder does not own window-system behavior.

Shaders require their own translation path:

```text
guest Maxwell shader code
          │
          ▼
decode and validate guest shader ISA
          │
          ▼
Nixe shader intermediate representation
          │
          ├── control flow and predicates
          ├── registers and local/shared memory
          ├── textures and samplers
          └── stage inputs and outputs
          │
          ▼
host shader representation + pipeline state
```

Translation must preserve Switch-visible numerical behavior, resource
bindings, texture formats, barriers, and stage semantics. Unsupported GPU
methods or shader operations must produce precise diagnostics rather than
silently behaving as no-ops.

## Composition and presentation are not rendering

Rendering creates an image. Composition chooses how one or more layer images
appear on a display. Presentation makes the composed result visible at a
display boundary.

```text
render target A ──► layer A ──┐
                              │ position, scale, crop,
render target B ──► layer B ──┼─► compositor ─► displayed frame
                              │ alpha, transform, Z order
system layer ─────► layer C ──┘
```

For a first single-application implementation, Nixe may have one visible layer
and present it directly. The state model should still retain explicit display
and layer objects so that scaling, docked/handheld resolution changes,
overlays, applets, screenshots, and multiple layers do not require replacing
the architecture later.

The compositor output should flow into a host-independent presentation
interface. An SDL window can be one consumer, while tests can use a headless
consumer that hashes or captures frames.

## VSync, fences, and buffer availability

These synchronization mechanisms are related but not interchangeable.

| Mechanism | Meaning |
| --- | --- |
| GPU fence or syncpoint | A submitted GPU operation reached a defined completion point. |
| Buffer acquire fence | The producer or consumer must not use the buffer until prior work completes. |
| Buffer availability event | A BufferQueue slot can be dequeued or its state changed. |
| VI VSync event | A display refresh boundary occurred. |

VI exposes `GetDisplayVsyncEvent` as command 5202 in pinned libnx
[`vi.c`](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/services/vi.c).
On Nixe, a virtual display clock should generate this guest-visible event at
the configured display cadence. The event wakes threads through the ordinary
emulated Horizon scheduler and wait mechanisms.

```text
virtual display clock
        │ refresh boundary
        ├── latch eligible queued layers
        ├── compose/present a frame
        └── signal VI VSync event
                    │
                    ▼
             WaitSynchronization wakes
```

The exact policy must define what happens if host presentation is late,
emulation is paused, fast-forward is active, or deterministic mode is enabled.
Host monitor VSync must not automatically become guest VSync: tying them
directly would make guest timing depend on the host monitor and window state.

GPU completion is a separate timeline. A frame cannot be consumed merely
because VSync occurred if its GPU fence has not completed. Conversely, a GPU
fence may complete long before the next VSync.

## Docked and handheld operation

Docked or handheld operation mode is Horizon runtime state, not a different
CPU instruction or a direct LCD-controller signal. A mode change may affect
the display resolution selected by software, performance configuration,
scaling, and application behavior.

Nixe should propagate a mode transition through the relevant emulated system
and applet state, update display capabilities where required, and emit the
same kind of notification the guest expects. Existing buffers do not
magically change layout: the application or its graphics library may react by
recreating swapchain resources.

The host window can resize independently. Host resizing should normally
change how Nixe scales the final image, not falsify a guest dock transition.

## What does not need physical emulation initially

For normal title execution, Nixe does not initially need:

- electrical panel timing;
- HDMI link training or packet encoding;
- physical DisplayPort-to-HDMI controller behavior;
- analog characteristics of the LCD;
- real Tegra MMIO register addresses exposed to application code; or
- a one-to-one reproduction of the physical display engine's internal
  implementation.

It does need every guest-observable consequence used by supported software:

- service results and object lifetimes;
- buffer formats and memory contents;
- queue blocking and release;
- layer properties;
- fences, syncpoints, events, and timeouts;
- presentation cadence;
- error behavior; and
- GPU command semantics for 3D titles.

Direct MMIO emulation would become relevant if Nixe later executes privileged
firmware or driver code that legitimately accesses those regions. That is a
different scope from serving a normal application's `nvdrv` IPC.

## Proposed implementation stages

The following ordering allows incremental progress without making the
software-framebuffer path a dead-end.

### Stage 1: VI object model

- implement the required VI service roots and child objects;
- open the default display;
- create or open managed and stray layers as required;
- return and validate native-window data;
- track layer size, scaling, crop, transform, visibility, and lifetime; and
- provide virtual VSync events.

### Stage 2: Binder and BufferQueue

- implement the VI Binder relay;
- parse and encode Parcel data with strict bounds checking;
- implement the required `IGraphicBufferProducer` transactions;
- maintain slot ownership and queue ordering;
- implement dequeue blocking, availability events, and disconnect behavior;
- retain graphic-buffer metadata instead of accepting opaque bytes; and
- test malformed Parcels and invalid state transitions.

### Stage 3: minimal `nvdrv` and software presentation

- initialize the application `nvdrv` session;
- implement `/dev/nvmap` create, allocate, ID, query, and free operations;
- bind nvmap objects to guest backing memory;
- support the fence behavior needed by the software framebuffer;
- identify queued RGB565 buffers;
- convert supported NVIDIA layouts into a linear host image; and
- expose both a headless frame sink and a windowed presenter.

At the end of this stage, the libnx hello-world console should be capable of
displaying text without a 3D GPU implementation.

### Stage 4: GPU memory and submission infrastructure

- implement GPU virtual address spaces;
- map nvmap allocations at GPU addresses;
- implement channels and GPFIFO submission;
- model syncpoints, events, timeouts, and error notifications;
- validate every submitted address and range; and
- preserve ordering between CPU memory, GPU work, and BufferQueue fences.

### Stage 5: Maxwell command execution

- decode pushbuffer packets and supported GPU classes;
- model graphics, compute, copy, and inline-to-memory state;
- implement render targets, textures, samplers, vertex input, and clears;
- translate shaders through an explicit intermediate representation;
- submit equivalent work to an accelerated host backend; and
- report unsupported methods and shader instructions with actionable
  diagnostics.

### Stage 6: composition and timing fidelity

- compose multiple layers;
- apply crop, scaling, transforms, alpha, and Z order;
- coordinate GPU completion, queue acquisition, VSync, and presentation;
- handle docked/handheld transitions and display-mode changes;
- implement pause, fast-forward, and deterministic display-clock policies;
- support capture and frame-debugging tools; and
- characterize latency and queue-depth behavior against public references and
  reproducible guest tests.

## Testing strategy

Graphics tests should not require a physical host window or nondeterministic
wall-clock timing.

Useful layers of testing include:

| Test level | Examples |
| --- | --- |
| Wire format | Parcel bounds, graphic-buffer metadata, ioctl structures, CMIF handles. |
| State machines | Slot transitions, disconnect, queue full, fence pending, invalid reuse. |
| Memory | CPU/GPU aliases, nvmap lifetime, address overflow, block-linear conversion. |
| Timing | VSync cadence, timeouts, blocked dequeue wakeup, late GPU completion. |
| Rendering | Known clears, triangles, textures, blend cases, shader microprograms. |
| Acceptance | A real redistributable homebrew creates a console and submits visible frames. |

A headless frame sink should expose deterministic image bytes, dimensions,
format, layer metadata, and presentation sequence numbers. Tests can compare
small golden images or hashes where appropriate, while lower-level conversion
tests should compare complete bytes so that failures remain diagnosable.

Real-content acceptance tests should assert observed boundaries precisely. For
example, successful VI initialization is not proof that a framebuffer was
created, and successful buffer registration is not proof that a frame was
queued or presented.

## Architectural rules

The following rules prevent convenient early shortcuts from becoming
long-term incompatibilities:

1. Do not treat guest CPU addresses, GPU addresses, and host pointers as the
   same namespace.
2. Do not return success from Binder or `nvdrv` commands without applying the
   state transition that success promises.
3. Do not treat VSync, GPU completion, and buffer availability as the same
   event.
4. Do not assume that every image is linear, RGB, single-plane, or
   CPU-readable.
5. Do not copy full buffers merely to simplify handle or slot ownership unless
   the copy is an explicit backend operation.
6. Do not put SDL, Vulkan, or another host API inside Horizon service objects.
7. Do not identify guest graphics libraries by fixed program counters when a
   stable driver or hardware boundary exists.
8. Do not interpret unknown GPU methods or shader instructions as no-ops.
9. Keep a deterministic, headless presentation path available for tests.
10. Keep software-produced and GPU-produced images on the same memory,
    BufferQueue, composition, and presentation model.

The last rule is the central design constraint. The hello-world framebuffer
is the smallest producer that reaches the display pipeline; it should exercise
the same ownership and presentation architecture that a later 3D render
target will use.
