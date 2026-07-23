# Nintendo Switch 1 IPC, HIPC, and CMIF

This document explains Horizon inter-process communication, the roles of HIPC
and CMIF, how they interact with ports, sessions, handles, services, SVCs,
memory, and process permissions, and how those layers are currently
implemented in Nixe.

The protocol information is based on public community reverse engineering.
Some names are descriptive community names rather than terminology officially
published by Nintendo.

> **Nixe implementation status**
>
> The implementation described here is functional, bounded, and covered by
> tests, but it is incomplete and still evolving. The available services,
> command coverage, limits, result mappings, wire behavior, and internal module
> boundaries may change as the implementation develops. This document
> describes the repository at the time of writing; it does not define a stable
> internal API.

## The short version

An application does not directly call a Rust function belonging to a service.
It writes an HIPC/CMIF message into guest memory, invokes an SVC with a session
handle, and receives a response that may contain data, handles, or CMIF domain
objects.

```text
Guest application
    │
    │ writes a message into the thread IPC command buffer
    │
    │ HIPC: transport, descriptors, PID, and handles
    │ CMIF: object, command ID, arguments, and result
    ▼
svcSendSyncRequest(handle)
    │
    ▼
Horizon kernel / Nixe SVC dispatcher
    │
    ├── session for a service implemented inside Nixe
    │       └── decode, validate, dispatch, and encode a response
    │
    └── generic session connected to another guest process
            └── deliver the request to the guest server endpoint
    │
    ▼
HIPC/CMIF response in the command buffer
```

The essential terms are:

| Term | Role |
| --- | --- |
| IPC | The general concept of communication between processes. |
| HIPC | Horizon's transport format: headers, handles, memory descriptors, and raw data. |
| CMIF | The common object and command protocol carried inside HIPC raw data. |
| SVC | The guest-to-kernel entry used to connect, send, wait, reply, or close. |
| Port | A published connection point from which sessions are created. |
| Session | A client/server channel used to exchange requests and responses. |
| Handle | A process-local integer referring to a session or another kernel object. |
| Service | A named interface such as `sm:`, `fsp-srv`, or `hid`. |
| CMIF domain | Multiple service objects multiplexed by object ID over one session handle. |

## The layers

IPC is not one monolithic binary format. A typical request crosses several
distinct layers:

```text
┌──────────────────────────────────────────────────────────────┐
│ Service semantics                                            │
│ Example: IFile::Read(offset, size), GetOperationMode()       │
├──────────────────────────────────────────────────────────────┤
│ CMIF                                                         │
│ Command ID, token/context, result, arguments, domain objects │
├──────────────────────────────────────────────────────────────┤
│ HIPC                                                         │
│ Message type, handles, PID, memory descriptors, raw data     │
├──────────────────────────────────────────────────────────────┤
│ Kernel sessions and objects                                  │
│ Client/server endpoints, queues, waits, replies, closure     │
├──────────────────────────────────────────────────────────────┤
│ SVC, CPU state, and virtual memory                           │
│ Registers, TLS or user buffer, and guest memory access       │
└──────────────────────────────────────────────────────────────┘
```

Each layer answers a different question:

- HIPC defines **how the message components are transported**.
- CMIF defines **which object and command are addressed**, how arguments are
  framed, and how a service result is returned.
- The service interface defines **what the command means**.
- The session and kernel define **who communicates with whom**, when a thread
  waits, and when it resumes.
- A handle lets a process refer to an object without seeing a kernel or host
  pointer.

CMIF does not replace HIPC: it normally travels inside an HIPC message. HIPC
also does not imply CMIF. TIPC is a smaller protocol used by some services on
later Horizon versions and also travels over the HIPC transport. Nixe
currently implements the CMIF path described here, not a general TIPC
implementation.

## Ports, sessions, services, and handles

### Ports and sessions

A port is a connection point. A server publishes or manages a port, and a
client connects to it. Each successful connection creates a session with two
endpoints:

```text
                   named or direct port
                            │ connect
                            ▼
Guest client     client endpoint ═══════ server endpoint     Guest server
    handle C ───────────────┘                  └──────────── handle S
```

Handle values are process-local and therefore do not need to match. The two
handles identify related endpoints, but they are not one global integer.

A synchronous request follows this lifecycle:

1. The client submits a request through its endpoint.
2. If no response is ready, the client thread is suspended.
3. The server waits with `ReplyAndReceive`, receives the request, and handles
   it.
4. The server submits a response.
5. The client wakes and continues with the response.

The shared session state also records whether each endpoint remains open.
Closing the last handle for an endpoint makes the peer closure observable.

### The Service Manager

Applications normally do not connect to every service port directly. They
first invoke `ConnectToNamedPort("sm:")`, which returns a session connected to
the Service Manager:

```text
ConnectToNamedPort("sm:")
        │
        ▼
sm: session handle
        │ CMIF command 0: RegisterClient, with PID
        ▼
registered client
        │ CMIF command 1: GetService("fsp-srv")
        ▼
fsp-srv session handle
```

`sm:` is a service registry and broker. It is not the IPC transport; the
requests sent to `sm:` are themselves HIPC/CMIF requests.

### Copy and move handles

An HIPC message may contain two handle lists:

- a **copy handle** gives the receiver another reference to the same object
  while the sender keeps its reference;
- a **move handle** transfers ownership and consumes the source handle.

This distinction matters when services return objects. Opening a filesystem or
file outside a domain normally returns a moved handle for the child object.
Events and shared-memory objects may be returned as copied handles when the
specific command ABI requires it.

The special values `0xffff8000` and `0xffff8001` represent the current thread
and current process in operations that accept those pseudo-handles. They are
not normal entries in the process handle table.

## The command buffer and TLS

For ordinary `SendSyncRequest`, the message occupies the IPC command buffer at
the beginning of the thread-local region. The CPU's TLS register identifies
that region:

| CPU mode | Register used by Nixe |
| --- | --- |
| AArch64 | `TPIDR_EL0` |
| AArch32 | `TPIDRURW` |

Nixe allocates and maps TLS while building the process, initializes the
appropriate register, and treats the first `0x100` bytes as the fixed IPC
command buffer. This is guest virtual memory. Neither the TLS address nor any
address contained in a descriptor is a Rust pointer.

```text
Guest thread-local region
┌──────────────────────────────────┐  ← TPIDR_EL0 / TPIDRURW
│ HIPC command buffer (0x100)      │
├──────────────────────────────────┤
│ remaining thread-local storage   │
└──────────────────────────────────┘
```

`SendSyncRequestWithUserBuffer` instead receives an explicit guest address and
size. Nixe checks page alignment, rejects an empty or overflowing range, and
uses the complete region for the generic guest-session transport.

> **Current limitation:** the codec used by Nixe's built-in services still
> assumes the fixed `0x100`-byte TLS command buffer. The user-buffer SVC is
> therefore not yet a generally usable path for those built-in services: its
> ABI requires a page-sized region, while that codec rejects input larger than
> the fixed command buffer. This boundary needs to be separated or generalized
> and should be considered unstable.

## HIPC: the transport layer

HIPC places transport metadata and guest-memory references around a raw-data
section:

```text
variable offset
┌────────────────────────────────────────────┐
│ HeaderData: two 32-bit words               │
├────────────────────────────────────────────┤
│ SpecialHeaderData, when present            │
│   ├── optional PID                         │
│   ├── copy handles                         │
│   └── move handles                         │
├────────────────────────────────────────────┤
│ send-static descriptors                    │
├────────────────────────────────────────────┤
│ send / receive / exchange buffer descs.    │
├────────────────────────────────────────────┤
│ raw data words                             │
│   └── CMIF data aligned to 16 bytes        │
├────────────────────────────────────────────┤
│ receive-static descriptors, when present   │
└────────────────────────────────────────────┘
```

### HIPC header

Nixe decodes the initial words as follows:

| Field | Bits | Meaning |
| --- | ---: | --- |
| `command_type` | word 0, 0..15 | Message type interpreted by CMIF. |
| send-static count | word 0, 16..19 | Number of static input descriptors. |
| send-buffer count | word 0, 20..23 | Number of mapped input buffers. |
| receive-buffer count | word 0, 24..27 | Number of mapped output buffers. |
| exchange-buffer count | word 0, 28..31 | Number of bidirectional buffers. |
| data-word count | word 1, 0..9 | Raw section size in 32-bit words. |
| receive-static mode | word 1, 10..13 | None, automatic, or an encoded entry count. |
| special-header present | word 1, bit 31 | Indicates PID and/or handle lists. |

Reserved bits must be zero. Counts and size calculations are checked before
memory is reserved or traversed.

The public format also defines `ReceiveListOffset` in bits 20..30 of the second
word. Nixe's current codec does not model that value independently and expects
the receive-static list after the raw data. Supporting every valid transport
layout will require this area to be extended.

### Special header

When present, the special header states:

- whether the sender supplies its process ID;
- how many handles are copied;
- how many handles are moved.

The PID is not an ordinary user-selected service argument. The kernel conveys
the sender's identity, and many commands require it. Examples include
`sm:RegisterClient`, `fsp-srv:SetCurrentProcess`,
`hid:CreateAppletResource`, and opening the `appletOE` application proxy.

### Memory descriptors

Large arguments do not fit in the command buffer. HIPC descriptors carry a
guest address, size, and, where applicable, mapping mode:

| Descriptor | Conceptual use |
| --- | --- |
| send static / pointer | A small region read by the server through its pointer buffer. |
| send buffer | A mapped input region supplied by the client. |
| receive buffer | A mapped output region written by the server. |
| exchange buffer | A region usable in both directions. |
| receive static / pointer | An output region described outside the main descriptor table. |

Mapped buffers carry one of the modes `Normal`, `NonSecure`, `Invalid`, or
`NonDevice`. The binary format splits addresses and sizes across multiple bit
fields; Nixe reconstructs them with checked arithmetic.

A descriptor does not inline the referenced bytes into the command buffer. It
describes memory that the kernel makes available to the operation. For
built-in services, Nixe currently validates the descriptor and reads or writes
the corresponding guest virtual memory through the emulated memory interface.

A simplified file read looks like this:

```text
CMIF raw arguments:
    option = 0
    offset = 0x200
    size   = 0x1000

HIPC receive-buffer descriptor:
    address = 0x80012000
    size    = 0x1000

IFile::Read
    ├── reads min(requested size, descriptor capacity, Nixe limit)
    ├── writes bytes into guest[0x80012000..]
    └── returns the actual byte count in the CMIF response
```

## CMIF: commands and objects over HIPC

CMIF begins at the 16-byte-aligned portion of the HIPC raw-data words. A
request on an ordinary non-domain session contains a 16-byte input header
followed by command-specific arguments:

```text
CMIF request
┌───────────────────────────────────┐
│ magic = "SFCI"                    │
│ version                           │
│ command ID                        │
│ token / inline context            │
├───────────────────────────────────┤
│ command-specific arguments        │
└───────────────────────────────────┘

CMIF response
┌───────────────────────────────────┐
│ magic = "SFCO"                    │
│ version                           │
│ Horizon result                    │
│ token/context value used by Nixe  │
├───────────────────────────────────┤
│ command-specific output values    │
└───────────────────────────────────┘
```

The magic values appear as `0x49434653` and `0x4f434653` when interpreted as
little-endian integers. The context variants carry an inline context value.
Nixe validates the relationship between the command type and CMIF header
version.

Nixe currently mirrors the request token into the final word of its output
header. Public documentation assigns that position an `InterfaceId` on
Horizon 14.0.0 and later. Nixe does not yet select this detail according to
firmware version, so this behavior belongs to the older CMIF profile currently
being emulated.

### CMIF message types recognized by Nixe

| HIPC value | CMIF interpretation |
| ---: | --- |
| 1 | Legacy request |
| 2 | Close |
| 3 | Legacy control |
| 4 | Request |
| 5 | Control |
| 6 | Request with context |
| 7 | Control with context |

A **request** invokes a command on a service object. A **control** acts on the
CMIF session itself rather than on the service interface. Control command 0
converts the current session into a domain, and control command 3 queries the
pointer-buffer size. A close message ends the session.

### Kernel results and service results

One request can produce two distinct results:

```text
svcSendSyncRequest
    │
    ├── X0: KernelResult
    │       Was the handle valid and could the request be transported?
    │
    └── CMIF OutHeader.result
            Did the service accept and execute the command?
```

For example, a nonexistent session handle produces a kernel result. A valid
session receiving an unknown command can complete the SVC successfully and
return `CMIF_UNKNOWN_COMMAND_ID` inside the CMIF response. Combining these
levels would make guest software misinterpret failures.

Horizon result values encode a 9-bit module and a 13-bit description:

```text
raw result = module | (description << 9)
```

Nixe keeps its internal semantic failures separate from guest-visible result
values and translates them at the CMIF boundary. A missing filesystem path
becomes an `fs` result, while a missing add-on may become the verified `lr`
result. CMIF framing and object errors use the `sf` module, and Service Manager
errors use the `sm` module.

## CMIF objects and domains

### Ordinary sessions

Without a domain, a child object normally consumes another process handle:

```text
fsp-srv handle
    │ OpenDataFileSystemByCurrentProcess
    ▼
IFileSystem handle
    │ OpenFile("/data.bin")
    ▼
IFile handle
```

Every live child consumes an entry in the process handle table.

### Domain conversion

CMIF control command 0 converts a session into a domain. Subsequent requests
use one kernel session while addressing multiple objects by ID:

```text
one session handle
        │
        ├── object ID 1: root service object
        ├── object ID 2: IFileSystem
        ├── object ID 3: IFile
        └── object ID 4: IDirectory
```

A domain request adds a domain header before the normal CMIF header. It
contains:

- the domain operation: invoke an object or close an object;
- the number of input objects;
- the CMIF payload size;
- the target object ID;
- token/context and reserved fields;
- a list of input object IDs after the CMIF payload.

A domain response may return child object IDs instead of new handles. This
reduces handle-table pressure while keeping every object under the lifetime of
the parent session.

Closing a domain child removes that ID without closing the session handle.
Object ID 1 is the root and Nixe does not allow it to be closed as a child.

Nixe currently limits each domain table to 64 objects. `IpcSession` retains
generic type-erased child objects, while `AppletSession` has a specialized
table of applet object kinds. That duplication is an implementation detail,
not a permanent architectural boundary, and may change as domain support is
generalized.

## End-to-end example: opening and reading a file

The following path shows how the layers cooperate:

```text
1. Guest/libnx
   ConnectToNamedPort("sm:")
             │
2. SVC 0x1f  │  Nixe creates ServiceManagerSession
             ▼
3. sm:RegisterClient (CMIF 0, sends PID)
             │
4. sm:GetService("fsp-srv") (CMIF 1)
             │  effective NPDM SAC authorizes the service name
             ▼
5. IFileSystemProxy session handle
             │ CMIF 1 SetCurrentProcess
             │ CMIF 2 OpenDataFileSystemByCurrentProcess
             │  NPDM filesystem permissions authorize content reads
             ▼
6. IFileSystem handle or domain object ID
             │ CMIF 8 OpenFile
             │ HIPC send-static/send-buffer points to "/file\0"
             ▼
7. IFile handle or domain object ID
             │ CMIF 0 Read(offset, size)
             │ HIPC receive-buffer identifies the destination
             ▼
8. Nixe reads the authorized read-only RomFS mount
             │ writes the result into guest memory
             ▼
9. HIPC/CMIF response
   KernelResult=Success, CMIF result=Success, bytes_read=N
```

The loader and IPC layers meet through the process mount namespace:

```text
NSP/XCI/NCA processed by the loaders
                 │
                 ▼
LaunchPlan + effective NPDM policy
                 │
                 ▼
ProcessMountNamespace
    ├── effective base/update RomFS
    ├── authorized add-on content
    ├── Service Access Control
    └── filesystem permissions
                 │
                 ▼
fsp-srv and aoc:u service semantics
```

The service does not open arbitrary host paths. It operates on read-only mounts
that the launch pipeline has already resolved and authorized for the process.

## Nixe implementation

### Component overview

```text
crates/cpu
    └── executes SVC instructions and exposes guest register state

crates/runtime
    ├── RunnableProcess and guest virtual memory
    ├── thread TLS
    ├── HandleTable and HandleObject
    ├── PortObject, SessionObject, EventObject, SharedMemoryObject
    └── ProcessMountNamespace and effective policy

crates/horizon
    ├── svc.rs             verified SVC number registry
    ├── svc_dispatch.rs    kernel ABI, waits, sessions, handle transport
    ├── ipc_message.rs     checked HIPC and CMIF codec
    ├── ipc_wire.rs        bridge between wire data, memory, and services
    ├── ipc.rs             typed semantic requests and responses
    ├── ipc_result.rs      guest-visible Horizon results
    └── object.rs          service objects and their state
```

These module boundaries are useful for understanding the current
implementation, but they are not intended to be permanent. In particular,
domain state, built-in service dispatch, and generic kernel session transport
may be reorganized as broader IPC behavior is implemented.

The current split between `ipc_wire.rs` and `ipc.rs` creates this validation
boundary:

```text
untrusted guest bytes
        │
        ▼
checked HipcRequest + CmifRequest
        │
        ▼
bounded typed IpcRequest
        │
        ▼
semantic IpcDispatcher
        │
        ▼
typed IpcResponse
        │
        ▼
checked CmifResponse → guest bytes
```

This allows service behavior to be tested without manually constructing wire
messages, and allows the codec to be tested without mounting real content.

### 1. SVC entry

`HorizonSvcDispatcher` recognizes the main IPC-related operations:

| SVC | Implemented operation |
| ---: | --- |
| `0x1f` | `ConnectToNamedPort` |
| `0x20` | `SendSyncRequestLight` |
| `0x21` | `SendSyncRequest` |
| `0x22` | `SendSyncRequestWithUserBuffer` |
| `0x40` / `0x41` | `CreateSession` / `AcceptSession` |
| `0x42` / `0x43` / `0x44` | `ReplyAndReceive` variants |
| `0x70` / `0x71` / `0x72` | create, manage, and connect to ports |
| `0x16` | `CloseHandle` |

When a handle contains a known built-in service object, the dispatcher invokes
the host-side HIPC/CMIF bridge in `ipc_wire.rs`. When the handle contains a
generic `SessionObject`, the message is delivered to its guest server endpoint.
The two paths coexist:

```text
SendSyncRequest
    │
    ├── built-in service handle
    │       └── immediate host-side dispatch and CMIF response
    │
    └── generic SessionObject handle
            ├── capture objects referenced by sent handles
            ├── enqueue the request
            ├── suspend the client
            ├── let the guest server ReplyAndReceive
            └── materialize response handles in the client
```

The generic path models guest IPC servers and preserves copy/move semantics
between process handle tables. Client requests may copy handles but may not
move them. Server responses may copy or move them, matching the public kernel
behavior used as the implementation reference.

### 2. Defensive wire decoding

For the fixed built-in-service command buffer, `ipc_message.rs`:

- rejects input larger than the expected command buffer;
- checks additions, multiplications, and alignments;
- bounds descriptor and handle counts;
- rejects nonzero reserved fields where implemented;
- ensures every table and payload remains inside the buffer;
- validates CMIF magic, version, message type, and domain framing;
- prevents domain objects from appearing in a non-domain response; and
- prevents an encoded response from exceeding HIPC or TLS limits.

Malformed wire input never reaches the semantic dispatcher. Guest memory
faults, malformed messages, and host resource exhaustion remain separate
`IpcWireError` categories.

### 3. Service registry and authorization

Nixe exposes `sm:` as a built-in named port. After `RegisterClient`,
`GetService` can create sessions for:

| Service | Current coverage |
| --- | --- |
| `fsp-srv` | Open the primary RomFS and read files/directories. |
| `aoc:u` | Count, list, prepare, and observe authorized add-on content. |
| `set:sys` | Return the emulated firmware version for the libnx startup path. |
| `apm` | Normal performance mode and per-mode configuration storage. |
| `appletOE` | A subset of the application applet domain object graph. |
| `hid` | Create `IAppletResource` and return read-only HID shared memory. |

Authorization is applied at two levels:

1. The effective NPDM Service Access Control must permit connecting to the
   named service.
2. Content operations also require filesystem permissions such as
   `ApplicationInfo`, `ContentManager`, or `FullPermission`.

Homebrew processes without an NPDM are currently allowed to access the
platform service registry. This is current policy rather than a permanent
guarantee.

### 4. Semantic objects

The generic `HandleTable` stores type-erased objects with shared identity.
IPC-related handles may contain:

- `ServiceManagerSession` or `IpcSession`;
- `ReadOnlyFileSystem`, `ReadOnlyFile`, or `ReadOnlyDirectory`;
- events and shared memory;
- settings, performance, applet, or HID session objects;
- generic kernel `SessionObject` and `PortObject` endpoints.

Outside a domain, a child is inserted into the process handle table and its
handle is returned. Inside a domain, Nixe removes the temporary handle,
retains the same object in the domain table, and returns an object ID.

### 5. Filesystem and add-on semantics

The semantic layer uses `IpcRequest` and `IpcResponse` rather than raw CMIF
fields. Its current defensive limits include:

| Limit | Current value |
| --- | ---: |
| Semantic path | `0x300` bytes |
| One file read | 1 MiB |
| Entries returned by one list request | 1024 |
| Wire directory-entry record | `0x310` bytes |
| Objects in one domain table | 64 |

Paths must be valid UTF-8, absolute, nonempty, and canonical. Nixe rejects NUL
bytes, empty components, `.` and `..`, a trailing slash, and excessive length.
No host path resolution occurs.

Directory objects retain a shared cursor. Each `ReadDirectory` advances that
cursor. File reads are bounded by the file end, the output descriptor capacity,
and Nixe's per-request limit.

### 6. Response encoding

The wire layer maps typed semantic responses as follows:

| Semantic response | Wire representation |
| --- | --- |
| `None` | Successful CMIF response without output data. |
| `Size` | Little-endian integer in the CMIF payload. |
| `Handle` | Move handle, or object ID when the session is a domain. |
| `Event` | Copy handle. |
| `Data` | Bytes in the receive buffer and byte count in CMIF. |
| `DirectoryEntries` | `0x310`-byte records in the receive buffer. |
| `AddOnContentEntries` | `u32` indices in the receive buffer and count in CMIF. |

If writing a response to guest memory fails after a new handle has been
created, Nixe closes that handle to avoid leaking a process resource.

## Implemented service commands

This is a practical coverage map, not a stable compatibility contract:

| Object | Main command IDs |
| --- | --- |
| `sm:` | 0 `RegisterClient`, 1 `GetService` |
| `IFileSystemProxy` (`fsp-srv`) | 1 `SetCurrentProcess`, 2 `OpenDataFileSystemByCurrentProcess` |
| `IFileSystem` | 8 `OpenFile`, 9 `OpenDirectory` |
| `IFile` | 0 `Read`, 4 `GetSize` |
| `IDirectory` | 0 `Read`, 1 `GetEntryCount` |
| `aoc:u` | 0/2 count, 1/3 list, 6/7 prepare, 8 changed event |
| `set:sys` | 3/4 firmware version |
| root `apm` | 0 open session, 1 get performance mode |
| `apm` child session | 0 set configuration, 1 get configuration |
| `hid` | 0 create applet resource |
| `IAppletResource` | 0 get shared-memory handle |

`appletOE` supports domain conversion, opening `IApplicationProxy`, and a
subset of `ICommonStateGetter`, `ISelfController`, `IWindowController`, and
`IApplicationFunctions`. Other child types may exist in its domain table while
still returning `CMIF_UNKNOWN_COMMAND_ID` for their methods.

## Current limitations and unstable areas

The present coverage is not a complete implementation of Horizon IPC:

- not every Switch service or command is implemented;
- TIPC is not implemented as a general protocol;
- `SendSyncRequestWithUserBuffer` works for generic session transport, but the
  built-in service codec does not accept its full page-sized region;
- the codec expects the receive-static list after raw data and does not yet
  apply every valid `ReceiveListOffset` layout;
- the built-in service bridge does not reproduce every mapping, permission,
  cacheability, and copy rule applied by the real kernel to descriptors;
- several commands accept only the descriptor forms used by the tested libnx
  call paths;
- applet, HID, settings, and performance behavior exposes a coherent minimum,
  not the complete hardware or OS behavior;
- service ABIs are not selected generally by emulated firmware version;
- some limits are Nixe safety limits rather than normative Horizon limits;
- generic and specialized domain tables may be unified later; and
- semantic types and module boundaries may change when writable storage, GPU,
  audio, network, and additional system services are introduced.

When the implementation changes, the command tables, dispatch diagrams,
limits, result behavior, and this section should be reviewed.

## Debugging by layer

The first useful step when debugging IPC is to identify the failing layer:

| Symptom | Likely layer |
| --- | --- |
| The SVC returns `InvalidHandle` | Handle table, endpoint, or session type. |
| A client remains suspended | Session queue, server wait/reply, or peer closure. |
| The SVC succeeds but CMIF reports failure | Command ID, arguments, permissions, or service operation. |
| The request is rejected before dispatch | HIPC header, offsets, CMIF magic/version, or domain framing. |
| The service succeeds but the output buffer is unchanged | Descriptor address, size, mode, or guest memory. |
| A child disappears or handles are exhausted | Copy/move ownership, close behavior, or domain conversion. |
| `sm:GetService` rejects a service name | Service registry or NPDM Service Access Control. |
| `fsp-srv` connects but cannot open content | NPDM filesystem permission or missing mount. |

`ipc_wire.rs` logs the target handle, CMIF message type, command ID, PID
presence, descriptor counts, and handle counts. Those fields usually
distinguish wire framing, routing, and semantic failures quickly.

## References

- [Switchbrew HIPC, fixed revision](https://switchbrew.org/w/index.php?title=HIPC&oldid=14205),
  including the HIPC layout, CMIF framing, descriptors, domains, and TIPC
  distinction.
- [Switchbrew SVC, fixed revision](https://switchbrew.org/w/index.php?title=SVC&oldid=14679),
  documenting the public Horizon supervisor-call ABI.
- [libnx `sf/service.h`, pinned commit](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/service.h),
  used for CMIF service construction, conversion, and lifetime behavior.
- [libnx `sf/cmif.h`, pinned commit](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/cmif.h),
  defining the CMIF helper structures used by the reference client.
- [libnx `sf/hipc.h`, pinned commit](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/sf/hipc.h),
  defining the HIPC helper structures used by the reference client.
- [Atmosphère kernel IPC, pinned commit](https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/source/svc/kern_svc_ipc.cpp),
  used as the public reference for ports, sessions, and IPC SVC behavior.
- [Atmosphère common result encoding, pinned commit](https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libvapours/include/vapours/results/results_common.hpp),
  documenting the Horizon result layout.
