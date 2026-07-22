# Nintendo Switch 1 Runtime Linker (`rtld`)

This document explains the role of the runtime linker used by packaged
Nintendo Switch 1 applications, how it cooperates with the system loader, and
what an emulator must reproduce when executing it as guest software.

The behavior described here is based on public community reverse engineering.
`rtld` is part of each title's guest software, not a component supplied by the
emulator.

## The short version

A packaged title commonly contains several NSO executable modules:

```text
Program NCA
└── ExeFS (PFS0)
    ├── rtld        ─ runtime linker and initial process entry point
    ├── main        ─ application code
    ├── subsdk0     ─ optional library module
    ├── sdk         ─ Nintendo SDK module
    └── main.npdm   ─ process metadata
```

The system loader maps these NSOs into the new process, but does not perform
the complete dynamic link itself. Execution begins in the guest `rtld`, which
discovers the mapped modules, relocates them, resolves symbols between them,
initializes the runtime, and eventually calls the application.

```text
Package and NCA loader
        │
        │ load, decompress, validate, place, and map NSOs
        ▼
Guest virtual memory
        │
        │ start the main thread at the rtld base address
        ▼
Guest rtld
        │
        ├── relocate itself
        ├── discover main, subsdk*, and sdk
        ├── relocate those modules
        ├── resolve symbols between modules
        ├── initialize the SDK and module constructors
        └── call the application
```

Loading and linking are therefore separate operations:

| Component | Primary responsibility |
| --- | --- |
| System or emulator loader | Make validated NSO images and process resources available to the guest. |
| Guest `rtld` | Turn the mapped but unlinked modules into one runnable program. |

## `rtld` is guest code

The `rtld` file in ExeFS is an NSO belonging to the title. It contains ordinary
AArch64 instructions which execute on the emulated CPU. Calls such as
`svcQueryMemory`, `svcBreak`, and `svcExitProcess` cross from that guest code
into the emulated Horizon kernel interface.

```text
rtld NSO bytes
      │
      ▼
guest AArch64 instructions
      │
      ▼
emulated CPU
      │ SVC instruction
      ▼
emulated Horizon service-call dispatcher
```

An emulator may also contain a host-side ELF linker for inspection, tests, or
an alternative loading strategy. That host linker is not the title's `rtld`,
even when both pieces of code understand the same dynamic-symbol and
relocation formats.

## What the loader must do first

Before `rtld` can execute, the system loader must construct a valid process.
For an emulator, this normally consists of the following steps.

### 1. Resolve the effective title content

The loader selects the effective Program content from the base application and
any installed update. It opens the Program NCA and locates its ExeFS section.
This package-level work is unrelated to dynamic linking.

### 2. Load the NSO images

For each executable entry, the loader:

- validates the NSO header and segment ranges;
- decompresses LZ4 or other supported segment encodings;
- materializes the initialized `.text`, `.rodata`, and `.data` bytes; and
- allocates zero-initialized memory for `.bss`.

NSO decompression is not relocation. Decompressing a segment reconstructs the
bytes produced by the static linker; relocation later adjusts values inside
those bytes for their actual guest addresses.

### 3. Select guest addresses

Each module receives an ASLR-selected, page-aligned base address. Its segments
are mapped with appropriate permissions and Horizon memory-state metadata.

```text
Example guest layout

0x71000000  rtld.text      R-X  CodeStatic
    ...     rtld.rodata    R--  CodeStatic
    ...     rtld.data/bss  RW-  CodeMutable

0x71014000  main.text      R-X  CodeStatic
    ...     main.rodata    R--  CodeStatic
    ...     main.data/bss  RW-  CodeMutable

0x75000000  sdk.text       R-X  CodeStatic
    ...     sdk.rodata     R--  CodeStatic
    ...     sdk.data/bss   RW-  CodeMutable
```

The precise addresses are not important. Their mapping boundaries,
permissions, memory states, and query results are important because `rtld`
uses `svcQueryMemory` to discover modules.

#### Guest addresses are the shared coordinate system

The addresses selected by the emulator are the same addresses that `rtld`
uses when it calculates and writes relocations. They are **guest virtual
addresses**, not native pointers in the emulator's host process.

For example, Nixe might choose this process layout:

```text
Guest virtual address space

0x71000000 ─ rtld
0x71010000 ─ main
0x73000000 ─ sdk
```

Those values describe the emulated Switch process. Internally, the emulator
may store the corresponding bytes in Rust allocations at unrelated host
addresses:

```text
Guest address               Emulator-internal representation

0x71013200 ────────────────► guest page-table lookup
                                      │
                                      ▼
                             emulated physical page
                                      │
                                      ▼
                             Rust-owned byte buffer
                             at an unrelated host address
```

The guest never observes the Rust pointer or the emulator's physical-page
identifier. Every guest instruction supplies a guest virtual address, and the
emulated memory subsystem translates it to the appropriate backing storage.

This creates one consistent coordinate system:

| Value | Visible to `rtld`? | Meaning |
| --- | --- | --- |
| `0x71013200` | Yes | Virtual address inside the emulated Switch process. |
| An emulator physical-page ID | No | Internal identity used to implement mappings and aliases. |
| A native Rust pointer such as `0x7f...` | No | Host address of an implementation-owned allocation. |

`rtld` learns module bases from that same guest address space. It derives its
own base with position-independent instructions and discovers other module
mappings through `svcQueryMemory`. The emulator answers the SVC from the
memory map it created earlier, so the result contains exactly the guest bases
that the emulator selected:

```text
ProcessBuilder selects 0x71010000 for main
                 │
                 ├── maps main at guest address 0x71010000
                 │
                 └── later answers QueryMemory
                                  │
                                  ▼
                         base = 0x71010000
                                  │
                                  ▼
                         rtld uses that base
```

A relative relocation therefore writes valid guest addresses without knowing
anything about the emulator's host memory. Suppose `main` is based at
`0x71010000` and one relocation contains:

```text
target offset = 0x3200
addend        = 0x9000
```

Conceptually, `rtld` performs:

```text
target guest address = 0x71010000 + 0x3200 = 0x71013200
relocated value      = 0x71010000 + 0x9000 = 0x71019000

guest_memory[0x71013200] = 0x71019000
```

The emulated `STR` instruction sends the target address `0x71013200` to the
memory subsystem. The emulator resolves that address through its guest page
tables and changes the Rust-owned backing bytes. A later guest `LDR` follows
the same translation and reads back `0x71019000` as a Switch virtual pointer.

This relationship is why memory layout and `QueryMemory` must agree. If the
emulator maps `main` at one guest base but reports another base to `rtld`, the
linker will create pointers that do not address the mapped module.

### 4. Leave dynamic relocations pending

The loader must not apply the same relocations that the guest `rtld` is about
to apply. It may parse and validate the dynamic tables, but relocation targets
must retain their pre-runtime values.

For example, an `R_AARCH64_RELATIVE` relocation conceptually requests:

```text
result = module_base + addend
```

Before `rtld` runs:

```text
module base              = 0x71014000
stored addend            = 0x00001234
final relocated pointer  = not written yet
```

When `rtld` processes the relocation:

```text
0x71014000 + 0x1234 = 0x71015234
```

Applying the relocation in the host and then allowing `rtld` to process the
same target can add the base twice or otherwise make the guest interpret a
final pointer as an unprocessed addend.

## The NSO module header (`MOD0`)

The metadata needed by `rtld` is largely embedded in each module. An NSO's
text image contains a locator at image offset `+0x04` which identifies its
`MOD0` header. The header contains image-relative references to data such as:

- the ELF-style `.dynamic` table;
- the beginning and end of `.bss`;
- exception-frame metadata; and
- storage reserved for a runtime-generated module object.

Conceptually:

```text
NSO image
├── .text
│   ├── entry/bootstrap code
│   ├── MOD0 locator at +0x04
│   └── MOD0
│       ├── dynamic-table offset
│       ├── BSS range
│       ├── exception-frame range
│       └── module-object storage offset
├── .rodata
│   ├── .dynstr
│   ├── .dynsym
│   └── hash and relocation metadata
└── .data / .bss
    ├── relocation targets and GOT
    └── runtime-generated ModuleObject storage
```

The loader validates these ranges and ensures that their backing pages exist.
It does not need to create the final `ModuleObject`. The guest `rtld`
initializes that object and links it into its own module lists.

## Initial process state

Execution begins at the first instruction of the `rtld` text mapping, not at
the first byte of `main`. For a normal packaged AArch64 process, the important
initial state includes:

```text
PC        = rtld image base
X0        = 0
X1        = main-thread handle
SP        = top of the main-thread stack
TPIDR_EL0 = main-thread TLS address
```

The loader must also create the process address space, stack, thread-local
storage, handle table, and the kernel-visible memory descriptions needed by
the SVC interface.

`main` is not a substitute initial entry point. Its image offset zero may be
module bootstrap metadata rather than an application entry function, and its
imports, GOT, runtime state, and constructors are not ready until `rtld`
finishes.

## Runtime-linking sequence

Once the main thread starts, `rtld` performs several ordered phases.

### Phase 1: bootstrap and self-relocation

`rtld` begins in a state where its own absolute addresses are not yet
available. Its bootstrap code derives its image base from position-independent
control flow, locates its `MOD0`, and parses its `.dynamic` table.

It first applies the relative relocations required to make its own data
structures usable. Only after this bootstrap can it safely use ordinary
absolute pointers.

```text
position-independent bootstrap
        │
        ├── determine rtld base
        ├── locate MOD0 and .dynamic
        ├── find relative-relocation tables
        └── apply rtld's relative relocations
```

### Phase 2: initialize the `rtld` module object

`rtld` initializes load lists and creates a `ModuleObject` describing itself.
This object formalizes the relationship between its base address, dynamic
metadata, symbols, relocations, and neighboring modules.

### Phase 3: discover the other NSOs

Rather than receiving a completed host-side module list, `rtld` scans guest
memory with `svcQueryMemory`. Candidate regions must have the expected
executable permissions and static-code memory state.

For each candidate, it:

1. identifies the module through its `MOD0` locator and magic;
2. reads the dynamic-table and BSS information;
3. clears or initializes the module's BSS as required;
4. constructs its runtime `ModuleObject`;
5. applies its relative relocations; and
6. adds the module to the runtime load lists.

This is why an emulator's `QueryMemory` result is part of the loader ABI. A
module can be present in memory but remain invisible to `rtld` if its memory
state, permissions, base, or extent are reported incorrectly.

### Phase 4: resolve symbols across modules

After all modules have usable relative pointers, `rtld` resolves imports and
exports across the complete load scope. It uses ELF-style dynamic metadata,
including:

- `.dynsym` symbol records;
- `.dynstr` symbol names;
- hash tables used for lookup;
- global and weak bindings;
- symbol visibility; and
- REL or RELA relocation tables.

Common AArch64 relocation kinds include:

| Relocation | Purpose |
| --- | --- |
| `R_AARCH64_RELATIVE` | Form a pointer from the module base and an addend. |
| `R_AARCH64_ABS64` | Write a 64-bit resolved symbol value plus an addend. |
| `R_AARCH64_GLOB_DAT` | Populate a global-data or GOT entry. |
| `R_AARCH64_JUMP_SLOT` | Populate a callable import slot. |

An import in `main` may therefore resolve to a definition in `sdk` or a
`subsdk` module. Scope order, binding, and visibility determine which matching
definition wins.

### Phase 5: initialize and enter the application

Once linking is complete, `rtld` becomes the process startup runtime. It
initializes the standard runtime and SDK, invokes module initialization
routines, and transfers control to the application-defined entry routine.

At process shutdown it coordinates finalization and ultimately requests
process termination from Horizon.

```text
linked modules
      │
      ├── runtime initialization
      ├── SDK initialization
      ├── module constructors
      ├── application entry/main
      ├── module destructors and SDK finalization
      └── svcExitProcess
```

## Horizon behavior that an emulator must reproduce

The emulator does not need to replace `rtld`; it needs to provide the
environment in which the guest `rtld` behaves as it would on the console.

The minimum relevant contract includes:

- correct NSO placement and segment contents;
- pending, not pre-applied, guest relocations;
- zero-backed BSS and writable relocation targets;
- accurate code-static and code-mutable memory states;
- accurate `svcQueryMemory` boundaries, permissions, and states;
- the expected initial register, stack, TLS, and thread-handle state;
- SVC implementations used during startup; and
- memory-permission transitions requested by the runtime.

These pieces are interdependent. Correct executable bytes are insufficient if
`QueryMemory` hides a module, and a correct memory map is insufficient if the
host has already modified values that `rtld` expects to relocate itself.

## Host linking and guest linking are alternative models

An emulator can theoretically choose either of two architectures.

### Host-linked model

```text
host loader maps modules
      └── host linker applies all relocations and resolves all symbols
              └── emulator bypasses guest rtld and enters a known final entry
```

This can be useful for analysis or specialized runtimes, but it must reproduce
all startup work normally performed by `rtld`, including initialization order
and ABI details. Merely jumping to NSO offset zero is not sufficient.

### Guest-linked model

```text
host loader maps unrelocated modules
      └── guest rtld performs the real dynamic link
              └── guest rtld enters the application normally
```

This model executes the title's intended startup path and avoids duplicating
version-specific linker behavior in the host. It requires a more faithful
Horizon memory and process environment.

The two models must not be combined accidentally:

```text
host applies relocations
      +
guest rtld applies relocations again
      =
double relocation, corrupted pointers, or rtld invariant failure
```

## Implications for Nixe

Nixe retains host-side executable preparation capable of applying NSO
relocations and resolving symbols across a batch of modules. That machinery is
useful for validation and tests, but the packaged-title path uses a distinct
guest-relocation mode.

That path:

1. prepares each NSO with `prepare_for_guest_relocation`;
2. parses and validates MOD0, dynamic tables, symbols, and relocations without
   committing relocation writes;
3. maps every NSO at its selected guest base with the required memory metadata;
4. preserves writable data, BSS, and module-object storage for `rtld`;
5. starts the main thread at the `rtld` base with the packaged-process ABI;
6. exposes those same mappings through the emulated `QueryMemory` SVC; and
7. lets guest `rtld` perform relocation, symbol resolution, and initialization.

A fatal `svcBreak` during startup should still be treated as an execution
failure even when its numeric reason is zero.

The host batch linker can remain available for loader tests, executable
inspection, and any future explicitly host-linked mode. It should not mutate
the same launch image that will subsequently be linked by the guest `rtld`.

## Diagnosing early `rtld` failures

An early `svcBreak` normally means that `rtld` reached an invariant it considers
fatal. Useful diagnostics include:

- the PC and disassembly around the break call;
- the `Break` reason, information pointer, and size from `X0`-`X2`;
- the queried memory region immediately preceding the failure;
- the module and MOD0 being processed;
- the current dynamic tag or relocation record; and
- whether the affected bytes were already modified by host relocation.

A break payload of `{ reason: 0, info: 0, size: 0 }` is still a fatal guest
break. A zero reason must not automatically be interpreted as successful
process completion.

## References

Public information about retail `rtld` behavior is based primarily on
community reverse engineering rather than a complete official Nintendo
specification.

- [Switchbrew: rtld](https://switchbrew.org/wiki/Rtld)
- [Switchbrew: MOD](https://switchbrew.org/wiki/MOD)
- [Switchbrew: NSO0](https://switchbrew.org/wiki/NSO0)
- [Switchbrew: SVC](https://switchbrew.org/wiki/SVC)
- [ARM: ELF for the Arm 64-bit Architecture](https://github.com/ARM-software/abi-aa/blob/main/aaelf64/aaelf64.rst)
