# Maxwell Macro Method Expander (MME)

This document explains the Maxwell Macro Method Expander, why Switch 1
graphics command streams use it, how its programs execute, and how Nixe models
it. The MME is a command-generation processor. It is not a shader stage and it
does not normally execute on the host GPU.

The public NVIDIA `MAXWELL_B` class header documents the MME upload and call
methods, but it does not publish the instruction-set semantics. The ISA details
in this document are therefore based on pinned public emulator implementations
listed in [References](#references).

## Position in the graphics pipeline

The MME sits between the decoded pushbuffer and the Maxwell 3D engine:

```text
guest graphics library
        │
        │ builds GPFIFO entries and pushbuffers
        ▼
Maxwell pushbuffer frontend
        │
        ├── ordinary class method ─────────────────────┐
        │                                               │
        └── CALL_MME_MACRO(index, arguments)            │
                         │                              │
                         ▼                              │
                  MME interpreter                       │
                         │                              │
                         │ emits Maxwell class methods  │
                         ▼                              ▼
                  Maxwell 3D dispatcher and typed engine state
                                         │
                                         ├── state changes
                                         ├── clears
                                         └── draws
                                                │
                                                ▼
                                      host graphics backend
```

An ordinary method directly changes engine state or triggers work. An MME call
first runs a small program that may inspect existing state, calculate values,
consume call arguments, and emit more ordinary methods.

## Why the MME exists

Graphics drivers repeatedly produce command sequences with small variations.
Sending every command separately costs pushbuffer space and frontend bandwidth.
An MME program moves that repetitive control work into the GPU command
processor.

Typical uses include:

- expanding one compact draw request into several register writes;
- selecting different method values from the current raster or shader state;
- calculating counts, offsets, masks, or method addresses;
- implementing loops for repeated draws or state updates;
- consuming a variable number of arguments; and
- avoiding duplicated command sequences in every guest pushbuffer.

The result is conceptually similar to a parameterized command-list macro. It
is more capable than textual substitution because it has registers,
arithmetic, conditionals, state reads, and a method-output mechanism.

## The small virtual machine

Nixe models the MME as a tiny 32-bit virtual machine. Its execution context
contains:

| Component                       | Purpose                                                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Eight general-purpose registers | Hold arguments, values read from Maxwell state, and intermediate results. Register zero is hardwired to zero. |
| Program counter                 | Selects the next word in MME instruction RAM.                                                                 |
| Carry flag                      | Supports addition with carry and subtraction with borrow.                                                     |
| Parameter cursor                | Selects the next value supplied by `CALL_MME_MACRO/DATA`.                                                     |
| Method address                  | Selects the Maxwell method written by the next send operation.                                                |
| Method increment                | Advances the method address after each send.                                                                  |
| Delayed branch target           | Models Maxwell branch and exit delay slots.                                                                   |

This is not a general-purpose processor. It has no process, operating system,
stack, virtual address space, or arbitrary instruction fetch from guest RAM.
Its instruction set is narrowly designed to transform parameters and GPU
state into Maxwell method writes.

### Instruction families

The captured Maxwell ISA supports the following families:

- integer add, add-with-carry, subtract, and subtract-with-borrow;
- bitwise XOR, OR, AND, AND-NOT, and NAND;
- add with a signed immediate;
- bitfield extract, insert, and shifted extract;
- reads from the Maxwell 3D register file;
- conditional zero or nonzero branches;
- branch annulment and delay slots;
- parameter fetches;
- method-address selection; and
- method sends.

Every instruction is one 32-bit word. The meaning of some fields depends on
the selected operation and result mode.

## Memory and state domains

It is important not to treat every value used by the MME as ordinary guest
memory. Four separate domains participate:

```text
MME instruction RAM          MME start-address RAM
┌────────────────────┐       ┌────────────────────────┐
│ 32-bit instructions│       │ macro index → first PC │
└─────────┬──────────┘       └────────────┬───────────┘
          │                               │
          └──────────────┬────────────────┘
                         ▼
                  MME execution context
                  ┌─────────────────────┐
call arguments ──►│ GPRs, carry, PC,    │
                  │ method address      │
                  └───────┬───────┬─────┘
                          │ READ  │ SEND
                          ▼       ▼
                  Maxwell register file
```

### Instruction RAM

The guest driver uploads microcode through:

- `LOAD_MME_INSTRUCTION_RAM_POINTER`; and
- `LOAD_MME_INSTRUCTION_RAM`.

The pointer selects an instruction-RAM address. Each data write stores one
word and advances the pointer. This is private MME program storage, not guest
CPU memory or general GPU VRAM.

### Start-address RAM

The guest driver binds macro indices to entry points through:

- `LOAD_MME_START_ADDRESS_RAM_POINTER`; and
- `LOAD_MME_START_ADDRESS_RAM`.

For example, a command stream may establish mappings such as:

```text
macro index 5 ──► instruction address 0x002d
macro index 6 ──► instruction address 0x001c
```

The index used by `CALL_MME_MACRO(5)` therefore selects an entry in this table,
not a byte address supplied directly by the call.

### Call parameters

`CALL_MME_MACRO(j)` begins an invocation and supplies its first parameter.
Additional writes through the corresponding `CALL_MME_DATA(j)` aperture supply
more parameters. The first parameter is initially available in general-purpose
register 1. Fetch result modes consume subsequent parameters in order.

Parameters are transient invocation input. They are neither persistent MME RAM
nor arbitrary guest memory.

### Maxwell register reads

An MME `READ` addresses the Maxwell 3D register file using a method dword
address. It does not directly dereference a guest GPU virtual address. The
program can therefore inspect values such as polygon mode or pipeline-shader
configuration and use them in later decisions.

Nixe retains a source-preserving raw register view alongside typed state for
this purpose. If a macro reads a register that has neither been programmed nor
given a verified reset value, execution stops with a typed host error. Treating
every unknown register as zero would fabricate hardware state.

An emitted method may itself describe a memory operation, resource address, or
draw. Any resulting guest-memory access happens later through that method's
normal validated semantics; it is not an arbitrary memory access performed by
the MME ISA.

## Upload and invocation lifecycle

The normal lifecycle is:

```text
1. Select instruction-RAM address
        │
2. Upload instruction words
        │
3. Select start-address-table index
        │
4. Store the program entry point
        │
5. Submit CALL_MME_MACRO(index, first argument)
        │
6. Supply optional CALL_MME_DATA arguments
        │
7. Execute until an exit instruction and its delay slot complete
        │
8. Dispatch every emitted Maxwell method
```

Programs are normally uploaded during graphics initialization and called many
times afterward. Different indices can refer to different programs, and the
same instruction RAM may contain several adjacent programs.

## How results leave the MME

Intermediate results remain in the eight MME registers. The externally useful
output is normally produced by a result mode that sends a 32-bit value to the
current method address.

The method-address register contains two relevant fields:

```text
┌───────────────────────────────┬────────────────────────────┐
│ next Maxwell method address   │ increment after each send  │
└───────────────────────────────┴────────────────────────────┘
```

After a send, the method address advances by its configured increment. This
lets a small loop fill consecutive or regularly spaced Maxwell registers.

In Nixe, generated methods re-enter the same 3D dispatcher used by ordinary
pushbuffer methods:

```text
MME SEND(method, value)
        │
        ▼
construct source-preserving Maxwell method
        │
        ▼
validate method encoding and semantics
        │
        ├── update typed candidate state
        ├── record raw register value
        └── retain a draw or clear trigger with its exact state snapshot
```

This is essential. Applying MME writes through a second, less strict path
would let macros bypass reserved-bit checks, missing-method errors, resource
validation, or draw-state capture.

## A conceptual example

The following is illustrative pseudocode rather than MME assembly syntax:

```text
r1 = first_call_argument
r2 = READ(SET_FRONT_POLYGON_MODE)
r3 = READ(SET_BACK_POLYGON_MODE)
r4 = r2 OR r3

if (r4 AND LINE_MODE_BIT) != 0:
    SET_METHOD(SET_PIPELINE_SHADER(4))
    SEND(line_pipeline_configuration)
else:
    SET_METHOD(SET_PIPELINE_SHADER(4))
    SEND(fill_pipeline_configuration)

EXIT
execute one final delay-slot instruction
```

The program does not render pixels itself. It chooses and emits the Maxwell
state that a later draw will use.

## Transactional execution in Nixe

MME execution happens during packet preflight against a cloned candidate
state. Generated methods update that candidate in program order. The real
channel state is committed only if all of the following succeed:

- the macro index has a captured entry point;
- every fetched instruction exists;
- every opcode and ALU operation is supported;
- every requested parameter is available;
- every register read has a known value;
- every generated method validates;
- all cross-register invariants remain valid; and
- execution stays within host safety limits.

```text
committed state S0
        │ clone
        ▼
candidate state C0
        │ execute macro
        ├── emitted method A ──► C1
        ├── emitted method B ──► C2
        └── emitted method C ──► error
                                   │
                                   ▼
                         discard C0, C1, and C2
                         committed state remains S0
```

This also preserves packet atomicity when a macro succeeds but a later method
in the same packet fails.

### Bounded execution

A malformed program can loop forever or emit an unbounded number of methods.
Nixe therefore maintains separate limits for retired MME instructions and
generated methods. Exceeding either limit produces a typed host-side coverage
error and rolls back the invocation.

Other typed failures include:

- `CALL_MME_DATA` without a preceding matching call;
- an absent start-address entry;
- an absent instruction word;
- an invalid operation or ALU encoding;
- a branch inside a delay slot;
- program-counter overflow;
- missing or unconsumed parameters;
- a read from unknown register state; and
- a recursively emitted MME call.

These errors describe missing or invalid emulator semantics. They are not
fabricated guest-visible GPU error codes.

## Relationship to shaders and the host GPU

MME programs and shaders are both GPU-associated programs, but they operate at
different layers:

| Program               | Runs conceptually in           | Main input                                 | Main output                       |
| --------------------- | ------------------------------ | ------------------------------------------ | --------------------------------- |
| MME program           | Command frontend               | Call parameters and Maxwell registers      | Maxwell method writes             |
| Vertex shader         | Programmable graphics pipeline | Vertices, attributes, constants            | Clip-space vertices and varyings  |
| Fragment shader       | Programmable graphics pipeline | Interpolated varyings, textures, constants | Fragment colors, depth, and masks |
| Host backend commands | Host graphics API              | Validated emulated state and resources     | Work executed by the host GPU     |

The normal translation path is therefore:

```text
MME microcode
    │ interpreted on the host CPU
    ▼
Maxwell methods and typed Switch GPU state
    │ lowered by the emulator
    ▼
host Vulkan/OpenGL pipeline, resources, draws, and clears
    │
    ▼
host GPU execution
```

Nixe does not need to translate MME instructions into a host GPU shader. The
MME performs command preparation, which naturally belongs in the emulator's
CPU-side frontend. Maxwell shader programs and validated draw operations are
the parts that eventually become host-native GPU work.

## Possible future optimizations

Interpretation is the correctness baseline. Later implementations may cache a
decoded program, compile frequently used macros to host CPU code, or replace a
well-known program with a verified high-level implementation. Such an
optimization must preserve:

- parameter consumption order;
- 32-bit arithmetic and carry behavior;
- register-read visibility;
- branch and delay-slot semantics;
- emitted method order and values;
- exact draw or clear state snapshots;
- typed failure behavior; and
- transactional rollback.

An optimization changes how the host executes a macro, not what Maxwell
methods the guest observes it producing.

## References

- NVIDIA, pinned `MAXWELL_B` class header containing the MME RAM and indexed
  call methods:
  [`clb197.h`](https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L55-L65)
  and
  [`CALL_MME_MACRO/DATA`](https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L4159-L4163).
- yuzu, pinned Maxwell MME instruction fields and enums:
  [`macro.h`](https://source.hodakov.me/hdkv/yuzu/src/commit/8a674958a730a36dbcc43910412521420a804c69/src/video_core/macro/macro.h).
- yuzu, pinned low-level interpreter behavior:
  [`macro_interpreter.cpp`](https://source.hodakov.me/hdkv/yuzu/src/commit/8a674958a730a36dbcc43910412521420a804c69/src/video_core/macro/macro_interpreter.cpp).
- Ryujinx, independently implemented pinned MME interpreter:
  [`MacroInterpreter.cs`](https://git.axenov.dev/Museum/ryujinx/src/commit/ec3e848d7998038ce22c41acdbf81032bf47991f/Ryujinx.Graphics.Gpu/Engine/MME/MacroInterpreter.cs).
