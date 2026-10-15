# Maxwell engine boundaries

In this crate, an **engine** is a Maxwell command class with its own method
namespace and persistent channel state. It does not need to be a large
programmable unit such as the 3D or compute engine. Small, specialized classes
such as `MAXWELL_INLINE_TO_MEMORY_A` are engines at the same architectural
layer because they receive class methods directly from the pushbuffer and own
state that is independent from the other classes.

## Class and method identity

A numeric method offset is not globally unique. Its meaning is determined by
the pair:

```text
(bound class, method offset)
```

For example, method `0x0188` in `MAXWELL_INLINE_TO_MEMORY_A` programs the upper
part of an output address. The same offset in another class belongs to that
class's method namespace and must not be routed here merely because the
operation looks similar.

`SetObject` establishes or verifies the class bound to a subchannel. Later
pushbuffer methods are decoded against that binding and dispatched to the
corresponding engine handler. The Switch 1 GM20B profile currently validates
this fixed graphics-subchannel layout:

```text
subchannel 0 -> MAXWELL_B (3D)
subchannel 1 -> MAXWELL_COMPUTE_B
subchannel 2 -> MAXWELL_INLINE_TO_MEMORY_A
subchannel 3 -> FERMI_TWOD_A
subchannel 4 -> MAXWELL_DMA_COPY_A
```

Subchannels identify routing slots; classes define semantics. Engine state is
therefore stored separately even when two classes expose similar facilities.
For instance, compute and inline-to-memory both support inline uploads, but
their registers, launch rules, and cursors belong to different classes. They
may share a verified neutral memory-write abstraction without sharing frontend
state or method decoding.

## Frontend and execution responsibilities

An engine handler should:

- declare verified methods and field masks;
- validate method sequences and cross-register requirements;
- preserve the exact method source for diagnostics;
- apply validated methods to live channel state in submission order; and
- emit typed, host-independent operations for execution when required.

Engine handlers must not access Horizon objects, guest mappings, scheduler
ownership, or host graphics APIs directly. Address resolution and execution
belong to the submission execution layer; backend-specific work belongs behind
the neutral GPU contracts. Unsupported semantics must remain explicit typed
errors rather than fabricated success.

The retained pushbuffer iterator validates submission identity once before its
first read. Each entry read still validates its own retained mappings. A monotonic
segment cursor preserves exact source locations across fragmented entries without
searching their preceding segments again for every word.

MME instruction uploads retain dense predecoded execution pages separately from
raw words and their diagnostic provenance. Replacing a word updates both; active
program snapshots keep their original instructions through copy-on-write. Calls
still execute parameters, register reads, shadow behavior and emitted methods in
order. Cached code does not cache a macro's state-dependent effects.

The live raw and MME shadow register files keep valid words in dense arrays and
write provenance in separate cold storage. Reads, replay and change detection
consume only the words; diagnostic inspection reconstructs a source-preserving
register on demand. Method replay updates the argument in place, and preparation
returns only effects/metadata instead of copying the complete dispatch back to
its caller. Rewriting the same value still updates its exact write provenance.

Native preparation shares immutable binding-role and consumed-role arrays across
draws. Program binding comparison allocates a replacement only at the first
physical binding change, retaining the previous program when all bindings match.
Canonical access collection appends directly into one vector, without allocating
an intermediate vector for each resource. After successful backend acceptance,
the owner recycles the native command vector; acceptance has already retained
native uses through GPU completion. Pending preparation keeps its segment intact,
and ordered canonical writes and completion boundaries are unchanged.

Native shader preparation derives consumed constant-buffer extents from live
optimized SSA. Static loads bound the byte window, while dynamic addresses retain
the complete declared range. Stage requirements and texture-handle dependencies
are unioned before resolving resources. Resource resolution builds a mask of
required constant slots and visits only those slots in hardware order, rather
than searching every slot against every required role.
