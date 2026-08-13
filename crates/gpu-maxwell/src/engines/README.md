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
- update a candidate state atomically; and
- emit typed, host-independent operations for execution when required.

Engine handlers must not access Horizon objects, guest mappings, scheduler
ownership, or host graphics APIs directly. Address resolution and execution
belong to the submission execution layer; backend-specific work belongs behind
the neutral GPU contracts. Unsupported semantics must remain explicit typed
errors rather than fabricated success.
