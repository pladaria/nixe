# Task 8: close lifecycle, fault and pressure races

[Specification index](../README.md) · [Task sequence and evidence](README.md) · [Open review items](../review-status.md)

## Entry conditions and required reading

Status: pending; this file is a specification, not an implementation completion record.

Prerequisite: [Task 7](07-reshape.md) must be accepted first.

Read [CONTRIBUTING.md](../../../../CONTRIBUTING.md), the
[architecture and invariants](../architecture.md), the
[task execution contract](README.md), and these task-specific contracts before
editing production code: [Direct memory and fault authority](../memory-authority.md); [Maintenance records, cleanup and mapping requests](../maintenance-records.md); [Cohort workspace, arbitration and mutation plans](../cohort.md); [Shutdown requests, result ownership and wait protocols](../shutdown-and-waits.md); [Request execution, terminal transfer and cohort handoff](../coordinator-execution.md); [Native fault transport and completion](../faults.md); [Epoch reclamation and terminal teardown](../epochs-and-shutdown.md); [Executable cache, metadata and backend ownership](../cache.md). Follow their related-contract links when
the change consumes a referenced protocol.

## Work

Every safety mechanism required by Tasks 2 through 7 is already mandatory at
that task's exit. This task composes, stress-tests and hardens them; it may not
defer a known publication, ownership, fault, unlink or reclamation race from an
earlier vertical slice.

Build deterministic barrier/fault-injection matrices for publication admitted
immediately before Closing, attempted after Closing, a request inserted at the
reopen handoff, simultaneous reshape/invalidation/eviction, a resolver which
requests MappingChange, Shutdown with a suspended retry, old PageFaultTable
readers during removal, configured-vCPU deactivate/reactivate, exact virtual-
address reuse and every fallible preparation boundary. Include a requester
paused after PrequeueProducerToken but before queue visibility, shutdown between
every ActiveBuild state, an LCQ owner plus waiter in Building, worker i winning
the terminal driver while worker j observes Requested, and two external
shutdown callers competing to consume the final JoinHandle. Tests synchronize at the
named linearization points; sleeps alone are forbidden.

Exhaust every fault disposition and MemoryEffectPlan class at every subaccess/
commit stage, including permitted prefix visibility, all-or-nothing failure,
base writeback, barriers, atomics/exclusives and lazy NZCV/FP state. Write each
observed physical page through every CPU alias and every interpreter/DMA/GPU/
loader/debugger/service API. Force ledger, span, dispatch, bridge, registry,
island and metadata pressure plus near-exhaustion checked counters. Exercise
partial worker startup, WorkerControl parking reserve exhaustion and all
shutdown states.

## Acceptance criteria

 every reachable pointer has complete live metadata and a
root/pin; a pre-Closing publisher is included and a post-Closing publisher
changes nothing; the handoff loses no request; mixed payloads remain semantic;
stale jobs cannot modify newer ownership; no promised LCQ is reclaimed. Only
ordinary unobserved tracking RAM restores a captured PC, identical repaired
tuples fail instead of livelocking, and MMIO/SMC have exactly-one effect with
canonical continuation. No observed write is visible before every dependent
root is synchronized/cut. Forced reclaimable pressure returns charged total to
at most 512 MiB without crossing 640 MiB and reuses spans/slots only after grace.
Shutdown leaves no worker, waiter, build/family reservation, active epoch,
UnitPin, PageFaultTable, bridge, executable alias/span or charged arena page.
