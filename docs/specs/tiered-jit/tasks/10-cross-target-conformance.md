# Task 10: prove cross-target conformance

[Specification index](../README.md) · [Task sequence and evidence](README.md) · [Open review items](../review-status.md)

## Entry conditions and required reading

Status: pending; this file is a specification, not an implementation completion record.

Prerequisite: [Task 9](09-legacy-removal.md) must be accepted first.

Read [CONTRIBUTING.md](../../../../CONTRIBUTING.md), the
[architecture and invariants](../architecture.md), the
[task execution contract](README.md), and these task-specific contracts before
editing production code: [Supported baseline and generated manifest](../baseline.md); [Gateway, state transfer and helper ABI](../native-abi.md); [Static links, indirect dispatch, returns and target protection](../linking.md); [Native fault transport and completion](../faults.md); [External performance acceptance](../performance.md); [Final conformance gate](../conformance.md). Follow their related-contract links when
the change consumes a referenced protocol.

## Work

Run focused differential, concurrency and native-shape tests throughout the
work, then execute the complete final matrix on native Linux x86-64 and native
Linux AArch64 hosts with 4096-byte pages. Cross-compilation, emulation and
encoder byte tests are supplemental and do not substitute for either native
run.

The matrix covers:

- cold LCQ and hot HCQ semantics for every [Task 0](00-baseline.md) accepted `native` or `helper`
  CoverageId/`A64Instruction` variant which is legal inside HCQ, differentially
  against the independent interpreter and its architectural vectors; every
  accepted `boundary` variant instead executes through LCQ/canonical handling
  and proves that HCQ terminates immediately before it;
- every selected entry and direct/indirect/return link kind;
- zero/small/large block budgets, loops and forced control requests;
- FPCR/FPSR, lazy NZCV and helper success/failure boundaries;
- all fault dispositions/MemoryEffectPlan stages and identical-PC retry only for
  ordinary tracked RAM;
- concurrent compilation, publication, reshape, invalidation and shutdown;
- W^X, BTI, CET/IBT, epoch retirement, segment reuse and cache pressure;
- self-modifying code and mapping changes during every compilation stage; and
- the versioned external-protocol scripts for `hello-world`, `es2gears`,
  `textured_cube` and every frozen commercial-workload startup and sustained
  case, each checked against its recorded correctness-marker hash.

Native AArch64 additionally patches/unpatches while alternating the executor
across physical cores; both hosts repeatedly reuse the same RX virtual address
and must resolve a synthetic native fault to the current segment generation and
CodeVersion. A checked-in disassembly verifier consumes production binaries and
backend metadata and validates every named gateway, entry, static bridge,
dynamic bridge, helper, poll, fault and canonical-exit shape on both hosts.
External profiles and the existing private allocation-enforcement ledger remain
diagnostic inputs; no production benchmark/telemetry path is added.

## Acceptance criteria

 both native host targets pass the same architectural suite;
every scripted smoke/workload case produces its frozen marker, the disassembly
verifier accepts every enumerated production shape, repeated invalidation and
reclaimable pressure return charged storage to at most the soft watermark, and
the external comparison protocol reaches every stated threshold. No result is
inferred from another ISA, a vsync-limited frame rate, manual inspection or
obsolete tests.
