# Task 9: remove the superseded architecture

[Specification index](../README.md) · [Task sequence and evidence](README.md) · [Open review items](../review-status.md)

## Entry conditions and required reading

Status: pending; this file is a specification, not an implementation completion record.

Prerequisite: [Task 8](08-lifecycle-stress.md) must be accepted first.

Read [CONTRIBUTING.md](../../../../CONTRIBUTING.md), the
[architecture and invariants](../architecture.md), the
[task execution contract](README.md), and these task-specific contracts before
editing production code: [Migration and removal map](../migration.md); [Final conformance gate](../conformance.md). Follow their related-contract links when
the change consumes a referenced protocol.

## Work

Delete every item in the migration map and all tests that exist only for those
items. Collapse abstractions which have one remaining implementation. Retain
tests only when they assert the new guest-visible, concurrency, ABI, allocator
or lifecycle contract.

## Acceptance criteria

 repository inspection finds one decoder/semantics path, one
native ABI, one executable cache, one dispatch identity, one tiering policy and
no callable legacy route. Production contains no implementation-detail
measurement fields or test-driven compatibility branches.
