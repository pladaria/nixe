# Task 4: complete native chaining and safe cutover

[Specification index](../README.md) · [Task sequence and evidence](README.md) · [Open review items](../review-status.md)

## Entry conditions and required reading

Status: pending; this file is a specification, not an implementation completion record.

Prerequisite: [Task 3](03-lcq-cutover.md) must be accepted first.

Read [CONTRIBUTING.md](../../../../CONTRIBUTING.md), the
[architecture and invariants](../architecture.md), the
[task execution contract](README.md), and these task-specific contracts before
editing production code: [Static links, indirect dispatch, returns and target protection](../linking.md); [Gateway, state transfer and helper ABI](../native-abi.md); [Compilation and publication pipeline](../publication.md); [Cohort workspace, arbitration and mutation plans](../cohort.md); [Epoch reclamation and terminal teardown](../epochs-and-shutdown.md). Follow their related-contract links when
the change consumes a referenced protocol.

## Work

Implement static bridges/backlinks, embedded pending patch records, the
maintenance records and exact Fallback/InstallPending/Installed/UnlinkPending
machine/root ordering. Add static island preflight, cache synchronization and
source/target generational validation. Then add the source-keyed per-vCPU PIC,
weak BridgeKey index, bounded registry/UnitPins, absolute BridgeUnits and the
per-thread 16-entry BlockKey-only RSB. A miss is canonical/cold; capacity loss
leaves the chosen PIC way/fallback unchanged.
At the same time extend terminal cleanup with PatchRecord,
AwaitingLinkCleanup, static/dynamic bridge, PIC and RSB-root handling before any
of those roots is enabled.

Extend [Task 3](03-lcq-cutover.md) invalidation, eviction, mapping change and shutdown to every new
root before enabling a link kind. No old target/bridge span or metadata may be
retired until its patch is globally restored and every matching PIC/suspended
retry root is cleared. Enable BTI/CET landing kinds only with their native
feature checks.

## Acceptance criteria

 after the required block-budget sub/test, a compatible
in-range empty static bridge is exactly guest branch decision plus one host
branch, with no dispatch lookup, Rust call, PC store, A64State write or full
state round trip. Nonempty/far/helper shapes match their specified bridge/
island forms. PIC and matched-RSB hits compare the full ExitSiteKey/target,
perform no A64State write or epoch exit, and never reuse another source's
bridge; a named CanonicalClean live-in may cause only its declared load.
The PIC-miss resolver test deliberately destroys every system-ABI
caller-clobbered register, proves that resolving transfer reaches target
canonical ingress rather than the new bridge, and then proves that a later
native hit uses that bridge with the original source physical contract.
Deterministic tests stop between provisional-root creation, patch store, local
cache clear, cross-core flush and final root-state change: install never exposes
an unpinned target and unlink never releases one before fallback visibility.
All active/retired/registry/ledger counts stay within declared bounds under
collision and rapid replacement. Native AArch64 cross-core patch/unpatch and
x86 tests prove an old target cannot execute after root removal, and every
indirect target passes enabled BTI/CET/IBT. Both production empty and nonempty
bridge paths subsume the [Task 2](02-lifetime-foundation.md) ABI proof on both native hosts; the test-prewired
bridge route and only its obsolete scaffolding are then removed.
