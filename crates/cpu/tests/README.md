# CPU frontend test provenance

The architectural source of truth is Arm's public documentation:

- [Arm Architecture Reference Manual for A-profile architecture (DDI 0487)](https://developer.arm.com/documentation/ddi0487/latest)
- [A64 Instruction Set Architecture (DDI 0602)](https://developer.arm.com/documentation/ddi0602/latest)
Raw-encoding cases trace the `A64` encoding index in those
documents. Semantic tests trace the pseudocode for the named instruction and
the shared architectural pseudocode it invokes. The links are evidence and a
review aid; they are not executable assertions. The tests validate our decoder,
normalized operands and reference semantics with independently written
expectations.

Use stable document identifiers in comments and reviews. A dated document
revision may additionally be recorded when behavior differs between revisions,
but tests should not depend on network access or scrape Arm's website.

The interpreter/IR differential suite and optional QEMU oracle are owned by
`nixe-cpu-interpreter`. Run them with:

```bash
cargo test-diff
```

Robustness fuzzing is documented in [`fuzz/README.md`](../../../fuzz/README.md).

## Region diagnostics

`translate_raw_region_report` accepts raw little-endian bytes, a base guest PC,
an execution state, and a CPU profile. It runs the ordinary decoder, lifter, IR
builder, region former, and verifier against a bounded synthetic instruction
view, then emits a deterministic report containing source disassembly,
pre-optimization IR, basic-block cut reasons, region entries and exits, and
every physical code-page generation.

`translate_region_report` provides the same opt-in report path for an existing
process-memory implementation. Ordinary `translate_region` calls do not compute
disassembly strings or print IR.
