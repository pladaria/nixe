# Semantic shader SSA

`nixe_gpu::ssa` is the native compiler's representation. It contains typed values,
blocks, phis, instructions and terminators, rather than a guest register file or a
shader-language interpreter. Block/value IDs are dense `u32` identities; resource identities also use `u32`;
neither guest register counts nor predicate slots appear in the shared passes.
Scalar reinterpretation is an explicit bitcast. Memory addresses are explicit byte
offsets with a declared `U32` or `U64` type, not an implicit guest word width or
alignment rule. Host feature legalization belongs to SPIR-V generation.

`FunctionBuilder` accepts temporary mutable variables with an explicit initial
value: undefined, a bit-exact constant or a parameter. Its prologue has no incoming
edges. The frontend selects hardware initialization rules; the builder never
invents reset values. Construction removes unreachable blocks, computes immediate
dominators/frontiers, inserts live-in phis and renames variables along the dominator
tree. All graph walks use explicit stacks. There is one SSA construction algorithm,
including for frontend predication and loops.

Loads, stores, outputs and barriers thread an effect value. Effect phis join actual
predecessor effects, and every terminator retains its final effect, including early
return/discard. Texture samples and output reads also participate in this chain.
Unused nonvolatile reads can be deleted by splicing their effect tokens; live
reads cannot move across writes. Tokens constrain compiler transformations; they do not request a
Vulkan barrier after every instruction. Execution/memory barrier scopes are explicit
semantic operations with explicit memory classes. They lower to the required
SPIR-V scopes and semantics, without a blanket fence.

Optimization propagates copies/trivial phis, folds exact integer/boolean/bitcast
constants and selects, performs local pure-expression CSE, removes dead values,
folds constant branches/switches, prunes edges and fuses linear blocks. Values and
definition metadata are compacted after transformations. Undefined choices and
ordered operations are excluded from CSE. Floating arithmetic is not folded using
CPU arithmetic, reassociated or contracted; floating controls and exact constant
bits, including signed zero, survive unchanged.

`Function::verify` checks unique dense definitions, opcode types/arity, dominance,
phi predecessor coverage, switch cases and the entire effect chain. Construction
and optimization invoke it in debug builds. Release compilation has no automatic
full verifier, and execution has no per-draw check or register interpreter.
`Function::dump` retains block and instruction PCs for diagnostics.

## Maxwell frontend boundary

`nixe_gpu_maxwell::compile_maxwell_shader_ssa` reads a bounded canonical shader
snapshot and returns a `MaxwellSsaProgram`: semantic SSA, live guest binding facts,
source locations and captured global-buffer extent expressions. The pipeline
entry point captures all stages, links vertex A/B and adjacent interfaces, and
retains canonical source identities/invalidation inputs for the native cache.

Checked ISA decoding, register/predicate bookkeeping, predication, scheduling
bundles, reconvergence and multi-result input snapshots belong to Maxwell. Its
frontend construction protocol is owned by `gpu-maxwell`; all 56 decoded semantic
operation variants lower directly to SSA. The new path never consumes `ShaderIr`,
its verifier/interpreter, WGSL or an old-IR importer. Temporary and resource IDs
are `u32`; the oracle's 16-bit temporary and 8-bit resource limits apply only when
exporting to that retiring renderer. Hardware field widths remain ISA facts.

Vertex A/B uses one SSA function: A returns enter B, and B attribute loads consume
A's written values without an intermediate host buffer. Separate source-PC ranges
retain A/B diagnostics. Adjacent linkage preserves scalar types, components and
interpolation qualifiers; disabled Maxwell generic fragment inputs receive the
specified `(0,0,0,1)` components. The oracle's synthetic R0–R3 input preload is
confined to its export adapter; native register inputs follow explicit guest
attribute loads and undefined live-ins.

## Native SPIR-V

`ssa::spirv::emit` emits SPIR-V directly using `rspirv`. `emit_pipeline` assigns a
shared, dense descriptor ABI to live logical resource identities already
namespaced by the guest frontend. Returned modules contain 32-bit binary words,
read/write binding reflection and the required push-constant range. This is cold
compiler work on cache misses, with no guest-model/profile dispatch per draw.

The Vulkan control-flow pass splits irreducible entries and overlapping regions,
funnels multiple loop exits, and reconstructs SSA with the same `FunctionBuilder`.
Emission uses native phi instructions, selections, loops and early exits, with
explicit continue blocks for multiple backedges. No GPU program-counter dispatcher
or mutable guest register array is generated. Structured-control requirements:
[SPIR-V specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_structured_control_flow).

Numerical helpers are shared independently of the old register emitter. They
preserve exact constant bits, rounding, signed zero, saturation and declared
DAZ/FTZ behavior. Integer lowering implements exact half packing/unpacking,
integral rounding and conversions; binary16 FMA corrects double rounding.
Float32 FMA requires the correctly rounded `OpFmaKHR`, rather than relying on
GLSL's optionally unfused FMA. Host capabilities are explicit compilation inputs:
[VK_KHR_shader_fma](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_shader_fma.html).

| Area | Implemented | Explicitly unsupported | Host-dependent |
| --- | --- | --- | --- |
| Stages/interfaces | Vertex A/B, fragment, compute, guest TCS/TES; dynamic default TCS; precise scalar/component and patch/control-point IO | Geometry emission and stage/opcode forms already rejected by Maxwell | Tessellation and storage-write stage features must be enabled by the Vulkan pipeline consumer |
| Scalar operations | Integer arithmetic/bit operations/carry/condition codes; predicates; float arithmetic/comparisons/min-max/special functions; half operations; rounding/conversions | Guest FP64 arithmetic remains outside current Maxwell coverage | Float control guarantees; exact F32 FMA requires `shaderFmaFloat32`; optional integer64 |
| Resources | Live CB/storage buffers; dynamic byte addresses; existing global extent proofs; 2D/array/cube/depth textures, supported LOD/compare/fetch/offset/mip queries; separate images/samplers | Bindless/dynamically selected descriptors and TEX forms already rejected by the decoder | Device descriptor/format limits are consumed by resource/pipeline creation |
| Memory/compute | QMD workgroup size, ordered effects, U32/U64 addresses, 32-bit word memory, explicit shared/private arrays and scoped barriers | Narrow/wide scalar memory requests without an exact lowering; device-wide execution barriers; currently unsupported Maxwell shared-memory instruction forms | Actual shared-memory capacity and enabled features |
| Flow | Predication, branches, bounded decoded BRX tables, loops, early return/discard, multiple continues/exits and static irreducible CFG splitting | Path-dependent reconvergence token destinations and unbounded frontend stack exploration remain precise errors | No vendor-name heuristic |
| Fragment quads | Native subgroup quad operations; lane coefficients use subgroup quad position | Missing quad feature reports a source diagnostic | Fragment subgroup QUAD support |

The current games still use the oracle until native renderer integration/cutover.
Its register IR, export adapters and WGSL retire in tasks 15–16. Shader modules
alone do not implement full guest raster execution or presentation. Task 09 adds
[native caches](../../../../gpu-vulkan/src/cache/README.md); guest recording/integration
remain tasks 10–11.

## Checks

CPU compiler checks:

```sh
cargo test -p nixe-gpu -p nixe-gpu-maxwell shader::ssa --lib
NIXE_SPIRV_VAL=/path/to/pinned/spirv-val cargo test -p nixe-gpu shader::ssa::spirv -- --include-ignored
NIXE_SPIRV_VAL=/path/to/pinned/spirv-val cargo test -p nixe-gpu-maxwell complete_guest_patch_programs_compile_directly_through_ssa -- --include-ignored
```

These cover SSA/dominance, actual Maxwell fixtures, A/B forwarding, pipeline
capture/linkage, descriptor ABI, structured flow, all texture classes, quad
interfaces, default tessellation domains and complete deko guest TCS/TES programs.
Oracle comparisons exist only in test code.

Real-device checks use the existing Vulkan owner/submission/allocator and a small
test compute pipeline, never a software GPU or a production compatibility layer:

```sh
NIXE_SPIRV_VAL=/path/to/pinned/spirv-val NIXE_VULKAN_TEST_ADAPTER=NVIDIA \
VK_LAYER_PATH=/path/to/validation/layers VK_LAYER_VALIDATE_SYNC=1 \
cargo test -p nixe-gpu-vulkan --lib shader_tests -- --nocapture
```

They compare output words for native branches/loops/phis, shared-memory exchange,
rounding/conversions, fused FMA, binary16 midpoint cases and randomized operands,
min/max signed-zero/subnormal cases, and bounded irreducible/multiple-exit cycles.
When supplied, the pinned validator checks each module before driver compilation.
Tests skip only when no real GPU is available; device/profile/validation failures
on available hardware fail. Vulkan 1.2 validation exercises the common SPIR-V
contract, including explicitly declared extensions.

Devices without native float32 gradual underflow use exact capability-based
lowerings for preserving/mixed add, multiply and FMA: normal values keep the native
fast path; underflow uses enabled float64 and explicit float32 rounding. FMA keeps
the wide residual so float64-to-float32 double rounding cannot change a midpoint.
Inverse operations and explicitly approximate special functions normalize tiny
inputs/results while retaining their guest DAZ/FTZ policy. No vendor/title rule or
silently enabled `DenormPreserve` mode selects these paths. Hardware checks cover
tiny values, signed zeros, infinities, mixed controls and midpoint counterexamples.

Float32 multiplication toward zero with DAZ/FTZ uses an exact 24×24-bit integer
product (`OpUMulExtended`), including finite overflow, signed zero and quiet NaN
propagation. It requires no float64 or native floating-point rounding mode.
Other directed arithmetic modes still require an explicit numerical lowering.

## Native compilation caching

`ssa::artifact` defines stable content/namespace/compiler identities, with explicit
little-endian emission specialization. SHA-256 runs only during capture/cache-miss
or persistence work; in-memory changed-state indexes use FxHash. `ssa::compile`
provides one bounded worker queue and deduplicated publication shared by frontend
and host artifact/pipeline compilation. Prepared native references bypass both.
No old `GpuCacheConfiguration` or oracle fingerprint policy selects these caches.

Task 10 adds native 32-bit scaled vertex legalization and a direct SPIR-V host
geometry stage for exact quad assembly. It forwards linked outputs, selects the
fourth corner's flat values and does not re-evaluate guest vertex effects. This
host primitive lowering does not implement guest geometry shader instructions.
