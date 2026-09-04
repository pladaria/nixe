# Evidence from reference engines

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Architecture, objectives and invariants](architecture.md); [Cranelift fork and backend contract](backend.md).

## Evidence from reference engines

The architecture deliberately combines two proven families instead of copying
one engine wholesale.

Ryujinx ARMeilleure separates fast low-quality compilation from higher-quality
background recompilation. It uses roughly 500 instructions for low quality,
2500 for high quality and requests retranslation after about 100 entry calls.
Its exact-root multi-block cache can retain overlapping functions, so its
constants do not by themselves solve Nixe's overlap problem. See the pinned
[decoder](https://git.axenov.dev/Museum/ryujinx/src/commit/a0624de3fdaa125d51187f90069c7150219b3b55/src/ARMeilleure/Decoders/Decoder.cs),
[translator](https://git.axenov.dev/Museum/ryujinx/src/commit/a0624de3fdaa125d51187f90069c7150219b3b55/src/ARMeilleure/Translation/Translator.cs)
and
[translation cache](https://git.axenov.dev/Museum/ryujinx/src/commit/a0624de3fdaa125d51187f90069c7150219b3b55/src/ARMeilleure/Translation/TranslatorCache.cs).

FEX uses multi-block compilation with selected public entries and preserves
post-call continuations without making every internal block public. It permits
cheap duplicate frontend work in a race, then rechecks before expensive
backend work. Its 5000-instruction default explicitly carries a compilation
stutter warning, so that number is not copied. See the pinned
[multiblock frontend](https://github.com/FEX-Emu/FEX/blob/511c45c4c63ae2958027ca7bfdb88cea457afceb/FEXCore/Source/Interface/Core/Frontend.cpp#L1137-L1220),
[entry emission](https://github.com/FEX-Emu/FEX/blob/511c45c4c63ae2958027ca7bfdb88cea457afceb/FEXCore/Source/Interface/Core/JIT/JIT.cpp#L930-L978),
[race recheck](https://github.com/FEX-Emu/FEX/blob/511c45c4c63ae2958027ca7bfdb88cea457afceb/FEXCore/Source/Interface/Core/Core.cpp#L779-L829)
and
[configuration](https://github.com/FEX-Emu/FEX/blob/511c45c4c63ae2958027ca7bfdb88cea457afceb/FEXCore/Source/Interface/Config/Config.json.in).

QEMU normally translates single-entry TBs with a general ceiling of 512 guest
instructions, then avoids dispatcher round trips through direct block
chaining. It accepts rare differently rooted overlap because units are small
and its code buffer is reclaimable. See the pinned
[TB bound](https://github.com/qemu/qemu/blob/93b9a2436564a9df25a0b978c8245fed255264f2/include/exec/translation-block.h#L75-L88),
[direct-chaining design](https://github.com/qemu/qemu/blob/93b9a2436564a9df25a0b978c8245fed255264f2/docs/devel/tcg.rst#L33-L125)
and
[concurrent publication](https://github.com/qemu/qemu/blob/93b9a2436564a9df25a0b978c8245fed255264f2/accel/tcg/translate-all.c#L524-L539).

Dynarmic/Yuzu uses exact LocationDescriptor blocks, direct LinkBlock patching,
fast dispatch and a return-stack hint. Dolphin likewise compiles exact
root/state blocks, directly links them and follows only a shallow bounded
branch shape. Their small units are successful because their dispatch and
cross-block boundary are designed to be cheap. See Dynarmic's
[design](https://github.com/Borked3DS/dynarmic/blob/bd287ce645117040abb393357f82fa55e7a16242/docs/Design.md),
[A64 translator](https://github.com/Borked3DS/dynarmic/blob/bd287ce645117040abb393357f82fa55e7a16242/src/dynarmic/frontend/A64/translate/a64_translate.cpp)
and
[A64 linker](https://github.com/Borked3DS/dynarmic/blob/bd287ce645117040abb393357f82fa55e7a16242/src/dynarmic/backend/arm64/address_space.cpp),
and Dolphin's
[analyzer](https://github.com/dolphin-emu/dolphin/blob/a1e636d72c8469acf747ac6542f0b7ace7cea02f/Source/Core/Core/PowerPC/PPCAnalyst.cpp#L806-L985)
and
[block cache](https://github.com/dolphin-emu/dolphin/blob/a1e636d72c8469acf747ac6542f0b7ace7cea02f/Source/Core/Core/PowerPC/JitCommon/JitCache.cpp).

These references support the chosen hybrid:

- LCQ uses small canonical basic blocks and cheap native chaining.
- HCQ uses execution-informed multi-block regions to recover cross-block SSA
  and higher-quality allocation.
- A bounded cache and versioned replacement make temporary code duplication
  recoverable.
- Exact-key deduplication and a short pre-backend reservation prevent duplicate
  expensive compilation without serializing all discovery.
- A classic linear trace JIT is rejected. Its duplicated tails, side-exit
  proliferation, edge instrumentation and deoptimization surface are not
  justified for already-compiled A64 software.
