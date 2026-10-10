// Nixe-owned interchange payload. This is not Dynarmic's private JitState.
#pragma once
#include <stdint.h>

struct NixeState {
  uint64_t x[31], sp, pc, vectors[32][2], tpidr, tpidrro;
  uint32_t nzcv, fpcr, fpsr;
};

#ifdef __cplusplus
#include <cstddef>
#include <type_traits>
static_assert(std::is_standard_layout_v<NixeState>);
static_assert(std::is_trivially_copyable_v<NixeState>);
static_assert(sizeof(NixeState) == 808 && alignof(NixeState) == 8);
static_assert(offsetof(NixeState, x) == 0);
static_assert(offsetof(NixeState, sp) == 248);
static_assert(offsetof(NixeState, pc) == 256);
static_assert(offsetof(NixeState, vectors) == 264);
static_assert(offsetof(NixeState, tpidr) == 776);
static_assert(offsetof(NixeState, tpidrro) == 784);
static_assert(offsetof(NixeState, nzcv) == 792);
static_assert(offsetof(NixeState, fpcr) == 796);
static_assert(offsetof(NixeState, fpsr) == 800);
#endif
