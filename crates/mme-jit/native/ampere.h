// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "runtime.h"
#include <cstddef>

struct AmpereAlu {
  uint8_t op, dst, x, y;
  uint16_t imm;
};
struct AmpereInstruction {
  uint8_t end, predicate, predicate_reg;
  AmpereAlu alu[2];
  uint8_t output[2][2], loads[2];
};
struct AmpereInvocation {
  uint32_t regs[24];
  uint32_t carry, method, increment;
  uint32_t loops, loop_start, loop_end, next_pc, budget;
  uint32_t scratch_x, scratch_y;
  const uint32_t *params;
  size_t params_size, consumed;
  uint32_t *dram;
  size_t dram_size;
  const uint32_t *bank, *valid_bank;
  void *host;
  uint32_t (*emit)(AmpereInvocation *, uint32_t, uint32_t);
  uint32_t (*poll)(AmpereInvocation *);
  uint32_t start, error_pc, error_value;
  uint8_t end_next, stopped, cancel_end, padding;
  const uint64_t *owned;
  uint32_t (*resolve)(AmpereInvocation *, uint32_t);
  uint32_t state_value;
};
static_assert(sizeof(AmpereAlu) == 6);
static_assert(sizeof(AmpereInstruction) == 22);
static_assert(offsetof(AmpereInvocation, params) == 136);
static_assert(offsetof(AmpereInvocation, start) == 216);
static_assert(offsetof(AmpereInvocation, owned) == 232);
static_assert(sizeof(AmpereInvocation) == 256);
using AmpereProgram = uint32_t (*)(AmpereInvocation *);
extern "C" bool nixe_ampere_mme_compile(void *, const AmpereInstruction *,
                                        const uint8_t *, size_t, bool,
                                        AmpereProgram *, size_t *,
                                        NixeMmeCompileError *) noexcept;
