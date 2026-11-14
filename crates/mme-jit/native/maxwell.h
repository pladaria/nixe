// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include "runtime.h"

constexpr uint32_t NIXE_MME_BANK_WORDS = 0xe00;
using NixeMmeEmit = uint32_t (*)(void *, uint32_t, uint32_t);
struct NixeMmeState {
  uint32_t *bank;
  void *host;
  uint32_t registers[8];
  uint32_t carry_flag;
  NixeMmeEmit emit;
  const uint32_t *next_parameter;
  uint32_t error;
  uint32_t error_pc;
  uint32_t error_value;
};
using NixeMmeProgram = void (*)(NixeMmeState *, const uint32_t *,
                                const uint32_t *);
extern "C" {
bool nixe_mme_compile(void *, const uint32_t *, size_t, NixeMmeProgram *,
                      NixeMmeCompileError *) noexcept;
}
