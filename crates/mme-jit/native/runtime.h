// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <asmjit/x86.h>
#include <cstdint>

struct NixeMmeRuntime {
  asmjit::JitRuntime jit;
};
struct NixeMmeCompileError {
  uint32_t pc;
  char message[256];
};
