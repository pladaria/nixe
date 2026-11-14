// SPDX-License-Identifier: GPL-3.0-or-later
#include "runtime.h"
extern "C" void *nixe_mme_runtime_new() noexcept {
  try {
    return new NixeMmeRuntime;
  } catch (...) {
    return nullptr;
  }
}
extern "C" void nixe_mme_runtime_delete(void *runtime) noexcept {
  delete static_cast<NixeMmeRuntime *>(runtime);
}
extern "C" void nixe_mme_release(void *runtime, const void *program) noexcept {
  static_cast<NixeMmeRuntime *>(runtime)->jit.release(program);
}
