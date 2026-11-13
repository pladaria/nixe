// SPDX-License-Identifier: 0BSD
#pragma once

#include <array>

#include <mcl/stdint.hpp>

namespace Dynarmic::A64 {
// Immutable EL0 execution environment. Keys belong to the guest process, never
// to a host worker. No privileged key register is exposed to EL0.
// ARM Cortex-A78C TRM B2.80: QARMA5, FEAT_PAuth2, no FPAC/FPACCOMBINED.
// https://documentation-service.arm.com/static/6193c9bef45f0b1fbf3a85dd
struct PointerAuthentication {
    // APIA, APIB, APDA, APDB, APGA; each pair is {low, high}.
    std::array<std::array<u64, 2>, 5> keys{};
    // Zero means FEAT_PAuth is absent. Otherwise use this VA width with TBI=0.
    // The execution environment must use the same VA interpretation for memory.
    unsigned address_bits = 0;
};
}  // namespace Dynarmic::A64
