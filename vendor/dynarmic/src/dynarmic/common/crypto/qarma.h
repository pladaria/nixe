// SPDX-License-Identifier: LGPL-2.1-or-later
#pragma once
#include <cstdint>

namespace Dynarmic::Common::Crypto {
std::uint64_t ComputePAC(std::uint64_t data, std::uint64_t modifier, std::uint64_t key_hi, std::uint64_t key_lo);
}
