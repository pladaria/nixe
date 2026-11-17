/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <string_view>

#include <mcl/bit_cast.hpp>

namespace Dynarmic::Backend::X64 {

#ifdef __linux__
/// Diagnostic symbol naming is selected once for the process.
bool PerfMapEnabled() noexcept;
namespace detail {
void PerfMapRegister(const void* start, const void* end, std::string_view friendly_name);
}  // namespace detail
/// Remove symbols overlapping this cache range; other JITs retain their names.
void PerfMapClear(const void* begin, const void* end);
#else
inline bool PerfMapEnabled() noexcept {
    return false;
}
namespace detail {
inline void PerfMapRegister(const void*, const void*, std::string_view) {}
}  // namespace detail
inline void PerfMapClear(const void*, const void*) {}
#endif

template<typename T>
void PerfMapRegister(T start, const void* end, std::string_view friendly_name) {
    if (PerfMapEnabled()) {
        detail::PerfMapRegister(mcl::bit_cast<const void*>(start), end, friendly_name);
    }
}

}  // namespace Dynarmic::Backend::X64
