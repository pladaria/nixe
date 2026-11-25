/* This file is part of the dynarmic project.
 * SPDX-License-Identifier: 0BSD
 */
#pragma once

#include <cstdint>

namespace Dynarmic::A64 {

// Written only by the thread executing this JIT. Consumers sample at Run
// boundaries; no generated instruction or process-wide counter is involved.
struct Compilation {
    std::uint64_t descriptor{}, fingerprint{}, host_address{}, native_bytes{};
    std::uint64_t guest_bytes{}, ir_instructions{};
    std::uint64_t near_bytes{}, cold_bytes{};
    std::uint64_t translation_ns{}, optimization_ns{}, emission_ns{};
};

struct Diagnostics {
    std::uint64_t lookups{}, hits{}, compiled{}, guest_bytes{}, native_bytes{};
    std::uint64_t mainline_bytes{}, deferred_bytes{};
    std::uint64_t near_bytes{}, cold_bytes{};
    std::uint64_t dispatch_hits{}, dispatch_misses{}, rsb_hits{};
    std::uint64_t capacity_clears{}, requested_clears{}, targeted_blocks{}, fastmem_blocks{};
    std::uint64_t fastmem_retained{};
    std::uint64_t host_calls{}, host_returns{};
    std::uint64_t translation_ns{}, optimization_ns{}, emission_ns{};
    void* context{};
    void (*compile)(void*, const Compilation&){};
    void (*invalidate)(void*, std::uint64_t descriptor, bool fastmem){};
    void (*clear)(void*, bool capacity){};
};

}  // namespace Dynarmic::A64
