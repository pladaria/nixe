/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/x64/perf_map.h"

#include <cstddef>
#include <string>

#ifdef __linux__

#    include <cstdio>
#    include <cstdlib>
#    include <iterator>
#    include <map>
#    include <mutex>

#    include <fmt/format.h>
#    include <mcl/stdint.hpp>
#    include <sys/types.h>
#    include <unistd.h>

namespace Dynarmic::Backend::X64 {

namespace {
std::mutex mutex;
std::FILE* file = nullptr;
struct Symbol {
    u64 bytes;
    std::string name;
};
// Only enabled diagnostic registration populates this process-wide table.
// Cache resets retain the other cores' symbols and the owner's prelude.
std::map<u64, Symbol>& Symbols() {
    static std::map<u64, Symbol> symbols;
    return symbols;
}

void WriteSymbol(u64 address, const Symbol& symbol) {
    const std::string line = fmt::format("{:016x} {:016x} {:s}\n", address, symbol.bytes, symbol.name);
    std::fwrite(line.data(), sizeof *line.data(), line.size(), file);
}

void OpenFile() {
    const char* perf_dir = std::getenv("PERF_BUILDID_DIR");
    if (!perf_dir) {
        file = nullptr;
        return;
    }

    const pid_t pid = getpid();
    const std::string filename = fmt::format("{:s}/perf-{:d}.map", perf_dir, pid);

    file = std::fopen(filename.c_str(), "w");
    if (!file) {
        return;
    }

    std::setvbuf(file, nullptr, _IONBF, 0);
    for (const auto& [address, symbol] : Symbols()) {
        WriteSymbol(address, symbol);
    }
}
}  // anonymous namespace

namespace detail {
void PerfMapRegister(const void* start, const void* end, std::string_view friendly_name) {
    if (start == end) {
        // Nothing to register
        return;
    }

    std::lock_guard guard{mutex};

    if (!file) {
        OpenFile();
        if (!file) {
            return;
        }
    }

    const auto address = reinterpret_cast<u64>(start);
    const auto symbol = Symbols().insert_or_assign(
                                     address, Symbol{reinterpret_cast<u64>(end) - address, std::string{friendly_name}})
                            .first;
    WriteSymbol(address, symbol->second);
}

}  // namespace detail

bool PerfMapEnabled() noexcept {
    static const bool enabled = std::getenv("PERF_BUILDID_DIR") != nullptr;
    return enabled;
}

void PerfMapClear(const void* begin, const void* end) {
    if (!PerfMapEnabled()) {
        return;
    }
    std::lock_guard guard{mutex};
    auto& symbols = Symbols();
    const auto start = reinterpret_cast<u64>(begin);
    auto first = symbols.lower_bound(start);
    if (first != symbols.begin()) {
        const auto previous = std::prev(first);
        if (previous->first + previous->second.bytes > start) {
            first = previous;
        }
    }
    const auto last = symbols.lower_bound(reinterpret_cast<u64>(end));
    if (first == last) {
        return;
    }
    symbols.erase(first, last);
    if (file) {
        std::fclose(file);
        file = nullptr;
    }
    // Rebuild the live map only on an actual cache retirement, never per block.
    OpenFile();
}

}  // namespace Dynarmic::Backend::X64

#endif
