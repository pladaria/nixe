/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <algorithm>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <mcl/bit/bit_count.hpp>
#include <mcl/stdint.hpp>

#include "dynarmic/frontend/decoder/decoder_detail.h"
#include "dynarmic/frontend/decoder/matcher.h"

namespace Dynarmic::A64 {

template<typename Visitor>
using Matcher = Decoder::Matcher<Visitor, u32>;

template<typename Visitor>
struct Handler {
    typename Matcher<Visitor>::handler_function fn;
    u32 mask;
    u32 expected;

    bool Matches(u32 instruction) const {
        return (instruction & mask) == expected;
    }

    typename Visitor::instruction_return_type call(Visitor& visitor, u32 instruction) const {
        DEBUG_ASSERT(Matches(instruction));
        return fn(visitor, instruction);
    }
};

template<typename Visitor>
using DecodeTable = std::array<std::vector<Handler<Visitor>>, 0x1000>;

namespace detail {
inline size_t ToFastLookupIndex(u32 instruction) {
    return ((instruction >> 10) & 0x00F) | ((instruction >> 18) & 0xFF0);
}
}  // namespace detail

template<typename V>
std::vector<Matcher<V>> GetMatchers() {
    std::vector<Matcher<V>> list = {
#define INST(fn, name, bitstring) DYNARMIC_DECODER_GET_MATCHER(Matcher, fn, name, Decoder::detail::StringToArray<32>(bitstring)),
#include "./a64.inc"
#undef INST
    };

    std::stable_sort(list.begin(), list.end(), [](const auto& matcher1, const auto& matcher2) {
        // If a matcher has more bits in its mask it is more specific, so it should come first.
        return mcl::bit::count_ones(matcher1.GetMask()) > mcl::bit::count_ones(matcher2.GetMask());
    });

    // Exceptions to the above rule of thumb.
    const std::set<std::string> comes_first{
        "MOVI, MVNI, ORR, BIC (vector, immediate)",
        "FMOV (vector, immediate)",
        "Unallocated SIMD modified immediate",
    };

    std::stable_partition(list.begin(), list.end(), [&](const auto& matcher) {
        return comes_first.count(matcher.GetName()) > 0;
    });

    return list;
}

template<typename V>
DecodeTable<V> GetDecodeTable() {
    // Eden keeps only the direct callback and opcode match in the hot table.
    // Preserve this frontend's specificity and SIMD exception order while
    // discarding names after admission; diagnostics can resolve them on demand.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/dynarmic/src/dynarmic/frontend/A64/decoder/a64.h
    const auto list = GetMatchers<V>();
    static_assert(sizeof(Handler<V>) == sizeof(void*) + 2 * sizeof(u32));
    DecodeTable<V> table{};
    for (size_t i = 0; i < table.size(); ++i) {
        for (auto matcher : list) {
            const auto expect = detail::ToFastLookupIndex(matcher.GetExpected());
            const auto mask = detail::ToFastLookupIndex(matcher.GetMask());
            if ((i & mask) == expect) {
                table[i].push_back({matcher.GetHandler(), matcher.GetMask(), matcher.GetExpected()});
            }
        }
    }
    return table;
}

template<typename V>
std::optional<std::reference_wrapper<const Handler<V>>> Decode(u32 instruction) {
    alignas(64) static const auto table = GetDecodeTable<V>();

    const auto matches_instruction = [instruction](const auto& matcher) { return matcher.Matches(instruction); };

    const auto& subtable = table[detail::ToFastLookupIndex(instruction)];
    auto iter = std::find_if(subtable.begin(), subtable.end(), matches_instruction);
    return iter != subtable.end() ? std::optional<std::reference_wrapper<const Handler<V>>>(*iter) : std::nullopt;
}

template<typename V>
const char* GetName(u32 instruction) {
    const auto list = GetMatchers<V>();
    const auto iter = std::find_if(list.begin(), list.end(), [instruction](const auto& matcher) {
        return matcher.Matches(instruction);
    });
    return iter != list.end() ? iter->GetName() : "<null>";
}

}  // namespace Dynarmic::A64
