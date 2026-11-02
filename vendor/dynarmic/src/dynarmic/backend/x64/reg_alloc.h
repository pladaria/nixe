/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <optional>
#include <utility>

#include <boost/container/small_vector.hpp>
#include <mcl/stdint.hpp>

#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/hostloc.h"
#include "dynarmic/backend/x64/oparg.h"
#include "dynarmic/backend/x64/stack_layout.h"
#include "dynarmic/backend/x64/xbyak.h"
#include "dynarmic/ir/cond.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/value.h"

namespace Dynarmic::IR {
enum class AccType;
}  // namespace Dynarmic::IR

namespace Dynarmic::Backend::X64 {

class RegAlloc;

struct HostLocInfo {
public:
    HostLocInfo() = default;
    HostLocInfo(const HostLocInfo&) = delete;
    HostLocInfo& operator=(const HostLocInfo&) = delete;
    HostLocInfo(HostLocInfo&&) = delete;
    HostLocInfo& operator=(HostLocInfo&&) = delete;

    bool IsLocked() const {
        return is_being_used_count > 0;
    }
    bool IsEmpty() const {
        return is_being_used_count == 0 && values.empty();
    }
    bool IsLastUse() const {
        return is_being_used_count == 0 && current_references == 1 && accumulated_uses + 1 == total_uses;
    }

    void SetLastUse();

    void ReadLock();
    void WriteLock();
    void AddArgReference();
    void ReleaseOne();
    void ReleaseAll();

    size_t GetMaxBitWidth() const {
        return max_bit_width;
    }

    void AddValue(HostLoc loc, IR::Inst* inst);
    void SetLocation(HostLoc loc);
    void MoveFrom(HostLocInfo& from);
    void ExchangeWith(HostLocInfo& other);

    void EmitVerboseDebuggingOutput(BlockOfCode& code, size_t host_loc_index) const;

private:
    void ClearValues();

    // Current instruction state
    size_t is_being_used_count = 0;
    bool is_scratch = false;
    bool is_set_last_use = false;

    // Block state
    size_t current_references = 0;
    size_t accumulated_uses = 0;
    size_t total_uses = 0;

    // Value state
    boost::container::small_vector<IR::Inst*, 3> values;
    size_t max_bit_width = 0;
};

struct Argument {
public:
    using copyable_reference = std::reference_wrapper<Argument>;

    IR::Type GetType() const {
        return value.GetType();
    }
    bool IsImmediate() const {
        return value.IsImmediate();
    }
    bool IsVoid() const {
        return GetType() == IR::Type::Void;
    }

    bool FitsInImmediateU32() const;
    bool FitsInImmediateS32() const;

    bool GetImmediateU1() const;
    u8 GetImmediateU8() const;
    u16 GetImmediateU16() const;
    u32 GetImmediateU32() const;
    u64 GetImmediateS32() const;
    u64 GetImmediateU64() const;
    IR::Cond GetImmediateCond() const;
    IR::AccType GetImmediateAccType() const;

    /// Is this value currently in a GPR?
    bool IsInGpr() const;
    /// Is this value currently in a XMM?
    bool IsInXmm() const;
    /// Is this value currently in memory?
    bool IsInMemory() const;

private:
    friend class RegAlloc;
    explicit Argument(RegAlloc& reg_alloc)
            : reg_alloc(reg_alloc) {}

    bool allocated = false;
    RegAlloc& reg_alloc;
    IR::Value value;
};

class RegAlloc final {
public:
    using ArgumentInfo = std::array<Argument, IR::max_arg_count>;

    explicit RegAlloc(BlockOfCode& code, HostLocSet gpr_candidates, HostLocSet xmm_candidates);

    ArgumentInfo GetArgumentInfo(IR::Inst* inst);
    void RegisterPseudoOperation(IR::Inst* inst);
    bool IsValueLive(IR::Inst* inst) const;

    Xbyak::Reg64 UseGpr(Argument& arg);
    Xbyak::Xmm UseXmm(Argument& arg);
    OpArg UseOpArg(Argument& arg);
    void Use(Argument& arg, HostLoc host_loc);

    Xbyak::Reg64 UseScratchGpr(Argument& arg);
    Xbyak::Xmm UseScratchXmm(Argument& arg);
    void UseScratch(Argument& arg, HostLoc host_loc);

    void DefineValue(IR::Inst* inst, const Xbyak::Reg& reg);
    void DefineValue(IR::Inst* inst, Argument& arg);

    void Release(const Xbyak::Reg& reg);

    Xbyak::Reg64 ScratchGpr();
    Xbyak::Reg64 ScratchGpr(HostLoc desired_location);
    Xbyak::Xmm ScratchXmm();
    Xbyak::Xmm ScratchXmm(HostLoc desired_location);

    void HostCall(IR::Inst* result_def = nullptr,
                  std::optional<Argument::copyable_reference> arg0 = {},
                  std::optional<Argument::copyable_reference> arg1 = {},
                  std::optional<Argument::copyable_reference> arg2 = {},
                  std::optional<Argument::copyable_reference> arg3 = {});

    // TODO: Values in host flags

    void AllocStackSpace(size_t stack_space);
    void ReleaseStackSpace(size_t stack_space);

    void EndOfAllocScope();

    void AssertNoMoreUses() const {
#ifndef NDEBUG
        ASSERT(std::all_of(hostloc_info.begin(), hostloc_info.end(), [](const auto& i) { return i.IsEmpty(); }));
        ASSERT(std::all_of(scope_locations.begin(), scope_locations.end(), [](u64 word) { return word == 0; }));
        ASSERT(reserved_stack_space == 0);
#endif
    }

    void EmitVerboseDebuggingOutput();

private:
    friend struct Argument;

    HostLocSet gpr_candidates;
    HostLocSet xmm_candidates;

    HostLoc SelectARegister(HostLocSet desired_locations) const;
    std::optional<HostLoc> ValueLocation(const IR::Inst* value) const {
        const auto location = value->GetAllocationLocation();
        if (location == IR::Inst::InvalidAllocationLocation) {
            return std::nullopt;
        }
        DEBUG_ASSERT(location < hostloc_info.size());
        return static_cast<HostLoc>(location);
    }

    HostLoc UseImpl(IR::Value use_value, HostLocSet desired_locations);
    HostLoc UseScratchImpl(IR::Value use_value, HostLocSet desired_locations);
    HostLoc ScratchImpl(HostLocSet desired_locations);
    void DefineValueImpl(IR::Inst* def_inst, HostLoc host_loc);
    void DefineValueImpl(IR::Inst* def_inst, const IR::Value& use_inst);

    HostLoc LoadImmediate(IR::Value imm, HostLoc host_loc);
    void Move(HostLoc to, HostLoc from);
    void CopyToScratch(size_t bit_width, HostLoc to, HostLoc from);
    void Exchange(HostLoc a, HostLoc b);
    void MoveOutOfTheWay(HostLoc reg);

    void SpillRegister(HostLoc loc);
    HostLoc FindFreeSpill() const;

    std::array<HostLocInfo, NonSpillHostLocCount + SpillCount> hostloc_info;
    std::array<u64, (NonSpillHostLocCount + SpillCount + 63) / 64> scope_locations{};
    HostLocInfo& LocInfo(HostLoc loc) {
        DEBUG_ASSERT(loc != HostLoc::RSP && loc != HostLoc::R15);
        const auto index = static_cast<size_t>(loc);
        scope_locations[index / 64] |= u64{1} << (index % 64);
        return hostloc_info[index];
    }
    const HostLocInfo& LocInfo(HostLoc loc) const {
        DEBUG_ASSERT(loc != HostLoc::RSP && loc != HostLoc::R15);
        return hostloc_info[static_cast<size_t>(loc)];
    }

    BlockOfCode& code;
    size_t reserved_stack_space = 0;
    void EmitMove(size_t bit_width, HostLoc to, HostLoc from);
    void EmitExchange(HostLoc a, HostLoc b);

    Xbyak::Address SpillToOpArg(HostLoc loc);
};

}  // namespace Dynarmic::Backend::X64
