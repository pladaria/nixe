/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/x64/reg_alloc.h"

#include <algorithm>
#include <bit>
#include <numeric>
#include <utility>

#include <fmt/ostream.h>
#include <mcl/assert.hpp>
#include <mcl/bit_cast.hpp>

#include "dynarmic/backend/x64/abi.h"
#include "dynarmic/backend/x64/stack_layout.h"
#include "dynarmic/backend/x64/verbose_debugging_output.h"
#include "dynarmic/backend/x64/xbyak.h"

namespace Dynarmic::Backend::X64 {

static_assert(NonSpillHostLocCount + SpillCount <= IR::Inst::InvalidAllocationLocation);

#define MAYBE_AVX(OPCODE, ...)                       \
    [&] {                                            \
        if (code.HasHostFeature(HostFeature::AVX)) { \
            code.v##OPCODE(__VA_ARGS__);             \
        } else {                                     \
            code.OPCODE(__VA_ARGS__);                \
        }                                            \
    }()

static bool CanExchange(HostLoc a, HostLoc b) {
    return HostLocIsGPR(a) && HostLocIsGPR(b);
}

// Minimum number of bits required to represent a type
static size_t GetBitWidth(IR::Type type) {
    switch (type) {
    case IR::Type::A32Reg:
    case IR::Type::A32ExtReg:
    case IR::Type::A64Reg:
    case IR::Type::A64Vec:
    case IR::Type::CoprocInfo:
    case IR::Type::Cond:
    case IR::Type::Void:
    case IR::Type::Table:
    case IR::Type::AccType:
        ASSERT_FALSE("Type {} cannot be represented at runtime", type);
    case IR::Type::Opaque:
        ASSERT_FALSE("Not a concrete type");
    case IR::Type::U1:
        return 8;
    case IR::Type::U8:
        return 8;
    case IR::Type::U16:
        return 16;
    case IR::Type::U32:
        return 32;
    case IR::Type::U64:
        return 64;
    case IR::Type::U128:
        return 128;
    case IR::Type::NZCVFlags:
        return 32;  // TODO: Update to 16 when flags optimization is done
    }
    UNREACHABLE();
}

static bool IsValuelessType(IR::Type type) {
    switch (type) {
    case IR::Type::Table:
        return true;
    default:
        return false;
    }
}

void HostLocInfo::SetLastUse() {
    ASSERT(IsLastUse());
    is_set_last_use = true;
}

void HostLocInfo::ReadLock() {
    ASSERT(!is_scratch);
    is_being_used_count++;
}

void HostLocInfo::WriteLock() {
    ASSERT(is_being_used_count == 0);
    is_being_used_count++;
    is_scratch = true;
}

void HostLocInfo::AddArgReference() {
    current_references++;
    ASSERT(accumulated_uses + current_references <= total_uses);
}

void HostLocInfo::ReleaseOne() {
    is_being_used_count--;
    is_scratch = false;

    if (current_references == 0)
        return;

    accumulated_uses++;
    current_references--;

    if (current_references == 0)
        ReleaseAll();
}

void HostLocInfo::ReleaseAll() {
    accumulated_uses += current_references;
    current_references = 0;

    is_set_last_use = false;

    if (total_uses == accumulated_uses) {
        ClearValues();
        accumulated_uses = 0;
        total_uses = 0;
        max_bit_width = 0;
    }

    is_being_used_count = 0;
    is_scratch = false;
}

void HostLocInfo::ClearValues() {
    for (auto* value : values) {
        value->SetAllocationLocation(IR::Inst::InvalidAllocationLocation);
    }
    values.clear();
}

void HostLocInfo::SetLocation(HostLoc loc) {
    for (auto* value : values) {
        value->SetAllocationLocation(static_cast<u8>(loc));
    }
}

void HostLocInfo::MoveFrom(HostLocInfo& from) {
    ASSERT(this != &from && IsEmpty() && !from.IsLocked());
    // Storage belongs to the resident host slot. Moving the small_vector itself
    // leaves its source without even its inline capacity and discards the
    // destination's retained allocation. Transfer values, keeping both buffers.
    values.assign(from.values.begin(), from.values.end());
    from.values.clear();
    is_being_used_count = std::exchange(from.is_being_used_count, 0);
    is_scratch = std::exchange(from.is_scratch, false);
    is_set_last_use = std::exchange(from.is_set_last_use, false);
    current_references = std::exchange(from.current_references, 0);
    accumulated_uses = std::exchange(from.accumulated_uses, 0);
    total_uses = std::exchange(from.total_uses, 0);
    max_bit_width = std::exchange(from.max_bit_width, 0);
}

void HostLocInfo::ExchangeWith(HostLocInfo& other) {
    values.swap(other.values);
    std::swap(is_being_used_count, other.is_being_used_count);
    std::swap(is_scratch, other.is_scratch);
    std::swap(is_set_last_use, other.is_set_last_use);
    std::swap(current_references, other.current_references);
    std::swap(accumulated_uses, other.accumulated_uses);
    std::swap(total_uses, other.total_uses);
    std::swap(max_bit_width, other.max_bit_width);
}

void HostLocInfo::AddValue(HostLoc loc, IR::Inst* inst) {
    if (is_set_last_use) {
        is_set_last_use = false;
        ClearValues();
    }
    values.push_back(inst);
    inst->SetAllocationLocation(static_cast<u8>(loc));
    total_uses += inst->UseCount();
    max_bit_width = std::max(max_bit_width, GetBitWidth(inst->GetType()));
}

void HostLocInfo::EmitVerboseDebuggingOutput(BlockOfCode& code, size_t host_loc_index) const {
    using namespace Xbyak::util;
    for (IR::Inst* value : values) {
        code.mov(code.ABI_PARAM1, rsp);
        code.mov(code.ABI_PARAM2, host_loc_index);
        code.mov(code.ABI_PARAM3, value->GetName());
        code.mov(code.ABI_PARAM4, GetBitWidth(value->GetType()));
        code.CallFunction(PrintVerboseDebuggingOutputLine);
    }
}

bool Argument::FitsInImmediateU32() const {
    if (!IsImmediate())
        return false;
    const u64 imm = value.GetImmediateAsU64();
    return imm < 0x100000000;
}

bool Argument::FitsInImmediateS32() const {
    if (!IsImmediate())
        return false;
    const s64 imm = static_cast<s64>(value.GetImmediateAsU64());
    return -s64(0x80000000) <= imm && imm <= s64(0x7FFFFFFF);
}

bool Argument::GetImmediateU1() const {
    return value.GetU1();
}

u8 Argument::GetImmediateU8() const {
    const u64 imm = value.GetImmediateAsU64();
    ASSERT(imm < 0x100);
    return u8(imm);
}

u16 Argument::GetImmediateU16() const {
    const u64 imm = value.GetImmediateAsU64();
    ASSERT(imm < 0x10000);
    return u16(imm);
}

u32 Argument::GetImmediateU32() const {
    const u64 imm = value.GetImmediateAsU64();
    ASSERT(imm < 0x100000000);
    return u32(imm);
}

u64 Argument::GetImmediateS32() const {
    ASSERT(FitsInImmediateS32());
    return value.GetImmediateAsU64();
}

u64 Argument::GetImmediateU64() const {
    return value.GetImmediateAsU64();
}

IR::Cond Argument::GetImmediateCond() const {
    ASSERT(IsImmediate() && GetType() == IR::Type::Cond);
    return value.GetCond();
}

IR::AccType Argument::GetImmediateAccType() const {
    ASSERT(IsImmediate() && GetType() == IR::Type::AccType);
    return value.GetAccType();
}

bool Argument::IsInGpr() const {
    if (IsImmediate())
        return false;
    return HostLocIsGPR(*reg_alloc.ValueLocation(value.GetInst()));
}

bool Argument::IsInXmm() const {
    if (IsImmediate())
        return false;
    return HostLocIsXMM(*reg_alloc.ValueLocation(value.GetInst()));
}

bool Argument::IsInMemory() const {
    if (IsImmediate())
        return false;
    return HostLocIsSpill(*reg_alloc.ValueLocation(value.GetInst()));
}

RegAlloc::RegAlloc(BlockOfCode& code, HostLocSet gpr_candidates, HostLocSet xmm_candidates)
        : gpr_candidates(gpr_candidates)
        , xmm_candidates(xmm_candidates)
        , code(code) {}

RegAlloc::ArgumentInfo RegAlloc::GetArgumentInfo(IR::Inst* inst) {
    ArgumentInfo ret = {Argument{*this}, Argument{*this}, Argument{*this}, Argument{*this}};
    for (size_t i = 0; i < inst->NumArgs(); i++) {
        const IR::Value arg = inst->GetArg(i);
        ret[i].value = arg;
        if (!arg.IsImmediate() && !IsValuelessType(arg.GetType())) {
            ASSERT_MSG(ValueLocation(arg.GetInst()), "argument must already been defined");
            LocInfo(*ValueLocation(arg.GetInst())).AddArgReference();
        }
    }
    return ret;
}

void RegAlloc::RegisterPseudoOperation(IR::Inst* inst) {
    ASSERT(IsValueLive(inst) || !inst->HasUses());

    for (size_t i = 0; i < inst->NumArgs(); i++) {
        const IR::Value arg = inst->GetArg(i);
        if (!arg.IsImmediate() && !IsValuelessType(arg.GetType())) {
            if (const auto loc = ValueLocation(arg.GetInst())) {
                // May not necessarily have a value (e.g. CMP variant of Sub32).
                LocInfo(*loc).AddArgReference();
            }
        }
    }
}

bool RegAlloc::IsValueLive(IR::Inst* inst) const {
    return !!ValueLocation(inst);
}

Xbyak::Reg64 RegAlloc::UseGpr(Argument& arg) {
    ASSERT(!arg.allocated);
    arg.allocated = true;
    return HostLocToReg64(UseImpl(arg.value, gpr_candidates));
}

Xbyak::Xmm RegAlloc::UseXmm(Argument& arg) {
    ASSERT(!arg.allocated);
    arg.allocated = true;
    return HostLocToXmm(UseImpl(arg.value, xmm_candidates));
}

OpArg RegAlloc::UseOpArg(Argument& arg) {
    return UseGpr(arg);
}

void RegAlloc::Use(Argument& arg, HostLoc host_loc) {
    ASSERT(!arg.allocated);
    arg.allocated = true;
    UseImpl(arg.value, HostLocBit(host_loc));
}

Xbyak::Reg64 RegAlloc::UseScratchGpr(Argument& arg) {
    ASSERT(!arg.allocated);
    arg.allocated = true;
    return HostLocToReg64(UseScratchImpl(arg.value, gpr_candidates));
}

Xbyak::Xmm RegAlloc::UseScratchXmm(Argument& arg) {
    ASSERT(!arg.allocated);
    arg.allocated = true;
    return HostLocToXmm(UseScratchImpl(arg.value, xmm_candidates));
}

void RegAlloc::UseScratch(Argument& arg, HostLoc host_loc) {
    ASSERT(!arg.allocated);
    arg.allocated = true;
    UseScratchImpl(arg.value, HostLocBit(host_loc));
}

void RegAlloc::DefineValue(IR::Inst* inst, const Xbyak::Reg& reg) {
    ASSERT(reg.getKind() == Xbyak::Operand::XMM || reg.getKind() == Xbyak::Operand::REG);
    const auto hostloc = static_cast<HostLoc>(reg.getIdx() + static_cast<size_t>(reg.getKind() == Xbyak::Operand::XMM ? HostLoc::XMM0 : HostLoc::RAX));
    DefineValueImpl(inst, hostloc);
}

void RegAlloc::DefineValue(IR::Inst* inst, Argument& arg) {
    ASSERT(!arg.allocated);
    arg.allocated = true;
    DefineValueImpl(inst, arg.value);
}

void RegAlloc::Release(const Xbyak::Reg& reg) {
    ASSERT(reg.getKind() == Xbyak::Operand::XMM || reg.getKind() == Xbyak::Operand::REG);
    const auto hostloc = static_cast<HostLoc>(reg.getIdx() + static_cast<size_t>(reg.getKind() == Xbyak::Operand::XMM ? HostLoc::XMM0 : HostLoc::RAX));
    LocInfo(hostloc).ReleaseOne();
}

Xbyak::Reg64 RegAlloc::ScratchGpr() {
    return HostLocToReg64(ScratchImpl(gpr_candidates));
}

Xbyak::Reg64 RegAlloc::ScratchGpr(HostLoc desired_location) {
    return HostLocToReg64(ScratchImpl(HostLocBit(desired_location)));
}

Xbyak::Xmm RegAlloc::ScratchXmm() {
    return HostLocToXmm(ScratchImpl(xmm_candidates));
}

Xbyak::Xmm RegAlloc::ScratchXmm(HostLoc desired_location) {
    return HostLocToXmm(ScratchImpl(HostLocBit(desired_location)));
}

HostLoc RegAlloc::UseImpl(IR::Value use_value, HostLocSet desired_locations) {
    if (use_value.IsImmediate()) {
        return LoadImmediate(use_value, ScratchImpl(desired_locations));
    }

    const IR::Inst* use_inst = use_value.GetInst();
    const HostLoc current_location = *ValueLocation(use_inst);
    const size_t max_bit_width = LocInfo(current_location).GetMaxBitWidth();

    const bool can_use_current_location = HostLocIsRegister(current_location) && desired_locations.test(static_cast<size_t>(current_location));
    if (can_use_current_location) {
        LocInfo(current_location).ReadLock();
        return current_location;
    }

    if (LocInfo(current_location).IsLocked()) {
        return UseScratchImpl(use_value, desired_locations);
    }

    const HostLoc destination_location = SelectARegister(desired_locations);
    if (max_bit_width > HostLocBitWidth(destination_location)) {
        return UseScratchImpl(use_value, desired_locations);
    } else if (CanExchange(destination_location, current_location)) {
        Exchange(destination_location, current_location);
    } else {
        MoveOutOfTheWay(destination_location);
        Move(destination_location, current_location);
    }
    LocInfo(destination_location).ReadLock();
    return destination_location;
}

HostLoc RegAlloc::UseScratchImpl(IR::Value use_value, HostLocSet desired_locations) {
    if (use_value.IsImmediate()) {
        return LoadImmediate(use_value, ScratchImpl(desired_locations));
    }

    const IR::Inst* use_inst = use_value.GetInst();
    const HostLoc current_location = *ValueLocation(use_inst);
    const size_t bit_width = GetBitWidth(use_inst->GetType());

    const bool can_use_current_location = HostLocIsRegister(current_location) && desired_locations.test(static_cast<size_t>(current_location));
    if (can_use_current_location && !LocInfo(current_location).IsLocked()) {
        if (!LocInfo(current_location).IsLastUse()) {
            MoveOutOfTheWay(current_location);
        } else {
            LocInfo(current_location).SetLastUse();
        }
        LocInfo(current_location).WriteLock();
        return current_location;
    }

    const HostLoc destination_location = SelectARegister(desired_locations);
    MoveOutOfTheWay(destination_location);
    CopyToScratch(bit_width, destination_location, current_location);
    LocInfo(destination_location).WriteLock();
    return destination_location;
}

HostLoc RegAlloc::ScratchImpl(HostLocSet desired_locations) {
    const HostLoc location = SelectARegister(desired_locations);
    MoveOutOfTheWay(location);
    LocInfo(location).WriteLock();
    return location;
}

void RegAlloc::HostCall(IR::Inst* result_def,
                        std::optional<Argument::copyable_reference> arg0,
                        std::optional<Argument::copyable_reference> arg1,
                        std::optional<Argument::copyable_reference> arg2,
                        std::optional<Argument::copyable_reference> arg3) {
    constexpr size_t args_count = 4;
    constexpr std::array<HostLoc, args_count> args_hostloc = {ABI_PARAM1, ABI_PARAM2, ABI_PARAM3, ABI_PARAM4};
    const std::array<std::optional<Argument::copyable_reference>, args_count> args = {arg0, arg1, arg2, arg3};

    static const HostLocSet other_caller_save = [args_hostloc]() {
        HostLocSet result;
        for (auto loc : ABI_ALL_CALLER_SAVE) {
            result |= HostLocBit(loc);
        }
        result.reset(static_cast<size_t>(ABI_RETURN));
        for (auto loc : args_hostloc) {
            result.reset(static_cast<size_t>(loc));
        }
        return result;
    }();

    ScratchGpr(ABI_RETURN);
    if (result_def) {
        DefineValueImpl(result_def, ABI_RETURN);
    }

    for (size_t i = 0; i < args_count; i++) {
        if (args[i] && !args[i]->get().IsVoid()) {
            UseScratch(*args[i], args_hostloc[i]);

            // LLVM puts the burden of zero-extension of 8 and 16 bit values on the caller instead of the callee
            const Xbyak::Reg64 reg = HostLocToReg64(args_hostloc[i]);
            switch (args[i]->get().GetType()) {
            case IR::Type::U8:
                code.movzx(reg.cvt32(), reg.cvt8());
                break;
            case IR::Type::U16:
                code.movzx(reg.cvt32(), reg.cvt16());
                break;
            case IR::Type::U32:
                code.mov(reg.cvt32(), reg.cvt32());
                break;
            default:
                break;  // Nothing needs to be done
            }
        }
    }

    for (size_t i = 0; i < args_count; i++) {
        if (!args[i]) {
            // TODO: Force spill
            ScratchGpr(args_hostloc[i]);
        }
    }

    for (size_t index = 0; index < other_caller_save.size(); ++index) {
        if (other_caller_save.test(index)) {
            ScratchImpl(HostLocBit(static_cast<HostLoc>(index)));
        }
    }
}

void RegAlloc::AllocStackSpace(size_t stack_space) {
    ASSERT(stack_space < static_cast<size_t>(std::numeric_limits<s32>::max()));
    ASSERT(reserved_stack_space == 0);
    reserved_stack_space = stack_space;
    code.sub(code.rsp, static_cast<u32>(stack_space));
}

void RegAlloc::ReleaseStackSpace(size_t stack_space) {
    ASSERT(stack_space < static_cast<size_t>(std::numeric_limits<s32>::max()));
    ASSERT(reserved_stack_space == stack_space);
    reserved_stack_space = 0;
    code.add(code.rsp, static_cast<u32>(stack_space));
}

void RegAlloc::EndOfAllocScope() {
    // Unchanged locations already closed their references in an earlier scope.
    // Operand admission, moves, aliases and scratches all pass through LocInfo;
    // release only those locations instead of writing the entire spill table
    // after every emitted IR instruction.
    for (size_t word = 0; word < scope_locations.size(); ++word) {
        auto remaining = std::exchange(scope_locations[word], 0);
        while (remaining != 0) {
            const auto index = word * 64 + std::countr_zero(remaining);
            remaining &= remaining - 1;
            hostloc_info[index].ReleaseAll();
        }
    }
}

void RegAlloc::EmitVerboseDebuggingOutput() {
    for (size_t i = 0; i < hostloc_info.size(); i++) {
        hostloc_info[i].EmitVerboseDebuggingOutput(code, i);
    }
}

HostLoc RegAlloc::SelectARegister(HostLocSet desired_locations) const {
    // Keep candidates in a register mask, with small inline value lists, like
    // Eden. Allocating and partitioning a vector for every IR operand dominates
    // cold block emission and is unnecessary for this fixed register file.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/dynarmic/src/dynarmic/backend/x64/reg_alloc.cpp
    HostLoc occupied = HostLoc::FirstSpill;
    auto remaining = desired_locations.to_ullong();
    while (remaining != 0) {
        const auto loc = static_cast<HostLoc>(std::countr_zero(remaining));
        remaining &= remaining - 1;
        const auto& info = LocInfo(loc);
        if (info.IsLocked()) {
            continue;
        }
        if (info.IsEmpty()) {
            return loc;
        }
        if (occupied == HostLoc::FirstSpill) {
            occupied = loc;
        }
    }
    ASSERT_MSG(occupied != HostLoc::FirstSpill, "All candidate registers have already been allocated");
    return occupied;
}

void RegAlloc::DefineValueImpl(IR::Inst* def_inst, HostLoc host_loc) {
    ASSERT_MSG(!ValueLocation(def_inst), "def_inst has already been defined");
    LocInfo(host_loc).AddValue(host_loc, def_inst);
}

void RegAlloc::DefineValueImpl(IR::Inst* def_inst, const IR::Value& use_inst) {
    ASSERT_MSG(!ValueLocation(def_inst), "def_inst has already been defined");

    if (use_inst.IsImmediate()) {
        const HostLoc location = ScratchImpl(gpr_candidates);
        DefineValueImpl(def_inst, location);
        LoadImmediate(use_inst, location);
        return;
    }

    ASSERT_MSG(ValueLocation(use_inst.GetInst()), "use_inst must already be defined");
    const HostLoc location = *ValueLocation(use_inst.GetInst());
    DefineValueImpl(def_inst, location);
}

HostLoc RegAlloc::LoadImmediate(IR::Value imm, HostLoc host_loc) {
    ASSERT_MSG(imm.IsImmediate(), "imm is not an immediate");

    if (HostLocIsGPR(host_loc)) {
        const Xbyak::Reg64 reg = HostLocToReg64(host_loc);
        const u64 imm_value = imm.GetImmediateAsU64();
        if (imm_value == 0) {
            code.xor_(reg.cvt32(), reg.cvt32());
        } else {
            code.mov(reg, imm_value);
        }
        return host_loc;
    }

    if (HostLocIsXMM(host_loc)) {
        const Xbyak::Xmm reg = HostLocToXmm(host_loc);
        const u64 imm_value = imm.GetImmediateAsU64();
        if (imm_value == 0) {
            MAYBE_AVX(xorps, reg, reg);
        } else {
            MAYBE_AVX(movaps, reg, code.Const(code.xword, imm_value));
        }
        return host_loc;
    }

    UNREACHABLE();
}

void RegAlloc::Move(HostLoc to, HostLoc from) {
    const size_t bit_width = LocInfo(from).GetMaxBitWidth();

    ASSERT(LocInfo(to).IsEmpty() && !LocInfo(from).IsLocked());
    ASSERT(bit_width <= HostLocBitWidth(to));

    if (LocInfo(from).IsEmpty()) {
        return;
    }

    EmitMove(bit_width, to, from);

    LocInfo(to).MoveFrom(LocInfo(from));
    LocInfo(to).SetLocation(to);
}

void RegAlloc::CopyToScratch(size_t bit_width, HostLoc to, HostLoc from) {
    ASSERT(LocInfo(to).IsEmpty() && !LocInfo(from).IsEmpty());

    EmitMove(bit_width, to, from);
}

void RegAlloc::Exchange(HostLoc a, HostLoc b) {
    ASSERT(!LocInfo(a).IsLocked() && !LocInfo(b).IsLocked());
    ASSERT(LocInfo(a).GetMaxBitWidth() <= HostLocBitWidth(b));
    ASSERT(LocInfo(b).GetMaxBitWidth() <= HostLocBitWidth(a));

    if (LocInfo(a).IsEmpty()) {
        Move(a, b);
        return;
    }

    if (LocInfo(b).IsEmpty()) {
        Move(b, a);
        return;
    }

    EmitExchange(a, b);

    LocInfo(a).ExchangeWith(LocInfo(b));
    LocInfo(a).SetLocation(a);
    LocInfo(b).SetLocation(b);
}

void RegAlloc::MoveOutOfTheWay(HostLoc reg) {
    ASSERT(!LocInfo(reg).IsLocked());
    if (!LocInfo(reg).IsEmpty()) {
        SpillRegister(reg);
    }
}

void RegAlloc::SpillRegister(HostLoc loc) {
    ASSERT_MSG(HostLocIsRegister(loc), "Only registers can be spilled");
    ASSERT_MSG(!LocInfo(loc).IsEmpty(), "There is no need to spill unoccupied registers");
    ASSERT_MSG(!LocInfo(loc).IsLocked(), "Registers that have been allocated must not be spilt");

    const HostLoc new_loc = FindFreeSpill();
    Move(new_loc, loc);
}

HostLoc RegAlloc::FindFreeSpill() const {
    for (size_t i = static_cast<size_t>(HostLoc::FirstSpill); i < hostloc_info.size(); i++) {
        const auto loc = static_cast<HostLoc>(i);
        if (LocInfo(loc).IsEmpty()) {
            return loc;
        }
    }

    ASSERT_FALSE("All spill locations are full");
}

void RegAlloc::EmitMove(size_t bit_width, HostLoc to, HostLoc from) {
    if (HostLocIsXMM(to) && HostLocIsXMM(from)) {
        MAYBE_AVX(movaps, HostLocToXmm(to), HostLocToXmm(from));
    } else if (HostLocIsGPR(to) && HostLocIsGPR(from)) {
        ASSERT(bit_width != 128);
        if (bit_width == 64) {
            code.mov(HostLocToReg64(to), HostLocToReg64(from));
        } else {
            code.mov(HostLocToReg64(to).cvt32(), HostLocToReg64(from).cvt32());
        }
    } else if (HostLocIsXMM(to) && HostLocIsGPR(from)) {
        ASSERT(bit_width != 128);
        if (bit_width == 64) {
            MAYBE_AVX(movq, HostLocToXmm(to), HostLocToReg64(from));
        } else {
            MAYBE_AVX(movd, HostLocToXmm(to), HostLocToReg64(from).cvt32());
        }
    } else if (HostLocIsGPR(to) && HostLocIsXMM(from)) {
        ASSERT(bit_width != 128);
        if (bit_width == 64) {
            MAYBE_AVX(movq, HostLocToReg64(to), HostLocToXmm(from));
        } else {
            MAYBE_AVX(movd, HostLocToReg64(to).cvt32(), HostLocToXmm(from));
        }
    } else if (HostLocIsXMM(to) && HostLocIsSpill(from)) {
        const Xbyak::Address spill_addr = SpillToOpArg(from);
        ASSERT(spill_addr.getBit() >= bit_width);
        switch (bit_width) {
        case 128:
            MAYBE_AVX(movaps, HostLocToXmm(to), spill_addr);
            break;
        case 64:
            MAYBE_AVX(movsd, HostLocToXmm(to), spill_addr);
            break;
        case 32:
        case 16:
        case 8:
            MAYBE_AVX(movss, HostLocToXmm(to), spill_addr);
            break;
        default:
            UNREACHABLE();
        }
    } else if (HostLocIsSpill(to) && HostLocIsXMM(from)) {
        const Xbyak::Address spill_addr = SpillToOpArg(to);
        ASSERT(spill_addr.getBit() >= bit_width);
        switch (bit_width) {
        case 128:
            MAYBE_AVX(movaps, spill_addr, HostLocToXmm(from));
            break;
        case 64:
            MAYBE_AVX(movsd, spill_addr, HostLocToXmm(from));
            break;
        case 32:
        case 16:
        case 8:
            MAYBE_AVX(movss, spill_addr, HostLocToXmm(from));
            break;
        default:
            UNREACHABLE();
        }
    } else if (HostLocIsGPR(to) && HostLocIsSpill(from)) {
        ASSERT(bit_width != 128);
        if (bit_width == 64) {
            code.mov(HostLocToReg64(to), SpillToOpArg(from));
        } else {
            code.mov(HostLocToReg64(to).cvt32(), SpillToOpArg(from));
        }
    } else if (HostLocIsSpill(to) && HostLocIsGPR(from)) {
        ASSERT(bit_width != 128);
        if (bit_width == 64) {
            code.mov(SpillToOpArg(to), HostLocToReg64(from));
        } else {
            code.mov(SpillToOpArg(to), HostLocToReg64(from).cvt32());
        }
    } else {
        ASSERT_FALSE("Invalid RegAlloc::EmitMove");
    }
}

void RegAlloc::EmitExchange(HostLoc a, HostLoc b) {
    if (HostLocIsGPR(a) && HostLocIsGPR(b)) {
        code.xchg(HostLocToReg64(a), HostLocToReg64(b));
    } else if (HostLocIsXMM(a) && HostLocIsXMM(b)) {
        ASSERT_FALSE("Check your code: Exchanging XMM registers is unnecessary");
    } else {
        ASSERT_FALSE("Invalid RegAlloc::EmitExchange");
    }
}

Xbyak::Address RegAlloc::SpillToOpArg(HostLoc loc) {
    ASSERT(HostLocIsSpill(loc));

    size_t i = static_cast<size_t>(loc) - static_cast<size_t>(HostLoc::FirstSpill);
    ASSERT_MSG(i < SpillCount, "Spill index greater than number of available spill locations");

    using namespace Xbyak::util;
    return xword[rsp + reserved_stack_space + ABI_SHADOW_SPACE + offsetof(StackLayout, spill) + i * sizeof(StackLayout::spill[0])];
}

}  // namespace Dynarmic::Backend::X64
