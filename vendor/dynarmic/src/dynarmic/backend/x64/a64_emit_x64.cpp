// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */
// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "dynarmic/backend/x64/a64_emit_x64.h"

#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <mcl/assert.hpp>
#include <mcl/scope_exit.hpp>
#include <mcl/stdint.hpp>
#include <mcl/type_traits/integer_of_size.hpp>

#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/backend/x64/abi.h"
#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/devirtualize.h"
#include "dynarmic/backend/x64/emit_x64.h"
#include "dynarmic/backend/x64/nzcv_util.h"
#include "dynarmic/backend/x64/perf_map.h"
#include "dynarmic/backend/x64/stack_layout.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/frontend/A64/a64_types.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/cond.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"

// TODO: Have ARM flags in host flags and not have them use up GPR registers unless necessary.
// TODO: Actually implement that proper instruction selector you've always wanted to sweetheart.

namespace Dynarmic::Backend::X64 {

using namespace Xbyak::util;

// JITs on one arena share faulted sites, as on ARM64 in Lichii.
// https://github.com/zortos293/Lichii/pull/1861
class FastmemFaults {
public:
    static std::shared_ptr<FastmemFaults> ForArena(uintptr_t arena) {
        static std::mutex registry_mutex;
        static std::unordered_map<uintptr_t, std::weak_ptr<FastmemFaults>> registry;
        std::scoped_lock lock{registry_mutex};
        std::erase_if(registry, [](const auto& entry) { return entry.second.expired(); });
        auto& slot = registry[arena];
        auto faults = slot.lock();
        if (!faults) {
            faults = std::make_shared<FastmemFaults>();
            slot = faults;
        }
        return faults;
    }

    FastmemFaultSites Sites(IR::LocationDescriptor block) const {
        std::shared_lock lock{mutex};
        const auto found = records.find(block);
        return found == records.end() ? FastmemFaultSites{} : found->second;
    }

    void Remember(IR::LocationDescriptor block, unsigned instruction) {
        std::unique_lock lock{mutex};
        auto& sites = records[block];
        if (std::find(sites.begin(), sites.end(), instruction) == sites.end()) {
            sites.push_back(instruction);
        }
    }

private:
    mutable std::shared_mutex mutex;
    tsl::robin_map<IR::LocationDescriptor, FastmemFaultSites, CodeCacheHash> records;
};

void A64EmitX64::RememberFastmemFault(const DoNotFastmemMarker& marker) {
    fastmem_faults->Remember(std::get<0>(marker), std::get<1>(marker));
}

A64EmitContext::A64EmitContext(const A64::UserConfig& conf, RegAlloc& reg_alloc, IR::Block& block, boost::container::stable_vector<Xbyak::Label>& labels, boost::container::stable_vector<DeferredEmit>& deferred_emits, boost::container::stable_vector<DeferredEmit>& cold_emits)
        : EmitContext(reg_alloc, block, labels, deferred_emits, cold_emits), conf(conf) {}

// Eden Auto omits the fastmem width check globally. Retain that direct operand
// only when the block's SSA expression proves the complete starting address.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/arm/dynarmic/arm_dynarmic_64.cpp
bool A64EmitContext::CanOmitFastmemBoundsCheck(IR::Value address) const {
    if (address.IsEmpty()) {
        return false;
    }
    const u64 limit = u64{1} << conf.fastmem_address_space_bits;
    u64 offset = 0;
    IR::Inst* root = nullptr;
    for (unsigned depth = 0; depth < 16; ++depth) {
        if (address.IsImmediate()) {
            return address.GetImmediateAsU64() + offset < limit;
        }
        root = address.GetInstRecursive();
        if (root->GetOpcode() != IR::Opcode::Add64 || !root->GetArg(2).IsZero()) {
            break;
        }
        if (root->GetArg(1).IsImmediate()) {
            offset += root->GetArg(1).GetImmediateAsU64();
            address = root->GetArg(0);
        } else if (root->GetArg(0).IsImmediate()) {
            offset += root->GetArg(0).GetImmediateAsU64();
            address = root->GetArg(1);
        } else {
            break;
        }
        root = nullptr;
    }
    if (!root) {
        return false;
    }
    std::optional<u64> maximum;
    if (root->GetOpcode() == IR::Opcode::ZeroExtendWordToLong) {
        maximum = 0xffff'ffff;
    } else if (root->GetOpcode() == IR::Opcode::And64) {
        for (size_t index : {0, 1}) {
            if (root->GetArg(index).IsImmediate()) {
                maximum = root->GetArg(index).GetImmediateAsU64();
            }
        }
    }
    if (maximum && *maximum < limit && offset < limit - *maximum) {
        return true;
    }
    return false;
}

A64::LocationDescriptor A64EmitContext::Location() const {
    return A64::LocationDescriptor{block.Location()};
}

bool A64EmitContext::IsSingleStep() const {
    return Location().SingleStepping();
}

FP::FPCR A64EmitContext::FPCR(bool fpcr_controlled) const {
    return fpcr_controlled ? Location().FPCR() : Location().FPCR().ASIMDStandardValue();
}

A64EmitX64::A64EmitX64(BlockOfCode& code, A64::UserConfig conf, A64::Jit* jit_interface)
        : EmitX64(
            code, [&conf] {
                HostLocSet gprs{any_gpr};
                if (conf.page_table) {
                    gprs.reset(static_cast<size_t>(HostLoc::R14));
                }
                if (conf.fastmem_pointer) {
                    gprs.reset(static_cast<size_t>(HostLoc::R13));
                }
                return gprs;
            }(),
            conf.spill_to_free_gpr,
            conf.scratch_copy_to_free_reg)
        , conf(conf)
        , jit_interface{jit_interface}
        , fastmem_faults{conf.fastmem_pointer ? FastmemFaults::ForArena(*conf.fastmem_pointer) : nullptr} {
    code.elide_redundant_zext = conf.elide_redundant_zext;
    code.reuse_host_flags = conf.reuse_host_flags && !conf.enable_cycle_counting;
    if (conf.far_code) {
        // Memory-abort exits enlarge recovery stubs. Reserve their share up
        // front so the cold region does not clear a half-empty cache.
        const size_t cold_size = (conf.check_halt_on_memory_access || conf.check_memory_callback_abort)
                                   ? conf.code_cache_size / 16 * 7
                                   : conf.code_cache_size / 4;
        code.EnableFarCode(cold_size);
    }
    GenMemory128Accessors();
    GenFastmemFallbacks();
    GenTerminalHandlers();
    code.PreludeComplete();
    ClearFastDispatchTable();

    exception_handler.SetFastmemCallback([this](u64 rip_) {
        return FastmemCallback(rip_);
    });
}

A64EmitX64::~A64EmitX64() = default;

A64EmitX64::BlockDescriptor A64EmitX64::Emit(IR::Block& block) {
    if (conf.very_verbose_debugging_output) {
        std::puts(IR::DumpBlock(block).c_str());
    }

    code.EnableWriting();
    SCOPE_EXIT {
        labels.clear();
        code.DisableWriting();
    };

    A64EmitContext ctx{conf, reg_alloc, block, labels, deferred_emits, cold_emits};
    // Use the same snapshot for address folding and code emission. A fault on
    // another core can update the shared table without blocking this emission.
    if (fastmem_faults) {
        ctx.fastmem_fault_sites = fastmem_faults->Sites(block.Location());
    }

    emitting_block = true;

    if (conf.fold_address_offsets && conf.fastmem_address_space_bits == 64) {
        for (auto& inst : block) {
            inst.SetAddressFolded(false);
            switch (inst.GetOpcode()) {
            case IR::Opcode::A64ReadMemory8:
            case IR::Opcode::A64ReadMemory16:
            case IR::Opcode::A64ReadMemory32:
            case IR::Opcode::A64ReadMemory64:
            case IR::Opcode::A64WriteMemory8:
            case IR::Opcode::A64WriteMemory16:
            case IR::Opcode::A64WriteMemory32:
            case IR::Opcode::A64WriteMemory64:
                break;
            default:
                continue;
            }
            const auto address = inst.GetArg(1);
            if (address.IsImmediate()) {
                continue;
            }
            auto* const add = address.GetInst();
            if (add->GetOpcode() != IR::Opcode::Add64 || add->UseCount() != 1 || add->GetArg(0).IsImmediate() || !add->GetArg(1).IsImmediate() || !add->GetArg(2).IsZero()) {
                continue;
            }
            const auto displacement = static_cast<s64>(add->GetArg(1).GetImmediateAsU64());
            // Fault recovery also needs the inverse displacement to fit in s32.
            if (displacement < -0x7fff'ffff || displacement > 0x7fff'ffff || !ShouldFastmem(ctx, &inst)) {
                continue;
            }
            add->SetAddressFolded(true);
        }
    }

    // Start emitting.
    code.align();
    const u8* const entrypoint = code.getCurr();
    code.nzcv_flags_pos = nullptr;
    code.nzcv_live_pos = nullptr;
    check_bit_in_zf = false;

    ASSERT(block.GetCondition() == IR::Cond::AL);

    for (auto iter = block.begin(); iter != block.end(); ++iter) {
        IR::Inst* inst = &*iter;

        // Call the relevant Emit* member function.
        switch (inst->GetOpcode()) {
#define OPCODE(name, type, ...)            \
    case IR::Opcode::name:                 \
        A64EmitX64::Emit##name(ctx, inst); \
        break;
#define A32OPC(...)
#define A64OPC(name, type, ...)               \
    case IR::Opcode::A64##name:               \
        A64EmitX64::EmitA64##name(ctx, inst); \
        break;
#include "dynarmic/ir/opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC

        default:
            ASSERT_MSG(false, "Invalid opcode: {}", inst->GetOpcode());
            break;
        }

        ctx.reg_alloc.EndOfAllocScope();

        if (conf.very_verbose_debugging_output) {
            EmitVerboseDebuggingOutput(reg_alloc);
        }
    }

    if (conf.enable_cycle_counting) {
        EmitAddCycles(block.CycleCount());
        code.nzcv_live_pos = nullptr;
    }
    EmitX64::EmitTerminal(block.GetTerminal(), ctx.Location().SetSingleStepping(false), ctx.IsSingleStep());
    code.nzcv_flags_pos = nullptr;
    code.nzcv_live_pos = nullptr;
    check_bit_in_zf = false;
    code.int3();
    [[maybe_unused]] const auto mainline_end = code.getCurr<const u8*>();
    if (code.HasFarCode())
        code.SwitchToFarCode();
    const auto cold_begin = code.getCurr<const u8*>();
    FlushPendingExitStubs();
    emitting_block = false;
    for (auto& emit : ctx.cold_emits)
        emit();
    const size_t cold_size = code.getCurr<const u8*>() - cold_begin;
    if (code.HasFarCode())
        code.SwitchToNearCode();
    for (auto& emit : ctx.deferred_emits)
        emit();
    reg_alloc.AssertNoMoreUses();
    code.int3();

    const size_t size = static_cast<size_t>(code.getCurr<const u8*>() - entrypoint);
#ifdef DYNARMIC_JIT_DIAGNOSTICS
    if (conf.diagnostics) {
        conf.diagnostics->mainline_bytes += mainline_end - entrypoint;
        conf.diagnostics->deferred_bytes += size - (mainline_end - entrypoint) + (code.HasFarCode() ? cold_size : 0);
    }
#endif

    const A64::LocationDescriptor descriptor{block.Location()};
    const A64::LocationDescriptor end_location{block.EndLocation()};

    const auto range = boost::icl::discrete_interval<u64>::closed(descriptor.PC(), end_location.PC() - 1);
    block_ranges.AddRange(range, descriptor);

    return RegisterBlock(descriptor, entrypoint, size, code.HasFarCode() ? cold_begin : nullptr, code.HasFarCode() ? cold_size : 0);
}

void A64EmitX64::ClearCache() {
    EmitX64::ClearCache();
    block_ranges.ClearCache();
    ClearFastDispatchTable();
    fastmem_patch_info.clear();
    exit_stubs.clear();
    ASSERT(pending_exit_stubs.empty() && !emitting_block);
}

void A64EmitX64::InvalidateCacheRanges(const boost::icl::interval_set<u64>& ranges) {
    const auto locations = block_ranges.InvalidateRanges(ranges);
#ifdef DYNARMIC_JIT_DIAGNOSTICS
    if (conf.diagnostics) {
        for (const auto descriptor : locations) {
            if (GetBasicBlock(descriptor)) {
                conf.diagnostics->invalidate(conf.diagnostics->context, descriptor.Value(), false);
            }
        }
    }
#endif
    InvalidateBasicBlocks(locations);
}

void A64EmitX64::ClearFastDispatchTable() {
    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        fast_dispatch_table.fill({});
    }
}

void A64EmitX64::GenTerminalHandlers() {
    const auto check_exit = [this] {
        code.cmp(dword[r15 + offsetof(A64JitState, halt_reason)], 0);
        code.jne(code.GetForceReturnFromRunCodeAddress());
        if (conf.enable_cycle_counting) {
            code.cmp(qword[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, cycles_remaining)], 0);
            code.jle(code.GetForceReturnFromRunCodeAddress());
        }
    };
#ifdef DYNARMIC_JIT_DIAGNOSTICS
    const auto count = [this](std::uint64_t A64::Diagnostics::*counter) {
        if (conf.diagnostics) {
            code.mov(rcx, reinterpret_cast<u64>(&(conf.diagnostics->*counter)));
            code.inc(qword[rcx]);
        }
    };
#endif
    // PC ends up in rbp, location_descriptor ends up in rbx
    const auto calculate_location_descriptor = [this] {
        // This calculation has to match up with A64::LocationDescriptor::UniqueHash
        // TODO: Optimization is available here based on known state of fpcr.
        code.mov(rbp, qword[r15 + offsetof(A64JitState, pc)]);
        code.mov(rcx, A64::LocationDescriptor::pc_mask);
        code.and_(rcx, rbp);
        code.mov(ebx, dword[r15 + offsetof(A64JitState, fpcr)]);
        code.and_(ebx, A64::LocationDescriptor::fpcr_mask);
        code.shl(rbx, A64::LocationDescriptor::fpcr_shift);
        code.or_(rbx, rcx);
    };

    Xbyak::Label fast_dispatch_cache_miss, rsb_cache_miss;

    code.align();
    terminal_handler_pop_rsb_hint = code.getCurr<const void*>();
    check_exit();
    calculate_location_descriptor();
    code.mov(eax, dword[r15 + offsetof(A64JitState, rsb_ptr)]);
    code.sub(eax, 1);
    code.and_(eax, u32(A64JitState::RSBPtrMask));
    code.mov(dword[r15 + offsetof(A64JitState, rsb_ptr)], eax);
    code.cmp(rbx, qword[r15 + offsetof(A64JitState, rsb_location_descriptors) + rax * sizeof(u64)]);
    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        code.jne(rsb_cache_miss);
    } else {
        code.jne(code.GetReturnFromRunCodeAddress());
    }
    code.mov(rax, qword[r15 + offsetof(A64JitState, rsb_codeptrs) + rax * sizeof(u64)]);
#ifdef DYNARMIC_JIT_DIAGNOSTICS
    count(&A64::Diagnostics::rsb_hits);
#endif
    code.jmp(rax);
    PerfMapRegister(terminal_handler_pop_rsb_hint, code.getCurr(), "a64_terminal_handler_pop_rsb_hint");

    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        code.align();
        terminal_handler_fast_dispatch_hint = code.getCurr<const void*>();
        check_exit();
        calculate_location_descriptor();
        code.L(rsb_cache_miss);
        code.mov(r12, reinterpret_cast<u64>(fast_dispatch_table.data()));
        code.mov(rbp, rbx);
        if (code.HasHostFeature(HostFeature::SSE42)) {
            code.crc32(rbp, r12);
        }
        code.and_(ebp, fast_dispatch_table_mask);
        code.lea(rbp, ptr[r12 + rbp]);
        for (size_t way = 0; way < fast_dispatch_set_size; ++way) {
            const size_t entry = way * sizeof(FastDispatchEntry);
            Xbyak::Label next_way;
            code.cmp(rbx, qword[rbp + entry + offsetof(FastDispatchEntry, location_descriptor)]);
            code.jne(way + 1 < fast_dispatch_set_size ? next_way : fast_dispatch_cache_miss, code.T_NEAR);
#ifdef DYNARMIC_JIT_DIAGNOSTICS
            count(&A64::Diagnostics::dispatch_hits);
#endif
            code.jmp(ptr[rbp + entry + offsetof(FastDispatchEntry, code_ptr)]);
            code.L(next_way);
        }
        code.L(fast_dispatch_cache_miss);
#ifdef DYNARMIC_JIT_DIAGNOSTICS
        count(&A64::Diagnostics::dispatch_misses);
#endif
        code.LookupBlock();
        // Compilation may clear the code cache and this table. Publish only after
        // it returns; rbp (set), rbx (full tag) and the table allocation stay stable.
        for (size_t way = fast_dispatch_set_size - 1; way > 0; --way) {
            const size_t to = way * sizeof(FastDispatchEntry);
            const size_t from = to - sizeof(FastDispatchEntry);
            code.mov(rcx, qword[rbp + from + offsetof(FastDispatchEntry, location_descriptor)]);
            code.mov(rdx, qword[rbp + from + offsetof(FastDispatchEntry, code_ptr)]);
            code.mov(qword[rbp + to + offsetof(FastDispatchEntry, location_descriptor)], rcx);
            code.mov(qword[rbp + to + offsetof(FastDispatchEntry, code_ptr)], rdx);
        }
        code.mov(qword[rbp + offsetof(FastDispatchEntry, location_descriptor)], rbx);
        code.mov(ptr[rbp + offsetof(FastDispatchEntry, code_ptr)], rax);
        code.jmp(rax);
        PerfMapRegister(terminal_handler_fast_dispatch_hint, code.getCurr(), "a64_terminal_handler_fast_dispatch_hint");

        code.align();
        fast_dispatch_table_lookup = code.getCurr<FastDispatchEntry& (*)(u64)>();
        code.mov(code.ABI_PARAM2, reinterpret_cast<u64>(fast_dispatch_table.data()));
        if (code.HasHostFeature(HostFeature::SSE42)) {
            code.crc32(code.ABI_PARAM1, code.ABI_PARAM2);
        }
        code.and_(code.ABI_PARAM1.cvt32(), fast_dispatch_table_mask);
        code.lea(code.ABI_RETURN, code.ptr[code.ABI_PARAM2 + code.ABI_PARAM1]);
        code.ret();
        PerfMapRegister(fast_dispatch_table_lookup, code.getCurr(), "a64_fast_dispatch_table_lookup");
    }
}

void A64EmitX64::EmitPushRSB(EmitContext& ctx, IR::Inst* inst) {
    if (!conf.HasOptimization(OptimizationFlag::ReturnStackBuffer)) {
        return;
    }

    EmitX64::EmitPushRSB(ctx, inst);
}

void A64EmitX64::EmitA64SetCheckBit(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg8 to_store = ctx.reg_alloc.UseGpr(args[0]).cvt8();
    // No code after this final instruction changes ZF before the CheckBit terminal.
    if (code.reuse_host_flags && !conf.very_verbose_debugging_output && &ctx.block.back() == inst && std::holds_alternative<IR::Term::CheckBit>(ctx.block.GetTerminal())) {
        code.test(to_store, to_store);
        check_bit_in_zf = true;
        return;
    }
    code.mov(code.byte[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, check_bit)], to_store);
}

void A64EmitX64::EmitA64GetCFlag(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr().cvt32();
    code.mov(result, dword[r15 + offsetof(A64JitState, cpsr_nzcv)]);
    code.shr(result, NZCV::x64_c_flag_bit);
    code.and_(result, 1);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetNZCVRaw(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 nzcv_raw = ctx.reg_alloc.ScratchGpr().cvt32();

    code.mov(nzcv_raw, dword[r15 + offsetof(A64JitState, cpsr_nzcv)]);

    if (code.HasHostFeature(HostFeature::FastBMI2)) {
        const Xbyak::Reg32 tmp = ctx.reg_alloc.ScratchGpr().cvt32();
        code.mov(tmp, NZCV::x64_mask);
        code.pext(nzcv_raw, nzcv_raw, tmp);
        code.shl(nzcv_raw, 28);
    } else {
        code.and_(nzcv_raw, NZCV::x64_mask);
        code.imul(nzcv_raw, nzcv_raw, NZCV::from_x64_multiplier);
        code.and_(nzcv_raw, NZCV::arm_mask);
    }

    ctx.reg_alloc.DefineValue(inst, nzcv_raw);
}

void A64EmitX64::EmitA64SetNZCVRaw(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg32 nzcv_raw = ctx.reg_alloc.UseScratchGpr(args[0]).cvt32();

    code.shr(nzcv_raw, 28);
    if (code.HasHostFeature(HostFeature::FastBMI2)) {
        const Xbyak::Reg32 tmp = ctx.reg_alloc.ScratchGpr().cvt32();
        code.mov(tmp, NZCV::x64_mask);
        code.pdep(nzcv_raw, nzcv_raw, tmp);
    } else {
        code.imul(nzcv_raw, nzcv_raw, NZCV::to_x64_multiplier);
        code.and_(nzcv_raw, NZCV::x64_mask);
    }
    code.mov(dword[r15 + offsetof(A64JitState, cpsr_nzcv)], nzcv_raw);
}

void A64EmitX64::EmitA64SetNZCV(A64EmitContext& ctx, IR::Inst* inst) {
    const bool flags_live = code.reuse_host_flags && code.nzcv_flags_pos != nullptr && code.nzcv_flags_pos == code.getCurr() && !inst->GetArg(0).IsImmediate() && code.nzcv_flags_inst == inst->GetArg(0).GetInst();
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg32 to_store = ctx.reg_alloc.UseScratchGpr(args[0]).cvt32();
    code.mov(dword[r15 + offsetof(A64JitState, cpsr_nzcv)], to_store);
    code.nzcv_live_pos = flags_live ? code.getCurr() : nullptr;
}

void A64EmitX64::EmitA64GetW(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr().cvt32();

    code.mov(result, dword[r15 + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)]);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetX(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr();

    code.mov(result, qword[r15 + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)]);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetS(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = qword[r15 + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
    code.movd(result, addr);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetD(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = qword[r15 + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
    code.movq(result, addr);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetQ(A64EmitContext& ctx, IR::Inst* inst) {
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[r15 + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm result = ctx.reg_alloc.ScratchXmm();
    code.movaps(result, addr);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetSP(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr();
    code.mov(result, qword[r15 + offsetof(A64JitState, sp)]);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetFPCR(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr().cvt32();
    code.mov(result, dword[r15 + offsetof(A64JitState, fpcr)]);
    ctx.reg_alloc.DefineValue(inst, result);
}

static u32 GetFPSRImpl(A64JitState* jit_state) {
    return jit_state->GetFpsr();
}

void A64EmitX64::EmitA64GetFPSR(A64EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.HostCall(inst);
    code.mov(code.ABI_PARAM1, code.r15);
    code.stmxcsr(code.dword[code.r15 + offsetof(A64JitState, guest_MXCSR)]);
    code.CallFunction(GetFPSRImpl);
}

void A64EmitX64::EmitA64SetW(A64EmitContext& ctx, IR::Inst* inst) {
    // Register materialization and stores below preserve host flags.
    const bool flags_live = code.nzcv_live_pos != nullptr && code.nzcv_live_pos == code.getCurr();
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    const auto addr = qword[r15 + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)];
    if (args[1].FitsInImmediateS32()) {
        code.mov(addr, args[1].GetImmediateS32());
    } else if (conf.elide_redundant_zext && !args[1].IsImmediate() && IsZeroExtendedU32Producer(args[1].GetInst())) {
        // Reloading a spilled U32 into a GPR also clears its high bits.
        const auto to_store = ctx.reg_alloc.UseGpr(args[1]);
        code.mov(addr, to_store);
    } else {
        // TODO: zext tracking, xmm variant
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseScratchGpr(args[1]);
        code.mov(to_store.cvt32(), to_store.cvt32());
        code.mov(addr, to_store);
    }
    if (flags_live)
        code.nzcv_live_pos = code.getCurr();
}

void A64EmitX64::EmitA64SetX(A64EmitContext& ctx, IR::Inst* inst) {
    // Register materialization and stores below preserve host flags.
    const bool flags_live = code.nzcv_live_pos != nullptr && code.nzcv_live_pos == code.getCurr();
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Reg reg = inst->GetArg(0).GetA64RegRef();
    const auto addr = qword[r15 + offsetof(A64JitState, reg) + sizeof(u64) * static_cast<size_t>(reg)];
    if (args[1].FitsInImmediateS32()) {
        code.mov(addr, args[1].GetImmediateS32());
    } else if (args[1].IsInXmm()) {
        const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(args[1]);
        code.movq(addr, to_store);
    } else {
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseGpr(args[1]);
        code.mov(addr, to_store);
    }
    if (flags_live)
        code.nzcv_live_pos = code.getCurr();
}

void A64EmitX64::EmitA64SetS(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[r15 + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(args[1]);
    const Xbyak::Xmm tmp = ctx.reg_alloc.ScratchXmm();
    // TODO: Optimize
    code.pxor(tmp, tmp);
    code.movss(tmp, to_store);
    code.movaps(addr, tmp);
}

void A64EmitX64::EmitA64SetD(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[r15 + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm to_store = ctx.reg_alloc.UseScratchXmm(args[1]);
    code.movq(to_store, to_store);  // TODO: Remove when able
    code.movaps(addr, to_store);
}

void A64EmitX64::EmitA64SetQ(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const A64::Vec vec = inst->GetArg(0).GetA64VecRef();
    const auto addr = xword[r15 + offsetof(A64JitState, vec) + sizeof(u64) * 2 * static_cast<size_t>(vec)];

    const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(args[1]);
    code.movaps(addr, to_store);
}

void A64EmitX64::EmitA64SetSP(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const auto addr = qword[r15 + offsetof(A64JitState, sp)];
    if (args[0].FitsInImmediateS32()) {
        code.mov(addr, args[0].GetImmediateS32());
    } else if (args[0].IsInXmm()) {
        const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(args[0]);
        code.movq(addr, to_store);
    } else {
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseGpr(args[0]);
        code.mov(addr, to_store);
    }
}

static void SetFPCRImpl(A64JitState* jit_state, u32 value) {
    jit_state->SetFpcr(value);
}

void A64EmitX64::EmitA64SetFPCR(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.HostCall(nullptr, {}, args[0]);
    code.mov(code.ABI_PARAM1, code.r15);
    code.CallFunction(SetFPCRImpl);
    code.ldmxcsr(code.dword[code.r15 + offsetof(A64JitState, guest_MXCSR)]);
}

static void SetFPSRImpl(A64JitState* jit_state, u32 value) {
    jit_state->SetFpsr(value);
}

void A64EmitX64::EmitA64SetFPSR(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.HostCall(nullptr, {}, args[0]);
    code.mov(code.ABI_PARAM1, code.r15);
    code.CallFunction(SetFPSRImpl);
    code.ldmxcsr(code.dword[code.r15 + offsetof(A64JitState, guest_MXCSR)]);
}

void A64EmitX64::EmitA64SetPC(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const auto addr = qword[r15 + offsetof(A64JitState, pc)];
    if (args[0].FitsInImmediateS32()) {
        code.mov(addr, args[0].GetImmediateS32());
    } else if (args[0].IsInXmm()) {
        const Xbyak::Xmm to_store = ctx.reg_alloc.UseXmm(args[0]);
        code.movq(addr, to_store);
    } else {
        const Xbyak::Reg64 to_store = ctx.reg_alloc.UseGpr(args[0]);
        code.mov(addr, to_store);
    }
}

void A64EmitX64::EmitA64CallSupervisor(A64EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.HostCall(nullptr);
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(args[0].IsImmediate());
    const u32 imm = args[0].GetImmediateU32();
    Devirtualize<&A64::UserCallbacks::CallSVC>(conf.callbacks).EmitCall(code, [&](RegList param) {
        code.mov(param[0], imm);
    });
    // The kernel would have to execute ERET to get here, which would clear exclusive state.
    code.mov(code.byte[r15 + offsetof(A64JitState, exclusive_state)], u8(0));
}

void A64EmitX64::EmitA64CheckMemoryAlignment(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const auto address = ctx.reg_alloc.UseGpr(args[1]);
    Xbyak::Label aligned;
    code.test(address, static_cast<u32>(args[2].GetImmediateU64()));
    code.jz(aligned, code.T_NEAR);
    code.lock();
    code.or_(code.dword[r15 + offsetof(A64JitState, halt_reason)], static_cast<u32>(HaltReason::MemoryAbort));
    code.mov(rax, args[0].GetImmediateU64());
    code.mov(code.qword[r15 + offsetof(A64JitState, pc)], rax);
    code.ForceReturnFromRunCode();
    code.L(aligned);
}

void A64EmitX64::EmitA64ExceptionRaised(A64EmitContext& ctx, IR::Inst* inst) {
    ctx.reg_alloc.HostCall(nullptr);
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ASSERT(args[0].IsImmediate() && args[1].IsImmediate());
    const u64 pc = args[0].GetImmediateU64();
    const u64 exception = args[1].GetImmediateU64();
    Devirtualize<&A64::UserCallbacks::ExceptionRaised>(conf.callbacks).EmitCall(code, [&](RegList param) {
        code.mov(param[0], pc);
        code.mov(param[1], exception);
    });
}

void A64EmitX64::EmitA64DataCacheOperationRaised(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.HostCall(nullptr, {}, args[1], args[2]);
    Devirtualize<&A64::UserCallbacks::DataCacheOperationRaised>(conf.callbacks).EmitCall(code);
}

void A64EmitX64::EmitA64InstructionCacheOperationRaised(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.HostCall(nullptr, {}, args[0], args[1]);
    Devirtualize<&A64::UserCallbacks::InstructionCacheOperationRaised>(conf.callbacks).EmitCall(code);
}

void A64EmitX64::EmitA64DataSynchronizationBarrier(A64EmitContext&, IR::Inst*) {
    code.mfence();
    code.lfence();
}

void A64EmitX64::EmitA64DataMemoryBarrier(A64EmitContext&, IR::Inst*) {
    code.mfence();
}

void A64EmitX64::EmitA64InstructionSynchronizationBarrier(A64EmitContext& ctx, IR::Inst*) {
    if (!conf.hook_isb) {
        return;
    }

    ctx.reg_alloc.HostCall(nullptr);
    Devirtualize<&A64::UserCallbacks::InstructionSynchronizationBarrierRaised>(conf.callbacks).EmitCall(code);
}

void A64EmitX64::EmitA64GetCNTFRQ(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr().cvt32();
    code.mov(result, conf.cntfrq_el0);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetCNTPCT(A64EmitContext& ctx, IR::Inst* inst) {
    if (conf.wall_clock_cntpct && conf.cntpct_tsc_base) {
        const auto low = ctx.reg_alloc.ScratchGpr(HostLoc::RAX);
        const auto high = ctx.reg_alloc.ScratchGpr(HostLoc::RDX);
        const auto address = ctx.reg_alloc.ScratchGpr();
        const auto base = ctx.reg_alloc.ScratchGpr();
        Xbyak::Label retry;
        code.mov(address, reinterpret_cast<std::uintptr_t>(conf.cntpct_tsc_base));
        code.L(retry);
        code.mov(base, qword[address]);
        code.lfence();
        code.rdtsc();
        code.lfence();
        code.shl(high, 32);
        code.or_(low, high);
        code.sub(low, base);
        // Retry if pause or resume changed the origin during the sample.
        code.cmp(base, qword[address]);
        code.jne(retry);
        code.test(base, base);
        code.cmovs(low, base);
        code.btr(low, 63);
        code.mov(high, conf.cntpct_tsc_multiplier);
        code.mul(high);
        ctx.reg_alloc.DefineValue(inst, high);
        return;
    }

    ctx.reg_alloc.HostCall(inst);
    if (!conf.wall_clock_cntpct) {
        code.UpdateTicks();
    }
    Devirtualize<&A64::UserCallbacks::GetCNTPCT>(conf.callbacks).EmitCall(code);
}

void A64EmitX64::EmitA64GetCTR(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr().cvt32();
    code.mov(result, conf.ctr_el0);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetDCZID(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg32 result = ctx.reg_alloc.ScratchGpr().cvt32();
    code.mov(result, conf.dczid_el0);
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetTPIDR(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr();
    if (conf.tpidr_el0) {
        code.mov(result, u64(conf.tpidr_el0));
        code.mov(result, qword[result]);
    } else {
        code.xor_(result.cvt32(), result.cvt32());
    }
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64GetTPIDRRO(A64EmitContext& ctx, IR::Inst* inst) {
    const Xbyak::Reg64 result = ctx.reg_alloc.ScratchGpr();
    if (conf.tpidrro_el0) {
        code.mov(result, u64(conf.tpidrro_el0));
        code.mov(result, qword[result]);
    } else {
        code.xor_(result.cvt32(), result.cvt32());
    }
    ctx.reg_alloc.DefineValue(inst, result);
}

void A64EmitX64::EmitA64SetTPIDR(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const Xbyak::Reg64 value = ctx.reg_alloc.UseGpr(args[0]);
    const Xbyak::Reg64 addr = ctx.reg_alloc.ScratchGpr();
    if (conf.tpidr_el0) {
        code.mov(addr, u64(conf.tpidr_el0));
        code.mov(qword[addr], value);
    }
}

std::string A64EmitX64::LocationDescriptorToFriendlyName(const IR::LocationDescriptor& ir_descriptor) const {
    const A64::LocationDescriptor descriptor{ir_descriptor};
    return fmt::format("a64_{:016X}_fpcr{:08X}",
                       descriptor.PC(),
                       descriptor.FPCR().Value());
}

void A64EmitX64::EmitTerminalImpl(IR::Term::Interpret terminal, IR::LocationDescriptor, bool) {
    code.SwitchMxcsrOnExit();
    Devirtualize<&A64::UserCallbacks::InterpreterFallback>(conf.callbacks).EmitCall(code, [&](RegList param) {
        code.mov(param[0], A64::LocationDescriptor{terminal.next}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], param[0]);
        code.mov(param[1].cvt32(), terminal.num_instructions);
    });
    code.ReturnFromRunCode(true);  // TODO: Check cycles
}

void A64EmitX64::EmitTerminalImpl(IR::Term::ReturnToDispatch, IR::LocationDescriptor, bool) {
    code.ReturnFromRunCode();
}

namespace {
bool UncheckedForwardLink(const A64::UserConfig& conf, IR::LocationDescriptor next, IR::LocationDescriptor initial) {
    const auto from = A64::LocationDescriptor{initial}.PC();
    const auto to = A64::LocationDescriptor{next}.PC();
    // Every cycle contains a backward or indirect edge. Checking page crossings
    // also bounds an acyclic chain to at most 1024 distinct forward block starts,
    // without adding a counter to generated code. Blocking callbacks and cold
    // compilation retain their existing interruption boundaries.
    return conf.halt_check_on_back_edges && !conf.enable_cycle_counting && to > from && (to ^ from) < 4096;
}
}  // namespace

void A64EmitX64::EmitTerminalImpl(IR::Term::LinkBlock terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    if (!conf.HasOptimization(OptimizationFlag::BlockLinking) || is_single_step) {
        code.mov(rax, A64::LocationDescriptor{terminal.next}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        code.ReturnFromRunCode();
        return;
    }

    if (UncheckedForwardLink(conf, terminal.next, initial_location)) {
        EmitTerminalImpl(IR::Term::LinkBlockFast{terminal.next}, initial_location, is_single_step);
        return;
    }

    Xbyak::Label stopped;
    if (conf.enable_cycle_counting) {
        code.cmp(dword[r15 + offsetof(A64JitState, halt_reason)], 0);
        code.jne(stopped, code.T_NEAR);
        code.cmp(qword[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, cycles_remaining)], 0);

        patch_information[terminal.next].jg.push_back(code.getCurr());
        if (const auto next_bb = GetBasicBlock(terminal.next)) {
            EmitPatchJg(terminal.next, next_bb->entrypoint);
        } else {
            EmitPatchJg(terminal.next);
        }
    } else {
        code.cmp(dword[r15 + offsetof(A64JitState, halt_reason)], 0);

        patch_information[terminal.next].jz.push_back(code.getCurr());
        if (const auto next_bb = GetBasicBlock(terminal.next)) {
            EmitPatchJz(terminal.next, next_bb->entrypoint);
        } else {
            EmitPatchJz(terminal.next);
        }
    }

    code.L(stopped);
    EmitUnlinkedExit(terminal.next);
}

void A64EmitX64::EmitTerminalImpl(IR::Term::LinkBlockFast terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    if (conf.halt_check_on_back_edges && !conf.enable_cycle_counting && !UncheckedForwardLink(conf, terminal.next, initial_location)) {
        EmitTerminalImpl(IR::Term::LinkBlock{terminal.next}, initial_location, is_single_step);
        return;
    }
    if (!conf.HasOptimization(OptimizationFlag::BlockLinking) || is_single_step) {
        code.mov(rax, A64::LocationDescriptor{terminal.next}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        code.ReturnFromRunCode();
        return;
    }

    patch_information[terminal.next].jmp.push_back(code.getCurr());
    if (auto next_bb = GetBasicBlock(terminal.next)) {
        EmitPatchJmp(terminal.next, next_bb->entrypoint);
    } else {
        EmitPatchJmp(terminal.next);
    }
}

void A64EmitX64::EmitTerminalImpl(IR::Term::PopRSBHint, IR::LocationDescriptor, bool is_single_step) {
    if (!conf.HasOptimization(OptimizationFlag::ReturnStackBuffer) || is_single_step) {
        code.ReturnFromRunCode();
        return;
    }

    code.jmp(terminal_handler_pop_rsb_hint);
}

void A64EmitX64::EmitTerminalImpl(IR::Term::FastDispatchHint, IR::LocationDescriptor, bool is_single_step) {
    if (!conf.HasOptimization(OptimizationFlag::FastDispatch) || is_single_step) {
        code.ReturnFromRunCode();
        return;
    }

    code.jmp(terminal_handler_fast_dispatch_hint);
}

void A64EmitX64::EmitTerminalImpl(IR::Term::If terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    switch (terminal.if_) {
    case IR::Cond::AL:
    case IR::Cond::NV:
        EmitTerminal(terminal.then_, initial_location, is_single_step);
        break;
    default:
        Xbyak::Label pass = EmitCond(terminal.if_);
        EmitTerminal(terminal.else_, initial_location, is_single_step);
        code.L(pass);
        EmitTerminal(terminal.then_, initial_location, is_single_step);
        break;
    }
}

void A64EmitX64::EmitTerminalImpl(IR::Term::CheckBit terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    Xbyak::Label fail;
    if (check_bit_in_zf) {
        check_bit_in_zf = false;
    } else {
        code.cmp(code.byte[rsp + ABI_SHADOW_SPACE + offsetof(StackLayout, check_bit)], u8(0));
    }
    code.jz(fail);
    EmitTerminal(terminal.then_, initial_location, is_single_step);
    code.L(fail);
    EmitTerminal(terminal.else_, initial_location, is_single_step);
}

void A64EmitX64::EmitTerminalImpl(IR::Term::CheckHalt terminal, IR::LocationDescriptor initial_location, bool is_single_step) {
    code.cmp(dword[r15 + offsetof(A64JitState, halt_reason)], 0);
    code.jne(code.GetForceReturnFromRunCodeAddress());
    EmitTerminal(terminal.else_, initial_location, is_single_step);
}

Xbyak::Label* A64EmitX64::PendingExitStub(const IR::LocationDescriptor& target) {
    for (const auto& [value, label] : pending_exit_stubs) {
        if (value == target.Value())
            return label;
    }
    auto* label = &labels.emplace_back();
    pending_exit_stubs.emplace_back(target.Value(), label);
    return label;
}

void A64EmitX64::FlushPendingExitStubs() {
    for (const auto& [value, label] : pending_exit_stubs) {
        code.L(*label);
        exit_stubs.emplace(value, code.getCurr<CodePtr>());
        code.mov(rax, A64::LocationDescriptor{IR::LocationDescriptor{value}}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        code.ReturnFromRunCode();
    }
    pending_exit_stubs.clear();
}

void A64EmitX64::EmitUnlinkedExit(const IR::LocationDescriptor& target) {
    if (conf.compact_block_exits) {
        if (const auto it = exit_stubs.find(target.Value()); it != exit_stubs.end()) {
            code.jmp(it->second, code.T_NEAR);
        } else {
            ASSERT(emitting_block);
            code.jmp(*PendingExitStub(target), code.T_NEAR);
        }
    } else {
        code.mov(rax, A64::LocationDescriptor{target}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        code.ForceReturnFromRunCode();
    }
}

// Each target owns an epoch-lived stub even when initially linked: invalidation
// can always retarget every fixed-size site without allocating executable code.
void A64EmitX64::EmitCompactPatch(const IR::LocationDescriptor& target, CodePtr target_code_ptr, int condition) {
    const CodePtr patch_location = code.getCurr();
    const auto it = exit_stubs.find(target.Value());
    Xbyak::Label* pending = nullptr;
    if (it == exit_stubs.end()) {
        ASSERT(emitting_block);
        pending = PendingExitStub(target);
    }
    const auto destination = target_code_ptr ? target_code_ptr : (it == exit_stubs.end() ? nullptr : it->second);
    if (condition == 0) {
        if (destination)
            code.jmp(destination, code.T_NEAR);
        else
            code.jmp(*pending, code.T_NEAR);
    } else if (condition == 1) {
        if (destination)
            code.jg(destination);
        else
            code.jg(*pending, code.T_NEAR);
    } else {
        if (destination)
            code.jz(destination);
        else
            code.jz(*pending, code.T_NEAR);
    }
    code.EnsurePatchLocationSize(patch_location, condition == 0 ? 5 : 6);
}

void A64EmitX64::EmitPatchJg(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr) {
    const CodePtr patch_location = code.getCurr();
    if (conf.compact_block_exits) {
        EmitCompactPatch(target_desc, target_code_ptr, 1);
        return;
    }
    if (target_code_ptr) {
        code.jg(target_code_ptr);
    } else {
        code.mov(rax, A64::LocationDescriptor{target_desc}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        code.jg(code.GetReturnFromRunCodeAddress());
    }
    code.EnsurePatchLocationSize(patch_location, 23);
}

void A64EmitX64::EmitPatchJz(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr) {
    const CodePtr patch_location = code.getCurr();
    if (conf.compact_block_exits) {
        EmitCompactPatch(target_desc, target_code_ptr, 2);
        return;
    }
    if (target_code_ptr) {
        code.jz(target_code_ptr);
    } else {
        code.mov(rax, A64::LocationDescriptor{target_desc}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        code.jz(code.GetReturnFromRunCodeAddress());
    }
    code.EnsurePatchLocationSize(patch_location, 23);
}

void A64EmitX64::EmitPatchJmp(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr) {
    const CodePtr patch_location = code.getCurr();
    if (conf.compact_block_exits) {
        EmitCompactPatch(target_desc, target_code_ptr, 0);
        return;
    }
    if (target_code_ptr) {
        code.jmp(target_code_ptr);
    } else {
        code.mov(rax, A64::LocationDescriptor{target_desc}.PC());
        code.mov(qword[r15 + offsetof(A64JitState, pc)], rax);
        code.jmp(code.GetReturnFromRunCodeAddress());
    }
    code.EnsurePatchLocationSize(patch_location, 22);
}

void A64EmitX64::EmitPatchMovRcx(CodePtr target_code_ptr) {
    if (!target_code_ptr) {
        target_code_ptr = code.GetReturnFromRunCodeAddress();
    }
    const CodePtr patch_location = code.getCurr();
    code.mov(code.rcx, reinterpret_cast<u64>(target_code_ptr));
    code.EnsurePatchLocationSize(patch_location, 10);
}

void A64EmitX64::Unpatch(const IR::LocationDescriptor& location) {
    EmitX64::Unpatch(location);
    if (conf.HasOptimization(OptimizationFlag::FastDispatch)) {
        code.DisableWriting();
        auto* set = &(*fast_dispatch_table_lookup)(location.Value());
        for (size_t way = 0; way < fast_dispatch_set_size; ++way) {
            if (set[way].location_descriptor == location.Value())
                set[way] = {};
        }
        code.EnableWriting();
    }
}

}  // namespace Dynarmic::Backend::X64
