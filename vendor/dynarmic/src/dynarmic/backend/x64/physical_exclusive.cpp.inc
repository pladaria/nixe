/* Physical reservation protocol shared with Nixe canonical memory.
 * SPDX-License-Identifier: 0BSD
 *
 * ARM reservations track intervening physical writes, including ABA, rather
 * than just expected value equality. See DDI0487, Synchronization and semaphores:
 * https://developer.arm.com/documentation/ddi0487/latest/
 *
 * The host contract is x86 TSO plus aligned atomic version words. The sparse
 * table is initialized under execution exclusion, and retained physical pages
 * outlive every generated access. Never call a fault callback with a word locked.
 */

namespace {
void LockPhysicalReservation(BlockOfCode& code, Xbyak::Reg64 monitor) {
    Xbyak::Label retry, acquired;
    code.L(retry);
    code.lock();
    code.bts(qword[monitor], 0);
    code.jnc(acquired);
    code.pause();
    code.jmp(retry);
    code.L(acquired);
}

void LookupPhysicalReservation(BlockOfCode& code, Xbyak::Reg64 vaddr,
                               Xbyak::Reg64 monitor, Xbyak::Reg64 tmp) {
    code.mov(tmp, vaddr);
    code.shr(tmp, 12);
    code.mov(monitor, qword[r14 + tmp * 8]);
}

void SelectPhysicalGranule(BlockOfCode& code, Xbyak::Reg64 vaddr,
                          Xbyak::Reg64 monitor, Xbyak::Reg64 tmp) {
    code.mov(tmp, vaddr);
    code.and_(tmp, 0xfc0);
    code.shr(tmp, 3);
    code.lea(monitor, ptr[monitor + tmp + 16]);
}
}

template<size_t bitsize>
void A64EmitX64::EmitPhysicalMemoryWrite(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const bool ordered = IsOrdered(args[3].GetImmediateAccType());
    if (ordered && bitsize == 128) {
        ctx.reg_alloc.ScratchGpr(HostLoc::RAX);
        ctx.reg_alloc.ScratchGpr(HostLoc::RBX);
        ctx.reg_alloc.ScratchGpr(HostLoc::RCX);
        ctx.reg_alloc.ScratchGpr(HostLoc::RDX);
    }
    const auto vaddr = ctx.reg_alloc.UseGpr(args[1]);
    const int value = bitsize == 128 ? ctx.reg_alloc.UseXmm(args[2]).getIdx()
        : (ordered ? ctx.reg_alloc.UseScratchGpr(args[2]).getIdx() : ctx.reg_alloc.UseGpr(args[2]).getIdx());
    const auto monitor = ctx.reg_alloc.ScratchGpr();
    const auto tmp = ctx.reg_alloc.ScratchGpr();
    const auto marker = ShouldFastmem(ctx, inst);
    const auto fallback = write_fallbacks.at({ordered, bitsize, vaddr.getIdx(), value});
    const auto abort = GenSharedLabel(), locked_fault = GenSharedLabel(), end = GenSharedLabel();
    const auto tracked = GenSharedLabel();
    bool handling;
    const auto destination = EmitFastmemVAddr(code, ctx, *abort, vaddr, handling, tmp);
    // Complete cross-page writes on the checked path, before publishing bytes.
    if constexpr (bitsize > 8) {
        code.mov(tmp, vaddr);
        code.and_(tmp, 0xfff);
        code.cmp(tmp, 4096 - bitsize / 8);
        code.ja(*abort, code.T_NEAR);
    }
    LookupPhysicalReservation(code, vaddr, monitor, tmp);
    code.sub(monitor, 1);
    code.jnz(*tracked, code.T_NEAR);
    const auto untracked_location = EmitWriteMemoryMov<bitsize>(code, destination, value, ordered);
    code.L(*end);
    ctx.deferred_emits.emplace_back([=, this, &ctx] {
        // Keep the untracked RAM path contiguous. Only pages that have been
        // used for exclusives enter this out-of-line monitor protocol.
        code.L(*tracked);
        code.add(monitor, 1);
        code.jz(*abort, code.T_NEAR);
        if constexpr (bitsize > 8) {
            code.mov(tmp, vaddr);
            code.and_(tmp, 63);
            code.cmp(tmp, 64 - bitsize / 8);
            code.ja(*abort, code.T_NEAR);
        }
        SelectPhysicalGranule(code, vaddr, monitor, tmp);
        LockPhysicalReservation(code, monitor);
        const auto tracked_location = EmitWriteMemoryMov<bitsize>(code, destination, value, ordered);
        code.add(qword[monitor], 1);
        code.jmp(*end, code.T_NEAR);
        code.L(*locked_fault);
        code.sub(qword[monitor], 1); // faulting store committed no bytes
        code.jmp(reinterpret_cast<const void*>(fallback));
        code.L(*abort);
        code.call(fallback);
        const auto resume = mcl::bit_cast<u64>(code.getCurr());
        if (marker) {
            for (const auto& [location, callback] : {
                     std::pair{tracked_location, locked_fault->getAddress()},
                     std::pair{untracked_location, reinterpret_cast<const u8*>(fallback)}}) {
                fastmem_patch_info.emplace(mcl::bit_cast<u64>(location),
                    FastmemPatchInfo{resume, mcl::bit_cast<u64>(callback), *marker, false});
            }
        }
        EmitCheckMemoryAbort(ctx, inst, end.get());
        code.jmp(*end, code.T_NEAR);
    });
}

template<size_t bitsize, bool write>
void A64EmitX64::EmitPhysicalExclusive(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    if constexpr (write && bitsize == 128) {
        ctx.reg_alloc.ScratchGpr(HostLoc::RAX);
        ctx.reg_alloc.ScratchGpr(HostLoc::RBX);
        ctx.reg_alloc.ScratchGpr(HostLoc::RCX);
        ctx.reg_alloc.ScratchGpr(HostLoc::RDX);
    }
    const auto vaddr = ctx.reg_alloc.UseGpr(args[1]);
    const int value = [&] {
        if constexpr (write) {
            return bitsize == 128 ? ctx.reg_alloc.UseXmm(args[2]).getIdx() : ctx.reg_alloc.UseGpr(args[2]).getIdx();
        } else {
            return bitsize == 128 ? ctx.reg_alloc.ScratchXmm().getIdx() : ctx.reg_alloc.ScratchGpr().getIdx();
        }
    }();
    const auto monitor = ctx.reg_alloc.ScratchGpr();
    const auto tmp = ctx.reg_alloc.ScratchGpr();
    const auto physical = ctx.reg_alloc.ScratchGpr();
    const auto status = write ? ctx.reg_alloc.ScratchGpr().cvt32() : tmp.cvt32();
    const auto abort = GenSharedLabel(), locked_fault = GenSharedLabel(), cold = GenSharedLabel(), end = GenSharedLabel();
    const auto marker = ShouldFastmem(ctx, inst);
    if constexpr (write) {
        code.mov(status, 1);
        code.cmp(code.byte[r15 + offsetof(A64JitState, exclusive_state)], 0);
        code.mov(code.byte[r15 + offsetof(A64JitState, exclusive_state)], 0);
        code.je(*end, code.T_NEAR);
    } else {
        code.mov(code.byte[r15 + offsetof(A64JitState, exclusive_state)], 0);
    }
    bool handling;
    const auto address = EmitFastmemVAddr(code, ctx, *abort, vaddr, handling, tmp);
    code.test(vaddr, bitsize / 8 - 1);
    code.jnz(*abort, code.T_NEAR);
    LookupPhysicalReservation(code, vaddr, monitor, tmp);
    code.cmp(monitor, 1);
    code.jbe(*abort, code.T_NEAR);
    code.mov(tmp, qword[monitor + 8]);
    if constexpr (write) {
        code.cmp(tmp, qword[r15 + offsetof(A64JitState, physical_exclusive_identity)]);
        code.jne(*end, code.T_NEAR);
    } else {
        code.mov(qword[r15 + offsetof(A64JitState, physical_exclusive_identity)], tmp);
    }
    code.mov(physical, qword[monitor]);
    code.mov(tmp, vaddr);
    code.and_(tmp, 0xfff);
    code.add(physical, tmp);
    if constexpr (write) {
        code.cmp(physical, qword[r15 + offsetof(A64JitState, physical_exclusive_address)]);
        code.jne(*end, code.T_NEAR);
        code.cmp(code.byte[r15 + offsetof(A64JitState, physical_exclusive_size)], bitsize / 8);
        code.jne(*end, code.T_NEAR);
    }
    SelectPhysicalGranule(code, vaddr, monitor, tmp);
    LockPhysicalReservation(code, monitor);
    const void* location;
    if constexpr (write) {
        Xbyak::Label valid;
        code.mov(tmp, qword[r15 + offsetof(A64JitState, physical_exclusive_version)]);
        code.add(tmp, 1);
        code.cmp(qword[monitor], tmp);
        code.je(valid);
        code.sub(qword[monitor], 1);
        code.jmp(*end, code.T_NEAR);
        code.L(valid);
        location = EmitWriteMemoryMov<bitsize>(code, address, value, bitsize == 128);
        code.add(qword[monitor], 1);
        code.mov(status, 0);
    } else {
        // Every writer to an activated physical granule uses this monitor.
        // Holding it makes even the paired 128-bit read indivisible relative
        // to those writers. Use a genuine read from the guest alias: a host
        // no-change CMPXCHG would write through observation protection and race
        // the immutable byte borrow of a snapshot.
        location = EmitReadMemoryMov<bitsize>(code, value, address, false);
        code.mov(qword[r15 + offsetof(A64JitState, physical_exclusive_address)], physical);
        code.mov(tmp, qword[monitor]);
        code.sub(tmp, 1);
        code.mov(qword[r15 + offsetof(A64JitState, physical_exclusive_version)], tmp);
        code.mov(code.byte[r15 + offsetof(A64JitState, physical_exclusive_size)], bitsize / 8);
        code.mov(code.byte[r15 + offsetof(A64JitState, exclusive_state)], 1);
        code.sub(qword[monitor], 1);
    }
    code.L(*end);
    if constexpr (write) ctx.reg_alloc.DefineValue(inst, status);
    else if constexpr (bitsize == 128) ctx.reg_alloc.DefineValue(inst, Xbyak::Xmm{value});
    else ctx.reg_alloc.DefineValue(inst, Xbyak::Reg64{value});

    ctx.deferred_emits.emplace_back([=, this, &ctx] {
        code.L(*locked_fault);
        code.sub(qword[monitor], 1);
        code.jmp(*cold, code.T_NEAR);
        code.L(*cold);
        if constexpr (write) ABI_PushCallerSaveRegistersAndAdjustStackExcept(code, HostLocRegIdx(status.getIdx()));
        else ABI_PushCallerSaveRegistersAndAdjustStack(code);
        code.sub(rsp, 32);
        if constexpr (write) {
            if constexpr (bitsize == 128) code.movups(xword[rsp], Xbyak::Xmm{value});
            else { code.mov(qword[rsp], Xbyak::Reg64{value}); code.mov(qword[rsp + 8], 0); }
            code.mov(tmp, qword[r15 + offsetof(A64JitState, physical_exclusive_identity)]);
            code.mov(qword[rsp + 16], tmp);
            code.mov(tmp, qword[r15 + offsetof(A64JitState, physical_exclusive_version)]);
            code.mov(qword[rsp + 24], tmp);
        } else {
            code.pxor(xmm0, xmm0);
            code.movups(xword[rsp], xmm0);
            code.movups(xword[rsp + 16], xmm0);
        }
        code.mov(code.ABI_PARAM2, vaddr);
        code.mov(code.ABI_PARAM1, reinterpret_cast<u64>(&conf));
        code.lea(code.ABI_PARAM3, ptr[rsp]);
        code.CallLambda([](A64::UserConfig& conf, u64 address, u64* data) {
            conf.callbacks->PreparePhysicalExclusive(address, bitsize / 8, write, data);
        });
        if constexpr (write) { code.mov(status, dword[rsp]); code.xor_(status, 1); }
        code.add(rsp, 32);
        if constexpr (write) ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLocRegIdx(status.getIdx()));
        else ABI_PopCallerSaveRegistersAndAdjustStack(code);
        if constexpr (!write && bitsize == 128) code.pxor(Xbyak::Xmm{value}, Xbyak::Xmm{value});
        else if constexpr (!write) code.xor_(Xbyak::Reg32{value}, Xbyak::Reg32{value});
        code.ret();
        code.L(*abort);
        code.call(*cold);
        const auto resume = mcl::bit_cast<u64>(code.getCurr());
        if (marker) fastmem_patch_info.emplace(mcl::bit_cast<u64>(location),
            FastmemPatchInfo{resume, mcl::bit_cast<u64>(locked_fault->getAddress()), *marker, false});
        EmitCheckMemoryAbort(ctx, inst, end.get());
        code.jmp(*end, code.T_NEAR);
    });
}
