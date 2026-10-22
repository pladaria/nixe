/* Physical reservation protocol shared with Nixe canonical memory.
 * SPDX-License-Identifier: 0BSD
 *
 * Exclusive operations retain physical identity and the observed value. Ordinary
 * native stores remain plain fastmem, as in Eden; exclusive commit uses CAS.
 * See DDI0487, Synchronization and semaphores:
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

void LookupPhysicalPage(BlockOfCode& code, Xbyak::Reg64 vaddr,
                        Xbyak::Reg64 page, Xbyak::Reg64 tmp) {
    code.mov(tmp, vaddr);
    code.shr(tmp, 12);
    code.mov(page, qword[r14 + tmp * 8]);
}

template<bool write>
void LookupPhysicalReservation(BlockOfCode& code, Xbyak::Reg64 vaddr,
                               Xbyak::Reg64 monitor, Xbyak::Reg64 tmp) {
    Xbyak::Label tracked, callback, end;
    LookupPhysicalPage(code, vaddr, monitor, tmp);
    if constexpr (write) {
        code.mov(tmp, monitor);
        code.and_(tmp, 0x18);
        code.cmp(tmp, 0x18);
        code.jne(callback);
    } else {
        code.test(monitor.cvt8(), 8);
        code.jz(callback);
    }
    code.test(monitor.cvt8(), 3); // observed or device-dependent
    code.jnz(callback);
    code.test(monitor.cvt8(), 4); // activated physical reservation
    code.jnz(tracked);
    code.mov(monitor, 1);
    code.jmp(end);
    code.L(tracked);
    code.and_(monitor, -64);
    code.jmp(end);
    code.L(callback);
    code.xor_(monitor, monitor);
    code.L(end);
}

void SelectPhysicalGranule(BlockOfCode& code, Xbyak::Reg64 vaddr,
                          Xbyak::Reg64 monitor, Xbyak::Reg64 tmp) {
    code.mov(tmp, vaddr);
    code.and_(tmp, 0xfc0);
    code.shr(tmp, 3);
    code.lea(monitor, ptr[monitor + tmp + 24]);
}
}

// After a protection fault, consult the same canonical PTE used by the
// process. Clean RAM returns to direct native accesses; tracked/device pages
// enter callbacks without repeatedly raising host signals. Eden uses this
// fastmem-to-page-table fallback for both reads and writes:
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/arm/dynarmic/arm_dynarmic_64.cpp
// Sites that have never faulted retain their original plain fastmem access.
template<size_t bitsize, bool write>
void A64EmitX64::EmitPhysicalMemoryAccess(A64EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    const bool ordered = IsOrdered(args[write ? 3 : 2].GetImmediateAccType());
    if constexpr (write && bitsize == 128) {
        if (ordered) {
            ctx.reg_alloc.ScratchGpr(HostLoc::RAX);
            ctx.reg_alloc.ScratchGpr(HostLoc::RBX);
            ctx.reg_alloc.ScratchGpr(HostLoc::RCX);
            ctx.reg_alloc.ScratchGpr(HostLoc::RDX);
        }
    }
    const auto vaddr = ctx.reg_alloc.UseGpr(args[1]);
    const int value = [&] {
        if constexpr (write) {
            return bitsize == 128 ? ctx.reg_alloc.UseXmm(args[2]).getIdx()
                : (ordered ? ctx.reg_alloc.UseScratchGpr(args[2]).getIdx() : ctx.reg_alloc.UseGpr(args[2]).getIdx());
        } else {
            return bitsize == 128 ? ctx.reg_alloc.ScratchXmm().getIdx() : ctx.reg_alloc.ScratchGpr().getIdx();
        }
    }();
    const auto tmp = ctx.reg_alloc.ScratchGpr();
    const auto abort = GenSharedLabel(), end = GenSharedLabel();
    const auto fallback = [&] {
        if constexpr (write) return write_fallbacks.at({ordered, bitsize, vaddr.getIdx(), value});
        else return read_fallbacks.at({ordered, bitsize, vaddr.getIdx(), value});
    }();
    bool handling;
    const auto address = EmitFastmemVAddr(code, ctx, *abort, vaddr, handling, tmp);
    if constexpr (bitsize > 8) {
        code.mov(tmp, vaddr);
        code.and_(tmp, 0xfff);
        code.cmp(tmp, 4096 - bitsize / 8);
        code.ja(*abort, code.T_NEAR);
    }
    code.mov(tmp, vaddr);
    code.shr(tmp, 12);
    code.mov(tmp, qword[r14 + tmp * 8]);
    const void* location;
    if constexpr (write) {
        // Linux fastmem cannot expose write-only guest mappings. Their checked
        // callback uses the canonical physical pointer with guest WRITE rights.
        code.and_(tmp, 0x1b); // READ|WRITE permissions, OBSERVED|DEVICE attributes
        code.cmp(tmp, 0x18);
        code.jne(*abort, code.T_NEAR);
        location = EmitWriteMemoryMov<bitsize>(code, address, value, ordered);
    } else {
        code.test(tmp.cvt8(), 8); // guest READ permission
        code.jz(*abort, code.T_NEAR);
        code.test(tmp.cvt8(), 2); // DEVICE-dependent page
        code.jnz(*abort, code.T_NEAR);
        if constexpr (bitsize == 128) {
            // Ordered-128 loads use a locked write and require the callback on
            // read-only mappings instead of generating a second native fault.
            if (ordered) code.jmp(*abort, code.T_NEAR);
        }
        if (ordered) code.mfence();
        location = EmitReadMemoryMov<bitsize>(code, value, address, false);
    }
    code.L(*end);
    if constexpr (!write && bitsize == 128) ctx.reg_alloc.DefineValue(inst, Xbyak::Xmm{value});
    else if constexpr (!write) ctx.reg_alloc.DefineValue(inst, Xbyak::Reg64{value});
    const auto marker = std::make_tuple(ctx.Location(), inst->GetName());
    ctx.deferred_emits.emplace_back([=, this, &ctx] {
        code.L(*abort);
        code.call(fallback);
        // Tracking can tighten the kernel alias after this instruction loaded
        // its PTE. Attribute that race through the existing native fault map;
        // this site already uses the PTE and needs no further recompilation.
        fastmem_patch_info.emplace(mcl::bit_cast<u64>(location),
            FastmemPatchInfo{mcl::bit_cast<u64>(code.getCurr()), mcl::bit_cast<u64>(fallback), marker, false});
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
    } else if constexpr (write) {
        ctx.reg_alloc.ScratchGpr(HostLoc::RAX);
    }
    const int pair_scratch = write && bitsize == 128 ? ctx.reg_alloc.ScratchXmm().getIdx() : 0;
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
    const auto device_epoch = ctx.reg_alloc.ScratchGpr();
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
    LookupPhysicalReservation<write>(code, vaddr, monitor, tmp);
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
    code.mov(device_epoch, qword[monitor + 16]);
    SelectPhysicalGranule(code, vaddr, monitor, tmp);
    LockPhysicalReservation(code, monitor);
    const void* location;
    if constexpr (write) {
        Xbyak::Label valid;
        code.mov(tmp, qword[r15 + offsetof(A64JitState, physical_exclusive_version)]);
        code.add(tmp, 1);
        code.cmp(qword[monitor], tmp);
        const auto invalid = GenSharedLabel();
        code.jne(*invalid);
        code.mov(tmp, qword[device_epoch]);
        code.cmp(tmp, qword[r15 + offsetof(A64JitState, physical_exclusive_device_version)]);
        code.je(valid);
        code.L(*invalid);
        code.sub(qword[monitor], 1);
        code.jmp(*end, code.T_NEAR);
        code.L(valid);
        // Expected-value CAS is the commit point, including interference from
        // ordinary direct stores which never consult this reservation metadata.
        // Eden's EmitExclusiveWriteMemoryInline:
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/dynarmic/src/dynarmic/backend/x64/emit_x64_memory.cpp.inc
        code.mov(rax, qword[r15 + offsetof(A64JitState, physical_exclusive_expected)]);
        if constexpr (bitsize == 128) {
            code.mov(rdx, qword[r15 + offsetof(A64JitState, physical_exclusive_expected) + 8]);
            code.movq(rbx, Xbyak::Xmm{value});
            code.movaps(Xbyak::Xmm{pair_scratch}, Xbyak::Xmm{value});
            code.punpckhqdq(Xbyak::Xmm{pair_scratch}, Xbyak::Xmm{pair_scratch});
            code.movq(rcx, Xbyak::Xmm{pair_scratch});
        }
        location = code.getCurr();
        code.lock();
        if constexpr (bitsize == 8) code.cmpxchg(code.byte[address], Xbyak::Reg8{value});
        else if constexpr (bitsize == 16) code.cmpxchg(code.word[address], Xbyak::Reg16{value});
        else if constexpr (bitsize == 32) code.cmpxchg(code.dword[address], Xbyak::Reg32{value});
        else if constexpr (bitsize == 64) code.cmpxchg(code.qword[address], Xbyak::Reg64{value});
        else code.cmpxchg16b(code.ptr[address]);
        code.setnz(status.cvt8());
        Xbyak::Label failed, released;
        code.test(status, status);
        code.jnz(failed);
        code.add(qword[monitor], 1);
        code.jmp(released);
        code.L(failed);
        code.sub(qword[monitor], 1);
        code.L(released);
    } else {
        // The load is read-only, including on observed/read-only aliases. The
        // later CAS validates the observed value against direct ordinary stores.
        location = EmitReadMemoryMov<bitsize>(code, value, address, false);
        if constexpr (bitsize == 128) code.movups(xword[r15 + offsetof(A64JitState, physical_exclusive_expected)], Xbyak::Xmm{value});
        else {
            code.mov(qword[r15 + offsetof(A64JitState, physical_exclusive_expected)], Xbyak::Reg64{value});
            code.mov(qword[r15 + offsetof(A64JitState, physical_exclusive_expected) + 8], 0);
        }
        code.mov(qword[r15 + offsetof(A64JitState, physical_exclusive_address)], physical);
        code.mov(tmp, qword[monitor]);
        code.sub(tmp, 1);
        code.mov(qword[r15 + offsetof(A64JitState, physical_exclusive_version)], tmp);
        code.mov(tmp, qword[device_epoch]);
        code.mov(qword[r15 + offsetof(A64JitState, physical_exclusive_device_version)], tmp);
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
        code.sub(rsp, 80);
        if constexpr (write) {
            if constexpr (bitsize == 128) code.movups(xword[rsp], Xbyak::Xmm{value});
            else { code.mov(qword[rsp], Xbyak::Reg64{value}); code.mov(qword[rsp + 8], 0); }
            code.mov(tmp, qword[r15 + offsetof(A64JitState, physical_exclusive_identity)]);
            code.mov(qword[rsp + 16], tmp);
            code.mov(tmp, qword[r15 + offsetof(A64JitState, physical_exclusive_version)]);
            code.mov(qword[rsp + 24], tmp);
            code.mov(tmp, qword[r15 + offsetof(A64JitState, physical_exclusive_device_version)]);
            code.mov(qword[rsp + 32], tmp);
            code.mov(tmp, qword[r15 + offsetof(A64JitState, physical_exclusive_address)]);
            code.mov(qword[rsp + 40], tmp);
            code.movzx(tmp.cvt32(), code.byte[r15 + offsetof(A64JitState, physical_exclusive_size)]);
            code.mov(qword[rsp + 48], tmp);
            code.mov(tmp, qword[r15 + offsetof(A64JitState, physical_exclusive_expected)]);
            code.mov(qword[rsp + 56], tmp);
            code.mov(tmp, qword[r15 + offsetof(A64JitState, physical_exclusive_expected) + 8]);
            code.mov(qword[rsp + 64], tmp);
            code.mov(qword[rsp + 72], 0);
        } else {
            code.pxor(xmm0, xmm0);
            code.movups(xword[rsp], xmm0);
            code.movups(xword[rsp + 16], xmm0);
            code.movups(xword[rsp + 32], xmm0);
            code.movups(xword[rsp + 48], xmm0);
            code.movups(xword[rsp + 64], xmm0);
        }
        code.mov(code.ABI_PARAM2, vaddr);
        code.mov(code.ABI_PARAM1, reinterpret_cast<u64>(&conf));
        code.lea(code.ABI_PARAM3, ptr[rsp]);
        code.CallLambda([](A64::UserConfig& conf, u64 address, u64* data) {
            conf.callbacks->PreparePhysicalExclusive(address, bitsize / 8, write, data);
        });
        if constexpr (write) { code.mov(status, dword[rsp]); code.xor_(status, 1); }
        code.add(rsp, 80);
        if constexpr (write) ABI_PopCallerSaveRegistersAndAdjustStackExcept(code, HostLocRegIdx(status.getIdx()));
        else ABI_PopCallerSaveRegistersAndAdjustStack(code);
        if constexpr (!write && bitsize == 128) code.pxor(Xbyak::Xmm{value}, Xbyak::Xmm{value});
        else if constexpr (!write) code.xor_(Xbyak::Reg32{value}, Xbyak::Reg32{value});
        code.ret();
        code.L(*abort);
        code.call(*cold);
        const auto resume = mcl::bit_cast<u64>(code.getCurr());
        if (marker) fastmem_patch_info.emplace(mcl::bit_cast<u64>(location),
            FastmemPatchInfo{resume, mcl::bit_cast<u64>(locked_fault->getAddress()), *marker, conf.recompile_on_exclusive_fastmem_failure});
        EmitCheckMemoryAbort(ctx, inst, end.get());
        code.jmp(*end, code.T_NEAR);
    });
}
