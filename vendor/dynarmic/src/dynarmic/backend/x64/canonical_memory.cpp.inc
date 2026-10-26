/* Canonical tagged-page fallback for ordinary memory accesses.
 * SPDX-License-Identifier: 0BSD
 */

// After a protection fault, consult the same canonical PTE used by the
// process. Clean RAM returns to direct native accesses; tracked/device pages
// enter callbacks without repeatedly raising host signals. Eden uses this
// fastmem-to-page-table fallback for both reads and writes:
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/arm/dynarmic/arm_dynarmic_64.cpp
// Sites that have never faulted retain their original plain fastmem access.
template<size_t bitsize, bool write>
void A64EmitX64::EmitCanonicalMemoryAccess(A64EmitContext& ctx, IR::Inst* inst) {
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
