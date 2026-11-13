// SPDX-License-Identifier: 0BSD
#include "dynarmic/frontend/A64/translate/impl/impl.h"

namespace Dynarmic::A64 {

bool TranslatorVisitor::HasPAuth() const {
    return options.pointer_authentication && options.pointer_authentication->address_bits != 0;
}

IR::U64 TranslatorVisitor::StripPAC(const IR::U64& pointer) {
    const unsigned bits = options.pointer_authentication->address_bits;
    // The selector remains bit 55 even when top-byte-ignore is disabled.
    const auto extension = ir.ArithmeticShiftRight(ir.LogicalShiftLeft(pointer, ir.Imm8(8)), ir.Imm8(63));
    const u64 mask = ~((u64{1} << bits) - 1);
    return ir.Or(ir.And(pointer, ir.Imm64(~mask)), ir.And(extension, ir.Imm64(mask)));
}

IR::U64 TranslatorVisitor::PointerAuth(const IR::U64& pointer, const IR::U64& modifier, unsigned key, bool authenticate) {
    const auto& config = *options.pointer_authentication;
    const u64 low = (u64{1} << config.address_bits) - 1;
    const u64 selector = u64{1} << 55;
    const u64 mask = ~(low | selector);
    // FEAT_PAuth2: PAC insertion/removal uses XOR. Authentication failure leaves
    // a noncanonical address; this CPU has neither FPAC nor FPACCOMBINED.
    // ARM pseudocode: InsertPAC(), Auth(), Strip().
    const auto extension = ir.ArithmeticShiftRight(pointer, ir.Imm8(63));
    const IR::U64 original = authenticate ? StripPAC(pointer) : IR::U64{ir.Or(ir.And(pointer, ir.Imm64(low)), ir.And(extension, ir.Imm64(~low)))};
    const auto pac = ir.ComputePAC(original, modifier, ir.Imm64(config.keys[key][1]), ir.Imm64(config.keys[key][0]));
    const auto result = ir.Eor(pointer, ir.And(pac, ir.Imm64(mask)));
    if (authenticate) {
        return result;
    }
    return ir.Or(ir.And(result, ir.Imm64(~selector)), ir.And(extension, ir.Imm64(selector)));
}

bool TranslatorVisitor::PAuthHint(unsigned immediate) {
    if (immediate == 7) {
        X(64, Reg::LR, StripPAC(X(64, Reg::LR)));
    } else if (immediate < 24) {
        X(64, Reg::R17, PointerAuth(X(64, Reg::R17), X(64, Reg::R16), (immediate >> 1) & 1, immediate >= 12));
    } else {
        const IR::U64 modifier = (immediate & 1) ? IR::U64{SP(64)} : ir.Imm64(0);
        X(64, Reg::LR, PointerAuth(X(64, Reg::LR), modifier, (immediate >> 1) & 1, immediate >= 28));
    }
    return true;
}

bool TranslatorVisitor::PACAUT_reg(bool Z, Imm<3> op, Reg Rn, Reg Rd) {
    if (!HasPAuth())
        return InterpretThisInstruction();
    if (Z && Rn != Reg::R31)
        return UnallocatedEncoding();
    const IR::U64 modifier = Z ? ir.Imm64(0) : Rn == Reg::SP ? IR::U64{SP(64)}
                                                             : IR::U64{X(64, Rn)};
    const auto operation = op.ZeroExtend<unsigned>();
    X(64, Rd, PointerAuth(X(64, Rd), modifier, operation & 3, operation >= 4));
    return true;
}

bool TranslatorVisitor::XPAC_reg([[maybe_unused]] bool D, Reg Rd) {
    if (!HasPAuth())
        return InterpretThisInstruction();
    X(64, Rd, StripPAC(X(64, Rd)));
    return true;
}

bool TranslatorVisitor::PACGA(Reg Rm, Reg Rn, Reg Rd) {
    if (!HasPAuth())
        return InterpretThisInstruction();
    const auto& key = options.pointer_authentication->keys[4];
    const IR::U64 modifier = Rm == Reg::SP ? IR::U64{SP(64)} : IR::U64{X(64, Rm)};
    X(64, Rd, ir.And(ir.ComputePAC(X(64, Rn), modifier, ir.Imm64(key[1]), ir.Imm64(key[0])), ir.Imm64(0xffffffff00000000)));
    return true;
}

bool TranslatorVisitor::RETA(bool M) {
    if (!HasPAuth())
        return InterpretThisInstruction();
    ir.SetPC(PointerAuth(X(64, Reg::LR), SP(64), M ? 1 : 0, true));
    ir.SetTerm(IR::Term::PopRSBHint{});
    return false;
}

bool TranslatorVisitor::BRA_BLRA(bool Z, bool L, bool M, Reg Rn, Reg Rm) {
    if (!HasPAuth())
        return InterpretThisInstruction();
    if (!Z && Rm != Reg::R31)
        return UnallocatedEncoding();
    const IR::U64 modifier = !Z ? ir.Imm64(0) : Rm == Reg::SP ? IR::U64{SP(64)}
                                                              : IR::U64{X(64, Rm)};
    const auto target = PointerAuth(X(64, Rn), modifier, M ? 1 : 0, true);
    if (L) {
        X(64, Reg::LR, ir.Imm64(ir.PC() + 4));
        ir.PushRSB(ir.current_location->AdvancePC(4));
    }
    ir.SetPC(target);
    ir.SetTerm(IR::Term::FastDispatchHint{});
    return false;
}

bool TranslatorVisitor::LDRA(bool M, bool S, Imm<9> imm9, bool W, Reg Rn, Reg Rt) {
    if (!HasPAuth())
        return InterpretThisInstruction();
    // Writeback overlap is constrained-unpredictable, matching other loads.
    if (W && Rn == Rt && Rn != Reg::SP)
        return UnpredictableInstruction();
    const IR::U64 base = Rn == Reg::SP ? IR::U64{SP(64)} : IR::U64{X(64, Rn)};
    const s64 offset = concatenate(Imm<1>{S ? 1u : 0u}, imm9).SignExtend<s64>() * 8;
    const auto address = ir.Add(PointerAuth(base, ir.Imm64(0), M ? 3 : 2, true), ir.Imm64(static_cast<u64>(offset)));
    const auto value = Mem(address, 8, IR::AccType::NORMAL);
    if (W) {
        if (Rn == Reg::SP)
            SP(64, address);
        else
            X(64, Rn, address);
    }
    X(64, Rt, value);
    return true;
}
}  // namespace Dynarmic::A64
