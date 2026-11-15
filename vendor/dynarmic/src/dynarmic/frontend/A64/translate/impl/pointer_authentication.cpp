// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// @SIMPLIFICATION Keep pointers unsigned and omit authentication checks. PACGA
// returns a deterministic masked XOR instead of QARMA, avoiding cryptographic work.
// @SPECULATIVE This assumes guests only compare recomputed codes; forged or
// externally signed pointers are not modeled.
// Emit no IR for PAC/AUT/XPAC and use direct branches/loads, without host calls.
// https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/dynarmic/src/dynarmic/frontend/A64/translate/impl/pointer_authentication.cpp
#include "dynarmic/frontend/A64/translate/impl/impl.h"

namespace Dynarmic::A64 {

bool TranslatorVisitor::PACAUT_reg(bool Z, [[maybe_unused]] Imm<3> op, Reg Rn, [[maybe_unused]] Reg Rd) {
    if (!options.enable_pauth)
        return InterpretThisInstruction();
    if (Z && Rn != Reg::R31)
        return UnallocatedEncoding();
    return true;
}

bool TranslatorVisitor::XPAC_reg([[maybe_unused]] bool D, [[maybe_unused]] Reg Rd) {
    if (!options.enable_pauth)
        return InterpretThisInstruction();
    return true;
}

bool TranslatorVisitor::PACGA(Reg Rm, Reg Rn, Reg Rd) {
    if (!options.enable_pauth)
        return InterpretThisInstruction();
    const IR::U64 modifier = Rm == Reg::SP ? IR::U64{SP(64)} : IR::U64{X(64, Rm)};
    X(64, Rd, ir.And(ir.Eor(X(64, Rn), modifier), ir.Imm64(0xffffffff00000000)));
    return true;
}

bool TranslatorVisitor::RETA([[maybe_unused]] bool M) {
    if (!options.enable_pauth)
        return InterpretThisInstruction();
    ir.SetPC(X(64, Reg::LR));
    ir.SetTerm(IR::Term::PopRSBHint{});
    return false;
}

bool TranslatorVisitor::BRA_BLRA(bool Z, bool L, [[maybe_unused]] bool M, Reg Rn, Reg Rm) {
    if (!options.enable_pauth)
        return InterpretThisInstruction();
    if (!Z && Rm != Reg::R31)
        return UnallocatedEncoding();
    const auto target = X(64, Rn);
    if (L) {
        X(64, Reg::LR, ir.Imm64(ir.PC() + 4));
        ir.PushRSB(ir.current_location->AdvancePC(4));
    }
    ir.SetPC(target);
    ir.SetTerm(IR::Term::FastDispatchHint{});
    return false;
}

bool TranslatorVisitor::LDRA([[maybe_unused]] bool M, bool S, Imm<9> imm9, bool W, Reg Rn, Reg Rt) {
    if (!options.enable_pauth)
        return InterpretThisInstruction();
    // Writeback overlap is constrained-unpredictable, matching other loads.
    if (W && Rn == Rt && Rn != Reg::SP)
        return UnpredictableInstruction();
    const IR::U64 base = Rn == Reg::SP ? IR::U64{SP(64)} : IR::U64{X(64, Rn)};
    const s64 offset = concatenate(Imm<1>{S ? 1u : 0u}, imm9).SignExtend<s64>() * 8;
    const auto address = ir.Add(base, ir.Imm64(static_cast<u64>(offset)));
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
