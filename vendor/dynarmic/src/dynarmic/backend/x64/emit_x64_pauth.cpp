// SPDX-License-Identifier: 0BSD
#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/emit_x64.h"
#include "dynarmic/common/crypto/qarma.h"
#include "dynarmic/ir/microinstruction.h"

namespace Dynarmic::Backend::X64 {
void EmitX64::EmitComputePAC(EmitContext& ctx, IR::Inst* inst) {
    auto args = ctx.reg_alloc.GetArgumentInfo(inst);
    ctx.reg_alloc.HostCall(inst, args[0], args[1], args[2], args[3]);
    code.CallFunction(&Common::Crypto::ComputePAC);
}
}  // namespace Dynarmic::Backend::X64
