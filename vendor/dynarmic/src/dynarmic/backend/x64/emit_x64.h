/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <bit>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <boost/container/stable_vector.hpp>
#include <boost/container_hash/hash.hpp>
#include <mcl/bitsizeof.hpp>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include "dynarmic/backend/exception_handler.h"
#include "dynarmic/backend/x64/deferred_emit.h"
#include "dynarmic/backend/x64/reg_alloc.h"
#include "dynarmic/backend/x64/xbyak.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/ir/location_descriptor.h"
#include "dynarmic/ir/terminal.h"

namespace Dynarmic::IR {
class Block;
class Inst;
}  // namespace Dynarmic::IR

namespace Dynarmic {
enum class OptimizationFlag : u32;
}  // namespace Dynarmic

namespace Dynarmic::Backend::X64 {

class BlockOfCode;

using A64FullVectorWidth = std::integral_constant<size_t, 128>;

// Generated memory wrappers form a fixed domain: ordering, width and two host
// register indices. Direct indexing avoids an ordered-tree search per emitted
// load/store; the generated code and the fault handlers remain the same.
using MemoryFallbacks = std::array<std::array<std::array<std::array<void (*)(), 16>, 16>, 5>, 2>;

inline size_t MemoryFallbackSizeIndex(size_t bitsize) {
    DEBUG_ASSERT(bitsize >= 8 && bitsize <= 128 && std::has_single_bit(bitsize));
    return static_cast<size_t>(std::countr_zero(bitsize) - 3);
}

// Array alias that always sizes itself according to the given type T
// relative to the size of a vector register. e.g. T = u32 would result
// in a std::array<u32, 4>.
template<typename T>
using VectorArray = std::array<T, A64FullVectorWidth::value / mcl::bitsizeof<T>>;

template<typename T>
using HalfVectorArray = std::array<T, A64FullVectorWidth::value / mcl::bitsizeof<T> / 2>;

struct EmitContext {
    EmitContext(RegAlloc& reg_alloc, IR::Block& block, boost::container::stable_vector<Xbyak::Label>& labels, boost::container::stable_vector<DeferredEmit>& deferred_emits);
    virtual ~EmitContext();

    void EraseInstruction(IR::Inst* inst);

    virtual FP::FPCR FPCR(bool fpcr_controlled = true) const = 0;

    virtual bool HasOptimization(OptimizationFlag flag) const = 0;

    Xbyak::Label* GenLabel() {
        return &labels.emplace_back();
    }

    RegAlloc& reg_alloc;
    IR::Block& block;

    boost::container::stable_vector<DeferredEmit>& deferred_emits;
    boost::container::stable_vector<Xbyak::Label>& labels;
};

class EmitX64 {
public:
    struct BlockDescriptor {
        CodePtr entrypoint;  // Entrypoint of emitted code
        size_t size;         // Length in bytes of emitted code
    };

    explicit EmitX64(BlockOfCode& code, HostLocSet gpr_candidates);
    virtual ~EmitX64();

    /// Looks up an emitted host block in the cache.
    std::optional<BlockDescriptor> GetBasicBlock(IR::LocationDescriptor descriptor) const;

    /// Empties the entire cache.
    virtual void ClearCache();

    /// Invalidates a selection of basic blocks.
    void InvalidateBasicBlocks(const tsl::robin_set<IR::LocationDescriptor>& locations);

protected:
    using DoNotFastmemMarker = std::tuple<IR::LocationDescriptor, unsigned>;
    struct FastmemMarkerHash {
        size_t operator()(const DoNotFastmemMarker& marker) const noexcept {
            size_t seed = std::hash<IR::LocationDescriptor>{}(std::get<0>(marker));
            boost::hash_combine(seed, std::get<1>(marker));
            return seed;
        }
    };

    // Microinstruction emitters
#define OPCODE(name, type, ...) void Emit##name(EmitContext& ctx, IR::Inst* inst);
#define A32OPC(...)
#define A64OPC(...)
#include "dynarmic/ir/opcodes.inc"
#undef OPCODE
#undef A32OPC
#undef A64OPC

    // Helpers
    virtual std::string LocationDescriptorToFriendlyName(const IR::LocationDescriptor&) const = 0;
    void EmitAddCycles(size_t cycles);
    Xbyak::Label EmitCond(IR::Cond cond);
    BlockDescriptor RegisterBlock(const IR::LocationDescriptor& location_descriptor, CodePtr entrypoint, size_t size);
    void PushRSBHelper(Xbyak::Reg64 loc_desc_reg, Xbyak::Reg64 index_reg, IR::LocationDescriptor target);

    void EmitVerboseDebuggingOutput(RegAlloc& reg_alloc);

    // Terminal instruction emitters
    void EmitTerminal(const IR::Terminal& terminal, IR::LocationDescriptor initial_location, bool is_single_step);
    void EmitTerminal(const IR::Term::LeafTerminal& terminal, IR::LocationDescriptor initial_location, bool is_single_step);
    virtual void EmitTerminalImpl(IR::Term::Interpret terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::ReturnToDispatch terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::LinkBlock terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::LinkBlockFast terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::PopRSBHint terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::FastDispatchHint terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::If terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::CheckBit terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;
    virtual void EmitTerminalImpl(IR::Term::CheckHalt terminal, IR::LocationDescriptor initial_location, bool is_single_step) = 0;

    // Patching
    struct PatchInformation {
        std::vector<CodePtr> jg;
        std::vector<CodePtr> jz;
        std::vector<CodePtr> jmp;
        std::vector<CodePtr> mov_rcx;
    };
    void Patch(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr);
    virtual void Unpatch(const IR::LocationDescriptor& target_desc);
    virtual void EmitPatchJg(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) = 0;
    virtual void EmitPatchJz(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) = 0;
    virtual void EmitPatchJmp(const IR::LocationDescriptor& target_desc, CodePtr target_code_ptr = nullptr) = 0;
    virtual void EmitPatchMovRcx(CodePtr target_code_ptr = nullptr) = 0;

    // State
    BlockOfCode& code;
    // Completed IR scopes release all values and locks. Reuse the allocator and
    // its value-list capacity across blocks, as in Eden's resident allocator.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/dynarmic/src/dynarmic/backend/x64/a64_emit_x64.h
    RegAlloc reg_alloc;
    // Deferred branches borrow stable labels until the block has been emitted.
    // Keep their storage between blocks, as in Eden's native code emitter.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/dynarmic/src/dynarmic/backend/x64/emit_x64.h
    boost::container::stable_vector<Xbyak::Label> labels;
    // Preserve Eden's ordered deferred pass; retain closure storage alongside
    // its labels rather than allocating std::function captures per access.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/dynarmic/src/dynarmic/backend/x64/a64_emit_x64.cpp
    boost::container::stable_vector<DeferredEmit> deferred_emits;
    ExceptionHandler exception_handler;
    tsl::robin_map<IR::LocationDescriptor, BlockDescriptor> block_descriptors;
    tsl::robin_map<IR::LocationDescriptor, PatchInformation> patch_information;
};

}  // namespace Dynarmic::Backend::X64
