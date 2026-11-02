/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <variant>

#include <mcl/stdint.hpp>

#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/frontend/A64/translate/a64_translate.h"
#include "dynarmic/interface/A64/config.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/opt/passes.h"

namespace Dynarmic::Optimization {

void A64MergeInterpretBlocksPass(IR::Block& block, A64::UserCallbacks* cb) {
    const auto is_interpret_instruction = [cb](A64::LocationDescriptor location) {
        const auto instruction = cb->MemoryReadCode(location.PC());
        if (!instruction)
            return false;

        IR::Block new_block{location};
        A64::TranslateSingleInstruction(new_block, location, *instruction);

        if (!new_block.Instructions().empty())
            return false;

        if (const auto* leaf = std::get_if<IR::Term::LeafTerminal>(&new_block.GetTerminal())) {
            if (const auto* term = std::get_if<IR::Term::Interpret>(leaf)) {
                return term->next == location;
            }
        }

        return false;
    };

    const auto* leaf = std::get_if<IR::Term::LeafTerminal>(&block.GetTerminal());
    const auto* term = leaf ? std::get_if<IR::Term::Interpret>(leaf) : nullptr;
    if (!term)
        return;

    A64::LocationDescriptor location{term->next};
    size_t num_instructions = 1;

    while (is_interpret_instruction(location.AdvancePC(static_cast<int>(num_instructions * 4)))) {
        num_instructions++;
    }

    auto merged = *term;
    merged.num_instructions = num_instructions;
    block.ReplaceTerminal(merged);
    block.CycleCount() += num_instructions - 1;
}

}  // namespace Dynarmic::Optimization
