/* This file is part of the dynarmic project.
 * SPDX-License-Identifier: 0BSD
 */
#pragma once

#include <mcl/hash/xmrx.hpp>
#include <tsl/robin_set.h>

#include "dynarmic/ir/location_descriptor.h"

namespace Dynarmic::Backend {

// Power-of-two buckets must use upper PC/context bits as well as low address bits.
// The same policy covers block ownership, incoming links and invalidation sets.
struct CodeCacheHash {
    size_t operator()(IR::LocationDescriptor location) const noexcept {
        return (*this)(location.Value());
    }
    size_t operator()(u64 value) const noexcept {
        return mcl::hash::xmrx(value);
    }
};

using LocationDescriptorSet = tsl::robin_set<IR::LocationDescriptor, CodeCacheHash>;

}  // namespace Dynarmic::Backend
