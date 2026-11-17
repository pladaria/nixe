/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <vector>

#include <boost/icl/interval_set.hpp>
#include <tsl/robin_set.h>

#include "dynarmic/ir/location_descriptor.h"

namespace Dynarmic::Backend {

template<typename ProgramCounterType>
class BlockRangeInformation {
public:
    void AddRange(boost::icl::discrete_interval<ProgramCounterType> range, IR::LocationDescriptor location);
    void ClearCache();
    tsl::robin_set<IR::LocationDescriptor> InvalidateRanges(const boost::icl::interval_set<ProgramCounterType>& ranges);

private:
    // Compilation appends an extent; the less frequent invalidation scans it.
    // Retain capacity across cache clears without allocating interval/set nodes
    // for every compiled block. Extents may overlap and have distinct contexts.
    struct Range {
        ProgramCounterType first;
        ProgramCounterType last;
        IR::LocationDescriptor location;
    };
    std::vector<Range> block_ranges;
};

}  // namespace Dynarmic::Backend
