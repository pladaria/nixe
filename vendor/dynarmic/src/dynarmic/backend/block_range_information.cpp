/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/block_range_information.h"

#include <algorithm>

#include <boost/icl/interval_set.hpp>
#include <mcl/stdint.hpp>
#include <tsl/robin_set.h>

namespace Dynarmic::Backend {

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::AddRange(boost::icl::discrete_interval<ProgramCounterType> range, IR::LocationDescriptor location) {
    if (!boost::icl::is_empty(range)) {
        block_ranges.push_back({boost::icl::first(range), boost::icl::last(range), location});
    }
}

template<typename ProgramCounterType>
void BlockRangeInformation<ProgramCounterType>::ClearCache() {
    block_ranges.clear();
}

template<typename ProgramCounterType>
tsl::robin_set<IR::LocationDescriptor> BlockRangeInformation<ProgramCounterType>::InvalidateRanges(const boost::icl::interval_set<ProgramCounterType>& ranges) {
    tsl::robin_set<IR::LocationDescriptor> erase_locations;
    if (ranges.empty()) {
        return erase_locations;
    }
    std::erase_if(block_ranges, [&](const Range& range) {
        // Interval-set lookup finds any overlap in O(log M), including batches
        // with many disjoint invalidations, without a temporary range index.
        const auto extent = boost::icl::discrete_interval<ProgramCounterType>::closed(range.first, range.last);
        if (ranges.find(extent) == ranges.end()) {
            return false;
        }
        erase_locations.insert(range.location);
        return true;
    });
    return erase_locations;
}

template class BlockRangeInformation<u32>;
template class BlockRangeInformation<u64>;

}  // namespace Dynarmic::Backend
