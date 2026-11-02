/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <boost/icl/interval_map.hpp>
#include <boost/icl/interval_set.hpp>
#include <tsl/robin_set.h>

#include "dynarmic/ir/location_descriptor.h"

namespace Dynarmic::Backend {

namespace detail {
// Boost ICL's default codomain combiner assumes operator+=. Flat hash sets
// express the same union directly, without an ordered tree per interval.
template<typename Set>
struct BlockRangeUnion : boost::icl::identity_based_inplace_combine<Set> {
    void operator()(Set& target, const Set& incoming) const {
        target.insert(incoming.begin(), incoming.end());
    }
};
}  // namespace detail

template<typename ProgramCounterType>
class BlockRangeInformation {
public:
    void AddRange(boost::icl::discrete_interval<ProgramCounterType> range, IR::LocationDescriptor location);
    void ClearCache();
    tsl::robin_set<IR::LocationDescriptor> InvalidateRanges(const boost::icl::interval_set<ProgramCounterType>& ranges);

private:
    // Eden uses a flat hash set for each interval's compilation contexts.
    // Prime buckets also distribute descriptors sharing aligned PCs/FPCR bits.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/dynarmic/src/dynarmic/backend/block_range_information.h
    using Locations = tsl::robin_pg_set<IR::LocationDescriptor>;
    boost::icl::interval_map<ProgramCounterType, Locations, boost::icl::partial_absorber, std::less, detail::BlockRangeUnion> block_ranges;
};

}  // namespace Dynarmic::Backend
