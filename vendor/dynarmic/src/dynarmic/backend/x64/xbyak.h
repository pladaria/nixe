/* This file is part of the dynarmic project.
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <boost/unordered_map.hpp>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

// Label definitions and live pointers need no stable container references.
// Keep them in flat tables, as Eden does, while unresolved branches retain the
// multimap needed for multiple references to one label. Prime growth also
// distributes aligned label pointers without depending on their low bits.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/dynarmic/src/dynarmic/backend/x64/xbyak.h
#define XBYAK_STD_UNORDERED_SET tsl::robin_pg_set
#define XBYAK_STD_UNORDERED_MAP tsl::robin_pg_map
#define XBYAK_STD_UNORDERED_MULTIMAP boost::unordered_multimap
#include <xbyak/xbyak.h>
#include <xbyak/xbyak_util.h>
