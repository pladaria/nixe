#include "tz.h"

#include <bit>
#include <cstring>

// The service ABI is little-endian; host calendar conversion stays in Rust.
static_assert(std::endian::native == std::endian::little);

extern "C" int nixe_timezone_parse(const uint8_t* data, size_t size, uint8_t* output) {
    Tz::Rule rule{};
    // Padding is part of the serialized guest ABI too.
    std::memset(&rule, 0, sizeof(rule));
    const int result = Tz::ParseTimeZoneBinary(rule, {data, size});
    if (result == 0) {
        std::memcpy(output, &rule, sizeof(rule));
    }
    return result;
}
