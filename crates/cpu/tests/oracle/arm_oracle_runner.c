// SPDX-License-Identifier: GPL-3.0-or-later
// Shared A64/A32/T32 oracle runner compiled for QEMU user-mode tests.
// The initial protocol exposes ADDS and will grow by instruction family.

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint64_t parse(const char *text) {
    char *end = NULL;
    const uint64_t value = strtoull(text, &end, 16);
    if (end == text || *end != '\0') {
        fprintf(stderr, "invalid hexadecimal operand: %s\n", text);
        exit(2);
    }
    return value;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s LHS RHS\n", argv[0]);
        return 2;
    }

#if defined(__aarch64__)
    const uint64_t lhs = parse(argv[1]);
    const uint64_t rhs = parse(argv[2]);
    uint64_t result;
    uint64_t nzcv;
    __asm__ volatile(
        "adds %0, %2, %3\n\t"
        "mrs %1, nzcv"
        : "=&r"(result), "=r"(nzcv)
        : "r"(lhs), "r"(rhs)
        : "cc");
    printf("arch=a64 profile=armv8-a result=%016" PRIx64 " flags=%08" PRIx32 "\n",
           result, (uint32_t)nzcv & UINT32_C(0xf0000000));
#elif defined(__arm__)
    const uint32_t lhs = (uint32_t)parse(argv[1]);
    const uint32_t rhs = (uint32_t)parse(argv[2]);
    uint32_t result;
    uint32_t apsr;
    __asm__ volatile(
        "adds %0, %2, %3\n\t"
        "mrs %1, apsr"
        : "=&r"(result), "=r"(apsr)
        : "r"(lhs), "r"(rhs)
        : "cc");
#if defined(__thumb__)
    const char *arch = "t32";
#else
    const char *arch = "a32";
#endif
    printf("arch=%s profile=armv8-a result=%08" PRIx32 " flags=%08" PRIx32 "\n",
           arch, result, apsr & UINT32_C(0xf0000000));
#else
#error "This oracle must be cross-compiled for Arm"
#endif
    return 0;
}
