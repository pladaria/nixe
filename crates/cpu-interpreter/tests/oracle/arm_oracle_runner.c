// SPDX-License-Identifier: GPL-3.0-or-later
// A64 oracle runner compiled for optional QEMU user-mode tests.

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// QEMU's GDB stub patches this slot and single-steps exactly one instruction.
// Keeping the scratch area in the same static executable gives the Nixe and
// QEMU fixtures stable, equally mapped code and data addresses.
__asm__(
    ".pushsection .nixe_oracle,\"awx\",@progbits\n"
    ".balign 16\n"
    ".global nixe_oracle_slot\n"
    "nixe_oracle_slot:\n"
    ".rept 1024\n"
    ".word 0xd503201f\n"
    ".endr\n"
    ".global nixe_oracle_after\n"
    "nixe_oracle_after:\n"
    ".word 0xd4200000\n"
    ".popsection\n");

__attribute__((aligned(4096), used)) uint8_t nixe_oracle_scratch[4096];

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
    return 0;
}
