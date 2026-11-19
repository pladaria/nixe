// Freestanding compiler intrinsics. x0 is the original destination on return.
.global memcpy, memmove
.type memcpy,%function
.type memmove,%function
memcpy:
memmove:
    mov x3, x0
    sub x4, x0, x1
    cmp x4, x2
    b.lo 3f
1:  cmp x2, #16
    b.lo 2f
    ldr q0, [x1], #16
    str q0, [x3], #16
    sub x2, x2, #16
    b 1b
2:  cbz x2, 6f
    ldrb w4, [x1], #1
    strb w4, [x3], #1
    sub x2, x2, #1
    b 2b
3:  add x1, x1, x2
    add x3, x3, x2
4:  cmp x2, #16
    b.lo 5f
    ldr q0, [x1, #-16]!
    str q0, [x3, #-16]!
    sub x2, x2, #16
    b 4b
5:  cbz x2, 6f
    ldrb w4, [x1, #-1]!
    strb w4, [x3, #-1]!
    sub x2, x2, #1
    b 5b
6:  ret

.global memset
.type memset,%function
memset:
    mov x3, x0
    dup v0.16b, w1
1:  cmp x2, #16
    b.lo 2f
    str q0, [x3], #16
    sub x2, x2, #16
    b 1b
2:  cbz x2, 3f
    strb w1, [x3], #1
    sub x2, x2, #1
    b 2b
3:  ret
