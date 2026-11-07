// SPDX-FileCopyrightText: 2011 Google, Inc.
// SPDX-FileContributor: Geoff Pike
// SPDX-FileContributor: Jyrki Alakuijala
// SPDX-License-Identifier: MIT
//! CityHash64, used by Eden's shader bytecode cache.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/common/cityhash.cpp
use std::num::Wrapping as W;
const K0: W<u64> = W(0xc3a5c85c97cb3127);
const K1: W<u64> = W(0xb492b66fbe98f273);
const K2: W<u64> = W(0x9ae16a3b2f90404f);
const KMUL: W<u64> = W(0x9ddfea08eb382d69);
fn fetch64(s: &[u8], at: usize) -> W<u64> {
    W(u64::from_le_bytes(s[at..at + 8].try_into().unwrap()))
}
fn fetch32(s: &[u8], at: usize) -> W<u64> {
    W(u64::from(u32::from_le_bytes(
        s[at..at + 4].try_into().unwrap(),
    )))
}
fn rotate(x: W<u64>, shift: u32) -> W<u64> {
    W(x.0.rotate_right(shift))
}
fn mix(x: W<u64>) -> W<u64> {
    x ^ W(x.0 >> 47)
}
fn hash16(a: W<u64>, b: W<u64>, mul: W<u64>) -> W<u64> {
    let a = mix((a ^ b) * mul);
    mix((b ^ a) * mul) * mul
}
fn weak32(s: &[u8], at: usize, mut a: W<u64>, mut b: W<u64>) -> (W<u64>, W<u64>) {
    let w = fetch64(s, at);
    let x = fetch64(s, at + 8);
    let y = fetch64(s, at + 16);
    let z = fetch64(s, at + 24);
    a += w;
    b = rotate(b + a + z, 21);
    let c = a;
    a += x + y;
    b += rotate(a, 44);
    (a + z, b + c)
}
/// Hash exactly the byte span selected by GenericEnvironment::Analyze.
pub fn city_hash64(s: &[u8]) -> u64 {
    let len = s.len();
    let n = W(len as u64);
    if len <= 16 {
        let mul = K2 + n * W(2);
        return if len >= 8 {
            let a = fetch64(s, 0) + K2;
            let b = fetch64(s, len - 8);
            hash16(rotate(b, 37) * mul + a, (rotate(a, 25) + b) * mul, mul).0
        } else if len >= 4 {
            hash16(n + W(fetch32(s, 0).0 << 3), fetch32(s, len - 4), mul).0
        } else if len != 0 {
            let y = W(u64::from(s[0]) + (u64::from(s[len >> 1]) << 8));
            let z = n + W(u64::from(s[len - 1]) << 2);
            (mix((y * K2) ^ (z * K0)) * K2).0
        } else {
            K2.0
        };
    }
    if len <= 32 {
        let mul = K2 + n * W(2);
        let a = fetch64(s, 0) * K1;
        let b = fetch64(s, 8);
        let c = fetch64(s, len - 8) * mul;
        let d = fetch64(s, len - 16) * K2;
        return hash16(
            rotate(a + b, 43) + rotate(c, 30) + d,
            a + rotate(b + K2, 18) + c,
            mul,
        )
        .0;
    }
    if len <= 64 {
        let mul = K2 + n * W(2);
        let mut a = fetch64(s, 0) * K2;
        let mut b = fetch64(s, 8);
        let c = fetch64(s, len - 24);
        let d = fetch64(s, len - 32);
        let e = fetch64(s, 16) * K2;
        let f = fetch64(s, 24) * W(9);
        let g = fetch64(s, len - 8);
        let h = fetch64(s, len - 16) * mul;
        let u = rotate(a + g, 43) + (rotate(b, 30) + c) * W(9);
        let v = ((a + g) ^ d) + f + W(1);
        let w = W(((u + v) * mul).0.swap_bytes()) + h;
        let x = rotate(e + f, 42) + c;
        let y = (W(((v + w) * mul).0.swap_bytes()) + g) * mul;
        let z = e + f + c;
        a = W(((x + z) * mul + y).0.swap_bytes()) + b;
        b = mix((z + a) * mul + d + h) * mul;
        return (b + x).0;
    }
    let mut x = fetch64(s, len - 40);
    let mut y = fetch64(s, len - 16) + fetch64(s, len - 56);
    let mut z = hash16(fetch64(s, len - 48) + n, fetch64(s, len - 24), KMUL);
    let mut v = weak32(s, len - 64, n, z);
    let mut w = weak32(s, len - 32, y + K1, x);
    x = x * K1 + fetch64(s, 0);
    let end = (len - 1) & !63;
    let mut at = 0;
    while at < end {
        x = rotate(x + y + v.0 + fetch64(s, at + 8), 37) * K1;
        y = rotate(y + v.1 + fetch64(s, at + 48), 42) * K1;
        x ^= w.1;
        y += v.0 + fetch64(s, at + 40);
        z = rotate(z + w.0, 33) * K1;
        v = weak32(s, at, v.1 * K1, x + w.0);
        w = weak32(s, at + 32, z + w.1, y + fetch64(s, at + 16));
        std::mem::swap(&mut z, &mut x);
        at += 64;
    }
    hash16(
        hash16(v.0, w.0, KMUL) + mix(y) * K1 + z,
        hash16(v.1, w.1, KMUL) + x,
        KMUL,
    )
    .0
}
