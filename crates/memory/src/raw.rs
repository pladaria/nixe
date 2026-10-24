//! Guest RAM has no racing Rust references. These instruction routines permit
//! overlapping mixed-width guest accesses, with tearing across bulk words.
//! Callers retain allocation/mapping lifetime and resolve tracking/authority.
use core::arch::asm;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("native RAM instruction primitives require an x86-64 or AArch64 host");

#[cfg(target_arch = "x86_64")]
unsafe fn load(address: *const u8, width: usize) -> u64 {
    let value: u64;
    unsafe {
        match width {
            1 => {
                asm!("movzx {value:e}, byte ptr [{address}]", value=out(reg) value, address=in(reg) address, options(nostack, preserves_flags, readonly))
            }
            2 => {
                asm!("movzx {value:e}, word ptr [{address}]", value=out(reg) value, address=in(reg) address, options(nostack, preserves_flags, readonly))
            }
            4 => {
                asm!("mov {value:e}, dword ptr [{address}]", value=out(reg) value, address=in(reg) address, options(nostack, preserves_flags, readonly))
            }
            8 => {
                asm!("mov {value}, qword ptr [{address}]", value=out(reg) value, address=in(reg) address, options(nostack, preserves_flags, readonly))
            }
            _ => unreachable!(),
        }
    }
    value
}
#[cfg(target_arch = "x86_64")]
unsafe fn store(address: *mut u8, value: u64, width: usize) {
    unsafe {
        match width {
            1 => {
                asm!("mov byte ptr [{address}], {value:l}", address=in(reg) address, value=in(reg) value, options(nostack, preserves_flags))
            }
            2 => {
                asm!("mov word ptr [{address}], {value:x}", address=in(reg) address, value=in(reg) value, options(nostack, preserves_flags))
            }
            4 => {
                asm!("mov dword ptr [{address}], {value:e}", address=in(reg) address, value=in(reg) value, options(nostack, preserves_flags))
            }
            8 => {
                asm!("mov qword ptr [{address}], {value}", address=in(reg) address, value=in(reg) value, options(nostack, preserves_flags))
            }
            _ => unreachable!(),
        }
    }
}
// Host instruction reference: https://developer.arm.com/documentation/ddi0602/latest/
#[cfg(target_arch = "aarch64")]
unsafe fn load(address: *const u8, width: usize) -> u64 {
    let value: u64;
    unsafe {
        match width {
            1 => {
                asm!("ldrb {value:w}, [{address}]", value=out(reg) value, address=in(reg) address, options(nostack, preserves_flags, readonly))
            }
            2 => {
                asm!("ldrh {value:w}, [{address}]", value=out(reg) value, address=in(reg) address, options(nostack, preserves_flags, readonly))
            }
            4 => {
                asm!("ldr {value:w}, [{address}]", value=out(reg) value, address=in(reg) address, options(nostack, preserves_flags, readonly))
            }
            8 => {
                asm!("ldr {value}, [{address}]", value=out(reg) value, address=in(reg) address, options(nostack, preserves_flags, readonly))
            }
            _ => unreachable!(),
        }
    }
    value
}
#[cfg(target_arch = "aarch64")]
unsafe fn store(address: *mut u8, value: u64, width: usize) {
    unsafe {
        match width {
            1 => {
                asm!("strb {value:w}, [{address}]", address=in(reg) address, value=in(reg) value, options(nostack, preserves_flags))
            }
            2 => {
                asm!("strh {value:w}, [{address}]", address=in(reg) address, value=in(reg) value, options(nostack, preserves_flags))
            }
            4 => {
                asm!("str {value:w}, [{address}]", address=in(reg) address, value=in(reg) value, options(nostack, preserves_flags))
            }
            8 => {
                asm!("str {value}, [{address}]", address=in(reg) address, value=in(reg) value, options(nostack, preserves_flags))
            }
            _ => unreachable!(),
        }
    }
}
fn width(address: usize, remaining: usize) -> usize {
    if remaining >= 8 && address.is_multiple_of(8) {
        8
    } else if remaining >= 4 && address.is_multiple_of(4) {
        4
    } else if remaining >= 2 && address.is_multiple_of(2) {
        2
    } else {
        1
    }
}
#[cfg(target_arch = "x86_64")]
unsafe fn words(source: *const u8, destination: *mut u8, count: usize) {
    // The System V ABI keeps DF clear. REP MOVSQ performs eight-byte transfers
    // without expressing guest bytes as a Rust allocation/reference or memcpy.
    unsafe {
        asm!("rep movsq", inout("rsi") source => _, inout("rdi") destination => _, inout("rcx") count => _, options(nostack, preserves_flags));
    }
}
#[cfg(target_arch = "aarch64")]
unsafe fn words(source: *const u8, destination: *mut u8, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        asm!("2:", "ldr {value}, [{source}], #8", "str {value}, [{destination}], #8", "subs {count}, {count}, #1", "b.ne 2b", source=inout(reg) source=>_, destination=inout(reg) destination=>_, count=inout(reg) count=>_, value=out(reg) _, options(nostack));
    }
}
/// Copy disjoint live RAM without creating Rust references to either range.
///
/// # Safety
/// Both ranges are retained and admitted; source coherence and destination
/// tracking are resolved. The byte ranges must not overlap.
pub(crate) unsafe fn copy(source: *const u8, destination: *mut u8, bytes: usize) {
    let count = bytes / 8;
    unsafe {
        words(source, destination, count);
    }
    for offset in count * 8..bytes {
        unsafe {
            store(destination.add(offset), load(source.add(offset), 1), 1);
        }
    }
}
/// # Safety
/// The source is admitted guest RAM with stable physical lifetime. Destination
/// is exclusively owned host memory, disjoint from the guest backing.
pub(crate) unsafe fn capture(mut source: *const u8, destination: &mut [u8]) {
    let mut offset = 0;
    while offset < destination.len() {
        let remaining = destination.len() - offset;
        let width = width(source as usize, remaining);
        if width == 8 {
            let bytes = remaining / 8 * 8;
            unsafe {
                words(source, destination.as_mut_ptr().add(offset), bytes / 8);
            }
            source = unsafe { source.add(bytes) };
            offset += bytes;
        } else {
            let value = unsafe { load(source, width) }.to_ne_bytes();
            destination[offset..offset + width].copy_from_slice(&value[..width]);
            source = unsafe { source.add(width) };
            offset += width;
        }
    }
}
/// # Safety
/// The destination is admitted/tracked guest RAM with stable physical lifetime.
/// Source is owned host memory, disjoint from the guest backing.
pub(crate) unsafe fn write(mut destination: *mut u8, source: &[u8]) {
    let mut offset = 0;
    while offset < source.len() {
        let remaining = source.len() - offset;
        let width = width(destination as usize, remaining);
        if width == 8 {
            let bytes = remaining / 8 * 8;
            unsafe {
                words(source.as_ptr().add(offset), destination, bytes / 8);
            }
            destination = unsafe { destination.add(bytes) };
            offset += bytes;
        } else {
            let mut value = [0; 8];
            value[..width].copy_from_slice(&source[offset..offset + width]);
            unsafe {
                store(destination, u64::from_ne_bytes(value), width);
            }
            destination = unsafe { destination.add(width) };
            offset += width;
        }
    }
}

/// # Safety
/// One validated/admitted RAM interval covers the width (1, 2, 4, 8 or 16).
/// Sixteen-byte ordinary values may tear between their two words.
pub(crate) unsafe fn read_value(address: *const u8, bytes: usize) -> u128 {
    #[cfg(target_endian = "little")]
    unsafe {
        if bytes == 16 {
            u128::from(load(address, 8)) | (u128::from(load(address.add(8), 8)) << 64)
        } else {
            u128::from(load(address, bytes))
        }
    }
    #[cfg(target_endian = "big")]
    {
        let mut value = [0; 16];
        unsafe {
            capture(address, &mut value[..bytes]);
        }
        u128::from_le_bytes(value)
    }
}
/// # Safety
/// One validated/admitted writable RAM interval covers the architectural width.
pub(crate) unsafe fn write_value(address: *mut u8, bytes: usize, value: u128) {
    #[cfg(target_endian = "little")]
    unsafe {
        if bytes == 16 {
            store(address, value as u64, 8);
            store(address.add(8), (value >> 64) as u64, 8);
        } else {
            store(address, value as u64, bytes);
        }
    }
    #[cfg(target_endian = "big")]
    unsafe {
        write(address, &value.to_le_bytes()[..bytes]);
    }
}

/// Host compare/exchange on potentially concurrently accessed guest RAM.
/// Returns the value observed at the atomic linearization point.
///
/// # Safety
/// Caller admits a naturally aligned writable RAM span of 1/2/4/8/16 bytes.
/// x86-64 requires CMPXCHG16B, established once by Machine::new.
#[cfg(target_arch = "x86_64")]
pub(crate) unsafe fn compare_exchange(
    address: *mut u8,
    bytes: usize,
    expected: u128,
    replacement: u128,
) -> u128 {
    unsafe {
        match bytes {
            1 => {
                let mut previous = expected as u8;
                asm!("lock cmpxchg byte ptr [{address}], {replacement}", address=in(reg) address, replacement=in(reg_byte) replacement as u8, inout("al") previous, options(nostack));
                u128::from(previous)
            }
            2 => {
                let mut previous = expected as u16;
                asm!("lock cmpxchg word ptr [{address}], {replacement:x}", address=in(reg) address, replacement=in(reg) replacement as u16, inout("ax") previous, options(nostack));
                u128::from(previous)
            }
            4 => {
                let mut previous = expected as u32;
                asm!("lock cmpxchg dword ptr [{address}], {replacement:e}", address=in(reg) address, replacement=in(reg) replacement as u32, inout("eax") previous, options(nostack));
                u128::from(previous)
            }
            8 => {
                let mut previous = expected as u64;
                asm!("lock cmpxchg qword ptr [{address}], {replacement}", address=in(reg) address, replacement=in(reg) replacement as u64, inout("rax") previous, options(nostack));
                u128::from(previous)
            }
            16 => {
                let mut low = expected as u64;
                let mut high = (expected >> 64) as u64;
                // RBX belongs to the caller/LLVM; preserve it within this asm.
                asm!("mov {saved}, rbx", "mov rbx, rsi", "lock cmpxchg16b [rdi]", "mov rbx, {saved}", saved=out(reg) _, in("rdi") address, in("rsi") replacement as u64, in("rcx") (replacement >> 64) as u64, inout("rax") low, inout("rdx") high, options(nostack));
                u128::from(low) | (u128::from(high) << 64)
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn compare_exchange(
    address: *mut u8,
    bytes: usize,
    expected: u128,
    replacement: u128,
) -> u128 {
    let previous: u64;
    unsafe {
        match bytes {
            1 => {
                asm!("2:", "ldaxrb {previous:w}, [{address}]", "cmp {previous:w}, {expected:w}", "b.ne 3f", "stlxrb {status:w}, {replacement:w}, [{address}]", "cbnz {status:w}, 2b", "b 4f", "3:", "clrex", "4:", address=in(reg) address, expected=in(reg) expected as u8 as u64, replacement=in(reg) replacement as u64, previous=out(reg) previous, status=out(reg) _, options(nostack))
            }
            2 => {
                asm!("2:", "ldaxrh {previous:w}, [{address}]", "cmp {previous:w}, {expected:w}", "b.ne 3f", "stlxrh {status:w}, {replacement:w}, [{address}]", "cbnz {status:w}, 2b", "b 4f", "3:", "clrex", "4:", address=in(reg) address, expected=in(reg) expected as u16 as u64, replacement=in(reg) replacement as u64, previous=out(reg) previous, status=out(reg) _, options(nostack))
            }
            4 => {
                asm!("2:", "ldaxr {previous:w}, [{address}]", "cmp {previous:w}, {expected:w}", "b.ne 3f", "stlxr {status:w}, {replacement:w}, [{address}]", "cbnz {status:w}, 2b", "b 4f", "3:", "clrex", "4:", address=in(reg) address, expected=in(reg) expected as u32 as u64, replacement=in(reg) replacement as u64, previous=out(reg) previous, status=out(reg) _, options(nostack))
            }
            8 => {
                asm!("2:", "ldaxr {previous}, [{address}]", "cmp {previous}, {expected}", "b.ne 3f", "stlxr {status:w}, {replacement}, [{address}]", "cbnz {status:w}, 2b", "b 4f", "3:", "clrex", "4:", address=in(reg) address, expected=in(reg) expected as u64, replacement=in(reg) replacement as u64, previous=out(reg) previous, status=out(reg) _, options(nostack))
            }
            16 => {
                let high: u64;
                // A paired exclusive load may tear. Validate even a mismatch
                // with a no-change store before returning an atomic pair.
                asm!("2:", "ldaxp {previous}, {high}, [{address}]", "cmp {previous}, {expected_low}", "ccmp {high}, {expected_high}, #0, eq", "b.ne 3f", "stlxp {status:w}, {replacement_low}, {replacement_high}, [{address}]", "cbnz {status:w}, 2b", "b 4f", "3:", "stlxp {status:w}, {previous}, {high}, [{address}]", "cbnz {status:w}, 2b", "4:", address=in(reg) address, expected_low=in(reg) expected as u64, expected_high=in(reg) (expected >> 64) as u64, replacement_low=in(reg) replacement as u64, replacement_high=in(reg) (replacement >> 64) as u64, previous=out(reg) previous, high=out(reg) high, status=out(reg) _, options(nostack));
                return u128::from(previous) | (u128::from(high) << 64);
            }
            _ => unreachable!(),
        }
    }
    u128::from(previous)
}
