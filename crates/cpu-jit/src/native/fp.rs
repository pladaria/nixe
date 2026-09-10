//! Lazy activation leaf, entered only after native-FP eligibility guards.
//! No system call, host-stack access, guest-state commit or allocator-register
//! clobber. The source boundary must retain NZCV as SSA, not live host flags.
//! Host control instructions: Intel SDM LDMXCSR and Arm FPCR/FPSR registers.
//! https://cdrdv2-public.intel.com/868137/325462-089-sdm-vol-1-2abcd-3abcd-4.pdf
//! https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Registers/FPCR--Floating-point-Control-Register

use super::moves::Emitter;
use crate::abi::{
    CanonicalState, HostAbi, HostFpState, NativeFrame, RegisterClass, TRANSFER_BYTES,
};
use std::mem::{offset_of, size_of};

const OWNER: u32 = offset_of!(NativeFrame<'static>, host_fp) as u32;
const ACTIVE: u32 = OWNER + offset_of!(HostFpState, active) as u32;
const SUSPENDED: u32 = OWNER + offset_of!(HostFpState, suspended) as u32;
const FPCR: u32 =
    (offset_of!(NativeFrame<'static>, canonical) + offset_of!(CanonicalState, fpcr)) as u32;
const SAVE_RAX: u32 = TRANSFER_BYTES - 32;
const CONTROL: u32 = TRANSFER_BYTES - 16;
const _: () = {
    assert!(offset_of!(HostFpState, saved) == offset_of!(HostFpState, active) + 4);
    assert!(size_of::<HostFpState>() == offset_of!(HostFpState, suspended) + 8);
    assert!(ACTIVE.is_multiple_of(8) && SUSPENDED.is_multiple_of(8));
};

/// The invocation has saved its caller, canonical FPCR is unchanged, and its
/// value passed the shared eligibility guard. The continuation remains in the
/// same epoch. On an already active segment this leaves sticky status intact.
/// Installation uses the same control encodings and ownership fields as
/// `HostFpState::ensure`; the gateway remains responsible for completion.
pub(crate) fn emit_fp_activation(abi: HostAbi) -> Vec<u8> {
    use RegisterClass::Integer;
    let mut e = Emitter::new(abi);
    let scratch = abi.reserved().link_scratch[0];
    let active_branch;
    let table_address;
    if abi == HostAbi::X86_64 {
        // CMP active,0; JNE done. Only infrastructure condition flags change.
        e.x64(&[], false, &[0x83], 7, 15, Some(ACTIVE));
        e.code_byte(0);
        e.code.extend([0x0f, 0x85]);
        active_branch = e.code.len();
        e.word(0);
        e.memory(false, Integer, 0, SAVE_RAX, 8);
        e.memory(true, Integer, 11, FPCR, 8);
        e.memory_at(true, Integer, 0, 11, 0, 4);
        e.code.extend([0xc1, 0xe8, 22, 0x83, 0xe0, 15]); // SHR EAX,22; AND EAX,15
        e.code.extend([0x4c, 0x8d, 0x1d]); // LEA R11,[RIP+table]
        table_address = e.code.len();
        e.word(0);
        e.code.extend([0x45, 0x8b, 0x1c, 0x83]); // MOV R11D,[R11+RAX*4]
        e.memory(false, Integer, 11, CONTROL, 4);
        e.x64(&[], false, &[0x0f, 0xae], 2, 15, Some(CONTROL)); // LDMXCSR
        e.memory(true, Integer, 0, SAVE_RAX, 8);
    } else {
        e.memory(true, Integer, 16, ACTIVE, 8);
        active_branch = e.code.len();
        e.word(0); // CBNZ W16,done (ignore the adjacent saved field).
        e.memory(true, Integer, 16, FPCR, 8);
        e.memory_at(true, Integer, 17, 16, 0, 4);
        e.word(0x53000000 | (22 << 16) | (25 << 10) | (17 << 5) | 17); // UBFX
        table_address = e.code.len();
        e.word(0); // ADR X16,table
        e.word(0x8b000000 | (17 << 16) | (2 << 10) | (16 << 5) | 16);
        e.memory_at(true, Integer, 16, 16, 0, 4);
        e.word(0xd51b4400 | 16); // MSR FPCR,X16
        e.word(0xd51b4420 | 31); // MSR FPSR,XZR
    }
    // active=1, saved=1; suspended=0 plus struct padding. Never replace the
    // saved caller control/status. Aligned u64 stores fit both native encoders.
    e.constant(scratch, 0x1_0000_0001, 8);
    e.memory(false, Integer, scratch, ACTIVE, 8);
    e.constant(scratch, 0, 8);
    e.memory(false, Integer, scratch, SUSPENDED, 8);
    let skip_table = e.code.len();
    if abi == HostAbi::X86_64 {
        e.code_byte(0xe9);
    }
    e.word(0);
    let table = e.code.len();
    for index in 0..16 {
        e.word(crate::fp_env::native_control(abi, index << 22));
    }
    let done = e.code.len();
    if abi == HostAbi::X86_64 {
        for (patch, target) in [
            (active_branch, done),
            (table_address, table),
            (skip_table + 1, done),
        ] {
            e.code[patch..patch + 4].copy_from_slice(&((target - patch - 4) as i32).to_le_bytes());
        }
    } else {
        let delta = (table - table_address) as u32;
        for (patch, word) in [
            (
                active_branch,
                0x35000010 | (((done - active_branch) as u32 / 4) << 5),
            ),
            (
                table_address,
                0x10000010 | ((delta & 3) << 29) | ((delta >> 2) << 5),
            ),
            (skip_table, 0x14000000 | ((done - skip_table) as u32 / 4)),
        ] {
            e.code[patch..patch + 4].copy_from_slice(&word.to_le_bytes());
        }
    }
    e.finish()
}
