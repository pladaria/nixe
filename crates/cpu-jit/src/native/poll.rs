//! Bounded native cold control. No guest memory, host calls, FP instructions or
//! stack changes: the active FP segment and exclusive reservation survive.
//! All mapped SSA registers survive; infrastructure condition flags are dead.
//! Requests arriving after their acquire load wait at most another interval.
//! The active epoch/lease prevents maintenance mutation until a genuine exit;
//! this leaf neither acknowledges requests nor waits for a maintenance owner.

use super::moves::Emitter;
use crate::abi::{
    HostAbi, NativeFrame, PollBudget, RegisterClass::Integer, SAMPLE_INTERVAL, TRANSFER_BYTES,
};
use std::mem::offset_of;

const SAVE: u32 = TRANSFER_BYTES - 32;
const BUDGET: u32 = offset_of!(NativeFrame<'static>, budget) as u32;
const SAMPLE: u32 = BUDGET + offset_of!(PollBudget, sample_remaining) as u32;
const SLICE: u32 = BUDGET + offset_of!(PollBudget, slice_remaining) as u32;
const ARMED: u32 = BUDGET + offset_of!(PollBudget, armed_span) as u32;
const _: () = assert!(SAMPLE_INTERVAL == 4096); // AArch64 shifted ADD immediate below.

/// Return bytes and three local branch patches: resume, slice exit, control
/// exit. Resume goes to the already-completed source's hot patch. Exit adapters
/// charge zero: the gateway reconciles the still-unmodified budget. Only the
/// resumable sample-only path updates both balances here. No functional sample
/// is emitted yet; Task 5 will consume the source terminal's retained identity.
pub(crate) fn emit_poll(abi: HostAbi) -> (Vec<u8>, [u32; 3]) {
    let mut e = Emitter::new(abi);
    let scratch = abi.reserved().link_scratch[0];
    // Two borrowed GPRs on x86, one on Arm. Transfer scratch is disjoint from
    // allocator spills. Neither helper calls nor a transfer can overlap this leaf.
    e.memory(false, Integer, 0, SAVE, 8);
    if abi == HostAbi::X86_64 {
        e.memory(false, Integer, 1, SAVE + 8, 8);
    }
    let mut controls = Vec::new();
    for index in 0..3 {
        e.memory(
            true,
            Integer,
            scratch,
            offset_of!(NativeFrame<'static>, poll_requests) as u32 + index * 8,
            8,
        );
        if abi == HostAbi::X86_64 {
            // Aligned MOV is an acquire load on x86-64 (Intel SDM vol. 3,
            // memory ordering). No atomic RMW or consumption of the request.
            // https://cdrdv2-public.intel.com/868137/325462-089-sdm-vol-1-2abcd-3abcd-4.pdf
            e.memory_at(true, Integer, 11, 11, 0, 4);
            e.code.extend([0x45, 0x85, 0xdb]); // TEST R11D,R11D
            e.code.extend([0x0f, 0x85]);
        } else {
            // LDAR W16,[X16]; CBNZ W16,control.
            // https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/LDAR--Load-Acquire-Register-
            e.word(0x88dffe10);
        }
        controls.push(e.code.len());
        e.word(0);
    }
    let exhausted;
    if abi == HostAbi::X86_64 {
        e.memory(true, Integer, 0, ARMED, 8);
        e.code.extend([0x4c, 0x29, 0xf0]); // SUB RAX,R14: spent
        e.memory(true, Integer, 11, SLICE, 8);
        e.code.extend([0x49, 0x29, 0xc3]); // SUB R11,RAX: new slice
        e.code.extend([0x0f, 0x8e]); // JLE slice exit
        exhausted = e.code.len();
        e.word(0);
        e.memory(false, Integer, 11, SLICE, 8);
        e.memory(true, Integer, 1, SAMPLE, 8);
        e.code.extend([0x48, 0x29, 0xc1]); // SUB RCX,RAX
        e.code.extend([0x48, 0x81, 0xc1]); // ADD RCX,4096
        e.word(SAMPLE_INTERVAL as u32);
        e.memory(false, Integer, 1, SAMPLE, 8);
        e.code.extend([0x4c, 0x39, 0xd9]); // CMP RCX,R11
        e.code.extend([0x49, 0x0f, 0x4f, 0xcb]); // CMOVG RCX,R11: min
        e.memory(false, Integer, 1, ARMED, 8);
        e.code.extend([0x49, 0x89, 0xce]); // MOV R14,RCX
    } else {
        e.memory(true, Integer, 16, ARMED, 8);
        e.word(0xcb140210); // SUB X16,X16,X20: spent
        e.memory(true, Integer, 17, SLICE, 8);
        e.word(0xeb100231); // SUBS X17,X17,X16: new slice
        exhausted = e.code.len();
        e.word(0); // B.LE slice exit
        e.memory(false, Integer, 17, SLICE, 8);
        e.memory(true, Integer, 0, SAMPLE, 8);
        e.word(0xcb100000); // SUB X0,X0,X16
        e.word(0x91400400); // ADD X0,X0,#1,LSL #12
        e.memory(false, Integer, 0, SAMPLE, 8);
        e.word(0xeb11001f); // CMP X0,X17
        e.word(0x9a91d014); // CSEL X20,X0,X17,LE: min
        e.memory(false, Integer, 20, ARMED, 8);
    }
    // The counter was armed to min(sample,slice). If the slice still has work,
    // this deadline crossed the sample. A <=2048-instruction terminal overshoots
    // by at most 2047, so one +4096 is exactly PollBudget's modulo wrap.
    let mut patches = [0; 3];
    let mut labels = [0; 3];
    for (index, patch) in patches.iter_mut().enumerate() {
        labels[index] = e.code.len();
        e.memory(true, Integer, 0, SAVE, 8);
        if abi == HostAbi::X86_64 {
            e.memory(true, Integer, 1, SAVE + 8, 8);
        }
        while !e.code.len().is_multiple_of(8) {
            if abi == HostAbi::X86_64 {
                e.code_byte(0x90);
            } else {
                e.word(0xd503201f);
            }
        }
        *patch = e.code.len() as u32;
        if abi == HostAbi::X86_64 {
            e.code
                .extend([0x0f, 0x0b, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90]);
        } else {
            e.word(0xd4200000);
        }
    }
    for (offset, target, arm) in controls
        .into_iter()
        .map(|offset| (offset, labels[2], 0x35000010))
        .chain([(exhausted, labels[1], 0x5400000d)])
    {
        let word = if abi == HostAbi::X86_64 {
            (target as i32 - offset as i32 - 4) as u32
        } else {
            arm | (((target - offset) as u32 / 4) << 5)
        };
        e.code[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
    }
    (e.finish(), patches)
}
