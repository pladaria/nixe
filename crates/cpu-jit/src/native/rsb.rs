//! Flag-preserving guest return-stack operations. No host stack, calls or
//! canonical guest-state traffic; entries contain only scalar guest keys.

use super::{
    TransferError,
    moves::{Copy, Emitter},
    pic::probe::{RECORD, TARGET, compare, conditional, lookup, patch, preserve_flags},
};
use crate::{
    abi::{
        BlockKey, ExitStateMap, HostAbi, NativeFrame, NzcvLocation, RegisterClass::Integer,
        ValueLocation,
    },
    rsb::{CAPACITY, Continuation, ReturnStack},
};
use std::mem::offset_of;

const POINTER: u32 = offset_of!(NativeFrame, return_stack) as u32;
const HEAD: u32 = offset_of!(ReturnStack, head) as u32;
const DEPTH: u32 = offset_of!(ReturnStack, depth) as u32;
const _: () = assert!(CAPACITY == 16 && size_of::<Continuation>() == 40);

/// A matched prediction pops before the ordinary source-keyed PIC probe.
/// Mismatch/underflow clears the chain and bypasses even a populated PIC.
/// A matched PIC miss retains the remaining predictions. Every miss falls
/// through to the caller's canonical adapter with architectural state intact.
pub(crate) fn emit_return_probe(
    source: &ExitStateMap,
    target: BlockKey,
    pc: ValueLocation,
) -> Result<Vec<u8>, TransferError> {
    emit_return(source, target, pc, true)
}

/// Complete RET prediction bookkeeping on an exhausted/control exit without
/// entering any successor, even when the per-vCPU PIC contains a matching way.
pub(crate) fn emit_return_update(
    source: &ExitStateMap,
    target: BlockKey,
    pc: ValueLocation,
) -> Result<Vec<u8>, TransferError> {
    emit_return(source, target, pc, false)
}

fn emit_return(
    source: &ExitStateMap,
    target: BlockKey,
    pc: ValueLocation,
    probe: bool,
) -> Result<Vec<u8>, TransferError> {
    source.validate().map_err(TransferError::InvalidContract)?;
    if !pc.valid(source.abi, 8) {
        return Err(TransferError::InvalidContract(
            "invalid RSB target location",
        ));
    }
    let mut e = Emitter::new(source.abi);
    let scratch = source.abi.reserved().link_scratch[0];
    let flags = source.live.nzcv != 0 && matches!(source.nzcv, NzcvLocation::Host { .. });
    preserve_flags(&mut e, flags, false);
    e.copy(Copy {
        source: pc,
        destination: ValueLocation::Spill {
            offset: TARGET,
            bytes: 8,
        },
        bytes: 8,
    });
    e.memory(true, Integer, scratch, POINTER, 8);
    let absent = if source.abi == HostAbi::X86_64 {
        e.x64(&[], true, &[0x85], scratch, scratch, None);
        conditional(&mut e, true)
    } else {
        let at = e.code.len();
        e.word(0xb4000010); // CBZ x16,miss (no owner to clear)
        at
    };
    let mut failures = Vec::new();
    if source.abi == HostAbi::X86_64 {
        e.x64(&[], false, &[0x83], 7, scratch, Some(DEPTH));
        e.code_byte(0); // CMP depth,0
        failures.push(conditional(&mut e, true));
        e.memory_at(true, Integer, scratch, scratch, HEAD, 4);
        e.x64(&[], false, &[0x83], 5, scratch, None);
        e.code_byte(1); // SUB r11d,1
        e.x64(&[], false, &[0x83], 4, scratch, None);
        e.code_byte(15); // AND r11d,15
        e.x64(&[], true, &[0x6b], scratch, scratch, None);
        e.code_byte(40);
        e.x64(
            &[],
            true,
            &[0x03],
            scratch,
            source.abi.reserved().frame,
            Some(POINTER),
        );
    } else {
        e.memory_at(true, Integer, 17, 16, DEPTH, 4);
        failures.push(e.code.len());
        e.word(0x34000011); // CBZ w17,clear
        e.memory_at(true, Integer, 17, 16, HEAD, 4);
        e.word(0x51000631); // SUB w17,w17,#1
        e.word(0x53000e31); // UBFX w17,w17,#0,#4
        e.word(0x8b110a31); // 5*index
        e.word(0x8b110e10); // root+40*index
    }
    e.memory(false, Integer, scratch, RECORD, 8);
    e.memory_at(
        true,
        Integer,
        scratch,
        scratch,
        offset_of!(Continuation, pc) as u32,
        8,
    );
    if source.abi == HostAbi::X86_64 {
        e.x64(
            &[],
            true,
            &[0x3b],
            scratch,
            source.abi.reserved().frame,
            Some(TARGET),
        );
    } else {
        e.memory(true, Integer, 17, TARGET, 8);
        e.word(0xeb11021f); // CMP x16,x17
    }
    failures.push(conditional(&mut e, false));
    e.memory(true, Integer, scratch, RECORD, 8);
    let expected = Continuation::from(target);
    for (offset, value) in [
        (
            offset_of!(Continuation, address_space),
            expected.address_space,
        ),
        (offset_of!(Continuation, profile), expected.profile),
        (offset_of!(Continuation, platform), expected.platform),
        (offset_of!(Continuation, fp), expected.fp),
    ] {
        compare(&mut e, offset as u32, value, &mut failures);
    }
    e.memory(true, Integer, scratch, POINTER, 8);
    if source.abi == HostAbi::X86_64 {
        e.x64(&[], false, &[0x83], 5, scratch, Some(HEAD));
        e.code_byte(1);
        e.x64(&[], false, &[0x83], 4, scratch, Some(HEAD));
        e.code_byte(15);
        e.x64(&[], false, &[0x83], 5, scratch, Some(DEPTH));
        e.code_byte(1);
    } else {
        e.memory_at(true, Integer, 17, 16, HEAD, 4);
        e.word(0x51000631); // SUB w17,w17,#1
        e.word(0x53000e31); // UBFX w17,w17,#0,#4
        e.memory_at(false, Integer, 17, 16, HEAD, 4);
        e.memory_at(true, Integer, 17, 16, DEPTH, 4);
        e.word(0x51000631);
        e.memory_at(false, Integer, 17, 16, DEPTH, 4);
    }
    // Reuse the saved PC and original host flags directly. Do not save/restore
    // them a second time between the successful RSB check and the PIC probe.
    if probe {
        lookup(&mut e, source, target, flags);
    } else {
        preserve_flags(&mut e, flags, true);
    }
    // A matched PIC miss (or a matched poll-only update) reaches this cold
    // branch. Skip mismatch cleanup, retaining older predictions and flags.
    let skip = e.code.len();
    if source.abi == HostAbi::X86_64 {
        e.code_byte(0xe9);
        e.word(0);
    } else {
        e.word(0x14000000);
    }
    let clear = e.code.len();
    for branch in failures {
        patch(&mut e, branch, clear);
    }
    e.memory(true, Integer, scratch, POINTER, 8);
    if source.abi == HostAbi::X86_64 {
        e.x64(&[], true, &[0xc7], 0, scratch, Some(HEAD));
        e.word(0); // Adjacent head/depth are both cleared by one qword store.
    } else {
        e.memory_at(false, Integer, 31, 16, HEAD, 8); // STR xzr,[x16,#HEAD]
    }
    let miss = e.code.len();
    patch(&mut e, absent, miss);
    preserve_flags(&mut e, flags, true);
    let end = e.code.len();
    if source.abi == HostAbi::X86_64 {
        e.code[skip + 1..skip + 5]
            .copy_from_slice(&i32::try_from(end - skip - 5).unwrap().to_le_bytes());
    } else {
        e.code[skip..skip + 4].copy_from_slice(
            &(0x14000000 | u32::try_from((end - skip) / 4).unwrap()).to_le_bytes(),
        );
    }
    Ok(e.finish())
}

/// Push a compile-time continuation, after the architectural X30 update. The
/// caller must place this on every completed BL/BLR path, exactly once, before
/// entering the successor or leaving canonically. A bare frame's null stack
/// is a no-op.
pub(crate) fn emit_push(
    source: &ExitStateMap,
    continuation: BlockKey,
) -> Result<Vec<u8>, TransferError> {
    source.validate().map_err(TransferError::InvalidContract)?;
    if continuation.pc.get() & 3 != 0 {
        return Err(TransferError::InvalidContract("unaligned RSB continuation"));
    }
    let mut e = Emitter::new(source.abi);
    let scratch = source.abi.reserved().link_scratch[0];
    let flags = source.live.nzcv != 0 && matches!(source.nzcv, NzcvLocation::Host { .. });
    preserve_flags(&mut e, flags, false);
    e.memory(true, Integer, scratch, POINTER, 8);
    let absent = if source.abi == HostAbi::X86_64 {
        e.x64(&[], true, &[0x85], scratch, scratch, None); // TEST r11,r11
        conditional(&mut e, true)
    } else {
        let at = e.code.len();
        e.word(0xb4000010); // CBZ x16,done
        at
    };
    match source.abi {
        HostAbi::X86_64 => {
            e.memory_at(true, Integer, scratch, scratch, HEAD, 4);
            e.x64(&[], true, &[0x6b], scratch, scratch, None);
            e.code_byte(40); // IMUL r11,r11,40
            e.x64(
                &[],
                true,
                &[0x03],
                scratch,
                source.abi.reserved().frame,
                Some(POINTER),
            );
        }
        HostAbi::Aarch64 => {
            // ADD shifted-register addressing for 40-byte entries.
            // https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/ADD--shifted-register---Add-optionally-shifted-register-
            e.memory_at(true, Integer, 17, 16, HEAD, 4);
            e.word(0x8b110a31); // ADD x17,x17,x17,LSL #2 (5*head)
            e.word(0x8b110e10); // ADD x16,x16,x17,LSL #3
        }
    }
    let key = Continuation::from(continuation);
    for (offset, value) in [
        (offset_of!(Continuation, pc), key.pc),
        (offset_of!(Continuation, address_space), key.address_space),
        (offset_of!(Continuation, profile), key.profile),
        (offset_of!(Continuation, platform), key.platform),
        (offset_of!(Continuation, fp), key.fp),
    ] {
        if source.abi == HostAbi::X86_64 {
            // Two immediate dword stores need no allocator-visible temporary,
            // including for full-width keys (MOV r/m64,imm32 sign-extends).
            for half in 0..2 {
                e.x64(
                    &[],
                    false,
                    &[0xc7],
                    0,
                    scratch,
                    Some(offset as u32 + half * 4),
                );
                e.word((value >> (half * 32)) as u32);
            }
        } else {
            e.constant(17, value, 8);
            e.memory_at(false, Integer, 17, 16, offset as u32, 8);
        }
    }
    e.memory(true, Integer, scratch, POINTER, 8);
    match source.abi {
        HostAbi::X86_64 => {
            e.x64(&[], false, &[0x83], 0, scratch, Some(HEAD));
            e.code_byte(1); // ADD head,1
            e.x64(&[], false, &[0x83], 4, scratch, Some(HEAD));
            e.code_byte(15); // AND head,15
            e.x64(&[], false, &[0x83], 7, scratch, Some(DEPTH));
            e.code_byte(16); // CMP depth,16
        }
        HostAbi::Aarch64 => {
            e.memory_at(true, Integer, 17, 16, HEAD, 4);
            e.word(0x11000631); // ADD w17,w17,#1
            e.word(0x53000e31); // UBFX w17,w17,#0,#4
            e.memory_at(false, Integer, 17, 16, HEAD, 4);
            e.memory_at(true, Integer, 17, 16, DEPTH, 4);
            e.word(0x7100423f); // CMP w17,#16
        }
    }
    let full = conditional(&mut e, true);
    if source.abi == HostAbi::X86_64 {
        e.x64(&[], false, &[0x83], 0, scratch, Some(DEPTH));
        e.code_byte(1);
    } else {
        e.word(0x11000631); // ADD w17,w17,#1
        e.memory_at(false, Integer, 17, 16, DEPTH, 4);
    }
    let done = e.code.len();
    patch(&mut e, absent, done);
    patch(&mut e, full, done);
    preserve_flags(&mut e, flags, true);
    Ok(e.finish())
}
