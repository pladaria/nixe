//! Two-way native lookup. Misses fall through to the caller's canonical exit;
//! hits branch through a strongly owned record. No host stack, calls, atomics,
//! shared-generation reads or recency writes occur here.

use super::{Record, SETS};
use crate::abi::{
    BlockKey, ExitStateMap, HostAbi, NativeFrame, NzcvLocation, RegisterClass::Integer,
    ValueLocation,
};
use crate::native::{
    TransferError,
    moves::{Copy, Emitter},
};
use std::mem::offset_of;

// Probe-local values in the ABI transfer partition. No allocated guest value
// may name it; all are dead before a bridge or the canonical adapter starts.
pub(in crate::native) const TARGET: u32 = 0;
const SET: u32 = 8;
pub(in crate::native) const RECORD: u32 = 16;
const RAX: u32 = 24;
const FLAGS: u32 = 32;

/// The PC in `target` is supplied dynamically by `pc`; all other BlockKey
/// fields and the source site are immutable compilation inputs. The source
/// checkpoint must already have charged this terminal's instruction prefix.
pub(crate) fn emit(
    source: &ExitStateMap,
    target: BlockKey,
    pc: ValueLocation,
) -> Result<Vec<u8>, TransferError> {
    source.validate().map_err(TransferError::InvalidContract)?;
    if !pc.valid(source.abi, 8) {
        return Err(TransferError::InvalidContract(
            "invalid PIC target location",
        ));
    }
    let mut e = Emitter::new(source.abi);
    let host_flags = source.live.nzcv != 0 && matches!(source.nzcv, NzcvLocation::Host { .. });
    preserve_flags(&mut e, host_flags, false);
    e.copy(Copy {
        source: pc,
        destination: ValueLocation::Spill {
            offset: TARGET,
            bytes: 8,
        },
        bytes: 8,
    });
    lookup(&mut e, source, target, host_flags);
    Ok(e.finish())
}

/// Shared tail for ordinary indirect probes and matched RSB returns. TARGET
/// and the original host flags have already been saved in transfer storage.
/// A hit restores those flags and jumps; a miss restores them and falls through.
pub(in crate::native) fn lookup(
    e: &mut Emitter,
    source: &ExitStateMap,
    target: BlockKey,
    host_flags: bool,
) {
    let scratch = source.abi.reserved().link_scratch[0];
    e.memory(
        true,
        Integer,
        scratch,
        offset_of!(NativeFrame, indirect_pic) as u32,
        8,
    );
    let absent = if source.abi == HostAbi::X86_64 {
        e.x64(&[], true, &[0x85], scratch, scratch, None); // TEST
        conditional(e, true)
    } else {
        let at = e.code.len();
        e.word(0xb4000010); // CBZ x16,miss
        at
    };
    e.memory(true, Integer, scratch, TARGET, 8);
    let salt =
        ((source.site.source.get() ^ u64::from(source.site.state_map)) & (SETS as u64 - 1)) as u32;
    match source.abi {
        HostAbi::X86_64 => {
            e.x64(&[], true, &[0xc1], 5, scratch, None);
            e.code_byte(2); // SHR
            e.x64(&[], true, &[0x81], 6, scratch, None);
            e.word(salt); // XOR
            e.x64(&[], true, &[0x81], 4, scratch, None);
            e.word(SETS as u32 - 1); // AND
            e.x64(&[], true, &[0xc1], 4, scratch, None);
            e.code_byte(4); // SHL, two pointers/set
            e.x64(
                &[],
                true,
                &[0x03],
                scratch,
                source.abi.reserved().frame,
                Some(offset_of!(NativeFrame, indirect_pic) as u32),
            );
        }
        HostAbi::Aarch64 => {
            // UBFM/LSR, EOR, UBFM low 11 bits, then ADD shifted register.
            // https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions
            e.word(0xd342fe10); // LSR x16,x16,#2
            e.constant(17, u64::from(salt), 8);
            e.word(0xca110210); // EOR x16,x16,x17
            e.word(0xd3402a10); // UBFX x16,x16,#0,#11
            e.memory(
                true,
                Integer,
                17,
                offset_of!(NativeFrame, indirect_pic) as u32,
                8,
            );
            e.word(0x8b101230); // ADD x16,x17,x16,LSL #4
        }
    }
    e.memory(false, Integer, scratch, SET, 8);
    let expected = Record::new(source.site, target, 0);
    for way in 0..2 {
        if way != 0 {
            e.memory(true, Integer, scratch, SET, 8);
        }
        e.memory_at(true, Integer, scratch, scratch, way * 8, 8);
        let mut failures = Vec::new();
        if source.abi == HostAbi::X86_64 {
            e.x64(&[], true, &[0x85], scratch, scratch, None);
            failures.push(conditional(e, true));
        } else {
            failures.push(e.code.len());
            e.word(0xb4000010); // CBZ x16,next
        }
        e.memory(false, Integer, scratch, RECORD, 8);
        // Compare PC first; a hash collision must not use a guest address as a
        // host pointer. Only validated records supply the eventual branch.
        e.memory_at(
            true,
            Integer,
            scratch,
            scratch,
            offset_of!(Record, pc) as u32,
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
        failures.push(conditional(e, false));
        e.memory(true, Integer, scratch, RECORD, 8);
        for (offset, value) in [
            (offset_of!(Record, source), expected.source),
            (
                offset_of!(Record, state_map),
                u64::from(expected.state_map) | (u64::from(expected.platform) << 32),
            ),
            (offset_of!(Record, address_space), expected.address_space),
            (offset_of!(Record, profile), expected.profile),
            (offset_of!(Record, fp), expected.fp),
        ] {
            compare(e, offset as u32, value, &mut failures);
        }
        e.memory_at(
            true,
            Integer,
            scratch,
            scratch,
            offset_of!(Record, address) as u32,
            8,
        );
        preserve_flags(e, host_flags, true);
        e.jump_register(scratch);
        let next = e.code.len();
        for branch in failures {
            patch(e, branch, next);
        }
    }
    let miss = e.code.len();
    patch(e, absent, miss);
    preserve_flags(e, host_flags, true);
}

pub(in crate::native) fn compare(
    e: &mut Emitter,
    offset: u32,
    value: u64,
    failures: &mut Vec<usize>,
) {
    let scratch = e.abi.reserved().link_scratch[0];
    if e.abi == HostAbi::X86_64 {
        if value == (value as i32 as i64) as u64 {
            e.x64(&[], true, &[0x81], 7, scratch, Some(offset));
            e.word(value as u32);
            failures.push(conditional(e, false));
        } else {
            for half in 0..2 {
                e.x64(&[], false, &[0x81], 7, scratch, Some(offset + half * 4));
                e.word((value >> (half * 32)) as u32);
                failures.push(conditional(e, false));
            }
        }
    } else {
        e.memory_at(true, Integer, 17, 16, offset, 8);
        if value < 4096 {
            e.word(0xf100023f | ((value as u32) << 10)); // CMP x17,#imm
            failures.push(conditional(e, false));
        } else {
            e.constant(16, value, 8);
            e.word(0xeb10023f); // CMP x17,x16
            failures.push(conditional(e, false));
            e.memory(true, Integer, 16, RECORD, 8);
        }
    }
}

pub(in crate::native) fn conditional(e: &mut Emitter, equal: bool) -> usize {
    if e.abi == HostAbi::X86_64 {
        e.code.extend([0x0f, if equal { 0x84 } else { 0x85 }]);
        let at = e.code.len();
        e.word(0);
        at
    } else {
        let at = e.code.len();
        e.word(if equal { 0x54000000 } else { 0x54000001 });
        at
    }
}

pub(in crate::native) fn patch(e: &mut Emitter, at: usize, target: usize) {
    if e.abi == HostAbi::X86_64 {
        let delta = i32::try_from(target as i64 - (at + 4) as i64).unwrap();
        e.code[at..at + 4].copy_from_slice(&delta.to_le_bytes());
    } else {
        let delta = (target as i64 - at as i64) / 4;
        assert!((-(1 << 18)..(1 << 18)).contains(&delta));
        let original = u32::from_le_bytes(e.code[at..at + 4].try_into().unwrap());
        e.code[at..at + 4]
            .copy_from_slice(&(original | (((delta as u32) & 0x7ffff) << 5)).to_le_bytes());
    }
}

pub(in crate::native) fn preserve_flags(e: &mut Emitter, live: bool, restore: bool) {
    if !live {
        return;
    }
    if e.abi == HostAbi::Aarch64 {
        if restore {
            e.memory(true, Integer, 17, FLAGS, 8);
            e.word(0xd51b4211); // MSR NZCV,x17 (x16 retains the hit address)
        } else {
            e.word(0xd53b4211); // MRS x17,NZCV
            e.memory(false, Integer, 17, FLAGS, 8);
        }
    } else {
        // LAHF/SAHF require AH: borrow RAX only for capture/restore, then put
        // back all 64 bits. AL retains OF separately; ADD 0x7f restores OF and
        // SAHF restores SF/ZF/CF without altering it. No PUSHF/POPF or SP use.
        // https://cdrdv2-public.intel.com/782151/253667-sdm-vol-2b.pdf
        e.memory(false, Integer, 0, RAX, 8);
        if restore {
            e.memory(true, Integer, 0, FLAGS, 4);
            e.code.extend([0x04, 0x7f, 0x9e]); // ADD al,0x7f; SAHF
        } else {
            e.code.extend([0x9f, 0x0f, 0x90, 0xc0]); // LAHF; SETO al
            e.memory(false, Integer, 0, FLAGS, 4);
        }
        e.memory(true, Integer, 0, RAX, 8);
    }
}
