//! Register preservation for a resumable sampling callback: mapped values at
//! external transfers, all volatile registers for internal SSA continuations.
//! Emission allocates; the generated saves/restores do not. No canonical state,
//! backend spill, SP, pinned register or FP environment is changed here.

use super::moves::{Copy, Emitter};
use super::{TransferError, flags};
use crate::abi::{
    ExitStateMap, HostAbi, LazyFlags, NativeFrame, NzcvLocation, RegisterClass, ValueLocation,
};
use std::mem::offset_of;

// At most 16 caller-clobbered GPRs and 32 full vectors fit below this slot.
// Poll's borrowed registers and flag materialization use the top of the transfer
// partition. A callback must not reenter native code or reuse this partition.
pub(crate) const DESTINATION: u32 = 768;
const _: () = assert!(16 * 8 + 32 * 16 <= DESTINATION);
const _: () = assert!(DESTINATION + 8 < flags::RESULT - 32);

pub(crate) struct Preservation {
    pub save: Vec<u8>,
    pub restore: Vec<u8>,
}

/// Save precisely the caller-clobbered locations named by the source, including
/// clean values, recipe operands and an otherwise unbound dynamic destination.
/// Aliases save once at the largest live width. The callback reads DESTINATION
/// from the frame; it must not mutate any architectural or saved state.
///
/// Host NZCV is supported only if `save` runs before any flag-clobbering poll
/// arithmetic. Native poll maps instead carry packed/deferred NZCV.
/// Restore must run on both success and failure before a source continuation or
/// canonical exit. FP pause/resume and call/return belong to the callback veneer.
pub(crate) fn emit(
    source: &ExitStateMap,
    destination: ValueLocation,
) -> Result<Preservation, TransferError> {
    preservation(source, destination, false)
}

fn preservation(
    source: &ExitStateMap,
    destination: ValueLocation,
    all_volatile: bool,
) -> Result<Preservation, TransferError> {
    source.validate().map_err(TransferError::InvalidContract)?;
    if !destination.valid(source.abi, 8) {
        return Err(TransferError::InvalidContract(
            "invalid sampling destination",
        ));
    }
    let mut integer = [0u8; 32];
    let mut vector = [0u8; 32];
    if all_volatile {
        // Internal SSA may keep optimizer-created temporaries not represented
        // by any guest binding. Save the complete volatile bank on this cold
        // callback, without constraining allocation or spilling on the hot path.
        let registers = if source.abi == HostAbi::X86_64 {
            16
        } else {
            32
        };
        integer[..registers].fill(8);
        vector[..registers].fill(16);
    }
    let mut retain = |location, bytes: u8| {
        if let ValueLocation::Register { class, index } = location {
            let widths = match class {
                RegisterClass::Integer => &mut integer,
                RegisterClass::Vector => &mut vector,
            };
            widths[usize::from(index)] = widths[usize::from(index)].max(bytes.max(8));
        }
    };
    for binding in &source.bindings {
        retain(binding.location, binding.value.bytes());
    }
    retain(destination, 8);
    if source.live.nzcv != 0 {
        match &source.nzcv {
            NzcvLocation::Packed(value) => retain(*value, 4),
            NzcvLocation::Deferred(recipe) => retain_recipe(recipe, &mut retain),
            _ => {}
        }
    }
    let mut save = Emitter::new(source.abi);
    let mut restore = Emitter::new(source.abi);
    let mut offset = 0u32;
    for (class, widths) in [
        (RegisterClass::Integer, integer),
        (RegisterClass::Vector, vector),
    ] {
        for (index, bytes) in widths.into_iter().enumerate() {
            let index = index as u8;
            if bytes == 0 || !clobbered(source.abi, class, index, bytes) {
                continue;
            }
            offset = offset.next_multiple_of(u32::from(bytes));
            assert!(offset + u32::from(bytes) <= DESTINATION);
            save.memory(false, class, index, offset, bytes);
            restore.memory(true, class, index, offset, bytes);
            offset += u32::from(bytes);
        }
    }
    save.copy(Copy {
        source: destination,
        destination: ValueLocation::Spill {
            offset: DESTINATION,
            bytes: 8,
        },
        bytes: 8,
    });
    if source.live.nzcv != 0
        && let NzcvLocation::Host { carry_inverted } = source.nzcv
    {
        flags::materialize(&mut save, &source.nzcv, source.live.nzcv);
        flags::install_host(&mut restore, carry_inverted);
    }
    Ok(Preservation {
        save: save.finish(),
        restore: restore.finish(),
    })
}

/// Called only after the cold poll has reconciled a sample-only deadline and
/// restored its borrowed registers. The callback runs under the same epoch and
/// mapping lease; it cannot reenter native code. Return local patches for the
/// already-charged hot continuation and the canonical failure exit.
/// Internal SSA checks preserve all volatile registers, including optimizer
/// temporaries absent from the guest map; external transfers need only the map.
pub(crate) fn emit_callback(
    source: &ExitStateMap,
    destination: ValueLocation,
    internal: bool,
) -> Result<(Vec<u8>, [u32; 2]), TransferError> {
    if source.flags_to_preserve(crate::analysis::NZCV) != 0 {
        return Err(TransferError::InvalidContract(
            "sampling after poll arithmetic requires packed or deferred NZCV",
        ));
    }
    let preservation = if internal {
        preservation(source, destination, true)?
    } else {
        emit(source, destination)?
    };
    let mut e = Emitter::new(source.abi);
    e.code.extend(preservation.save);
    let scratch = source.abi.reserved().link_scratch[0];
    e.memory(
        true,
        RegisterClass::Integer,
        scratch,
        offset_of!(NativeFrame<'static>, sample_observer) as u32,
        8,
    );
    let absent;
    let failed;
    if source.abi == HostAbi::X86_64 {
        e.code.extend([0x4d, 0x85, 0xdb]); // TEST R11,R11
        e.code.extend([0x0f, 0x84]);
        absent = e.code.len();
        e.word(0);
        e.memory(
            true,
            RegisterClass::Integer,
            7,
            offset_of!(NativeFrame<'static>, dispatch_context) as u32,
            8,
        );
        e.code.extend([0x4c, 0x89, 0xfe]); // MOV RSI,R15: frame
        e.code.extend([0x48, 0x8d, 0x15, 0, 0, 0, 0]); // LEA RDX,[RIP+0]: source
        e.constant(1, source.site.source.get(), 8); // RCX: version
        e.constant(8, u64::from(source.site.state_map), 4); // R8D: map
        e.code.extend([0x41, 0xff, 0xd3]); // CALL R11; gateway SP already aligned
        e.code.extend([0x85, 0xc0, 0x0f, 0x84]); // TEST EAX,EAX; JZ failure
        failed = e.code.len();
        e.word(0);
    } else {
        absent = e.code.len();
        e.word(0); // CBZ X16,resume
        e.memory(
            true,
            RegisterClass::Integer,
            0,
            offset_of!(NativeFrame<'static>, dispatch_context) as u32,
            8,
        );
        e.word(0xaa1503e1); // MOV X1,X21: frame
        e.word(0x10000002); // ADR X2,.: source
        e.constant(3, source.site.source.get(), 8);
        e.constant(4, u64::from(source.site.state_map), 4);
        e.word(0xd63f0200); // BLR X16; native continuations do not use X30
        failed = e.code.len();
        e.word(0); // CBZ W0,failure
    }
    let mut labels = [0; 2];
    let mut patches = [0; 2];
    for index in 0..2 {
        labels[index] = e.code.len();
        e.code.extend_from_slice(&preservation.restore);
        while !e.code.len().is_multiple_of(8) {
            if source.abi == HostAbi::X86_64 {
                e.code_byte(0x90);
            } else {
                e.word(0xd503201f);
            }
        }
        patches[index] = e.code.len() as u32;
        if source.abi == HostAbi::X86_64 {
            e.code
                .extend([0x0f, 0x0b, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90]);
        } else {
            e.word(0xd4200000);
        }
    }
    for (offset, target, arm) in [
        (absent, labels[0], 0xb4000010),
        (failed, labels[1], 0x34000000),
    ] {
        let word = if source.abi == HostAbi::X86_64 {
            (target as i32 - offset as i32 - 4) as u32
        } else {
            arm | (((target - offset) as u32 / 4) << 5)
        };
        e.code[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
    }
    Ok((e.finish(), patches))
}

fn clobbered(abi: HostAbi, class: RegisterClass, index: u8, bytes: u8) -> bool {
    // Same System ABIs as gateway.rs:
    // https://gitlab.com/x86-psABIs/x86-64-ABI
    // https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst
    // AAPCS64 preserves only the low 64 bits of v8-v15, not full Q values.
    match (abi, class) {
        (HostAbi::X86_64, RegisterClass::Integer) => matches!(index, 0..=2 | 6..=10),
        (HostAbi::X86_64, RegisterClass::Vector) => true,
        (HostAbi::Aarch64, RegisterClass::Integer) => index < 16,
        (HostAbi::Aarch64, RegisterClass::Vector) => bytes == 16 || !(8..=15).contains(&index),
    }
}

fn retain_recipe(recipe: &LazyFlags<ValueLocation>, retain: &mut impl FnMut(ValueLocation, u8)) {
    use LazyFlags::*;
    match recipe {
        Canonical(value) | Packed(value) => retain(*value, 4),
        Add {
            lhs,
            rhs,
            result,
            width,
        }
        | Subtract {
            lhs,
            rhs,
            result,
            width,
        } => {
            for value in [lhs, rhs, result] {
                retain(*value, width / 8);
            }
        }
        AddCarry {
            lhs,
            rhs,
            carry,
            result,
            width,
        }
        | SubtractCarry {
            lhs,
            rhs,
            carry,
            result,
            width,
        } => {
            for value in [lhs, rhs, result] {
                retain(*value, width / 8);
            }
            retain(*carry, 1);
        }
        Logical { result, width } => retain(*result, width / 8),
        Conditional {
            predicate,
            when_true,
            ..
        } => {
            retain(*predicate, 1);
            retain_recipe(when_true, retain);
        }
    }
}
