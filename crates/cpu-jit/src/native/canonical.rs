//! Canonical data adapters over NativeFrame's borrowed architectural pointers.
//! No A64State layout offsets, duplicate state image or system-ABI calls.

use super::TransferError;
use super::moves::{Copy, Emitter};
use crate::abi::{
    CanonicalState, EntryContract, ExitStateMap, GuestValue, HostAbi, NativeExitReason,
    NativeFrame, NzcvLocation, RegisterClass, TRANSFER_BYTES, ValueLocation,
};
use std::mem::offset_of;

// Does not overlap constant materialization's final 16-byte slot. Canonical
// adapters do not run concurrently with an in-progress fast transfer.
const BORROW_SAVE: u32 = TRANSFER_BYTES - 32;
const CANONICAL: u32 = offset_of!(NativeFrame<'static>, canonical) as u32;
const NZCV_POINTER: u32 = CANONICAL + offset_of!(CanonicalState, nzcv) as u32;

#[derive(Clone, Copy)]
struct Operand {
    pointer: u32,
    offset: u32,
    bytes: u8,
    location: ValueLocation,
}

fn operand(value: GuestValue, location: ValueLocation) -> Operand {
    let (pointer, offset) = match value {
        GuestValue::General(index) => (offset_of!(CanonicalState, x), u32::from(index) * 8),
        GuestValue::Vector(index) => (offset_of!(CanonicalState, vector), u32::from(index) * 16),
        GuestValue::Sp => (offset_of!(CanonicalState, sp), 0),
        GuestValue::Fpcr => (offset_of!(CanonicalState, fpcr), 0),
        GuestValue::Fpsr => (offset_of!(CanonicalState, fpsr), 0),
        GuestValue::TpidrEl0 => (offset_of!(CanonicalState, tpidr_el0), 0),
        GuestValue::TpidrroEl0 => (offset_of!(CanonicalState, tpidrro_el0), 0),
    };
    Operand {
        pointer: CANONICAL + pointer as u32,
        offset,
        bytes: value.bytes(),
        location,
    }
}

/// Load precisely the target's live physical inputs from canonical A64State.
/// Append the fast-entry branch separately. This adapter assumes canonical state
/// is authoritative; it neither publishes an epoch nor installs the guest FP
/// environment. Host flags change only when the target requests host NZCV;
/// packed ingress leaves host flags intact on both hosts. x86-64 host-flag ingress
/// assumes the documented SAHF minimum, checked by the execution owner at setup.
/// On x86-64 spill ingress can use RAX temporarily; if RAX is a target input it
/// is loaded last. Unbound registers need not survive canonical ingress.
pub fn emit_canonical_entry(target: &EntryContract) -> Result<Vec<u8>, TransferError> {
    target.validate().map_err(TransferError::InvalidContract)?;
    let mut operands: Vec<_> = target
        .bindings
        .iter()
        .map(|binding| operand(binding.value, binding.location))
        .collect();
    if target.live_in.nzcv != 0 {
        let location = match target.nzcv {
            NzcvLocation::Packed(location) => location,
            NzcvLocation::Host { .. } => crate::abi::ValueLocation::Spill {
                offset: super::flags::RESULT,
                bytes: 4,
            },
            _ => unreachable!("validated live NZCV ingress is packed or host flags"),
        };
        operands.push(Operand {
            pointer: NZCV_POINTER,
            offset: 0,
            bytes: 4,
            location,
        });
    }
    let mut emitter = Emitter::new(target.abi);
    emit_operands(&mut emitter, target.abi, true, operands);
    if target.live_in.nzcv != 0
        && let NzcvLocation::Host { carry_inverted } = target.nzcv
    {
        super::flags::install_host(&mut emitter, carry_inverted);
    }
    Ok(emitter.finish())
}

/// Store only dirty live architectural values; merge only dirty NZCV bits.
/// This is the writeback portion of a canonical exit, not the complete exit:
/// PC/reason publication, budget reconciliation, FP ownership and the gateway
/// return belong to the caller. Physical source registers remain available for
/// subsequent boundary work. Recipes/host flags are materialized before data
/// writeback can clobber them, and only dirty NZCV bits are computed.
/// When host FPSR is pending, this stores the mapped
/// software contribution (or leaves canonical software FPSR alone if unmapped).
/// Then the caller MUST run [NativeFrame::finish_fp] on exit, or
/// [NativeFrame::suspend_fp] before a helper, to collect sticky host status and
/// restore the caller environment BEFORE general Rust work or epoch quiescence.
/// Collecting before this writeback would let the software store erase flags.
/// The final NZCV merge may clobber x86-64 condition flags, never host FP state.
pub fn emit_canonical_writeback(source: &ExitStateMap) -> Result<Vec<u8>, TransferError> {
    source.validate().map_err(TransferError::InvalidContract)?;
    let mut emitter = Emitter::new(source.abi);
    writeback(&mut emitter, source, &source.nzcv);
    Ok(emitter.finish())
}

// The bridge may supply an already materialized NZCV in ABI-owned transfer
// storage. Public maps still prohibit that storage; only emission creates it.
pub(super) fn writeback(emitter: &mut Emitter, source: &ExitStateMap, nzcv: &NzcvLocation) {
    let operands = source
        .bindings
        .iter()
        .filter(|binding| {
            !source
                .dirty_live
                .intersection(binding.value.state().unwrap())
                .is_empty()
        })
        .map(|binding| operand(binding.value, binding.location))
        .collect();
    let nzcv = if source.dirty_live.nzcv != 0 && !matches!(nzcv, NzcvLocation::Packed(_)) {
        super::flags::materialize(emitter, nzcv, source.dirty_live.nzcv);
        NzcvLocation::Packed(ValueLocation::Spill {
            offset: super::flags::RESULT,
            bytes: 4,
        })
    } else {
        nzcv.clone()
    };
    emit_operands(emitter, source.abi, false, operands);
    if source.dirty_live.nzcv != 0 {
        let NzcvLocation::Packed(location) = nzcv else {
            unreachable!()
        };
        let pointer = source.abi.reserved().link_scratch[0];
        let value = temporary_register(source.abi);
        if source.abi == HostAbi::X86_64 {
            emitter.memory(false, RegisterClass::Integer, value, BORROW_SAVE, 8);
        }
        emitter.copy(Copy {
            source: location,
            destination: ValueLocation::Register {
                class: RegisterClass::Integer,
                index: value,
            },
            bytes: 4,
        });
        emitter.memory(true, RegisterClass::Integer, pointer, NZCV_POINTER, 8);
        emitter.merge_nzcv(value, pointer, source.dirty_live.nzcv);
        if source.dirty_live.nzcv != crate::analysis::NZCV {
            emitter.memory(true, RegisterClass::Integer, pointer, NZCV_POINTER, 8);
        }
        emitter.memory_at(false, RegisterClass::Integer, value, pointer, 0, 4);
        if source.abi == HostAbi::X86_64 {
            emitter.memory(true, RegisterClass::Integer, value, BORROW_SAVE, 8);
        }
    }
}

/// Load missing clean bridge inputs after physical copies. Unlike canonical
/// ingress, already-installed inputs (including RAX) must remain untouched.
pub(super) fn load_missing(emitter: &mut Emitter, bindings: &[crate::abi::ValueBinding]) {
    let preserve_rax = emitter.abi == HostAbi::X86_64
        && bindings
            .iter()
            .any(|b| matches!(b.location, ValueLocation::Spill { .. }))
        && !bindings.iter().any(|b| {
            b.location
                == ValueLocation::Register {
                    class: RegisterClass::Integer,
                    index: 0,
                }
        });
    if preserve_rax {
        emitter.memory(false, RegisterClass::Integer, 0, BORROW_SAVE, 8);
    }
    emit_operands(
        emitter,
        emitter.abi,
        true,
        bindings
            .iter()
            .map(|b| operand(b.value, b.location))
            .collect(),
    );
    if preserve_rax {
        emitter.memory(true, RegisterClass::Integer, 0, BORROW_SAVE, 8);
    }
}

/// Fill flags absent from the source from their authoritative canonical home.
/// RESULT already holds the source bits; preserve every allocated register.
pub(super) fn fill_missing_flags(emitter: &mut Emitter, source_bits: u8) {
    let abi = emitter.abi;
    let value = temporary_register(abi);
    let pointer = abi.reserved().link_scratch[0];
    if abi == HostAbi::X86_64 {
        emitter.memory(false, RegisterClass::Integer, value, BORROW_SAVE, 8);
    }
    emitter.memory(true, RegisterClass::Integer, value, super::flags::RESULT, 4);
    emitter.memory(true, RegisterClass::Integer, pointer, NZCV_POINTER, 8);
    emitter.merge_nzcv(value, pointer, source_bits);
    emitter.memory(
        false,
        RegisterClass::Integer,
        value,
        super::flags::RESULT,
        4,
    );
    if abi == HostAbi::X86_64 {
        emitter.memory(true, RegisterClass::Integer, value, BORROW_SAVE, 8);
    }
}

/// Emit dirty state writeback, dynamic/constant destination PC and exit identity,
/// followed by a jump to this invocation's gateway continuation. FP completion
/// and poll reconciliation run there before returning to the lifetime owner.
/// No host call, RET or SP adjustment is emitted. After data writeback no guest
/// register remains live, so the epilogue may use RAX/X0 without saving it.
/// Lazy or host NZCV is materialized by the writeback adapter before publication.
/// `completed` is the uncharged work on this path (at most one 2048-instruction
/// unit). Charge it once without changing host flags; PRE helper/fault exits
/// exclude the instruction still awaiting completion.
pub fn emit_canonical_exit(
    source: &ExitStateMap,
    pc: ValueLocation,
    reason: NativeExitReason,
    completed: u16,
) -> Result<Vec<u8>, TransferError> {
    emit_exit(
        source,
        pc,
        reason,
        completed,
        offset_of!(NativeFrame<'static>, gateway_exit),
    )
}

/// Canonicalize a link miss and enter the invocation's cold resolver instead
/// of leaving its gateway/epoch. Budget and control exits use the ordinary exit.
pub(crate) fn emit_dispatch_fallback(
    source: &ExitStateMap,
    pc: ValueLocation,
    completed: u16,
) -> Result<Vec<u8>, TransferError> {
    emit_exit(
        source,
        pc,
        NativeExitReason::Dispatch,
        completed,
        offset_of!(NativeFrame<'static>, dispatch_fallback),
    )
}

fn emit_exit(
    source: &ExitStateMap,
    pc: ValueLocation,
    reason: NativeExitReason,
    completed: u16,
    continuation: usize,
) -> Result<Vec<u8>, TransferError> {
    if reason == NativeExitReason::None {
        return Err(TransferError::InvalidContract(
            "canonical exit needs a reason",
        ));
    }
    let mut emitter = exit_state(source, pc, completed)?;
    exit_identity(&mut emitter, source, reason);
    exit_continuation(&mut emitter, continuation);
    Ok(emitter.finish())
}

/// Cold entries with one state materialization and one RSB update. Entry zero
/// is the dispatch fallback; returned offsets are [slice, control]. A lookup
/// miss has already updated the RSB, whereas both poll exits must still do so.
/// The reason/map pair is
/// selected before writeback using only link scratch and flag-transparent moves.
/// No new frame storage, relocation, shared owner or hot-edge work is required.
/// Both entries have already charged their terminal checkpoint.
pub(crate) fn emit_polled_exit(
    source: &ExitStateMap,
    pc: ValueLocation,
    operation: &[u8],
    indirect: bool,
) -> Result<(Vec<u8>, [usize; 2]), TransferError> {
    let mut emitter = Emitter::new(source.abi);
    exit_identity(&mut emitter, source, NativeExitReason::Dispatch);
    let skip = emitter.code.len();
    if source.abi == HostAbi::X86_64 {
        emitter.code_byte(0xe9);
        emitter.word(0);
    } else {
        emitter.word(0x14000000);
    }
    let control = emitter.code.len();
    exit_identity(&mut emitter, source, NativeExitReason::Control);
    let mut control_skip = None;
    let slice = if indirect && !operation.is_empty() {
        control_skip = Some(emitter.code.len());
        if source.abi == HostAbi::X86_64 {
            emitter.code_byte(0xe9);
            emitter.word(0);
        } else {
            emitter.word(0x14000000);
        }
        let slice = emitter.code.len();
        exit_identity(&mut emitter, source, NativeExitReason::Dispatch);
        slice
    } else {
        0
    };
    let update = emitter.code.len();
    emitter.code.extend_from_slice(operation);
    let common = emitter.code.len();
    for (branch, target) in std::iter::once((skip, if indirect { common } else { update }))
        .chain(control_skip.map(|branch| (branch, update)))
    {
        if source.abi == HostAbi::X86_64 {
            emitter.code[branch + 1..branch + 5]
                .copy_from_slice(&((target - branch - 5) as i32).to_le_bytes());
        } else {
            emitter.code[branch..branch + 4]
                .copy_from_slice(&(0x14000000 | ((target - branch) as u32 / 4)).to_le_bytes());
        }
    }
    emitter.code.extend(exit_state(source, pc, 0)?.finish());
    // Architectural values are now canonical, including lazy flags. Only these
    // cold paths inspect the selected reason; a linked hot exit bypasses both.
    let branch;
    if source.abi == HostAbi::X86_64 {
        emitter.x64(
            &[],
            false,
            &[0xf6],
            0,
            source.abi.reserved().frame,
            Some(offset_of!(NativeFrame<'static>, exit_reason) as u32),
        );
        emitter.code_byte(2); // TEST byte [frame+reason],2: Control=3, Dispatch=1
        emitter.code.extend([0x0f, 0x85]); // JNZ gateway
        branch = emitter.code.len();
        emitter.word(0);
    } else {
        emitter.memory(
            true,
            RegisterClass::Integer,
            16,
            offset_of!(NativeFrame<'static>, exit_state_map) as u32,
            8,
        );
        branch = emitter.code.len();
        emitter.word(0); // TBNZ X16,#33,gateway (reason bit 1 in packed pair)
    }
    const _: () = assert!(NativeExitReason::Dispatch as u32 == 1);
    const _: () = assert!(NativeExitReason::Control as u32 == 3);
    exit_continuation(
        &mut emitter,
        offset_of!(NativeFrame<'static>, dispatch_fallback),
    );
    let gateway = emitter.code.len();
    let patch = if source.abi == HostAbi::X86_64 {
        (gateway - branch - 4) as u32
    } else {
        0xb7080010 | (((gateway - branch) as u32 / 4) << 5)
    };
    emitter.code[branch..branch + 4].copy_from_slice(&patch.to_le_bytes());
    exit_continuation(&mut emitter, offset_of!(NativeFrame<'static>, gateway_exit));
    Ok((emitter.finish(), [slice, control]))
}

fn exit_state(
    source: &ExitStateMap,
    pc: ValueLocation,
    completed: u16,
) -> Result<Emitter, TransferError> {
    if completed > 2048 {
        return Err(TransferError::InvalidContract(
            "exit work exceeds unit bound",
        ));
    }
    if !pc.valid(source.abi, 8) {
        return Err(TransferError::InvalidContract("invalid exit PC location"));
    }
    let mut emitter = Emitter::new(source.abi);
    emitter.code = emit_canonical_writeback(source)?;
    let scratch = source.abi.reserved().link_scratch[0];
    emitter.copy(Copy {
        source: pc,
        destination: ValueLocation::Register {
            class: RegisterClass::Integer,
            index: 0,
        },
        bytes: 8,
    });
    emitter.memory(
        false,
        RegisterClass::Integer,
        0,
        offset_of!(NativeFrame<'static>, exit_pc) as u32,
        8,
    );
    emitter.memory(
        true,
        RegisterClass::Integer,
        scratch,
        CANONICAL + offset_of!(CanonicalState, pc) as u32,
        8,
    );
    emitter.memory_at(false, RegisterClass::Integer, 0, scratch, 0, 8);
    if completed != 0 {
        let poll = source.abi.reserved().poll;
        if source.abi == HostAbi::X86_64 {
            // LEA r14,[r14-completed]: no flags, guest registers or memory touched.
            emitter.x64(
                &[],
                true,
                &[0x8d],
                poll,
                poll,
                Some((-i32::from(completed)) as u32),
            );
        } else {
            // SUB x20,x20,#completed (not SUBS).
            emitter.word(
                0xd1000000
                    | (u32::from(completed) << 10)
                    | (u32::from(poll) << 5)
                    | u32::from(poll),
            );
        }
    }
    // Self-relative identity survives staging/copying without a relocation.
    // No guest operand is live here. Hot links never publish a current owner.
    if source.abi == HostAbi::X86_64 {
        // LEA scratch,[RIP+0]: address of the following store in this adapter.
        emitter.code.extend([
            0x48 | ((scratch >> 3) << 2),
            0x8d,
            0x05 | ((scratch & 7) << 3),
        ]);
        emitter.code.extend(0i32.to_le_bytes());
    } else {
        // ADR scratch,.: address of this instruction in the exit adapter.
        emitter.word(0x10000000 | u32::from(scratch));
    }
    emitter.memory(
        false,
        RegisterClass::Integer,
        scratch,
        offset_of!(NativeFrame<'static>, exit_native_pc) as u32,
        8,
    );
    emitter.constant(scratch, source.site.source.get(), 8);
    emitter.memory(
        false,
        RegisterClass::Integer,
        scratch,
        offset_of!(NativeFrame<'static>, exit_source_version) as u32,
        8,
    );
    Ok(emitter)
}

fn exit_identity(emitter: &mut Emitter, source: &ExitStateMap, reason: NativeExitReason) {
    let scratch = source.abi.reserved().link_scratch[0];
    // The two adjacent u32 fields are one little-endian store. Besides being
    // smaller, this keeps the AArch64 metadata access within scaled LDR/STR's
    // 64-bit immediate range above the 16 KiB spill arena.
    const _: () = assert!(
        offset_of!(NativeFrame<'static>, exit_reason)
            == offset_of!(NativeFrame<'static>, exit_state_map) + 4
    );
    const _: () = assert!(offset_of!(NativeFrame<'static>, exit_state_map) % 8 == 0);
    emitter.constant(
        scratch,
        u64::from(source.site.state_map) | ((reason as u64) << 32),
        8,
    );
    emitter.memory(
        false,
        RegisterClass::Integer,
        scratch,
        offset_of!(NativeFrame<'static>, exit_state_map) as u32,
        8,
    );
}

fn exit_continuation(emitter: &mut Emitter, continuation: usize) {
    let scratch = emitter.abi.reserved().link_scratch[0];
    emitter.memory(
        true,
        RegisterClass::Integer,
        scratch,
        continuation as u32,
        8,
    );
    emitter.jump_register(scratch);
}

fn temporary_register(abi: HostAbi) -> u8 {
    match abi {
        HostAbi::X86_64 => 0,
        HostAbi::Aarch64 => 17,
    }
}

fn emit_operands(emitter: &mut Emitter, abi: HostAbi, load: bool, mut operands: Vec<Operand>) {
    let pointer = abi.reserved().link_scratch[0];
    let temporary = temporary_register(abi);
    let borrow = abi == HostAbi::X86_64
        && operands
            .iter()
            .any(|operand| !matches!(operand.location, ValueLocation::Register { .. }));
    // Reuse each field pointer for contiguous X/V elements. When loading, a
    // bound RAX must be initialized after all memory-to-memory transfers.
    operands.sort_by_key(|operand| {
        (
            load && borrow
                && matches!(
                    operand.location,
                    ValueLocation::Register {
                        class: RegisterClass::Integer,
                        index: 0
                    }
                ),
            operand.pointer,
        )
    });
    if borrow && !load {
        emitter.memory(false, RegisterClass::Integer, temporary, BORROW_SAVE, 8);
        // Canonical stores may refer to RAX after it is borrowed for a spill.
        for operand in &mut operands {
            if matches!(
                operand.location,
                ValueLocation::Register {
                    class: RegisterClass::Integer,
                    index: 0
                }
            ) {
                operand.location = ValueLocation::Spill {
                    offset: BORROW_SAVE,
                    bytes: operand.bytes,
                };
            }
        }
    }
    let mut last_pointer = None;
    for operand in operands {
        if last_pointer != Some(operand.pointer) {
            emitter.memory(true, RegisterClass::Integer, pointer, operand.pointer, 8);
            last_pointer = Some(operand.pointer);
        }
        match operand.location {
            ValueLocation::Register { class, index } => {
                emitter.memory_at(load, class, index, pointer, operand.offset, operand.bytes)
            }
            ValueLocation::Spill { .. } | ValueLocation::Constant(_) => {
                let part = operand.bytes.min(8);
                for delta in (0..operand.bytes).step_by(usize::from(part)) {
                    if load {
                        emitter.memory_at(
                            true,
                            RegisterClass::Integer,
                            temporary,
                            pointer,
                            operand.offset + u32::from(delta),
                            part,
                        );
                    } else {
                        match operand.location {
                            ValueLocation::Spill { offset, .. } => emitter.memory(
                                true,
                                RegisterClass::Integer,
                                temporary,
                                offset + u32::from(delta),
                                part,
                            ),
                            ValueLocation::Constant(value) => emitter.constant(
                                temporary,
                                (value.get() >> (delta * 8)) as u64,
                                part,
                            ),
                            _ => unreachable!(),
                        }
                    }
                    if load {
                        let ValueLocation::Spill { offset, .. } = operand.location else {
                            unreachable!("validated ingress cannot be a constant")
                        };
                        emitter.memory(
                            false,
                            RegisterClass::Integer,
                            temporary,
                            offset + u32::from(delta),
                            part,
                        );
                    } else {
                        emitter.memory_at(
                            false,
                            RegisterClass::Integer,
                            temporary,
                            pointer,
                            operand.offset + u32::from(delta),
                            part,
                        );
                    }
                }
            }
        }
    }
    if borrow && !load {
        emitter.memory(true, RegisterClass::Integer, temporary, BORROW_SAVE, 8);
    }
}
