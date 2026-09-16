//! Reconstruct the semantic subaccess, not Linux's first inaccessible byte.
//! Addressing follows the same Arm rules as memory_lowering (DDI 0602):
//! https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/LDR--register---Load-Register--register--
//! https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/LDP--Load-pair-of-registers-
//! https://documentation-service.arm.com/static/6245c734b059dc5ff9a8bdab#page=1363

use super::{captured_state, read_location};
use crate::{
    abi::{GuestValue, NativeFrame},
    lifetime::{Fault, unit},
};
use nixe_cpu::{
    decode::{
        self, DecodeResult,
        a64::{A64Instruction, fp_simd, memory},
    },
    location::LocationDescriptor,
    memory::{
        CpuMemory, DataAccessFault, DataAccessFaultReason, DataAccessKind, DirectFaultResolution,
        MemoryAccessSize,
    },
    semantics::a64::{
        ScalarTransfer, compare_exchange_pair_sizes, exclusive_transfer_sizes, literal_load,
        memory_size, pair_transfer, scalar_transfer, signed_immediate, simd_memory_access_size,
        simd_multiple_structure_shape, simd_pair_access_size, simd_single_structure_shape,
    },
};
use nixe_cpu_direct_memory::CapturedFault;
use nixe_memory::{AddressSpaceId, DirectAddressSpaceView, GuestVirtualAddress};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Access {
    pub address: GuestVirtualAddress,
    pub size: MemoryAccessSize,
    pub kind: DataAccessKind,
    pub alignment: u8,
}

impl Access {
    /// Check before asking the memory authority for mapping repair. A guard
    /// access due to guest misalignment or range overflow can never be retried.
    pub fn guest_fault(self, space: AddressSpaceId) -> Option<DataAccessFault> {
        let bytes = self.size.bytes() as u64;
        let reason = if self.address.get() & (u64::from(self.alignment) - 1) != 0 {
            DataAccessFaultReason::Misaligned {
                required_alignment: self.alignment,
            }
        } else if self.address.get().checked_add(bytes - 1).is_none() {
            DataAccessFaultReason::AddressOverflow
        } else {
            return None;
        };
        Some(DataAccessFault::new(space, self.address, self.kind, reason))
    }
}

/// Inspect and consult the memory authority on the normal dispatcher stack.
/// Cold requests carry no completed device operation; the owner must escape
/// before reconstructing state and executing the typed subaccess.
///
/// # Safety
/// Same captured-frame/epoch requirements as inspect. The caller also retains
/// the memory execution lease, keeping mappings stable through resolution/retry.
pub(crate) unsafe fn resolve(
    frame: &NativeFrame<'_>,
    captured: &CapturedFault<'_>,
    fault: &Fault<'_>,
    arena: DirectAddressSpaceView,
    memory: &dyn CpuMemory,
) -> Result<(Access, DirectFaultResolution), &'static str> {
    let space = fault.record.instruction.block_key().address_space;
    if memory.direct_address_space_view(space) != Some(arena) {
        return Err("fault memory authority does not own the captured arena");
    }
    let access = unsafe { inspect(frame, captured, fault, arena)? };
    let resolution = if fault.record.access == unit::Access::CacheProbe {
        // The byte read is only a proof of the fast case. In particular, read
        // permissions and MMIO callbacks are not CIVAC semantics. The cold
        // owner validates the original address after this epoch/lease ends.
        DirectFaultResolution::Cold
    } else if let Some(fault) = access.guest_fault(space) {
        DirectFaultResolution::Fault(fault)
    } else if fault.record.access == unit::Access::Atomic {
        memory.resolve_direct_atomic_fault(space, access.address, access.size)
    } else {
        memory.resolve_direct_fault(space, access.address, access.size, access.kind)
    };
    if resolution == DirectFaultResolution::Cold
        && matches!(
            instruction(fault)?,
            A64Instruction::Memory(
                memory::Instruction::StoreExclusive(_) | memory::Instruction::StoreExclusivePair(_)
            )
        )
    {
        // This native CAS is reachable only after a same-VA/width native load
        // in the same invocation. Its lease still protects that RAM mapping:
        // a write either retries after tracking repair or reports a guest fault.
        // Aliases/incoming reservations instead leave through a typed PRE exit.
        return Err("same-invocation exclusive store unexpectedly requires cold completion");
    }
    if resolution == DirectFaultResolution::Retry
        && access
            .address
            .get()
            .checked_add(access.size.bytes() as u64)
            .is_none_or(|end| end > arena.address_space_size as u64)
    {
        return Err("memory authority requested retry of a confined guard access");
    }
    Ok((access, resolution))
}

/// Inspect without canonicalizing, merging FPSR, repairing memory or changing
/// the saved image. The caller may still return Retry after successful repair.
///
/// # Safety
/// `frame` is the unmoved captured invocation with initialized mapped spills;
/// `fault` is protected by its execution epoch and `arena` belongs to it.
pub(crate) unsafe fn inspect(
    frame: &NativeFrame<'_>,
    captured: &CapturedFault<'_>,
    fault: &Fault<'_>,
    arena: DirectAddressSpaceView,
) -> Result<Access, &'static str> {
    let map = captured_state(frame, captured, fault)?;
    if captured.integer(map.abi.reserved().arena) != Some(arena.base as u64) {
        return Err("captured arena does not match the invocation");
    }
    let key = fault.record.instruction.block_key();
    let read = |index: u8, sp: bool| -> Result<u64, &'static str> {
        if index == 31 && !sp {
            return Ok(0);
        }
        let value = if index == 31 {
            GuestValue::Sp
        } else {
            GuestValue::General(index)
        };
        if let Some(binding) = map.bindings.iter().find(|binding| binding.value == value) {
            return Ok(unsafe {
                read_location(&frame.spill, captured, map.abi, binding.location, 8)?
            } as u64);
        }
        if !map
            .dirty_live
            .intersection(value.state().unwrap())
            .is_empty()
        {
            return Err("fault address operand has no physical location");
        }
        Ok(unsafe {
            if index == 31 {
                *frame.canonical.sp
            } else {
                *frame.canonical.x.add(index as usize)
            }
        })
    };
    let access = decode_access(instruction(fault)?, key.pc.get(), fault.record, &read)?;
    // Confinement redirects an invalid start/alignment to the trailing guard;
    // a valid start may cross into that guard or another inaccessible page.
    let bytes = access.size.bytes();
    let address = access.address.get();
    let offset = if address >= arena.address_space_size as u64
        || address & (u64::from(access.alignment) - 1) != 0
    {
        arena.address_space_size
    } else {
        address as usize
    };
    let start = arena
        .base
        .checked_add(offset)
        .ok_or("host access address overflow")?;
    let end = start
        .checked_add(bytes)
        .ok_or("host access extent overflow")?;
    if !(start..end).contains(&captured.fault_address()) {
        return Err("captured fault byte is outside the reconstructed native access");
    }
    Ok(access)
}

pub(super) fn instruction(fault: &Fault<'_>) -> Result<A64Instruction, &'static str> {
    let key = fault.record.instruction.block_key();
    let bits = fault
        .unit
        .instructions
        .iter()
        .find(|instruction| instruction.key == fault.record.instruction)
        .ok_or("fault instruction is absent from the published image")?
        .bits;
    let DecodeResult::Decoded(decoded) = decode::decode(
        key.platform,
        LocationDescriptor::new(key.pc, key.profile),
        bits.into(),
    ) else {
        return Err("published fault instruction does not decode");
    };
    Ok(decode::a64::normalize(
        &decoded.instruction,
        decoded.encoding,
    ))
}

pub(super) fn decode_access(
    instruction: A64Instruction,
    pc: u64,
    record: &unit::FaultRecord,
    read: &impl Fn(u8, bool) -> Result<u64, &'static str>,
) -> Result<Access, &'static str> {
    if record.access == unit::Access::CacheProbe {
        let A64Instruction::System(instruction) = instruction else {
            return Err("cache probe has a non-system instruction");
        };
        if !crate::lcq::system::is_cache_probe(record.instruction.block_key().platform, instruction)
            || record.bytes != 1
            || record.subaccess != 0
            || record.commit_stage != 0
            || record.completed_read.is_some()
        {
            return Err("invalid cache probe fault metadata");
        }
        return Ok(Access {
            address: GuestVirtualAddress::new(read(instruction.operands().rt, false)?),
            size: MemoryAccessSize::Byte,
            kind: DataAccessKind::Read,
            alignment: 1,
        });
    }
    let index = record.subaccess;
    let mut count = 1;
    let mut paired = false;
    let mut natural_alignment = false;
    let mut minimum_alignment = 1;
    let mut atomic = false;
    let (address, size, load) = match instruction {
        A64Instruction::Memory(instruction) => {
            use memory::Instruction::*;
            let f = instruction.operands();
            if let Literal(_) = instruction {
                let (size, _) = literal_load(f.size).ok_or("invalid literal access")?;
                (
                    pc.wrapping_add_signed(signed_immediate(u64::from(f.immediate_19), 19) << 2),
                    size,
                    true,
                )
            } else if let Pair(_) = instruction {
                let (size, _) = pair_transfer(f.size, f.load).ok_or("invalid pair access")?;
                count = 2;
                paired = true;
                (
                    pair_address(read(f.rn, true)?, f.mode, f.immediate_7, size, index),
                    size,
                    f.load,
                )
            } else if let CompareAndSwapPair(_) = instruction {
                if f.rm & 1 != 0 || f.rt & 1 != 0 {
                    return Err("invalid CASP register pair");
                }
                let (_, size) = compare_exchange_pair_sizes(f.size).ok_or("invalid CASP size")?;
                atomic = true;
                natural_alignment = true;
                (read(f.rn, true)?, size, true)
            } else if matches!(instruction, CompareAndSwap(_) | AtomicReadModifyWrite(_)) {
                atomic = true;
                natural_alignment = true;
                // Typed atomics validate the read before the conditional write.
                (read(f.rn, true)?, memory_size(f.size), true)
            } else if matches!(instruction, StoreExclusive(_) | StoreExclusivePair(_)) {
                atomic = true;
                natural_alignment = true;
                let (_, size) =
                    exclusive_transfer_sizes(f.size, matches!(instruction, StoreExclusivePair(_)))
                        .ok_or("invalid exclusive store size")?;
                // A successful native load already established read access
                // under this lease. The exclusive-store guest fault is Write.
                (read(f.rn, true)?, size, false)
            } else if let LoadExclusivePair(_) = instruction {
                if !matches!(f.size, 2 | 3) || f.rt == f.rt2 {
                    return Err("unsupported exclusive pair load fault access");
                }
                natural_alignment = true;
                if f.size == 3 {
                    count = 2;
                    paired = true;
                    if index == 0 {
                        minimum_alignment = 16;
                    }
                }
                (
                    read(f.rn, true)?.wrapping_add(u64::from(index) * 8),
                    MemoryAccessSize::Doubleword,
                    true,
                )
            } else if matches!(
                instruction,
                LoadAcquire(_) | StoreRelease(_) | LoadExclusive(_)
            ) {
                natural_alignment = true;
                (
                    read(f.rn, true)?,
                    memory_size(f.size),
                    matches!(instruction, LoadAcquire(_) | LoadExclusive(_)),
                )
            } else {
                let size = memory_size(f.size);
                let transfer = scalar_transfer(f.opc, size).ok_or("invalid scalar access")?;
                let base = read(f.rn, true)?;
                let address = match instruction {
                    Unsigned(_) => {
                        base.wrapping_add(u64::from(f.immediate_12) * size.bytes() as u64)
                    }
                    Unscaled(_) | PreIndex(_) => {
                        base.wrapping_add_signed(signed_immediate(u64::from(f.immediate_9), 9))
                    }
                    PostIndex(_) => base,
                    Register(_) => {
                        register_address(base, read(f.rm, false)?, f.option, f.scaled, size)?
                    }
                    _ => return Err("unsupported scalar fault access"),
                };
                (address, size, matches!(transfer, ScalarTransfer::Load(_)))
            }
        }
        A64Instruction::FpSimd(instruction) => {
            use fp_simd::Instruction::*;
            let f = instruction.operands();
            let base = read(f.rn, true)?;
            if let MemoryPair(_) = instruction {
                let size = simd_pair_access_size(f.size).ok_or("invalid vector pair access")?;
                count = 2;
                paired = true;
                (
                    pair_address(base, f.mode, f.immediate_7, size, index),
                    size,
                    f.load,
                )
            } else if matches!(
                instruction,
                MemorySingleStructure(_) | MemorySingleStructurePostIndex(_)
            ) {
                let shape =
                    simd_single_structure_shape(f).ok_or("invalid single-structure access")?;
                count = u16::from(shape.structure_registers);
                (
                    base.wrapping_add(u64::from(index) * shape.element_size.bytes() as u64),
                    shape.element_size,
                    f.load,
                )
            } else if matches!(
                instruction,
                MemoryMultipleStructures(_) | MemoryMultipleStructuresPostIndex(_)
            ) {
                let shape = simd_multiple_structure_shape(f)
                    .ok_or("invalid multiple-structure fault access")?;
                count = u16::from(shape.transfer_bytes) / shape.element_size.bytes() as u16;
                let grouped = record.bytes != shape.element_size.bytes() as u8;
                let size = if grouped {
                    if shape.structure_registers != 1
                        || record.bytes != shape.vector_bytes
                        || !index.is_multiple_of(u16::from(shape.elements_per_register))
                        || (base & (nixe_memory::DIRECT_PAGE_SIZE as u64 - 1))
                            > nixe_memory::DIRECT_PAGE_SIZE as u64 - u64::from(shape.transfer_bytes)
                    {
                        return Err("grouped structure fault violates its single-page contract");
                    }
                    if shape.vector_bytes == 16 {
                        MemoryAccessSize::Quadword
                    } else {
                        MemoryAccessSize::Doubleword
                    }
                } else {
                    shape.element_size
                };
                (
                    base.wrapping_add(u64::from(index) * shape.element_size.bytes() as u64),
                    size,
                    f.load,
                )
            } else {
                let size = simd_memory_access_size(f.size, f.opc).ok_or("invalid vector access")?;
                let address = match instruction {
                    MemoryUnsigned(_) => {
                        base.wrapping_add(u64::from(f.immediate_12) * size.bytes() as u64)
                    }
                    MemoryUnscaled(_) | MemoryPreIndex(_) => {
                        base.wrapping_add_signed(signed_immediate(u64::from(f.immediate_9), 9))
                    }
                    MemoryPostIndex(_) => base,
                    MemoryRegister(_) => {
                        register_address(base, read(f.rm, false)?, f.option, f.scaled, size)?
                    }
                    _ => return Err("unsupported vector fault access"),
                };
                (address, size, f.load)
            }
        }
        _ => return Err("non-memory instruction has a fault record"),
    };
    let kind = if load {
        DataAccessKind::Read
    } else {
        DataAccessKind::Write
    };
    let expected_access = if atomic {
        unit::Access::Atomic
    } else if load {
        unit::Access::Read
    } else {
        unit::Access::Write
    };
    if index >= count
        || record.bytes as usize != size.bytes()
        || record.access != expected_access
        || record.commit_stage != if paired && load { 0 } else { index }
        || record.completed_read.is_some() != (paired && load && index == 1)
    {
        return Err("fault metadata disagrees with the published instruction's subaccess");
    }
    Ok(Access {
        address: GuestVirtualAddress::new(address),
        size,
        kind,
        alignment: minimum_alignment.max(if natural_alignment {
            size.bytes() as u8
        } else {
            1
        }),
    })
}

fn pair_address(base: u64, mode: u8, immediate: u8, size: MemoryAccessSize, index: u16) -> u64 {
    let first = if mode == 1 {
        base
    } else {
        base.wrapping_add_signed(signed_immediate(u64::from(immediate), 7) * size.bytes() as i64)
    };
    first.wrapping_add(u64::from(index) * size.bytes() as u64)
}

fn register_address(
    base: u64,
    raw: u64,
    option: u8,
    scaled: bool,
    size: MemoryAccessSize,
) -> Result<u64, &'static str> {
    let offset = match option {
        2 => u64::from(raw as u32),
        6 => raw as i32 as i64 as u64,
        3 | 7 => raw,
        _ => return Err("invalid address register extension"),
    };
    Ok(base.wrapping_add(if scaled {
        offset << size.bytes().trailing_zeros()
    } else {
        offset
    }))
}
