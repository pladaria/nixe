//! Cold memory completion. No native PC, frame, code owner or borrowed
//! metadata survives preparation; no guest instruction is replayed.
//! Arm transfer/writeback rules shared with memory_lowering:
//! https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/LDR--immediate---Load-Register--immediate--
//! https://developer.arm.com/documentation/ddi0602/2024-03/SIMD-FP-Instructions/LDR--immediate--SIMD-FP---Load-SIMD-FP-Register--immediate-offset--
//! https://documentation-service.arm.com/static/62a304f231ea212bb662321d#page=22
//! Pair loads commit both destinations together; stores preserve their prefix:
//! https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/LDP--Load-pair-of-registers-
//! SIMD structures commit each element, including 64-bit arrangement clearing:
//! https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85

use super::access;
use crate::{
    abi::{BlockKey, NativeFrame},
    lifetime::Fault,
};
use nixe_cpu::{
    decode::a64::{A64Instruction, fp_simd, memory},
    exclusive::ExclusiveMonitorState,
    memory::{
        AtomicRmwKind, CpuMemory, DataAccessFault, DataAccessFaultReason, DataAccessKind,
        MemoryAccess, MemoryAccessClass, MemoryAccessSize, MemoryAlignment, MemoryOrdering,
        MemoryValue,
    },
    semantics::a64::{
        LoadSpec, ScalarTransfer, SimdMemoryMode, SimdMemoryShape, atomic_ordering,
        atomic_rmw_kind, compare_exchange_pair_sizes, exclusive_transfer_sizes, literal_load,
        pair_transfer, scalar_transfer, signed_immediate, simd_multiple_structure_shape,
        simd_single_structure_shape,
    },
    state::a64::A64State,
};

enum Transfer {
    General { register: u8, load: LoadSpec },
    Vector(u8),
    Store(MemoryValue),
}

enum Transfers {
    ExclusiveLoad {
        register: u8,
        second: Option<u8>,
        element_size: MemoryAccessSize,
    },
    Rmw {
        register: u8,
        kind: AtomicRmwKind,
        operand: MemoryValue,
    },
    CompareExchange {
        register: u8,
        element_size: MemoryAccessSize,
        expected: MemoryValue,
        replacement: MemoryValue,
    },
    Single(Transfer),
    Pair {
        items: [Transfer; 2],
        start: usize,
        retained: Option<MemoryValue>,
    },
    Structure {
        shape: SimdMemoryShape,
        register: u8,
        load: bool,
        start: u16,
    },
}

pub(crate) struct Completion {
    key: BlockKey,
    address: nixe_memory::GuestVirtualAddress,
    descriptor: MemoryAccess,
    transfers: Transfers,
    writeback: Option<(u8, u64)>,
}

#[derive(Debug)]
pub(crate) enum Error {
    Data(DataAccessFault),
    Internal(&'static str),
}
impl From<DataAccessFault> for Error {
    fn from(fault: DataAccessFault) -> Self {
        Self::Data(fault)
    }
}

fn perform(
    transfer: &Transfer,
    memory: &dyn CpuMemory,
    key: BlockKey,
    address: nixe_memory::GuestVirtualAddress,
    descriptor: MemoryAccess,
) -> Result<Option<MemoryValue>, Error> {
    if let Transfer::Store(value) = transfer {
        memory.write(key.address_space, address, descriptor, *value)?;
        return Ok(None);
    }
    read_value(memory, key, address, descriptor).map(Some)
}

fn read_value(
    memory: &dyn CpuMemory,
    key: BlockKey,
    address: nixe_memory::GuestVirtualAddress,
    descriptor: MemoryAccess,
) -> Result<MemoryValue, Error> {
    let value = memory.read(key.address_space, address, descriptor)?.value;
    if value.size() != descriptor.size {
        return Err(DataAccessFault::new(
            key.address_space,
            address,
            DataAccessKind::Read,
            DataAccessFaultReason::ValueSizeMismatch,
        )
        .into());
    }
    Ok(value)
}

fn commit(transfer: Transfer, state: &mut A64State, value: Option<MemoryValue>) {
    match transfer {
        Transfer::Store(_) => {}
        Transfer::Vector(register) => {
            state.set_vector(register, value.unwrap().bits());
        }
        Transfer::General { register, load } => {
            let value = value.unwrap();
            let raw = value.bits() as u64;
            let value = if load.signed {
                signed_immediate(raw, (value.size().bytes() * 8) as u8) as u64
            } else {
                raw
            };
            if register != 31 {
                state.general_register_storage_mut()[register as usize] =
                    if load.destination_bits == 32 {
                        u64::from(value as u32)
                    } else {
                        value
                    };
            }
        }
    }
}

impl Completion {
    /// Prepare only after Cold resolution and successful fault reconstruction,
    /// while the owning Invocation still protects the published instruction.
    /// A pair resumes at its published subaccess, retaining the first load's
    /// uncommitted bits. Structures resume after the reconstructed committed
    /// prefix; store sources remain in canonical vectors until consumption.
    ///
    /// # Safety
    /// frame contains this fault's reconstructed canonical state. The owner
    /// will not run more guest code before consuming the returned completion.
    /// retained_read is this fault's Reconstructed.completed_read, not a new read.
    pub(crate) unsafe fn prepare(
        frame: &NativeFrame<'_>,
        fault: &Fault<'_>,
        retained_read: Option<u128>,
    ) -> Result<Self, &'static str> {
        let key = fault.instruction().key.block_key();
        if frame.host_fp.saved != 0 || unsafe { *frame.canonical.pc } != key.pc.get() {
            return Err("cold completion requires reconstructed state and finished FP ownership");
        }
        let instruction = access::instruction(fault)?;
        let read = |index: u8, sp: bool| -> Result<u64, &'static str> {
            Ok(unsafe {
                if index == 31 {
                    if sp { *frame.canonical.sp } else { 0 }
                } else {
                    *frame.canonical.x.add(index as usize)
                }
            })
        };
        let access = access::decode_access(instruction, key, fault.record, &read)?;
        if retained_read.is_some() != fault.record.completed_read.is_some() {
            return Err("cold pair completion is missing or has unexpected retained read bits");
        }
        if let A64Instruction::Memory(
            i @ (memory::Instruction::LoadExclusive(_) | memory::Instruction::LoadExclusivePair(_)),
        ) = instruction
        {
            let f = i.operands();
            let pair = matches!(i, memory::Instruction::LoadExclusivePair(_));
            let (element_size, total_size) =
                exclusive_transfer_sizes(f.size, pair).ok_or("invalid cold exclusive load size")?;
            // An aligned X pair is confined to one page. Once its first read
            // succeeds, the memory execution lease prevents that RAM page
            // from turning into an ineligible mapping before the second read.
            // Do not silently repeat the first observation if a provider
            // violates that contract and requests Cold at the second site.
            if fault.record.subaccess != 0 {
                return Err("exclusive pair second read unexpectedly requires cold completion");
            }
            return Ok(Self {
                key,
                address: access.address,
                descriptor: MemoryAccess::new(
                    total_size,
                    MemoryAlignment::Natural,
                    if f.ordered {
                        MemoryOrdering::Acquire
                    } else {
                        MemoryOrdering::Relaxed
                    },
                    MemoryAccessClass::Exclusive,
                ),
                transfers: Transfers::ExclusiveLoad {
                    register: f.rt,
                    second: pair.then_some(f.rt2),
                    element_size,
                },
                writeback: None,
            });
        }
        if let A64Instruction::Memory(memory::Instruction::AtomicReadModifyWrite(f)) = instruction {
            return Ok(Self {
                key,
                address: access.address,
                descriptor: MemoryAccess::new(
                    access.size,
                    MemoryAlignment::Natural,
                    // RMW acquire semantics are suppressed when Rt is XZR.
                    // https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85#page=438
                    atomic_ordering(f.acquire && f.rt != 31, f.release),
                    MemoryAccessClass::Atomic,
                ),
                transfers: Transfers::Rmw {
                    register: f.rt,
                    kind: atomic_rmw_kind(f.atomic_opcode)
                        .ok_or("invalid cold atomic RMW operation")?,
                    operand: MemoryValue::from_bits(access.size, u128::from(read(f.rm, false)?)),
                },
                writeback: None,
            });
        }
        if let A64Instruction::Memory(
            i @ (memory::Instruction::CompareAndSwap(_)
            | memory::Instruction::CompareAndSwapPair(_)),
        ) = instruction
        {
            let f = i.operands();
            let pair = matches!(i, memory::Instruction::CompareAndSwapPair(_));
            let element_size = if pair {
                compare_exchange_pair_sizes(f.size)
                    .ok_or("invalid cold CASP size")?
                    .0
            } else {
                access.size
            };
            let operand = |first| -> Result<MemoryValue, &'static str> {
                let low =
                    MemoryValue::from_bits(element_size, u128::from(read(first, false)?)).bits();
                let high = if pair {
                    u128::from(read(first + 1, false)?) << (element_size.bytes() * 8)
                } else {
                    0
                };
                Ok(MemoryValue::from_bits(access.size, low | high))
            };
            return Ok(Self {
                key,
                address: access.address,
                descriptor: MemoryAccess::new(
                    access.size,
                    MemoryAlignment::Natural,
                    atomic_ordering(f.acquire, f.release),
                    MemoryAccessClass::Atomic,
                ),
                transfers: Transfers::CompareExchange {
                    register: f.rm,
                    element_size,
                    expected: operand(f.rm)?,
                    replacement: operand(f.rt)?,
                },
                writeback: None,
            });
        }
        if let A64Instruction::FpSimd(instruction) = instruction {
            use fp_simd::Instruction::*;
            let structure = match instruction {
                MemorySingleStructure(f) | MemorySingleStructurePostIndex(f) => Some((
                    simd_single_structure_shape(f).ok_or("invalid cold single structure")?,
                    matches!(instruction, MemorySingleStructurePostIndex(_)),
                )),
                MemoryMultipleStructures(f) | MemoryMultipleStructuresPostIndex(f) => Some((
                    simd_multiple_structure_shape(f).ok_or("invalid cold multiple structure")?,
                    matches!(instruction, MemoryMultipleStructuresPostIndex(_)),
                )),
                _ => None,
            };
            if let Some((shape, post_index)) = structure {
                let f = instruction.operands();
                let writeback = if post_index {
                    let offset = if f.rm == 31 {
                        u64::from(shape.immediate_post_index)
                    } else {
                        read(f.rm, false)?
                    };
                    Some((f.rn, read(f.rn, true)?.wrapping_add(offset)))
                } else {
                    None
                };
                return Ok(Self {
                    key,
                    address: access.address,
                    descriptor: MemoryAccess {
                        alignment: MemoryAlignment::Unaligned,
                        ..MemoryAccess::normal(shape.element_size)
                    },
                    transfers: Transfers::Structure {
                        shape,
                        register: f.rd,
                        load: f.load,
                        start: fault.record.subaccess,
                    },
                    writeback,
                });
            }
        }
        let pair = match instruction {
            A64Instruction::Memory(memory::Instruction::Pair(f)) => Some((
                f.rn,
                f.mode,
                f.immediate_7,
                [f.rt, f.rt2],
                f.load,
                Some(pair_transfer(f.size, f.load).ok_or("invalid cold pair")?.1),
            )),
            A64Instruction::FpSimd(fp_simd::Instruction::MemoryPair(f)) => {
                Some((f.rn, f.mode, f.immediate_7, [f.rd, f.rt2], f.load, None))
            }
            _ => None,
        };
        if let Some((rn, mode, immediate, registers, load, scalar)) = pair {
            let transfer = |register| -> Result<Transfer, &'static str> {
                Ok(if load {
                    if let Some(load) = scalar {
                        Transfer::General { register, load }
                    } else {
                        Transfer::Vector(register)
                    }
                } else {
                    let bits = if scalar.is_some() {
                        u128::from(read(register, false)?)
                    } else {
                        unsafe { *frame.canonical.vector.add(register as usize) }
                    };
                    Transfer::Store(MemoryValue::from_bits(access.size, bits))
                })
            };
            let writeback = if matches!(mode, 1 | 3) {
                Some((
                    rn,
                    read(rn, true)?.wrapping_add_signed(
                        signed_immediate(u64::from(immediate), 7) * access.size.bytes() as i64,
                    ),
                ))
            } else {
                None
            };
            return Ok(Self {
                key,
                address: access.address,
                descriptor: MemoryAccess {
                    alignment: MemoryAlignment::Unaligned,
                    ..MemoryAccess::normal(access.size)
                },
                transfers: Transfers::Pair {
                    items: [transfer(registers[0])?, transfer(registers[1])?],
                    start: usize::from(fault.record.subaccess),
                    retained: retained_read.map(|bits| MemoryValue::from_bits(access.size, bits)),
                },
                writeback,
            });
        }
        let mut ordering = MemoryOrdering::Relaxed;
        let (transfer, writeback) = match instruction {
            A64Instruction::Memory(instruction)
                if crate::memory_lowering::is_scalar(instruction) =>
            {
                use memory::Instruction::*;
                let f = instruction.operands();
                let transfer = match instruction {
                    Literal(_) => {
                        ScalarTransfer::Load(literal_load(f.size).ok_or("invalid cold literal")?.1)
                    }
                    LoadAcquire(_) => {
                        ordering = MemoryOrdering::Acquire;
                        ScalarTransfer::Load(LoadSpec::unsigned(access.size))
                    }
                    StoreRelease(_) => {
                        ordering = MemoryOrdering::Release;
                        ScalarTransfer::Store
                    }
                    _ => {
                        scalar_transfer(f.opc, access.size).ok_or("invalid cold scalar transfer")?
                    }
                };
                let transfer = match transfer {
                    ScalarTransfer::Load(load) => Transfer::General {
                        register: f.rt,
                        load,
                    },
                    ScalarTransfer::Store => Transfer::Store(MemoryValue::from_bits(
                        access.size,
                        u128::from(read(f.rt, false)?),
                    )),
                };
                let writeback = if matches!(instruction, PreIndex(_) | PostIndex(_)) {
                    Some((
                        f.rn,
                        read(f.rn, true)?
                            .wrapping_add_signed(signed_immediate(u64::from(f.immediate_9), 9)),
                    ))
                } else {
                    None
                };
                (transfer, writeback)
            }
            A64Instruction::FpSimd(instruction) => {
                use fp_simd::Instruction::*;
                if !matches!(
                    instruction,
                    MemoryUnsigned(_)
                        | MemoryUnscaled(_)
                        | MemoryPreIndex(_)
                        | MemoryPostIndex(_)
                        | MemoryRegister(_)
                ) {
                    return Err("compound SIMD cold completion is not implemented");
                }
                let f = instruction.operands();
                let transfer = if f.load {
                    Transfer::Vector(f.rd)
                } else {
                    Transfer::Store(MemoryValue::from_bits(access.size, unsafe {
                        *frame.canonical.vector.add(f.rd as usize)
                    }))
                };
                let writeback = if matches!(instruction, MemoryPreIndex(_) | MemoryPostIndex(_)) {
                    Some((
                        f.rn,
                        read(f.rn, true)?
                            .wrapping_add_signed(signed_immediate(u64::from(f.immediate_9), 9)),
                    ))
                } else {
                    None
                };
                (transfer, writeback)
            }
            _ => return Err("compound or unsupported cold completion is not implemented"),
        };
        Ok(Self {
            key,
            address: access.address,
            descriptor: MemoryAccess::new(
                access.size,
                // Ordered byte accesses still carry the Natural descriptor,
                // although every address satisfies their one-byte alignment.
                if access.alignment > 1 || ordering != MemoryOrdering::Relaxed {
                    MemoryAlignment::Natural
                } else {
                    MemoryAlignment::Unaligned
                },
                ordering,
                MemoryAccessClass::Normal,
            ),
            transfers: Transfers::Single(transfer),
            writeback,
        })
    }

    /// Consume once, outside native execution and its code/memory leases.
    /// The memory provider owns ordering, mappings and device synchronization.
    /// Only full success commits base writeback and PC. Single/pair destinations
    /// stay PRE on error; structures retain every successfully loaded element.
    /// Earlier stores/device effects survive; errors never imply a retry.
    /// `exclusive` is the persistent thread monitor, after the escaped frame's
    /// pending exclusive-load handoff, not a new per-completion monitor.
    pub(crate) fn complete(
        self,
        state: &mut A64State,
        memory: &dyn CpuMemory,
        exclusive: &mut ExclusiveMonitorState,
    ) -> Result<(), Error> {
        if state.pc() != self.key.pc.get() {
            return Err(Error::Internal(
                "cold completion PC no longer matches its instruction",
            ));
        }
        match self.transfers {
            Transfers::ExclusiveLoad {
                register,
                second,
                element_size,
            } => {
                let (read, reservation) =
                    memory.load_exclusive(self.key.address_space, self.address, self.descriptor)?;
                if read.value.size() != self.descriptor.size
                    || reservation.expected != read.value
                    || usize::from(reservation.access_size) != self.descriptor.size.bytes()
                {
                    return Err(Error::Internal(
                        "cold exclusive load returned an inconsistent reservation",
                    ));
                }
                exclusive.reserve(reservation);
                commit(
                    Transfer::General {
                        register,
                        load: LoadSpec::unsigned(element_size),
                    },
                    state,
                    Some(MemoryValue::from_bits(element_size, read.value.bits())),
                );
                if let Some(register) = second {
                    commit(
                        Transfer::General {
                            register,
                            load: LoadSpec::unsigned(element_size),
                        },
                        state,
                        Some(MemoryValue::from_bits(
                            element_size,
                            read.value.bits() >> (element_size.bytes() * 8),
                        )),
                    );
                }
            }
            Transfers::Rmw {
                register,
                kind,
                operand,
            } => {
                let value = memory
                    .atomic_read_modify_write(
                        self.key.address_space,
                        self.address,
                        self.descriptor,
                        kind,
                        operand,
                    )?
                    .previous;
                if value.size() != self.descriptor.size {
                    return Err(Error::Internal(
                        "cold RMW returned an incorrect result width",
                    ));
                }
                commit(
                    Transfer::General {
                        register,
                        load: LoadSpec::unsigned(value.size()),
                    },
                    state,
                    Some(value),
                );
            }
            Transfers::CompareExchange {
                register,
                element_size,
                expected,
                replacement,
            } => {
                let value = memory
                    .atomic_compare_exchange(
                        self.key.address_space,
                        self.address,
                        self.descriptor,
                        expected,
                        replacement,
                    )?
                    .previous;
                if value.size() != self.descriptor.size {
                    return Err(Error::Internal(
                        "cold CAS returned an incorrect result width",
                    ));
                }
                if element_size != value.size() {
                    commit(
                        Transfer::General {
                            register: register + 1,
                            load: LoadSpec::unsigned(element_size),
                        },
                        state,
                        Some(MemoryValue::from_bits(
                            element_size,
                            value.bits() >> (element_size.bytes() * 8),
                        )),
                    );
                }
                commit(
                    Transfer::General {
                        register,
                        load: LoadSpec::unsigned(element_size),
                    },
                    state,
                    Some(MemoryValue::from_bits(element_size, value.bits())),
                );
            }
            Transfers::Single(transfer) => {
                let value = perform(&transfer, memory, self.key, self.address, self.descriptor)?;
                commit(transfer, state, value);
            }
            Transfers::Pair {
                items,
                start,
                retained,
            } => {
                let mut values = [retained, None];
                for index in start..2 {
                    let address = self
                        .address
                        .wrapping_add(((index - start) * self.descriptor.size.bytes()) as u64);
                    values[index] =
                        perform(&items[index], memory, self.key, address, self.descriptor)?;
                }
                // No destination changes until both reads succeed. Stores were
                // committed in order by perform; failure never rolls them back.
                for (transfer, value) in items.into_iter().zip(values) {
                    commit(transfer, state, value);
                }
            }
            Transfers::Structure {
                shape,
                register,
                load,
                start,
            } => {
                let registers = u16::from(shape.structure_registers);
                let count = u16::from(shape.transfer_bytes) / shape.element_size.bytes() as u16;
                let lane_bits = shape.element_size.bytes() * 8;
                let mask = (1u128 << lane_bits) - 1;
                for index in start..count {
                    let (offset, lane) = match shape.mode {
                        SimdMemoryMode::Multiple => shape.multiple_element(index as u8),
                        SimdMemoryMode::Lane(lane) => ((index % registers) as u8, lane),
                        SimdMemoryMode::Replicate => ((index % registers) as u8, 0),
                    };
                    let register = (register + offset) & 31;
                    let shift = usize::from(lane) * lane_bits;
                    let address = self
                        .address
                        .wrapping_add(u64::from(index - start) * shape.element_size.bytes() as u64);
                    let previous = state.vector(register).unwrap();
                    if load {
                        let value = read_value(memory, self.key, address, self.descriptor)?.bits();
                        let mut vector = if shape.mode == SimdMemoryMode::Replicate {
                            let mut vector = 0;
                            for lane in 0..shape.elements_per_register {
                                vector |= value << (usize::from(lane) * lane_bits);
                            }
                            vector
                        } else {
                            (previous & !(mask << shift)) | (value << shift)
                        };
                        if shape.mode == SimdMemoryMode::Multiple && shape.vector_bytes == 8 {
                            vector &= u128::from(u64::MAX);
                        }
                        // Commit before issuing the next access. Reconstruction
                        // already committed all elements before start.
                        state.set_vector(register, vector);
                    } else {
                        memory.write(
                            self.key.address_space,
                            address,
                            self.descriptor,
                            MemoryValue::from_bits(shape.element_size, previous >> shift),
                        )?;
                    }
                }
            }
        }
        if let Some((register, value)) = self.writeback {
            if register == 31 {
                *state.stack_pointer_storage_mut() = value;
            } else {
                state.general_register_storage_mut()[register as usize] = value;
            }
        }
        state.set_pc(self.key.pc.get().wrapping_add(4));
        Ok(())
    }
}
