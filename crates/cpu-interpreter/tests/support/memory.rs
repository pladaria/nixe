//! ISA tests exercise the canonical machine through the instruction contract.
use nixe_cpu::memory::*;
use nixe_memory::{
    AccessOrdering, AccessResult, AddressSpace, AddressSpaceId, AddressWidth, AtomicAccess,
    AtomicOperation, AtomicTransform, GuestVirtualAddress, Machine, PAGE_SIZE, Reservation,
};
use std::sync::Arc;

thread_local! { pub static UNMAPPED: Memory = Memory::new(); }

pub struct Memory {
    pub space: Arc<AddressSpace>,
}
impl Memory {
    pub fn new() -> Self {
        Self {
            space: AddressSpace::new(Machine::new(PAGE_SIZE).unwrap(), AddressWidth::Bits39)
                .unwrap(),
        }
    }
    pub fn mapped(base: u64, initial: &[u8]) -> Self {
        let memory = Self::new();
        let allocation = memory
            .space
            .machine()
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        memory
            .space
            .map(
                GuestVirtualAddress::new(base),
                allocation,
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        if !initial.is_empty() {
            memory
                .space
                .write(GuestVirtualAddress::new(base), initial)
                .unwrap()
                .into_ready()
                .unwrap();
        }
        memory
    }
    fn validate(
        &self,
        space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        kind: DataAccessKind,
    ) -> Result<(), DataAccessFault> {
        let alignment = match access.alignment {
            MemoryAlignment::Unaligned => 1,
            MemoryAlignment::Natural => access.size.bytes(),
            MemoryAlignment::Bytes2 => 2,
            MemoryAlignment::Bytes4 => 4,
            MemoryAlignment::Bytes8 => 8,
            MemoryAlignment::Bytes16 => 16,
        };
        if !address.get().is_multiple_of(alignment as u64) {
            return Err(DataAccessFault::new(
                space,
                address,
                kind,
                DataAccessFaultReason::Misaligned {
                    required_alignment: alignment as u8,
                },
            ));
        }
        Ok(())
    }
    fn atomic(
        &self,
        space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        operation: AtomicOperation,
    ) -> Result<AccessResult, DataAccessFault> {
        self.validate(space, address, access, DataAccessKind::Write)?;
        let ordering = match access.ordering {
            MemoryOrdering::Relaxed => AccessOrdering::Relaxed,
            MemoryOrdering::Acquire => AccessOrdering::Acquire,
            MemoryOrdering::Release => AccessOrdering::Release,
            MemoryOrdering::AcquireRelease => AccessOrdering::AcquireRelease,
            MemoryOrdering::SequentiallyConsistent => AccessOrdering::SequentiallyConsistent,
        };
        self.space
            .atomic(
                address,
                AtomicAccess {
                    bytes: access.size.bytes() as u8,
                    ordering,
                    operation,
                },
            )
            .map_err(|e| DataAccessFault::new(space, address, DataAccessKind::Write, (&e).into()))
            .map(|value| value.into_ready().expect("ISA fixture contains only RAM"))
    }
}
impl InstructionMemoryAccess for Memory {
    type Reservation = Reservation;
    fn read(
        &self,
        space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<DataReadResult, DataAccessFault> {
        self.validate(space, address, access, DataAccessKind::Read)?;
        self.space
            .read_value(address, access.size.bytes())
            .map_err(|e| DataAccessFault::new(space, address, DataAccessKind::Read, (&e).into()))
            .map(|bits| DataReadResult {
                value: MemoryValue::from_bits(access.size, bits.into_ready().unwrap()),
                region: MemoryRegionKind::Ram,
            })
    }
    fn write(
        &self,
        space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
    ) -> Result<DataWriteResult, DataAccessFault> {
        self.validate(space, address, access, DataAccessKind::Write)?;
        self.space
            .write_value(address, access.size.bytes(), value.bits())
            .map_err(|e| DataAccessFault::new(space, address, DataAccessKind::Write, (&e).into()))?
            .into_ready()
            .unwrap();
        Ok(DataWriteResult {
            region: MemoryRegionKind::Ram,
        })
    }
    fn atomic_read_modify_write(
        &self,
        space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        kind: AtomicRmwKind,
        operand: MemoryValue,
    ) -> Result<AtomicMemoryResult, DataAccessFault> {
        let transform = match kind {
            AtomicRmwKind::Add => AtomicTransform::Add,
            AtomicRmwKind::Clear => AtomicTransform::Clear,
            AtomicRmwKind::Xor => AtomicTransform::Xor,
            AtomicRmwKind::Set => AtomicTransform::Set,
            AtomicRmwKind::SignedMaximum => AtomicTransform::SignedMaximum,
            AtomicRmwKind::SignedMinimum => AtomicTransform::SignedMinimum,
            AtomicRmwKind::UnsignedMaximum => AtomicTransform::UnsignedMaximum,
            AtomicRmwKind::UnsignedMinimum => AtomicTransform::UnsignedMinimum,
            AtomicRmwKind::Swap => AtomicTransform::Swap,
        };
        let AccessResult::Atomic { previous, written } = self.atomic(
            space,
            address,
            access,
            AtomicOperation::ReadModifyWrite {
                transform,
                operand: operand.bits(),
            },
        )?
        else {
            unreachable!()
        };
        Ok(AtomicMemoryResult {
            previous: MemoryValue::from_bits(access.size, previous),
            stored: written,
            region: MemoryRegionKind::Ram,
        })
    }
    fn atomic_compare_exchange(
        &self,
        space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        expected: MemoryValue,
        replacement: MemoryValue,
    ) -> Result<AtomicMemoryResult, DataAccessFault> {
        let AccessResult::Atomic { previous, written } = self.atomic(
            space,
            address,
            access,
            AtomicOperation::CompareExchange {
                expected: expected.bits(),
                replacement: replacement.bits(),
            },
        )?
        else {
            unreachable!()
        };
        Ok(AtomicMemoryResult {
            previous: MemoryValue::from_bits(access.size, previous),
            stored: written,
            region: MemoryRegionKind::Ram,
        })
    }
    fn maintain_cache(
        &self,
        space: AddressSpaceId,
        kind: CacheMaintenanceKind,
        address: Option<GuestVirtualAddress>,
    ) -> Result<(), DataAccessFault> {
        if kind == CacheMaintenanceKind::InstructionInvalidate {
            let (address, bytes) = address
                .map_or((GuestVirtualAddress::MIN, (1u64 << 39) - 1), |address| {
                    (address, 4)
                });
            self.space.invalidate_code(address, bytes).map_err(|e| {
                DataAccessFault::new(space, address, DataAccessKind::Write, (&e).into())
            })?;
        }
        Ok(())
    }
    fn load_exclusive(
        &self,
        space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<(DataReadResult, Reservation), DataAccessFault> {
        let AccessResult::ExclusiveRead { value, reservation } =
            self.atomic(space, address, access, AtomicOperation::LoadExclusive)?
        else {
            unreachable!()
        };
        Ok((
            DataReadResult {
                value: MemoryValue::from_bits(access.size, value),
                region: MemoryRegionKind::Ram,
            },
            reservation,
        ))
    }
    fn store_exclusive(
        &self,
        space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
        reservation: Reservation,
    ) -> Result<(DataWriteResult, bool), DataAccessFault> {
        let AccessResult::ExclusiveWritten(written) = self.atomic(
            space,
            address,
            access,
            AtomicOperation::StoreExclusive {
                reservation,
                replacement: value.bits(),
            },
        )?
        else {
            unreachable!()
        };
        Ok((
            DataWriteResult {
                region: MemoryRegionKind::Ram,
            },
            written,
        ))
    }
}
pub struct Timer;
impl nixe_cpu::execution::ArchitecturalTimer for Timer {
    fn snapshot(&self) -> nixe_cpu::execution::TimerSnapshot {
        nixe_cpu::execution::TimerSnapshot {
            counter: 0,
            frequency: 0,
        }
    }
}
