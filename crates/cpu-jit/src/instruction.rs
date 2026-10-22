//! Owned completion of an interrupted memory instruction. The first pass
//! derives effects with the shared ISA semantics under native admission. The
//! second pass consumes completed results; it never accesses guest memory.
//! https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions
use crate::{JitError, execution::CapturedAccess};
use nixe_cpu::{
    exclusive::ExclusiveMonitorState,
    execution::{ArchitecturalTimer, CpuExit, VcpuEventState},
    memory::{
        AtomicMemoryResult, AtomicRmwKind, CacheMaintenanceKind, DataAccessFault,
        DataAccessFaultReason, DataAccessKind, DataReadResult, DataWriteResult,
        InstructionMemoryAccess, MemoryAccess, MemoryAccessClass, MemoryAlignment, MemoryOrdering,
        MemoryRegionKind, MemoryValue,
    },
    profile::ProcessCpuContext,
    state::a64::A64State,
};
use nixe_cpu_interpreter::{
    InstructionStep, InterpreterContext, InterpreterError, execute_one_with_context,
};
use nixe_memory::{
    Access, AccessOperation, AccessOrdering, AccessResult, AddressSpace, AddressSpaceId,
    AllocationRange, AtomicAccess, AtomicOperation, AtomicTransform, GuestVirtualAddress, Machine,
    MemoryError, MemoryPermissions, NativeAdmission, NativeReadArea, NativeReservation,
    PendingAccess, Reservation,
};
use std::{
    cell::{Cell, RefCell},
    sync::{
        Arc,
        atomic::{Ordering, fence},
    },
};

#[derive(Clone, Copy, Eq, PartialEq)]
enum Kind {
    Read,
    Write,
    Atomic,
    ExclusiveRead,
    ExclusiveWrite,
}
impl Kind {
    fn of(operation: &AccessOperation) -> Self {
        match operation {
            AccessOperation::Read | AccessOperation::ReadNative => Self::Read,
            AccessOperation::Write(_) => Self::Write,
            AccessOperation::Atomic(AtomicAccess {
                operation: AtomicOperation::LoadExclusive,
                ..
            }) => Self::ExclusiveRead,
            AccessOperation::Atomic(AtomicAccess {
                operation: AtomicOperation::StoreExclusive { .. },
                ..
            }) => Self::ExclusiveWrite,
            AccessOperation::Atomic(_) => Self::Atomic,
            AccessOperation::ReadCode(_) | AccessOperation::ReadObserved(_) => {
                unreachable!("instruction data cannot request an observed capture")
            }
        }
    }
    fn permissions(self) -> MemoryPermissions {
        match self {
            Self::Read | Self::ExclusiveRead => MemoryPermissions::READ,
            Self::Write | Self::ExclusiveWrite => MemoryPermissions::WRITE,
            Self::Atomic => MemoryPermissions::READ_WRITE,
        }
    }
}
#[derive(Clone, Copy)]
enum Operation {
    Read,
    Write(MemoryValue),
    Atomic(AtomicAccess),
}
impl Operation {
    fn kind(self) -> Kind {
        match self {
            Self::Read => Kind::Read,
            Self::Write(_) => Kind::Write,
            Self::Atomic(AtomicAccess {
                operation: AtomicOperation::LoadExclusive,
                ..
            }) => Kind::ExclusiveRead,
            Self::Atomic(AtomicAccess {
                operation: AtomicOperation::StoreExclusive { .. },
                ..
            }) => Kind::ExclusiveWrite,
            Self::Atomic(_) => Kind::Atomic,
        }
    }
}
#[derive(Clone, Copy)]
enum Value {
    Read(u128),
    Written,
    Atomic {
        previous: u128,
        written: bool,
    },
    ExclusiveRead {
        value: u128,
        reservation: Reservation,
    },
    ExclusiveWritten(bool),
}
struct Record {
    address: GuestVirtualAddress,
    access: MemoryAccess,
    kind: Kind,
    operation: Operation,
    group: usize,
    input: Option<(Vec<AllocationRange>, AccessOperation)>,
    result: Option<Result<Value, DataAccessFault>>,
}
enum Stage<'a> {
    Capture {
        first: Option<GuestVirtualAddress>,
        reached: Cell<bool>,
        records: RefCell<Vec<Record>>,
    },
    Complete {
        records: &'a [Record],
        next: Cell<usize>,
    },
}
struct Effects<'a> {
    stage: Stage<'a>,
}
enum Outcome<'a> {
    Planned,
    Completed(&'a Value),
}

/// No borrowed CPU stack, address-space admission, core or native frame is
/// retained while an exact graphics operation is pending.
pub struct InstructionContinuation {
    space: AddressSpaceId,
    cpu: ProcessCpuContext,
    state: A64State,
    word: u32,
    records: Vec<Record>,
    next: usize,
    pending: Option<PendingAccess>,
    wake: Option<Arc<dyn Fn() + Send + Sync>>,
    reservation: Option<Reservation>,
    finished: bool,
    ready: Option<CompletedInstruction>,
    read_area: Option<NativeReadArea>,
    events: VcpuEventState,
}
pub(crate) enum ExclusiveUpdate {
    Keep,
    Clear,
    Set(NativeReservation),
}
pub struct CompletedInstruction {
    pub(crate) space: AddressSpaceId,
    pub(crate) state: A64State,
    pub(crate) exclusive: ExclusiveUpdate,
    pub(crate) read_area: Option<NativeReadArea>,
    pub stop: InstructionStep,
    pub ticks: u64,
}
impl InstructionContinuation {
    /// Cancel the unexecuted suffix without starting another operation. Preserve
    /// an already completed device failure instead of replacing it with Stop.
    pub fn cancel(mut self) -> Option<Arc<MemoryError>> {
        self.finished = true;
        self.pending
            .as_mut()
            .and_then(PendingAccess::poll)
            .and_then(Result::err)
    }
    pub(crate) fn capture(
        cpu: ProcessCpuContext,
        state: A64State,
        (mut first, word): (Option<CapturedAccess>, u32),
        space: &AddressSpace,
        admission: &NativeAdmission<'_>,
        timer: &dyn ArchitecturalTimer,
        events: &VcpuEventState,
    ) -> Result<Self, JitError> {
        let reservation = match first.as_ref().map(|first| &first.operation) {
            Some(AccessOperation::Atomic(AtomicAccess {
                operation: AtomicOperation::StoreExclusive { reservation, .. },
                ..
            })) => Some(*reservation),
            _ => None,
        };
        let mut ranges = first
            .as_mut()
            .map(|first| std::mem::take(&mut first.ranges))
            .filter(|ranges| !ranges.is_empty());
        let effects = Effects {
            stage: Stage::Capture {
                first: first.as_ref().map(|first| first.address),
                reached: Cell::new(
                    !first
                        .as_ref()
                        .is_some_and(|first| matches!(first.operation, AccessOperation::Write(_))),
                ),
                records: RefCell::new(Vec::new()),
            },
        };
        let mut monitor = ExclusiveMonitorState::default();
        if let Some(reservation) = reservation {
            monitor.reserve(Some(reservation));
        }
        let monitor = RefCell::new(monitor);
        let mut planned = state.clone();
        let result = execute_one_with_context(
            InterpreterContext::new(cpu, &effects, &monitor, timer, events),
            &mut planned,
            word,
        );
        let (stop, semantic_failure) = match result {
            Ok(stop) => (stop, false),
            Err(error) => {
                let stop = match error {
                    InterpreterError::UnsupportedInstruction {
                        source,
                        encoding,
                        disassembly,
                        coverage_id,
                    } => CpuExit::UnsupportedSemantics {
                        source,
                        encoding,
                        disassembly,
                        coverage_id,
                    },
                    InterpreterError::InvalidInstructionStream {
                        source,
                        encoding,
                        reason,
                    } => CpuExit::UnallocatedEncoding {
                        error: nixe_cpu::error::UnallocatedEncoding::new(
                            nixe_cpu::error::InstructionDiagnostic::new(source, encoding),
                            reason,
                        ),
                    },
                };
                (InstructionStep::Exit(stop), true)
            }
        };
        let Stage::Capture {
            reached, records, ..
        } = effects.stage
        else {
            unreachable!()
        };
        if !semantic_failure && !reached.get() {
            return Err(JitError(
                "instruction effects did not reach the interrupted native store".into(),
            ));
        }
        let mut records = records.into_inner();
        if semantic_failure {
            records.clear();
        }
        if !semantic_failure && first.is_some() && records.is_empty() {
            return Err(JitError(
                "interrupted instruction produced no memory effects".into(),
            ));
        }
        if !records.is_empty() {
            prepare(&mut records, first.as_ref(), &mut ranges, space, admission);
        }
        let ready = records.is_empty().then(|| CompletedInstruction {
            space: space.identity(),
            state: if semantic_failure {
                state.clone()
            } else {
                planned
            },
            exclusive: ExclusiveUpdate::Keep,
            read_area: None,
            ticks: u64::from(matches!(stop, InstructionStep::Continue)),
            stop,
        });
        Ok(Self {
            space: space.identity(),
            cpu,
            state,
            word,
            records,
            next: 0,
            pending: None,
            wake: None,
            reservation,
            finished: false,
            ready,
            read_area: None,
            events: events.clone(),
        })
    }
    pub fn set_waker(&mut self, wake: Arc<dyn Fn() + Send + Sync>) {
        if let Some(pending) = &mut self.pending {
            pending.set_waker(Arc::clone(&wake));
        }
        self.wake = Some(wake);
    }
    pub fn poll(
        &mut self,
        machine: &Machine,
        timer: &dyn ArchitecturalTimer,
    ) -> Option<Result<CompletedInstruction, Arc<MemoryError>>> {
        assert!(!self.finished, "instruction completion already consumed");
        if let Some(ready) = self.ready.take() {
            self.finished = true;
            return Some(Ok(ready));
        }
        while self.next < self.records.len() {
            let record = &mut self.records[self.next];
            let group = record.group;
            let acquire = record.access.ordering.has_acquire();
            let result = if let Some(pending) = &mut self.pending {
                let result = pending.poll()?;
                self.pending = None;
                Some(result)
            } else if let Some((ranges, operation)) = record.input.take() {
                if record.access.ordering.has_release() {
                    fence(Ordering::Release);
                }
                match machine.access_captured(&ranges, operation) {
                    Ok(Access::Ready(result)) => Some(Ok(result)),
                    Ok(Access::Pending(mut pending)) => {
                        if let Some(wake) = &self.wake {
                            pending.set_waker(Arc::clone(wake));
                        }
                        self.pending = Some(pending);
                        continue;
                    }
                    Err(error) => Some(Err(Arc::new(error))),
                }
            } else {
                None
            };
            if let Some(result) = result {
                match result {
                    Ok(result) => {
                        if let Some(area) = distribute(&mut self.records, self.next, result) {
                            self.read_area = Some(area);
                        }
                    }
                    Err(error) => {
                        self.finished = true;
                        return Some(Err(error));
                    }
                }
                if acquire {
                    fence(Ordering::Acquire);
                }
            }
            self.next += group;
        }
        self.finished = true;
        let effects = Effects {
            stage: Stage::Complete {
                records: &self.records,
                next: Cell::new(0),
            },
        };
        let mut monitor = ExclusiveMonitorState::default();
        if let Some(reservation) = self.reservation {
            monitor.reserve(Some(reservation));
        }
        let monitor = RefCell::new(monitor);
        let stop = match execute_one_with_context(
            InterpreterContext::new(self.cpu, &effects, &monitor, timer, &self.events),
            &mut self.state,
            self.word,
        ) {
            Ok(stop) => stop,
            Err(error) => {
                return Some(Err(Arc::new(MemoryError::Execution(
                    error.to_string().into(),
                ))));
            }
        };
        let Stage::Complete { next, .. } = effects.stage else {
            unreachable!()
        };
        if next.get() != self.records.len() {
            return Some(Err(Arc::new(MemoryError::Invalid(
                "instruction did not consume every captured effect",
            ))));
        }
        let exclusive = if self
            .records
            .iter()
            .any(|record| record.kind == Kind::ExclusiveRead)
        {
            match monitor.borrow().reservation().flatten() {
                Some(reservation) => match machine.export_native_reservation(reservation) {
                    Ok(proof) => ExclusiveUpdate::Set(proof),
                    Err(error) => return Some(Err(Arc::new(error))),
                },
                None => ExclusiveUpdate::Clear,
            }
        } else if self
            .records
            .iter()
            .any(|record| record.kind == Kind::ExclusiveWrite)
        {
            ExclusiveUpdate::Clear
        } else {
            ExclusiveUpdate::Keep
        };
        Some(Ok(CompletedInstruction {
            space: self.space,
            state: std::mem::take(&mut self.state),
            exclusive,
            read_area: self.read_area.take(),
            ticks: u64::from(matches!(stop, InstructionStep::Continue)),
            stop,
        }))
    }
}
impl Effects<'_> {
    fn access(
        &self,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        operation: Operation,
    ) -> Result<Outcome<'_>, DataAccessFault> {
        let kind = operation.kind();
        match &self.stage {
            Stage::Capture {
                first,
                reached,
                records,
            } => {
                let skipped = kind == Kind::Write && !reached.get() && Some(address) != *first;
                if kind == Kind::Write && !reached.get() && !skipped {
                    reached.set(true);
                }
                records.borrow_mut().push(Record {
                    address,
                    access,
                    kind,
                    operation,
                    group: 1,
                    input: None,
                    result: skipped.then_some(Ok(Value::Written)),
                });
                Ok(Outcome::Planned)
            }
            Stage::Complete { records, next } => {
                let record = records
                    .get(next.get())
                    .filter(|record| {
                        record.address == address && record.access == access && record.kind == kind
                    })
                    .expect("instruction effect differs from its captured inputs");
                next.set(next.get() + 1);
                match record
                    .result
                    .as_ref()
                    .expect("instruction effect is incomplete")
                {
                    Ok(result) => Ok(Outcome::Completed(result)),
                    Err(fault) => Err(fault.clone()),
                }
            }
        }
    }
}
fn alignment(record: &Record) -> u64 {
    match record.access.alignment {
        MemoryAlignment::Unaligned => 1,
        MemoryAlignment::Natural => record.access.size.bytes() as u64,
        MemoryAlignment::Bytes2 => 2,
        MemoryAlignment::Bytes4 => 4,
        MemoryAlignment::Bytes8 => 8,
        MemoryAlignment::Bytes16 => 16,
    }
}
fn prepare(
    records: &mut Vec<Record>,
    first: Option<&CapturedAccess>,
    ranges: &mut Option<Vec<AllocationRange>>,
    space: &AddressSpace,
    admission: &NativeAdmission<'_>,
) {
    let mut index = 0;
    while index < records.len() {
        if records[index].result.is_some() {
            index += 1;
            continue;
        }
        let head = &records[index];
        let address = head.address;
        let kind = head.kind;
        let mut bytes = head.access.size.bytes() as u64;
        let mut end = index + 1;
        // Coalesce normal same-page accesses, including individual SIMD lanes.
        // Stop at a page boundary so a later mapping fault preserves the valid
        // unexecuted prefix. Atomic and ordered accesses remain individual.
        if matches!(kind, Kind::Read | Kind::Write)
            && head.access.class == MemoryAccessClass::Normal
            && head.access.ordering == MemoryOrdering::Relaxed
        {
            while let Some(next) = records.get(end).filter(|next| {
                next.result.is_none()
                    && next.kind == kind
                    && next.access.class == MemoryAccessClass::Normal
                    && next.access.ordering == MemoryOrdering::Relaxed
                    && Some(next.address.get()) == address.get().checked_add(bytes)
                    && next.address.get().is_multiple_of(alignment(next))
                    && next
                        .address
                        .get()
                        .checked_add(next.access.size.bytes() as u64 - 1)
                        .is_some_and(|last| {
                            last / nixe_memory::PAGE_SIZE == address.get() / nixe_memory::PAGE_SIZE
                        })
            }) {
                bytes += next.access.size.bytes() as u64;
                end += 1;
            }
        }
        let physical = if !address.get().is_multiple_of(alignment(head)) {
            Err(MemoryError::Invalid(
                "unaligned architectural memory access",
            ))
        } else if first.is_some_and(|first| {
            address == first.address
                && bytes == u64::from(first.bytes)
                && Kind::of(&first.operation)
                    .permissions()
                    .contains(kind.permissions())
        }) && ranges.is_some()
        {
            Ok(ranges.take().unwrap())
        } else {
            space.capture_native(admission, address, bytes, kind.permissions())
        };
        match physical {
            Ok(physical) => {
                let operation = match records[index].operation {
                    Operation::Read => AccessOperation::ReadNative,
                    Operation::Atomic(access) => AccessOperation::Atomic(access),
                    Operation::Write(_) => {
                        let mut data = Vec::with_capacity(bytes as usize);
                        for record in &records[index..end] {
                            let Operation::Write(value) = record.operation else {
                                unreachable!()
                            };
                            data.extend_from_slice(
                                &value.bits().to_le_bytes()[..record.access.size.bytes()],
                            );
                        }
                        AccessOperation::Write(data.into_boxed_slice())
                    }
                };
                records[index].group = end - index;
                records[index].input = Some((physical, operation));
                index = end;
            }
            Err(error) => {
                let reason = if !address.get().is_multiple_of(alignment(head)) {
                    DataAccessFaultReason::Misaligned {
                        required_alignment: alignment(head) as u8,
                    }
                } else {
                    DataAccessFaultReason::from(&error)
                };
                let address = match error {
                    MemoryError::Unmapped(address)
                    | MemoryError::PermissionDenied { address, .. } => address,
                    _ => address,
                };
                records[index].result =
                    Some(Err(data_fault(space.identity(), address, kind, reason)));
                records.truncate(index + 1);
                break;
            }
        }
    }
}
fn distribute(
    records: &mut [Record],
    index: usize,
    result: AccessResult,
) -> Option<NativeReadArea> {
    let group = records[index].group;
    let mut offset = 0;
    for record in &mut records[index..index + group] {
        let value = match &result {
            AccessResult::Read(bytes) | AccessResult::NativeRead { bytes, .. } => {
                let width = record.access.size.bytes();
                let mut scalar = [0u8; 16];
                scalar[..width].copy_from_slice(&bytes[offset..offset + width]);
                offset += width;
                Value::Read(u128::from_le_bytes(scalar))
            }
            AccessResult::Written => Value::Written,
            AccessResult::Atomic { previous, written } => Value::Atomic {
                previous: *previous,
                written: *written,
            },
            AccessResult::ExclusiveRead { value, reservation } => Value::ExclusiveRead {
                value: *value,
                reservation: *reservation,
            },
            AccessResult::ExclusiveWritten(written) => Value::ExclusiveWritten(*written),
        };
        record.result = Some(Ok(value));
    }
    if let AccessResult::NativeRead { area, .. } = result {
        area
    } else {
        None
    }
}
fn data_fault(
    space: AddressSpaceId,
    address: GuestVirtualAddress,
    kind: Kind,
    reason: DataAccessFaultReason,
) -> DataAccessFault {
    DataAccessFault::new(
        space,
        address,
        if matches!(kind, Kind::Read | Kind::ExclusiveRead) {
            DataAccessKind::Read
        } else {
            DataAccessKind::Write
        },
        reason,
    )
}
fn value(access: MemoryAccess, bits: u128) -> DataReadResult {
    DataReadResult {
        value: MemoryValue::from_bits(access.size, bits),
        region: MemoryRegionKind::Ram,
    }
}
fn ordering(access: MemoryAccess) -> AccessOrdering {
    match access.ordering {
        MemoryOrdering::Relaxed => AccessOrdering::Relaxed,
        MemoryOrdering::Acquire => AccessOrdering::Acquire,
        MemoryOrdering::Release => AccessOrdering::Release,
        MemoryOrdering::AcquireRelease => AccessOrdering::AcquireRelease,
        MemoryOrdering::SequentiallyConsistent => AccessOrdering::SequentiallyConsistent,
    }
}
impl InstructionMemoryAccess for Effects<'_> {
    type Reservation = Option<Reservation>;
    fn read(
        &self,
        _: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<DataReadResult, DataAccessFault> {
        let bits = match self.access(address, access, Operation::Read)? {
            Outcome::Planned => 0,
            Outcome::Completed(Value::Read(bits)) => *bits,
            _ => unreachable!("invalid read completion"),
        };
        Ok(value(access, bits))
    }
    fn write(
        &self,
        _: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
    ) -> Result<DataWriteResult, DataAccessFault> {
        self.access(address, access, Operation::Write(value))?;
        Ok(DataWriteResult {
            region: MemoryRegionKind::Ram,
        })
    }
    fn atomic_read_modify_write(
        &self,
        _: AddressSpaceId,
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
        self.atomic_result(
            address,
            access,
            AtomicOperation::ReadModifyWrite {
                transform,
                operand: operand.bits(),
            },
        )
    }
    fn atomic_compare_exchange(
        &self,
        _: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        expected: MemoryValue,
        replacement: MemoryValue,
    ) -> Result<AtomicMemoryResult, DataAccessFault> {
        self.atomic_result(
            address,
            access,
            AtomicOperation::CompareExchange {
                expected: expected.bits(),
                replacement: replacement.bits(),
            },
        )
    }
    fn maintain_cache(
        &self,
        space: AddressSpaceId,
        _: CacheMaintenanceKind,
        address: Option<GuestVirtualAddress>,
    ) -> Result<(), DataAccessFault> {
        Err(data_fault(
            space,
            address.unwrap_or(GuestVirtualAddress::MIN),
            Kind::Write,
            DataAccessFaultReason::HostBacking(
                "cache operation is not a memory instruction".into(),
            ),
        ))
    }
    fn load_exclusive(
        &self,
        _: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<(DataReadResult, Self::Reservation), DataAccessFault> {
        let (bits, reservation) = match self.access(
            address,
            access,
            Operation::Atomic(AtomicAccess {
                bytes: access.size.bytes() as u8,
                ordering: ordering(access),
                operation: AtomicOperation::LoadExclusive,
            }),
        )? {
            Outcome::Planned => (0, None),
            Outcome::Completed(Value::ExclusiveRead { value, reservation }) => {
                (*value, Some(*reservation))
            }
            _ => unreachable!("invalid exclusive read completion"),
        };
        Ok((value(access, bits), reservation))
    }
    fn store_exclusive(
        &self,
        _: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
        reservation: Self::Reservation,
    ) -> Result<(DataWriteResult, bool), DataAccessFault> {
        let reservation =
            reservation.expect("native store-exclusive must retain its physical reservation");
        let written = match self.access(
            address,
            access,
            Operation::Atomic(AtomicAccess {
                bytes: access.size.bytes() as u8,
                ordering: ordering(access),
                operation: AtomicOperation::StoreExclusive {
                    reservation,
                    replacement: value.bits(),
                },
            }),
        )? {
            Outcome::Planned => true,
            Outcome::Completed(Value::ExclusiveWritten(written)) => *written,
            _ => unreachable!("invalid exclusive write completion"),
        };
        Ok((
            DataWriteResult {
                region: MemoryRegionKind::Ram,
            },
            written,
        ))
    }
}
impl Effects<'_> {
    fn atomic_result(
        &self,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        operation: AtomicOperation,
    ) -> Result<AtomicMemoryResult, DataAccessFault> {
        let (previous, written) = match self.access(
            address,
            access,
            Operation::Atomic(AtomicAccess {
                bytes: access.size.bytes() as u8,
                ordering: ordering(access),
                operation,
            }),
        )? {
            Outcome::Planned => (0, true),
            Outcome::Completed(Value::Atomic { previous, written }) => (*previous, *written),
            _ => unreachable!("invalid atomic completion"),
        };
        Ok(AtomicMemoryResult {
            previous: MemoryValue::from_bits(access.size, previous),
            stored: written,
            region: MemoryRegionKind::Ram,
        })
    }
}
