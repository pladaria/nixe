use std::cell::RefCell;
use std::sync::Mutex;

use nixe_cpu::error::InstructionFetchFault;
use nixe_cpu::exclusive::{ExclusiveMonitorState, ExclusiveReservation};
use nixe_cpu::location::InstructionEncoding;
use nixe_cpu::memory::{
    AtomicMemoryResult, AtomicRmwKind, CacheMaintenanceKind, CodePageSpan, CpuMemory,
    DataAccessFault, DataReadResult, DataWriteResult, FetchedCode, InstructionMemory, MemoryAccess,
    MemoryAccessClass, MemoryAccessSize, MemoryOrdering, MemoryPermissions, MemoryQueryResult,
    MemoryValue, SyntheticMemory,
};
use nixe_cpu::profile::{GuestCpuProfile, ProcessCpuContext};
use nixe_cpu::state::{ThreadCpuState, a32::A32GeneralRegister};
use nixe_cpu_engine_interpreter::{InterpreterContext, execute_one_with_context};
use nixe_memory::{
    AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress, MemoryInvalidation,
    MemoryInvalidationCursor, MemoryInvalidationError, MemoryInvalidationSource,
};

const SPACE: AddressSpaceId = AddressSpaceId::new(7);
const PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(9);
const PRIMARY: GuestVirtualAddress = GuestVirtualAddress::new(0x1000);
const ALIAS: GuestVirtualAddress = GuestVirtualAddress::new(0x2000);

struct ObservedMemory<'a> {
    inner: &'a SyntheticMemory,
    accesses: Mutex<Vec<MemoryAccess>>,
}

impl<'a> ObservedMemory<'a> {
    fn new(inner: &'a SyntheticMemory) -> Self {
        Self {
            inner,
            accesses: Mutex::default(),
        }
    }

    fn take_accesses(&self) -> Vec<MemoryAccess> {
        std::mem::take(&mut *self.accesses.lock().unwrap())
    }

    fn record(&self, access: MemoryAccess) {
        self.accesses.lock().unwrap().push(access);
    }
}

impl InstructionMemory for ObservedMemory<'_> {
    fn content_mutation_epoch(&self) -> nixe_memory::ContentMutationEpoch {
        self.inner.content_mutation_epoch()
    }

    fn code_page_span(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<CodePageSpan, InstructionFetchFault> {
        self.inner.code_page_span(address_space, address)
    }

    fn fetch16(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<FetchedCode<u16>, InstructionFetchFault> {
        self.inner.fetch16(address_space, address)
    }

    fn fetch32(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<FetchedCode<u32>, InstructionFetchFault> {
        self.inner.fetch32(address_space, address)
    }
}

impl MemoryInvalidationSource for ObservedMemory<'_> {
    fn invalidation_cursor(&self) -> MemoryInvalidationCursor {
        self.inner.invalidation_cursor()
    }

    fn read_invalidations_since(
        &self,
        after: MemoryInvalidationCursor,
        output: &mut Vec<MemoryInvalidation>,
    ) -> Result<MemoryInvalidationCursor, MemoryInvalidationError> {
        self.inner.read_invalidations_since(after, output)
    }
}

impl CpuMemory for ObservedMemory<'_> {
    fn maintain_cache(
        &self,
        address_space: AddressSpaceId,
        kind: CacheMaintenanceKind,
        address: Option<GuestVirtualAddress>,
    ) -> Result<(), DataAccessFault> {
        self.inner.maintain_cache(address_space, kind, address)
    }

    fn read(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<DataReadResult, DataAccessFault> {
        self.record(access);
        self.inner.read(address_space, address, access)
    }

    fn write(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
    ) -> Result<DataWriteResult, DataAccessFault> {
        self.record(access);
        self.inner.write(address_space, address, access, value)
    }

    fn atomic_read_modify_write(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        kind: AtomicRmwKind,
        operand: MemoryValue,
    ) -> Result<AtomicMemoryResult, DataAccessFault> {
        self.record(access);
        self.inner
            .atomic_read_modify_write(address_space, address, access, kind, operand)
    }

    fn atomic_compare_exchange(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        expected: MemoryValue,
        replacement: MemoryValue,
    ) -> Result<AtomicMemoryResult, DataAccessFault> {
        self.record(access);
        self.inner
            .atomic_compare_exchange(address_space, address, access, expected, replacement)
    }

    fn query_memory(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        end_exclusive: GuestVirtualAddress,
    ) -> Option<MemoryQueryResult> {
        self.inner
            .query_memory(address_space, address, end_exclusive)
    }

    fn load_exclusive(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<(DataReadResult, ExclusiveReservation), DataAccessFault> {
        self.record(access);
        self.inner.load_exclusive(address_space, address, access)
    }

    fn store_exclusive(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
        reservation: ExclusiveReservation,
    ) -> Result<(DataWriteResult, bool), DataAccessFault> {
        self.record(access);
        self.inner
            .store_exclusive(address_space, address, access, value, reservation)
    }
}

fn register(index: u8) -> A32GeneralRegister {
    A32GeneralRegister::new(index).unwrap()
}

#[test]
fn a32_ldrex_strex_uses_local_monitor_and_canonical_alias_generation() {
    let mut memory = SyntheticMemory::new();
    assert!(memory.add_ram_page(PAGE));
    assert!(memory.map_page(SPACE, PRIMARY, PAGE, MemoryPermissions::READ_WRITE));
    assert!(memory.map_page(SPACE, ALIAS, PAGE, MemoryPermissions::READ_WRITE));
    assert!(memory.initialize_ram(PAGE, 0, &7_u32.to_le_bytes()));

    let observed = ObservedMemory::new(&memory);
    let monitor = RefCell::new(ExclusiveMonitorState::default());
    let context =
        InterpreterContext::new(ProcessCpuContext::new(GuestCpuProfile::switch_1(), SPACE))
            .with_memory(&observed)
            .with_exclusive_monitor(&monitor);
    let mut state = ThreadCpuState::A32(Box::default());
    let ThreadCpuState::A32(a32) = &mut state else {
        unreachable!()
    };
    a32.write_r(register(3), PRIMARY.get() as u32);

    // LDREX r0, [r3]
    execute_one_with_context(
        context,
        &mut state,
        InstructionEncoding::from_u32(0xe193_0f9f),
    )
    .unwrap();
    let accesses = observed.take_accesses();
    assert_eq!(accesses.len(), 1);
    assert_eq!(accesses[0].ordering, MemoryOrdering::Relaxed);
    assert_eq!(accesses[0].class, MemoryAccessClass::Exclusive);
    let ThreadCpuState::A32(a32) = &mut state else {
        unreachable!()
    };
    assert_eq!(a32.read_r(register(0)), 7);
    a32.write_r(register(0), 9);

    // Any write through a physical alias invalidates the reservation.
    memory
        .write(
            SPACE,
            ALIAS,
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(8),
        )
        .unwrap();

    // STREX r1, r0, [r3]
    execute_one_with_context(
        context,
        &mut state,
        InstructionEncoding::from_u32(0xe183_1f90),
    )
    .unwrap();
    let accesses = observed.take_accesses();
    assert_eq!(accesses.len(), 1);
    assert_eq!(accesses[0].ordering, MemoryOrdering::Relaxed);
    assert_eq!(accesses[0].class, MemoryAccessClass::Exclusive);
    let ThreadCpuState::A32(a32) = &state else {
        unreachable!()
    };
    assert_eq!(a32.read_r(register(1)), 1);
    assert_eq!(
        memory
            .read(SPACE, PRIMARY, MemoryAccess::normal(MemoryAccessSize::Word),)
            .unwrap()
            .value,
        MemoryValue::U32(8),
    );
}

#[test]
fn a32_lda_stl_transfer_words_without_using_the_exclusive_monitor() {
    let mut memory = SyntheticMemory::new();
    assert!(memory.add_ram_page(PAGE));
    assert!(memory.map_page(SPACE, PRIMARY, PAGE, MemoryPermissions::READ_WRITE));
    assert!(memory.initialize_ram(PAGE, 0, &7_u32.to_le_bytes()));

    let observed = ObservedMemory::new(&memory);
    let monitor = RefCell::new(ExclusiveMonitorState::default());
    let context =
        InterpreterContext::new(ProcessCpuContext::new(GuestCpuProfile::switch_1(), SPACE))
            .with_memory(&observed)
            .with_exclusive_monitor(&monitor);
    let mut state = ThreadCpuState::A32(Box::default());
    let ThreadCpuState::A32(a32) = &mut state else {
        unreachable!()
    };
    a32.write_r(register(3), PRIMARY.get() as u32);

    // LDA r0, [r3]. Arm ARM DDI 0602 AArch32 LDA:
    // https://developer.arm.com/documentation/ddi0602/latest/AArch32-Instructions/LDA
    execute_one_with_context(
        context,
        &mut state,
        InstructionEncoding::from_u32(0xe193_0e8f),
    )
    .unwrap();
    let accesses = observed.take_accesses();
    assert_eq!(accesses.len(), 1);
    assert_eq!(accesses[0].ordering, MemoryOrdering::Acquire);
    assert_eq!(accesses[0].class, MemoryAccessClass::Normal);
    let ThreadCpuState::A32(a32) = &mut state else {
        unreachable!()
    };
    assert_eq!(a32.read_r(register(0)), 7);
    a32.write_r(register(0), 11);

    // STL r0, [r3]. Arm ARM DDI 0602 AArch32 STL:
    // https://developer.arm.com/documentation/ddi0602/latest/AArch32-Instructions/STL
    execute_one_with_context(
        context,
        &mut state,
        InstructionEncoding::from_u32(0xe183_0e80),
    )
    .unwrap();
    let accesses = observed.take_accesses();
    assert_eq!(accesses.len(), 1);
    assert_eq!(accesses[0].ordering, MemoryOrdering::Release);
    assert_eq!(accesses[0].class, MemoryAccessClass::Normal);
    assert_eq!(
        memory
            .read(SPACE, PRIMARY, MemoryAccess::normal(MemoryAccessSize::Word),)
            .unwrap()
            .value,
        MemoryValue::U32(11)
    );
    assert!(monitor.borrow().reservation().is_none());
}
