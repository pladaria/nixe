use super::*;
use crate::lcq::{compiler::PublishError, invocation};
use nixe_cpu::error::InstructionFetchFault;
use nixe_cpu::memory::{
    AtomicRmwKind, CacheMaintenanceKind, CodePageSpan, CpuMemory, ExecutableMemory, FetchedCode,
    InstructionImage, MemoryAccess, MemoryAccessClass, MemoryAccessSize, MemoryAlignment,
    MemoryOrdering, MemoryValue,
};
use nixe_memory::MemoryInvalidation;
use nixe_memory::{MemoryInvalidationCursor, MemoryInvalidationError};
use std::num::NonZeroU16;
use std::sync::atomic::{AtomicBool, AtomicUsize};

const OLD: u32 = 0xd4200020; // BRK #1
const NEW: u32 = 0xd4200120; // BRK #9
const ALIAS: GuestVirtualAddress = GuestVirtualAddress::new(0x3000);

fn setup() -> (Arc<Lifetime>, ExecutionMemory) {
    let (process, memory) = fixture();
    writes::writable_alias(&memory);
    memory
        .overwrite_mapped_ram(SPACE, ALIAS, &OLD.to_le_bytes())
        .unwrap();
    (process, memory)
}

fn guest_write(memory: &ExecutionMemory, operation: u8) {
    let access = MemoryAccess::new(
        MemoryAccessSize::Word,
        MemoryAlignment::Natural,
        MemoryOrdering::AcquireRelease,
        if operation == 3 {
            MemoryAccessClass::Exclusive
        } else {
            MemoryAccessClass::Atomic
        },
    );
    match operation {
        0 => {
            memory
                .write(
                    SPACE,
                    ALIAS,
                    MemoryAccess::normal(MemoryAccessSize::Word),
                    MemoryValue::U32(NEW),
                )
                .unwrap();
        }
        1 => assert!(
            memory
                .atomic_compare_exchange(
                    SPACE,
                    ALIAS,
                    access,
                    MemoryValue::U32(OLD),
                    MemoryValue::U32(NEW),
                )
                .unwrap()
                .stored
        ),
        2 => assert!(
            memory
                .atomic_read_modify_write(
                    SPACE,
                    ALIAS,
                    access,
                    AtomicRmwKind::Swap,
                    MemoryValue::U32(NEW),
                )
                .unwrap()
                .stored
        ),
        3 => {
            let (_, reservation) = memory.load_exclusive(SPACE, ALIAS, access).unwrap();
            assert!(
                memory
                    .store_exclusive(SPACE, ALIAS, access, MemoryValue::U32(NEW), reservation)
                    .unwrap()
                    .1
            );
        }
        _ => unreachable!(),
    }
}

fn ic(memory: &ExecutionMemory) {
    memory
        .maintain_cache(
            SPACE,
            CacheMaintenanceKind::InstructionInvalidate,
            Some(ALIAS),
        )
        .unwrap();
}

fn breakpoint(process: &Arc<Lifetime>, memory: &ExecutionMemory) -> u16 {
    let mut reader = process.register().unwrap();
    let mut state = A64State::default();
    state.set_pc(0x1000);
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
    let mut monitor = nixe_cpu::exclusive::ExclusiveMonitorState::default();
    let mut worker = nixe_cpu_direct_memory::WorkerFaultContext::register().unwrap();
    let invocation::Exit::Native { guest, .. } = unsafe {
        invocation::run(
            &mut crate::sampling::Samples::new(),
            &mut reader,
            &mut frame,
            memory,
            &mut worker,
            &mut monitor,
            key(0x1000),
        )
    }
    .unwrap()
    .unwrap() else {
        panic!("expected native breakpoint")
    };
    let unit::EdgeKind::Breakpoint(value) = guest.kind else {
        panic!("expected captured breakpoint")
    };
    value
}

// The real compiler obtains its cursor signal after its final image check and
// before prepare_unit. Inject one mutation at that trait boundary, with no
// production hooks and without replacing capture, validation or publication.
struct BeforePublication<'a> {
    memory: &'a ExecutionMemory,
    action: &'a (dyn Fn() + Sync),
    called: AtomicBool,
    validations: AtomicUsize,
}

impl InstructionMemory for BeforePublication<'_> {
    fn code_page_span(
        &self,
        space: AddressSpaceId,
        pc: GuestVirtualAddress,
    ) -> Result<CodePageSpan, InstructionFetchFault> {
        self.memory.code_page_span(space, pc)
    }
    fn fetch32(
        &self,
        space: AddressSpaceId,
        pc: GuestVirtualAddress,
    ) -> Result<FetchedCode<u32>, InstructionFetchFault> {
        self.memory.fetch32(space, pc)
    }
}

impl ExecutableMemory for BeforePublication<'_> {
    fn capture_instructions(
        &self,
        space: AddressSpaceId,
        start: GuestVirtualAddress,
        limit: NonZeroU16,
        stop: &dyn Fn(GuestVirtualAddress, u32) -> bool,
    ) -> InstructionImage {
        self.memory.capture_instructions(space, start, limit, stop)
    }
    fn image_is_current(&self, image: &InstructionImage) -> bool {
        self.validations.fetch_add(1, Ordering::Relaxed);
        self.memory.image_is_current(image)
    }
}

impl MemoryInvalidationSource for BeforePublication<'_> {
    fn invalidation_cursor(&self) -> MemoryInvalidationCursor {
        self.memory.invalidation_cursor()
    }
    fn invalidation_signal(&self) -> &std::sync::atomic::AtomicU64 {
        assert_eq!(self.validations.load(Ordering::Relaxed), 2);
        assert!(!self.called.swap(true, Ordering::Relaxed));
        (self.action)();
        self.memory.invalidation_signal()
    }
    fn read_invalidations_since(
        &self,
        after: MemoryInvalidationCursor,
        output: &mut Vec<MemoryInvalidation>,
    ) -> Result<MemoryInvalidationCursor, MemoryInvalidationError> {
        self.memory.read_invalidations_since(after, output)
    }
}

#[test]
fn checked_guest_writes_before_validation_reject_the_captured_candidate() {
    for operation in 0..4 {
        let (process, memory) = setup();
        let mut reader = process.register().unwrap();
        let compile::Request::Owner(claim) = reader.claim(key(0x1000)).unwrap() else {
            panic!()
        };
        let captured = Compilation::capture(claim, &memory).unwrap();
        guest_write(&memory, operation);
        assert!(matches!(
            compiler().publish(captured, &process, &process.cache, &memory),
            Err(PublishError::StaleCapture)
        ));
        assert!(process.lock().keys.is_empty());
        publish(&process, &memory, 0x1000);
        assert_eq!(breakpoint(&process, &memory), 9);
    }
}

#[test]
fn guest_write_after_final_validation_can_publish_old_code_only_until_ic() {
    for operation in 0..4 {
        for invalidate_before_publication in [false, true] {
            let (process, memory) = setup();
            let mut reader = process.register().unwrap();
            let compile::Request::Owner(claim) = reader.claim(key(0x1000)).unwrap() else {
                panic!()
            };
            let captured = Compilation::capture(claim, &memory).unwrap();
            let cursor = memory.invalidation_cursor();
            let action = || {
                guest_write(&memory, operation);
                assert_eq!(memory.invalidation_cursor(), cursor);
                if invalidate_before_publication {
                    ic(&memory);
                }
            };
            let boundary = BeforePublication {
                memory: &memory,
                action: &action,
                called: AtomicBool::new(false),
                validations: AtomicUsize::new(0),
            };
            let result = compiler().publish(captured, &process, &process.cache, &boundary);
            assert!(boundary.called.load(Ordering::Relaxed));
            if invalidate_before_publication {
                assert!(matches!(
                    result,
                    Err(PublishError::Lifetime(Error::StalePublication))
                ));
                assert!(process.lock().keys.is_empty());
            } else {
                let old = result.unwrap();
                assert_eq!(breakpoint(&process, &memory), 1);
                ic(&memory);
                assert!(matches!(process.snapshot(old), Err(Error::StaleUnit)));
            }
            assert!(memory.invalidation_cursor() > cursor);
            assert_eq!(process.lock().phase, Phase::Open);
            publish(&process, &memory, 0x1000);
            assert_eq!(breakpoint(&process, &memory), 9);
        }
    }
}

#[test]
fn host_write_or_permission_change_after_final_validation_rejects_publication() {
    for permission in [false, true] {
        let (process, memory) = setup();
        let mut reader = process.register().unwrap();
        let compile::Request::Owner(claim) = reader.claim(key(0x1000)).unwrap() else {
            panic!()
        };
        let captured = Compilation::capture(claim, &memory).unwrap();
        let action = || {
            if permission {
                memory
                    .set_permissions(
                        SPACE,
                        GuestVirtualAddress::new(0x1000),
                        4096,
                        MemoryPermissions::READ,
                    )
                    .unwrap();
            } else {
                memory
                    .overwrite_mapped_ram(SPACE, ALIAS, &NEW.to_le_bytes())
                    .unwrap();
            }
        };
        let boundary = BeforePublication {
            memory: &memory,
            action: &action,
            called: AtomicBool::new(false),
            validations: AtomicUsize::new(0),
        };
        assert!(matches!(
            compiler().publish(captured, &process, &process.cache, &boundary),
            Err(PublishError::Lifetime(Error::StalePublication))
        ));
        assert!(boundary.called.load(Ordering::Relaxed));
        assert!(process.lock().keys.is_empty());
        assert_eq!(process.lock().phase, Phase::Open);
    }
}
