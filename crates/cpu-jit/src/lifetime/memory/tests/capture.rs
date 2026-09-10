use super::*;
use nixe_cpu::error::{InstructionFetchFault, InstructionFetchFaultReason};
use nixe_cpu::memory::{
    CodePageSpan, CpuMemory, ExecutableMemory, FetchedCode, InstructionImage, MemoryAccess,
    MemoryAccessSize, MemoryValue,
};
use nixe_memory::DirectProtection;
use std::num::NonZeroU16;

#[test]
fn already_armed_capture_preserves_other_compilers_and_needs_no_fault_reader_stop() {
    let (process, memory) = fixture();
    let code = publish(&process, &memory, 0x1000);
    let snapshot = process.snapshot(code).unwrap();
    let mut compiling = process.register().unwrap();
    let compile::Request::Owner(other) = compiling.claim(key(0x2000)).unwrap() else {
        panic!()
    };
    let mut reader = process.register().unwrap();
    let mut state = A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
    let invocation = unsafe { reader.admit(&mut frame, key(0x1000)) }
        .unwrap()
        .unwrap();
    let admission = process.lock().admission;
    let cursor = memory.invalidation_cursor();
    let (done, received) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let fault = invocation
            .fault(snapshot.code.allocation.address() + snapshot.faults[0].native_start as usize)
            .unwrap();
        let worker = scope.spawn(|| {
            let code = publish(&process, &memory, 0x1004);
            done.send(code).unwrap();
        });
        let result = received.recv_timeout(Duration::from_secs(5));
        assert_eq!(fault.unit.id, snapshot.id);
        // Release before asserting the timeout, so a regression cannot deadlock
        // the test by leaving a worker waiting for this same reader.
        drop(invocation);
        let new = result.unwrap();
        worker.join().unwrap();
        assert_eq!(process.snapshot(new).unwrap().instructions.len(), 1);
    });
    assert_eq!(process.lock().admission, admission);
    assert_eq!(memory.invalidation_cursor(), cursor);
    other.validate().unwrap();
    assert!(process.snapshot(code).is_ok());
}

#[test]
fn instruction_tracking_drains_fault_readers_then_captures_with_new_compile_admission() {
    let (process, memory) = fixture();
    writes::writable_alias(&memory);
    let code = publish(&process, &memory, 0x1000);
    let snapshot = process.snapshot(code).unwrap();
    // Disarm tracking without changing guest code or issuing IC maintenance.
    memory
        .write(
            SPACE,
            GuestVirtualAddress::new(0x3000),
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(0xf9400020),
        )
        .unwrap();
    let mut compiling = process.register().unwrap();
    let compile::Request::Owner(old) = compiling.claim(key(0x2000)).unwrap() else {
        panic!()
    };
    let cursor = memory.invalidation_cursor();
    let mut reader = process.register().unwrap();
    let mut state = A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
    let lease = memory.acquire_execution_lease();
    let invocation = unsafe { reader.admit(&mut frame, key(0x1000)) }
        .unwrap()
        .unwrap();
    let new = std::thread::scope(|scope| {
        let fault = invocation
            .fault(snapshot.code.allocation.address() + snapshot.faults[0].native_start as usize)
            .unwrap();
        drop(lease);
        let worker = scope.spawn(|| publish(&process, &memory, 0x1004));
        let (locked, timeout) = process
            .changed
            .wait_timeout_while(process.lock(), Duration::from_secs(5), |state| {
                state.phase == Phase::Open
            })
            .unwrap();
        assert!(!timeout.timed_out());
        assert_eq!(locked.phase, Phase::Closing);
        drop(locked);
        // A memory-only stop is insufficient: fault metadata remains borrowed
        // after releasing the memory lease, until Invocation itself finishes.
        assert_eq!(fault.unit.id, snapshot.id);
        assert_eq!(process.lock().phase, Phase::Closing);
        assert_eq!(
            memory.direct_protection_at(SPACE, GuestVirtualAddress::new(0x3000)),
            Some(DirectProtection::ReadWrite)
        );
        assert_eq!(memory.invalidation_cursor(), cursor);
        drop(invocation);
        worker.join().unwrap()
    });
    assert_eq!(process.lock().phase, Phase::Open);
    assert_eq!(old.validate(), Err(Error::StalePublication));
    assert_eq!(memory.invalidation_cursor(), cursor);
    assert_eq!(
        memory.direct_protection_at(SPACE, GuestVirtualAddress::new(0x3000)),
        Some(DirectProtection::Read)
    );
    assert!(process.snapshot(code).is_ok());
    assert_eq!(
        process.snapshot(new).unwrap().instructions[0].bits,
        0xd4200000
    );
}

// Insert the race after a real bounded capture releases its gate, before the
// compiler obtains its new admission identity and validates the owned image.
struct ChangeAfterCapture<'a>(&'a ExecutionMemory);

impl InstructionMemory for ChangeAfterCapture<'_> {
    fn code_page_span(
        &self,
        space: AddressSpaceId,
        pc: GuestVirtualAddress,
    ) -> Result<CodePageSpan, InstructionFetchFault> {
        self.0.code_page_span(space, pc)
    }
    fn fetch32(
        &self,
        space: AddressSpaceId,
        pc: GuestVirtualAddress,
    ) -> Result<FetchedCode<u32>, InstructionFetchFault> {
        self.0.fetch32(space, pc)
    }
}

impl ExecutableMemory for ChangeAfterCapture<'_> {
    fn capture_instructions(
        &self,
        space: AddressSpaceId,
        start: GuestVirtualAddress,
        limit: NonZeroU16,
        stop: &dyn Fn(GuestVirtualAddress, u32) -> bool,
    ) -> InstructionImage {
        let image = self.0.capture_instructions(space, start, limit, stop);
        self.0
            .overwrite_mapped_ram(space, start, &0xd4200120_u32.to_le_bytes())
            .unwrap();
        image
    }
    fn image_is_current(&self, image: &InstructionImage) -> bool {
        self.0.image_is_current(image)
    }
}

#[test]
fn new_capture_admission_does_not_make_an_intervening_code_write_current() {
    let (process, memory) = fixture();
    let mut reader = process.register().unwrap();
    let compile::Request::Owner(claim) = reader.claim(key(0x1000)).unwrap() else {
        panic!()
    };
    assert!(matches!(
        Compilation::capture(claim, &ChangeAfterCapture(&memory)),
        Err(Error::StalePublication)
    ));
    assert!(process.lock().keys.is_empty());
    let new = publish(&process, &memory, 0x1000);
    assert_eq!(
        process.snapshot(new).unwrap().instructions[0].bits,
        0xd4200120
    );
}

#[test]
fn rejected_instruction_capture_preserves_diagnostic_without_arming_or_copying() {
    let (process, memory) = fixture();
    writes::writable_alias(&memory);
    let cursor = memory.invalidation_cursor();
    process.fail(&mut process.lock(), Error::CacheFailed);
    let image = memory.capture_instructions(
        SPACE,
        GuestVirtualAddress::new(0x1000),
        NonZeroU16::new(2).unwrap(),
        &|_, _| panic!("a refused capture must not read/classify instructions"),
    );
    assert!(image.words().is_empty());
    assert_eq!(
        image.fault().unwrap().reason,
        InstructionFetchFaultReason::Memory(Error::CacheFailed.to_string().into())
    );
    assert_eq!(
        memory.direct_protection_at(SPACE, GuestVirtualAddress::new(0x3000)),
        Some(DirectProtection::ReadWrite)
    );
    assert_eq!(memory.invalidation_cursor(), cursor);
    assert_eq!(process.lock().memory_mutations, 0);
    drop(memory.acquire_execution_lease()); // Rejection released the memory gate.
}
