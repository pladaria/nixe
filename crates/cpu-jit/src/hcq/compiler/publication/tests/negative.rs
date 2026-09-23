use super::*;
use crate::abi::InstructionKey;
use crate::lifetime::background::Outcome;
use crate::sampling::{BoundaryKey, FamilyIdentity};

pub(super) fn real_backend_limit(frozen: &Frozen<'_, '_>) -> Limit {
    let compiler = Compiler::new(host(), 0x10000).unwrap();
    let mut context = Context::new();
    let mut frontend = FunctionBuilderContext::new();
    let body = compiler
        .emit(
            &mut context,
            &mut frontend,
            frozen.graph(),
            &frozen.analyze().unwrap(),
            frozen.entries(),
        )
        .unwrap();
    // Each slot is legal; their aggregate exceeds the fixed native frame.
    for _ in 0..2 {
        context.func.create_sized_stack_slot(ir::StackSlotData::new(
            ir::StackSlotKind::ExplicitSlot,
            8192,
            4,
        ));
    }
    let Err(Failure::Rejected(limit)) = compiler.finish(
        &mut context,
        body,
        frozen.graph(),
        CodeVersion::new(1).unwrap(),
    ) else {
        panic!("expected real backend frame limit")
    };
    assert!(context.func.layout.blocks().next().is_none());
    limit
}

#[test]
fn real_backend_limit_records_reshape_negative_without_poisoning_seed() {
    let (process, memory, mut reader) = setup();
    for expected in [true, false] {
        if !expected {
            let (queue, outcome) = admit(&process, &mut reader, 0x1000, 0x1004, 0x2000);
            assert_eq!(outcome, Outcome::Suppressed);
            assert!(queue.pop().unwrap().is_none());
            continue;
        }
        let work = reshape(&process, &mut reader, 0x1000, 0x1004, 0x2000);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        let before = process.executable_cache().usage().unwrap().committed;
        let limit = real_backend_limit(&frozen);
        assert_eq!(
            record_backend_rejection(&frozen, limit, &memory).unwrap(),
            expected
        );
        assert_eq!(
            process.executable_cache().usage().unwrap().committed,
            before
        );
    }
    let work = work(&process, &mut reader);
    work.reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap()
        .analyze()
        .unwrap();
}

#[test]
fn backend_negative_rejects_code_changed_after_record_preparation() {
    let (process, memory, mut reader) = setup();
    let work = reshape(&process, &mut reader, 0x1000, 0x1004, 0x2000);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let limit = real_backend_limit(&frozen);
    let change = || {
        memory
            .write(
                AddressSpaceId::new(1),
                GuestVirtualAddress::new(0x2000),
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(0x91000c00),
            )
            .unwrap();
    };
    let observed = Observed {
        memory: &memory,
        runs: Mutex::new(Vec::new()),
        validations: AtomicUsize::new(0),
        during_validation: Some((2, &change)),
    };
    assert!(matches!(
        record_backend_rejection(&frozen, limit, &observed),
        Err(Failure::Cancelled)
    ));
    assert!(observed.validations.load(Ordering::Relaxed) >= 3);
}

fn optimized() -> (Arc<Lifetime>, ExecutionMemory, Reader) {
    let (process, memory, mut reader) = setup();
    {
        let work = work(&process, &mut reader);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        Compiler::new(host(), 0x10000)
            .unwrap()
            .publish(
                &mut Context::new(),
                &mut FunctionBuilderContext::new(),
                &frozen,
                &memory,
            )
            .unwrap();
    }
    process.try_service_links().unwrap();
    (process, memory, reader)
}

pub(super) fn reshape<'a>(
    process: &'a Lifetime,
    reader: &mut Reader,
    root: u64,
    source_pc: u64,
    target_pc: u64,
) -> Work<'a> {
    let (queue, outcome) = admit(process, reader, root, source_pc, target_pc);
    assert_eq!(outcome, Outcome::Queued);
    process
        .accept_background(queue.pop().unwrap().unwrap())
        .unwrap()
        .unwrap()
}

fn admit(
    process: &Lifetime,
    reader: &mut Reader,
    root: u64,
    source_pc: u64,
    target_pc: u64,
) -> (Queue, Outcome) {
    let queue = Queue::new(1, process).unwrap().unwrap();
    let outcome = admit_to(process, &queue, reader, root, source_pc, target_pc);
    (queue, outcome)
}

pub(super) fn admit_to(
    process: &Lifetime,
    queue: &Queue,
    reader: &mut Reader,
    root: u64,
    source_pc: u64,
    target_pc: u64,
) -> Outcome {
    let mut state = A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 100).unwrap());
    let mut payload = |pc| {
        unsafe { reader.admit(&mut frame, key(pc)) }
            .unwrap()
            .unwrap()
            .payload()
            .clone()
    };
    let source = payload(root);
    let target = payload(target_pc);
    let family = |entry: crate::abi::HcqEntry| FamilyIdentity {
        id: entry.family,
        version: entry.family_version,
    };
    let boundary = BoundaryKey {
        source: InstructionKey::new(key(source_pc)).unwrap(),
        target: InstructionKey::new(key(target_pc)).unwrap(),
        source_version: source.reachability(),
        target_version: target.reachability(),
        source_family: source.hcq().map(family),
        target_family: target.hcq().map(family),
    };
    let mut samples = Samples::new();
    let mut snapshot = None;
    for _ in 0..4 {
        snapshot = samples.boundary(boundary, true);
    }
    process
        .admit_reshape(queue, &mut samples, key(root), snapshot.unwrap())
        .unwrap()
}

#[test]
fn no_op_result_uses_real_memory_capture_and_does_not_emit_native_code() {
    let (process, memory, mut reader) = optimized();
    for expected in [true, false] {
        if !expected {
            let (queue, outcome) = admit(&process, &mut reader, 0x1000, 0x1004, 0x2000);
            assert_eq!(outcome, Outcome::Suppressed);
            assert!(queue.pop().unwrap().is_none());
            continue;
        }
        let work = reshape(&process, &mut reader, 0x1000, 0x1004, 0x2000);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        assert!(frozen.unchanged());
        let before = process.executable_cache().usage().unwrap().committed;
        assert_eq!(record_unchanged(&frozen, &memory).unwrap(), expected);
        assert_eq!(
            process.executable_cache().usage().unwrap().committed,
            before
        );
    }
}

#[test]
fn no_op_result_rejects_changed_code_during_final_memory_validation() {
    let (process, memory, mut reader) = optimized();
    let work = reshape(&process, &mut reader, 0x1000, 0x1004, 0x2000);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let change = || {
        memory
            .write(
                AddressSpaceId::new(1),
                GuestVirtualAddress::new(0x2000),
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(0x91000c00),
            )
            .unwrap();
    };
    let observed = Observed {
        memory: &memory,
        runs: Mutex::new(Vec::new()),
        validations: AtomicUsize::new(0),
        // Two disjoint runs are checked during capture; change the second run
        // after preparation, at the first check immediately before insertion.
        during_validation: Some((2, &change)),
    };
    assert!(matches!(
        record_unchanged(&frozen, &observed),
        Err(Failure::Cancelled)
    ));
    assert!(observed.validations.load(Ordering::Relaxed) >= 3);
}

#[test]
fn structural_capture_includes_disconnected_input_and_returns_pin_charge() {
    let (process, memory, mut reader) = setup();
    let work = reshape(&process, &mut reader, 0x7000, 0x7000, 0x2000);
    let Err(crate::hcq::DiscoveryError::Structural(result)) = Graph::discover(&work) else {
        panic!("return boundary must be disconnected")
    };
    assert_eq!(result.inspected(), 2);
    let observed = Observed {
        memory: &memory,
        runs: Mutex::new(Vec::new()),
        validations: AtomicUsize::new(0),
        during_validation: None,
    };
    let before = process.executable_cache().usage().unwrap();
    let image = Image::capture_structural(&result, &observed).unwrap();
    let mut runs = observed.runs.lock().unwrap().clone();
    runs.sort_unstable();
    assert_eq!(runs, [(0x2000, 2), (0x7000, 1)]);
    let after = process.executable_cache().usage().unwrap();
    assert_eq!(after.committed, before.committed);
    assert_eq!(after.metadata, before.metadata);
    memory
        .write(
            AddressSpaceId::new(1),
            GuestVirtualAddress::new(0x2000),
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(0x91000c00),
        )
        .unwrap();
    assert!(matches!(image.validate(&memory), Err(Failure::Cancelled)));
}

#[test]
fn structural_capture_rejects_discarded_input_mutation_during_validation() {
    let (process, memory, mut reader) = setup();
    let work = reshape(&process, &mut reader, 0x7000, 0x7000, 0x2000);
    let Err(crate::hcq::DiscoveryError::Structural(result)) = Graph::discover(&work) else {
        panic!("return boundary must be disconnected")
    };
    let change = || {
        memory
            .write(
                AddressSpaceId::new(1),
                GuestVirtualAddress::new(0x2000),
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(0x91000c00),
            )
            .unwrap();
    };
    let observed = Observed {
        memory: &memory,
        runs: Mutex::new(Vec::new()),
        validations: AtomicUsize::new(0),
        during_validation: Some((0, &change)),
    };
    assert!(matches!(
        Image::capture_structural(&result, &observed),
        Err(Failure::Cancelled)
    ));
    assert!(observed.validations.load(Ordering::Relaxed) > 0);
}

#[test]
fn disconnected_installation_uses_real_memory_without_backend_or_executable_allocation() {
    let (process, memory, mut reader) = setup();
    for expected in [true, false] {
        if !expected {
            let (queue, outcome) = admit(&process, &mut reader, 0x7000, 0x7000, 0x2000);
            assert_eq!(outcome, Outcome::Suppressed);
            assert!(queue.pop().unwrap().is_none());
            continue;
        }
        let work = reshape(&process, &mut reader, 0x7000, 0x7000, 0x2000);
        let Err(crate::hcq::DiscoveryError::Structural(result)) = Graph::discover(&work) else {
            panic!()
        };
        let before = process.executable_cache().usage().unwrap().committed;
        assert_eq!(record_structural(&result, &memory).unwrap(), expected);
        assert_eq!(
            process.executable_cache().usage().unwrap().committed,
            before
        );
    }
}

#[test]
fn structural_installation_rejects_code_mutation_after_preparing_record() {
    let (process, memory, mut reader) = setup();
    let work = reshape(&process, &mut reader, 0x7000, 0x7000, 0x2000);
    let Err(crate::hcq::DiscoveryError::Structural(result)) = Graph::discover(&work) else {
        panic!()
    };
    let change = || {
        memory
            .write(
                AddressSpaceId::new(1),
                GuestVirtualAddress::new(0x2000),
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(0x91000c00),
            )
            .unwrap();
    };
    let observed = Observed {
        memory: &memory,
        runs: Mutex::new(Vec::new()),
        validations: AtomicUsize::new(0),
        during_validation: Some((2, &change)),
    };
    assert!(matches!(
        record_structural(&result, &observed),
        Err(Failure::Cancelled)
    ));
    assert!(observed.validations.load(Ordering::Relaxed) >= 3);
}
