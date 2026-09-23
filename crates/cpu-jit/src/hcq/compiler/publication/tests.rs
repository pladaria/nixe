use super::*;
use crate::{
    abi::{CodeVersion, NativeFrame, PollBudget},
    executable::Cache,
    hcq::tests::key,
    lcq::{Compilation, compiler::Compiler as Lcq, invocation},
    lifetime::{
        Lifetime, Reader,
        background::{Queue, Work},
        compile::Request,
    },
    sampling::{AdmissionSnapshot, Samples},
};
use nixe_cpu::{
    error::InstructionFetchFault,
    exclusive::ExclusiveMonitorState,
    memory::{
        CodePageSpan, CpuMemory, ExecutionMemory, FetchedCode, InstructionMemory, MemoryAccess,
        MemoryAccessSize, MemoryPermissions, MemoryValue,
    },
    state::a64::A64State,
};
use nixe_cpu_direct_memory::WorkerFaultContext;
use nixe_memory::MemoryInvalidationCursor;
use nixe_memory::{
    AddressSpaceId, DirectBackendPolicy, GuestPhysicalPageId, MemoryInvalidation,
    MemoryInvalidationError,
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

mod coordination;
mod faults;
mod lifecycle;
mod mutation;
mod negative;
mod replacement;
mod worker;

fn host() -> HostAbi {
    if cfg!(target_arch = "x86_64") {
        HostAbi::X86_64
    } else {
        HostAbi::Aarch64
    }
}

fn setup() -> (Arc<Lifetime>, ExecutionMemory, Reader) {
    let mut memory = ExecutionMemory::new();
    for (pc, words) in [
        (0x1000, &[0x91000400u32, 0x140003ff][..]), // ADD X0,X0,#1; B 0x2000
        (0x2000, &[0x91000800, 0xd61f0040][..]),    // ADD X0,X0,#2; BR X2
        (0x3000, &[0x17fff800][..]),                // B 0x1000: external ingress at seed
        (0x4000, &[0x17fff800][..]),                // B 0x2000: select the second public entry
        (0x5000, &[0xd42000e0][..]),                // external BRK #7
        (0x6000, &[0xd61f0060][..]),                // BR X3: external PIC ingress
        (0x7000, &[0xd65f03c0][..]),                // RET X30: external return ingress
    ] {
        let page = GuestPhysicalPageId::new(pc / 4096);
        assert!(memory.add_ram_page(page));
        let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        memory.initialize_ram(page, 0, &bytes).unwrap();
        assert!(memory.map_page(
            AddressSpaceId::new(1),
            GuestVirtualAddress::new(pc),
            page,
            MemoryPermissions::READ_WRITE_EXECUTE
        ));
    }
    memory
        .bind_cpu_memory_backend(
            AddressSpaceId::new(1),
            0x10000,
            DirectBackendPolicy::Required,
        )
        .unwrap();
    let cache = Cache::new().unwrap();
    let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
    memory.set_mutation_observer(process.clone()).unwrap();
    let mut reader = process.register().unwrap();
    let mut compiler = Lcq::for_arena(host(), 0x10000).unwrap();
    for pc in [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000] {
        let Request::Owner(claim) = reader.claim(key(pc)).unwrap() else {
            panic!()
        };
        compiler
            .publish(
                Compilation::capture(claim, &memory).unwrap(),
                &process,
                &cache,
                &memory,
            )
            .unwrap();
        process.try_service_links().unwrap();
    }
    (process, memory, reader)
}

fn work<'a>(process: &'a Lifetime, reader: &mut Reader) -> Work<'a> {
    work_at(process, reader, 0x1000)
}

fn work_at<'a>(process: &'a Lifetime, reader: &mut Reader, pc: u64) -> Work<'a> {
    let mut state = A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 100).unwrap());
    let version = unsafe { reader.admit(&mut frame, key(pc)) }
        .unwrap()
        .unwrap()
        .payload()
        .reachability();
    let snapshot = AdmissionSnapshot {
        key: key(pc),
        version,
        sequence: 8,
        last_edge: None,
        successors: [None; 4],
    };
    let queue = Queue::new(1, process).unwrap().unwrap();
    process
        .admit_seed(&queue, &mut Samples::new(), snapshot)
        .unwrap();
    process
        .accept_background(queue.pop().unwrap().unwrap())
        .unwrap()
        .unwrap()
}

#[test]
fn hcq_trimmed_candidate_publishes_and_executes_a_native_cross_family_edge() {
    let (process, memory, mut reader) = setup();
    let first = work(&process, &mut reader);
    let graph = Graph::discover(&first).unwrap();
    let second = work_at(&process, &mut reader, 0x2000);
    let second = second
        .reserve_candidate(Graph::discover(&second).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let first = first.reserve_candidate(graph).unwrap().freeze().unwrap();
    assert_eq!(first.graph().instructions.len(), 2);
    assert_eq!(first.graph().units.len(), 1);
    assert_eq!(first.dependencies().len(), 1);
    assert_eq!(first.entries().len(), 1);
    assert_eq!(
        first.graph().blocks[0].exit,
        crate::hcq::Exit::Jump(crate::hcq::Target::External(key(0x2000)))
    );
    let compiler = Compiler::new(host(), 0x10000).unwrap();
    for frozen in [&second, &first] {
        compiler
            .publish(
                &mut Context::new(),
                &mut FunctionBuilderContext::new(),
                frozen,
                &memory,
            )
            .unwrap();
    }
    process.try_service_links().unwrap();
    let mut state = A64State::default();
    state.set_pc(0x1000);
    state.general_register_storage_mut()[2] = 0x5000;
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 100).unwrap());
    for pc in [0x1000, 0x2000] {
        assert!(
            unsafe { reader.admit(&mut frame, key(pc)) }
                .unwrap()
                .unwrap()
                .payload()
                .hcq()
                .is_some()
        );
    }
    let exit = unsafe {
        invocation::run(
            &mut Samples::new(),
            &mut reader,
            &mut frame,
            &memory,
            &mut WorkerFaultContext::register().unwrap(),
            &mut ExclusiveMonitorState::default(),
            key(0x1000),
        )
    }
    .unwrap()
    .unwrap();
    let invocation::Exit::Native { guest, .. } = exit else {
        panic!()
    };
    assert_eq!(guest.pc.get(), 0x5000);
    assert_eq!(guest.kind, EdgeKind::Breakpoint(7));
    assert_eq!(state.general_register_storage_mut()[0], 3);
}

#[test]
fn hcq_disjoint_claims_after_trimming_allow_parallel_backend_compilation() {
    use std::sync::mpsc;
    use std::time::Duration;
    let (process, _, mut reader) = setup();
    let first = work(&process, &mut reader);
    let graph = Graph::discover(&first).unwrap();
    let second = work_at(&process, &mut reader, 0x2000);
    let second = second
        .reserve_candidate(Graph::discover(&second).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let first = first.reserve_candidate(graph).unwrap().freeze().unwrap();
    let compiler = Compiler::new(host(), 0x10000).unwrap();
    std::thread::scope(|scope| {
        let (ready, received) = mpsc::channel();
        let mut releases = Vec::new();
        let mut joins = Vec::new();
        for frozen in [&first, &second] {
            let compiler = &compiler;
            let ready = ready.clone();
            let (release, wait) = mpsc::channel();
            releases.push(release);
            joins.push(scope.spawn(move || {
                let analysis = frozen.analyze().unwrap();
                let mut context = Context::new();
                let body = compiler
                    .emit(
                        &mut context,
                        &mut FunctionBuilderContext::new(),
                        frozen.graph(),
                        &analysis,
                        frozen.entries(),
                    )
                    .unwrap();
                ready.send(()).unwrap();
                wait.recv_timeout(Duration::from_secs(10)).unwrap();
                let staged = compiler
                    .finish(
                        &mut context,
                        body,
                        frozen.graph(),
                        CodeVersion::new(1).unwrap(),
                    )
                    .unwrap();
                assert_eq!(staged.entries.len(), 1);
                frozen.check().unwrap();
            }));
        }
        for _ in 0..2 {
            received.recv_timeout(Duration::from_secs(10)).unwrap();
        }
        // Both real frontends reached the backend with disjoint live claims;
        // finish one while the other still holds its compiler state and claims.
        releases.pop().unwrap().send(()).unwrap();
        joins.pop().unwrap().join().unwrap();
        releases.pop().unwrap().send(()).unwrap();
        joins.pop().unwrap().join().unwrap();
    });
}

struct Observed<'a> {
    memory: &'a ExecutionMemory,
    runs: Mutex<Vec<(u64, u16)>>,
    validations: AtomicUsize,
    during_validation: Option<(usize, &'a (dyn Fn() + Sync))>,
}

impl InstructionMemory for Observed<'_> {
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
impl ExecutableMemory for Observed<'_> {
    fn capture_instructions(
        &self,
        space: AddressSpaceId,
        start: GuestVirtualAddress,
        limit: NonZeroU16,
        stop: &dyn Fn(GuestVirtualAddress, u32) -> bool,
    ) -> InstructionImage {
        self.runs.lock().unwrap().push((start.get(), limit.get()));
        self.memory.capture_instructions(space, start, limit, stop)
    }
    fn image_is_current(&self, image: &InstructionImage) -> bool {
        let count = self.validations.fetch_add(1, Ordering::Relaxed);
        if let Some((at, action)) = self.during_validation
            && count == at
        {
            action();
        }
        self.memory.image_is_current(image)
    }
}
impl MemoryInvalidationSource for Observed<'_> {
    fn invalidation_cursor(&self) -> MemoryInvalidationCursor {
        self.memory.invalidation_cursor()
    }
    fn invalidation_signal(&self) -> &AtomicU64 {
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
fn hcq_publication_compiles_real_frozen_graph_and_executes_mixed_tier_entries() {
    let (process, memory, mut reader) = setup();
    let work = work(&process, &mut reader);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    assert_eq!(frozen.graph().instructions.len(), 4);
    assert_eq!(frozen.entries().len(), 2);
    let memory_view = Observed {
        memory: &memory,
        runs: Mutex::new(Vec::new()),
        validations: AtomicUsize::new(0),
        during_validation: None,
    };
    let handle = Compiler::new(host(), 0x10000)
        .unwrap()
        .publish(
            &mut Context::new(),
            &mut FunctionBuilderContext::new(),
            &frozen,
            &memory_view,
        )
        .unwrap();
    assert_eq!(
        *memory_view.runs.lock().unwrap(),
        [(0x1000, 2), (0x2000, 2)]
    );
    drop(frozen);
    drop(work);
    process.try_service_links().unwrap();
    let unit = process.snapshot(handle).unwrap();
    assert_eq!(unit.tier, Tier::Hcq);
    assert_eq!(unit.entries.len(), 2);
    assert_eq!(unit.dependencies.len(), 2);
    let mut worker = WorkerFaultContext::register().unwrap();
    for (pc, expected) in [(0x1000, 3), (0x2000, 2), (0x3000, 3), (0x4000, 2)] {
        let mut state = A64State::default();
        state.set_pc(pc);
        state.general_register_storage_mut()[2] = 0x5000;
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 100).unwrap());
        {
            let admission = unsafe { reader.admit(&mut frame, key(pc)) }
                .unwrap()
                .unwrap();
            assert_eq!(admission.payload().hcq().is_some(), pc < 0x3000);
        }
        let exit = unsafe {
            invocation::run(
                &mut Samples::new(),
                &mut reader,
                &mut frame,
                &memory,
                &mut worker,
                &mut ExclusiveMonitorState::default(),
                key(pc),
            )
        }
        .unwrap()
        .unwrap();
        let invocation::Exit::Native { guest, .. } = exit else {
            panic!()
        };
        assert_eq!(guest.pc.get(), 0x5000);
        assert_eq!(guest.kind, EdgeKind::Breakpoint(7));
        assert_eq!(state.general_register_storage_mut()[0], expected);
        process.try_service_links().unwrap();
    }
}

#[test]
fn hcq_publication_rejects_memory_changed_after_output_preparation() {
    let (process, memory, mut reader) = setup();
    let work = work(&process, &mut reader);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    // Native output and baseline pins exist when the final image check calls
    // this mutation. Compiler references must not block memory authority.
    let mutate = || {
        memory
            .overwrite_mapped_ram(
                AddressSpaceId::new(1),
                GuestVirtualAddress::new(0x2000),
                &0x91000c00u32.to_le_bytes(),
            )
            .unwrap();
    };
    let view = Observed {
        memory: &memory,
        runs: Mutex::new(Vec::new()),
        validations: AtomicUsize::new(0),
        during_validation: Some((4, &mutate)),
    };
    let mut context = Context::new();
    let result = Compiler::new(host(), 0x10000).unwrap().publish(
        &mut context,
        &mut FunctionBuilderContext::new(),
        &frozen,
        &view,
    );
    assert!(matches!(result, Err(Failure::Cancelled)));
    assert!(view.validations.load(Ordering::Relaxed) >= 5);
    assert!(context.compiled_code().is_none());
    assert_eq!(context.func.layout.blocks().count(), 0);
    drop(frozen);
    drop(work);
    process.try_service_links().unwrap();
    let mut state = A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 100).unwrap());
    if let Some(admission) = unsafe { reader.admit(&mut frame, key(0x1000)) }.unwrap() {
        assert!(admission.payload().hcq().is_none());
    }
}

#[test]
fn hcq_publication_detects_changed_captured_bytes_before_backend_work() {
    let (process, memory, mut reader) = setup();
    let work = work(&process, &mut reader);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    // A guest-style executable store may leave the old LCQ image callable
    // before IC. Revalidation still cannot compile different captured bytes.
    memory
        .write(
            AddressSpaceId::new(1),
            GuestVirtualAddress::new(0x2000),
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(0x91000c00),
        )
        .unwrap();
    let mut context = Context::new();
    let result = Compiler::new(host(), 0x10000).unwrap().publish(
        &mut context,
        &mut FunctionBuilderContext::new(),
        &frozen,
        &memory,
    );
    assert!(matches!(result, Err(Failure::Cancelled)));
    assert_eq!(context.func.layout.blocks().count(), 0);
}

#[test]
fn hcq_publication_keeps_output_across_unrelated_memory_mutations_at_each_phase() {
    for phase in [0, 2, 4] {
        let (process, memory, mut reader) = setup();
        let work = work(&process, &mut reader);
        let frozen = work
            .reserve_candidate(Graph::discover(&work).unwrap())
            .unwrap()
            .freeze()
            .unwrap();
        let cursor = memory.invalidation_cursor();
        let mutate = || {
            // This input belongs to a different LCQ unit, not this region.
            memory
                .overwrite_mapped_ram(
                    AddressSpaceId::new(1),
                    GuestVirtualAddress::new(0x5000),
                    &0xd42000e0u32.to_le_bytes(),
                )
                .unwrap();
        };
        let view = Observed {
            memory: &memory,
            runs: Mutex::new(Vec::new()),
            validations: AtomicUsize::new(0),
            during_validation: Some((phase, &mutate)),
        };
        let handle = Compiler::new(host(), 0x10000)
            .unwrap()
            .publish(
                &mut Context::new(),
                &mut FunctionBuilderContext::new(),
                &frozen,
                &view,
            )
            .unwrap();
        assert_ne!(memory.invalidation_cursor(), cursor);
        process.try_service_links().unwrap();
        assert_eq!(process.snapshot(handle).unwrap().instructions.len(), 4);
        assert_eq!(view.runs.lock().unwrap().len(), 2); // No recapture/recompilation loop.
    }
}
