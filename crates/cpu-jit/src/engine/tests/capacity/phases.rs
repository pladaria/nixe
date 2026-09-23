use super::*;
use crate::{
    executable::SEGMENT_BYTES,
    lifetime::{
        background::{Work, workers::Resources},
        unit::Snapshot,
    },
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

fn fixture() -> (JitThread, Arc<AtomicUsize>) {
    let memory = memory(DirectBackendPolicy::Required);
    for (offset, add, brk) in [
        (0, 0x91000400u32, 0xd4200020u32),
        (16, 0x91000800, 0xd4200040),
    ] {
        memory
            .overwrite_mapped_ram(
                SPACE,
                PC.checked_add(offset).unwrap(),
                &add.to_le_bytes()
                    .into_iter()
                    .chain(brk.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    }
    let compilations = Arc::new(AtomicUsize::new(0));
    let counted = compilations.clone();
    let process = Arc::new(
        JitProcess::with_compiler(cpu(), memory, 1, |size, memory| {
            let compiler = crate::hcq::worker::consumer(
                if cfg!(target_arch = "x86_64") {
                    HostAbi::X86_64
                } else {
                    HostAbi::Aarch64
                },
                size,
                memory,
            )?;
            Ok(move |resources: &mut Resources, work: Work<'_>| {
                let result = compiler(resources, work);
                counted.fetch_add(1, Ordering::Release);
                result
            })
        })
        .unwrap(),
    );
    (JitThread::new(process).unwrap(), compilations)
}

fn await_jobs(compilations: &AtomicUsize, count: usize) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while compilations.load(Ordering::Acquire) < count {
        assert!(
            Instant::now() < deadline,
            "compiler callback did not complete"
        );
        std::thread::yield_now();
    }
    assert_eq!(compilations.load(Ordering::Acquire), count);
}

fn resident(thread: &mut JitThread, pc: GuestVirtualAddress, optimized: bool) -> Option<Snapshot> {
    let mut state = A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 10).unwrap());
    let key = thread.key(pc).unwrap();
    let mut invocation = match unsafe { thread.reader.admit(&mut frame, key) } {
        Ok(value) => value?,
        Err(lifetime::Error::Closed) => return None,
        Err(error) => panic!("admission: {error:?}"),
    };
    let payload = invocation.payload();
    let entry = if optimized {
        payload.hcq()?.entry
    } else {
        payload.lcq()?
    };
    let (_, directory) = invocation.frame_and_faults();
    let handle = directory
        .unit(entry.canonical.get())
        .unwrap()
        .registered_handle()
        .unwrap();
    drop(invocation);
    thread.process.lifetime.snapshot(handle).ok()
}

fn promote(thread: &mut JitThread, native: &mut NativeWorker, pc: GuestVirtualAddress) -> Snapshot {
    assert!(matches!(thread.demand(pc).unwrap(), Demand::Ready));
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        thread.process.lifetime.try_service_links().unwrap();
        if let Some(unit) = resident(thread, pc, true) {
            return unit;
        }
        assert!(Instant::now() < deadline, "HCQ promotion did not finish");
        let mut state = A64State::default();
        state.set_pc(pc.get());
        match thread.invoke(
            &mut crate::ReturnStack::default(),
            native,
            &mut state,
            PollBudget::new(1, 4).unwrap(),
            &VcpuEventState::default(),
        ) {
            Ok(_) | Err(invocation::Error::Lifetime(lifetime::Error::Closed)) => {}
            Err(error) => panic!("sampling: {error:?}"),
        }
        std::thread::yield_now();
    }
}

fn execute(
    thread: &mut JitThread,
    native: &mut NativeWorker,
    pc: GuestVirtualAddress,
    increment: u64,
) {
    let mut state = A64State::default();
    state.set_pc(pc.get());
    state.general_register_storage_mut()[0] = 100;
    let report = thread
        .run_slice(
            &mut crate::ReturnStack::default(),
            native,
            &mut state,
            10,
            &Timer,
            &VcpuEventState::default(),
        )
        .unwrap();
    assert_eq!(report.progress, 2); // ADD; BRK, not compilation or maintenance.
    assert!(
        matches!(report.stop, CpuExit::ArchitecturalException { syndrome: Some(value), .. } if value == increment)
    );
    assert_eq!(state.general_register_storage_mut()[0], 100 + increment);
    assert_eq!(state.pc(), pc.get() + 4);
}

#[test]
fn changing_hot_regions_recover_real_hcq_and_baselines_then_stabilize() {
    let (mut thread, compilations) = fixture();
    let process = thread.process.clone();
    let cache = process.lifetime.executable_cache();
    let mut native = NativeWorker::default();
    let mut warmed_usage = None;
    for phase in 0..4 {
        let pc = PC.checked_add((phase % 2) * 16).unwrap();
        let next = PC.checked_add(((phase + 1) % 2) * 16).unwrap();
        let optimized = promote(&mut thread, &mut native, pc);
        await_jobs(&compilations, phase as usize + 1);
        let baseline = resident(&mut thread, pc, false).unwrap();
        let old_hcq = optimized.registered_handle().unwrap();
        let old_lcq = baseline.registered_handle().unwrap();
        let old_address = baseline.code.allocation.address();
        let old_generation = baseline.code.allocation.generation;
        assert_eq!(cache.usage().unwrap().committed, 2 * SEGMENT_BYTES);
        assert!(matches!(
            process.lifetime.retire_unit(old_lcq),
            Err(lifetime::Error::PinnedBaseline)
        ));
        drop(optimized);
        drop(baseline);
        let charge = cache
            .charge_metadata(SOFT_BYTES - cache.usage().unwrap().total(), Tier::Lcq)
            .unwrap();
        // The miss takes the production run_slice capacity path. The live HCQ
        // must release its LCQ promise before both actual segments can decommit.
        execute(&mut thread, &mut native, next, (phase + 1) % 2 + 1);
        assert!(matches!(
            process.lifetime.snapshot(old_hcq),
            Err(lifetime::Error::StaleUnit)
        ));
        assert!(matches!(
            process.lifetime.snapshot(old_lcq),
            Err(lifetime::Error::StaleUnit)
        ));
        assert_eq!(cache.usage().unwrap().committed, SEGMENT_BYTES);
        assert!(cache.usage().unwrap().total() < SOFT_BYTES);
        let current = resident(&mut thread, next, false).unwrap();
        assert_eq!(current.code.allocation.address(), old_address);
        assert_ne!(current.code.allocation.generation, old_generation);
        let identity = (current.id, current.version);
        drop(current);
        drop(charge);
        // A fitting, unchanged set must neither evict nor recompile each slice.
        // Sixty-four instructions stay below the next ordinary hotness sample.
        let jobs = compilations.load(Ordering::Acquire);
        let usage = cache.usage().unwrap();
        for _ in 0..32 {
            execute(&mut thread, &mut native, next, (phase + 1) % 2 + 1);
            let current = resident(&mut thread, next, false).unwrap();
            assert_eq!((current.id, current.version), identity);
        }
        assert_eq!(compilations.load(Ordering::Acquire), jobs);
        assert_eq!(cache.usage().unwrap(), usage);
        // Thread profile/root capacity warms once; steady live/cache ownership
        // must not grow with repeated eviction and native recompilation.
        if phase > 0 {
            assert_eq!(*warmed_usage.get_or_insert(usage), usage);
        }
    }
    assert!(process.try_shutdown().unwrap());
    assert_eq!(cache.usage().unwrap().committed, 0);
    native.finish().unwrap();
}

#[test]
fn retained_hcq_and_unpublished_bridges_preserve_capacity_then_allow_lcq_reserve_and_hcq_retry() {
    use crate::executable::output::{Metadata, Output};
    let (mut thread, compilations) = fixture();
    let process = thread.process.clone();
    let cache = process.lifetime.executable_cache();
    let mut native = NativeWorker::default();
    let old = promote(&mut thread, &mut native, PC);
    let baseline = resident(&mut thread, PC, false).unwrap();
    await_jobs(&compilations, 1);
    let handle = old.registered_handle().unwrap();
    let host = if cfg!(target_arch = "x86_64") {
        HostAbi::X86_64
    } else {
        HostAbi::Aarch64
    };
    let stage = || {
        let (bytes, tail) = crate::native::link::bridge(host, &[]);
        let mut bytes = bytes.into_vec();
        bytes.resize(tail + 16, 0);
        let output = Output {
            bytes: bytes.into_boxed_slice(),
            alignment: 16,
            metadata: Metadata {
                abi: host,
                frame_extent: 0,
                entries: Box::new([]),
                states: Box::new([]),
                faults: Box::new([]),
                traps: Box::new([]),
                relocations: Box::new([]),
            },
        };
        cache
            .install_with_inline_branch(output, Tier::Hcq, tail, old.code.allocation.address())
            .unwrap()
    };
    // These are actual W^X installed bridge spans, not published unit records.
    // Keep their target snapshot until all unpublished references are released.
    let first = stage();
    let middle = stage();
    let last = stage();
    let hole = middle.allocation.address();
    drop(middle);
    let reused = stage();
    assert_eq!(reused.allocation.address(), hole);
    drop(reused);
    let charge = cache
        .charge_metadata(HARD_BYTES - cache.usage().unwrap().total(), Tier::Lcq)
        .unwrap();
    let next = PC.checked_add(16).unwrap();
    let mut state = A64State::default();
    state.set_pc(next.get());
    state.general_register_storage_mut()[0] = 100;
    let before = state.clone();
    let error = thread
        .run_slice(
            &mut crate::ReturnStack::default(),
            &mut native,
            &mut state,
            10,
            &Timer,
            &VcpuEventState::default(),
        )
        .unwrap_err();
    assert_eq!(error.kind, CpuFaultKind::Unavailable);
    assert!(error.message.contains("LCQ capacity at"));
    assert_eq!(error.progress, 0);
    assert_eq!(state, before);
    assert_eq!(*error.context, state.register_context());
    assert!(matches!(
        process.lifetime.snapshot(handle),
        Err(lifetime::Error::StaleUnit)
    ));
    assert_eq!(old.instructions.get(0).unwrap().bits, 0x91000400);
    assert!(cache.usage().unwrap().total() <= HARD_BYTES);
    assert!(cache.usage().unwrap().committed >= SEGMENT_BYTES);
    // Releasing one unpublished span is insufficient: snapshots and the other
    // unpublished span still protect the two retired segments.
    drop(first);
    process.lifetime.try_service_links().unwrap();
    assert_eq!(cache.usage().unwrap().committed, 2 * SEGMENT_BYTES);
    drop(last);
    drop(old);
    drop(baseline);
    execute(&mut thread, &mut native, next, 2);
    assert_eq!(cache.usage().unwrap().committed, SEGMENT_BYTES);
    assert!(cache.usage().unwrap().total() > SOFT_BYTES);
    assert!(cache.usage().unwrap().total() > HARD_BYTES - 32 * 1024 * 1024);
    assert!(cache.usage().unwrap().total() <= HARD_BYTES);
    // LCQ can execute inside its headroom; hotness must not start background
    // compilation while the external charge keeps optimization suppressed.
    for _ in 0..16 {
        state.set_pc(next.get());
        thread
            .invoke(
                &mut crate::ReturnStack::default(),
                &mut native,
                &mut state,
                PollBudget::new(1, 4).unwrap(),
                &VcpuEventState::default(),
            )
            .unwrap();
    }
    assert!(resident(&mut thread, next, true).is_none());
    assert_eq!(compilations.load(Ordering::Acquire), 1);
    drop(charge);
    let optimized = promote(&mut thread, &mut native, next);
    await_jobs(&compilations, 2);
    assert_eq!(optimized.instructions.get(0).unwrap().bits, 0x91000800);
    drop(optimized);
    execute(&mut thread, &mut native, next, 2);
    assert!(process.try_shutdown().unwrap());
    assert_eq!(cache.usage().unwrap().committed, 0);
    native.finish().unwrap();
}

#[test]
fn cold_demand_pressure_waits_for_old_fault_epoch_then_executes_the_new_region() {
    let (mut thread, compilations) = fixture();
    let process = thread.process.clone();
    let cache = process.lifetime.executable_cache();
    // ADD X0,X0,#1; LDR X3,literal; BRK #1. Reuse the tested scalar-literal
    // lowering to give the real optimized region a precise native fault map.
    let words = [0x91000400u32, 0x58000023, 0xd4200020];
    process
        .memory
        .overwrite_mapped_ram(
            SPACE,
            PC,
            &words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let mut native = NativeWorker::default();
    let old = promote(&mut thread, &mut native, PC);
    await_jobs(&compilations, 1);
    let handle = old.registered_handle().unwrap();
    let identity = (old.id, old.version);
    let fault_pc = old.code.allocation.address() + old.faults[0].native_start as usize;
    drop(old);
    let charge = cache
        .charge_metadata(SOFT_BYTES - cache.usage().unwrap().total(), Tier::Lcq)
        .unwrap();
    let mut state = A64State::default();
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 10).unwrap());
    let key = thread.key(PC).unwrap();
    let lease = process.memory.acquire_execution_lease();
    let invocation = unsafe { thread.reader.admit(&mut frame, key) }
        .unwrap()
        .unwrap();
    let found = invocation.fault(fault_pc).unwrap();
    assert_eq!((found.unit.id, found.unit.version), identity);
    let worker_process = process.clone();
    let pressure = std::thread::spawn(move || {
        let mut thread = JitThread::new(worker_process).unwrap();
        let mut native = NativeWorker::default();
        execute(&mut thread, &mut native, PC.checked_add(16).unwrap(), 2);
        native.finish().unwrap();
    });
    let deadline = Instant::now() + Duration::from_secs(30);
    while process.lifetime.control_word().load(Ordering::Acquire) == 0 {
        assert!(
            Instant::now() < deadline,
            "demand did not request pressure recovery"
        );
        std::thread::yield_now();
    }
    assert!(
        !pressure.is_finished(),
        "pressure bypassed the active fault reader"
    );
    assert_eq!(cache.usage().unwrap().committed, 2 * SEGMENT_BYTES);
    assert_eq!((found.unit.id, found.unit.version), identity);
    assert_eq!(
        invocation.fault(fault_pc).unwrap().instruction().bits,
        0x58000023
    );
    // No strong compiler snapshot is left: only the real invocation/lease
    // prevents the synchronous pressure caller from unlinking/reusing code.
    drop(invocation);
    drop(lease);
    pressure.join().unwrap();
    assert!(matches!(
        process.lifetime.snapshot(handle),
        Err(lifetime::Error::StaleUnit)
    ));
    assert_eq!(cache.usage().unwrap().committed, SEGMENT_BYTES);
    assert!(cache.usage().unwrap().total() < SOFT_BYTES);
    drop(charge);
    assert!(process.try_shutdown().unwrap());
    assert_eq!(cache.usage().unwrap().committed, 0);
    native.finish().unwrap();
}
