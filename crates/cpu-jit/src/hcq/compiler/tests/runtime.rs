use super::*;
use crate::{
    abi::{InstructionKey, NativeFrame, PollBudget},
    executable::{Cache, Tier},
    hcq::tests::key,
    lcq::{Compilation, compiler::Compiler, invocation},
    lifetime::{Lifetime, Reader, compile::Request, unit::Input},
    sampling::Samples,
};
use nixe_cpu::{
    exclusive::ExclusiveMonitorState,
    memory::{ExecutionMemory, MemoryPermissions},
    state::a64::A64State,
};
use nixe_cpu_direct_memory::WorkerFaultContext;
use nixe_memory::{
    AddressSpaceId, DirectBackendPolicy, GuestPhysicalPageId, MemoryInvalidationSource,
};
use std::sync::Arc;

const ARENA: usize = 0x10000;

mod flags;
mod memory;

#[test]
fn hcq_fused_element_reads_and_updates_the_vector_accumulator() {
    #[cfg(target_arch = "x86_64")]
    if !(std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma")) {
        // Baseline x86 terminates capture at each exact FMA boundary, covered
        // by the LCQ typed-completion tests instead of this native HCQ graph.
        return;
    }
    use nixe_cpu_interpreter::execute_one;
    let words = [0x4f99_12fb, 0x4f99_5afb, 0xd420_0000];
    let graph = graph(&[(0x1000, &words)]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        if abi == HostAbi::X86_64 && !cfg!(target_arch = "x86_64") {
            // A foreign baseline x86 target has no FMA feature. Its exact
            // boundaries are not the native multi-instruction graph tested here.
            continue;
        }
        let _ = staged(&graph, &[0], abi, CodeVersion::new(1).unwrap());
    }
    let (mut reader, memory) = fixture(&graph, &[0], &[]);
    let mut worker = WorkerFaultContext::register().unwrap();
    let mut state = A64State::default();
    state.set_pc(0x1000);
    state.set_fpsr(1 << 27);
    let pack = |lanes: [f32; 4]| {
        lanes
            .into_iter()
            .enumerate()
            .fold(0, |v, (i, x)| v | (u128::from(x.to_bits()) << (32 * i)))
    };
    state.set_vector(23, pack([1.5, 2.5, 3.5, 4.5]));
    state.set_vector(25, pack([2.0, 3.0, 4.0, 5.0]));
    state.set_vector(27, pack([0.5; 4]));
    let mut expected = state.clone();
    for word in &words[..2] {
        execute_one(
            &nixe_cpu::platform::TargetPlatform::Switch1,
            &mut expected,
            *word,
        )
        .unwrap();
    }
    loop {
        let entry = key(state.pc());
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 100).unwrap());
        let exit = unsafe {
            invocation::run(
                &mut Samples::new(),
                &mut reader,
                &mut frame,
                &memory,
                &mut worker,
                &mut ExclusiveMonitorState::default(),
                entry,
            )
        }
        .unwrap()
        .unwrap();
        let invocation::Exit::Native { guest, .. } = exit else {
            panic!("expected native exit")
        };
        match guest.kind {
            EdgeKind::VectorFpFusedElement(operation) => {
                crate::lcq::fp::complete_vector_fused_element(operation, &mut state).unwrap()
            }
            EdgeKind::Breakpoint(0) => break,
            other => panic!("{other:?}"),
        }
    }
    assert_eq!(state, expected);
    assert_eq!(state.vector(27), Some(pack([-2.5, -4.5, -6.5, -8.5])));
}

#[test]
fn hcq_constant_maps_cross_native_links_and_exit_canonically() {
    // Integer/vector literals cross the published HCQ -> LCQ link directly.
    let graph = graph(&[(
        0x1000,
        &[
            0xd280_0540, // MOVZ X0,#42
            0x9100_0400, // ADD X0,X0,#1 (folded by HCQ)
            0x4f07_e7e0, // MOVI V0.16B,#0xff
            0x1400_0005, // B 0x1020 (external LCQ)
        ],
    )]);
    let extra = [(0x1020, &[0x9100_0401, 0x4ea0_1c01, 0xd420_0000][..])];
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (context, body) = emitted(&graph, &[0], abi);
        let code = context.compiled_code().unwrap();
        let mut found = false;
        for (index, exit) in body.exits.iter().enumerate() {
            let map = code
                .buffer
                .nixe_states
                .iter()
                .find(|map| !map.entry && map.id == index as u64 + 1)
                .unwrap();
            let state = exit
                .state
                .allocate(
                    abi,
                    CodeVersion::new(1).unwrap(),
                    index as u32,
                    &AllocatedBoundary::new(abi, code, map).unwrap(),
                )
                .unwrap();
            if let Some(binding) = state
                .bindings
                .iter()
                .find(|b| b.value == GuestValue::General(0))
            {
                assert_eq!(binding.location, crate::abi::ValueLocation::constant(43));
                found = true;
            }
        }
        assert!(found);
    }
    let (mut reader, memory) = fixture(&graph, &[0], &extra);
    let mut worker = WorkerFaultContext::register().unwrap();
    let mut state = A64State::default();
    state.set_pc(0x1000);
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 100).unwrap());
    let exit = unsafe {
        invocation::run(
            &mut Samples::new(),
            &mut reader,
            &mut frame,
            &memory,
            &mut worker,
            &mut ExclusiveMonitorState::default(),
            key(0x1000),
        )
    }
    .unwrap()
    .unwrap();
    let invocation::Exit::Native {
        returned, guest, ..
    } = exit
    else {
        panic!()
    };
    assert_eq!(returned.reason, NativeExitReason::Architectural);
    assert_eq!(guest.pc.get(), 0x1028);
    assert_eq!(frame.budget.slice_remaining, 94);
    assert_eq!(state.general_register_storage_mut()[0], 43);
    assert_eq!(state.general_register_storage_mut()[1], 44);
    assert_eq!(state.vector(0), Some(u128::MAX));
    assert_eq!(state.vector(1), Some(u128::MAX));
}

#[test]
fn hcq_constant_maps_reconstruct_the_prefix_on_native_fault() {
    let graph = graph(&[(
        0x1000,
        &[0xd280_0540, 0x4f07_e7e0, 0xf940_0022, 0xd420_0000],
    )]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (context, _) = emitted(&graph, &[0], abi);
        let code = context.compiled_code().unwrap();
        assert_eq!(code.buffer.nixe_faults.len(), 1);
        let map = &code.buffer.nixe_faults[0];
        for bits in [[42, 0], [u64::MAX; 2]] {
            assert!(
                map.values.iter().any(
                    |value| value.location == cranelift_codegen::nixe::Location::Constant(bits)
                )
            );
        }
    }
    let (mut reader, memory) = fixture(&graph, &[0], &[]);
    let mut worker = WorkerFaultContext::register().unwrap();
    let mut state = A64State::default();
    state.set_pc(0x1000);
    state.general_register_storage_mut()[1] = 0x4000; // Unmapped, inside the arena.
    state.general_register_storage_mut()[2] = 0x1234;
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 100).unwrap());
    let exit = unsafe {
        invocation::run(
            &mut Samples::new(),
            &mut reader,
            &mut frame,
            &memory,
            &mut worker,
            &mut ExclusiveMonitorState::default(),
            key(0x1000),
        )
    }
    .unwrap()
    .unwrap();
    assert!(matches!(
        exit,
        invocation::Exit::Memory {
            outcome: invocation::MemoryExit::Fault(_),
            ..
        }
    ));
    assert_eq!(frame.execution_epoch, 0);
    assert_eq!(frame.budget.slice_remaining, 98);
    assert_eq!(state.pc(), 0x1008);
    assert_eq!(state.general_register_storage_mut()[0], 42);
    assert_eq!(state.general_register_storage_mut()[2], 0x1234);
    assert_eq!(state.vector(0), Some(u128::MAX));
}

// Publish real code through existing lifetime machinery, without activating
// background admission or the production HCQ publication consumer (steps 6/7).
fn fixture(graph: &Graph, entries: &[usize], extra: &[(u64, &[u32])]) -> (Reader, ExecutionMemory) {
    fixture_with_memory(graph, entries, extra, ExecutionMemory::new())
}

fn fixture_with_memory(
    graph: &Graph,
    entries: &[usize],
    extra: &[(u64, &[u32])],
    mut memory: ExecutionMemory,
) -> (Reader, ExecutionMemory) {
    let abi = if cfg!(target_arch = "x86_64") {
        HostAbi::X86_64
    } else {
        HostAbi::Aarch64
    };
    let space = AddressSpaceId::new(1);
    let page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(page));
    for word in &graph.instructions {
        memory
            .initialize_ram(
                page,
                (word.instruction.key.block_key().pc.get() - 0x1000) as usize,
                &word.instruction.bits.to_le_bytes(),
            )
            .unwrap();
    }
    for &(pc, words) in extra {
        let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        memory
            .initialize_ram(page, (pc - 0x1000) as usize, &bytes)
            .unwrap();
    }
    assert!(memory.map_page(
        space,
        GuestVirtualAddress::new(0x1000),
        page,
        MemoryPermissions::READ_EXECUTE
    ));
    memory
        .bind_cpu_memory_backend(space, ARENA as u64, DirectBackendPolicy::Required)
        .unwrap();
    let cache = Cache::new().unwrap();
    let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
    let mut reader = process.register().unwrap();
    let mut lcq = Compiler::for_arena(abi, ARENA).unwrap();
    for key in graph
        .blocks
        .iter()
        .map(|block| block.key)
        .chain(extra.iter().map(|&(pc, _)| key(pc)))
    {
        let Request::Owner(claim) = reader.claim(key).unwrap() else {
            panic!()
        };
        lcq.publish(
            Compilation::capture(claim, &memory).unwrap(),
            &process,
            &cache,
            &memory,
        )
        .unwrap();
        process.try_service_links().unwrap();
    }
    let identity = process.begin_unit(Tier::Hcq).unwrap();
    let compiler = backend::Compiler::new(abi, ARENA).unwrap();
    let mut context = Context::new();
    let body = compiler
        .emit(
            &mut context,
            &mut FunctionBuilderContext::new(),
            graph,
            &Analysis::build(graph, entries),
            entries,
        )
        .unwrap();
    let image = compiler
        .finish(&mut context, body, graph, identity.version())
        .unwrap();
    let islands = image
        .states
        .iter()
        .filter(|state| {
            state
                .transfer
                .as_ref()
                .is_some_and(|t| t.static_target.is_some())
        })
        .count();
    let code = cache
        .install_with_islands(image.output, Tier::Hcq, islands, |_| None)
        .unwrap();
    let publications: Vec<_> = image
        .entries
        .iter()
        .map(|entry| process.reserve(entry.key).unwrap())
        .collect();
    let input = Input {
        identity,
        code,
        tier: Tier::Hcq,
        instructions: graph
            .instructions
            .iter()
            .map(|word| word.instruction)
            .collect(),
        entries: image.entries,
        // This fixture does not mutate memory or test publication revalidation.
        dependencies: Box::new([]),
        cursor: memory.invalidation_cursor(),
        states: image.states,
        faults: image.faults,
    };
    process
        .prepare_unit(&publications, input, memory.invalidation_signal())
        .unwrap()
        .publish()
        .unwrap();
    process.try_service_links().unwrap();
    (reader, memory)
}

#[test]
fn hcq_runtime_constant_branch_skips_dead_fault_and_preserves_accounting() {
    let graph = graph(&[
        (0x1000, &[0xd2800000, 0xb4000060]), // MOVZ X0,#0; CBZ X0,0x1010
        (0x1008, &[0xf9400021, RET]),        // dead load from unmapped X1
        (0x1010, &[0xd4200000]),
    ]);
    let (mut reader, memory) = fixture(&graph, &[0], &[]);
    let mut worker = WorkerFaultContext::register().unwrap();
    let mut state = A64State::default();
    state.set_pc(0x1000);
    state.general_register_storage_mut()[0] = 99;
    state.general_register_storage_mut()[1] = 0xffff;
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(100, 100).unwrap());
    let exit = unsafe {
        invocation::run(
            &mut Samples::new(),
            &mut reader,
            &mut frame,
            &memory,
            &mut worker,
            &mut ExclusiveMonitorState::default(),
            key(0x1000),
        )
    }
    .unwrap()
    .unwrap();
    let invocation::Exit::Native {
        returned, guest, ..
    } = exit
    else {
        panic!()
    };
    assert_eq!(returned.reason, NativeExitReason::Architectural);
    assert_eq!(guest.pc.get(), 0x1010);
    assert_eq!(frame.budget.slice_remaining, 98);
    assert_eq!(frame.execution_epoch, 0);
    assert_eq!(state.pc(), 0x1010);
    assert_eq!(state.general_register_storage_mut()[0], 0);
    assert_eq!(state.general_register_storage_mut()[1], 0xffff);
}

#[test]
fn hcq_runtime_internal_samples_resume_and_resolve_noncontiguous_source() {
    // Seed/first public entry is not the first captured instruction. Cycle
    // samples must resume SSA and never turn into family boundary heat.
    let graph = graph(&[
        (0x1040, &[0xf1000421, 0x54fffde1]), // SUBS X1,X1,#1; B.NE 0x1000
        (0x1000, &[0x91000400, 0x1400000f]), // ADD X0,X0,#1; B 0x1040
        (0x1048, &[0xd4200000]),
    ]);
    let entries = [block(&graph, 0x1040), block(&graph, 0x1000)];
    let (mut reader, memory) = fixture(&graph, &entries, &[]);
    let mut worker = WorkerFaultContext::register().unwrap();
    for entry in entries {
        for slice in [17, 8193] {
            let mut state = A64State::default();
            state.set_pc(graph.blocks[entry].key.pc.get());
            state.general_register_storage_mut()[1] = 10000;
            let mut samples = Samples::new();
            let mut frame = NativeFrame::new(&mut state, PollBudget::new(1, slice).unwrap());
            let exit = unsafe {
                invocation::run(
                    &mut samples,
                    &mut reader,
                    &mut frame,
                    &memory,
                    &mut worker,
                    &mut ExclusiveMonitorState::default(),
                    graph.blocks[entry].key,
                )
            }
            .unwrap()
            .unwrap();
            let invocation::Exit::Native {
                returned,
                guest,
                instruction,
                ..
            } = exit
            else {
                panic!()
            };
            assert_eq!(returned.reason, NativeExitReason::BudgetExhausted);
            assert_eq!(frame.execution_epoch, 0);
            let image: Vec<_> = graph
                .instructions
                .iter()
                .map(|word| word.instruction)
                .collect();
            let source = guest.source(|i| image.get(i).copied()).unwrap().1;
            assert_eq!(source.key, instruction.key);
            assert_eq!(source.bits, instruction.bits);
            assert!(matches!(guest.pc.get(), 0x1004 | 0x1044));
            assert_eq!(instruction.key.block_key().pc, guest.pc);
            let executed = slice - frame.budget.slice_remaining;
            let mut expected = A64State::default();
            expected.set_pc(graph.blocks[entry].key.pc.get());
            expected.general_register_storage_mut()[1] = 10000;
            for _ in 0..executed {
                let word = image
                    .iter()
                    .find(|word| word.key.block_key().pc.get() == expected.pc())
                    .unwrap();
                nixe_cpu_interpreter::execute_one(
                    &graph.blocks[entry].key.platform,
                    &mut expected,
                    word.bits,
                )
                .unwrap();
            }
            assert_eq!(state, expected);
            for block in &graph.blocks {
                assert!(samples.seed_snapshot(block.key).is_none());
                for source in &image {
                    assert!(
                        samples
                            .boundary_snapshot(source.key, InstructionKey::new(block.key).unwrap())
                            .is_none()
                    );
                }
            }
        }
    }
}

#[test]
fn hcq_runtime_canonical_pre_exit_uses_noncontiguous_image_and_block_prefix() {
    let graph = graph(&[
        (0x1000, &[0x91000400, 0x1400000f]), // ADD X0,X0,#1; B 0x1040
        (0x1040, &[0x91000400, 0xd42000e0]), // ADD X0,X0,#1; BRK #7
    ]);
    let entries = [0, block(&graph, 0x1040)];
    let (mut reader, memory) = fixture(&graph, &entries, &[]);
    let mut worker = WorkerFaultContext::register().unwrap();
    for (entry, completed, value) in [(0, 3, 2), (entries[1], 1, 1)] {
        let mut state = A64State::default();
        state.set_pc(graph.blocks[entry].key.pc.get());
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 100).unwrap());
        let exit = unsafe {
            invocation::run(
                &mut Samples::new(),
                &mut reader,
                &mut frame,
                &memory,
                &mut worker,
                &mut ExclusiveMonitorState::default(),
                graph.blocks[entry].key,
            )
        }
        .unwrap()
        .unwrap();
        let invocation::Exit::Native {
            returned,
            guest,
            instruction,
            ..
        } = exit
        else {
            panic!()
        };
        assert_eq!(returned.reason, NativeExitReason::Architectural);
        assert_eq!(guest.kind, EdgeKind::Breakpoint(7));
        assert_eq!(guest.pc.get(), 0x1044);
        assert_eq!(guest.block_index, 2);
        assert_eq!(guest.instruction_index, 3);
        assert_eq!(instruction.bits, 0xd42000e0);
        assert_eq!(frame.budget.slice_remaining, 100 - completed);
        assert_eq!(frame.execution_epoch, 0);
        assert_eq!(state.pc(), 0x1044);
        assert_eq!(state.general_register_storage_mut()[0], value);
    }
}

#[test]
fn hcq_runtime_fault_escape_charges_only_executed_blocks_and_local_prefix() {
    let graph = graph(&[
        (0x1000, &[0x91000400, 0x1400000f]), // ADD X0,X0,#1; B 0x1040
        (0x1040, &[0x91000400, 0xf9400040, 0xd4200000]), // ADD; LDR X0,[X2]; BRK
    ]);
    let entries = [0, block(&graph, 0x1040)];
    let (mut reader, memory) = fixture(&graph, &entries, &[]);
    let mut worker = WorkerFaultContext::register().unwrap();
    for (entry, completed, value) in [(0, 3, 2), (entries[1], 1, 1)] {
        let mut state = A64State::default();
        state.set_pc(graph.blocks[entry].key.pc.get());
        state.general_register_storage_mut()[2] = 0x2000; // Unmapped guest page.
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 100).unwrap());
        let exit = unsafe {
            invocation::run(
                &mut Samples::new(),
                &mut reader,
                &mut frame,
                &memory,
                &mut worker,
                &mut ExclusiveMonitorState::default(),
                graph.blocks[entry].key,
            )
        }
        .unwrap()
        .unwrap();
        let invocation::Exit::Memory {
            instruction,
            outcome: invocation::MemoryExit::Fault(fault),
            ..
        } = exit
        else {
            panic!()
        };
        assert_eq!(instruction.key.block_key(), key(0x1044));
        assert_eq!(instruction.bits, 0xf9400040);
        assert_eq!(fault.address.get(), 0x2000);
        assert_eq!(frame.budget.slice_remaining, 100 - completed);
        assert_eq!(frame.budget.sample_remaining, 4096 - completed);
        assert_eq!(frame.execution_epoch, 0);
        assert_eq!(state.pc(), 0x1044);
        assert_eq!(state.general_register_storage_mut()[0], value);
    }
}

#[test]
fn hcq_runtime_external_sample_uses_actual_source_not_root_or_last_word() {
    let graph = graph(&[
        (0x1000, &[0x14000010]), // B 0x1040
        (0x1040, &[0xd61f0040]), // BR X2
        (0x1060, &[0xd4200000]), // unrelated public entry, last word in image
    ]);
    let entries = [0, block(&graph, 0x1040), block(&graph, 0x1060)];
    let (mut reader, memory) = fixture(&graph, &entries, &[(0x1080, &[0xd4200000])]);
    let mut worker = WorkerFaultContext::register().unwrap();
    // Sample-only native callback and coincident sample/slice canonical exit.
    for slice in [100, 1] {
        let mut samples = Samples::new();
        let mut state = A64State::default();
        state.set_pc(0x1040);
        state.general_register_storage_mut()[2] = 0x1080;
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(1, slice).unwrap());
        unsafe {
            invocation::run(
                &mut samples,
                &mut reader,
                &mut frame,
                &memory,
                &mut worker,
                &mut ExclusiveMonitorState::default(),
                key(0x1040),
            )
        }
        .unwrap()
        .unwrap();
        let (snapshot, score) = samples
            .boundary_snapshot(
                InstructionKey::new(key(0x1040)).unwrap(),
                InstructionKey::new(key(0x1080)).unwrap(),
            )
            .unwrap();
        assert_eq!(score, 1);
        assert!(snapshot.key.source_family.is_some());
        assert!(snapshot.key.target_family.is_none());
        for wrong in [0x1000, 0x1060] {
            assert!(samples.seed_snapshot(key(wrong)).is_none());
            assert!(
                samples
                    .boundary_snapshot(
                        InstructionKey::new(key(wrong)).unwrap(),
                        InstructionKey::new(key(0x1080)).unwrap()
                    )
                    .is_none()
            );
        }
    }
}
