use super::*;

#[test]
fn civac_probes_keep_distinct_byte_fault_sites_without_runtime_exits() {
    // Repeated addresses and an unused result must not remove either access.
    let memory = memory(&[0xd50b7e20, 0xd50b7e20, 0xd50b7e3f, 0xd4200000]);
    let fragment = Fragment::capture(&memory, key()).unwrap();
    assert_eq!(fragment.instructions.len(), 4);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let lowered = Compiler::for_arena(abi, 0x10000)
            .unwrap()
            .lower(&fragment, CodeVersion::new(1).unwrap())
            .unwrap();
        assert_eq!(lowered.faults.len(), 3);
        for (index, fault) in lowered.faults.iter().enumerate() {
            assert_eq!(fault.access, crate::lifetime::unit::Access::CacheProbe);
            assert_eq!(fault.bytes, 1);
            assert_eq!(
                fault.instruction.block_key().pc.get(),
                PC + index as u64 * 4
            );
            assert_eq!(fault.commit_stage, 0);
            assert!(fault.completed_read.is_none());
            let bytes =
                &lowered.output.bytes[fault.native_start as usize..fault.native_end as usize];
            if abi == HostAbi::Aarch64 {
                // LDRB Wt,[Xn] (unsigned immediate), not PRFM or an elided read.
                assert_eq!(
                    u32::from_le_bytes(bytes.try_into().unwrap()) & 0xffc00000,
                    0x39400000
                );
            } else {
                // MOVZX r32, byte ptr [...], optionally preceded by REX.
                let opcode = usize::from(bytes[0] & 0xf0 == 0x40);
                assert_eq!(&bytes[opcode..opcode + 2], &[0x0f, 0xb6]);
            }
        }
        assert!(!lowered.states.iter().any(|map| {
            map.exit
                .is_some_and(|exit| matches!(exit.kind, EdgeKind::RuntimeSystem(_)))
        }));
    }
}

fn terminal_hot_offset(record: &StateRecord, key: BlockKey, bytes: &[u8]) -> u32 {
    let transfer = record.transfer.as_ref().unwrap();
    if transfer.static_target.is_none()
        && record.exit.is_some_and(|exit| {
            matches!(
                exit.kind,
                EdgeKind::Indirect | EdgeKind::Call | EdgeKind::Return
            )
        })
    {
        let exit = record.exit.unwrap();
        let probe = if exit.kind == EdgeKind::Return {
            crate::native::rsb::emit_return_probe(&record.state, key, transfer.destination).unwrap()
        } else {
            let mut prefix = if exit.kind == EdgeKind::Call {
                crate::native::rsb::emit_push(
                    &record.state,
                    key.at(GuestVirtualAddress::new(exit.pc.get().wrapping_add(4)))
                        .unwrap(),
                )
                .unwrap()
            } else {
                Vec::new()
            };
            prefix.extend(
                crate::native::pic::probe::emit(&record.state, key, transfer.destination).unwrap(),
            );
            prefix
        };
        let start = transfer.fallback_offset - probe.len() as u32;
        assert_eq!(
            &bytes[start as usize..transfer.fallback_offset as usize],
            probe
        );
        start
    } else {
        transfer.fallback_offset
    }
}

#[test]
fn published_lcq_reserves_islands_only_for_linkable_static_dispatch_sites() {
    for (words, count) in [
        (vec![0x1400_0000], 1),      // B self
        (vec![0x9400_0002], 1),      // BL
        (vec![0x5400_0040], 2),      // B.EQ: taken and fallthrough
        (vec![0xb400_0040], 2),      // CBZ
        (vec![0x3600_0040], 2),      // TBZ
        (vec![0xd61f_00a0], 0),      // BR X5
        (vec![0xd63f_00a0], 0),      // BLR X5
        (vec![0xd65f_03c0], 0),      // RET
        (vec![0xd400_0001], 0),      // SVC
        (vec![0xd420_0000], 0),      // BRK
        (vec![0xd50b_7520], 0),      // IC IVAU,X0: runtime completion, not a link
        (vec![0x0000_0000], 0),      // Unsupported word
        (vec![0xd503_201f; 512], 1), // Emergency fragment cut
    ] {
        let memory = memory(&words);
        let cache = Cache::new().unwrap();
        let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
        let mut reader = process.register().unwrap();
        let Request::Owner(claim) = reader.claim(key()).unwrap() else {
            panic!()
        };
        let handle = Compiler::new(native_abi())
            .unwrap()
            .publish(
                Compilation::capture(claim, &memory).unwrap(),
                &process,
                &cache,
                &memory,
            )
            .unwrap_or_else(|error| panic!("publication of {:08x?}: {error:?}", &words[..1]));
        // Inspect the unpublished patch shape after explicit performance-work
        // deferral; the self edge is now automatically registered at publication.
        if let Some(mut transition) = process.try_transition().unwrap() {
            transition.wait_closed().unwrap();
            transition
                .batch()
                .unwrap()
                .complete_with_links_deferred()
                .unwrap();
            assert!(transition.try_reopen().unwrap());
        }
        let code = process.snapshot(handle).unwrap();
        assert_eq!(
            code.code.allocation.island_count(),
            count,
            "{:08x?}",
            &words[..1]
        );
        let targets: Vec<_> = code
            .states
            .iter()
            .filter_map(|state| {
                state
                    .transfer
                    .as_ref()
                    .and_then(|transfer| transfer.static_target)
            })
            .collect();
        assert_eq!(targets.len(), count);
        // No bytes were redirected by reservation/publication itself. Every
        // source still carries its exact static fallback or native PIC probe.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                code.code.allocation.address() as *const u8,
                code.code.allocation.len(),
            )
        };
        for state in &code.states {
            let Some(_) = &state.transfer else {
                continue;
            };
            let branch = crate::native::link::emit(
                native_abi(),
                (code.code.allocation.address() + state.native_offset as usize) as u64,
                (code.code.allocation.address() + terminal_hot_offset(state, key(), bytes) as usize)
                    as u64,
                0,
            )
            .unwrap();
            assert!(branch.island.is_none());
            assert_eq!(
                &bytes[state.native_offset as usize..][..branch.patch().len()],
                branch.patch()
            );
        }
        drop(code);
        assert!(process.try_shutdown().unwrap());
        assert_eq!(cache.usage().unwrap().committed, 0);
    }
}

#[test]
fn terminal_descriptors_keep_destinations_costs_and_safe_fallbacks() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for (word, expected) in [
            (0x1400_0000, vec![(EdgeKind::Static, Some(PC + 4), 2)]), // self-loop
            (0x9400_0002, vec![(EdgeKind::Call, Some(PC + 12), 2)]),
            (
                0x5400_0040,
                vec![
                    (EdgeKind::Taken, Some(PC + 12), 2),
                    (EdgeKind::NotTaken, Some(PC + 8), 2),
                ],
            ),
            (
                0xb400_0040,
                vec![
                    (EdgeKind::Taken, Some(PC + 12), 2),
                    (EdgeKind::NotTaken, Some(PC + 8), 2),
                ],
            ),
            (
                0x3600_0040,
                vec![
                    (EdgeKind::Taken, Some(PC + 12), 2),
                    (EdgeKind::NotTaken, Some(PC + 8), 2),
                ],
            ),
            (0xd61f_00a0, vec![(EdgeKind::Indirect, None, 2)]),
            (0xd63f_00a0, vec![(EdgeKind::Call, None, 2)]),
            (0xd65f_03c0, vec![(EdgeKind::Return, None, 2)]),
            (0xd400_0001, vec![(EdgeKind::SupervisorCall(0), None, 1)]),
            (0xd420_0000, vec![(EdgeKind::Breakpoint(0), None, 1)]),
            (0x0000_0000, vec![(EdgeKind::InvalidInstruction, None, 1)]),
        ] {
            let fragment = Fragment::capture(&memory(&[0xd503_201f, word]), key()).unwrap();
            let lowered = Compiler::new(abi)
                .unwrap()
                .lower(&fragment, CodeVersion::new(1).unwrap())
                .unwrap();
            assert_eq!(lowered.states.len(), expected.len(), "{word:08x}");
            for (record, (kind, target, completed)) in lowered.states.iter().zip(expected) {
                assert_eq!(
                    record.exit.unwrap(),
                    GuestExit {
                        pc: GuestVirtualAddress::new(PC + 4),
                        kind,
                        block_index: 0,
                        instruction_index: 1,
                    }
                );
                let transfer = record.transfer.as_ref().unwrap();
                assert_eq!(transfer.completed, completed, "{word:08x}");
                assert_eq!(
                    transfer.static_target,
                    target.map(|pc| key().at(GuestVirtualAddress::new(pc)).unwrap())
                );
                assert!(transfer.destination.valid(abi, 8));
                let backend = lowered
                    .output
                    .metadata
                    .states
                    .iter()
                    .find(|map| map.offset == record.native_offset && !map.entry)
                    .unwrap();
                assert_eq!(transfer.patch_bytes, backend.patch_bytes);
                assert_eq!(transfer.poll_offset, backend.poll.map(|poll| poll.offset));
                assert_eq!(backend.poll.is_some(), completed == 2);
                let mut repatched = lowered.output.bytes.to_vec();
                backend
                    .patch_exit(
                        &mut repatched,
                        0,
                        u64::from(terminal_hot_offset(
                            record,
                            fragment.key,
                            &lowered.output.bytes,
                        )),
                    )
                    .unwrap();
                if let Some(poll) = backend.poll {
                    assert_eq!(poll.completed, completed);
                    let end = backend.offset as usize;
                    let expected = match abi {
                        HostAbi::X86_64 => vec![0x49, 0x81, 0xee, 2, 0, 0, 0, 0x7e, 8],
                        HostAbi::Aarch64 => [0xf1000a94u32, 0x5400004d]
                            .into_iter()
                            .flat_map(u32::to_le_bytes)
                            .collect(),
                    };
                    assert_eq!(&repatched[end - expected.len()..end], expected);
                }
                assert_eq!(
                    &*repatched, &*lowered.output.bytes,
                    "retained fallback differs from emitted jump"
                );
            }
        }
        let fragment = Fragment::capture(&memory(&vec![0xd503_201f; 512]), key()).unwrap();
        let lowered = Compiler::new(abi)
            .unwrap()
            .lower(&fragment, CodeVersion::new(1).unwrap())
            .unwrap();
        let record = &lowered.states[0];
        assert_eq!(record.exit.unwrap().kind, EdgeKind::FragmentLimit);
        let transfer = record.transfer.as_ref().unwrap();
        assert_eq!(transfer.completed, 512);
        assert_eq!(transfer.static_target.unwrap().pc.get(), PC + 512 * 4);
    }
}

#[test]
fn straight_line_nops_add_no_per_instruction_scaffolding() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let mut compiler = Compiler::new(abi).unwrap();
        let mut measure = |count| {
            let mut words = vec![0xd503_201f; count];
            words.push(0xd420_0000);
            let memory = memory(&words);
            let fragment = Fragment::capture(&memory, key()).unwrap();
            let lowered = compiler
                .lower(&fragment, CodeVersion::new(1).unwrap())
                .unwrap();
            let instructions: usize = compiler
                .context
                .func
                .layout
                .blocks()
                .map(|block| compiler.context.func.layout.block_insts(block).count())
                .sum();
            (
                instructions,
                lowered.output.bytes.len(),
                lowered.states.len(),
            )
        };
        let short = measure(1);
        for count in [64, 511] {
            let long = measure(count);
            assert_eq!(long.0, short.0, "{abi:?}: CLIF grows with NOP count");
            // Only the terminal PC/work immediates change, not one poll per NOP.
            assert!(
                long.1 <= short.1 + 16,
                "{abi:?}: short={short:?} long={long:?}"
            );
            assert_eq!(long.2, short.2);
        }
    }
}

#[test]
fn maximum_fragment_has_one_attributed_fault_record_per_load() {
    const LOAD_COUNT: usize = 511;
    let mut words = vec![0xf940_0020; LOAD_COUNT]; // LDR X0,[X1]
    words.push(0xd420_0000);
    let memory = memory(&words);
    let fragment = Fragment::capture(&memory, key()).unwrap();
    assert_eq!(fragment.instructions.len(), 512);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let lowered = Compiler::for_arena(abi, 1 << 20)
            .unwrap()
            .lower(&fragment, CodeVersion::new(1).unwrap())
            .unwrap();
        assert_eq!(lowered.faults.len(), LOAD_COUNT);
        assert_eq!(lowered.output.metadata.faults.len(), LOAD_COUNT);
        for (index, fault) in lowered.faults.iter().enumerate() {
            assert!(fault.native_start < fault.native_end);
            assert_eq!(fault.bytes, 8);
            assert_eq!(fault.subaccess, 0);
            assert_eq!(fault.commit_stage, 0);
            lowered.states[fault.state_map as usize]
                .state
                .validate()
                .unwrap();
            if index > 0 {
                assert!(lowered.faults[index - 1].native_end <= fault.native_start);
                assert_ne!(lowered.faults[index - 1].state_map, fault.state_map);
            }
        }
    }
}
