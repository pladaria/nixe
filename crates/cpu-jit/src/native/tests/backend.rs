//! Real fork output -> shared contracts -> Nixe adapters -> system gateway.
//! The byte owner below is test-only, pending Task 2's publication allocator.
use super::*;
use cranelift_codegen::{
    CompiledCode, Context,
    control::ControlPlane,
    cursor::{Cursor, FuncCursor},
    ir::{self, InstBuilder, MemFlagsData, types},
    isa::{self, CallConv},
    nixe::{EntryConstraint, Location, StateMap},
    settings::{self, Configurable},
};
use nixe_cpu::state::a64::{A64State, Nzcv};

fn compile(abi: HostAbi, allocator: &str, mut function: ir::Function) -> CompiledCode {
    let mut flags = settings::builder();
    for (name, value) in [
        ("enable_pinned_reg", "true"),
        ("enable_nixe_abi", "true"),
        ("regalloc_checker", "true"),
        ("machine_code_cfg_info", "true"),
        ("regalloc_algorithm", allocator),
        (
            "opt_level",
            if allocator == "single_pass" {
                "none"
            } else {
                "speed"
            },
        ),
    ] {
        flags.set(name, value).unwrap();
    }
    let triple = if abi == HostAbi::X86_64 {
        "x86_64-unknown-linux-gnu"
    } else {
        "aarch64-unknown-linux-gnu"
    };
    let mut builder = isa::lookup(triple.parse().unwrap()).unwrap();
    if abi == HostAbi::X86_64 {
        flags.set("enable_nixe_ibt", "true").unwrap();
    } else {
        builder.set("use_bti", "true").unwrap();
    }
    let target = builder.finish(settings::Flags::new(flags)).unwrap();
    let entry = function.layout.entry_block().unwrap();
    cranelift_codegen::nixe::set_entries(&mut function, &[entry]).unwrap();
    let mut context = Context::for_function(function);
    context.set_disasm(true);
    context
        .compile(&*target, &mut ControlPlane::default())
        .unwrap();
    let code = context.take_compiled_code().unwrap();
    // These fixtures intentionally have no external symbols. Never drop real
    // relocations when transferring compiled bytes to their eventual owner.
    assert!(code.buffer.relocs().is_empty());
    let text = code.vcode.as_ref().unwrap();
    assert!(!text.contains("%rsp"), "{text}");
    assert!(!text.contains(", sp"), "{text}");
    assert!(
        !text.lines().any(|line| {
            let line = line.trim_start();
            line.starts_with("call") || line.starts_with("ret")
        }),
        "{text}"
    );
    code
}

fn operands(count: u8) -> Vec<(GuestValue, usize)> {
    let mut operands = Vec::new();
    for i in 0..count {
        operands.push((GuestValue::General(i), operands.len()));
        operands.push((GuestValue::Vector(i), operands.len()));
    }
    operands.push((GuestValue::Fpsr, operands.len()));
    operands
}

fn live(operands: &[(GuestValue, usize)]) -> StateSet {
    operands.iter().fold(
        StateSet {
            nzcv: NZCV,
            ..StateSet::default()
        },
        |live, &(value, _)| live.union(value.state().unwrap()),
    )
}

fn fragment(
    operands: &[(GuestValue, usize)],
    ingress: Option<&[Location]>,
    fp: bool,
) -> ir::Function {
    let mut function = ir::Function::new();
    let block = function.dfg.make_block();
    function.layout.append_block(block);
    let mut signature = ir::Signature::new(CallConv::SystemV);
    signature.returns = operands
        .iter()
        .map(|(value, _)| {
            ir::AbiParam::new(match value {
                GuestValue::Vector(_) => types::I8X16,
                GuestValue::Fpsr => types::I32,
                _ => types::I64,
            })
        })
        .chain([ir::AbiParam::new(types::I32)])
        .collect();
    let signature = function.import_signature(signature);
    if let Some(ingress) = ingress {
        function.nixe_entry_constraints.insert(
            10,
            ingress
                .iter()
                .map(|location| match *location {
                    Location::Register { index, vector } => {
                        EntryConstraint::Register { index, vector }
                    }
                    _ => panic!("small link fixture must fit in registers"),
                })
                .collect(),
        );
    }
    let mut cursor = FuncCursor::new(&mut function).at_bottom(block);
    let entry = cursor.ins().nixe_entry(signature, 10);
    let mut values = cursor.func.dfg.inst_results(entry).to_vec();
    for &(value, index) in operands {
        if matches!(value, GuestValue::General(_)) {
            values[index] = cursor.ins().iadd_imm_s(values[index], 1);
        }
    }
    if fp {
        let vector = cursor.ins().bitcast(
            types::F64X2,
            MemFlagsData::new().with_endianness(ir::Endianness::Little),
            values[1],
        );
        let numerator = cursor.ins().extractlane(vector, 0);
        let zero = cursor.ins().f64const(0.0);
        let quotient = cursor.ins().fdiv(numerator, zero);
        let vector = cursor.ins().insertlane(vector, quotient, 0);
        values[1] = cursor.ins().bitcast(
            types::I8X16,
            MemFlagsData::new().with_endianness(ir::Endianness::Little),
            vector,
        );
    }
    cursor.ins().nixe_exit(20, &values);
    function
}

fn boundary(abi: HostAbi, code: &CompiledCode, id: u64) -> AllocatedBoundary<'_> {
    let map = code
        .buffer
        .nixe_states
        .iter()
        .find(|map| map.id == id)
        .unwrap();
    if map.entry {
        assert!(
            code.buffer
                .nixe_entries
                .iter()
                .any(|&(_, offset)| offset == map.offset)
        );
        let landing = if abi == HostAbi::X86_64 {
            [0xf3, 0x0f, 0x1e, 0xfa]
        } else {
            0xd503249fu32.to_le_bytes()
        };
        assert_eq!(
            &code.code_buffer()[map.offset as usize..map.offset as usize + 4],
            &landing
        );
    }
    AllocatedBoundary::new(abi, code, map).unwrap()
}

fn entry(
    abi: HostAbi,
    map: &AllocatedBoundary<'_>,
    operands: &[(GuestValue, usize)],
) -> EntryContract {
    let entry = EntryContract {
        abi,
        live_in: live(operands),
        bindings: map.bindings(operands).unwrap(),
        nzcv: NzcvLocation::Packed(map.location(operands.len(), types::I32).unwrap()),
    };
    entry.validate().unwrap();
    entry
}

fn exit(
    abi: HostAbi,
    map: &AllocatedBoundary<'_>,
    operands: &[(GuestValue, usize)],
    version: u64,
    deferred: bool,
) -> ExitStateMap {
    let mut dirty = live(operands);
    // Clean vectors must remain canonical, although they are physically live.
    dirty.vector.fill(false);
    dirty.vector[0] = true;
    let exit = ExitStateMap {
        abi,
        site: ExitSiteKey {
            source: CodeVersion::new(version).unwrap(),
            state_map: map.map.id.try_into().unwrap(),
        },
        live: live(operands),
        dirty_live: dirty,
        bindings: map.bindings(operands).unwrap(),
        nzcv: if deferred {
            NzcvLocation::Deferred(LazyFlags::Logical {
                result: map.location(0, types::I64).unwrap(),
                width: 64,
            })
        } else {
            NzcvLocation::Packed(map.location(operands.len(), types::I32).unwrap())
        },
        host_fpsr_pending: true,
    };
    exit.validate().unwrap();
    exit
}

fn append(bytes: &mut Vec<u8>, part: &[u8]) -> usize {
    bytes.resize(bytes.len().next_multiple_of(16), 0);
    let offset = bytes.len();
    bytes.extend_from_slice(part);
    offset
}

fn branch(abi: HostAbi, bytes: &mut [u8], from: usize, to: usize) {
    StateMap {
        id: 0,
        offset: from.try_into().unwrap(),
        entry: false,
        patch_bytes: if abi == HostAbi::X86_64 { 8 } else { 4 },
        values: vec![],
    }
    .patch_exit(bytes, 0, to as u64)
    .unwrap();
}

fn reserve_branch(abi: HostAbi, bytes: &mut Vec<u8>) -> usize {
    while !bytes.len().is_multiple_of(8) {
        match abi {
            HostAbi::X86_64 => bytes.push(0x90),
            HostAbi::Aarch64 => bytes.extend_from_slice(&0xd503201fu32.to_le_bytes()),
        }
    }
    let offset = bytes.len();
    bytes.resize(offset + 8, 0);
    offset
}

fn canonical_ingress(
    abi: HostAbi,
    bytes: &mut Vec<u8>,
    contract: &EntryContract,
    fast: usize,
) -> usize {
    let mut ingress = gateway::landing(abi);
    ingress.extend(emit_canonical_entry(contract).unwrap());
    let jump = reserve_branch(abi, &mut ingress);
    let offset = append(bytes, &ingress);
    branch(abi, bytes, offset + jump, fast);
    offset
}

fn execute(abi: HostAbi, bytes: &[u8], address_offset: usize, count: u8, increments: u64) {
    if !canonical::native(abi) {
        return;
    }
    let _restore = crate::fp_env::tests::RestoreHost::new();
    check_host().unwrap();
    let mut owner = JITModule::new(JITBuilder::new(default_libcall_names()).unwrap());
    let id = owner
        .declare_function(
            "compiled_boundary_proof",
            Linkage::Local,
            &owner.make_signature(),
        )
        .unwrap();
    // Patches above are relative within one test-owned allocation, so they
    // can be applied BEFORE finalization makes this memory executable.
    owner.define_function_bytes(id, 16, bytes, &[]).unwrap();
    owner.finalize_definitions().unwrap();
    let mut state = A64State::default();
    for (index, value) in state.general_register_storage_mut().iter_mut().enumerate() {
        *value = 100 + index as u64;
    }
    state.general_register_storage_mut()[0] = 0u64.wrapping_sub(increments);
    for (index, value) in state.vector_register_storage_mut().iter_mut().enumerate() {
        *value = (0x123456789abcdef0u128 << 64) | (index as u128 + 123);
    }
    state.vector_register_storage_mut()[0] =
        (0x123456789abcdef0u128 << 64) | u128::from(1.0f64.to_bits());
    state.set_nzcv(Nzcv::from_bits(0xb0000000));
    state.set_fpsr(1 << 27);
    state.set_tpidr_el0(0xaabbccdd);
    let mut expected = state.clone();
    for value in &mut expected.general_register_storage_mut()[..usize::from(count)] {
        *value = value.wrapping_add(increments);
    }
    expected.vector_register_storage_mut()[0] =
        (0x123456789abcdef0u128 << 64) | u128::from(f64::INFINITY.to_bits());
    expected.set_nzcv(Nzcv::from_bits(Nzcv::Z));
    expected.set_fpsr((1 << 27) | 2);
    expected.set_pc(0x12345678);
    {
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(17, 23).unwrap());
        unsafe { frame.begin_fp() };
        // Test-only invocation protection, not Task 2's publication protocol.
        frame.execution_epoch = 7;
        let address = unsafe { owner.get_finalized_function(id).add(address_offset) };
        let result = unsafe {
            frame.ensure_fp().unwrap();
            enter_protected(&mut frame, std::ptr::dangling_mut(), address)
        }
        .unwrap();
        assert_eq!(result.reason, NativeExitReason::Dispatch);
        assert_eq!((frame.exit_source_version, frame.exit_state_map), (2, 20));
        assert_eq!(
            (frame.budget.sample_remaining, frame.budget.slice_remaining),
            (17, 23)
        );
        assert_eq!(frame.execution_epoch, 7);
        assert_eq!(
            (
                frame.host_fp.saved,
                frame.host_fp.active,
                frame.gateway_exit
            ),
            (0, 0, 0)
        );
        let mut caller = HostFpState::default();
        unsafe {
            caller.begin();
            caller.finish();
        }
        assert_eq!(
            (caller.saved_control, caller.saved_status),
            (frame.host_fp.saved_control, frame.host_fp.saved_status)
        );
        frame.execution_epoch = 0;
        assert_eq!(frame.execution_epoch, 0);
    }
    assert_eq!(state, expected);
    unsafe { owner.free_memory() };
}

#[test]
fn final_allocations_drive_empty_and_cyclic_links_through_nixe_gateway() {
    let operands = operands(3);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for source_allocator in ["single_pass", "backtracking"] {
            for target_allocator in ["single_pass", "backtracking"] {
                for cycle in [false, true] {
                    let a = compile(abi, source_allocator, fragment(&operands, None, true));
                    let a_entry = entry(abi, &boundary(abi, &a, 10), &operands);
                    let a_map = boundary(abi, &a, 20);
                    let a_exit = exit(abi, &a_map, &operands, 1, false);
                    let mut locations: Vec<_> = a_map
                        .map
                        .values
                        .iter()
                        .map(|value| value.location)
                        .collect();
                    if cycle {
                        locations.swap(0, 2);
                        locations.swap(2, 4);
                        locations.swap(1, 3);
                        locations.swap(3, 5);
                    }
                    let b = compile(
                        abi,
                        target_allocator,
                        fragment(&operands, Some(&locations), false),
                    );
                    let b_entry = entry(abi, &boundary(abi, &b, 10), &operands);
                    let b_map = boundary(abi, &b, 20);
                    let b_exit = exit(abi, &b_map, &operands, 2, true);
                    let transfer = emit_fast_transfer(&a_exit, &b_entry).unwrap();
                    assert_eq!(transfer.is_empty(), !cycle);
                    let mut bytes = a.code_buffer().to_vec();
                    let b_offset = append(&mut bytes, b.code_buffer());
                    let exit_offset = append(
                        &mut bytes,
                        &emit_canonical_exit(
                            &b_exit,
                            ValueLocation::Constant(0x12345678),
                            NativeExitReason::Dispatch,
                        )
                        .unwrap(),
                    );
                    b_map
                        .map
                        .patch_exit(&mut bytes[b_offset..], b_offset as u64, exit_offset as u64)
                        .unwrap();
                    let destination = b_offset + boundary(abi, &b, 10).map.offset as usize;
                    let target = if cycle {
                        let mut bridge = transfer;
                        let jump = reserve_branch(abi, &mut bridge);
                        let offset = append(&mut bytes, &bridge);
                        branch(abi, &mut bytes, offset + jump, destination);
                        offset
                    } else {
                        destination
                    };
                    a_map.map.patch_exit(&mut bytes, 0, target as u64).unwrap();
                    let address_offset = canonical_ingress(
                        abi,
                        &mut bytes,
                        &a_entry,
                        boundary(abi, &a, 10).map.offset as usize,
                    );
                    execute(abi, &bytes, address_offset, 3, 2);
                }
            }
        }
    }
}

#[test]
fn allocation_chosen_entry_and_exit_spills_use_nixe_canonical_adapters() {
    let operands = operands(31);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for allocator in ["single_pass", "backtracking"] {
            let code = compile(abi, allocator, fragment(&operands, None, true));
            let input = boundary(abi, &code, 10);
            let output = boundary(abi, &code, 20);
            for map in [&input, &output] {
                assert!(
                    map.map
                        .values
                        .iter()
                        .any(|value| matches!(value.location, Location::Spill { .. }))
                );
            }
            let contract = entry(abi, &input, &operands);
            let mut state = exit(abi, &output, &operands, 2, true);
            // Require physical reconstruction of EVERY live vector here;
            // leaving the unchanged vectors canonical would hide bad spills.
            state.dirty_live.vector = state.live.vector;
            let mut bytes = code.code_buffer().to_vec();
            let exit_offset = append(
                &mut bytes,
                &emit_canonical_exit(
                    &state,
                    ValueLocation::Constant(0x12345678),
                    NativeExitReason::Dispatch,
                )
                .unwrap(),
            );
            output
                .map
                .patch_exit(&mut bytes, 0, exit_offset as u64)
                .unwrap();
            let start = canonical_ingress(abi, &mut bytes, &contract, input.map.offset as usize);
            execute(abi, &bytes, start, 31, 1);
        }
    }
}

#[test]
fn instruction_attached_prefault_maps_translate_real_registers_and_spills() {
    let operands = operands(31);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for allocator in ["single_pass", "backtracking"] {
            let mut function = fragment(&operands, None, false);
            let block = function.layout.entry_block().unwrap();
            let entry = function.layout.first_inst(block).unwrap();
            let values = function.dfg.inst_results(entry).to_vec();
            // The fault must describe PRE-instruction SSA values, even though
            // the rest of this unit later increments all architectural GPRs.
            let mut cursor = FuncCursor::new(&mut function).at_inst(entry);
            cursor.next_inst();
            cursor.ins().nixe_fault_start(30, &values);
            let address = cursor.ins().get_pinned_reg(types::I64);
            let loaded = cursor
                .ins()
                .load(types::I64, MemFlagsData::new(), address, 0);
            cursor.ins().store(MemFlagsData::new(), loaded, address, 8);
            cursor.ins().nixe_fault_end(30, &[]);
            let code = compile(abi, allocator, function);
            assert_eq!(code.buffer.nixe_faults.len(), 2);
            for map in &code.buffer.nixe_faults {
                assert_eq!(map.id, 30);
                assert!(!map.entry);
                assert_eq!(map.patch_bytes, 0);
                assert!(
                    code.buffer
                        .traps()
                        .iter()
                        .any(|trap| trap.offset == map.offset)
                );
                let allocated = AllocatedBoundary::new(abi, &code, map).unwrap();
                let mut state = exit(abi, &allocated, &operands, 2, false);
                state.host_fpsr_pending = false;
                state.dirty_live = StateSet::default();
                state.validate().unwrap();
                assert!(
                    state
                        .bindings
                        .iter()
                        .any(|binding| matches!(binding.location, ValueLocation::Spill { .. }))
                );
            }
        }
    }
}

#[test]
fn backend_mapping_rejects_missing_mistyped_reserved_and_out_of_extent_operands() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let code = compile(abi, "single_pass", fragment(&operands(3), None, false));
        let allocated = boundary(abi, &code, 20);
        assert!(allocated.location(100, types::I64).is_err());
        assert!(allocated.location(0, types::I32).is_err());
        assert!(allocated.bindings(&[(GuestValue::Vector(0), 0)]).is_err());
        let mut bad = allocated.map.clone();
        for location in [
            Location::Unused,
            Location::Register {
                index: abi.reserved().poll,
                vector: false,
            },
            Location::Register {
                index: 0,
                vector: true,
            },
            Location::Spill { offset: 0 },
            Location::Spill {
                offset: SPILL_BYTES - 8,
            },
        ] {
            bad.values[0].location = location;
            assert!(
                AllocatedBoundary::new(abi, &code, &bad).is_err(),
                "{location:?}"
            );
        }
        bad = allocated.map.clone();
        bad.offset = code.code_buffer().len() as u32;
        assert!(AllocatedBoundary::new(abi, &code, &bad).is_err());
        // An optimized-away definition is allowed only on entry, and cannot
        // be consumed as a physical input by a retained contract.
        bad = boundary(abi, &code, 10).map.clone();
        bad.values[0].location = Location::Unused;
        let unused = AllocatedBoundary::new(abi, &code, &bad).unwrap();
        assert!(unused.location(0, types::I64).is_err());
    }
}
