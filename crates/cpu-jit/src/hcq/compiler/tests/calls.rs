use super::*;
use crate::abi::{NativeFrame, PollBudget};
use crate::hcq::{
    Builder,
    tests::{key, words},
};
use nixe_cpu::state::a64::{A64State, Nzcv};

fn host() -> HostAbi {
    if cfg!(target_arch = "x86_64") {
        HostAbi::X86_64
    } else {
        HostAbi::Aarch64
    }
}

fn run(
    graph: &Graph,
    entries: &[usize],
    state: &mut A64State,
    slice: i64,
) -> crate::abi::NativeExitReason {
    let (process, owner) = published(graph, entries, host());
    let mut reader = process.register().unwrap();
    let entry = key(state.pc());
    let mut frame = NativeFrame::new(state, PollBudget::new(4096, slice).unwrap());
    let mut invocation = unsafe { reader.admit(&mut frame, entry) }.unwrap().unwrap();
    let address = invocation.payload().preferred().unwrap().canonical.get();
    let result = unsafe {
        crate::native::enter_protected(
            invocation.frame(),
            std::ptr::null_mut(),
            address as *const u8,
        )
    }
    .unwrap();
    drop(invocation);
    drop(owner);
    result.reason
}

fn expected(graph: &Graph, state: &mut A64State, limit: usize) {
    for _ in 0..limit {
        let Some(word) = graph
            .instructions
            .iter()
            .find(|word| word.instruction.key.block_key().pc.get() == state.pc())
        else {
            return;
        };
        if word.instruction.bits == 0xd4200000 {
            return;
        }
        nixe_cpu_interpreter::execute_one(
            &graph.blocks[0].key.platform,
            state,
            word.instruction.bits,
        )
        .unwrap();
    }
}

#[test]
fn nested_calls_and_guarded_returns_execute_in_one_allocation_domain() {
    // MOV X19,LR; BL callee; MOV LR,X19; BRK.
    // callee saves its own LR in X20 before calling a leaf.
    let graph = graph(&[
        (0x1000, &[0xaa1e03f3, 0x94000007, 0xaa1303fe, 0xd4200000]),
        (0x1020, &[0xaa1e03f4, 0x94000007, 0xaa1403fe, RET]),
        (0x1040, &[0x91000800, RET]),
    ]);
    let entries = [0, block(&graph, 0x1020), block(&graph, 0x1040)];
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        staged(&graph, &entries, abi, CodeVersion::new(1).unwrap());
    }
    for (pc, lr) in [
        (0x1000, 0x2000),
        (0x1020, 0x1008),
        (0x1040, 0x1028),
        (0x1040, 0x3000),
    ] {
        let mut state = A64State::default();
        state.set_pc(pc);
        for (index, value) in state.general_register_storage_mut().iter_mut().enumerate() {
            *value = index as u64 + 0x12345600;
        }
        state.general_register_storage_mut()[30] = lr;
        state.set_nzcv(Nzcv::from_bits(0xb0000000));
        state.set_vector(7, u128::MAX - 123);
        let mut reference = state.clone();
        expected(&graph, &mut reference, 20);
        run(&graph, &entries, &mut state, 100);
        assert_eq!(state, reference, "entry={pc:x}, LR={lr:x}");
    }
}

#[test]
fn recursive_calls_guard_both_recursive_and_outer_continuations() {
    let graph = graph(&[
        (
            0x1000,
            &[
                0xaa1e03f4, // MOV X20,LR
                0xd2800020, // MOV X0,#1
                0x94000006, // BL 0x1020
                0xaa1403fe, // MOV LR,X20
                0xd4200000,
            ],
        ),
        (
            0x1020,
            &[
                0xb4000100, // CBZ X0,0x1040
                0xaa1e03f3, // MOV X19,LR
                0xd1000400, // SUB X0,X0,#1
                0x97fffffd, // BL 0x1020
                0xaa1303fe, // MOV LR,X19
                RET,
            ],
        ),
        (0x1040, &[0x91000421, RET]),
    ]);
    let entries = [0, block(&graph, 0x1020)];
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        staged(&graph, &entries, abi, CodeVersion::new(1).unwrap());
    }
    let mut state = A64State::default();
    state.set_pc(0x1000);
    state.general_register_storage_mut()[30] = 0x2000;
    let mut reference = state.clone();
    expected(&graph, &mut reference, 32);
    assert_eq!(
        run(&graph, &entries, &mut state, 100),
        NativeExitReason::Architectural
    );
    assert_eq!(state, reference);
}

#[test]
fn captured_blr_reads_x30_before_writing_the_continuation() {
    let mut builder = Builder::new(key(0x1000));
    builder
        .merge(key(0x1000), words(0x1000, &[0xd63f03c0, 0xd4200000]))
        .unwrap();
    builder
        .merge(key(0x1020), words(0x1020, &[0x91000400, RET]))
        .unwrap();
    builder.observe(key(0x1000), key(0x1020)).unwrap();
    let (instructions, blocks) = builder.finish().unwrap();
    let graph = Graph {
        discovery: None,
        units: Vec::new(),
        inputs: Vec::new(),
        instructions,
        blocks,
    };
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        staged(&graph, &[0], abi, CodeVersion::new(1).unwrap());
    }
    for target in [0x1020, 0x2000] {
        let mut state = A64State::default();
        state.set_pc(0x1000);
        state.general_register_storage_mut()[30] = target;
        let mut reference = state.clone();
        expected(&graph, &mut reference, 10);
        run(&graph, &[0], &mut state, 100);
        assert_eq!(state, reference);
    }
}

#[test]
fn cycles_through_calls_keep_mandatory_budget_checks_and_exact_state() {
    let graph = graph(&[
        (0x1000, &[0x94000004, 0x17ffffff]),
        (0x1010, &[0x91000400, RET]),
    ]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        staged(&graph, &[0], abi, CodeVersion::new(1).unwrap());
    }
    for slice in [4, 32, 4096] {
        let mut state = A64State::default();
        state.set_pc(0x1000);
        let mut reference = state.clone();
        expected(&graph, &mut reference, slice as usize);
        assert_eq!(
            run(&graph, &[0], &mut state, slice),
            NativeExitReason::BudgetExhausted
        );
        assert_eq!(state, reference);
    }
}
