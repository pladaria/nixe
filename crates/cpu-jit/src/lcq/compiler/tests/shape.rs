use super::*;

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
