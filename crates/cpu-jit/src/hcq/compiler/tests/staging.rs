use super::*;
use crate::abi::ExitSiteKey;

#[test]
fn hcq_staging_owns_complete_entry_exit_fp_and_fault_metadata() {
    // Byte/metadata proof only; faulting memory is not executed here.
    let graph = graph(&[
        (0, &[0x1e222820, ADDS, 0x1400000e]), // FADD; ADDS; B 64
        (64, &[0xa9400c22, 0xd4200000]),      // LDP X2,X3,[X1]; BRK
    ]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let mut previous = None;
        for _ in 0..2 {
            let (mut context, body) = emitted(&graph, &[0, block(&graph, 64)], abi);
            let code = context.take_compiled_code().unwrap();
            let body_bytes = code.code_buffer().len();
            let pending_exits = body.exits.len();
            let activations = body.fp_activations.len();
            assert!(activations > 0);
            let version = CodeVersion::new(7).unwrap();
            let staged = body
                .stage(abi, code, &context.func, &graph, version)
                .unwrap();
            context.clear();
            assert_eq!(staged.entries.len(), 2);
            assert_eq!(staged.faults.len(), 2);
            assert_eq!(
                staged.states.len(),
                pending_exits + activations + staged.faults.len()
            );
            assert_eq!(staged.output.metadata.entries.len(), 2 + activations);
            for (entry, pc) in staged.entries.iter().zip([0, 64]) {
                assert_eq!(entry.key.pc.get(), pc);
                assert!((entry.fast_offset as usize) < body_bytes);
                assert!(entry.canonical_offset as usize >= body_bytes);
                assert!((entry.canonical_offset as usize) < staged.output.bytes.len());
                entry.contract.validate().unwrap();
            }
            for (index, record) in staged.states.iter().enumerate() {
                assert_eq!(
                    record.state.site,
                    ExitSiteKey {
                        source: version,
                        state_map: index as u32
                    }
                );
                record.state.validate().unwrap();
                assert!((record.native_offset as usize) < body_bytes);
                assert_eq!(record.exit.is_some(), index < pending_exits);
                assert_eq!(record.transfer.is_some(), index < pending_exits);
            }
            for (index, fault) in staged.faults.iter().enumerate() {
                assert_eq!(fault.instruction.block_key().pc.get(), 64);
                assert_eq!(
                    fault.state_map as usize,
                    pending_exits + activations + index
                );
                assert_eq!(
                    staged.states[fault.state_map as usize].native_offset,
                    fault.native_start
                );
                assert!(fault.native_start < fault.native_end);
                assert!(fault.native_end as usize <= body_bytes);
            }
            for map in staged
                .output
                .metadata
                .states
                .iter()
                .filter(|map| !map.entry)
            {
                assert!(
                    staged
                        .states
                        .iter()
                        .any(|state| state.native_offset == map.offset)
                );
            }
            assert_eq!(staged.output.metadata.faults.len(), staged.faults.len());
            // Identical captured input and explicit accounting produce identical
            // patched bytes, independent of the backend context's lifetime.
            if let Some(previous) = &previous {
                assert_eq!(previous, &staged.output.bytes);
            }
            previous = Some(staged.output.bytes);
        }
    }
}

#[test]
fn hcq_staging_requires_charged_dispatches() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for word in [0x14000004, RET] {
            let (graph, mut context, body) = exits::terminal(abi, word, false);
            let code = context.take_compiled_code().unwrap();
            let error = body
                .stage(
                    abi,
                    code,
                    &context.func,
                    &graph,
                    CodeVersion::new(1).unwrap(),
                )
                .err()
                .expect("static and dynamic dispatch must have a checkpoint");
            assert!(
                error
                    .to_string()
                    .contains("dispatch exit has no charged checkpoint")
            );
        }
    }
}
