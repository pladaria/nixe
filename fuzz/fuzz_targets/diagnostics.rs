#![no_main]

use libfuzzer_sys::fuzz_target;
use nixe_cpu::{
    address::GuestVirtualAddress,
    coverage::{
        CoverageId, MAX_MISSING_INSTRUCTION_EXPORT_BYTES, MAX_MISSING_INSTRUCTION_RECORDS,
        MAX_SURROUNDING_INSTRUCTION_BYTES, MissingInstructionObservation,
        MissingInstructionTracker, ModuleIdentity,
    },
    location::{ExecutionState, InstructionEncoding},
};

fuzz_target!(|data: &[u8]| {
    if data.len() > 64 * 1024 {
        return;
    }

    let mut tracker = MissingInstructionTracker::new();
    for (index, chunk) in data.chunks(16).enumerate() {
        let mut bytes = [0_u8; 16];
        bytes[..chunk.len()].copy_from_slice(chunk);
        let encoding = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let pc = u64::from_le_bytes(bytes[4..12].try_into().unwrap());
        let context_length = usize::from(bytes[12]) % (MAX_SURROUNDING_INSTRUCTION_BYTES + 2);
        let context = vec![bytes[13]; context_length];
        let observation = MissingInstructionObservation::new(
            CoverageId::new(index as u32),
            InstructionEncoding::from_u32(encoding),
            GuestVirtualAddress::new(pc),
            ModuleIdentity::new(u64::from(bytes[14])),
            match bytes[15] % 3 {
                0 => ExecutionState::A64,
                1 => ExecutionState::A32,
                _ => ExecutionState::T32,
            },
            context,
        );
        match observation {
            Ok(observation) => {
                let _ = tracker.record(observation);
            }
            Err(error) => assert!(error.supplied > error.maximum),
        }
    }

    assert!(tracker.unique_instructions() <= MAX_MISSING_INSTRUCTION_RECORDS);
    assert!(tracker.export_sanitized().len() <= MAX_MISSING_INSTRUCTION_EXPORT_BYTES);
    assert!(tracker.export_detailed().len() <= MAX_MISSING_INSTRUCTION_EXPORT_BYTES);
});
