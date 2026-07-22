#![no_main]

use libfuzzer_sys::fuzz_target;
use nixe_cpu::{
    address::{CodeGeneration, GuestPhysicalPageId, GuestVirtualAddress},
    ir::{
        block::{BlockExit, BlockExitKind, BlockMetadata, InstructionSource, IrBlock},
        op::{
            IntegerBinaryKind, IrOperation, OperationEffects, OperationKind, OperationResults,
            ScalarOperation, StateRegister,
        },
        terminator::{ControlTarget, ExceptionKind, Terminator},
        types::IrType,
        value::{Immediate, Operand, Value, ValueId},
        verify::verify_block,
    },
    location::{ExecutionState, InstructionEncoding, LocationDescriptor},
    memory::{CodeDependencies, CodePageDependency},
    profile::GuestCpuProfile,
    state::a64::A64GeneralRegister,
};

const MAX_OPERATIONS: usize = 64;
const MAX_SOURCES: usize = 8;

fuzz_target!(|data: &[u8]| {
    if data.len() > 4_096 {
        return;
    }

    let mut header = [0_u8; 8];
    let header_length = data.len().min(header.len());
    header[..header_length].copy_from_slice(&data[..header_length]);

    let profile = GuestCpuProfile::switch_1();
    let start = LocationDescriptor::new(
        GuestVirtualAddress::new(0x1000),
        ExecutionState::A64,
        profile.id(),
    );
    let dependency = CodePageDependency {
        page: GuestPhysicalPageId::new(1),
        generation: CodeGeneration::new(u64::from(header[0])),
    };
    let source_count = usize::from(header[1]) % (MAX_SOURCES + 1);
    let mut sources = Vec::with_capacity(source_count);
    for index in 0..source_count {
        let location = LocationDescriptor::new(
            GuestVirtualAddress::new(0x1000 + index as u64 * 4),
            ExecutionState::A64,
            profile.id(),
        );
        sources.push(InstructionSource::new(
            location,
            InstructionEncoding::from_u32(
                u32::from_le_bytes(header[0..4].try_into().unwrap()) ^ index as u32,
            ),
            CodeDependencies::one(dependency),
        ));
    }

    let operation_count = data.len().saturating_sub(8).min(MAX_OPERATIONS);
    let mut operations = Vec::with_capacity(operation_count);
    for index in 0..operation_count {
        let byte = data[8 + index];
        let source = LocationDescriptor::new(
            GuestVirtualAddress::new(0x1000 + u64::from(byte % (MAX_SOURCES as u8 + 2)) * 4),
            ExecutionState::A64,
            profile.id(),
        );
        let value = Value::new(ValueId::new(u32::from(byte >> 2)), integer_type(byte));
        let results = if byte & 1 == 0 {
            OperationResults::NONE
        } else {
            OperationResults::one(value)
        };
        let operand = if byte & 2 == 0 {
            Operand::Immediate(integer_immediate(byte))
        } else {
            Operand::Value(value)
        };
        let kind = match byte % 4 {
            0 => OperationKind::Constant(integer_immediate(byte)),
            1 => OperationKind::Scalar(ScalarOperation::Binary {
                kind: IntegerBinaryKind::Add,
                lhs: operand,
                rhs: Operand::Immediate(integer_immediate(byte.wrapping_add(1))),
            }),
            2 => OperationKind::ReadState(StateRegister::A64X(
                A64GeneralRegister::new(byte % 31).unwrap(),
            )),
            _ => OperationKind::WriteState {
                register: StateRegister::A64X(A64GeneralRegister::new(byte % 31).unwrap()),
                value: operand,
            },
        };
        let mut operation = IrOperation::new(source, results, kind);
        if byte & 0x80 != 0 {
            operation.effects = OperationEffects::default();
        }
        operations.push(operation);
    }

    let direct_target = ControlTarget::Direct {
        pc: GuestVirtualAddress::new(0x1000 + source_count as u64 * 4),
        execution_state: ExecutionState::A64,
    };
    let terminator = match header[2] % 4 {
        0 => Terminator::Direct {
            target: direct_target,
        },
        1 => Terminator::Conditional {
            condition: Operand::Immediate(Immediate::I1(header[3] & 1 != 0)),
            taken: direct_target,
            fallthrough: direct_target,
        },
        2 => Terminator::Exception {
            source: start,
            kind: ExceptionKind::UndefinedInstruction,
            syndrome: Some(u64::from(header[3])),
        },
        _ => Terminator::UnsupportedInstruction {
            source: start,
            encoding: InstructionEncoding::from_u32(u32::from(header[3])),
            coverage_id: u32::from(header[4]),
            disassembly: "fuzzed".into(),
            reason: "malformed verifier input".into(),
        },
    };
    let exits = if header[5] & 1 == 0 {
        Vec::new()
    } else {
        vec![BlockExit {
            kind: BlockExitKind::Direct,
            target: Some(GuestVirtualAddress::new(0x1000 + source_count as u64 * 4)),
        }]
    };
    let metadata = BlockMetadata::new(
        start,
        if header[6] & 1 == 0 {
            source_count as u32 * 4
        } else {
            u32::from(header[6])
        },
        if header[7] & 1 == 0 {
            source_count as u32
        } else {
            u32::from(header[7])
        },
        exits,
        if source_count == 0 {
            Vec::new()
        } else {
            vec![dependency]
        },
        sources,
    );
    let block = IrBlock::new(metadata, operations, terminator);
    if let Err(error) = verify_block(&block) {
        assert!(error.to_string().len() <= 4_096);
    }
});

fn integer_type(selector: u8) -> IrType {
    match selector % 5 {
        0 => IrType::I1,
        1 => IrType::I8,
        2 => IrType::I16,
        3 => IrType::I32,
        _ => IrType::I64,
    }
}

fn integer_immediate(selector: u8) -> Immediate {
    match selector % 5 {
        0 => Immediate::I1(selector & 1 != 0),
        1 => Immediate::I8(selector),
        2 => Immediate::I16(u16::from(selector)),
        3 => Immediate::I32(u32::from(selector)),
        _ => Immediate::I64(u64::from(selector)),
    }
}
