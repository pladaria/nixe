//! Stable textual IR diagnostics.

use core::fmt::Write;

use super::{block::IrBlock, types::IrType, value::Value};

/// Optional source comments included in a textual IR dump.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IrPrintOptions {
    pub raw_encoding_comments: bool,
    pub disassembly_comments: bool,
}

impl Default for IrPrintOptions {
    fn default() -> Self {
        Self {
            raw_encoding_comments: true,
            disassembly_comments: true,
        }
    }
}

/// Prints a block using only stable guest and semantic identities.
///
/// The format deliberately excludes host pointers and hash iteration order, so
/// it is suitable for golden tests and can be pasted directly into bug reports.
#[must_use]
pub fn print_block(block: &IrBlock, options: IrPrintOptions) -> String {
    let mut output = String::new();
    writeln!(
        output,
        "block {} {} {} bytes={} instructions={}",
        block.metadata.start.pc,
        block.metadata.start.execution_state,
        block.metadata.start.profile_id,
        block.metadata.guest_byte_count,
        block.metadata.guest_instruction_count
    )
    .expect("writing to a String cannot fail");

    for source in &block.metadata.sources {
        write!(output, "  source {}", source.location.pc).expect("writing to a String cannot fail");
        if options.raw_encoding_comments {
            write!(output, " ; raw={}", source.encoding).expect("writing to a String cannot fail");
        }
        if options.disassembly_comments
            && let Some(disassembly) = &source.disassembly
        {
            write!(output, " ; guest={disassembly:?}").expect("writing to a String cannot fail");
        }
        output.push('\n');
    }
    for dependency in &block.metadata.code_dependencies {
        writeln!(
            output,
            "  dependency {} {}",
            dependency.page, dependency.generation
        )
        .expect("writing to a String cannot fail");
    }
    for (index, operation) in block.operations.iter().enumerate() {
        output.push_str("  ");
        write_results(
            &mut output,
            operation.results.iter().collect::<Vec<_>>().as_slice(),
        );
        writeln!(
            output,
            "op{index} {:?} effects={:?} source={}",
            operation.kind, operation.effects, operation.source.pc
        )
        .expect("writing to a String cannot fail");
    }
    writeln!(output, "  terminator {:?}", block.terminator)
        .expect("writing to a String cannot fail");
    for exit in &block.metadata.exits {
        writeln!(output, "  exit {:?} target={:?}", exit.kind, exit.target)
            .expect("writing to a String cannot fail");
    }
    output.push_str("end\n");
    output
}

fn write_results(output: &mut String, results: &[Value]) {
    if results.is_empty() {
        return;
    }
    for (index, result) in results.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        write!(output, "%{}:{}", result.id.index(), type_name(result.ty))
            .expect("writing to a String cannot fail");
    }
    output.push_str(" = ");
}

const fn type_name(ty: IrType) -> &'static str {
    match ty {
        IrType::I1 => "i1",
        IrType::I8 => "i8",
        IrType::I16 => "i16",
        IrType::I32 => "i32",
        IrType::I64 => "i64",
        IrType::I128 => "i128",
        IrType::F16 => "f16",
        IrType::F32 => "f32",
        IrType::F64 => "f64",
        IrType::V64 => "v64",
        IrType::V128 => "v128",
        IrType::Address => "address",
        IrType::Flags => "flags",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        address::{CodeGeneration, GuestPhysicalPageId, GuestVirtualAddress},
        ir::{
            block::{BlockExit, BlockExitKind, BlockMetadata, InstructionSource},
            op::{IrOperation, OperationKind, OperationResults},
            terminator::{ControlTarget, Terminator},
            value::{Immediate, Value, ValueId},
        },
        location::{ExecutionState, InstructionEncoding, LocationDescriptor},
        memory::{CodeDependencies, CodePageDependency},
        profile::CpuProfileId,
    };

    fn block() -> IrBlock {
        let location = LocationDescriptor::new(
            GuestVirtualAddress::new(0x1000),
            ExecutionState::A64,
            CpuProfileId::new(1),
        );
        let dependency = CodePageDependency {
            page: GuestPhysicalPageId::new(2),
            generation: CodeGeneration::new(3),
        };
        IrBlock::new(
            BlockMetadata::new(
                location,
                4,
                1,
                vec![BlockExit {
                    kind: BlockExitKind::Direct,
                    target: Some(GuestVirtualAddress::new(0x1004)),
                }],
                vec![dependency],
                vec![
                    InstructionSource::new(
                        location,
                        InstructionEncoding::from_u32(0xd503_201f),
                        CodeDependencies::one(dependency),
                    )
                    .with_disassembly("nop"),
                ],
            ),
            vec![IrOperation::new(
                location,
                OperationResults::one(Value::new(ValueId::new(0), IrType::I64)),
                OperationKind::Constant(Immediate::I64(7)),
            )],
            Terminator::Direct {
                target: ControlTarget::Direct {
                    pc: GuestVirtualAddress::new(0x1004),
                    execution_state: ExecutionState::A64,
                },
            },
        )
    }

    #[test]
    fn printer_is_deterministic_and_has_a_golden_format() {
        let actual = print_block(&block(), IrPrintOptions::default());
        let expected = concat!(
            "block 0x0000000000001000 A64 profile=0x0000000000000001 bytes=4 instructions=1\n",
            "  source 0x0000000000001000 ; raw=0xd503201f ; guest=\"nop\"\n",
            "  dependency page=0x0000000000000002 generation=0x0000000000000003\n",
            "  %0:i64 = op0 Constant(I64(7)) effects=OperationEffects { side_effects: EffectSet(0), may_fault: false } source=0x0000000000001000\n",
            "  terminator Direct { target: Direct { pc: GuestVirtualAddress(4100), execution_state: A64 } }\n",
            "  exit Direct target=Some(GuestVirtualAddress(4100))\n",
            "end\n",
        );
        assert_eq!(actual, expected);
        assert_eq!(actual, print_block(&block(), IrPrintOptions::default()));
    }

    #[test]
    fn raw_encoding_and_disassembly_comments_are_optional() {
        let output = print_block(
            &block(),
            IrPrintOptions {
                raw_encoding_comments: false,
                disassembly_comments: false,
            },
        );
        assert!(output.contains("  source 0x0000000000001000\n"));
        assert!(!output.contains("raw="));
        assert!(!output.contains("guest="));
    }
}
