//! Stable textual IR diagnostics.

use core::fmt::Write;

use super::{
    block::{BlockId, IrBlock},
    region::IrRegion,
    types::IrType,
    value::Value,
};

/// Position of an IR dump in the Nixe IR optimization pipeline.
///
/// The frontend currently emits only pre-optimization IR. Keeping the stage in
/// the public diagnostic contract lets future optimization passes publish a
/// comparable post-optimization dump without changing the format or API.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum IrDumpStage {
    PreOptimization,
    PostOptimization,
}

impl core::fmt::Display for IrDumpStage {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(match self {
            Self::PreOptimization => "pre-optimization",
            Self::PostOptimization => "post-optimization",
        })
    }
}

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

/// Prints a region using only stable guest and semantic identities.
///
/// The format deliberately excludes host pointers and hash iteration order, so
/// it is suitable for golden tests and can be pasted directly into bug reports.
#[must_use]
pub fn print_region(region: &IrRegion, options: IrPrintOptions) -> String {
    let mut output = String::new();
    writeln!(
        output,
        "region {} {} {} bytes={} instructions={} operations={} blocks={}",
        region.metadata.start.pc,
        region.metadata.start.execution_state,
        region.metadata.start.profile_id,
        region.metadata.guest_byte_count,
        region.metadata.guest_instruction_count,
        region.metadata.ir_operation_count,
        region.blocks.len()
    )
    .expect("writing to a String cannot fail");

    for dependency in &region.metadata.code_dependencies {
        writeln!(
            output,
            "  dependency {} {} {}",
            dependency.page, dependency.generation, dependency.mapping_generation
        )
        .expect("writing to a String cannot fail");
    }
    for entry in &region.metadata.entries {
        writeln!(
            output,
            "  entry {} -> b{}",
            entry.location,
            entry.block.index()
        )
        .expect("writing to a String cannot fail");
    }
    for safepoint in &region.metadata.safepoints {
        writeln!(
            output,
            "  safepoint {:?} block=b{} target={:?}",
            safepoint.kind,
            safepoint.block.index(),
            safepoint.target.map(BlockId::index)
        )
        .expect("writing to a String cannot fail");
    }
    for (index, block) in region.blocks.iter().enumerate() {
        write_block(&mut output, BlockId::new(index as u32), block, options);
    }
    for exit in &region.metadata.exits {
        writeln!(
            output,
            "  exit b{} {:?} target={:?}",
            exit.block.index(),
            exit.kind,
            exit.target
        )
        .expect("writing to a String cannot fail");
    }
    output.push_str("end-region\n");
    output
}

fn write_block(output: &mut String, id: BlockId, block: &IrBlock, options: IrPrintOptions) {
    writeln!(
        output,
        "block b{} {} {} {} bytes={} instructions={}",
        id.index(),
        block.metadata.start.pc,
        block.metadata.start.execution_state,
        block.metadata.start.profile_id,
        block.metadata.guest_byte_count,
        block.metadata.guest_instruction_count
    )
    .expect("writing to a String cannot fail");

    for source in &block.metadata.sources {
        write!(
            output,
            "  source pc={} state={}",
            source.location.pc, source.location.execution_state
        )
        .expect("writing to a String cannot fail");
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
    writeln!(output, "  end-reason {}", block.metadata.end_reason)
        .expect("writing to a String cannot fail");
    for (index, operation) in block.operations.iter().enumerate() {
        output.push_str("  ");
        write_results(
            output,
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
    output.push_str("end-block\n");
}

/// Prints a stable stage-labelled IR dump.
///
/// Callers pass the region produced at the named stage. No optimization is
/// performed by this function, and printing remains entirely opt-in.
#[must_use]
pub fn print_ir_dump(region: &IrRegion, stage: IrDumpStage, options: IrPrintOptions) -> String {
    let mut output = String::new();
    writeln!(output, "ir-dump stage={stage}").expect("writing to a String cannot fail");
    output.push_str(&print_region(region, options));
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
    use nixe_memory::{
        ContentGeneration, GuestPhysicalPageId, GuestVirtualAddress, MappingGeneration,
    };

    use crate::{
        ir::{
            block::{BlockMetadata, InstructionSource},
            op::{IrOperation, OperationKind, OperationResults},
            region::{
                RegionEntry, RegionExit, RegionExitKind, RegionMetadata, RegionSafepoint,
                RegionSafepointKind,
            },
            terminator::{ControlTarget, Terminator},
            value::{Immediate, Value, ValueId},
        },
        location::{ExecutionState, InstructionEncoding, LocationDescriptor},
        memory::{CodeDependencies, CodePageDependency},
        profile::CpuProfileId,
    };

    fn region() -> IrRegion {
        let location = LocationDescriptor::new(
            GuestVirtualAddress::new(0x1000),
            ExecutionState::A64,
            CpuProfileId::new(1),
        );
        let dependency = CodePageDependency {
            page: GuestPhysicalPageId::new(2),
            generation: ContentGeneration::new(3),
            mapping_generation: MappingGeneration::new(1),
        };
        let target = ControlTarget::Direct {
            pc: GuestVirtualAddress::new(0x1004),
            execution_state: ExecutionState::A64,
        };
        let block = IrBlock::new(
            BlockMetadata::new(
                location,
                4,
                1,
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
            Terminator::Direct { target },
        );
        IrRegion::new(
            RegionMetadata {
                start: location,
                guest_byte_count: 4,
                guest_instruction_count: 1,
                ir_operation_count: 1,
                entries: vec![RegionEntry {
                    location,
                    block: BlockId::new(0),
                }]
                .into_boxed_slice(),
                exits: vec![RegionExit {
                    block: BlockId::new(0),
                    kind: RegionExitKind::Direct,
                    target: Some(target),
                }]
                .into_boxed_slice(),
                code_dependencies: vec![dependency].into_boxed_slice(),
                safepoints: vec![RegionSafepoint {
                    block: BlockId::new(0),
                    target: None,
                    kind: RegionSafepointKind::Entry,
                }]
                .into_boxed_slice(),
            },
            vec![block],
        )
    }

    #[test]
    fn printer_is_deterministic_and_has_a_golden_format() {
        let actual = print_region(&region(), IrPrintOptions::default());
        let expected = concat!(
            "region 0x0000000000001000 A64 profile=0x0000000000000001 bytes=4 instructions=1 operations=1 blocks=1\n",
            "  dependency page=0x0000000000000002 generation=0x0000000000000003 mapping-generation=1\n",
            "  entry pc=0x0000000000001000 state=A64 profile=0x0000000000000001 -> b0\n",
            "  safepoint Entry block=b0 target=None\n",
            "block b0 0x0000000000001000 A64 profile=0x0000000000000001 bytes=4 instructions=1\n",
            "  source pc=0x0000000000001000 state=A64 ; raw=0xd503201f ; guest=\"nop\"\n",
            "  end-reason explicit-terminator\n",
            "  %0:i64 = op0 Constant(I64(7)) effects=OperationEffects { side_effects: EffectSet(0), may_fault: false } source=0x0000000000001000\n",
            "  terminator Direct { target: Direct { pc: GuestVirtualAddress(4100), execution_state: A64 } }\n",
            "end-block\n",
            "  exit b0 Direct target=Some(Direct { pc: GuestVirtualAddress(4100), execution_state: A64 })\n",
            "end-region\n",
        );
        assert_eq!(actual, expected);
        assert_eq!(actual, print_region(&region(), IrPrintOptions::default()));
    }

    #[test]
    fn raw_encoding_and_disassembly_comments_are_optional() {
        let output = print_region(
            &region(),
            IrPrintOptions {
                raw_encoding_comments: false,
                disassembly_comments: false,
            },
        );
        assert!(output.contains("  source pc=0x0000000000001000 state=A64\n"));
        assert!(!output.contains("raw="));
        assert!(!output.contains("guest="));
    }

    #[test]
    fn staged_dump_api_distinguishes_pre_and_post_optimization_ir() {
        let pre = print_ir_dump(
            &region(),
            IrDumpStage::PreOptimization,
            IrPrintOptions::default(),
        );
        let post = print_ir_dump(
            &region(),
            IrDumpStage::PostOptimization,
            IrPrintOptions::default(),
        );
        assert!(pre.starts_with("ir-dump stage=pre-optimization\n"));
        assert!(post.starts_with("ir-dump stage=post-optimization\n"));
        assert_eq!(
            pre.lines().skip(1).collect::<Vec<_>>(),
            post.lines().skip(1).collect::<Vec<_>>()
        );
    }
}
