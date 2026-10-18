mod cleanup;
mod fold;
mod fold_patterns;
mod info;
mod info_usage;

pub use info::collect_shader_info;
mod lowering;
mod lowering_tables;
mod ssa;
mod storage;
mod storage_table;

pub use storage::{add_nvn_storage_buffers, global_memory_to_storage, join_storage_info};

pub use fold::constant_propagation;

pub use cleanup::{dead_code_elimination, position, remove_identities, verify_program};

pub use lowering::{lower_fp16, lower_fp64, lower_int64};
pub use ssa::rewrite_ssa;

use crate::{
    CompileError, HostTranslateInfo, Result,
    diagnostics::FailureCategory,
    ir::{Ir, Opcode, Program, SyntaxNode},
};

pub fn conditional_barriers(ir: &mut Ir, program: &Program) -> Result<()> {
    let mut depth = 0i32;
    let mut conditional_returns = 0u32;
    for node in &program.syntax {
        match node {
            SyntaxNode::If { .. } | SyntaxNode::Loop { .. } => depth += 1,
            SyntaxNode::EndIf { .. } | SyntaxNode::Repeat { .. } => depth -= 1,
            SyntaxNode::Return | SyntaxNode::Unreachable if depth > 0 => conditional_returns += 1,
            SyntaxNode::Block(block) if depth > 0 || conditional_returns > 0 => {
                let mut index = 0;
                while let Some(&id) = ir.block(*block)?.instructions().get(index) {
                    if ir.inst(id)?.opcode() == Opcode::Barrier {
                        ir.replace_opcode(id, Opcode::Identity)?;
                    }
                    index += 1;
                }
            }
            _ => (),
        }
    }
    if depth != 0 {
        return Err(CompileError::new(
            FailureCategory::Internal,
            "Unbalanced conditional barrier traversal",
        ));
    }
    Ok(())
}

pub fn early_passes(ir: &mut Ir, program: &Program, host: &HostTranslateInfo) -> Result<()> {
    if !host.support_float64 {
        lower_fp64(ir, program)?;
    }
    if !host.support_float16 {
        lower_fp16(ir, program)?;
    }
    if !host.support_int64 {
        lower_int64(ir, program)?;
    }
    if !host.support_conditional_barrier {
        conditional_barriers(ir, program)?;
    }
    rewrite_ssa(ir, program)
}

mod texture;
mod texture_descriptors;
mod texture_table;
mod texture_tracking;
pub use texture::texture_pass;
pub use texture_descriptors::join_texture_info;

mod stage;
pub use stage::{
    collect_interpolation, convert_legacy_to_generic, layer, merge_dual_vertex,
    tighten_output_vertices,
};
mod geometry;
pub use geometry::{
    configure_stage, generate_geometry_passthrough, geometry_compaction,
    has_broken_shuffle_pattern, lower_geometry_passthrough, topology_vertices,
};
mod rescaling;
mod vendor;
pub use rescaling::rescaling;
pub use vendor::vendor_workaround;
mod runtime;
pub use runtime::{GraphicsRuntimeState, make_runtime_info};
