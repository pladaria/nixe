use super::{EmitContext, Id};
use crate::{
    CompileError, CompilerSettings, Profile, Result, RuntimeInfo, StageArtifact,
    artifact::{Bindings, StageInterface},
    diagnostics::FailureCategory,
    ir::{Ir, Program, SyntaxNode},
};
use spirv::{AddressingModel, MemoryModel, Op, StorageClass};

pub fn emit_spirv(
    ir: &mut Ir,
    program: Program,
    profile: &Profile,
    runtime: &RuntimeInfo,
    settings: &CompilerSettings,
    bindings: &mut Bindings,
) -> Result<StageArtifact> {
    let mut ctx = EmitContext::new(profile, runtime, &program, bindings)?;
    ctx.fix_bloom_effects = settings.fix_bloom_effects;
    ctx.builder
        .memory_model(AddressingModel::Logical, MemoryModel::GLSL450)?;
    let main = ctx.emit_main(ir, &program, settings.disable_shader_loop_safety_checks)?;
    ctx.entry_point(&program, main)?;
    ctx.setup_capabilities(&program, main)?;
    ctx.patch_phis(ir)?;
    let words = ctx.assemble()?.into_boxed_slice();
    let interface = StageInterface {
        inputs: program.info.loads.clone(),
        outputs: program.info.stores.clone(),
        passthrough: program.info.passthrough.clone(),
        interpolation: program.info.interpolation,
        patches: program.info.uses_patches,
        fragment_colors: program.info.stores_frag_color,
        sample_mask: program.info.stores_sample_mask,
        fragment_depth: program.info.stores_frag_depth,
        tess_level_outer: program.info.stores_tess_level_outer,
        tess_level_inner: program.info.stores_tess_level_inner,
        clip_distances: program.info.used_clip_distances,
    };
    Ok(StageArtifact {
        stage: program.stage,
        words,
        info: program.info,
        bindings: ctx.resource_bindings.into_boxed_slice(),
        host_uniforms: ctx.host_uniforms.into_boxed_slice(),
        interface,
        execution: program.execution,
    })
}
impl EmitContext<'_> {
    fn emit_main(&mut self, ir: &mut Ir, p: &Program, disable_safety: bool) -> Result<Id> {
        let ty = self.builder.type_function(self.types.void, &[])?;
        let main = self.builder.begin_function(self.types.void, ty)?;
        for &block in &p.blocks {
            ir.block_mut(block)?.definition = self.builder.reserve_id()?.0;
        }
        let mut current = false;
        for node in &p.syntax {
            let label = |block| -> Result<u32> {
                let id = ir.block(block)?.definition;
                if id == 0 {
                    Err(CompileError::new(
                        FailureCategory::Internal,
                        "Structured branch targets an unallocated SPIR-V label",
                    ))
                } else {
                    Ok(id)
                }
            };
            match *node {
                SyntaxNode::Block(block) => {
                    let target = label(block)?;
                    if current {
                        self.builder.instruction(Op::Branch, &[target])?;
                    }
                    self.builder.label(Id(target))?;
                    let count = ir.block(block)?.instructions().len();
                    for i in 0..count {
                        let id = ir.block(block)?.instructions()[i];
                        self.emit_instruction(ir, id)?;
                    }
                }
                SyntaxNode::If {
                    condition,
                    body,
                    merge,
                } => {
                    let cond = self.def(ir, condition)?;
                    self.builder
                        .instruction(Op::SelectionMerge, &[label(merge)?, 0])?;
                    self.builder.instruction(
                        Op::BranchConditional,
                        &[cond.0, label(body)?, label(merge)?],
                    )?;
                }
                SyntaxNode::Loop {
                    body,
                    continue_block,
                    merge,
                } => {
                    self.builder
                        .instruction(Op::LoopMerge, &[label(merge)?, label(continue_block)?, 0])?;
                    self.builder.instruction(Op::Branch, &[label(body)?])?;
                }
                SyntaxNode::Break {
                    condition,
                    merge,
                    skip,
                } => {
                    let cond = self.def(ir, condition)?;
                    self.builder.instruction(
                        Op::BranchConditional,
                        &[cond.0, label(merge)?, label(skip)?],
                    )?;
                }
                SyntaxNode::EndIf { merge } => {
                    if current {
                        self.builder.instruction(Op::Branch, &[label(merge)?])?;
                    }
                }
                SyntaxNode::Repeat {
                    condition,
                    loop_header,
                    merge,
                } => {
                    let mut cond = self.def(ir, condition)?;
                    if !disable_safety {
                        let initial = self.const_u32(0x2000)?;
                        let counter = self.builder.variable(
                            self.types.private_u32,
                            StorageClass::Private,
                            Some(initial),
                        )?;
                        self.global_interface(counter);
                        let old = self
                            .builder
                            .result(Op::Load, self.types.u32[0], &[counter.0])?;
                        let one = self.const_u32(1)?;
                        let new =
                            self.builder
                                .result(Op::ISub, self.types.u32[0], &[old.0, one.0])?;
                        self.builder.instruction(Op::Store, &[counter.0, new.0])?;
                        let safe = self.builder.result(
                            Op::SGreaterThanEqual,
                            self.types.bool,
                            &[new.0, self.constants.u32_zero.0],
                        )?;
                        cond = self.builder.result(
                            Op::LogicalAnd,
                            self.types.bool,
                            &[cond.0, safe.0],
                        )?;
                    }
                    self.builder.instruction(
                        Op::BranchConditional,
                        &[cond.0, label(loop_header)?, label(merge)?],
                    )?;
                }
                SyntaxNode::Return => self.builder.instruction(Op::Return, &[])?,
                SyntaxNode::Unreachable => self.builder.instruction(Op::Unreachable, &[])?,
            }
            current = matches!(node, SyntaxNode::Block(_));
        }
        self.builder.end_function()?;
        Ok(main)
    }
}
