use crate::{
    CompileError, Environment, HostTranslateInfo, Result, RuntimeInfo,
    artifact::VaryingState,
    diagnostics::FailureCategory,
    ir::{Attribute, Ir, Opcode as O, Program, SyntaxNode, Value},
    types::{AttributeType, Interpolation, PixelImap, Stage},
};
use std::collections::{BTreeMap, VecDeque};

pub fn collect_interpolation(env: &dyn Environment, program: &mut Program) -> Result<()> {
    if program.stage != Stage::Fragment {
        return Ok(());
    }
    for index in 0..32 {
        let mut interpolation = None;
        for mode in env.metadata().sph.generic_input_map(index) {
            if mode == PixelImap::Unused {
                continue;
            }
            if interpolation.is_some_and(|old| old != mode) {
                return Err(CompileError::new(
                    FailureCategory::Unsupported,
                    "Per-component interpolation is unsupported",
                ));
            }
            interpolation = Some(mode);
        }
        if let Some(mode) = interpolation {
            program.info.interpolation[index] = match mode {
                PixelImap::Unused | PixelImap::Perspective => Interpolation::Smooth,
                PixelImap::Constant => Interpolation::Flat,
                PixelImap::ScreenLinear => Interpolation::NoPerspective,
            };
        }
    }
    Ok(())
}
pub fn layer(ir: &mut Ir, program: &mut Program, host: &HostTranslateInfo) -> Result<()> {
    if host.support_viewport_index_layer
        || !matches!(
            program.stage,
            Stage::VertexA | Stage::VertexB | Stage::TessellationControl | Stage::TessellationEval
        )
    {
        return Ok(());
    }
    let attribute = (0..32)
        .find(|&i| !program.info.stores.generic(i))
        .map(|i| Attribute::generic(i, 0).unwrap())
        .unwrap_or(Attribute::Layer);
    let mut changed = false;
    for &block in &program.post_order {
        for id in ir.block(block)?.instructions().to_vec() {
            if ir.inst(id)?.opcode() == O::SetAttribute
                && ir.inst(id)?.arg(0)? == Value::Attribute(Attribute::Layer)
            {
                ir.set_arg(id, 0, Value::Attribute(attribute))?;
                changed = true;
            }
        }
    }
    if changed {
        program.info.requires_layer_emulation = true;
        program.info.emulated_layer = attribute;
        program.info.stores.set(Attribute::Layer, false);
        program.info.stores.set(attribute, true);
    }
    Ok(())
}
pub fn tighten_output_vertices(ir: &Ir, program: &mut Program) -> Result<()> {
    if program.stage != Stage::Geometry
        || program.execution.is_geometry_passthrough
        || program
            .syntax
            .iter()
            .any(|s| matches!(s, SyntaxNode::Loop { .. }))
    {
        return Ok(());
    }
    let mut emits = 0u32;
    for &block in &program.blocks {
        for &id in ir.block(block)?.instructions() {
            if ir.inst(id)?.opcode() == O::EmitVertex {
                emits = emits.wrapping_add(1);
            }
        }
    }
    if emits != 0 {
        program.execution.output_vertices = program.execution.output_vertices.min(emits);
    }
    Ok(())
}
fn mapping(
    state: &VaryingState,
    mut unused: VecDeque<Attribute>,
    previous: &BTreeMap<Attribute, Attribute>,
) -> Result<BTreeMap<Attribute, Attribute>> {
    let mut mapping = BTreeMap::new();
    let colors = (0..4).map(|i| {
        (
            Attribute::from_index((Attribute::ColorFrontDiffuseR.index() + i * 4) as u16).unwrap(),
            4,
        )
    });
    let fog = std::iter::once((Attribute::FogCoordinate, 1));
    let tex = (0..10).map(|i| {
        (
            Attribute::from_index((Attribute::FixedFncTexture0S.index() + i * 4) as u16).unwrap(),
            4,
        )
    });
    for (base, count) in colors.chain(fog).chain(tex) {
        if !(if count == 1 {
            state.get(base)
        } else {
            state.any_component(base)
        }) {
            continue;
        }
        let allocated = if previous.contains_key(&base) {
            None
        } else {
            Some(unused.pop_front().ok_or_else(|| {
                CompileError::new(
                    FailureCategory::UnsupportedReferenceCase,
                    "No free generic attribute for legacy interface",
                )
            })?)
        };
        for component in 0..count {
            let source = Attribute::from_index((base.index() + component) as u16).unwrap();
            let dest = if let Some(allocated) = allocated {
                Attribute::from_index((allocated.index() + component) as u16).unwrap()
            } else {
                *previous.get(&source).ok_or_else(|| {
                    CompileError::new(
                        FailureCategory::InvalidArgument,
                        "Incomplete previous-stage legacy mapping",
                    )
                })?
            };
            mapping.insert(source, dest);
        }
    }
    Ok(mapping)
}
pub fn convert_legacy_to_generic(
    ir: &mut Ir,
    program: &mut Program,
    runtime: &RuntimeInfo,
) -> Result<()> {
    for output in [true, false] {
        let state = if output {
            &program.info.stores
        } else {
            &program.info.loads
        };
        if !state.legacy() {
            continue;
        }
        let unused = (0..32)
            .filter(|&i| {
                if output {
                    !state.generic(i)
                } else {
                    !runtime.previous_stage_stores.generic(i)
                        || !state.generic(i)
                        || runtime.generic_input_types[i] == AttributeType::Disabled
                }
            })
            .map(|i| Attribute::generic(i, 0).unwrap())
            .collect();
        let empty = BTreeMap::new();
        let mapping = mapping(
            state,
            unused,
            if output {
                &empty
            } else {
                &runtime.previous_stage_legacy_stores_mapping
            },
        )?;
        if output {
            program.info.legacy_stores_mapping = mapping.clone();
        }
        for &block in &program.post_order {
            for id in ir.block(block)?.instructions().to_vec() {
                if ir.inst(id)?.opcode()
                    != if output {
                        O::SetAttribute
                    } else {
                        O::GetAttribute
                    }
                {
                    continue;
                }
                let Value::Attribute(attribute) = ir.inst(id)?.arg(0)? else {
                    continue;
                };
                if !attribute.is_legacy() {
                    continue;
                }
                let mapped = *mapping.get(&attribute).ok_or_else(|| {
                    CompileError::new(
                        FailureCategory::Internal,
                        "Legacy attribute is missing from interface metadata",
                    )
                })?;
                if output {
                    program.info.stores.set(mapped, true);
                } else {
                    program.info.loads.set(mapped, true);
                }
                ir.set_arg(id, 0, Value::Attribute(mapped))?;
            }
        }
    }
    Ok(())
}
fn invalidate_first(ir: &mut Ir, program: &Program, opcode: O) -> Result<()> {
    for &block in &program.blocks {
        for id in ir.block(block)?.instructions().to_vec() {
            if ir.inst(id)?.opcode() == opcode {
                return ir.invalidate(id);
            }
        }
    }
    Ok(())
}
pub fn merge_dual_vertex<E: Environment>(
    ir: &mut Ir,
    a: Program,
    b: Program,
    env_b: &E,
    debug: bool,
) -> Result<Program> {
    invalidate_first(ir, &a, O::Epilogue)?;
    invalidate_first(ir, &b, O::Prologue)?;
    let mut result = Program::new(Stage::VertexB);
    result.execution.provenance = crate::artifact::StageProvenance::MergedVertex;
    result.syntax = a
        .syntax
        .into_iter()
        .filter(|s| !matches!(s, SyntaxNode::Return))
        .chain(b.syntax)
        .collect();
    result.blocks = result
        .syntax
        .iter()
        .filter_map(|s| {
            if let SyntaxNode::Block(b) = s {
                Some(*b)
            } else {
                None
            }
        })
        .collect();
    result.post_order = b.post_order.into_iter().chain(a.post_order).collect();
    result.info = a.info;
    result.execution.local_memory_size = a
        .execution
        .local_memory_size
        .max(b.execution.local_memory_size);
    for i in 0..8 {
        result.info.loads.mask[i] |= b.info.loads.mask[i];
        result.info.stores.mask[i] |= b.info.stores.mask[i];
    }
    super::join_texture_info(&mut result.info, &b.info);
    super::join_storage_info(&mut result.info, &b.info)?;
    super::dead_code_elimination(ir, &result)?;
    if debug {
        super::verify_program(ir, &result)?;
    }
    super::collect_shader_info(env_b, ir, &mut result)?;
    Ok(result)
}
