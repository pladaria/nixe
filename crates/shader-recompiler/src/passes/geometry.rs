use super::fold::{arg, node};
use crate::{
    HostTranslateInfo, Result,
    arena::InstId,
    artifact::VaryingState,
    ir::{Attribute, Emitter, Ir, Opcode as O, Program, Reg, SyntaxNode, Value},
    types::{OutputTopology, Stage},
};

pub fn topology_vertices(topology: OutputTopology) -> u32 {
    match topology {
        OutputTopology::PointList => 1,
        OutputTopology::LineStrip => 2,
        _ => 3,
    }
}
fn emit_passthrough(
    e: &mut Emitter<'_>,
    vertices: u32,
    mask: &VaryingState,
    position: bool,
    layer: Option<Attribute>,
    viewport_reg: Option<Reg>,
) -> Result<()> {
    let mut culled = Value::U1(false);
    let mut viewport = Value::U32(0);
    if let Some(reg) = viewport_reg {
        let mask = e.get_reg(reg)?;
        let neg = e.i_neg(mask)?;
        let low = e.bitwise_and(mask, neg)?;
        culled = e.emit(O::IEqual, &[mask, Value::U32(0)], 0)?;
        let msb = e.find_u_msb(low)?;
        viewport = e.select(culled, Value::U32(0), msb)?;
    }
    for vertex in 0..vertices {
        for index in 0..32 {
            if !mask.generic(index) {
                continue;
            }
            for component in 0..4 {
                let attribute = Attribute::generic(index, component).unwrap();
                let value = e.get_attribute_at_vertex(attribute, Value::U32(vertex))?;
                e.set_attribute(attribute, value, Value::U32(0))?;
            }
        }
        if position {
            for (component, clipped) in [2.0, 2.0, 2.0, 1.0].into_iter().enumerate() {
                let attribute =
                    Attribute::from_index((Attribute::PositionX.index() + component) as u16)
                        .unwrap();
                let mut value = e.get_attribute_at_vertex(attribute, Value::U32(vertex))?;
                if viewport_reg.is_some() {
                    value = e.select(culled, Value::f32(clipped), value)?;
                }
                e.set_attribute(attribute, value, Value::U32(0))?;
            }
        }
        if viewport_reg.is_some() {
            let value = e.emit(O::BitCastF32U32, &[viewport], 0)?;
            e.set_attribute(Attribute::ViewportIndex, value, Value::U32(0))?;
        }
        if let Some(attribute) = layer {
            let value = e.get_attribute(attribute)?;
            e.set_attribute(Attribute::Layer, value, Value::U32(0))?;
        }
        e.emit(O::EmitVertex, &[Value::U32(0)], 0)?;
    }
    e.emit(O::EndPrimitive, &[Value::U32(0)], 0)?;
    Ok(())
}
fn lower_viewport_mask(ir: &mut Ir, program: &Program) -> Result<Option<Reg>> {
    let mut used = [false; 256];
    for &block in &program.blocks {
        for &id in ir.block(block)?.instructions() {
            if matches!(ir.inst(id)?.opcode(), O::GetRegister | O::SetRegister)
                && let Value::Reg(reg) = ir.inst(id)?.arg(0)?
            {
                used[reg.index()] = true;
            }
        }
    }
    let Some(index) = (0..255).rev().find(|&i| !used[i]) else {
        return Ok(None);
    };
    let Some(&entry) = program.blocks.first() else {
        return Ok(None);
    };
    let reg = Reg(index as u8);
    let mut changed = false;
    for &block in &program.blocks {
        for id in ir.block(block)?.instructions().to_vec() {
            if ir.inst(id)?.opcode() != O::SetAttribute
                || ir.inst(id)?.arg(0)? != Value::Attribute(Attribute::ViewportMask)
            {
                continue;
            }
            let value = ir.inst(id)?.arg(1)?;
            let mut e = Emitter::before(ir, block, id)?;
            let value = e.emit(O::BitCastU32F32, &[value], 0)?;
            e.emit(O::SetRegister, &[Value::Reg(reg), value], 0)?;
            e.ir.invalidate(id)?;
            changed = true;
        }
    }
    if !changed {
        return Ok(None);
    }
    Emitter::at_position(ir, entry, 0)?.emit(
        O::SetRegister,
        &[Value::Reg(reg), Value::U32(1)],
        0,
    )?;
    Ok(Some(reg))
}
pub fn lower_geometry_passthrough(
    ir: &mut Ir,
    program: &Program,
    host: &HostTranslateInfo,
) -> Result<()> {
    let viewport = if host.support_viewport_mask {
        None
    } else {
        lower_viewport_mask(ir, program)?
    };
    for &block in &program.blocks {
        for id in ir.block(block)?.instructions().to_vec() {
            if ir.inst(id)?.opcode() == O::Epilogue {
                emit_passthrough(
                    &mut Emitter::before(ir, block, id)?,
                    program.execution.output_vertices,
                    &program.info.passthrough,
                    program.info.passthrough.any_component(Attribute::PositionX),
                    None,
                    viewport,
                )?;
            }
        }
    }
    Ok(())
}
pub fn generate_geometry_passthrough(
    ir: &mut Ir,
    source: &Program,
    topology: OutputTopology,
) -> Result<Program> {
    let mut p = Program::new(Stage::Geometry);
    p.execution.provenance = crate::artifact::StageProvenance::LayerPassthrough;
    p.execution.output_topology = Some(topology);
    p.execution.output_vertices = topology_vertices(topology);
    p.info.loads = source.info.stores.clone();
    p.info.stores = source.info.stores.clone();
    p.info.stores.set(Attribute::Layer, true);
    p.info.stores.set(source.info.emulated_layer, false);
    let block = ir.new_block()?;
    let end = ir.new_block()?;
    emit_passthrough(
        &mut Emitter::new(ir, block)?,
        p.execution.output_vertices,
        &p.info.stores,
        true,
        Some(source.info.emulated_layer),
        None,
    )?;
    Emitter::new(ir, end)?.emit(O::Epilogue, &[], 0)?;
    ir.add_branch(block, end)?;
    p.syntax = vec![
        SyntaxNode::Block(block),
        SyntaxNode::Block(end),
        SyntaxNode::Return,
    ];
    p.blocks = vec![block, end];
    p.post_order = ir.post_order(&p.syntax[0])?;
    super::rewrite_ssa(ir, &p)?;
    Ok(p)
}
pub fn configure_stage(
    ir: &mut Ir,
    p: &mut Program,
    env: &dyn crate::Environment,
    host: &HostTranslateInfo,
) -> Result<()> {
    let m = env.metadata();
    p.execution.local_memory_size = m.local_memory_size;
    match p.stage {
        Stage::TessellationControl => p.execution.invocations = m.sph.threads_per_input_primitive(),
        Stage::Geometry => {
            p.execution.output_topology = Some(m.sph.output_topology()?);
            p.execution.output_vertices = m.sph.max_output_vertices();
            p.execution.invocations = m.sph.threads_per_input_primitive();
            p.execution.is_geometry_passthrough = m.sph.geometry_passthrough() != 0;
            if p.execution.is_geometry_passthrough {
                for i in 0..256 {
                    p.info.passthrough.set(
                        Attribute::from_index(i as u16).unwrap(),
                        m.gp_passthrough_mask[i / 32] & (1 << (i % 32)) == 0,
                    );
                }
                if !host.support_geometry_shader_passthrough {
                    p.execution.output_vertices =
                        topology_vertices(p.execution.output_topology.unwrap());
                    lower_geometry_passthrough(ir, p, host)?;
                }
            }
        }
        Stage::Compute => {
            p.execution.workgroup_size = m.workgroup_size;
            p.execution.shared_memory_size = m.shared_memory_size;
        }
        _ => (),
    }
    Ok(())
}
fn ballot(ir: &Ir, value: Value) -> Result<Option<InstId>> {
    let mut stack = vec![(value, 0)];
    while let Some((value, depth)) = stack.pop() {
        if depth > 8 {
            continue;
        }
        let Value::Inst(id) = ir.resolve(value)? else {
            continue;
        };
        let inst = ir.inst(id)?;
        if inst.opcode() == O::SubgroupBallot {
            return Ok(Some(id));
        }
        if inst.opcode() == O::Phi {
            continue;
        }
        for i in (0..inst.num_args()).rev() {
            stack.push((inst.arg(i)?, depth + 1));
        }
    }
    Ok(None)
}
fn primitive_slot(e: &mut Emitter<'_>, invocations: u32, amount: Value) -> Result<Value> {
    let mut key = e.get_attribute_u32(Attribute::PrimitiveId)?;
    if invocations > 1 {
        key = e.i_mul(key, Value::U32(invocations))?;
        let invocation = e.emit(O::InvocationId, &[], 0)?;
        key = e.i_add(key, invocation)?;
    }
    e.i_mul(key, amount)
}
fn shuffle(op: O) -> bool {
    matches!(
        op,
        O::ShuffleIndex | O::ShuffleUp | O::ShuffleDown | O::ShuffleButterfly
    )
}
pub fn geometry_compaction(ir: &mut Ir, p: &Program, host: &HostTranslateInfo) -> Result<()> {
    if p.stage != Stage::Geometry || !host.single_lane_geometry_subgroups {
        return Ok(());
    }
    let mut atomics = Vec::new();
    for &block in &p.post_order {
        for &id in ir.block(block)?.instructions() {
            if ir.inst(id)?.opcode() == O::StorageAtomicIAdd32
                && let Some(ballot) = ballot(ir, ir.inst(id)?.arg(2)?)?
            {
                atomics.push((block, id, ballot));
            }
        }
    }
    for (block, atomic, ballot) in atomics {
        for successor in ir.block(block)?.successors().to_vec() {
            for phi in ir.block(successor)?.instructions().to_vec() {
                if ir.inst(phi)?.opcode() != O::Phi {
                    continue;
                }
                let mut matches = false;
                for i in 0..ir.inst(phi)?.num_args() {
                    matches |= ir.inst(phi)?.phi_block(i)? == block
                        && arg(ir, phi, i)? == Value::Inst(atomic);
                }
                if !matches {
                    continue;
                }
                for i in 0..ir.inst(phi)?.num_args() {
                    let Value::Inst(input) = arg(ir, phi, i)? else {
                        continue;
                    };
                    if !shuffle(ir.inst(input)?.opcode()) {
                        continue;
                    }
                    let amount = ir.inst(input)?.arg(0)?;
                    let predecessor = ir.inst(phi)?.phi_block(i)?;
                    let value = primitive_slot(
                        &mut Emitter::new(ir, predecessor)?,
                        p.execution.invocations,
                        amount,
                    )?;
                    ir.set_arg(phi, i, value)?;
                }
            }
        }
        let index = ir.inst(atomic)?.arg(0)?;
        let offset = ir.inst(atomic)?.arg(1)?;
        let amount = ir.inst(atomic)?.arg(2)?;
        let mut e = Emitter::before(ir, block, atomic)?;
        let slot = primitive_slot(&mut e, p.execution.invocations, amount)?;
        let end = e.i_add(slot, amount)?;
        e.emit(O::StorageAtomicUMax32, &[index, offset, end], 0)?;
        e.ir.replace_uses_with(atomic, slot)?;
        if let Value::Inst(predicate) = arg(e.ir, ballot, 0)? {
            e.ir.replace_uses_with(predicate, Value::U1(true))?;
        }
    }
    Ok(())
}
pub fn has_broken_shuffle_pattern(ir: &Ir, p: &Program) -> Result<bool> {
    if p.stage != Stage::Fragment {
        return Ok(false);
    }
    for &block in &p.post_order {
        for &id in ir.block(block)?.instructions() {
            if !shuffle(ir.inst(id)?.opcode()) {
                continue;
            }
            if let Some(cast) = node(ir, arg(ir, id, 0)?, O::BitCastU32F32)?
                && let Some(attribute) = node(ir, arg(ir, cast, 0)?, O::GetAttribute)?
                && matches!(
                    arg(ir, attribute, 0)?,
                    Value::Attribute(Attribute::PositionX | Attribute::PositionY)
                )
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
