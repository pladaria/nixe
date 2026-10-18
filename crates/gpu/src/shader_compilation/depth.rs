//! Window-depth legalization for hosts with viewport depth limited to [0, 1].
use crate::FragmentDepthTransform;
use nixe_shader_recompiler::{
    Compiler, PreparedStage, Result, Stage,
    ir::{Attribute, Emitter, FpControl, Opcode, Value},
};

pub fn specialize(
    compiler: &mut Compiler,
    stage: &PreparedStage,
    transform: FragmentDepthTransform,
) -> Result<()> {
    if stage.stage != Stage::Fragment {
        return Ok(());
    }
    let [scale, offset, minimum, maximum] = transform.coefficients();
    compiler.transform_prepared(stage, |ir, program| {
        for &block in &program.blocks {
            let instructions = ir.block(block)?.instructions().to_vec();
            for id in instructions {
                let instruction = ir.inst(id)?;
                match instruction.opcode() {
                    Opcode::GetAttribute
                        if instruction.arg(0)? == Value::Attribute(Attribute::PositionZ) =>
                    {
                        let vertex = instruction.arg(1)?;
                        let mut emitter = Emitter::before(ir, block, id)?;
                        let raw = emitter.get_attribute_at_vertex(Attribute::PositionZ, vertex)?;
                        let value = window_depth(&mut emitter, raw, scale, offset)?;
                        emitter.ir.replace_uses_with(id, value)?;
                    }
                    Opcode::SetFragDepth => {
                        let value = instruction.arg(0)?;
                        let mut emitter = Emitter::before(ir, block, id)?;
                        let value = clamp(&mut emitter, value, minimum, maximum)?;
                        emitter.ir.set_arg(id, 0, value)?;
                    }
                    _ => (),
                }
            }
        }
        // Later guest exports replace this initial value. Position.z reads keep
        // the unclamped window value; only attachment exports are clamped.
        let entry = program.blocks[0];
        let first = ir
            .block(entry)?
            .instructions()
            .iter()
            .take_while(|&&id| ir.inst(id).is_ok_and(|i| i.opcode() == Opcode::Phi))
            .count();
        let mut emitter = Emitter::at_position(ir, entry, first)?;
        let raw = emitter.get_attribute(Attribute::PositionZ)?;
        let value = window_depth(&mut emitter, raw, scale, offset)?;
        let value = clamp(&mut emitter, value, minimum, maximum)?;
        emitter.set_frag_depth(value)?;
        program.info.loads.set(Attribute::PositionZ, true);
        program.info.stores_frag_depth = true;
        #[cfg(debug_assertions)]
        nixe_shader_recompiler::passes::verify_program(ir, program)?;
        Ok(())
    })
}

fn window_depth(e: &mut Emitter<'_>, value: Value, scale: f32, offset: f32) -> Result<Value> {
    let control = FpControl {
        no_contraction: true,
        ..Default::default()
    };
    let value = e.fp_mul(value, Value::f32(scale), control)?;
    e.fp_add(value, Value::f32(offset), control)
}

fn clamp(e: &mut Emitter<'_>, value: Value, minimum: f32, maximum: f32) -> Result<Value> {
    let minimum = Value::f32(minimum);
    let maximum = Value::f32(maximum);
    let below = e.fp_less_than(value, minimum, FpControl::default(), true)?;
    let value = e.select(below, minimum, value)?;
    let above = e.fp_greater_than(value, maximum, FpControl::default(), true)?;
    e.select(above, maximum, value)
}
