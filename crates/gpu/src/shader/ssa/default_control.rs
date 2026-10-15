//! Vulkan's required TCS for Maxwell patches without a guest control program.
//! It forwards linked per-vertex data and consumes dynamic programmed levels.
//! https://registry.khronos.org/OpenGL/specs/gl/glspec46.core.pdf (11.2.3)
//! https://docs.vulkan.org/spec/latest/chapters/tessellation.html
use super::*;
use crate::{ShaderScalarType, TessellationControl, TessellationDomain};
use std::collections::BTreeMap;

pub struct DefaultControl {
    pub function: Function,
    pub required_levels: u8,
}
impl DefaultControl {
    pub const PARAMETER_BYTES: u32 = 24;
    pub fn parameters(&self, control: TessellationControl) -> Result<[u32; 6], Error> {
        crate::shader::spirv::default_control_parameters(self.required_levels, control)
            .map_err(|e| Error::at(ShaderSourceLocation::new(0), e.to_string()))
    }
}
/// Parameters 0..5 are outer[0..4], inner[0..2], raw F32 words. Level values
/// never specialize a shader. Only invocation zero writes patch-wide outputs.
pub fn default_control(
    vertex: &Function,
    evaluation: &Function,
    points: u32,
    domain: TessellationDomain,
) -> Result<DefaultControl, Error> {
    let source = ShaderSourceLocation::new(0);
    if vertex.stage != ShaderStage::Vertex
        || evaluation.stage != ShaderStage::TessellationEvaluation
        || points == 0
    {
        return Err(Error::at(
            source,
            "default control requires VS/TES and nonzero patch cardinality",
        ));
    }
    let mut required = match domain {
        TessellationDomain::Isolines => 0b00_0011,
        TessellationDomain::Triangles => 0b01_0111,
        TessellationDomain::Quads => 0b11_1111,
    };
    let producer: BTreeMap<_, _> = vertex
        .outputs
        .iter()
        .map(|e| ((e.location(), e.component()), e.scalar_type()))
        .collect();
    let mut forwarded = BTreeMap::new();
    for e in &evaluation.inputs {
        let location = e.location();
        if crate::shader::tessellation::per_vertex(location) {
            if producer.get(&(location, e.component())) != Some(&e.scalar_type()) {
                return Err(Error::at(
                    source,
                    format!(
                        "default patch input {location:?}.{} has no matching vertex output",
                        e.component()
                    ),
                ));
            }
            forwarded.insert(
                (location, e.component()),
                ShaderInterfaceElement::new(location, e.component(), e.scalar_type(), None)
                    .unwrap(),
            );
        } else {
            match location {
                ShaderIoLocation::TessLevelOuter => required |= 1 << e.component(),
                ShaderIoLocation::TessLevelInner => required |= 1 << (4 + e.component()),
                ShaderIoLocation::TessCoord
                | ShaderIoLocation::PrimitiveId
                | ShaderIoLocation::PatchVertices => {}
                _ => {
                    return Err(Error::at(
                        source,
                        "per-patch user inputs require a guest control shader",
                    ));
                }
            }
        }
    }
    let mut b = FunctionBuilder::new(ShaderStage::TessellationControl);
    let entry = b.entry();
    let write = b.block(source);
    let end = b.block(source);
    let invocation = b.emit(
        entry,
        source,
        Op::Input {
            location: ShaderIoLocation::InvocationId,
            component: 0,
            ty: Type::U32,
        },
        vec![],
    )[0];
    for &(location, component) in forwarded.keys() {
        let value = b.emit(
            entry,
            source,
            Op::ControlPointLoad {
                location,
                component: u32::from(component),
                output: false,
            },
            vec![invocation.into()],
        )[0];
        b.emit(
            entry,
            source,
            Op::ControlPointStore {
                location,
                component: u32::from(component),
            },
            vec![invocation.into(), value.into()],
        );
    }
    let zero = b.emit(entry, source, Op::Constant(Constant::u32(0)), vec![])[0];
    let first = b.emit(
        entry,
        source,
        Op::IntegerCompare {
            ty: Type::U32,
            comparison: ShaderIntegerComparison::Equal,
        },
        vec![invocation.into(), zero.into()],
    )[0];
    b.terminate(
        entry,
        source,
        Terminator::Branch {
            condition: first.into(),
            when_true: write,
            when_false: end,
        },
    );
    let mut outputs: Vec<_> = forwarded.values().copied().collect();
    for lane in 0..6 {
        if required & (1 << lane) == 0 {
            continue;
        }
        let (location, component) = if lane < 4 {
            (ShaderIoLocation::TessLevelOuter, lane)
        } else {
            (ShaderIoLocation::TessLevelInner, lane - 4)
        };
        let value = b.emit(
            write,
            source,
            Op::Parameter {
                index: lane,
                ty: Type::F32,
            },
            vec![],
        )[0];
        b.emit(
            write,
            source,
            Op::Output {
                location,
                component,
                ty: Type::F32,
            },
            vec![value.into()],
        );
        outputs.push(
            ShaderInterfaceElement::new(location, component as u8, ShaderScalarType::Float32, None)
                .unwrap(),
        );
    }
    b.terminate(write, source, Terminator::Jump(end));
    b.terminate(end, source, Terminator::Return);
    let mut function = b.finish()?;
    function.inputs = forwarded.values().copied().collect();
    function.inputs.push(
        ShaderInterfaceElement::new(
            ShaderIoLocation::InvocationId,
            0,
            ShaderScalarType::Unsigned32,
            None,
        )
        .unwrap(),
    );
    function.outputs = outputs;
    function.tessellation_control_points = Some(points);
    function.optimize()?;
    Ok(DefaultControl {
        function,
        required_levels: required,
    })
}
