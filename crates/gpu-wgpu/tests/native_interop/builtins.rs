//! Built-ins affect independent pixels, not just SPIR-V decorations.
use super::emitted::{immediate, interface, ir, load, store};
use ShaderIoLocation as L;
use ShaderOperation as O;
use ShaderRegister as R;
use ShaderScalarType as T;

const FLOAT: ShaderFloatControl = ShaderFloatControl::new(
    ShaderRoundingMode::NearestEven,
    ShaderNanMode::Propagate,
    true,
    true,
    false,
);
use nixe_gpu::*;

fn float(register: u16) -> O {
    O::ConvertIntegerToFloat32 {
        destination: R::new(register),
        source: R::new(register),
        source_type: T::Unsigned32,
    }
}
fn multiply(destination: u16, left: u16, right: u16) -> O {
    O::Multiply32 {
        destination: R::new(destination),
        left: R::new(left),
        right: R::new(right),
        scalar_type: T::Float32,
        float_control: FLOAT,
    }
}
fn add(destination: u16, left: u16, right: u16) -> O {
    O::Add32 {
        destination: R::new(destination),
        left: R::new(left),
        right: R::new(right),
        scalar_type: T::Float32,
        float_control: FLOAT,
    }
}
fn point(destination: u16, vertex: u16, component: u8) -> O {
    O::LoadControlPoint {
        destination: R::new(destination),
        vertex: R::new(vertex),
        output: false,
        location: L::Generic(0),
        component,
    }
}

pub fn shaders() -> Vec<VerifiedShaderIr> {
    let vs = ir(
        ShaderStage::Vertex,
        vec![
            interface(L::VertexId, 0, T::Unsigned32),
            interface(L::InstanceId, 0, T::Unsigned32),
        ],
        (0..2)
            .map(|c| interface(L::Generic(0), c, T::Float32))
            .collect(),
        vec![
            load(0, L::VertexId, 0, T::Unsigned32),
            float(0),
            immediate(2, 0.0625_f32.to_bits()),
            multiply(0, 0, 2),
            store(0, L::Generic(0), 0),
            load(1, L::InstanceId, 0, T::Unsigned32),
            float(1),
            immediate(2, 0.125_f32.to_bits()),
            multiply(1, 1, 2),
            store(1, L::Generic(0), 1),
            O::Exit,
        ],
    );
    let mut outputs: Vec<_> = (0..4)
        .map(|c| interface(L::Generic(0), c, T::Float32))
        .collect();
    outputs.extend((0..3).map(|c| interface(L::TessLevelOuter, c, T::Float32)));
    outputs.push(interface(L::TessLevelInner, 0, T::Float32));
    let mut code = vec![
        load(0, L::InvocationId, 0, T::Unsigned32),
        point(1, 0, 0),
        point(2, 0, 1),
        load(3, L::PatchVertices, 0, T::Unsigned32),
        float(3),
        immediate(5, 0.125_f32.to_bits()),
        multiply(3, 3, 5),
        load(4, L::PrimitiveId, 0, T::Unsigned32),
        float(4),
        immediate(5, 0.25_f32.to_bits()),
        multiply(4, 4, 5),
    ];
    for component in 0..4 {
        code.push(O::StoreControlPoint {
            source: R::new(u16::from(component) + 1),
            vertex: R::new(0),
            location: L::Generic(0),
            component,
        });
    }
    code.extend([
        immediate(5, 0),
        O::SetPredicateInteger32 {
            destinations: [Some(0), None],
            left: R::new(0),
            right: R::new(5),
            signed: false,
            comparison: ShaderIntegerComparison::Equal,
            accumulator: ShaderPredicate::Always,
            set_operation: ShaderPredicateSetOperation::And,
        },
        immediate(6, 1_f32.to_bits()),
        store(6, L::TessLevelOuter, 0),
        store(6, L::TessLevelOuter, 1),
        store(6, L::TessLevelOuter, 2),
        store(6, L::TessLevelInner, 0),
        O::Exit,
    ]);
    let mut inputs: Vec<_> = (0..2)
        .map(|c| interface(L::Generic(0), c, T::Float32))
        .collect();
    inputs.extend(
        [L::InvocationId, L::PatchVertices, L::PrimitiveId].map(|l| interface(l, 0, T::Unsigned32)),
    );
    let tc = ir(ShaderStage::TessellationControl, inputs, outputs, code);
    let code = tc
        .instructions()
        .iter()
        .map(|i| {
            ShaderInstruction::new(
                i.source(),
                if matches!(
                    i.operation(),
                    O::StoreOutput {
                        location: L::TessLevelOuter | L::TessLevelInner,
                        ..
                    }
                ) {
                    ShaderPredicate::Register {
                        register: 0,
                        inverted: false,
                    }
                } else {
                    i.predicate()
                },
                i.operation().clone(),
            )
        })
        .collect();
    let tc = ShaderIr::new(
        tc.stage(),
        tc.inputs().to_vec(),
        tc.outputs().to_vec(),
        vec![],
        code,
    )
    .with_tessellation_control_points(Some(3));

    let mut inputs: Vec<_> = (0..4)
        .map(|c| interface(L::Generic(0), c, T::Float32))
        .collect();
    inputs.extend((0..3).map(|c| interface(L::TessCoord, c, T::Float32)));
    inputs.extend([L::PatchVertices, L::PrimitiveId].map(|l| interface(l, 0, T::Unsigned32)));
    let mut outputs: Vec<_> = (0..4)
        .map(|c| interface(L::Position, c, T::Float32))
        .collect();
    outputs.extend((0..3).map(|c| {
        ShaderInterfaceElement::new(
            L::Generic(0),
            c,
            T::Float32,
            Some(ShaderInterpolation::Perspective),
        )
        .unwrap()
    }));
    // Two patches occupy the right/left top quadrants. PatchVertices must be
    // four in TCS and three in TES; PrimitiveId resets between draw instances.
    // https://docs.vulkan.org/refpages/latest/refpages/source/PatchVertices.html
    // https://docs.vulkan.org/refpages/latest/refpages/source/PrimitiveId.html
    let code = vec![
        load(0, L::TessCoord, 0, T::Float32),
        load(1, L::TessCoord, 1, T::Float32),
        load(2, L::TessCoord, 2, T::Float32),
        add(10, 0, 1),
        add(10, 10, 2),
        immediate(11, 0.0625_f32.to_bits()),
        multiply(10, 10, 11),
        load(3, L::PrimitiveId, 0, T::Unsigned32),
        float(3),
        O::FloatNegate32 {
            destination: R::new(3),
            source: R::new(3),
        },
        add(0, 0, 3),
        immediate(2, 0.25_f32.to_bits()),
        immediate(3, 1_f32.to_bits()),
        store(0, L::Position, 0),
        store(1, L::Position, 1),
        store(2, L::Position, 2),
        store(3, L::Position, 3),
        immediate(4, 2),
        point(5, 4, 0),
        point(6, 4, 1),
        point(7, 4, 2),
        point(8, 4, 3),
        add(5, 5, 8),
        store(5, L::Generic(0), 0),
        store(6, L::Generic(0), 1),
        load(9, L::PatchVertices, 0, T::Unsigned32),
        float(9),
        immediate(11, 0.125_f32.to_bits()),
        multiply(9, 9, 11),
        add(7, 7, 9),
        add(7, 7, 10),
        store(7, L::Generic(0), 2),
        O::Exit,
    ];
    let te = ir(ShaderStage::TessellationEvaluation, inputs, outputs, code);
    let [_, _, _, fs] = super::emitted::shaders();
    vec![
        VerifiedShaderIr::verify(vs).unwrap(),
        VerifiedShaderIr::verify(tc).unwrap(),
        VerifiedShaderIr::verify(te).unwrap(),
        fs,
    ]
}
