//! Exact storage-backed vertex fetches and quad constant-attribute assembly.

use super::{
    InterfaceGroup, ShaderBackendLoweringError, ShaderInterpolation, ShaderIoLocation,
    ShaderScalarType, wgsl_field_name,
};
use crate::{VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
enum Encoding {
    Float,
    Uint,
    Sint,
    Uscaled,
    Sscaled,
    Unorm,
    Snorm,
}

pub(super) fn attribute_value(
    slot: usize,
    layout: &VertexBufferLayout,
    attribute: &VertexAttribute,
    scalar_type: ShaderScalarType,
) -> Result<String, ShaderBackendLoweringError> {
    use Encoding::*;
    use VertexFormat::*;
    let format = attribute.format;
    let (encoding, width, components) = if let Some((signed, width, count)) = format.scaled_layout()
    {
        (
            if signed {
                Encoding::Sscaled
            } else {
                Encoding::Uscaled
            },
            width.bytes() as u32 * 8,
            count.get(),
        )
    } else if let Some((signed, width, count)) = format.integer_layout() {
        (
            if signed { Sint } else { Uint },
            width.bytes() as u32 * 8,
            count.get(),
        )
    } else {
        match format {
            Unorm8x2 => (Unorm, 8, 2),
            Unorm8x4 => (Unorm, 8, 4),
            Snorm8x2 => (Snorm, 8, 2),
            Snorm8x4 => (Snorm, 8, 4),
            Unorm16x2 => (Unorm, 16, 2),
            Unorm16x4 => (Unorm, 16, 4),
            Snorm16x2 => (Snorm, 16, 2),
            Snorm16x4 => (Snorm, 16, 4),
            Float16x2 => (Float, 16, 2),
            Float16x4 => (Float, 16, 4),
            Float32 => (Float, 32, 1),
            Float32x2 => (Float, 32, 2),
            Float32x3 => (Float, 32, 3),
            Float32x4 => (Float, 32, 4),
            Unorm10_10_10_2 => (Unorm, 32, 4),
            _ => unreachable!("integer and scaled formats handled above"),
        }
    };
    let expected = match encoding {
        Uint => ShaderScalarType::Unsigned32,
        Sint => ShaderScalarType::Signed32,
        _ => ShaderScalarType::Float32,
    };
    if scalar_type != expected {
        return Err(ShaderBackendLoweringError::VertexFetch(
            "vertex storage format and shader input scalar type differ",
        ));
    }
    let element = match layout.step_mode {
        VertexStepMode::Vertex => "host.vertex_id",
        VertexStepMode::Instance => "host.instance_id",
    };
    let base = layout
        .buffer
        .range
        .offset()
        .checked_add(attribute.offset)
        .and_then(|base| u32::try_from(base).ok())
        .ok_or(ShaderBackendLoweringError::VertexFetch(
            "vertex attribute base exceeds WGSL u32",
        ))?;
    let stride = u32::try_from(layout.array_stride)
        .map_err(|_| ShaderBackendLoweringError::VertexFetch("vertex stride exceeds WGSL u32"))?;
    let mut values = Vec::with_capacity(4);
    for component in 0..4 {
        if component >= components {
            values.push(
                match (expected, component == 3) {
                    (ShaderScalarType::Float32, true) => "1.0",
                    (ShaderScalarType::Float32, false) => "0.0",
                    (ShaderScalarType::Signed32, true) => "1i",
                    (ShaderScalarType::Signed32, false) => "0i",
                    (ShaderScalarType::Unsigned32, true) => "1u",
                    (ShaderScalarType::Unsigned32, false) => "0u",
                    _ => unreachable!(),
                }
                .to_owned(),
            );
            continue;
        }
        let packed = format == Unorm10_10_10_2;
        let offset = base
            .checked_add(if packed {
                0
            } else {
                u32::from(component) * (width / 8)
            })
            .ok_or(ShaderBackendLoweringError::VertexFetch(
                "vertex component offset exceeds WGSL u32",
            ))?;
        let address = format!("{offset}u + {element} * {stride}u");
        let raw = if width == 32 && offset % 4 == 0 && stride % 4 == 0 {
            format!("nixe_vertex_buffer_{slot}[({address}) >> 2u]")
        } else {
            format!("nixe_vertex_buffer_{slot}_u{width}({address})")
        };
        let shift = 32 - width;
        let signed = format!("(i32({raw} << {shift}u) >> {shift}u)");
        // Vertex format conversion and missing-component defaults follow the
        // same contract as fixed-function fetch, including SNORM's -1 clamp.
        // https://www.w3.org/TR/webgpu/#vertex-formats
        values.push(if packed {
            let (shift, mask) = if component == 3 {
                (30, 3)
            } else {
                (u32::from(component) * 10, 1023)
            };
            format!("(f32(({raw} >> {shift}u) & {mask}u) / {mask}.0)")
        } else {
            match encoding {
                Float if width == 16 => format!("unpack2x16float({raw}).x"),
                Float => format!("bitcast<f32>({raw})"),
                Uint => raw,
                Sint => signed,
                Encoding::Uscaled => format!("f32({raw})"),
                Encoding::Sscaled => format!("f32({signed})"),
                Unorm => format!("(f32({raw}) / {}.0)", (1_u32 << width) - 1),
                Snorm => format!(
                    "max(f32({signed}) / {}.0, -1.0)",
                    (1_u32 << (width - 1)) - 1
                ),
            }
        });
    }
    let scalar = match expected {
        ShaderScalarType::Float32 => "f32",
        ShaderScalarType::Signed32 => "i32",
        ShaderScalarType::Unsigned32 => "u32",
        _ => unreachable!(),
    };
    Ok(format!("vec4<{scalar}>({})", values.join(", ")))
}

pub(super) fn emit_quad_flat_entry_point(
    source: &mut String,
    outputs: &BTreeMap<ShaderIoLocation, InterfaceGroup>,
) {
    // Both triangles start with corner zero. Only that invocation needs the
    // last corner's constant outputs; positions and smooth outputs are never
    // replaced. Re-evaluate the guest shader, not merely its input attributes:
    // constants may depend on vertex ID, positions, resources, or control flow.
    // https://www.w3.org/TR/WGSL/#interpolation
    source.push_str(
        "\nvar<immediate> nixe_quad_first_vertex: u32;\n\
         @vertex\nfn nixe_quad_flat(host: NixeVertexPullInput) -> ShaderOutput {\n\
           var output = nixe_guest_vertex(nixe_vertex_pull_input(host));\n\
           if ((host.vertex_id - nixe_quad_first_vertex) % 4u == 0u) {\n\
             var provoking_host = host;\n\
             provoking_host.vertex_id += 3u;\n\
             let provoking = nixe_guest_vertex(nixe_vertex_pull_input(provoking_host));\n",
    );
    for (location, group) in outputs {
        if matches!(
            location,
            ShaderIoLocation::Generic(_) | ShaderIoLocation::Color(_)
        ) && group.interpolation == Some(ShaderInterpolation::Constant)
        {
            let field = wgsl_field_name(*location);
            source.push_str(&format!("    output.{field} = provoking.{field};\n"));
        }
    }
    source.push_str("  }\n  return output;\n}\n");
}
