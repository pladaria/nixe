//! Maxwell quad-local SHFL forms mapped to rasterizer quads rather than host
//! warp IDs. General warp shuffles still require a separate subgroup contract.
use super::decode::validate_register_range;
use super::error::{MaxwellShaderTranslationError, malformed};
use crate::MaxwellShaderStage;
use nixe_gpu::{
    ShaderIntegerComparison, ShaderOperation, ShaderPredicate, ShaderPredicateSetOperation,
    ShaderQuadOperation, ShaderRegister,
};

pub(super) const fn is_shuffle(encoding: u64) -> bool {
    encoding & 0xfff8_0000_0000_0000 == 0xef10_0000_0000_0000
}

pub(super) fn decode_shuffle(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
) -> Result<Vec<ShaderOperation>, MaxwellShaderTranslationError> {
    // Operand flags, 5-bit lane and 13-bit clamp/segment mask follow emitSHFL.
    // https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp
    // Quad mask fixes bits 2:4 to this quad and clamps the low lane ID to 3.
    // https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/frontend/maxwell/translate/impl/warp_shuffle.cpp
    if encoding & !0xffff_7ffc_f1ff_ffff != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "SHFL reserved field is nonzero",
        ));
    }
    let index = ((encoding >> 20) & 31) as u8;
    let mode = (encoding >> 30) & 3;
    let operation = match mode {
        0 if index <= 3 => Some(ShaderQuadOperation::Broadcast(index)),
        3 => match index {
            1 => Some(ShaderQuadOperation::SwapHorizontal),
            2 => Some(ShaderQuadOperation::SwapVertical),
            3 => Some(ShaderQuadOperation::SwapDiagonal),
            _ => None,
        },
        _ => None,
    };
    let Some(operation) = operation.filter(|_| {
        stage == MaxwellShaderStage::Pixel
            && (encoding >> 28) & 3 == 3
            && (encoding >> 34) & 0x1fff == 0x1c03
    }) else {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "SHFL requires an immediate fragment quad-local IDX or BFLY form",
        });
    };
    let destination = encoding as u8;
    let source = ((encoding >> 8) & 255) as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    validate_register_range(stage, offset, encoding, source, 1, register_count)?;
    let source = ShaderRegister::new(u16::from(source));
    let mut operations = Vec::with_capacity(2);
    operations.push(ShaderOperation::QuadShuffle32 {
        destination: ShaderRegister::new(u16::from(destination)),
        source,
        operation,
    });
    let predicate = ((encoding >> 48) & 7) as u8;
    if predicate != 7 {
        // Every source lane is in bounds for this quad mask and immediate index.
        operations.push(ShaderOperation::SetPredicateInteger32 {
            destinations: [Some(predicate), None],
            left: source,
            right: source,
            signed: false,
            comparison: ShaderIntegerComparison::Equal,
            accumulator: ShaderPredicate::Always,
            set_operation: ShaderPredicateSetOperation::And,
        });
    }
    Ok(operations)
}

pub(super) const fn is_swizzle_add(encoding: u64) -> bool {
    encoding & 0xfff8_0000_0000_0000 == 0x50f8_0000_0000_0000
}

pub(super) fn decode_swizzle_add(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
) -> Result<ShaderOperation, MaxwellShaderTranslationError> {
    // emitFSWZADD supplies the swizzle byte, rounding, FTZ, NDV and CC fields.
    // https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp
    // Lookup coefficients: A=[-1,+1,-1,0], B=[-1,-1,+1,-1].
    // https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/backend/spirv/emit_spirv_warp.cpp
    if stage != MaxwellShaderStage::Pixel
        || (encoding >> 39) & 3 != 0
        || encoding & ((1 << 38) | (1 << 47)) != 0
    {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "FSWZADD requires fragment stage, nearest-even rounding, no NDV or CC",
        });
    }
    if encoding & !0xffff_11cf_ffff_ffff != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "FSWZADD reserved field is nonzero",
        ));
    }
    let destination = encoding as u8;
    let left = ((encoding >> 8) & 255) as u8;
    let right = ((encoding >> 20) & 255) as u8;
    for reg in [destination, left, right] {
        validate_register_range(stage, offset, encoding, reg, 1, register_count)?;
    }
    let ftz = encoding & (1 << 44) != 0;
    Ok(ShaderOperation::FloatQuadSwizzleAdd32 {
        destination: ShaderRegister::new(u16::from(destination)),
        left: ShaderRegister::new(u16::from(left)),
        right: ShaderRegister::new(u16::from(right)),
        swizzle: (encoding >> 28) as u8,
        float_control: nixe_gpu::ShaderFloatControl::new(
            nixe_gpu::ShaderRoundingMode::NearestEven,
            nixe_gpu::ShaderNanMode::Propagate,
            ftz,
            ftz,
            false,
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quad_shuffle_decodes_lane_modes_and_keeps_bounds_predicate() {
        let captured = 0xef17700cf0270203;
        for (mode, index, expected) in [
            (0, 0, ShaderQuadOperation::Broadcast(0)),
            (0, 3, ShaderQuadOperation::Broadcast(3)),
            (3, 1, ShaderQuadOperation::SwapHorizontal),
            (3, 2, ShaderQuadOperation::SwapVertical),
            (3, 3, ShaderQuadOperation::SwapDiagonal),
        ] {
            let word = (captured & !((3 << 30) | (31 << 20) | (7 << 48)))
                | (mode << 30)
                | (index << 20)
                | (2 << 48);
            let ops = decode_shuffle(MaxwellShaderStage::Pixel, 0xf0, word, 4).unwrap();
            assert_eq!(
                ops[0],
                ShaderOperation::QuadShuffle32 {
                    destination: ShaderRegister::new(3),
                    source: ShaderRegister::new(2),
                    operation: expected
                }
            );
            assert!(matches!(
                ops[1],
                ShaderOperation::SetPredicateInteger32 {
                    destinations: [Some(2), None],
                    comparison: ShaderIntegerComparison::Equal,
                    ..
                }
            ));
        }
        assert!(decode_shuffle(MaxwellShaderStage::Pixel, 0xf0, captured ^ (1 << 34), 4).is_err());
        assert!(decode_shuffle(MaxwellShaderStage::Vertex, 0xf0, captured, 4).is_err());
    }
    #[test]
    fn fragment_shuffle_reaches_verified_wgsl_and_false_predicate_stays_dead() {
        use super::super::test_support::translated_fixture;
        let mut header = [0; 20];
        header[0] = 0x00021462;
        header[18] = 15;
        let word = 0xef17700cf0270203;
        let ir = translated_fixture(
            MaxwellShaderStage::Pixel,
            header,
            &[0, word, 0xe30000000007000f, 0],
        );
        let module = nixe_gpu::lower_shader_ir_to_wgsl(&ir).unwrap();
        assert!(module.source().contains("quadSwapY(registers[2])"));
        let parsed = naga::front::wgsl::parse_str(module.source()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::SUBGROUP,
        )
        .validate(&parsed)
        .unwrap();
        let ir = translated_fixture(
            MaxwellShaderStage::Pixel,
            header,
            &[0, word | (1 << 19), 0xe30000000007000f, 0],
        );
        assert!(
            !ir.ir()
                .instructions()
                .iter()
                .any(|i| matches!(i.operation(), ShaderOperation::QuadShuffle32 { .. }))
        );
    }
    #[test]
    fn swizzled_add_decodes_sources_selectors_and_exact_float_control() {
        let word = 0x50f8000a50270303;
        assert!(
            matches!(decode_swizzle_add(MaxwellShaderStage::Pixel, 0x108, word, 4).unwrap(),
            ShaderOperation::FloatQuadSwizzleAdd32 {destination,left,right,swizzle:0xa5,float_control}
            if destination.index()==3 && left.index()==3 && right.index()==2 && float_control == nixe_gpu::ShaderFloatControl::PRECISE)
        );
        let ShaderOperation::FloatQuadSwizzleAdd32 { float_control, .. } =
            decode_swizzle_add(MaxwellShaderStage::Pixel, 0x108, word | (1 << 44), 4).unwrap()
        else {
            panic!("expected swizzled add")
        };
        assert!(float_control.denormals_are_zero() && float_control.flush_denormals_to_zero());
        for bit in [36, 38, 39, 47] {
            assert!(
                decode_swizzle_add(MaxwellShaderStage::Pixel, 0x108, word | (1 << bit), 4).is_err()
            );
        }
    }
}
