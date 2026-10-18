use crate::ir::Opcode as O;

pub(super) fn indexed(op: O) -> Option<(O, bool)> {
    Some(match op {
        O::BoundImageSampleImplicitLod => (O::ImageSampleImplicitLod, false),
        O::BindlessImageSampleImplicitLod => (O::ImageSampleImplicitLod, true),
        O::BoundImageSampleExplicitLod => (O::ImageSampleExplicitLod, false),
        O::BindlessImageSampleExplicitLod => (O::ImageSampleExplicitLod, true),
        O::BoundImageSampleDrefImplicitLod => (O::ImageSampleDrefImplicitLod, false),
        O::BindlessImageSampleDrefImplicitLod => (O::ImageSampleDrefImplicitLod, true),
        O::BoundImageSampleDrefExplicitLod => (O::ImageSampleDrefExplicitLod, false),
        O::BindlessImageSampleDrefExplicitLod => (O::ImageSampleDrefExplicitLod, true),
        O::BoundImageGather => (O::ImageGather, false),
        O::BindlessImageGather => (O::ImageGather, true),
        O::BoundImageGatherDref => (O::ImageGatherDref, false),
        O::BindlessImageGatherDref => (O::ImageGatherDref, true),
        O::BoundImageFetch => (O::ImageFetch, false),
        O::BindlessImageFetch => (O::ImageFetch, true),
        O::BoundImageQueryDimensions => (O::ImageQueryDimensions, false),
        O::BindlessImageQueryDimensions => (O::ImageQueryDimensions, true),
        O::BoundImageQueryLod => (O::ImageQueryLod, false),
        O::BindlessImageQueryLod => (O::ImageQueryLod, true),
        O::BoundImageGradient => (O::ImageGradient, false),
        O::BindlessImageGradient => (O::ImageGradient, true),
        O::BoundImageRead => (O::ImageRead, false),
        O::BindlessImageRead => (O::ImageRead, true),
        O::BoundImageWrite => (O::ImageWrite, false),
        O::BindlessImageWrite => (O::ImageWrite, true),
        O::BoundImageAtomicIAdd32 => (O::ImageAtomicIAdd32, false),
        O::BindlessImageAtomicIAdd32 => (O::ImageAtomicIAdd32, true),
        O::BoundImageAtomicSMin32 => (O::ImageAtomicSMin32, false),
        O::BindlessImageAtomicSMin32 => (O::ImageAtomicSMin32, true),
        O::BoundImageAtomicUMin32 => (O::ImageAtomicUMin32, false),
        O::BindlessImageAtomicUMin32 => (O::ImageAtomicUMin32, true),
        O::BoundImageAtomicSMax32 => (O::ImageAtomicSMax32, false),
        O::BindlessImageAtomicSMax32 => (O::ImageAtomicSMax32, true),
        O::BoundImageAtomicUMax32 => (O::ImageAtomicUMax32, false),
        O::BindlessImageAtomicUMax32 => (O::ImageAtomicUMax32, true),
        O::BoundImageAtomicInc32 => (O::ImageAtomicInc32, false),
        O::BindlessImageAtomicInc32 => (O::ImageAtomicInc32, true),
        O::BoundImageAtomicDec32 => (O::ImageAtomicDec32, false),
        O::BindlessImageAtomicDec32 => (O::ImageAtomicDec32, true),
        O::BoundImageAtomicAnd32 => (O::ImageAtomicAnd32, false),
        O::BindlessImageAtomicAnd32 => (O::ImageAtomicAnd32, true),
        O::BoundImageAtomicOr32 => (O::ImageAtomicOr32, false),
        O::BindlessImageAtomicOr32 => (O::ImageAtomicOr32, true),
        O::BoundImageAtomicXor32 => (O::ImageAtomicXor32, false),
        O::BindlessImageAtomicXor32 => (O::ImageAtomicXor32, true),
        O::BoundImageAtomicExchange32 => (O::ImageAtomicExchange32, false),
        O::BindlessImageAtomicExchange32 => (O::ImageAtomicExchange32, true),
        _ => return None,
    })
}

pub(super) fn storage(op: O) -> bool {
    matches!(
        op,
        O::ImageRead
            | O::ImageWrite
            | O::ImageAtomicIAdd32
            | O::ImageAtomicSMin32
            | O::ImageAtomicUMin32
            | O::ImageAtomicSMax32
            | O::ImageAtomicUMax32
            | O::ImageAtomicInc32
            | O::ImageAtomicDec32
            | O::ImageAtomicAnd32
            | O::ImageAtomicOr32
            | O::ImageAtomicXor32
            | O::ImageAtomicExchange32
    )
}
