use crate::{
    artifact::ShaderInfo,
    ir::{Opcode as O, Type},
};

pub(super) fn simple_usages(info: &mut ShaderInfo, op: O) {
    if matches!(
        op,
        O::CompositeConstructF16x2
            | O::CompositeConstructF16x3
            | O::CompositeConstructF16x4
            | O::CompositeExtractF16x2
            | O::CompositeExtractF16x3
            | O::CompositeExtractF16x4
            | O::CompositeInsertF16x2
            | O::CompositeInsertF16x3
            | O::CompositeInsertF16x4
            | O::SelectF16
            | O::BitCastU16F16
            | O::BitCastF16U16
            | O::PackFloat2x16
            | O::UnpackFloat2x16
            | O::ConvertS16F16
            | O::ConvertS32F16
            | O::ConvertS64F16
            | O::ConvertU16F16
            | O::ConvertU32F16
            | O::ConvertU64F16
            | O::ConvertF16S8
            | O::ConvertF16S16
            | O::ConvertF16S32
            | O::ConvertF16S64
            | O::ConvertF16U8
            | O::ConvertF16U16
            | O::ConvertF16U32
            | O::ConvertF16U64
            | O::ConvertF16F32
            | O::ConvertF32F16
            | O::FPAbs16
            | O::FPAdd16
            | O::FPCeil16
            | O::FPFloor16
            | O::FPFma16
            | O::FPMul16
            | O::FPNeg16
            | O::FPRoundEven16
            | O::FPSaturate16
            | O::FPClamp16
            | O::FPTrunc16
            | O::FPOrdEqual16
            | O::FPUnordEqual16
            | O::FPOrdNotEqual16
            | O::FPUnordNotEqual16
            | O::FPOrdLessThan16
            | O::FPUnordLessThan16
            | O::FPOrdGreaterThan16
            | O::FPUnordGreaterThan16
            | O::FPOrdLessThanEqual16
            | O::FPUnordLessThanEqual16
            | O::FPOrdGreaterThanEqual16
            | O::FPUnordGreaterThanEqual16
            | O::FPIsNan16
            | O::GlobalAtomicAddF16x2
            | O::GlobalAtomicMinF16x2
            | O::GlobalAtomicMaxF16x2
            | O::StorageAtomicAddF16x2
            | O::StorageAtomicMinF16x2
            | O::StorageAtomicMaxF16x2
    ) {
        info.uses_fp16 = true;
    }
    if matches!(
        op,
        O::CompositeConstructF64x2
            | O::CompositeConstructF64x3
            | O::CompositeConstructF64x4
            | O::CompositeExtractF64x2
            | O::CompositeExtractF64x3
            | O::CompositeExtractF64x4
            | O::CompositeInsertF64x2
            | O::CompositeInsertF64x3
            | O::CompositeInsertF64x4
            | O::SelectF64
            | O::BitCastU64F64
            | O::BitCastF64U64
            | O::PackDouble2x32
            | O::UnpackDouble2x32
            | O::FPAbs64
            | O::FPAdd64
            | O::FPCeil64
            | O::FPFloor64
            | O::FPFma64
            | O::FPMax64
            | O::FPMin64
            | O::FPMul64
            | O::FPNeg64
            | O::FPRecip64
            | O::FPRecipSqrt64
            | O::FPRoundEven64
            | O::FPSaturate64
            | O::FPClamp64
            | O::FPTrunc64
            | O::FPOrdEqual64
            | O::FPUnordEqual64
            | O::FPOrdNotEqual64
            | O::FPUnordNotEqual64
            | O::FPOrdLessThan64
            | O::FPUnordLessThan64
            | O::FPOrdGreaterThan64
            | O::FPUnordGreaterThan64
            | O::FPOrdLessThanEqual64
            | O::FPUnordLessThanEqual64
            | O::FPOrdGreaterThanEqual64
            | O::FPUnordGreaterThanEqual64
            | O::FPIsNan64
            | O::ConvertS16F64
            | O::ConvertS32F64
            | O::ConvertS64F64
            | O::ConvertU16F64
            | O::ConvertU32F64
            | O::ConvertU64F64
            | O::ConvertF32F64
            | O::ConvertF64F32
            | O::ConvertF64S8
            | O::ConvertF64S16
            | O::ConvertF64S32
            | O::ConvertF64S64
            | O::ConvertF64U8
            | O::ConvertF64U16
            | O::ConvertF64U32
            | O::ConvertF64U64
    ) {
        info.uses_fp64 = true;
    }
    if matches!(
        op,
        O::GetCbufU8
            | O::GetCbufS8
            | O::UndefU8
            | O::LoadGlobalU8
            | O::LoadGlobalS8
            | O::WriteGlobalU8
            | O::WriteGlobalS8
            | O::LoadStorageU8
            | O::LoadStorageS8
            | O::WriteStorageU8
            | O::WriteStorageS8
            | O::LoadSharedU8
            | O::LoadSharedS8
            | O::WriteSharedU8
            | O::SelectU8
            | O::ConvertF16S8
            | O::ConvertF16U8
            | O::ConvertF32S8
            | O::ConvertF32U8
            | O::ConvertF64S8
            | O::ConvertF64U8
    ) {
        info.uses_int8 = true;
    }
    if matches!(
        op,
        O::GetCbufU16
            | O::GetCbufS16
            | O::UndefU16
            | O::LoadGlobalU16
            | O::LoadGlobalS16
            | O::WriteGlobalU16
            | O::WriteGlobalS16
            | O::LoadStorageU16
            | O::LoadStorageS16
            | O::WriteStorageU16
            | O::WriteStorageS16
            | O::LoadSharedU16
            | O::LoadSharedS16
            | O::WriteSharedU16
            | O::SelectU16
            | O::BitCastU16F16
            | O::BitCastF16U16
            | O::ConvertS16F16
            | O::ConvertS16F32
            | O::ConvertS16F64
            | O::ConvertU16F16
            | O::ConvertU16F32
            | O::ConvertU16F64
            | O::ConvertF16S16
            | O::ConvertF16U16
            | O::ConvertF32S16
            | O::ConvertF32U16
            | O::ConvertF64S16
            | O::ConvertF64U16
    ) {
        info.uses_int16 = true;
    }
    if matches!(
        op,
        O::UndefU64
            | O::LoadGlobalU8
            | O::LoadGlobalS8
            | O::LoadGlobalU16
            | O::LoadGlobalS16
            | O::LoadGlobal32
            | O::LoadGlobal64
            | O::LoadGlobal128
            | O::WriteGlobalU8
            | O::WriteGlobalS8
            | O::WriteGlobalU16
            | O::WriteGlobalS16
            | O::WriteGlobal32
            | O::WriteGlobal64
            | O::WriteGlobal128
            | O::SelectU64
            | O::BitCastU64F64
            | O::BitCastF64U64
            | O::PackUint2x32
            | O::UnpackUint2x32
            | O::IAdd64
            | O::ISub64
            | O::INeg64
            | O::ShiftLeftLogical64
            | O::ShiftRightLogical64
            | O::ShiftRightArithmetic64
            | O::ConvertS64F16
            | O::ConvertS64F32
            | O::ConvertS64F64
            | O::ConvertU64F16
            | O::ConvertU64F32
            | O::ConvertU64F64
            | O::ConvertU64U32
            | O::ConvertU32U64
            | O::ConvertF16U64
            | O::ConvertF32U64
            | O::ConvertF64U64
            | O::SharedAtomicExchange64
            | O::GlobalAtomicIAdd64
            | O::GlobalAtomicSMin64
            | O::GlobalAtomicUMin64
            | O::GlobalAtomicSMax64
            | O::GlobalAtomicUMax64
            | O::GlobalAtomicAnd64
            | O::GlobalAtomicOr64
            | O::GlobalAtomicXor64
            | O::GlobalAtomicExchange64
            | O::StorageAtomicIAdd64
            | O::StorageAtomicSMin64
            | O::StorageAtomicUMin64
            | O::StorageAtomicSMax64
            | O::StorageAtomicUMax64
            | O::StorageAtomicAnd64
            | O::StorageAtomicOr64
            | O::StorageAtomicXor64
            | O::StorageAtomicExchange64
    ) {
        info.uses_int64 = true;
    }
    if matches!(
        op,
        O::LoadGlobalU8
            | O::LoadGlobalS8
            | O::LoadGlobalU16
            | O::LoadGlobalS16
            | O::LoadGlobal32
            | O::LoadGlobal64
            | O::LoadGlobal128
    ) {
        info.uses_int64 = true;
        info.uses_global_memory = true;
        info.used_constant_buffer_types |= Type::U32 | Type::U32x2;
        info.used_storage_buffer_types |= Type::U32 | Type::U32x2 | Type::U32x4;
    }
    if matches!(op, O::LoadLocal | O::WriteLocal) {
        info.uses_local_memory = true;
    }
    if matches!(op, O::DemoteToHelperInvocation) {
        info.uses_demote_to_helper_invocation = true;
    }
    if matches!(op, O::GetAttributeIndexed) {
        info.loads_indexed_attributes = true;
    }
    if matches!(op, O::SetAttributeIndexed) {
        info.stores_indexed_attributes = true;
    }
    if matches!(op, O::SetSampleMask) {
        info.stores_sample_mask = true;
    }
    if matches!(op, O::SetFragDepth) {
        info.stores_frag_depth = true;
    }
    if matches!(op, O::WorkgroupId) {
        info.uses_workgroup_id = true;
    }
    if matches!(op, O::LocalInvocationId) {
        info.uses_local_invocation_id = true;
    }
    if matches!(op, O::InvocationId) {
        info.uses_invocation_id = true;
    }
    if matches!(op, O::InvocationInfo) {
        info.uses_invocation_info = true;
    }
    if matches!(op, O::SampleId) {
        info.uses_sample_id = true;
    }
    if matches!(op, O::IsHelperInvocation) {
        info.uses_is_helper_invocation = true;
    }
    if matches!(
        op,
        O::ResolutionDownFactor | O::IsTextureScaled | O::IsImageScaled
    ) {
        info.uses_rescaling_uniform = true;
    }
    if matches!(op, O::LaneId) {
        info.uses_subgroup_invocation_id = true;
    }
    if matches!(
        op,
        O::ShuffleIndex | O::ShuffleUp | O::ShuffleDown | O::ShuffleButterfly
    ) {
        info.uses_subgroup_shuffles = true;
    }
    if matches!(op, O::QuadBroadcast | O::QuadSwap) {
        info.uses_quad_shuffles = true;
    }
    if matches!(
        op,
        O::SubgroupEqMask
            | O::SubgroupLtMask
            | O::SubgroupLeMask
            | O::SubgroupGtMask
            | O::SubgroupGeMask
    ) {
        info.uses_subgroup_mask = true;
    }
    if matches!(
        op,
        O::VoteAll | O::VoteAny | O::VoteEqual | O::SubgroupBallot
    ) {
        info.uses_subgroup_vote = true;
    }
    if matches!(op, O::FSwizzleAdd) {
        info.uses_fswzadd = true;
    }
    if matches!(
        op,
        O::DPdxFine | O::DPdyFine | O::DPdxCoarse | O::DPdyCoarse
    ) {
        info.uses_derivatives = true;
    }
    if matches!(
        op,
        O::LoadStorageU8 | O::LoadStorageS8 | O::WriteStorageU8 | O::WriteStorageS8
    ) {
        info.used_storage_buffer_types |= Type::U8;
    }
    if matches!(
        op,
        O::LoadStorageU16 | O::LoadStorageS16 | O::WriteStorageU16 | O::WriteStorageS16
    ) {
        info.used_storage_buffer_types |= Type::U16;
    }
    if matches!(
        op,
        O::LoadStorage32
            | O::WriteStorage32
            | O::StorageAtomicIAdd32
            | O::StorageAtomicUMin32
            | O::StorageAtomicUMax32
            | O::StorageAtomicAnd32
            | O::StorageAtomicOr32
            | O::StorageAtomicXor32
            | O::StorageAtomicExchange32
    ) {
        info.used_storage_buffer_types |= Type::U32;
    }
    if matches!(
        op,
        O::LoadStorage64
            | O::WriteStorage64
            | O::StorageAtomicIAdd32x2
            | O::StorageAtomicSMin32x2
            | O::StorageAtomicUMin32x2
            | O::StorageAtomicSMax32x2
            | O::StorageAtomicUMax32x2
            | O::StorageAtomicAnd32x2
            | O::StorageAtomicOr32x2
            | O::StorageAtomicXor32x2
            | O::StorageAtomicExchange32x2
    ) {
        info.used_storage_buffer_types |= Type::U32x2;
    }
    if matches!(op, O::LoadStorage128 | O::WriteStorage128) {
        info.used_storage_buffer_types |= Type::U32x4;
    }
    if matches!(op, O::SharedAtomicSMin32) {
        info.uses_atomic_s32_min = true;
    }
    if matches!(op, O::SharedAtomicSMax32) {
        info.uses_atomic_s32_max = true;
    }
    if matches!(op, O::SharedAtomicInc32) {
        info.uses_shared_increment = true;
    }
    if matches!(op, O::SharedAtomicDec32) {
        info.uses_shared_decrement = true;
    }
    if matches!(op, O::SharedAtomicExchange64) {
        info.uses_int64_bit_atomics = true;
    }
    if matches!(op, O::GlobalAtomicInc32 | O::StorageAtomicInc32) {
        info.used_storage_buffer_types |= Type::U32;
        info.uses_global_increment = true;
    }
    if matches!(op, O::GlobalAtomicDec32 | O::StorageAtomicDec32) {
        info.used_storage_buffer_types |= Type::U32;
        info.uses_global_decrement = true;
    }
    if matches!(op, O::GlobalAtomicAddF32 | O::StorageAtomicAddF32) {
        info.used_storage_buffer_types |= Type::U32;
        info.uses_atomic_f32_add = true;
    }
    if matches!(op, O::GlobalAtomicAddF16x2 | O::StorageAtomicAddF16x2) {
        info.used_storage_buffer_types |= Type::U32;
        info.uses_atomic_f16x2_add = true;
    }
    if matches!(op, O::GlobalAtomicAddF32x2 | O::StorageAtomicAddF32x2) {
        info.used_storage_buffer_types |= Type::U32;
        info.uses_atomic_f32x2_add = true;
    }
    if matches!(op, O::GlobalAtomicMinF16x2 | O::StorageAtomicMinF16x2) {
        info.used_storage_buffer_types |= Type::U32;
        info.uses_atomic_f16x2_min = true;
    }
    if matches!(op, O::GlobalAtomicMinF32x2 | O::StorageAtomicMinF32x2) {
        info.used_storage_buffer_types |= Type::U32;
        info.uses_atomic_f32x2_min = true;
    }
    if matches!(op, O::GlobalAtomicMaxF16x2 | O::StorageAtomicMaxF16x2) {
        info.used_storage_buffer_types |= Type::U32;
        info.uses_atomic_f16x2_max = true;
    }
    if matches!(op, O::GlobalAtomicMaxF32x2 | O::StorageAtomicMaxF32x2) {
        info.used_storage_buffer_types |= Type::U32;
        info.uses_atomic_f32x2_max = true;
    }
    if matches!(op, O::StorageAtomicSMin32) {
        info.used_storage_buffer_types |= Type::U32;
        info.uses_atomic_s32_min = true;
    }
    if matches!(op, O::StorageAtomicSMax32) {
        info.used_storage_buffer_types |= Type::U32;
        info.uses_atomic_s32_max = true;
    }
    if matches!(
        op,
        O::GlobalAtomicIAdd64
            | O::GlobalAtomicSMin64
            | O::GlobalAtomicUMin64
            | O::GlobalAtomicSMax64
            | O::GlobalAtomicUMax64
            | O::GlobalAtomicAnd64
            | O::GlobalAtomicOr64
            | O::GlobalAtomicXor64
            | O::GlobalAtomicExchange64
            | O::StorageAtomicIAdd64
            | O::StorageAtomicSMin64
            | O::StorageAtomicUMin64
            | O::StorageAtomicSMax64
            | O::StorageAtomicUMax64
            | O::StorageAtomicAnd64
            | O::StorageAtomicOr64
            | O::StorageAtomicXor64
    ) {
        info.used_storage_buffer_types |= Type::U64 | Type::U32x2;
        info.uses_int64_bit_atomics = true;
    }
}
