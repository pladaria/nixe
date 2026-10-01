//! Compute programs start with a scheduling bundle, not a graphics SPH.
//! Their register count and workgroup dimensions come from the consumed QMD.
//! https://github.com/devkitPro/deko3d/blob/master/source/maxwell/gpu_compute.cpp
use super::binary::{
    MAXWELL_SCHEDULE_BUNDLE_SIZE, MaxwellShaderBinary, MaxwellShaderMemoryView,
    MaxwellShaderMetadata, read_shader_code,
};
use super::decode::validate_register_range;
use super::error::{MaxwellShaderTranslationError, malformed};
use super::global_memory;
use super::translate::translate_shader_binary;
use crate::{MaxwellGpuAddressSpace, MaxwellShaderStage};
use nixe_gpu::{ShaderBackendModule, ShaderOperation, ShaderRegister, VerifiedShaderIr};
use nixe_memory::{CanonicalCpuWriteDependency, CanonicalWriteBatch};
use std::collections::BTreeMap;

#[cfg(all(test, not(target_os = "macos")))]
mod guest_execution;

#[derive(Debug)]
pub(crate) struct MaxwellComputeProgram {
    binary: MaxwellShaderBinary,
    pub module: ShaderBackendModule,
    pub global_buffers: global_memory::GlobalBufferBindings,
    pub constant_buffer_extents: BTreeMap<u8, u64>,
}

pub(crate) fn translate_compute_program(
    address_space: &MaxwellGpuAddressSpace,
    staged_writes: &CanonicalWriteBatch,
    address: u64,
    register_count: u8,
    workgroup_size: [u32; 3],
) -> Result<MaxwellComputeProgram, MaxwellShaderTranslationError> {
    let memory = MaxwellShaderMemoryView::new(address_space, staged_writes);
    let binary = read_shader_code(
        &memory,
        MaxwellShaderStage::Compute,
        address,
        MaxwellShaderMetadata::Compute {
            workgroup_size,
            register_count,
        },
        None,
    )?;
    let translated = translate_shader_binary(&binary, register_count, &BTreeMap::new())?;
    // Cache the statically addressed constant-buffer requirements with the
    // translation; dispatch must not walk instructions to validate QMD sizes.
    let mut constant_buffer_extents = BTreeMap::<u8, u64>::new();
    for instruction in translated.ir.instructions() {
        if let ShaderOperation::LoadConstantBuffer32 {
            binding,
            byte_offset,
            ..
        } = instruction.operation()
        {
            let extent = constant_buffer_extents.entry(*binding).or_default();
            *extent = (*extent).max(u64::from(*byte_offset) + 4);
        }
    }
    Ok(MaxwellComputeProgram {
        binary,
        module: ShaderBackendModule::new(VerifiedShaderIr::verify(translated.ir)?),
        global_buffers: translated.global_buffers,
        constant_buffer_extents,
    })
}

impl MaxwellComputeProgram {
    pub(crate) fn source_is_current(
        &self,
        address_space: &MaxwellGpuAddressSpace,
        writes: &CanonicalWriteBatch,
    ) -> bool {
        let binary = &self.binary;
        if !binary
            .source_cpu_writes
            .iter()
            .all(CanonicalCpuWriteDependency::remains_current)
        {
            return false;
        }
        let end =
            binary.address + binary.bundles.len() as u64 * MAXWELL_SCHEDULE_BUNDLE_SIZE as u64;
        binary.source_mappings.iter().all(|mapping| {
            if !address_space.retained_mapping_is_current(mapping) {
                return false;
            }
            let start = binary.address.max(mapping.offset().get());
            let end = end.min(mapping.offset().get() + mapping.size());
            start >= end
                || writes.overlaps(
                    mapping.backing(),
                    mapping.backing_offset() + start - mapping.offset().get(),
                    end - start,
                ) == Ok(false)
        })
    }
}

pub(super) fn system_register(
    offset: u32,
    encoding: u64,
    register_count: u8,
) -> Result<ShaderOperation, MaxwellShaderTranslationError> {
    let stage = MaxwellShaderStage::Compute;
    // emitSYS and emitS2R define the selector and the reserved operand fields.
    // https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp
    if encoding & !0xffff_0000_0fff_00ff != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "S2R reserved field is nonzero",
        ));
    }
    let selector = ((encoding >> 20) & 255) as u8;
    let (builtin, component) = match selector {
        0x21..=0x23 => (
            nixe_gpu::ShaderComputeBuiltin::LocalInvocationId,
            selector - 0x21,
        ),
        0x25..=0x27 => (nixe_gpu::ShaderComputeBuiltin::WorkgroupId, selector - 0x25),
        _ => {
            return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                stage,
                instruction_offset: offset,
                encoding,
                detail: "compute S2R system register is not translated",
            });
        }
    };
    validate_register_range(stage, offset, encoding, encoding as u8, 1, register_count)?;
    Ok(ShaderOperation::LoadComputeBuiltin32 {
        destination: ShaderRegister::new(u16::from(encoding as u8)),
        builtin,
        component,
    })
}

#[cfg(test)]
mod tests;
