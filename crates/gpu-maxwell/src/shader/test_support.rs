//! Shared shader fixtures and validation helpers, compiled only for tests.

use super::binary::{MaxwellShaderMemoryView, read_shader_binary, validate_program_header};
use super::error::MaxwellShaderTranslationError;
use super::link::{MaxwellTranslatedShaderProgram, translate_prepared_maxwell_shader_programs};
use super::source::{
    prepare_maxwell_shader_translation_inputs_from_source,
    prepare_maxwell_shader_translation_source,
};
use super::translate::translate_shader_binary;
use crate::engines::dispatch_maxwell_engine_packet;
use crate::{
    MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
    MaxwellChannelId, MaxwellGpfifoSourceLocation, MaxwellGpuAddressSpace, MaxwellGpuChannel,
    MaxwellMapRequest, MaxwellMappingId, MaxwellPushbufferWord, MaxwellShaderStage,
    MaxwellThreeDState, SWITCH_1_GM20B_PROFILE, decode_maxwell_pushbuffer,
};
use nixe_gpu::{FrontendSubmissionId, GpuVirtualAddress, MappingGeneration, VerifiedShaderIr};
use nixe_memory::{CanonicalAllocation, CanonicalWriteBatch, MemoryPermissions};
use std::collections::BTreeMap;

pub(super) fn canonical_shader_writes(
    address_space: &MaxwellGpuAddressSpace,
    writes: &[(u64, u32)],
) -> CanonicalWriteBatch {
    let mut batch = CanonicalWriteBatch::new();
    for &(address, value) in writes {
        let range = address_space
            .resolve_range(
                address_space.address(address).unwrap(),
                4,
                MemoryPermissions::READ,
            )
            .unwrap();
        let mut offset = 0;
        let bytes = value.to_le_bytes();
        for segment in range.segments() {
            let end = offset + segment.size() as usize;
            batch
                .stage(
                    segment.mapping().backing(),
                    segment.backing_offset(),
                    &bytes[offset..end],
                )
                .unwrap();
            offset = end;
        }
    }
    batch
}

pub(super) fn translate_maxwell_shader_programs(
    state: &MaxwellThreeDState,
    address_space: &MaxwellGpuAddressSpace,
    staged_writes: &[(u64, u32)],
) -> Result<Vec<MaxwellTranslatedShaderProgram>, MaxwellShaderTranslationError> {
    let source = prepare_maxwell_shader_translation_source(state)?.materialize();
    let staged = canonical_shader_writes(address_space, staged_writes);
    let inputs =
        prepare_maxwell_shader_translation_inputs_from_source(&source, address_space, &staged)?;
    translate_prepared_maxwell_shader_programs(&inputs)
}

pub(super) fn preflight_maxwell_shader_translation(
    state: &MaxwellThreeDState,
    address_space: &MaxwellGpuAddressSpace,
    staged_writes: &[(u64, u32)],
) -> Result<(), MaxwellShaderTranslationError> {
    translate_maxwell_shader_programs(state, address_space, staged_writes).map(|_| ())
}

pub(super) fn mapped_memory() -> (CanonicalAllocation, MaxwellGpuAddressSpace, u64) {
    let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
    let mut address_space =
        MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
    address_space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    let mapping = address_space
        .map(MaxwellMapRequest {
            allocation: MaxwellAllocationId::new(1),
            backing: allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            backing_offset: 0,
            size: 0x1000,
            allocation_alignment: 0x1000,
            page_size: 0,
            kind: 0,
            cacheable: false,
            permissions: MemoryPermissions::READ_WRITE,
            fixed_offset: None,
        })
        .unwrap();
    (allocation, address_space, mapping.offset().get())
}

pub(super) fn program_three_d(channel: &mut MaxwellGpuChannel, method: u32, argument: u32) {
    let location = |word_offset| MaxwellGpfifoSourceLocation {
        channel: MaxwellChannelId::new(1),
        frontend: FrontendSubmissionId::new(2),
        entry_index: 0,
        pushbuffer: GpuVirtualAddress::try_new(0x8000, 40).unwrap(),
        word_offset,
        mapping: MaxwellMappingId::new(1),
        generation: MappingGeneration::new(1),
    };
    let packet = decode_maxwell_pushbuffer([
        Ok(MaxwellPushbufferWord::new(
            (1 << 29) | (1 << 16) | (method / 4),
            location(0),
        )),
        Ok(MaxwellPushbufferWord::new(argument, location(1))),
    ])
    .unwrap();
    dispatch_maxwell_engine_packet(channel, FrontendSubmissionId::new(2), &packet.packets()[0])
        .unwrap();
}

pub(super) fn translated_fixture(
    stage: MaxwellShaderStage,
    header_words: [u32; 20],
    code_words: &[u64],
) -> VerifiedShaderIr {
    translated_fixture_with_register_count(stage, header_words, code_words, 4)
}

pub(super) fn translated_fixture_with_register_count(
    stage: MaxwellShaderStage,
    header_words: [u32; 20],
    code_words: &[u64],
    register_count: u8,
) -> VerifiedShaderIr {
    let (allocation, address_space, address) = mapped_memory();
    let mut bytes = header_words
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    bytes.extend(code_words.iter().flat_map(|word| word.to_le_bytes()));
    allocation.write(0, &bytes).unwrap();
    let writes = CanonicalWriteBatch::new();
    let memory = MaxwellShaderMemoryView::new(&address_space, &writes);
    let binary = read_shader_binary(&memory, stage, address).unwrap();
    validate_program_header(stage, binary.header()).unwrap();
    VerifiedShaderIr::verify(
        translate_shader_binary(&binary, register_count, &BTreeMap::new())
            .unwrap()
            .ir,
    )
    .unwrap()
}

pub(super) fn validate_wgsl(module: &nixe_gpu::WgslShaderModule) {
    let parsed = naga::front::wgsl::parse_str(module.source()).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&parsed)
    .unwrap();
}
