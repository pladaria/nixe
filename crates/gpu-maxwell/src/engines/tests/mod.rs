use nixe_gpu::{
    BackendCapabilities, BackendFeatures, BackendLimits, GpuCommand, GpuVirtualAddress,
    GuestSyncpointId, GuestSyncpointValue, GuestTimeline, ImageFormat, MappingGeneration,
    QueryKind, RenderPassOperation, SampleCount, ShaderId, ShaderStage, TimelineInstanceId,
    TimelineOwnerId,
};
use nixe_memory::{CanonicalAllocation, CanonicalBackingRange, MemoryPermissions};

use super::*;
use crate::{
    MaxwellAamVersion, MaxwellAamVersionRange, MaxwellAddressSpaceId,
    MaxwellAddressSpaceInitialization, MaxwellAllocationId, MaxwellChannelId, MaxwellChannelOwner,
    MaxwellGpfifoSourceLocation, MaxwellGpuAddressSpace, MaxwellGpuMapping, MaxwellMapRequest,
    MaxwellMappingId, MaxwellPushbufferWord, MaxwellShaderProgramHeaderVersion,
    SWITCH_1_GM20B_PROFILE, decode_maxwell_pushbuffer,
};

fn channel() -> MaxwellGpuChannel {
    MaxwellGpuChannel::new(
        MaxwellChannelId::new(7),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    )
}

fn word(value: u32, index: u32) -> Result<MaxwellPushbufferWord, crate::MaxwellGpfifoSourceError> {
    Ok(MaxwellPushbufferWord::new(
        value,
        MaxwellGpfifoSourceLocation {
            channel: MaxwellChannelId::new(7),
            frontend: FrontendSubmissionId::new(3),
            entry_index: 0,
            pushbuffer: GpuVirtualAddress::try_new(0x8000, 40).unwrap(),
            word_offset: u64::from(index),
            mapping: MaxwellMappingId::new(2),
            generation: MappingGeneration::new(1),
        },
    ))
}

fn packet(method_dword: u32, argument: u32) -> MaxwellDecodedPushbuffer {
    packet_on_subchannel(0, method_dword, argument)
}

fn packet_on_subchannel(
    subchannel: u32,
    method_dword: u32,
    argument: u32,
) -> MaxwellDecodedPushbuffer {
    decode_maxwell_pushbuffer([
        word((1 << 29) | (1 << 16) | (subchannel << 13) | method_dword, 0),
        word(argument, 1),
    ])
    .unwrap()
}

fn incrementing_packet(method_dword: u32, arguments: &[u32]) -> MaxwellDecodedPushbuffer {
    incrementing_packet_on_subchannel(0, method_dword, arguments)
}

fn increment_once_packet(method_dword: u32, arguments: &[u32]) -> MaxwellDecodedPushbuffer {
    increment_once_packet_on_subchannel(0, method_dword, arguments)
}

fn increment_once_packet_on_subchannel(
    subchannel: u32,
    method_dword: u32,
    arguments: &[u32],
) -> MaxwellDecodedPushbuffer {
    let mut words = Vec::with_capacity(arguments.len() + 1);
    words.push(word(
        (5 << 29) | ((arguments.len() as u32) << 16) | (subchannel << 13) | method_dword,
        0,
    ));
    words.extend(
        arguments
            .iter()
            .enumerate()
            .map(|(index, argument)| word(*argument, index as u32 + 1)),
    );
    decode_maxwell_pushbuffer(words).unwrap()
}

fn load_mme_program(channel: &mut MaxwellGpuChannel, macro_index: u8, code: &[u32]) {
    let start = 0x100 + u32::from(macro_index) * 0x10;
    let start_packet = incrementing_packet(0x011c / 4, &[u32::from(macro_index), start]);
    dispatch_maxwell_engine_packet(
        channel,
        FrontendSubmissionId::new(3),
        &start_packet.packets()[0],
    )
    .unwrap();
    let mut arguments = Vec::with_capacity(code.len() + 1);
    arguments.push(start);
    arguments.extend_from_slice(code);
    let instruction_packet = increment_once_packet(0x0114 / 4, &arguments);
    dispatch_maxwell_engine_packet(
        channel,
        FrontendSubmissionId::new(3),
        &instruction_packet.packets()[0],
    )
    .unwrap();
}

fn incrementing_packet_on_subchannel(
    subchannel: u32,
    method_dword: u32,
    arguments: &[u32],
) -> MaxwellDecodedPushbuffer {
    let mut words = Vec::with_capacity(arguments.len() + 1);
    words.push(word(
        (1 << 29) | ((arguments.len() as u32) << 16) | (subchannel << 13) | method_dword,
        0,
    ));
    words.extend(
        arguments
            .iter()
            .enumerate()
            .map(|(index, argument)| word(*argument, index as u32 + 1)),
    );
    decode_maxwell_pushbuffer(words).unwrap()
}

fn non_incrementing_packet_on_subchannel(
    subchannel: u32,
    method_dword: u32,
    arguments: &[u32],
) -> MaxwellDecodedPushbuffer {
    let mut words = Vec::with_capacity(arguments.len() + 1);
    words.push(word(
        (3 << 29) | ((arguments.len() as u32) << 16) | (subchannel << 13) | method_dword,
        0,
    ));
    words.extend(
        arguments
            .iter()
            .enumerate()
            .map(|(index, argument)| word(*argument, index as u32 + 1)),
    );
    decode_maxwell_pushbuffer(words).unwrap()
}

fn bind_two_d(channel: &mut MaxwellGpuChannel) {
    let decoded = packet_on_subchannel(3, 0, twod::CLASS.0);
    dispatch_maxwell_engine_packet(channel, FrontendSubmissionId::new(3), &decoded.packets()[0])
        .unwrap();
}

fn bind_compute(channel: &mut MaxwellGpuChannel) {
    let decoded = packet_on_subchannel(1, 0, compute::CLASS.0);
    dispatch_maxwell_engine_packet(channel, FrontendSubmissionId::new(3), &decoded.packets()[0])
        .unwrap();
}

fn bind_inline_to_memory(channel: &mut MaxwellGpuChannel) {
    let decoded = packet_on_subchannel(2, 0, inline_to_memory::CLASS.0);
    dispatch_maxwell_engine_packet(channel, FrontendSubmissionId::new(3), &decoded.packets()[0])
        .unwrap();
}

fn bind_three_d(channel: &mut MaxwellGpuChannel) {
    let decoded = packet(0, threed::CLASS.0);
    dispatch_maxwell_engine_packet(channel, FrontendSubmissionId::new(3), &decoded.packets()[0])
        .unwrap();
}

fn program_three_d(channel: &mut MaxwellGpuChannel, method: u32, argument: u32) {
    let decoded = packet(method / 4, argument);
    dispatch_maxwell_engine_packet(channel, FrontendSubmissionId::new(3), &decoded.packets()[0])
        .unwrap();
}

fn resource_address_space() -> MaxwellGpuAddressSpace {
    let mut address_space =
        MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(9), SWITCH_1_GM20B_PROFILE);
    address_space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    address_space
}

fn map_resource(
    address_space: &mut MaxwellGpuAddressSpace,
    backing: CanonicalBackingRange,
    allocation: u64,
    kind: u8,
) -> MaxwellGpuMapping {
    let size = backing.size();
    address_space
        .map(MaxwellMapRequest {
            allocation: MaxwellAllocationId::new(allocation),
            backing,
            backing_offset: 0,
            size,
            allocation_alignment: 0x1000,
            page_size: 0,
            kind,
            cacheable: true,
            permissions: MemoryPermissions::READ_WRITE,
            fixed_offset: None,
        })
        .unwrap()
}

fn lowering_capabilities(features: BackendFeatures) -> BackendCapabilities {
    BackendCapabilities::new(
        features,
        [ImageFormat::Rgba8Unorm, ImageFormat::Bgra8Unorm],
        [SampleCount::One],
        [ShaderStage::Vertex, ShaderStage::Fragment],
        std::iter::empty::<QueryKind>(),
        BackendLimits {
            max_color_attachments: 8,
            max_descriptor_bindings: 32,
            max_compute_workgroups: [1, 1, 1],
        },
    )
}

fn color_target_selection_raw(count: u8, targets: [u8; 8]) -> u32 {
    targets
        .into_iter()
        .enumerate()
        .fold(u32::from(count), |raw, (index, target)| {
            raw | (u32::from(target) << (4 + index * 3))
        })
}

fn program_color_target(channel: &mut MaxwellGpuChannel, target: u8, address: u64, format: u32) {
    let base = 0x0800 + u32::from(target) * 0x40;
    for (offset, argument) in [
        (0x00, (address >> 32) as u32),
        (0x04, address as u32),
        (0x08, 64),
        (0x0c, 32),
        (0x10, format),
        (0x14, 0),
        (0x18, 1),
        (0x1c, 0),
        (0x20, 0),
    ] {
        program_three_d(channel, base + offset, argument);
    }
}

fn program_basic_draw_state(channel: &mut MaxwellGpuChannel, vertex: u64) {
    for (method, argument) in [
        (0x1c00, 0x1010),
        (0x1c04, (vertex >> 32) as u32),
        (0x1c08, vertex as u32),
        (0x1f00, (vertex >> 32) as u32),
        (0x1f04, (vertex + 0xff) as u32),
        (0x1160, 0x3820_0000),
        (0x0d74, 0),
        (0x0308, 3),
        (0x1618, 4),
        (0x1970, 4),
        (0x12e4, 0),
        (0x135c, 0),
        (0x2000, 0x11),
        (0x2010, 0),
        (0x2040, 0x51),
        (0x2050, 1),
        (0x15d0, 0),
    ] {
        program_three_d(channel, method, argument);
    }
}

fn translated_graphics_shaders() -> (MaxwellThreeDTranslatedShaders, MaxwellThreeDLoweringCache) {
    let shaders = MaxwellThreeDTranslatedShaders::new(
        vec![
            MaxwellThreeDTranslatedShader::new(
                ShaderStage::Vertex,
                ShaderId::new(1),
                MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
                0,
            ),
            MaxwellThreeDTranslatedShader::new(
                ShaderStage::Fragment,
                ShaderId::new(2),
                MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
                0,
            ),
        ],
        Vec::new(),
    )
    .unwrap();
    let mut cache = MaxwellThreeDLoweringCache::default();
    cache.seed_test_shader_translations(&shaders);
    (shaders, cache)
}

mod graphics_pipeline;
mod raster_pipeline;
mod resources_compute;
mod state_validation;
