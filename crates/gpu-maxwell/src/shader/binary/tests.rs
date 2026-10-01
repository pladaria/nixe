use super::super::error::MaxwellShaderTranslationError;
use super::super::test_support::{
    mapped_memory, preflight_maxwell_shader_translation, program_three_d,
};
use super::*;
use crate::{
    MaxwellChannelId, MaxwellChannelOwner, MaxwellGpuChannel, MaxwellShaderStage,
    SWITCH_1_GM20B_PROFILE,
};
use nixe_memory::CanonicalWriteBatch;

#[test]
fn shader_memory_view_overlays_ordered_submission_writes_without_publication() {
    let (allocation, address_space, address) = mapped_memory();
    allocation
        .write(0, &[0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80])
        .unwrap();
    let writes = [
        MaxwellStagedShaderWrite::new(address + 2, 0xaabb_ccdd),
        MaxwellStagedShaderWrite::new(address + 4, 0x1122_3344),
    ];
    let writes =
        canonical_shader_writes(&address_space, &writes, MaxwellShaderStage::Vertex).unwrap();
    let bytes = MaxwellShaderMemoryView::new(&address_space, &writes)
        .read(MaxwellShaderStage::Vertex, address, 8)
        .unwrap()
        .bytes;

    assert_eq!(bytes, [0x10, 0x20, 0xdd, 0xcc, 0x44, 0x33, 0x22, 0x11]);
    let mut canonical = [0_u8; 8];
    allocation.read(0, &mut canonical).unwrap();
    assert_eq!(canonical, [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80]);
}

#[test]
fn version_three_vertex_and_pixel_headers_decode_from_public_field_layout() {
    let mut vertex = [0_u8; MAXWELL_SHADER_PROGRAM_HEADER_SIZE];
    vertex[..4].copy_from_slice(&0x0002_0461_u32.to_le_bytes());
    let decoded = decode_program_header(&vertex).unwrap();
    assert_eq!(decoded.sph_type(), 1);
    assert_eq!(decoded.version(), 3);
    assert_eq!(decoded.stage(), MaxwellShaderStage::Vertex);
    assert_eq!(decoded.sass_version(), 1);
    validate_program_header(MaxwellShaderStage::Vertex, decoded).unwrap();

    let mut pixel = [0_u8; MAXWELL_SHADER_PROGRAM_HEADER_SIZE];
    pixel[..4].copy_from_slice(&0x0002_5462_u32.to_le_bytes());
    let decoded = decode_program_header(&pixel).unwrap();
    assert_eq!(decoded.sph_type(), 2);
    assert_eq!(decoded.version(), 3);
    assert_eq!(decoded.stage(), MaxwellShaderStage::Pixel);
    assert_eq!(decoded.sass_version(), 1);
    validate_program_header(MaxwellShaderStage::Pixel, decoded).unwrap();
}

#[test]
fn header_validation_rejects_unverified_sass_versions() {
    for version in [0_u32, 2, 4, 15] {
        let mut bytes = [0_u8; MAXWELL_SHADER_PROGRAM_HEADER_SIZE];
        let common = 0x0000_0461_u32 | (version << 17);
        bytes[..4].copy_from_slice(&common.to_le_bytes());
        let header = decode_program_header(&bytes).unwrap();

        assert_eq!(
            validate_program_header(MaxwellShaderStage::Vertex, header),
            Err(MaxwellShaderTranslationError::UnsupportedSassVersion {
                stage: MaxwellShaderStage::Vertex,
                version: version as u8,
            })
        );
    }
}

#[test]
fn header_validation_rejects_stage_contradictions() {
    let mut bytes = [0_u8; MAXWELL_SHADER_PROGRAM_HEADER_SIZE];
    bytes[..4].copy_from_slice(&0x0002_0461_u32.to_le_bytes());
    let header = decode_program_header(&bytes).unwrap();
    assert_eq!(
        validate_program_header(MaxwellShaderStage::Pixel, header),
        Err(MaxwellShaderTranslationError::InvalidHeaderType {
            stage: MaxwellShaderStage::Pixel,
            sph_type: 1,
        })
    );
}

#[test]
fn header_validation_rejects_unimplemented_semantic_flags() {
    for (bit, feature) in [
        (15, "pixel-kill"),
        (16, "global-store"),
        (26, "memory-load/store"),
        (27, "FP64"),
        (28, "stream-output"),
    ] {
        let mut bytes = [0_u8; MAXWELL_SHADER_PROGRAM_HEADER_SIZE];
        let common = 0x0002_5462_u32 | (1 << bit);
        bytes[..4].copy_from_slice(&common.to_le_bytes());
        let header = decode_program_header(&bytes).unwrap();
        assert_eq!(
            validate_program_header(MaxwellShaderStage::Pixel, header),
            Err(MaxwellShaderTranslationError::UnsupportedHeaderFeature {
                stage: MaxwellShaderStage::Pixel,
                feature,
            })
        );
    }
}

#[test]
fn shader_reads_are_bounded_before_address_space_access() {
    let (_, address_space, address) = mapped_memory();
    assert!(matches!(
        MaxwellShaderMemoryView::new(&address_space, &CanonicalWriteBatch::new()).read(
            MaxwellShaderStage::Vertex,
            address,
            MAXWELL_SHADER_READ_LIMIT + 1,
        ),
        Err(MaxwellShaderTranslationError::ReadTooLarge { requested, limit })
            if requested == MAXWELL_SHADER_READ_LIMIT + 1
                && limit == MAXWELL_SHADER_READ_LIMIT
    ));
}

#[test]
fn shader_fetches_cannot_escape_the_bound_executable_program_range() {
    let (_, address_space, address) = mapped_memory();
    let writes = CanonicalWriteBatch::new();
    let memory = MaxwellShaderMemoryView::new(&address_space, &writes);
    let executable =
        MaxwellShaderExecutableRange::new(MaxwellShaderStage::Vertex, address).unwrap();
    assert!(matches!(
        memory.read_executable(
            MaxwellShaderStage::Vertex,
            executable,
            address + MAXWELL_SHADER_READ_LIMIT as u64 - 4,
            8,
        ),
        Err(MaxwellShaderTranslationError::ReadOutsideExecutableRange {
            stage: MaxwellShaderStage::Vertex,
            size: 8,
            ..
        })
    ));
}

#[test]
fn staged_header_and_code_reach_the_precise_first_instruction_boundary() {
    let (_, address_space, address) = mapped_memory();
    let mut channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(1),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    program_three_d(
        &mut channel,
        0,
        SWITCH_1_GM20B_PROFILE.classes().three_d().0,
    );
    program_three_d(&mut channel, 0x1608, (address >> 32) as u32);
    program_three_d(&mut channel, 0x160c, address as u32);
    program_three_d(&mut channel, 0x2000, 0x11);
    program_three_d(&mut channel, 0x2004, 0);
    program_three_d(&mut channel, 0x200c, 4);

    let instruction = 0xf123_0000_0007_0000_u64;
    let exit = 0xe300_0000_0007_000f_u64;
    let writes = [
        MaxwellStagedShaderWrite::new(address, 0x0002_0461),
        MaxwellStagedShaderWrite::new(address + 88, instruction as u32),
        MaxwellStagedShaderWrite::new(address + 92, (instruction >> 32) as u32),
        MaxwellStagedShaderWrite::new(address + 104, exit as u32),
        MaxwellStagedShaderWrite::new(address + 108, (exit >> 32) as u32),
    ];
    assert_eq!(
        preflight_maxwell_shader_translation(channel.three_d(), &address_space, &writes),
        Err(MaxwellShaderTranslationError::UnsupportedInstruction {
            stage: MaxwellShaderStage::Vertex,
            program_address: address,
            instruction_offset: 8,
            encoding: instruction,
        })
    );
}
