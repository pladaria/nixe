use nixe_gpu::{
    BackendCapabilities, BackendFeatures, BackendInstanceId, BackendLimits,
    BackendResourceCreateInfo, BackendResourceHandle, FrontendSubmissionId, GpuVirtualAddress,
    ImageFormat, MappingGeneration, QueryKind, ResourceDependency, SampleCount, ShaderDescription,
    ShaderId, ShaderStage,
};
use nixe_gpu_headless::backend as headless_backend;
use nixe_gpu_maxwell::{
    MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
    MaxwellChannelId, MaxwellChannelOwner, MaxwellDecodedPushbuffer, MaxwellGpfifoSourceLocation,
    MaxwellGpuAddressSpace, MaxwellGpuChannel, MaxwellGpuMapping, MaxwellMapRequest,
    MaxwellMappingId, MaxwellPushbufferWord, MaxwellThreeDDirectlyAddressableMemory,
    MaxwellThreeDLoweringCache, MaxwellThreeDLoweringError, MaxwellThreeDTranslatedShader,
    MaxwellThreeDTranslatedShaders, MaxwellThreeDTriggeredOperation, SWITCH_1_GM20B_PROFILE,
    decode_maxwell_pushbuffer, dispatch_maxwell_engine_pushbuffer,
    preflight_maxwell_three_d_operation, resolve_maxwell_three_d_resources,
};
use nixe_memory::{CanonicalAllocation, CanonicalBackingRange, MemoryPermissions};

const CHANNEL_ID: MaxwellChannelId = MaxwellChannelId::new(7);
const FRONTEND_STATE: FrontendSubmissionId = FrontendSubmissionId::new(3);

fn capabilities() -> BackendCapabilities {
    BackendCapabilities::new(
        BackendFeatures::CLEAR
            .union(BackendFeatures::DRAW)
            .union(BackendFeatures::BARRIER)
            .union(BackendFeatures::RENDER_PASS),
        [ImageFormat::Rgba8Unorm],
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

fn channel() -> MaxwellGpuChannel {
    MaxwellGpuChannel::new(
        CHANNEL_ID,
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    )
}

fn address_space() -> MaxwellGpuAddressSpace {
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

fn method_stream(methods: &[(u32, u32)]) -> MaxwellDecodedPushbuffer {
    let mut words = Vec::with_capacity(methods.len() * 2);
    for (packet, &(method, argument)) in methods.iter().enumerate() {
        let word_offset = u64::try_from(packet * 2).unwrap();
        for (relative, value) in [((1 << 29) | (1 << 16) | (method / 4)), argument]
            .into_iter()
            .enumerate()
        {
            words.push(Ok(MaxwellPushbufferWord::new(
                value,
                MaxwellGpfifoSourceLocation {
                    channel: CHANNEL_ID,
                    frontend: FRONTEND_STATE,
                    entry_index: 0,
                    pushbuffer: GpuVirtualAddress::try_new(0x8000, 40).unwrap(),
                    word_offset: word_offset + u64::try_from(relative).unwrap(),
                    mapping: MaxwellMappingId::new(2),
                    generation: MappingGeneration::new(1),
                },
            )));
        }
    }
    decode_maxwell_pushbuffer(words).unwrap()
}

fn dispatch_stream(
    channel: &mut MaxwellGpuChannel,
    methods: &[(u32, u32)],
) -> Vec<MaxwellThreeDTriggeredOperation> {
    dispatch_maxwell_engine_pushbuffer(channel, FRONTEND_STATE, &method_stream(methods))
        .unwrap()
        .iter()
        .flat_map(|packet| packet.operations().iter().cloned())
        .collect()
}

fn clear_stream(target: u64) -> Vec<(u32, u32)> {
    vec![
        (0, SWITCH_1_GM20B_PROFILE.classes().three_d().0),
        (0x0800, (target >> 32) as u32),
        (0x0804, target as u32),
        (0x0808, 64),
        (0x080c, 32),
        (0x0810, 0xd5),
        (0x0814, 0),
        (0x0818, 1),
        (0x081c, 0),
        (0x0820, 0),
        (0x15d0, 0),
        (0x0d6c, 64 << 16),
        (0x0d70, 32 << 16),
        (0x0d80, 0x3f80_0000),
        (0x0d84, 0x3f00_0000),
        (0x0d88, 0),
        (0x0d8c, 0x3f80_0000),
        (0x19d0, 0x3c),
    ]
}

fn draw_stream(vertex: u64) -> Vec<(u32, u32)> {
    vec![
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
        (0x2000, 0x11),
        (0x2010, 0),
        (0x2040, 0x51),
        (0x2050, 1),
        (0x12e4, 0),
        (0x135c, 0),
        (0x121c, 1),
        (0x0d78, 3),
    ]
}

#[test]
fn synthetic_maxwell_clear_and_draw_execute_through_headless_contract() {
    let vertex_allocation = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
    let target_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let mut address_space = address_space();
    let vertex_mapping = map_resource(
        &mut address_space,
        vertex_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        31,
        0,
    );
    let target_mapping = map_resource(
        &mut address_space,
        target_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        32,
        0xfe,
    );

    let mut channel = channel();
    let clear = dispatch_stream(&mut channel, &clear_stream(target_mapping.offset().get()));
    assert_eq!(clear.len(), 1);
    let draw = dispatch_stream(&mut channel, &draw_stream(vertex_mapping.offset().get()));
    assert_eq!(draw.len(), 1);

    let translated = MaxwellThreeDTranslatedShaders::new(
        vec![
            MaxwellThreeDTranslatedShader::new(
                ShaderStage::Vertex,
                ShaderId::new(1),
                1,
                MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
            ),
            MaxwellThreeDTranslatedShader::new(
                ShaderStage::Fragment,
                ShaderId::new(2),
                1,
                MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
            ),
        ],
        Vec::new(),
    )
    .unwrap();
    let capabilities = capabilities();
    let (mut backend, completion) =
        headless_backend(BackendInstanceId::new(41), capabilities.clone());
    let mut handles = Vec::<(ResourceDependency, BackendResourceHandle)>::new();
    for (shader, stage) in [
        (ShaderId::new(1), ShaderStage::Vertex),
        (ShaderId::new(2), ShaderStage::Fragment),
    ] {
        let creation = BackendResourceCreateInfo::Shader {
            id: shader,
            description: ShaderDescription { stage },
        };
        let dependency = creation.dependency();
        let handle = backend.create_resource(creation).unwrap();
        handles.push((dependency, handle));
    }

    let mut cache = MaxwellThreeDLoweringCache::default();
    let clear_resources =
        resolve_maxwell_three_d_resources(clear[0].state(), &address_space).unwrap();
    let clear_plan = preflight_maxwell_three_d_operation(
        clear[0].state(),
        &clear_resources,
        clear[0].trigger(),
        None,
        FrontendSubmissionId::new(10),
        Vec::new(),
        &capabilities,
        &cache,
    )
    .unwrap();
    for creation in clear_plan.resource_creations() {
        let dependency = creation.dependency();
        let handle = backend.create_resource(creation.clone()).unwrap();
        handles.push((dependency, handle));
    }
    let clear_token = backend.submit(clear_plan.submission()).unwrap();
    clear_plan.commit_cache(&mut cache).unwrap();

    let draw_resources =
        resolve_maxwell_three_d_resources(draw[0].state(), &address_space).unwrap();
    let resources_before_rejection = backend.driver().resource_count();
    let missing_shader_plan = preflight_maxwell_three_d_operation(
        draw[0].state(),
        &draw_resources,
        draw[0].trigger(),
        None,
        FrontendSubmissionId::new(11),
        vec![FrontendSubmissionId::new(10)],
        &capabilities,
        &cache,
    );
    assert!(matches!(
        missing_shader_plan,
        Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
    ));
    assert_eq!(
        backend.driver().resource_count(),
        resources_before_rejection
    );

    let draw_plan = preflight_maxwell_three_d_operation(
        draw[0].state(),
        &draw_resources,
        draw[0].trigger(),
        Some(&translated),
        FrontendSubmissionId::new(11),
        vec![FrontendSubmissionId::new(10)],
        &capabilities,
        &cache,
    )
    .unwrap();
    for creation in draw_plan.resource_creations() {
        let dependency = creation.dependency();
        let handle = backend.create_resource(creation.clone()).unwrap();
        handles.push((dependency, handle));
    }
    let draw_token = backend.submit(draw_plan.submission()).unwrap();
    draw_plan.commit_cache(&mut cache).unwrap();

    assert!(backend.driver().resource_count() > resources_before_rejection);
    assert_eq!(backend.driver().submission_count(), 2);
    assert!(!backend.has_completed(clear_token).unwrap());
    assert!(!backend.has_completed(draw_token).unwrap());
    completion.complete(clear_token).unwrap();
    completion.complete(draw_token).unwrap();
    backend.release_submission(clear_token).unwrap();
    backend.release_submission(draw_token).unwrap();

    for (_, handle) in handles.into_iter().rev() {
        backend.destroy_resource(handle).unwrap();
    }
    assert_eq!(backend.driver().resource_count(), 0);
    assert_eq!(backend.driver().submission_count(), 0);
    backend.teardown().unwrap();
}
