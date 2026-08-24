use nixe_gpu::{
    BackendCapabilities, BackendFeatures, BackendInstanceId, BackendLimits, BackendResourceHandle,
    FrontendSubmissionId, GpuVirtualAddress, ImageFormat, MappingGeneration, QueryKind,
    ResourceDependency, SampleCount, ShaderStage,
};
use nixe_gpu_headless::backend as headless_backend;
use nixe_gpu_maxwell::{
    MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
    MaxwellChannelId, MaxwellChannelOwner, MaxwellDecodedPushbuffer, MaxwellGpfifoSourceLocation,
    MaxwellGpuAddressSpace, MaxwellGpuChannel, MaxwellGpuMapping, MaxwellMapRequest,
    MaxwellMappingId, MaxwellPushbufferWord, MaxwellSubmissionExecutionPlan,
    MaxwellSubmissionExecutionStep, MaxwellThreeDLoweringCache, SWITCH_1_GM20B_PROFILE,
    decode_maxwell_pushbuffer, lower_maxwell_pushbuffer,
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
    address_space: &MaxwellGpuAddressSpace,
    methods: &[(u32, u32)],
) -> MaxwellSubmissionExecutionPlan {
    lower_maxwell_pushbuffer(
        &method_stream(methods),
        channel,
        address_space,
        FRONTEND_STATE,
        Vec::new(),
        None,
        &mut MaxwellThreeDLoweringCache::default(),
    )
    .unwrap()
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

#[test]
fn synthetic_maxwell_clear_executes_through_headless_contract() {
    let target_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
    let mut address_space = address_space();
    let target_mapping = map_resource(
        &mut address_space,
        target_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap(),
        32,
        0xfe,
    );

    let mut channel = channel();
    let plan = dispatch_stream(
        &mut channel,
        &address_space,
        &clear_stream(target_mapping.offset().get()),
    );
    let [MaxwellSubmissionExecutionStep::ThreeD(clear)] = plan.steps() else {
        panic!("clear stream did not lower to one 3D work item");
    };
    let capabilities = capabilities();
    let (mut backend, completion) =
        headless_backend(BackendInstanceId::new(41), capabilities.clone());
    let mut handles = Vec::<(ResourceDependency, BackendResourceHandle)>::new();
    for creation in clear.resource_creations() {
        let dependency = creation.dependency();
        let handle = backend.create_resource(creation.clone()).unwrap();
        handles.push((dependency, handle));
    }
    let clear_token = backend.submit(clear.submission()).unwrap();

    assert!(!backend.has_completed(clear_token).unwrap());
    completion.complete(clear_token).unwrap();
    backend.release_submission(clear_token).unwrap();

    for (_, handle) in handles.into_iter().rev() {
        backend.destroy_resource(handle).unwrap();
    }
    assert_eq!(backend.driver().resource_count(), 0);
    backend.teardown().unwrap();
}
