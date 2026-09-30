use super::*;
use nixe_gpu::ShaderResourceAccess;

#[path = "residency_tests.rs"]
mod residency;

fn limits() -> crate::VulkanGraphicsLimits {
    crate::VulkanGraphicsLimits {
        storage_buffers_per_stage: 4,
        resources_per_stage: 5,
        storage_buffers_per_set: 8,
        storage_buffer_range: 65536,
        vertex_input_attributes: 16,
        vertex_input_bindings: 16,
        vertex_input_attribute_offset: 2047,
        vertex_input_binding_stride: 2048,
        vertex_output_components: 64,
        fragment_input_components: 64,
    }
}

fn binding(number: u8, stages: PipelineStages) -> SpirvPipelineBinding {
    SpirvPipelineBinding {
        resource: ShaderResourceAccess::new(
            number,
            ShaderResourceKind::ConstantBuffer,
            true,
            false,
        )
        .unwrap(),
        stages,
    }
}

#[test]
fn native_descriptor_layout_keeps_sparse_binding_numbers_and_exact_visibility() {
    let bindings = [
        binding(7, PipelineStages::VERTEX_SHADER),
        binding(
            253,
            PipelineStages::TESSELLATION_CONTROL_SHADER
                .union(PipelineStages::TESSELLATION_EVALUATION_SHADER)
                .union(PipelineStages::FRAGMENT_SHADER),
        ),
    ];
    let (layout, stages) = layout(&bindings, limits()).unwrap();
    assert_eq!(layout.len(), 2);
    assert_eq!(layout[0].binding, 7);
    assert_eq!(layout[0].stage_flags, vk::ShaderStageFlags::VERTEX);
    assert_eq!(layout[1].binding, 253);
    assert_eq!(layout[1].descriptor_count, 1);
    assert_eq!(
        layout[1].descriptor_type,
        vk::DescriptorType::STORAGE_BUFFER
    );
    assert_eq!(
        layout[1].stage_flags,
        vk::ShaderStageFlags::TESSELLATION_CONTROL
            | vk::ShaderStageFlags::TESSELLATION_EVALUATION
            | vk::ShaderStageFlags::FRAGMENT
    );
    assert_eq!(
        stages,
        vk::PipelineStageFlags::VERTEX_SHADER
            | vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER
            | vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER
            | vk::PipelineStageFlags::FRAGMENT_SHADER
    );
    let (empty, stages) = super::layout(&[], limits()).unwrap();
    assert!(empty.is_empty());
    assert!(stages.is_empty());
}

#[test]
fn native_descriptor_limits_count_per_stage_and_fragment_attachment() {
    let mut limits = limits();
    limits.storage_buffers_per_stage = 1;
    limits.resources_per_stage = 1;
    let vertex = binding(0, PipelineStages::VERTEX_SHADER);
    let fragment = binding(1, PipelineStages::FRAGMENT_SHADER);
    assert!(layout(&[vertex], limits).is_ok());
    assert!(layout(&[fragment], limits).is_err()); // Attachment counts too.
    limits.resources_per_stage = 2;
    assert!(layout(&[vertex, fragment], limits).is_ok()); // Not a global stage count.
    let second_vertex = binding(2, PipelineStages::VERTEX_SHADER);
    assert!(layout(&[vertex, second_vertex], limits).is_err());
    limits.storage_buffers_per_set = 1;
    assert!(layout(&[vertex, fragment], limits).is_err());
}

#[test]
fn native_descriptor_layout_rejects_unimplemented_resources_and_visibility() {
    for (kind, write) in [
        (ShaderResourceKind::ConstantBuffer, true),
        (ShaderResourceKind::StorageBuffer, false),
        (ShaderResourceKind::SampledImage, false),
        (ShaderResourceKind::StorageImage, true),
    ] {
        let binding = SpirvPipelineBinding {
            resource: ShaderResourceAccess::new(0, kind, true, write).unwrap(),
            stages: PipelineStages::FRAGMENT_SHADER,
        };
        assert!(layout(&[binding], limits()).is_err());
    }
    assert!(layout(&[binding(0, PipelineStages::COMPUTE_SHADER)], limits()).is_err());
}

#[test]
fn native_descriptor_cache_checks_full_identity_within_fingerprint_bucket() {
    let pipeline = Arc::new(7);
    let equal_value_distinct_pipeline = Arc::new(7);
    let bucket = [
        (Arc::clone(&equal_value_distinct_pipeline), [11, 13]),
        (Arc::clone(&pipeline), [11, 17]),
        (Arc::clone(&pipeline), [11, 13]),
    ];
    let match_index = bucket.iter().position(|(cached_pipeline, cached_buffers)| {
        same_descriptor_identity(cached_pipeline, &pipeline, cached_buffers, &[11, 13])
    });
    assert_eq!(match_index, Some(2));
    assert!(!same_descriptor_identity(
        &equal_value_distinct_pipeline,
        &pipeline,
        &[11, 13],
        &[11, 13],
    ));
    assert!(!same_descriptor_identity(
        &pipeline,
        &pipeline,
        &[11, 17],
        &[11, 13],
    ));
}
