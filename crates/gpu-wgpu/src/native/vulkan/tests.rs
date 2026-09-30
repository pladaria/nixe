use super::*;

#[path = "numerical_tests.rs"]
mod numerical;

// The ignored numerical and lifetime tests share the test-only device counter
// and process-wide validation logger. Keep them independent of test scheduling.
static NATIVE_DEVICE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn native_features_preserve_hal_requirements_and_do_not_invent_support() {
    let required = vk::PhysicalDeviceFeatures::default()
        .robust_buffer_access(true)
        .shader_int64(true)
        .independent_blend(true);
    for supported in [false, true] {
        let enabled = enabled_core(
            required,
            vk::PhysicalDeviceFeatures::default()
                .tessellation_shader(supported)
                .shader_float64(supported)
                .full_draw_index_uint32(supported)
                .fill_mode_non_solid(supported)
                .wide_lines(supported),
        );
        assert_eq!(enabled.tessellation_shader != 0, supported);
        assert_eq!(enabled.shader_float64 != 0, supported);
        assert_eq!(enabled.full_draw_index_uint32 != 0, supported);
        assert_eq!(enabled.robust_buffer_access, required.robust_buffer_access);
        assert_eq!(enabled.shader_int64, required.shader_int64);
        assert_eq!(enabled.independent_blend, required.independent_blend);
        assert_eq!(enabled.geometry_shader, 0);
        assert_eq!(enabled.fill_mode_non_solid != 0, supported);
        assert_eq!(enabled.wide_lines != 0, supported);
    }
    let mut extensions = vec![ash::khr::swapchain::NAME];
    add_extension(&mut extensions, fma::NAME);
    add_extension(&mut extensions, fma::NAME);
    assert_eq!(extensions, [ash::khr::swapchain::NAME, fma::NAME]);
}

#[test]
fn fma_extension_has_c_layout_and_preserves_the_feature_chain() {
    use std::mem::offset_of;
    assert_eq!(offset_of!(fma::Features<'_>, s_type), 0);
    assert_eq!(
        offset_of!(fma::Features<'_>, p_next),
        offset_of!(vk::BaseOutStructure<'_>, p_next)
    );
    assert_eq!(
        offset_of!(fma::Features<'_>, float16),
        offset_of!(
            vk::PhysicalDeviceShaderFloat16Int8Features<'_>,
            shader_float16
        )
    );
    assert_eq!(
        offset_of!(fma::Features<'_>, float32),
        offset_of!(fma::Features<'_>, float16) + 4
    );
    assert_eq!(
        offset_of!(fma::Features<'_>, float64),
        offset_of!(fma::Features<'_>, float32) + 4
    );
    let mut fma = fma::Features::default();
    assert_eq!(fma.s_type.as_raw(), 1_000_579_000);
    assert_eq!([fma.float16, fma.float32, fma.float64], [0; 3]);
    let mut tail = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let tail_address = std::ptr::from_mut(&mut tail).cast();
    let fma_address = std::ptr::from_mut(&mut fma).cast();
    let query = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut tail)
        .push_next(&mut fma);
    assert_eq!(query.p_next, fma_address);
    assert_eq!(fma.p_next, tail_address);
}

#[test]
fn native_limits_keep_distinct_stage_and_patch_cardinalities() {
    let result = limits(vk::PhysicalDeviceLimits {
        max_tessellation_generation_level: 64,
        max_tessellation_patch_size: 32,
        max_tessellation_control_per_vertex_input_components: 128,
        max_tessellation_control_per_vertex_output_components: 120,
        max_tessellation_control_per_patch_output_components: 110,
        max_tessellation_control_total_output_components: 4096,
        max_tessellation_evaluation_input_components: 112,
        max_tessellation_evaluation_output_components: 124,
        ..Default::default()
    });
    assert_eq!(
        result,
        VulkanTessellationLimits {
            generation_level: 64,
            patch_size: 32,
            control_per_vertex_input_components: 128,
            control_per_vertex_output_components: 120,
            control_per_patch_output_components: 110,
            control_total_output_components: 4096,
            evaluation_input_components: 112,
            evaluation_output_components: 124,
        }
    );
}

#[test]
#[ignore = "requires a Vulkan 1.1 adapter; verifies real imported-device lifetime"]
fn imported_device_drops_after_queue_and_resources_without_an_ownership_cycle() {
    let _guard = NATIVE_DEVICE_TEST_LOCK.lock().unwrap();
    use std::sync::atomic::Ordering;
    assert_eq!(LIVE_IMPORTED_DEVICES.load(Ordering::SeqCst), 0);
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let Some(adapter) = crate::test_hardware::adapter(&instance, wgpu::Backends::VULKAN) else {
        return;
    };
    let created = create_device(&instance, &adapter, &Default::default())
        .unwrap()
        .unwrap();
    eprintln!(
        "native adapter: {:?}; enabled: {:?}",
        adapter.get_info(),
        created.capabilities
    );
    assert_eq!(LIVE_IMPORTED_DEVICES.load(Ordering::SeqCst), 1);
    let buffer = created.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = created.device.create_command_encoder(&Default::default());
    encoder.clear_buffer(&buffer, 0, None);
    created.queue.submit([encoder.finish()]);
    created
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    drop(adapter);
    drop(instance);
    drop(created.queue);
    drop(created.device);
    assert_eq!(LIVE_IMPORTED_DEVICES.load(Ordering::SeqCst), 1);
    drop(buffer);
    assert_eq!(LIVE_IMPORTED_DEVICES.load(Ordering::SeqCst), 0);
}
