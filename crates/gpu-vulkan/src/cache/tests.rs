use super::*;
use crate::*;
use ash::vk;
use nixe_gpu::{
    ShaderInterfaceElement, ShaderIoLocation, ShaderScalarType, ShaderSourceLocation, ShaderStage,
    ssa::artifact::{ContentId, ShaderNamespace},
    ssa::*,
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
mod support {
    use crate as nixe_gpu_vulkan;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/device.rs"
    ));
}
mod gate {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/gate.rs"
    ));
}
const NS: ShaderNamespace = ShaderNamespace {
    guest: *b"native-test-gpu1",
    isa: 1,
};
fn ready<T>(p: Pending<T>) -> Arc<T> {
    p.wait_timeout(Duration::from_secs(30))
        .expect("compiler publication")
        .unwrap()
}
fn pc() -> ShaderSourceLocation {
    ShaderSourceLocation::new(8)
}
fn c(b: &mut FunctionBuilder, block: BlockId, bits: u32) -> Operand {
    b.emit(block, pc(), Op::Constant(Constant::u32(bits)), vec![])[0].into()
}
fn kernel(increment: u32) -> Function {
    let mut b = FunctionBuilder::new(ShaderStage::Compute);
    let entry = b.entry();
    let zero = c(&mut b, entry, 0);
    let input = b.emit(
        entry,
        pc(),
        Op::Load {
            space: MemorySpace::Storage,
            resource: Some(ResourceId::new(42)),
            address_type: Type::U32,
            ty: Type::U32,
        },
        vec![zero],
    )[0];
    let increment = c(&mut b, entry, increment);
    let value = b.emit(
        entry,
        pc(),
        Op::Binary {
            op: Binary::Add,
            ty: Type::U32,
            float_control: nixe_gpu::ShaderFloatControl::PRECISE,
        },
        vec![input.into(), increment],
    )[0];
    let address = c(&mut b, entry, 4);
    b.emit(
        entry,
        pc(),
        Op::Store {
            space: MemorySpace::Storage,
            resource: Some(ResourceId::new(42)),
            address_type: Type::U32,
            ty: Type::U32,
        },
        vec![address, value.into()],
    );
    b.terminate(entry, pc(), Terminator::Return);
    let mut f = b.finish().unwrap();
    f.workgroup_size = Some([1, 1, 1]);
    f.optimize().unwrap();
    f
}
#[test]
fn persistent_payload_is_checked_and_bounded_before_consumption() {
    let root = tempfile::tempdir().unwrap();
    let store = persistent::Store::new(root.path().to_path_buf(), 256);
    let key = ContentId::of(b"test", &[b"one"]);
    store.write(key, b"valid-data").unwrap();
    assert_eq!(store.read(key).unwrap(), b"valid-data");
    let mut bytes = std::fs::read(store.path(key)).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    std::fs::write(store.path(key), bytes).unwrap();
    assert!(store.read(key).is_none());
    let other = ContentId::of(b"test", &[b"two"]);
    store.write(other, &[3; 512]).unwrap();
    assert!(store.read(other).is_none());
    store.write(key, b"rebuilt").unwrap();
    assert_eq!(store.read(key).unwrap(), b"rebuilt");
}
#[test]
fn native_cached_compute_deduplicates_persists_and_retains_in_flight_objects() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let root = tempfile::tempdir().unwrap();
    let config = CacheConfiguration {
        workers: 2,
        directory: Some(root.path().to_path_buf()),
        ..Default::default()
    };
    let pool = Arc::new(CompilePool::new(config.workers, config.pending_jobs).unwrap());
    let shaders = ShaderCompiler::new(NS, Arc::clone(&pool), &config);
    let pipelines = PipelineCompiler::new(&device, Arc::clone(&pool), &config).unwrap();
    let source = ContentId::of(b"test-source", &[b"increment7"]);
    let calls = Arc::new(AtomicUsize::new(0));
    let submit = || {
        let calls = Arc::clone(&calls);
        shaders
            .request(
                source,
                Box::new([device.shader_options()]),
                move || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(Box::new([kernel(7)]))
                },
                |fs| fs.iter().collect(),
            )
            .unwrap()
    };
    let shader = ready(submit());
    let duplicate = ready(submit());
    assert!(Arc::ptr_eq(&shader, &duplicate));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    drop(duplicate);
    let pipeline = ready(
        pipelines
            .request(PipelineRecipe::Compute(Arc::clone(&shader)))
            .unwrap(),
    );
    let duplicate = ready(
        pipelines
            .request(PipelineRecipe::Compute(Arc::clone(&shader)))
            .unwrap(),
    );
    assert!(Arc::ptr_eq(&pipeline, &duplicate));
    drop(duplicate);
    pipelines.checkpoint().unwrap();
    let buffer = Arc::new(
        device
            .allocate_buffer(
                8,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuToCpu,
                AllocationMode::Managed,
                "cached SSA compute",
            )
            .unwrap(),
    );
    unsafe { buffer.write_range(0, &[17u32.to_le_bytes(), 0u32.to_le_bytes()].concat()) }.unwrap();
    let mut descriptors = DescriptorCache::new(&device, 64);
    let resources = || {
        Box::new([DescriptorResource::Buffer {
            buffer: Arc::clone(&buffer),
            offset: 0,
            range: 8,
        }]) as Box<[DescriptorResource]>
    };
    let record = descriptors.get(&pipeline.layout, &resources()).unwrap();
    assert!(Arc::ptr_eq(
        &record,
        &descriptors.get(&pipeline.layout, &resources()).unwrap()
    ));
    device.bind_native_pipeline(&pipeline).unwrap();
    device
        .bind_native_descriptors(
            &pipeline,
            &record,
            &vec![0; pipeline.layout.dynamic_bindings.len()],
        )
        .unwrap();
    let counts = (Arc::strong_count(&pipeline), Arc::strong_count(&record));
    for _ in 0..100 {
        device.bind_native_pipeline(&pipeline).unwrap();
        device
            .bind_native_descriptors(
                &pipeline,
                &record,
                &vec![0; pipeline.layout.dynamic_bindings.len()],
            )
            .unwrap();
    }
    assert_eq!(
        counts,
        (Arc::strong_count(&pipeline), Arc::strong_count(&record))
    );
    assert_eq!(descriptors.statistics().updates, 1);
    let slot = device.submission.recording.unwrap();
    let command = device.submission.slots[slot].command;
    unsafe {
        device.owner.device.cmd_dispatch(command, 1, 1, 1);
        let barriers = [vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(buffer.handle)
            .size(8)];
        device.dispatch.barrier(
            command,
            &vk::DependencyInfo::default().buffer_memory_barriers(&barriers),
        );
    }
    let mut gate = gate::Gate::new(&device);
    let waits = [vk::SemaphoreSubmitInfo::default()
        .semaphore(gate.handle)
        .value(1)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
    let work = unsafe {
        device
            .submission
            .flush(device.queue, &device.dispatch, &waits, &[])
    }
    .unwrap()
    .unwrap();
    gate.last_use = work.tick();
    let weak_pipeline = Arc::downgrade(&pipeline);
    let weak_record = Arc::downgrade(&record);
    let weak_buffer = Arc::downgrade(&buffer);
    drop(pipeline);
    drop(record);
    drop(buffer);
    drop(descriptors);
    drop(pipelines);
    drop(shaders);
    assert!(weak_pipeline.upgrade().is_some());
    assert!(weak_record.upgrade().is_some());
    let buffer = weak_buffer.upgrade().expect("batch must keep buffer alive");
    assert!(!device.wait_for_work(work, Duration::ZERO).unwrap());
    gate.signal();
    assert!(device.wait_for_work(work, Duration::from_secs(10)).unwrap());
    assert!(weak_pipeline.upgrade().is_none());
    assert!(weak_record.upgrade().is_none());
    assert_eq!(
        unsafe { buffer.read_range(4, 4) }.unwrap(),
        24u32.to_le_bytes()
    );
    drop(buffer);
    assert!(weak_buffer.upgrade().is_none());
    let shaders = ShaderCompiler::new(NS, Arc::clone(&pool), &config);
    let warm = ready(
        shaders
            .request(
                source,
                Box::new([device.shader_options()]),
                || -> Result<Box<[Function]>, CompileError> {
                    panic!("persisted SPIR-V must bypass translation")
                },
                |fs| fs.iter().collect(),
            )
            .unwrap(),
    );
    assert!(warm.timings.disk_hit);
    let path = root.path().join("spirv-v1").join(format!(
        "{}.cache",
        nixe_gpu::ssa::artifact::ArtifactKey::new(
            NS,
            source,
            nixe_gpu::ssa::artifact::specialization(&[device.shader_options()])
        )
        .0
        .hex()
    ));
    let mut bytes = std::fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    std::fs::write(path, bytes).unwrap();
    let rebuilt_compiler = ShaderCompiler::new(NS, Arc::clone(&pool), &config);
    let rebuilt = ready(
        rebuilt_compiler
            .request(
                source,
                Box::new([device.shader_options()]),
                || Ok(Box::new([kernel(7)])),
                |fs| fs.iter().collect(),
            )
            .unwrap(),
    );
    assert!(!rebuilt.timings.disk_hit);
    assert_eq!(rebuilt.identity, warm.identity);
    drop(rebuilt_compiler);
    let different_namespace = ShaderNamespace { isa: 2, ..NS };
    let isolated = ShaderCompiler::new(different_namespace, Arc::clone(&pool), &config);
    let same_code = ready(
        isolated
            .request(
                source,
                Box::new([device.shader_options()]),
                || Ok(Box::new([kernel(7)])),
                |fs| fs.iter().collect(),
            )
            .unwrap(),
    );
    assert!(!same_code.timings.disk_hit);
    assert_eq!(same_code.identity, warm.identity);
    drop(isolated);
    let pipelines = PipelineCompiler::new(&device, Arc::clone(&pool), &config).unwrap();
    let restarted = ready(pipelines.request(PipelineRecipe::Compute(warm)).unwrap());
    let shared = ready(
        pipelines
            .request(PipelineRecipe::Compute(same_code))
            .unwrap(),
    );
    assert!(Arc::ptr_eq(&shared, &restarted));
    drop(shared);
    eprintln!(
        "native cache cold/warm driver feedback: {:?}",
        restarted.metrics
    );
    drop(restarted);
    drop(pipelines);
    drop(shaders);
    drop(pool);
    drop(gate);
    drop(device);
    validation.check().unwrap();
}

fn graphics_functions(seed: u32) -> Box<[Function]> {
    let mut b = FunctionBuilder::new(ShaderStage::Vertex);
    let entry = b.entry();
    let id = b.emit(
        entry,
        pc(),
        Op::Input {
            location: ShaderIoLocation::VertexId,
            component: 0,
            ty: Type::U32,
        },
        vec![],
    )[0];
    let mut positions = Vec::new();
    for index in [1, 2] {
        let index = c(&mut b, entry, index);
        let test = b.emit(
            entry,
            pc(),
            Op::IntegerCompare {
                ty: Type::U32,
                comparison: nixe_gpu::ShaderIntegerComparison::Equal,
            },
            vec![id.into(), index],
        )[0];
        let yes = b.emit(
            entry,
            pc(),
            Op::Constant(Constant {
                ty: Type::F32,
                bits: 3f32.to_bits() as u64,
            }),
            vec![],
        )[0];
        let no = b.emit(
            entry,
            pc(),
            Op::Constant(Constant {
                ty: Type::F32,
                bits: (-1f32).to_bits() as u64,
            }),
            vec![],
        )[0];
        positions.push(
            b.emit(
                entry,
                pc(),
                Op::Select(Type::F32),
                vec![test.into(), yes.into(), no.into()],
            )[0],
        );
    }
    for value in [0f32, 1.] {
        positions.push(
            b.emit(
                entry,
                pc(),
                Op::Constant(Constant {
                    ty: Type::F32,
                    bits: value.to_bits() as u64,
                }),
                vec![],
            )[0],
        );
    }
    for (component, &value) in positions.iter().enumerate() {
        b.emit(
            entry,
            pc(),
            Op::Output {
                location: ShaderIoLocation::Position,
                component: component as u32,
                ty: Type::F32,
            },
            vec![value.into()],
        );
    }
    b.terminate(entry, pc(), Terminator::Return);
    let mut vertex = b.finish().unwrap();
    vertex.inputs = vec![
        ShaderInterfaceElement::new(
            ShaderIoLocation::VertexId,
            0,
            ShaderScalarType::Unsigned32,
            None,
        )
        .unwrap(),
    ];
    vertex.outputs = (0..4)
        .map(|i| {
            ShaderInterfaceElement::new(
                ShaderIoLocation::Position,
                i,
                ShaderScalarType::Float32,
                None,
            )
            .unwrap()
        })
        .collect();
    vertex.optimize().unwrap();
    let mut b = FunctionBuilder::new(ShaderStage::Fragment);
    let entry = b.entry();
    for (component, bits) in [(seed + 1) as f32 / 16., 0.25, 0.5, 1.]
        .map(f32::to_bits)
        .into_iter()
        .enumerate()
    {
        let value = b.emit(
            entry,
            pc(),
            Op::Constant(Constant {
                ty: Type::F32,
                bits: bits as u64,
            }),
            vec![],
        )[0];
        b.emit(
            entry,
            pc(),
            Op::Output {
                location: ShaderIoLocation::Color(0),
                component: component as u32,
                ty: Type::F32,
            },
            vec![value.into()],
        );
    }
    b.terminate(entry, pc(), Terminator::Return);
    let mut fragment = b.finish().unwrap();
    fragment.outputs = (0..4)
        .map(|i| {
            ShaderInterfaceElement::new(
                ShaderIoLocation::Color(0),
                i,
                ShaderScalarType::Float32,
                None,
            )
            .unwrap()
        })
        .collect();
    fragment.optimize().unwrap();
    Box::new([vertex, fragment])
}
#[test]
fn native_graphics_cache_normalizes_dynamic_state_and_reuses_library_parts() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let config = CacheConfiguration {
        workers: 2,
        ..Default::default()
    };
    let pool = Arc::new(CompilePool::new(2, 32).unwrap());
    let shaders = ShaderCompiler::new(NS, Arc::clone(&pool), &config);
    let shader = ready(
        shaders
            .request(
                ContentId::of(b"gfx", &[&[0]]),
                Box::new([device.shader_options(); 2]),
                || Ok(graphics_functions(0)),
                |fs| fs.iter().collect(),
            )
            .unwrap(),
    );
    let pipelines = PipelineCompiler::new(&device, Arc::clone(&pool), &config).unwrap();
    let state = GraphicsState::color(vk::Format::R8G8B8A8_UNORM);
    let first = ready(
        pipelines
            .request(PipelineRecipe::Graphics {
                shader: Arc::clone(&shader),
                state: state.clone(),
            })
            .unwrap(),
    );
    if device.profile().dynamic_state.base {
        let mut changed = state.clone();
        changed.cull = vk::CullModeFlags::BACK;
        changed.front_face = vk::FrontFace::CLOCKWISE;
        changed.depth_test = true;
        changed.depth_write = true;
        changed.depth_compare = vk::CompareOp::LESS;
        let same = ready(
            pipelines
                .request(PipelineRecipe::Graphics {
                    shader: Arc::clone(&shader),
                    state: changed,
                })
                .unwrap(),
        );
        assert!(Arc::ptr_eq(&first, &same));
    }
    let mut changed = state;
    changed.colors[0].1.write_mask =
        vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B;
    let other = ready(
        pipelines
            .request(PipelineRecipe::Graphics {
                shader,
                state: changed,
            })
            .unwrap(),
    );
    if pipelines.strategy() == PipelineStrategy::Libraries {
        assert_eq!(first.metrics.libraries_created, 4);
        assert_eq!(other.metrics.libraries_created, 1);
    }
    drop(first);
    drop(other);
    drop(pipelines);
    drop(shaders);
    drop(pool);
    drop(device);
    validation.check().unwrap();
}

mod probe;

fn descriptor_pressure(api: ApiPreference) {
    let Some(mut device) = support::device_for_test(api) else {
        return;
    };
    let validation = device.validation_monitor();
    let config = CacheConfiguration {
        workers: 1,
        ..Default::default()
    };
    let pool = Arc::new(CompilePool::new(1, 8).unwrap());
    let shaders = ShaderCompiler::new(NS, Arc::clone(&pool), &config);
    let pipelines = PipelineCompiler::new(&device, Arc::clone(&pool), &config).unwrap();
    let shader = ready(
        shaders
            .request(
                ContentId::of(b"pressure", &[b"increment"]),
                Box::new([device.shader_options()]),
                || Ok(Box::new([kernel(1)])),
                |fs| fs.iter().collect(),
            )
            .unwrap(),
    );
    let pipeline = ready(pipelines.request(PipelineRecipe::Compute(shader)).unwrap());
    let alignment = device
        .profile()
        .limits
        .min_storage_buffer_offset_alignment
        .max(8);
    let buffer = Arc::new(
        device
            .allocate_buffer(
                alignment * 3,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuToCpu,
                AllocationMode::Managed,
                "descriptor pressure slices",
            )
            .unwrap(),
    );
    let mut cache = DescriptorCache::new(&device, 2);
    let resources = |offset| {
        Box::new([DescriptorResource::Buffer {
            buffer: Arc::clone(&buffer),
            offset,
            range: 8,
        }]) as Box<[DescriptorResource]>
    };
    let first = cache.get(&pipeline.layout, &resources(0)).unwrap();
    let second = cache.get(&pipeline.layout, &resources(alignment)).unwrap();
    device.bind_native_pipeline(&pipeline).unwrap();
    device
        .bind_native_descriptors(
            &pipeline,
            &first,
            &vec![0; pipeline.layout.dynamic_bindings.len()],
        )
        .unwrap();
    device
        .bind_native_descriptors(
            &pipeline,
            &second,
            &vec![0; pipeline.layout.dynamic_bindings.len()],
        )
        .unwrap();
    let mut gate = gate::Gate::new(&device);
    let waits = [vk::SemaphoreSubmitInfo::default()
        .semaphore(gate.handle)
        .value(1)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
    let work = unsafe {
        device
            .submission
            .flush(device.queue, &device.dispatch, &waits, &[])
    }
    .unwrap()
    .unwrap();
    gate.last_use = work.tick();
    drop(first);
    drop(second);
    assert!(matches!(
        cache.get(&pipeline.layout, &resources(alignment * 2)),
        Err(VulkanError::DescriptorPressure)
    ));
    gate.signal();
    assert!(device.wait_for_work(work, Duration::from_secs(10)).unwrap());
    let third = cache
        .get(&pipeline.layout, &resources(alignment * 2))
        .unwrap();
    assert_eq!(cache.statistics().resets, 1);
    device.bind_native_pipeline(&pipeline).unwrap();
    device
        .bind_native_descriptors(
            &pipeline,
            &third,
            &vec![0; pipeline.layout.dynamic_bindings.len()],
        )
        .unwrap();
    let next = device.flush().unwrap().unwrap();
    assert!(device.wait_for_work(next, Duration::from_secs(10)).unwrap());
    drop(third);
    drop(cache);
    drop(buffer);
    drop(pipeline);
    drop(pipelines);
    drop(shaders);
    drop(pool);
    drop(gate);
    drop(device);
    validation.check().unwrap();
}
#[test]
fn native_descriptor_pressure_recycles_only_after_real_completion() {
    descriptor_pressure(ApiPreference::Auto);
}
#[test]
fn native_cache_and_descriptor_retirement_support_vulkan12_aliases() {
    descriptor_pressure(ApiPreference::Compatibility12);
}

#[test]
fn native_sampler_cache_reuses_semantics_and_evicts_only_cache_ownership() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let mut cache = SamplerCache::new(&device, 1);
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST);
    let first = cache.get(&info).unwrap();
    let mut same = info;
    same.compare_op = vk::CompareOp::LESS;
    same.max_anisotropy = 2.;
    assert!(Arc::ptr_eq(&first, &cache.get(&same).unwrap()));
    let mut changed = info;
    changed.mag_filter = vk::Filter::LINEAR;
    let second = cache.get(&changed).unwrap();
    assert!(!Arc::ptr_eq(&first, &second));
    let weak = Arc::downgrade(&first);
    drop(cache);
    assert!(weak.upgrade().is_some());
    drop(first);
    assert!(weak.upgrade().is_none());
    drop(second);
    drop(device);
    validation.check().unwrap();
}

#[test]
fn dynamic_offsets_reuse_one_set_and_preserve_bounds_and_gpu_leases() {
    let Some(mut device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let validation = device.validation_monitor();
    let root = tempfile::tempdir().unwrap();
    let config = CacheConfiguration {
        directory: Some(root.path().to_path_buf()),
        ..Default::default()
    };
    let pool = Arc::new(CompilePool::new(config.workers, config.pending_jobs).unwrap());
    let shaders = ShaderCompiler::new(NS, Arc::clone(&pool), &config);
    let pipelines = PipelineCompiler::new(&device, Arc::clone(&pool), &config).unwrap();
    let shader = ready(
        shaders
            .request(
                ContentId::of(b"test-source", &[b"dynamic-increment7"]),
                Box::new([device.shader_options()]),
                || Ok(Box::new([kernel(7)])),
                |fs| fs.iter().collect(),
            )
            .unwrap(),
    );
    let pipeline = ready(pipelines.request(PipelineRecipe::Compute(shader)).unwrap());
    assert_eq!(&*pipeline.layout.dynamic_bindings, &[0]);
    let alignment = device
        .profile()
        .limits
        .min_storage_buffer_offset_alignment
        .max(8);
    let buffer = Arc::new(
        device
            .allocate_buffer(
                alignment + 8,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuToCpu,
                AllocationMode::Managed,
                "dynamic descriptor windows",
            )
            .unwrap(),
    );
    unsafe {
        buffer.write_range(0, &17u32.to_le_bytes()).unwrap();
        buffer.write_range(alignment, &29u32.to_le_bytes()).unwrap();
    }
    let mut descriptors = DescriptorCache::new(&device, 64);
    let mut resources = [DescriptorResource::Buffer {
        buffer: Arc::clone(&buffer),
        offset: 0,
        range: 8,
    }];
    let first_offsets = pipeline
        .layout
        .split_dynamic_offsets(&mut resources, &device.profile().limits, Vec::new())
        .unwrap();
    let record = descriptors.get(&pipeline.layout, &resources).unwrap();
    let DescriptorResource::Buffer { offset, .. } = &mut resources[0] else {
        unreachable!()
    };
    *offset = alignment;
    let second_offsets = pipeline
        .layout
        .split_dynamic_offsets(&mut resources, &device.profile().limits, Vec::new())
        .unwrap();
    assert!(Arc::ptr_eq(
        &record,
        &descriptors.get(&pipeline.layout, &resources).unwrap()
    ));
    assert_eq!(descriptors.statistics().updates, 1);
    device.bind_native_pipeline(&pipeline).unwrap();
    device
        .bind_native_descriptors(&pipeline, &record, &first_offsets)
        .unwrap();
    let slot = device.submission.recording.unwrap();
    let command = device.submission.slots[slot].command;
    unsafe { device.owner.device.cmd_dispatch(command, 1, 1, 1) };
    device
        .bind_native_descriptors(&pipeline, &record, &second_offsets)
        .unwrap();
    let binds = device.recording_statistics().descriptor_binds;
    device
        .bind_native_descriptors(&pipeline, &record, &second_offsets)
        .unwrap();
    assert_eq!(device.recording_statistics().descriptor_binds, binds);
    assert!(
        device
            .bind_native_descriptors(&pipeline, &record, &[])
            .is_err()
    );
    assert!(
        device
            .bind_native_descriptors(&pipeline, &record, &[(alignment * 2) as u32])
            .is_err()
    );
    if device.profile().limits.min_storage_buffer_offset_alignment > 1 {
        assert!(
            device
                .bind_native_descriptors(&pipeline, &record, &[1])
                .is_err()
        );
    }
    unsafe {
        device.owner.device.cmd_dispatch(command, 1, 1, 1);
        let barriers = [vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(buffer.handle)
            .size(buffer.size())];
        device.dispatch.barrier(
            command,
            &vk::DependencyInfo::default().buffer_memory_barriers(&barriers),
        );
    }
    let work = device.flush().unwrap().unwrap();
    // Command slots retain the immutable set after its cache is destroyed.
    drop(descriptors);
    drop(record);
    drop(resources);
    assert!(device.wait_for_work(work, Duration::from_secs(10)).unwrap());
    assert_eq!(
        unsafe { buffer.read_range(4, 4) }.unwrap(),
        24u32.to_le_bytes()
    );
    assert_eq!(
        unsafe { buffer.read_range(alignment + 4, 4) }.unwrap(),
        36u32.to_le_bytes()
    );
    drop(buffer);
    drop(pipeline);
    drop(pipelines);
    drop(shaders);
    drop(pool);
    drop(device);
    validation.check().unwrap();
}

#[test]
fn stage_reuse_precedes_emission_across_programs_and_restored_disk_artifacts() {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return;
    };
    let monitor = device.validation_monitor();
    let root = tempfile::tempdir().unwrap();
    let config = CacheConfiguration {
        directory: Some(root.path().to_path_buf()),
        ..Default::default()
    };
    let pool = Arc::new(CompilePool::new(2, 16).unwrap());
    let shaders = ShaderCompiler::new(NS, Arc::clone(&pool), &config);
    let opts = device.shader_options();
    let source = |value| ContentId::of(b"stage-program-test", &[&[value]]);
    let request = |compiler: &ShaderCompiler, value| {
        ready(
            compiler
                .request(
                    source(value),
                    Box::new([opts]),
                    || Ok(Box::new([kernel(7)])),
                    |fs| fs.iter().collect(),
                )
                .unwrap(),
        )
    };
    let first = request(&shaders, 1);
    let related = request(&shaders, 2);
    assert!(!Arc::ptr_eq(&first, &related));
    assert!(
        Arc::ptr_eq(&first.stages[0], &related.stages[0]),
        "different program misses must share the pre-emission stage"
    );
    drop(shaders);
    let restored = ShaderCompiler::new(NS, pool, &config);
    let disk = ready(
        restored
            .request::<Box<[Function; 1]>>(
                source(1),
                Box::new([opts]),
                || panic!("disk hit must not translate"),
                |fs| fs.iter().collect(),
            )
            .unwrap(),
    );
    assert!(disk.timings.disk_hit);
    let fresh = request(&restored, 3);
    assert!(
        Arc::ptr_eq(&disk.stages[0], &fresh.stages[0]),
        "disk hydration must register its semantic stage key"
    );
    drop(restored);
    drop(device);
    monitor.check().unwrap();
}
