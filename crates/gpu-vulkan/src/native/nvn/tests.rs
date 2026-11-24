use super::*;
use nixe_memory::{AllocationRange, Machine, device_memory::DeviceMemory};

mod support {
    use crate as nixe_gpu_vulkan;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/device.rs"
    ));
}

#[test]
fn nvn2_compute_compiles_with_private_spill_memory() -> Result<(), Box<dyn std::error::Error>> {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return Ok(());
    };
    let validation = device.validation_monitor();
    let machine = Machine::new(16 << 20)?;
    let allocation = machine.account(4 << 20)?.allocate(4096)?;
    let memory = DeviceMemory::new(machine, 32);
    let address = 0x10000;
    let _mapping = memory.map(
        address,
        Arc::from([AllocationRange::new(allocation.pin(), 0, 4096)?]),
        true,
    )?;
    let code = [
        0x000fe2000010080000000444ff007387u128, // STL [RZ + 4], R68
        0x000fea0003800000000000000000794d,     // EXIT
    ];
    let bytes = code.map(u128::to_le_bytes);
    memory.write_block_unsafe(
        address,
        nixe_memory::WriteSource::Bytes(bytes.as_flattened()),
    )?;
    let mut gpu = NativeGpuExecution::new(
        memory,
        device,
        CacheConfiguration::default(),
        64 << 20,
        BufferDownloadPolicy::OnRead,
    )?;
    shader::compute(
        &mut gpu.execution(),
        nixe_nvn::Shader {
            address,
            bytes: 32,
            constants: 32,
            stage: 5,
            group: [1; 3],
            isa: 1,
            local_memory_size: 8,
            ..Default::default()
        },
        &nixe_nvn::Bindings {
            buffers: &Default::default(),
            textures: &Default::default(),
            bindless: None,
        },
    )?;
    drop(gpu);
    validation.check()?;
    Ok(())
}

#[test]
fn nvn_shader_survives_unrelated_mappings_until_its_code_is_unmapped()
-> Result<(), Box<dyn std::error::Error>> {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return Ok(());
    };
    let machine = Machine::new(16 << 20)?;
    let allocation = machine.account(4 << 20)?.allocate(8192)?;
    let memory = DeviceMemory::new(machine, 32);
    let mut gpu = NativeGpuExecution::new(
        memory.clone(),
        device,
        CacheConfiguration::default(),
        64 << 20,
        BufferDownloadPolicy::OnRead,
    )?;
    let size = 4096;
    let address = 0x10000;
    let code = memory.map(
        address,
        Arc::from([AllocationRange::new(allocation.pin(), 0, size)?]),
        true,
    )?;
    let isa = nixe_shader_recompiler::environment::ShaderIsa::Ampere;
    let version = gpu.execution().residency.shader_cache.register(
        address,
        isa,
        nixe_gpu::shader_memory::ShaderInfo::new(0, size as usize),
    )?;
    let usage = nixe_nvn::ShaderUsage {
        stages: Box::new([nixe_nvn::StageSource {
            shader: nixe_nvn::Shader {
                address,
                bytes: size as u32,
                ..Default::default()
            },
            bytes: Arc::from([0; 4096]),
            code_digest: [0; 32],
            version,
        }]),
        shapes: Arc::from([]),
        sparse_generation: None,
    };
    let generation = memory.generation();
    let unrelated = memory.map(
        0x20000,
        Arc::from([AllocationRange::new(allocation.pin(), size, size)?]),
        true,
    )?;
    assert_ne!(memory.generation(), generation);
    assert!(usage.current(&memory));
    drop(unrelated);
    assert!(usage.current(&memory));
    drop(code);
    assert!(!usage.current(&memory));
    Ok(())
}

#[test]
fn counter_comparison_snapshots_device_writes_in_queue_order()
-> Result<(), Box<dyn std::error::Error>> {
    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return Ok(());
    };
    let conditional = device.profile().conditional_rendering;
    let validation = device.validation_monitor();
    let machine = Machine::new(16 << 20)?;
    let allocation = machine.account(4 << 20)?.allocate(8192)?;
    let memory = DeviceMemory::new(machine, 32);
    let address = 0x10004;
    let framebuffer = 0x11000;
    let _mapping = memory.map(
        0x10000,
        Arc::from([AllocationRange::new(allocation.pin(), 0, 8192)?]),
        true,
    )?;
    let mut gpu = NativeGpuExecution::new(
        memory.clone(),
        device,
        CacheConfiguration::default(),
        64 << 20,
        BufferDownloadPolicy::OnRead,
    )?;
    let target = Texture {
        address: framebuffer,
        width: 8,
        height: 4,
        depth: 1,
        stride: 32,
        flags: 0x10,
        format: 0x25,
        levels: 1,
        target: 1,
        samples: 1,
        swizzle: [2, 3, 4, 5],
        ..Default::default()
    };
    let different = [7u64, 0, (1u64 << 32) | 7];
    let equal = [7u64, 0, 7];
    let counters = |words| Operation::Update {
        address,
        bytes: bytemuck::cast_slice(words),
    };
    let compare = |mode| Operation::RenderEnable { mode, address };
    let clear = |color, scissor| Operation::Clear {
        target,
        color,
        scissor,
        mask: 15,
    };
    let full = [0, 0, 8, 4];
    let red = [1., 0., 0., 1.];
    let green = [0., 1., 0., 1.];
    let mut selected = selections::Queue::default();
    {
        let mut native = gpu.execution();
        for op in [
            counters(&different),
            compare(1),
            counters(&equal),
            clear(green, full),
            compare(1),
            clear(red, full),
        ] {
            native.issue_nvn(1, op, &mut selected)?;
        }
        if conditional {
            assert!(matches!(native.state.nvn.conditions.draw(1), Some(Some(_))));
        }
        native.issue_nvn(
            2,
            clear(red, [4, 0, 4, 4]),
            &mut selections::Queue::default(),
        )?;
        native.issue_nvn(1, clear(red, full), &mut selected)?;
    }
    // The explicit flush submits the draws. Both later fences must observe
    // that same GPU prefix in FIFO order even if it is still in flight.
    use nixe_gpu::GpuPresentation;
    gpu.nvn_flush(false, None)?;
    let submitted = gpu.residency.device().scheduled_tick();
    let (send, done) = std::sync::mpsc::channel();
    for i in 0..2 {
        let send = send.clone();
        gpu.nvn_flush(
            false,
            Some(Box::new(move |result| {
                let _ = send.send(i);
                result
            })),
        )?;
    }
    gpu.nvn_flush(false, None)?;
    assert_eq!(gpu.residency.device().scheduled_tick(), submitted);
    for i in 0..2 {
        assert_eq!(done.recv_timeout(Duration::from_secs(5))?, i);
    }
    gpu.residency.access().flush_read_span(framebuffer, 128)?;
    let mut pixels = [0; 128];
    memory
        .reader()
        .read_block_unsafe(framebuffer, &mut pixels)?;
    for row in pixels.chunks_exact(32) {
        for (x, pixel) in row.chunks_exact(4).enumerate() {
            assert_eq!(
                pixel,
                if x < 4 {
                    [0, 255, 0, 255]
                } else {
                    [255, 0, 0, 255]
                }
            );
        }
    }
    drop(gpu);
    validation.check()?;
    Ok(())
}

#[test]
fn early_nvn_prefix_keeps_pending_publication_on_its_original_batch()
-> Result<(), Box<dyn std::error::Error>> {
    use nixe_gpu::GpuPresentation;

    let Some(device) = support::device_for_test(ApiPreference::Auto) else {
        return Ok(());
    };
    let validation = device.validation_monitor();
    let machine = Machine::new(16 << 20)?;
    let allocation = machine.account(4 << 20)?.allocate(4096)?;
    let memory = DeviceMemory::new(machine, 32);
    let address = 0x10000;
    let _mapping = memory.map(
        address,
        Arc::from([AllocationRange::new(allocation.pin(), 0, 4096)?]),
        true,
    )?;
    let mut gpu = NativeGpuExecution::new(
        memory,
        device,
        CacheConfiguration::default(),
        64 << 20,
        BufferDownloadPolicy::OnRead,
    )?;
    let target = Texture {
        address,
        width: 8,
        height: 4,
        depth: 1,
        stride: 32,
        flags: 0x10,
        format: 0x25,
        levels: 1,
        target: 1,
        samples: 1,
        swizzle: [2, 3, 4, 5],
        ..Default::default()
    };
    let record_pass =
        |gpu: &mut NativeGpuExecution, color| -> Result<WorkTicket, BackendRuntimeError> {
            gpu.begin_frame_work();
            let mut native = gpu.execution();
            native.issue_nvn(
                1,
                Operation::Clear {
                    target,
                    color,
                    scissor: [0, 0, 8, 4],
                    mask: 15,
                },
                &mut selections::Queue::default(),
            )?;
            let device = native.residency.device_mut();
            for _ in 0..1024 {
                assert!(!device.submission.nvn_prefix_full(&device.dispatch));
            }
            device.end_rendering().map_err(runtime)?;
            Ok(device.current_work())
        };
    let end_frame = |gpu: &mut NativeGpuExecution, work: WorkTicket| -> Result<(), VulkanError> {
        let mut residency = gpu.residency.access();
        let device = residency.device_mut();
        assert!(device.wait_for_work(work, Duration::from_secs(5))?);
        device.submission.end_frame(work.tick(), 1);
        Ok(())
    };

    let first = record_pass(&mut gpu, [0., 0., 1., 1.])?;
    gpu.nvn_flush(false, None)?;
    end_frame(&mut gpu, first)?;

    let pending = record_pass(&mut gpu, [1., 0., 0., 1.])?;
    let (send, done) = std::sync::mpsc::channel();
    gpu.state.fences.queue(
        Some(pending),
        false,
        Some(Box::new(move |result| {
            let _ = send.send(result.is_ok());
            result
        })),
    )?;
    gpu.execution().submit_nvn_prefix_at_pass_end()?;
    assert_eq!(gpu.residency.device().scheduled_tick(), first.tick());
    assert!(matches!(
        done.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty)
    ));
    gpu.nvn_flush(false, None)?;
    assert!(done.recv_timeout(Duration::from_secs(5))?);
    end_frame(&mut gpu, pending)?;

    let eligible = record_pass(&mut gpu, [0., 1., 0., 1.])?;
    gpu.execution().submit_nvn_prefix_at_pass_end()?;
    assert_eq!(gpu.residency.device().scheduled_tick(), eligible.tick());
    end_frame(&mut gpu, eligible)?;
    drop(gpu);
    validation.check()?;
    Ok(())
}
