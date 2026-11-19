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
    gpu.nvn_flush(None)?;
    let submitted = gpu.residency.device().scheduled_tick();
    let (send, done) = std::sync::mpsc::channel();
    for i in 0..2 {
        let send = send.clone();
        gpu.nvn_flush(Some(Box::new(move |result| {
            let _ = send.send(i);
            result
        })))?;
    }
    gpu.nvn_flush(None)?;
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
